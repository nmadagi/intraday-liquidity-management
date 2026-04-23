"""
forecasting.py — Intraday Liquidity Forecasting Engine
========================================================
Multi-model forecasting with proper feature engineering:
  • Seasonal features: hour-of-day, day-of-week, month-end, quarter-end
  • Lagged features: rolling means, stdev, prior-day same-hour
  • Channel-specific models for granular forecasting
  • Ensemble: XGBoost + seasonal baseline with confidence intervals
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def engineer_features(df: pd.DataFrame, target_col: str = "total_net") -> pd.DataFrame:
    """
    Build feature matrix from time-indexed balance/flow data.
    Expects df with 'timestamp' column and target_col.
    """
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values("timestamp").reset_index(drop=True)

    ts = out["timestamp"]

    # ── Calendar features ──
    out["hour"] = ts.dt.hour
    out["minute"] = ts.dt.minute
    out["day_of_week"] = ts.dt.dayofweek  # 0=Mon
    out["day_of_month"] = ts.dt.day
    out["month"] = ts.dt.month
    out["is_monday"] = (ts.dt.dayofweek == 0).astype(int)
    out["is_friday"] = (ts.dt.dayofweek == 4).astype(int)

    # Month-end / Quarter-end flags
    month_end = ts.dt.is_month_end
    out["is_month_end"] = month_end.astype(int)
    out["days_to_month_end"] = ts.apply(
        lambda x: ((x + pd.offsets.MonthEnd(0)) - x).days if not x.is_month_end else 0
    )
    out["is_quarter_end"] = (month_end & ts.dt.month.isin([3, 6, 9, 12])).astype(int)

    # ── Intraday position features ──
    # Minute of day (normalized 0-1)
    out["minute_of_day"] = (ts.dt.hour * 60 + ts.dt.minute) / (24 * 60)

    # Cyclical encoding of hour (sine/cosine)
    out["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 5)
    out["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 5)

    # ── Lagged features ──
    target = out[target_col]

    # Rolling statistics (various windows)
    for w in [4, 8, 16, 32]:  # 1hr, 2hr, 4hr, 8hr at 15-min freq
        out[f"roll_mean_{w}"] = target.rolling(w, min_periods=1).mean()
        out[f"roll_std_{w}"] = target.rolling(w, min_periods=1).std().fillna(0)
        out[f"roll_max_{w}"] = target.rolling(w, min_periods=1).max()
        out[f"roll_min_{w}"] = target.rolling(w, min_periods=1).min()

    # Lag features: same interval previous day (approx 40 intervals/day for 6AM-6PM)
    intervals_per_day = 48  # approximate
    for lag_days in [1, 2, 5]:
        lag = lag_days * intervals_per_day
        out[f"lag_{lag_days}d"] = target.shift(lag)

    # Cumulative daily flow
    out["date"] = ts.dt.date
    out["cum_daily_net"] = out.groupby("date")[target_col].cumsum()

    # Fill NAs from lagging
    out = out.fillna(0)

    return out


FEATURE_COLS = [
    "hour", "minute", "day_of_week", "day_of_month", "month",
    "is_monday", "is_friday", "is_month_end", "days_to_month_end", "is_quarter_end",
    "minute_of_day", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "roll_mean_4", "roll_std_4", "roll_max_4", "roll_min_4",
    "roll_mean_8", "roll_std_8", "roll_max_8", "roll_min_8",
    "roll_mean_16", "roll_std_16", "roll_max_16", "roll_min_16",
    "roll_mean_32", "roll_std_32", "roll_max_32", "roll_min_32",
    "lag_1d", "lag_2d", "lag_5d",
    "cum_daily_net",
]


def train_forecast_model(
    df: pd.DataFrame,
    target_col: str = "total_net",
    train_frac: float = 0.85,
):
    """
    Train GradientBoosting model on engineered features.
    Returns: model, feature_df, train/test split info, metrics.
    """
    feat_df = engineer_features(df, target_col)

    # Train/test split (time-based, no shuffle)
    n = len(feat_df)
    split_idx = int(n * train_frac)

    train = feat_df.iloc[:split_idx]
    test = feat_df.iloc[split_idx:]

    X_train = train[FEATURE_COLS].values
    y_train = train[target_col].values
    X_test = test[FEATURE_COLS].values
    y_test = test[target_col].values

    # GradientBoosting (similar to XGBoost but in sklearn)
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "train_mae": mean_absolute_error(y_train, y_pred_train),
        "test_mae": mean_absolute_error(y_test, y_pred_test),
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "train_size": len(train),
        "test_size": len(test),
    }

    # Feature importance
    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return model, feat_df, metrics, importance


def forecast_forward(
    model,
    last_known_df: pd.DataFrame,
    horizon_steps: int = 48,
    freq_minutes: int = 15,
    target_col: str = "total_net",
) -> pd.DataFrame:
    """
    Generate forward forecast by iteratively predicting one step ahead.
    Uses the last known data to bootstrap lag/rolling features.

    Key fix: overnight hours are NOT appended to the working history,
    so rolling features stay clean for next-day business hours.
    """
    last_ts = pd.to_datetime(last_known_df["timestamp"].iloc[-1])
    future_timestamps = pd.date_range(
        last_ts + pd.Timedelta(minutes=freq_minutes),
        periods=horizon_steps,
        freq=f"{freq_minutes}min",
    )

    # Build seasonal profile (median + std by dow + hour + minute)
    hist = last_known_df[["timestamp", target_col]].copy()
    hist["timestamp"] = pd.to_datetime(hist["timestamp"])
    hist["_dow"] = hist["timestamp"].dt.dayofweek
    hist["_hour"] = hist["timestamp"].dt.hour
    hist["_minute"] = hist["timestamp"].dt.minute
    seasonal = hist.groupby(["_dow", "_hour", "_minute"])[target_col].median().to_dict()
    seasonal_std_dict = hist.groupby(["_dow", "_hour", "_minute"])[target_col].std().to_dict()

    last_known_date = last_ts.date()

    # Working history — only business-hour data (no zeros to pollute rolling calcs)
    work = last_known_df[["timestamp", target_col]].copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])

    biz_mask = (work["timestamp"].dt.hour >= 6) & (work["timestamp"].dt.hour <= 18) & (work["timestamp"].dt.dayofweek < 5)
    work = work[biz_mask].copy().reset_index(drop=True)

    predictions = []
    for ts in future_timestamps:
        if ts.dayofweek >= 5 or ts.hour < 6 or ts.hour > 18:
            predictions.append({"timestamp": ts, "forecast": 0, "lower": 0, "upper": 0})
            continue

        seasonal_key = (ts.dayofweek, ts.hour, ts.minute)
        seasonal_val = seasonal.get(seasonal_key, 0)

        if ts.date() > last_known_date:
            # Beyond the current day: use seasonal baseline directly
            yhat = seasonal_val
            s_std = seasonal_std_dict.get(seasonal_key, abs(seasonal_val) * 0.5)
            if np.isnan(s_std) or s_std == 0:
                s_std = abs(seasonal_val) * 0.5 if seasonal_val != 0 else 100_000
            lower = yhat - 1.96 * s_std
            upper = yhat + 1.96 * s_std
        else:
            # Same day: ML model with rolling features from actual history
            seed_val = seasonal_val if abs(seasonal_val) > 1000 else (
                work[target_col].iloc[-1] if len(work) else 0
            )
            placeholder = pd.DataFrame([{"timestamp": ts, target_col: seed_val}])
            work_tmp = pd.concat([work, placeholder], ignore_index=True)

            feat = engineer_features(work_tmp, target_col)
            X = feat.iloc[-1:][FEATURE_COLS].values
            yhat = model.predict(X)[0]

            if abs(yhat) < abs(seasonal_val) * 0.1 and abs(seasonal_val) > 1000:
                yhat = 0.3 * yhat + 0.7 * seasonal_val

            recent_std = work[target_col].iloc[-32:].std() if len(work) > 32 else work[target_col].std()
            if np.isnan(recent_std) or recent_std == 0:
                recent_std = abs(seasonal_val) * 0.5 if seasonal_val != 0 else 100_000
            lower = yhat - 1.96 * recent_std
            upper = yhat + 1.96 * recent_std

            work = pd.concat(
                [work, pd.DataFrame([{"timestamp": ts, target_col: yhat}])],
                ignore_index=True,
            )

        predictions.append({
            "timestamp": ts,
            "forecast": yhat,
            "lower": lower,
            "upper": upper,
        })

    return pd.DataFrame(predictions)


def seasonal_baseline(df: pd.DataFrame, target_col: str = "total_net", freq_minutes: int = 15) -> pd.DataFrame:
    """
    Simple seasonal baseline: median by (day_of_week, hour, minute).
    Used for comparison against the ML model.
    """
    tmp = df.copy()
    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"])
    tmp["dow"] = tmp["timestamp"].dt.dayofweek
    tmp["hour"] = tmp["timestamp"].dt.hour
    tmp["minute"] = tmp["timestamp"].dt.minute

    profile = tmp.groupby(["dow", "hour", "minute"])[target_col].agg(["median", "std"]).reset_index()
    profile.columns = ["dow", "hour", "minute", "seasonal_median", "seasonal_std"]

    return profile
