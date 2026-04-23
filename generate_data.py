"""
generate_data.py — Realistic Intraday Liquidity Data Generator
===============================================================
Generates synthetic payment data that mirrors real-world USD payment patterns:
  • Fedwire: RTGS, heavy 9-11 AM & 3-5 PM, individual large-value payments
  • CHIPS: Multilateral netting, bulk pre-funding AM, net settlement ~4:30 PM
  • ACH: Batch processing windows (6 AM, 12 PM, 4 PM)
  • FedSecurities: Treasury/agency settlement, concentrated 10 AM-2 PM
  • CCP_Margin: Variation margin calls from CCPs, AM & PM windows

Includes: day-of-week effects, month-end surges, counterparty concentration,
          time-specific obligations, and realistic balance dynamics.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# ─── Configuration ───────────────────────────────────────────────────────────
CHANNELS = ["FEDWIRE", "CHIPS", "ACH", "FED_SECURITIES", "CCP_MARGIN"]
BUSINESS_LINES = ["Markets", "Treasury", "Commercial_Banking", "Asset_Management", "Retail"]
COUNTERPARTIES = [
    "Citi", "BofA", "Wells_Fargo", "Goldman", "Morgan_Stanley",
    "BNY_Mellon", "State_Street", "HSBC", "Barclays", "Deutsche_Bank",
    "CLS_Bank", "DTCC", "CME_Clearing", "ICE_Clear", "FICC"
]

# Intraday profiles: (hour, relative_weight) — different shape per channel
FEDWIRE_PROFILE = {
    8: 0.3, 9: 0.8, 10: 1.0, 11: 0.9, 12: 0.6, 13: 0.5,
    14: 0.7, 15: 0.9, 16: 1.0, 17: 0.7, 18: 0.2
}
CHIPS_PROFILE = {
    8: 0.2, 9: 0.6, 10: 0.4, 11: 0.3, 12: 0.3, 13: 0.3,
    14: 0.4, 15: 0.5, 16: 0.8, 17: 0.3, 18: 0.1  # net settlement ~4:30
}
ACH_PROFILE = {
    6: 0.8, 7: 0.3, 8: 0.1, 9: 0.1, 10: 0.1, 11: 0.1,
    12: 0.7, 13: 0.2, 14: 0.1, 15: 0.1, 16: 0.6, 17: 0.2
}
FED_SEC_PROFILE = {
    9: 0.3, 10: 0.8, 11: 1.0, 12: 0.7, 13: 0.6, 14: 0.5, 15: 0.3
}
CCP_MARGIN_PROFILE = {
    9: 0.5, 10: 1.0, 11: 0.3, 12: 0.2, 13: 0.2, 14: 0.8, 15: 0.6, 16: 0.3
}

PROFILES = {
    "FEDWIRE": FEDWIRE_PROFILE,
    "CHIPS": CHIPS_PROFILE,
    "ACH": ACH_PROFILE,
    "FED_SECURITIES": FED_SEC_PROFILE,
    "CCP_MARGIN": CCP_MARGIN_PROFILE,
}

# Base daily volumes per channel (in USD)
BASE_DAILY_VOLUME = {
    "FEDWIRE": 800_000_000,
    "CHIPS": 500_000_000,
    "ACH": 150_000_000,
    "FED_SECURITIES": 300_000_000,
    "CCP_MARGIN": 200_000_000,
}

# Typical inflow/outflow split (inflow fraction)
INFLOW_FRAC = {
    "FEDWIRE": 0.48,       # slightly net payer
    "CHIPS": 0.50,         # roughly balanced (netting)
    "ACH": 0.52,           # slightly net receiver
    "FED_SECURITIES": 0.49,
    "CCP_MARGIN": 0.45,    # net payer (margin calls)
}


def _day_of_week_factor(dow: int) -> float:
    """Mon=0..Fri=4. Tue-Thu heavier, Mon/Fri lighter."""
    return {0: 0.90, 1: 1.05, 2: 1.10, 3: 1.05, 4: 0.85}.get(dow, 0.0)


def _month_end_factor(dt: datetime) -> float:
    """Last 2 business days of month see 40-60% volume spike."""
    next_month = (dt.replace(day=28) + timedelta(days=4)).replace(day=1)
    last_day = next_month - timedelta(days=1)
    days_to_end = (last_day - dt.replace(hour=0, minute=0, second=0)).days
    if days_to_end <= 0:
        return 1.55
    elif days_to_end == 1:
        return 1.35
    elif days_to_end == 2:
        return 1.15
    return 1.0


def _quarter_end_factor(dt: datetime) -> float:
    """Quarter-end adds extra on top of month-end."""
    if dt.month in (3, 6, 9, 12):
        next_month = (dt.replace(day=28) + timedelta(days=4)).replace(day=1)
        last_day = next_month - timedelta(days=1)
        days_to_end = (last_day - dt.replace(hour=0, minute=0, second=0)).days
        if days_to_end <= 1:
            return 1.20
    return 1.0


def generate_intraday_data(
    start_date: str = "2024-09-01",
    end_date: str = "2025-04-15",
    freq_minutes: int = 15,
    seed: int = 42,
    output_path: str = "idl_payment_data.csv",
) -> pd.DataFrame:
    """
    Generate realistic intraday payment data.

    Returns DataFrame with columns:
        timestamp, channel, business_line, counterparty,
        inflow, outflow, net_flow, is_time_critical,
        day_type (normal/month_end/quarter_end)
    """
    rng = np.random.default_rng(seed)
    records = []

    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    while current <= end:
        # Skip weekends
        if current.dayofweek >= 5:
            current += timedelta(days=1)
            continue

        dow_factor = _day_of_week_factor(current.dayofweek)
        me_factor = _month_end_factor(current)
        qe_factor = _quarter_end_factor(current)
        combined_factor = dow_factor * me_factor * qe_factor

        # Determine day type
        if qe_factor > 1.0:
            day_type = "quarter_end"
        elif me_factor > 1.1:
            day_type = "month_end"
        else:
            day_type = "normal"

        # Add daily noise factor (some days are just busier)
        daily_noise = rng.normal(1.0, 0.08)
        daily_factor = max(0.6, combined_factor * daily_noise)

        for channel in CHANNELS:
            profile = PROFILES[channel]
            base_vol = BASE_DAILY_VOLUME[channel] * daily_factor
            inflow_frac = INFLOW_FRAC[channel]

            # Normalize profile weights
            total_weight = sum(profile.values())

            for hour, weight in profile.items():
                intervals_in_hour = 60 // freq_minutes
                hourly_volume = base_vol * (weight / total_weight)

                for interval in range(intervals_in_hour):
                    minute = interval * freq_minutes
                    ts = current.replace(hour=hour, minute=minute, second=0)

                    # Volume for this interval
                    interval_vol = hourly_volume / intervals_in_hour

                    # Add interval-level noise
                    noise = rng.lognormal(0, 0.25)
                    interval_vol *= noise

                    # Split into inflow/outflow
                    # Inflow fraction varies by interval (timing mismatch is the whole point)
                    inflow_shift = rng.normal(0, 0.08)
                    actual_inflow_frac = np.clip(inflow_frac + inflow_shift, 0.2, 0.8)

                    inflow = interval_vol * actual_inflow_frac
                    outflow = interval_vol * (1 - actual_inflow_frac)

                    # Assign business line (weighted by channel)
                    if channel == "CCP_MARGIN":
                        bl_weights = [0.50, 0.20, 0.10, 0.15, 0.05]
                    elif channel == "ACH":
                        bl_weights = [0.05, 0.10, 0.25, 0.10, 0.50]
                    elif channel == "FED_SECURITIES":
                        bl_weights = [0.35, 0.30, 0.15, 0.15, 0.05]
                    elif channel == "CHIPS":
                        bl_weights = [0.30, 0.25, 0.20, 0.15, 0.10]
                    else:
                        bl_weights = [0.25, 0.20, 0.20, 0.15, 0.20]

                    bl = rng.choice(BUSINESS_LINES, p=bl_weights)

                    # Assign counterparty
                    cp = rng.choice(COUNTERPARTIES)

                    # Time-critical flag (margin calls, large Fedwire, settlement deadlines)
                    is_critical = False
                    if channel == "CCP_MARGIN":
                        is_critical = True
                    elif channel == "FEDWIRE" and outflow > 15_000_000:
                        is_critical = True
                    elif channel == "FED_SECURITIES" and hour >= 14:
                        is_critical = rng.random() < 0.6

                    records.append({
                        "timestamp": ts,
                        "channel": channel,
                        "business_line": bl,
                        "counterparty": cp,
                        "inflow": round(inflow, 2),
                        "outflow": round(outflow, 2),
                        "net_flow": round(inflow - outflow, 2),
                        "is_time_critical": is_critical,
                        "day_type": day_type,
                    })

        current += timedelta(days=1)

    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Generated {len(df):,} records → {output_path}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Channels: {df['channel'].nunique()}")
        print(f"Business days: {df['timestamp'].dt.date.nunique()}")

    return df


def build_balance_series(df: pd.DataFrame, opening_balance: float = 25_000_000_000) -> pd.DataFrame:
    """
    Aggregate to 15-min intervals and compute running Fed reserve balance.

    Opening balance of $25B is realistic for a large bank's Fed account.
    Different channels settle differently:
      - FEDWIRE, FED_SECURITIES: real-time (100% pass-through)
      - CCP_MARGIN: real-time (100%)
      - CHIPS: deferred net settlement (accumulates, settles at end of day)
      - ACH: batch settlement (settles at batch windows)
    """
    # Real-time channels: immediate balance impact
    rt_channels = ["FEDWIRE", "FED_SECURITIES", "CCP_MARGIN"]
    # Deferred channels: we still show flows but mark settlement type
    deferred_channels = ["CHIPS", "ACH"]

    agg = (
        df.groupby([pd.Grouper(key="timestamp", freq="15min")])
        .agg(
            total_inflow=("inflow", "sum"),
            total_outflow=("outflow", "sum"),
            total_net=("net_flow", "sum"),
            n_transactions=("net_flow", "count"),
            n_critical=("is_time_critical", "sum"),
        )
        .reset_index()
    )

    # Channel-level aggregation
    for ch in CHANNELS:
        ch_data = df[df["channel"] == ch].groupby(
            pd.Grouper(key="timestamp", freq="15min")
        )["net_flow"].sum().reset_index()
        ch_data.columns = ["timestamp", f"net_{ch.lower()}"]
        agg = agg.merge(ch_data, on="timestamp", how="left")

    agg = agg.fillna(0).sort_values("timestamp").reset_index(drop=True)

    # Compute balance: all channels impact balance but CHIPS has delayed netting
    # For simplicity, we model CHIPS as settling with a lag
    balance = []
    b = opening_balance
    for _, row in agg.iterrows():
        # Real-time channels: immediate impact
        rt_impact = sum(row.get(f"net_{ch.lower()}", 0) for ch in rt_channels)
        # CHIPS & ACH: partial impact (netting reduces gross requirement)
        deferred_impact = sum(row.get(f"net_{ch.lower()}", 0) for ch in deferred_channels) * 0.3
        b += rt_impact + deferred_impact
        balance.append(b)

    agg["fed_balance"] = balance

    # Business line breakdown
    for bl in BUSINESS_LINES:
        bl_data = df[df["business_line"] == bl].groupby(
            pd.Grouper(key="timestamp", freq="15min")
        )["net_flow"].sum().reset_index()
        bl_data.columns = ["timestamp", f"net_{bl.lower()}"]
        agg = agg.merge(bl_data, on="timestamp", how="left")

    agg = agg.fillna(0)
    return agg


if __name__ == "__main__":
    df = generate_intraday_data()
    bal = build_balance_series(df)
    bal.to_csv("idl_balance_series.csv", index=False)
    print(f"\nBalance series: {len(bal):,} intervals")
    print(f"Opening balance: ${bal['fed_balance'].iloc[0]:,.0f}")
    print(f"Min balance: ${bal['fed_balance'].min():,.0f}")
    print(f"Max balance: ${bal['fed_balance'].max():,.0f}")
