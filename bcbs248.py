"""
bcbs248.py — BCBS 248 Intraday Liquidity Monitoring Indicators
================================================================
Implements the 7 quantitative monitoring tools defined by the
Basel Committee on Banking Supervision (BCBS 248, April 2013):

1. Daily maximum intraday liquidity usage
2. Available intraday liquidity at start of day
3. Total payments made
4. Time-specific obligations
5. Value of payments made on behalf of FMI participants (correspondent)
6. Intraday credit lines extended (to customers)
7. Intraday throughput (timing of payments — % settled by hour)

Reference: "Monitoring tools for intraday liquidity management"
           Basel Committee on Banking Supervision, April 2013
"""

import numpy as np
import pandas as pd


def compute_daily_max_usage(balance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Indicator 1: Daily maximum intraday liquidity usage.
    = Opening balance - Minimum balance during the day.
    This shows the peak intraday funding gap.
    """
    bal = balance_df.copy()
    bal["timestamp"] = pd.to_datetime(bal["timestamp"])
    bal["date"] = bal["timestamp"].dt.date

    daily = bal.groupby("date").agg(
        opening_balance=("fed_balance", "first"),
        min_balance=("fed_balance", "min"),
        closing_balance=("fed_balance", "last"),
        max_balance=("fed_balance", "max"),
    ).reset_index()

    daily["max_intraday_usage"] = daily["opening_balance"] - daily["min_balance"]
    daily["max_usage_pct"] = (daily["max_intraday_usage"] / daily["opening_balance"] * 100).round(2)

    return daily


def compute_available_liquidity(balance_df: pd.DataFrame, credit_line: float = 5_000_000_000) -> pd.DataFrame:
    """
    Indicator 2: Available intraday liquidity at start of business day.
    = Opening reserve balance + Undrawn Fed daylight overdraft capacity
      + Other committed intraday credit lines.
    """
    bal = balance_df.copy()
    bal["timestamp"] = pd.to_datetime(bal["timestamp"])
    bal["date"] = bal["timestamp"].dt.date

    daily = bal.groupby("date").agg(
        opening_balance=("fed_balance", "first"),
    ).reset_index()

    daily["daylight_overdraft_capacity"] = credit_line
    daily["total_available_liquidity"] = daily["opening_balance"] + daily["daylight_overdraft_capacity"]

    return daily


def compute_total_payments(payment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Indicator 3: Total value of payments made & received.
    Broken down by channel.
    """
    pmt = payment_df.copy()
    pmt["timestamp"] = pd.to_datetime(pmt["timestamp"])
    pmt["date"] = pmt["timestamp"].dt.date

    daily = pmt.groupby("date").agg(
        total_outflow=("outflow", "sum"),
        total_inflow=("inflow", "sum"),
        total_net=("net_flow", "sum"),
        n_transactions=("net_flow", "count"),
    ).reset_index()

    # Channel breakdown
    channel_daily = pmt.groupby(["date", "channel"]).agg(
        outflow=("outflow", "sum"),
        inflow=("inflow", "sum"),
    ).reset_index()

    return daily, channel_daily


def compute_time_specific_obligations(payment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Indicator 4: Value of time-specific and critical obligations.
    These are payments that MUST be made by a specific time
    (CCP margin calls, settlement deadlines, etc.)
    """
    pmt = payment_df.copy()
    pmt["timestamp"] = pd.to_datetime(pmt["timestamp"])
    pmt["date"] = pmt["timestamp"].dt.date

    critical = pmt[pmt["is_time_critical"] == True]

    daily_critical = critical.groupby("date").agg(
        critical_outflow=("outflow", "sum"),
        critical_inflow=("inflow", "sum"),
        n_critical=("outflow", "count"),
        critical_pct_of_total=("outflow", "sum"),
    ).reset_index()

    # Compute as % of total
    total_daily = pmt.groupby("date")["outflow"].sum().reset_index()
    total_daily.columns = ["date", "total_daily_outflow"]

    daily_critical = daily_critical.merge(total_daily, on="date", how="left")
    daily_critical["critical_pct_of_total"] = (
        daily_critical["critical_outflow"] / daily_critical["total_daily_outflow"] * 100
    ).round(2)

    return daily_critical


def compute_largest_counterparty_exposures(payment_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Indicator 5/6: Largest intraday bilateral exposures.
    Shows net exposure to each counterparty — identifies concentration risk.
    """
    pmt = payment_df.copy()
    pmt["timestamp"] = pd.to_datetime(pmt["timestamp"])
    pmt["date"] = pmt["timestamp"].dt.date

    cp_daily = pmt.groupby(["date", "counterparty"]).agg(
        gross_outflow=("outflow", "sum"),
        gross_inflow=("inflow", "sum"),
        net_exposure=("net_flow", "sum"),
    ).reset_index()

    cp_daily["abs_net_exposure"] = cp_daily["net_exposure"].abs()

    # Get top N counterparties by average absolute exposure
    avg_exposure = (
        cp_daily.groupby("counterparty")["abs_net_exposure"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )

    return cp_daily, avg_exposure


def compute_throughput(payment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Indicator 7: Intraday throughput — cumulative % of daily payments
    settled by each hour. Regulators want to see payments spread
    throughout the day, not concentrated at close.
    """
    pmt = payment_df.copy()
    pmt["timestamp"] = pd.to_datetime(pmt["timestamp"])
    pmt["date"] = pmt["timestamp"].dt.date
    pmt["hour"] = pmt["timestamp"].dt.hour

    # Hourly outflow
    hourly = pmt.groupby(["date", "hour"])["outflow"].sum().reset_index()
    daily_total = pmt.groupby("date")["outflow"].sum().reset_index()
    daily_total.columns = ["date", "daily_total"]

    hourly = hourly.merge(daily_total, on="date")
    hourly["hourly_pct"] = (hourly["outflow"] / hourly["daily_total"] * 100).round(2)

    # Cumulative by hour within each day
    hourly = hourly.sort_values(["date", "hour"])
    hourly["cum_pct"] = hourly.groupby("date")["hourly_pct"].cumsum()

    # Average throughput profile across all days
    avg_throughput = hourly.groupby("hour").agg(
        avg_hourly_pct=("hourly_pct", "mean"),
        avg_cum_pct=("cum_pct", "mean"),
    ).reset_index()

    return hourly, avg_throughput


def generate_bcbs248_summary(payment_df: pd.DataFrame, balance_df: pd.DataFrame) -> dict:
    """
    Generate a complete BCBS 248 summary report.
    Returns dict of DataFrames for each indicator.
    """
    indicator_1 = compute_daily_max_usage(balance_df)
    indicator_2 = compute_available_liquidity(balance_df)
    indicator_3_total, indicator_3_channel = compute_total_payments(payment_df)
    indicator_4 = compute_time_specific_obligations(payment_df)
    indicator_5_daily, indicator_5_top = compute_largest_counterparty_exposures(payment_df)
    indicator_7_daily, indicator_7_avg = compute_throughput(payment_df)

    summary = {
        "max_intraday_usage": indicator_1,
        "available_liquidity": indicator_2,
        "total_payments": indicator_3_total,
        "payments_by_channel": indicator_3_channel,
        "time_specific_obligations": indicator_4,
        "counterparty_exposures": indicator_5_daily,
        "top_counterparties": indicator_5_top,
        "throughput_daily": indicator_7_daily,
        "throughput_avg": indicator_7_avg,
    }

    return summary
