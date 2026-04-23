"""
app.py — Intraday Liquidity Management Dashboard v2
=====================================================
JPMorgan IDL Management — Senior Associate Interview Project
Built with: Python | Streamlit | Plotly | Scikit-learn | Pandas

Modules:
  1. Executive Summary — real-time balance, KPIs, alerts
  2. Channel & LOB Analytics — per-channel and per-business-line views
  3. BCBS 248 Monitoring — regulatory indicators dashboard
  4. Forecasting Engine — ML-based forward projection
  5. Stress Scenarios — pre-defined & custom what-if analysis
  6. IDL Playbook — breach response simulation
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from generate_data import generate_intraday_data, build_balance_series, CHANNELS, BUSINESS_LINES
from forecasting import train_forecast_model, forecast_forward, seasonal_baseline, FEATURE_COLS
from bcbs248 import generate_bcbs248_summary, compute_throughput, compute_daily_max_usage
from playbook import (
    STRESS_SCENARIOS, SEVERITY_LEVELS, classify_severity,
    apply_scenario_stress, generate_escalation_timeline,
    project_balance_with_remediation,
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intraday Liquidity Management — IDL Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 100%);
        border-radius: 8px;
        padding: 16px;
        color: white;
        border-left: 4px solid #4FC3F7;
    }
    .alert-critical {
        background: #B71C1C;
        color: white;
        padding: 12px;
        border-radius: 6px;
        font-weight: bold;
    }
    .alert-elevated {
        background: #E65100;
        color: white;
        padding: 12px;
        border-radius: 6px;
    }
    .alert-advisory {
        background: #F57F17;
        color: white;
        padding: 12px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Generating realistic payment data...")
def load_data():
    pmt_path = "idl_payment_data.csv"
    bal_path = "idl_balance_series.csv"

    if not os.path.exists(pmt_path):
        pmt_df = generate_intraday_data(output_path=pmt_path)
    else:
        pmt_df = pd.read_csv(pmt_path, parse_dates=["timestamp"])

    if not os.path.exists(bal_path):
        bal_df = build_balance_series(pmt_df)
        bal_df.to_csv(bal_path, index=False)
    else:
        bal_df = pd.read_csv(bal_path, parse_dates=["timestamp"])

    return pmt_df, bal_df


@st.cache_resource(show_spinner="Training forecasting model...")
def train_model(bal_csv_path):
    bal_df = pd.read_csv(bal_csv_path, parse_dates=["timestamp"])
    model, feat_df, metrics, importance = train_forecast_model(bal_df, target_col="total_net")
    return model, feat_df, metrics, importance


pmt_df, bal_df = load_data()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/48/bank-building.png", width=40)
    st.title("IDL Dashboard")
    st.caption("Intraday Liquidity Management")
    st.divider()

    # Date range filter
    min_date = pmt_df["timestamp"].dt.date.min()
    max_date = pmt_df["timestamp"].dt.date.max()
    date_range = st.date_input(
        "Date Range",
        value=(max_date - pd.Timedelta(days=30), max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if len(date_range) == 2:
        start_d, end_d = date_range
    else:
        start_d, end_d = min_date, max_date

    # Filter data
    mask_pmt = (pmt_df["timestamp"].dt.date >= start_d) & (pmt_df["timestamp"].dt.date <= end_d)
    mask_bal = (bal_df["timestamp"].dt.date >= start_d) & (bal_df["timestamp"].dt.date <= end_d)
    pmt_filtered = pmt_df[mask_pmt].copy()
    bal_filtered = bal_df[mask_bal].copy()

    st.divider()
    st.caption(f"**Data:** {pmt_filtered['timestamp'].dt.date.nunique()} business days")
    st.caption(f"**Records:** {len(pmt_filtered):,} payment records")
    st.caption(f"**Balance points:** {len(bal_filtered):,}")


# ─── Title ────────────────────────────────────────────────────────────────────
st.title("🏦 Intraday Liquidity Management — IDL Dashboard")
st.caption("North America Treasury Funding | USD Central Bank Balance Monitoring")

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Executive Summary",
    "🔀 Channel & LOB Analytics",
    "📋 BCBS 248 Monitoring",
    "🔮 Forecasting Engine",
    "🧪 Stress Scenarios",
    "🧭 IDL Playbook",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Executive Summary")

    if len(bal_filtered) > 0:
        latest_bal = bal_filtered["fed_balance"].iloc[-1]
        opening_bal = bal_filtered["fed_balance"].iloc[0]
        min_bal = bal_filtered["fed_balance"].min()
        max_bal = bal_filtered["fed_balance"].max()
        avg_bal = bal_filtered["fed_balance"].mean()
        total_outflow = pmt_filtered["outflow"].sum()
        total_inflow = pmt_filtered["inflow"].sum()

        # KPIs row
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Current Balance", f"${latest_bal / 1e9:.2f}B",
                   delta=f"${(latest_bal - opening_bal) / 1e6:+,.0f}M from period start")
        c2.metric("Min Balance (Period)", f"${min_bal / 1e9:.2f}B")
        c3.metric("Max Intraday Usage", f"${(opening_bal - min_bal) / 1e9:.2f}B")
        c4.metric("Total Outflows", f"${total_outflow / 1e9:.1f}B")
        c5.metric("Total Inflows", f"${total_inflow / 1e9:.1f}B")

        st.divider()

        # Balance time series
        col_chart, col_dist = st.columns([3, 1])

        with col_chart:
            fig_bal = go.Figure()
            fig_bal.add_trace(go.Scatter(
                x=bal_filtered["timestamp"], y=bal_filtered["fed_balance"],
                name="Fed Reserve Balance",
                line=dict(color="#4FC3F7", width=1.5),
                fill="tozeroy", fillcolor="rgba(79,195,247,0.1)",
            ))
            fig_bal.add_hline(y=avg_bal, line=dict(color="gray", dash="dash", width=1),
                              annotation_text=f"Avg: ${avg_bal/1e9:.1f}B")
            fig_bal.update_layout(
                title="Fed Reserve Balance — Intraday",
                yaxis_title="USD",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                template="plotly_dark",
            )
            st.plotly_chart(fig_bal, use_container_width=True)

        with col_dist:
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=bal_filtered["fed_balance"] / 1e9,
                nbinsx=30,
                marker_color="#4FC3F7",
                opacity=0.7,
            ))
            fig_dist.update_layout(
                title="Balance Distribution",
                xaxis_title="Balance ($B)",
                yaxis_title="Frequency",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                template="plotly_dark",
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        # Net flow chart
        fig_net = go.Figure()
        fig_net.add_trace(go.Bar(
            x=bal_filtered["timestamp"], y=bal_filtered["total_net"],
            name="Net Flow",
            marker_color=np.where(bal_filtered["total_net"] >= 0, "#66BB6A", "#EF5350"),
            opacity=0.7,
        ))
        fig_net.update_layout(
            title="Net Cash Flow (15-min intervals)",
            yaxis_title="USD",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            template="plotly_dark",
        )
        st.plotly_chart(fig_net, use_container_width=True)
    else:
        st.warning("No data available for selected date range.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: CHANNEL & LOB ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Channel & Business Line Analytics")

    col_ch, col_bl = st.columns(2)

    with col_ch:
        # Channel volume breakdown
        ch_vol = pmt_filtered.groupby("channel").agg(
            total_outflow=("outflow", "sum"),
            total_inflow=("inflow", "sum"),
            total_net=("net_flow", "sum"),
            n_txns=("net_flow", "count"),
        ).reset_index()
        ch_vol["gross_volume"] = ch_vol["total_outflow"] + ch_vol["total_inflow"]

        fig_ch = px.bar(
            ch_vol, x="channel", y=["total_inflow", "total_outflow"],
            barmode="group",
            color_discrete_map={"total_inflow": "#66BB6A", "total_outflow": "#EF5350"},
            title="Volume by Payment Channel",
        )
        fig_ch.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_ch, use_container_width=True)

    with col_bl:
        # Business line breakdown
        bl_vol = pmt_filtered.groupby("business_line").agg(
            total_outflow=("outflow", "sum"),
            total_inflow=("inflow", "sum"),
            total_net=("net_flow", "sum"),
        ).reset_index()

        fig_bl = px.bar(
            bl_vol, x="business_line", y=["total_inflow", "total_outflow"],
            barmode="group",
            color_discrete_map={"total_inflow": "#66BB6A", "total_outflow": "#EF5350"},
            title="Volume by Business Line",
        )
        fig_bl.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_bl, use_container_width=True)

    st.divider()

    # Intraday profile by channel (hourly heatmap)
    st.subheader("Intraday Flow Profile by Channel")
    pmt_hour = pmt_filtered.copy()
    pmt_hour["hour"] = pmt_hour["timestamp"].dt.hour
    heatmap_data = pmt_hour.groupby(["channel", "hour"])["outflow"].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index="channel", columns="hour", values="outflow").fillna(0)

    fig_heat = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values / 1e6,
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        colorscale="YlOrRd",
        colorbar_title="Avg Outflow ($M)",
    ))
    fig_heat.update_layout(
        title="Average Outflow by Channel & Hour (Millions USD)",
        xaxis_title="Hour of Day",
        height=350,
        template="plotly_dark",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Counterparty concentration
    st.subheader("Top Counterparty Exposures")
    cp_net = pmt_filtered.groupby("counterparty")["net_flow"].agg(["sum", "mean", "std", "count"]).reset_index()
    cp_net.columns = ["counterparty", "total_net", "avg_net", "std_net", "n_txns"]
    cp_net["abs_total"] = cp_net["total_net"].abs()
    cp_net = cp_net.sort_values("abs_total", ascending=False).head(10)

    fig_cp = go.Figure()
    colors = ["#EF5350" if v < 0 else "#66BB6A" for v in cp_net["total_net"]]
    fig_cp.add_trace(go.Bar(
        x=cp_net["counterparty"], y=cp_net["total_net"] / 1e9,
        marker_color=colors,
    ))
    fig_cp.update_layout(
        title="Net Exposure by Counterparty (Top 10)",
        yaxis_title="Net Exposure ($B)",
        height=350,
        template="plotly_dark",
    )
    st.plotly_chart(fig_cp, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: BCBS 248 MONITORING
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("BCBS 248 — Intraday Liquidity Monitoring Indicators")
    st.caption("Basel Committee on Banking Supervision | Monitoring Tools for Intraday Liquidity Management (2013)")

    bcbs = generate_bcbs248_summary(pmt_filtered, bal_filtered)

    # Indicator 1: Max Intraday Usage
    st.markdown("#### Indicator 1: Daily Maximum Intraday Liquidity Usage")
    usage_df = bcbs["max_intraday_usage"]
    if len(usage_df) > 0:
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Max Usage", f"${usage_df['max_intraday_usage'].mean() / 1e9:.2f}B")
        c2.metric("Peak Max Usage", f"${usage_df['max_intraday_usage'].max() / 1e9:.2f}B")
        c3.metric("Avg Usage % of Opening", f"{usage_df['max_usage_pct'].mean():.1f}%")

        fig_usage = go.Figure()
        fig_usage.add_trace(go.Bar(
            x=usage_df["date"].astype(str), y=usage_df["max_intraday_usage"] / 1e9,
            marker_color="#FF7043", name="Max Intraday Usage",
        ))
        fig_usage.update_layout(
            title="Daily Maximum Intraday Liquidity Usage",
            yaxis_title="USD (Billions)",
            height=300, template="plotly_dark",
        )
        st.plotly_chart(fig_usage, use_container_width=True)

    st.divider()

    # Indicator 4: Time-Specific Obligations
    st.markdown("#### Indicator 4: Time-Specific & Critical Obligations")
    tso = bcbs["time_specific_obligations"]
    if len(tso) > 0:
        c1, c2 = st.columns(2)
        c1.metric("Avg Daily Critical Outflow", f"${tso['critical_outflow'].mean() / 1e9:.2f}B")
        c2.metric("Critical as % of Total", f"{tso['critical_pct_of_total'].mean():.1f}%")

    st.divider()

    # Indicator 7: Throughput
    st.markdown("#### Indicator 7: Intraday Throughput Profile")
    _, avg_tp = compute_throughput(pmt_filtered)
    if len(avg_tp) > 0:
        fig_tp = make_subplots(specs=[[{"secondary_y": True}]])
        fig_tp.add_trace(go.Bar(
            x=avg_tp["hour"], y=avg_tp["avg_hourly_pct"],
            name="Hourly %", marker_color="#4FC3F7", opacity=0.7,
        ), secondary_y=False)
        fig_tp.add_trace(go.Scatter(
            x=avg_tp["hour"], y=avg_tp["avg_cum_pct"],
            name="Cumulative %", line=dict(color="#FFA726", width=2),
        ), secondary_y=True)
        fig_tp.update_layout(
            title="Payment Throughput: % of Daily Volume Settled by Hour",
            height=350, template="plotly_dark",
        )
        fig_tp.update_yaxes(title_text="Hourly %", secondary_y=False)
        fig_tp.update_yaxes(title_text="Cumulative %", secondary_y=True)
        st.plotly_chart(fig_tp, use_container_width=True)

    # Counterparty exposure table (Indicator 5)
    st.markdown("#### Indicator 5: Largest Counterparty Exposures")
    cp_exp = bcbs["top_counterparties"]
    if len(cp_exp) > 0:
        cp_table = cp_exp.reset_index()
        cp_table.columns = ["Counterparty", "Avg Abs Net Exposure"]
        cp_table["Avg Abs Net Exposure"] = cp_table["Avg Abs Net Exposure"].apply(lambda x: f"${x/1e6:,.1f}M")
        st.dataframe(cp_table, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: FORECASTING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("ML Forecasting Engine — Net Flow Prediction")
    st.caption("GradientBoosting model with 30+ engineered features (calendar, cyclical, rolling, lagged)")

    col_train, col_forecast = st.columns([1, 2])

    with col_train:
        st.markdown("#### Model Training")
        if st.button("Train / Retrain Model", type="primary"):
            st.session_state["model_trained"] = True

    if st.session_state.get("model_trained", False):
        model, feat_df, metrics, importance = train_model("idl_balance_series.csv")

        with col_train:
            st.metric("Train MAE", f"${metrics['train_mae']/1e6:.2f}M",
                      help="Average prediction error on training data — how well the model learned historical patterns")
            st.metric("Test MAE", f"${metrics['test_mae']/1e6:.2f}M",
                      help="Average prediction error on unseen data — the real measure of forecast accuracy in production")
            st.metric("Test RMSE", f"${metrics['test_rmse']/1e6:.2f}M",
                      help="Penalizes large errors more than MAE — shows worst-case forecast misses during unusual events")
            st.metric("Train/Test Split", f"{metrics['train_size']:,} / {metrics['test_size']:,}",
                      help="85% data for training, 15% held out for testing — time-based split, no data leakage")

            st.markdown("**Top Features**")
            _feature_descriptions = {
                "roll_mean_4":   "Avg net flow over last 1 hour — strongest signal for near-term momentum",
                "roll_std_4":    "Flow volatility over last 1 hour — widens confidence intervals when high",
                "roll_max_4":    "Largest single flow in last 1 hour — detects recent payment spikes",
                "roll_min_4":    "Smallest flow in last 1 hour — detects recent large outflows",
                "lag_2d":        "Net flow at this exact time 2 days ago — captures weekly rhythm",
                "minute_of_day": "Position in trading day (0-1) — 10 AM behaves differently than 3 PM",
                "roll_std_8":    "Flow volatility over last 2 hours — broader volatility context",
                "roll_mean_8":   "Avg net flow over last 2 hours — medium-term trend signal",
                "roll_mean_32":  "Avg net flow over last 8 hours — captures full-day directional trend",
                "minute":        "Minute within the hour (0/15/30/45) — some intervals are busier",
            }
            importance_display = importance.head(10).copy()
            importance_display["description"] = importance_display["feature"].map(
                lambda f: _feature_descriptions.get(f, "")
            )
            st.dataframe(
                importance_display.style.format({"importance": "{:.4f}"}),
                use_container_width=True, hide_index=True,
                column_config={"description": st.column_config.TextColumn("Description", width="large")},
            )

        with col_forecast:
            horizon = st.slider("Forecast Horizon (15-min steps)", 16, 96, 48, step=8, key="fc_horizon")

            with st.spinner("Generating forecast..."):
                fc = forecast_forward(model, bal_df, horizon_steps=horizon)

            # Plot
            fig_fc = go.Figure()

            # Historical (last 200 points)
            hist_tail = bal_df.tail(200)
            fig_fc.add_trace(go.Scatter(
                x=hist_tail["timestamp"], y=hist_tail["total_net"],
                name="Historical Net Flow",
                line=dict(color="#4FC3F7", width=1),
            ))

            # Forecast
            fig_fc.add_trace(go.Scatter(
                x=fc["timestamp"], y=fc["forecast"],
                name="ML Forecast",
                line=dict(color="#FFA726", width=2),
            ))

            # Confidence interval
            fig_fc.add_trace(go.Scatter(
                x=pd.concat([fc["timestamp"], fc["timestamp"][::-1]]),
                y=pd.concat([fc["upper"], fc["lower"][::-1]]),
                fill="toself", fillcolor="rgba(255,167,38,0.15)",
                line=dict(width=0),
                name="95% CI",
            ))

            fig_fc.update_layout(
                title="Net Flow: Historical + ML Forecast",
                yaxis_title="USD",
                height=450,
                template="plotly_dark",
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            # Store forecast in session for use in stress/playbook tabs
            st.session_state["forecast_df"] = fc
            st.session_state["model"] = model
    else:
        st.info("Click **Train / Retrain Model** to build the forecasting model.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: STRESS SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Stress Scenario Analysis")

    if "forecast_df" not in st.session_state:
        st.warning("Please train the forecasting model first (Tab 4).")
    else:
        fc = st.session_state["forecast_df"].copy()

        col_scn, col_custom = st.columns([1, 1])

        with col_scn:
            st.markdown("#### Pre-Defined Scenarios")
            scenario_key = st.selectbox(
                "Select Scenario",
                options=list(STRESS_SCENARIOS.keys()),
                format_func=lambda k: STRESS_SCENARIOS[k]["name"],
            )
            scn = STRESS_SCENARIOS[scenario_key]
            st.markdown(f"**Description:** {scn['description']}")
            st.markdown(f"**Inflow Shock:** {scn['inflow_shock_pct']:+d}% &nbsp; | &nbsp; "
                        f"**Outflow Shock:** {scn['outflow_shock_pct']:+d}%")
            st.markdown(f"**Affected Channels:** {', '.join(scn['affected_channels'])}")
            st.markdown(f"**Duration:** {scn['duration_hours']} hours &nbsp; | &nbsp; "
                        f"**Probability:** {scn['probability']}")

        with col_custom:
            st.markdown("#### Custom Adjustments")
            custom_inflow = st.slider("Additional inflow shock (%)", -50, 50, 0, step=5, key="stress_inf")
            custom_outflow = st.slider("Additional outflow shock (%)", -50, 50, 0, step=5, key="stress_out")
            start_balance = st.number_input(
                "Starting Balance ($B)", value=25.0, step=0.5, format="%.1f", key="stress_bal"
            ) * 1e9
            breach_threshold = st.number_input(
                "Breach Threshold ($B)", value=15.0, step=0.5, format="%.1f", key="stress_thresh"
            ) * 1e9

        # Apply scenario + custom adjustments
        fc["forecast_stressed"] = fc["forecast"]  # initialize
        stressed = apply_scenario_stress(fc, scenario_key, start_step=4)

        # Additional custom adjustments on top
        if custom_inflow != 0 or custom_outflow != 0:
            inf_m = 1 + custom_inflow / 100
            out_m = 1 + custom_outflow / 100
            stressed["forecast_stressed"] = stressed["forecast_stressed"].apply(
                lambda v: v * inf_m if v >= 0 else v * out_m
            )

        # Project balances
        b_baseline = start_balance
        baseline_path = []
        for nf in stressed["forecast"].values:
            b_baseline += nf
            baseline_path.append(b_baseline)
        stressed["balance_baseline"] = baseline_path

        b_stressed = start_balance
        stressed_path = []
        for nf in stressed["forecast_stressed"].values:
            b_stressed += nf
            stressed_path.append(b_stressed)
        stressed["balance_stressed"] = stressed_path

        # Find breach
        breach_idx = None
        for i, v in enumerate(stressed["balance_stressed"]):
            if v < breach_threshold:
                breach_idx = i
                break

        # Charts
        fig_stress = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Net Flow: Baseline vs Stressed", "Balance Path: Baseline vs Stressed"],
            row_heights=[0.4, 0.6],
            vertical_spacing=0.12,
        )

        fig_stress.add_trace(go.Scatter(
            x=stressed["timestamp"], y=stressed["forecast"],
            name="Baseline Forecast", line=dict(color="#4FC3F7", width=1),
        ), row=1, col=1)
        fig_stress.add_trace(go.Scatter(
            x=stressed["timestamp"], y=stressed["forecast_stressed"],
            name="Stressed Forecast", line=dict(color="#EF5350", width=2, dash="dash"),
        ), row=1, col=1)

        fig_stress.add_trace(go.Scatter(
            x=stressed["timestamp"], y=stressed["balance_baseline"] / 1e9,
            name="Balance (Baseline)", line=dict(color="#4FC3F7", width=1.5),
        ), row=2, col=1)
        fig_stress.add_trace(go.Scatter(
            x=stressed["timestamp"], y=stressed["balance_stressed"] / 1e9,
            name="Balance (Stressed)", line=dict(color="#EF5350", width=2),
        ), row=2, col=1)
        fig_stress.add_hline(
            y=breach_threshold / 1e9, line=dict(color="red", dash="dot", width=1),
            annotation_text=f"Breach: ${breach_threshold/1e9:.1f}B",
            row=2, col=1,
        )

        if breach_idx is not None:
            fig_stress.add_trace(go.Scatter(
                x=[stressed["timestamp"].iloc[breach_idx]],
                y=[stressed["balance_stressed"].iloc[breach_idx] / 1e9],
                mode="markers", marker=dict(color="red", size=12, symbol="x"),
                name="First Breach",
            ), row=2, col=1)

        fig_stress.update_layout(height=700, template="plotly_dark")
        fig_stress.update_yaxes(title_text="USD", row=1, col=1)
        fig_stress.update_yaxes(title_text="USD (Billions)", row=2, col=1)
        st.plotly_chart(fig_stress, use_container_width=True)

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Start Balance", f"${start_balance / 1e9:.1f}B")
        c2.metric("Min Stressed Balance", f"${min(stressed_path) / 1e9:.2f}B")
        c3.metric("Max Drawdown",
                   f"${(start_balance - min(stressed_path)) / 1e9:.2f}B")
        if breach_idx is not None:
            c4.metric("Breach At", stressed["timestamp"].iloc[breach_idx].strftime("%Y-%m-%d %H:%M"))
        else:
            c4.metric("Breach", "No breach")

        # Commentary
        st.markdown("#### Scenario Commentary")
        min_stressed_bal = min(stressed_path)
        drawdown = start_balance - min_stressed_bal
        scn_name = scn["name"]
        breach_text = (
            f"**A breach is projected at {stressed['timestamp'].iloc[breach_idx].strftime('%H:%M on %b %d')}**, "
            f"when the balance falls to ${stressed['balance_stressed'].iloc[breach_idx]/1e9:.2f}B. "
            f"Immediate playbook activation recommended (see Playbook tab)."
            if breach_idx is not None
            else "No breach is projected under this scenario against the current threshold."
        )
        st.markdown(
            f"Under the **{scn_name}** scenario (inflow {scn['inflow_shock_pct']:+d}%, "
            f"outflow {scn['outflow_shock_pct']:+d}% over {scn['duration_hours']} hours), "
            f"the peak drawdown from starting balance is **${drawdown/1e9:.2f}B** "
            f"({drawdown/start_balance*100:.1f}% of opening). "
            f"Minimum projected balance: **${min_stressed_bal/1e9:.2f}B**. {breach_text}"
        )

        st.session_state["stressed_df"] = stressed
        st.session_state["stress_start_balance"] = start_balance
        st.session_state["breach_threshold"] = breach_threshold


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: IDL PLAYBOOK
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("IDL Playbook — Breach Response Simulation")

    if "stressed_df" not in st.session_state:
        st.warning("Run a stress scenario first (Tab 5) to populate the playbook.")
    else:
        stressed = st.session_state["stressed_df"]
        start_balance = st.session_state["stress_start_balance"]
        threshold = st.session_state["breach_threshold"]

        # Find breach or worst point
        min_bal_idx = np.argmin(stressed["balance_stressed"].values)
        min_bal = stressed["balance_stressed"].iloc[min_bal_idx]
        severity = classify_severity(min_bal, threshold)

        # Severity display
        sev_config = SEVERITY_LEVELS[severity]
        st.markdown(
            f'<div style="background:{sev_config["color"]}; color:white; padding:12px; '
            f'border-radius:6px; font-size:18px; font-weight:bold;">'
            f'Severity: {severity} — {sev_config["description"]}'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.write("")

        col_pb1, col_pb2, col_pb3 = st.columns(3)
        with col_pb1:
            intercompany_amt = st.number_input(
                "Intercompany Funding ($B)", value=2.0, step=0.5, format="%.1f"
            ) * 1e9
        with col_pb2:
            repo_amt = st.number_input(
                "Intraday Repo ($B)", value=3.0, step=0.5, format="%.1f"
            ) * 1e9
        with col_pb3:
            st.markdown(f"**Escalation Path:**")
            for owner in sev_config["escalation"]:
                st.markdown(f"• {owner}")

        if st.button("Run Playbook Simulation", type="primary"):
            breach_time = stressed["timestamp"].iloc[min_bal_idx]

            # Generate timeline
            timeline = generate_escalation_timeline(
                breach_time=breach_time,
                severity=severity,
                intercompany_amount=intercompany_amt,
                repo_amount=repo_amt,
                breach_balance=min_bal,
                threshold=threshold,
            )

            # Project balance with remediation
            ic_step = min(min_bal_idx + 4, len(stressed) - 1)  # ~15 min after breach
            repo_step = min(min_bal_idx + 8, len(stressed) - 1) if severity == "CRITICAL" else None

            remediated = project_balance_with_remediation(
                stressed, start_balance,
                intercompany_amt, repo_amt,
                ic_step, repo_step, severity,
            )

            # Chart
            fig_pb = go.Figure()
            fig_pb.add_trace(go.Scatter(
                x=remediated["timestamp"], y=remediated["balance_no_action"] / 1e9,
                name="No Action", line=dict(color="#EF5350", width=2, dash="dash"),
            ))
            fig_pb.add_trace(go.Scatter(
                x=remediated["timestamp"], y=remediated["balance_with_actions"] / 1e9,
                name="With Playbook Actions", line=dict(color="#66BB6A", width=2.5),
            ))
            fig_pb.add_hline(
                y=threshold / 1e9, line=dict(color="red", dash="dot"),
                annotation_text=f"Threshold: ${threshold/1e9:.1f}B",
            )

            # Mark action points
            if severity in ("ELEVATED", "CRITICAL"):
                fig_pb.add_trace(go.Scatter(
                    x=[remediated["timestamp"].iloc[ic_step]],
                    y=[remediated["balance_with_actions"].iloc[ic_step] / 1e9],
                    mode="markers+text",
                    marker=dict(color="#FFA726", size=12),
                    text=[f"+${intercompany_amt/1e9:.1f}B IC"],
                    textposition="top center",
                    name="Intercompany Funding",
                ))
            if repo_step is not None:
                fig_pb.add_trace(go.Scatter(
                    x=[remediated["timestamp"].iloc[repo_step]],
                    y=[remediated["balance_with_actions"].iloc[repo_step] / 1e9],
                    mode="markers+text",
                    marker=dict(color="#AB47BC", size=12),
                    text=[f"+${repo_amt/1e9:.1f}B Repo"],
                    textposition="top center",
                    name="Intraday Repo",
                ))

            fig_pb.update_layout(
                title="Balance Path: No Action vs Playbook Response",
                yaxis_title="USD (Billions)",
                height=450, template="plotly_dark",
            )
            st.plotly_chart(fig_pb, use_container_width=True)

            # Escalation timeline table
            st.markdown("#### Escalation Timeline")
            st.dataframe(
                timeline[["step", "time", "action", "owner", "detail", "status"]],
                use_container_width=True,
                hide_index=True,
            )

            # Recovery check
            final_bal_no_action = remediated["balance_no_action"].iloc[-1]
            final_bal_action = remediated["balance_with_actions"].iloc[-1]
            recovery = final_bal_action - final_bal_no_action

            st.markdown("#### Post-Playbook Assessment")
            c1, c2, c3 = st.columns(3)
            c1.metric("Final Balance (No Action)", f"${final_bal_no_action/1e9:.2f}B")
            c2.metric("Final Balance (With Actions)", f"${final_bal_action/1e9:.2f}B")
            c3.metric("Recovery from Actions", f"+${recovery/1e9:.2f}B")

            # Download
            st.download_button(
                "Download Incident Report (CSV)",
                timeline.to_csv(index=False).encode("utf-8"),
                file_name="idl_incident_report.csv",
                mime="text/csv",
            )
