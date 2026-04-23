# 🏦 Intraday Liquidity Management — IDL Dashboard v2

**USD Central Bank Balance Monitoring | North America Treasury Funding**

Python | Streamlit | Plotly | Scikit-learn | Pandas

---

## Overview

Enterprise-grade intraday liquidity management dashboard designed for monitoring, forecasting, and stress-testing USD payment flows across Fedwire, CHIPS, ACH, Fed Securities, and CCP margin channels.

Built to demonstrate IDL management capabilities aligned with **BCBS 248** regulatory requirements and JPMorgan's Intraday Liquidity framework.

## Features

### 📊 Executive Summary
- Real-time Fed reserve balance monitoring with intraday granularity
- KPIs: current balance, min/max usage, total flows, balance distribution

### 🔀 Channel & Business Line Analytics
- Per-channel volume breakdown (Fedwire, CHIPS, ACH, Fed Securities, CCP Margin)
- Per-business-line flow analysis (Markets, Treasury, Commercial Banking, Asset Mgmt, Retail)
- Intraday flow heatmap by channel and hour
- Top counterparty exposure concentration

### 📋 BCBS 248 Regulatory Monitoring
- **Indicator 1:** Daily maximum intraday liquidity usage
- **Indicator 2:** Available intraday liquidity
- **Indicator 3:** Total payments (by channel)
- **Indicator 4:** Time-specific and critical obligations
- **Indicator 5:** Largest counterparty exposures
- **Indicator 7:** Intraday throughput profile (% settled by hour)

### 🔮 ML Forecasting Engine
- GradientBoosting model with 30+ engineered features
- Calendar features (hour, day-of-week, month-end, quarter-end)
- Cyclical encoding (sine/cosine for periodicity)
- Rolling statistics (mean, std, min, max at multiple windows)
- Lagged features (1-day, 2-day, 5-day lookback)
- 95% confidence intervals
- Feature importance analysis

### 🧪 Stress Scenarios
Six pre-defined operational scenarios:
1. **Major Counterparty Payment Delay** — top-5 counterparty delays $3B+ inflows
2. **CCP Variation Margin Spike** — 2x normal margin calls from market volatility
3. **Payment System Operational Failure** — ACH batch processing delays
4. **Broad Market Stress Event** — equity drop triggers cascading margin calls
5. **Month-End Settlement Surge** — elevated settlement flows exceed forecast
6. **Correspondent Bank Stress** — nostro settlement delays

Custom stress adjustments on top of any scenario.

### 🧭 IDL Playbook — Breach Response
- Severity classification: Advisory / Elevated / Critical
- Role-based escalation matrix
- Projected balance with and without remediation actions
- Intercompany funding and intraday repo facility simulation
- Incident report generation (downloadable CSV)

## Data Architecture

Realistic synthetic data generator mimicking real-world USD payment patterns:
- **Fedwire:** RTGS, bimodal peaks at open and close
- **CHIPS:** Multilateral netting, pre-funding AM, net settlement ~4:30 PM
- **ACH:** Batch windows at 6 AM, 12 PM, 4 PM
- **Fed Securities:** Treasury/agency settlement, 10 AM–2 PM concentration
- **CCP Margin:** Variation margin, AM & PM windows

Includes day-of-week effects, month-end/quarter-end surges, counterparty concentration, and time-critical obligation flags.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

## Project Structure

```
├── app.py                  # Main Streamlit dashboard (6-tab layout)
├── generate_data.py        # Realistic synthetic data generator
├── forecasting.py          # ML forecasting engine (GradientBoosting)
├── bcbs248.py              # BCBS 248 monitoring indicators
├── playbook.py             # IDL playbook & stress scenarios
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Tech Stack

- **Python 3.10+**
- **Streamlit** — interactive dashboard framework
- **Plotly** — interactive visualizations
- **Scikit-learn** — GradientBoosting forecasting model
- **Pandas / NumPy** — data processing and feature engineering

## Regulatory References

- [BCBS 248: Monitoring Tools for Intraday Liquidity Management (2013)](https://www.bis.org/publ/bcbs248.htm)
- [ECB: Sound Practices for Intraday Liquidity Risk Management (2024)](https://www.bankingsupervision.europa.eu/press/supervisory-newsletters/newsletter/2024/html/ssm.nl241113_2.en.html)

---

*Built by Nitin Madagi | Quantitative Risk & Finance*
