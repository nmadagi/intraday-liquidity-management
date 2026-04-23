"""
playbook.py — IDL Playbook: Breach Response & Stress Scenarios
================================================================
Scenario-driven playbook with:
  • Severity classification (Advisory / Elevated / Critical)
  • Pre-defined stress scenarios (counterparty failure, CCP margin spike, etc.)
  • Escalation matrix with role-based owners
  • Remediation actions with projected balance recovery
  • Post-mortem template
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


# ─── Severity Levels ─────────────────────────────────────────────────────────
SEVERITY_LEVELS = {
    "ADVISORY": {
        "color": "#FFA726",
        "threshold_pct": 80,  # Balance within 80-100% of limit
        "description": "Balance approaching limit. Monitor closely.",
        "escalation": ["IDL Manager"],
        "actions": ["Increase monitoring frequency", "Alert LOB contacts"],
    },
    "ELEVATED": {
        "color": "#EF5350",
        "threshold_pct": 60,  # Balance at 60-80% of limit
        "description": "Balance significantly below target. Action required.",
        "escalation": ["IDL Manager", "Treasury Head", "LOB Heads"],
        "actions": [
            "Throttle non-critical outgoing payments",
            "Accelerate expected inflows where possible",
            "Prepare intercompany funding",
            "Notify Fed liaison (if daylight overdraft near cap)",
        ],
    },
    "CRITICAL": {
        "color": "#B71C1C",
        "threshold_pct": 40,  # Balance below 40% of limit
        "description": "Breach imminent or occurred. Emergency response.",
        "escalation": [
            "IDL Manager", "Treasury Head", "CFO Office",
            "LOB Heads", "Operations", "Fed Liaison", "Legal/Compliance"
        ],
        "actions": [
            "HALT all non-critical wires immediately",
            "Execute intercompany cash movement",
            "Execute intraday repo facility",
            "Contact counterparties to expedite incoming payments",
            "Notify Federal Reserve (if daylight overdraft cap breached)",
            "Activate contingency funding plan",
            "Begin incident documentation",
        ],
    },
}


# ─── Pre-defined Stress Scenarios ────────────────────────────────────────────
STRESS_SCENARIOS = {
    "counterparty_delay": {
        "name": "Major Counterparty Payment Delay",
        "description": "A top-5 counterparty delays $3B+ in expected inflows by 3+ hours",
        "inflow_shock_pct": -35,
        "outflow_shock_pct": 0,
        "affected_channels": ["FEDWIRE", "CHIPS"],
        "duration_hours": 4,
        "probability": "Medium",
    },
    "ccp_margin_spike": {
        "name": "CCP Variation Margin Spike",
        "description": "Market volatility triggers 2x normal variation margin calls from CME/ICE",
        "inflow_shock_pct": 0,
        "outflow_shock_pct": 40,
        "affected_channels": ["CCP_MARGIN"],
        "duration_hours": 6,
        "probability": "Medium-High",
    },
    "operational_failure": {
        "name": "Payment System Operational Failure",
        "description": "Internal systems issue delays all ACH batch processing by 4 hours",
        "inflow_shock_pct": -20,
        "outflow_shock_pct": 10,
        "affected_channels": ["ACH"],
        "duration_hours": 4,
        "probability": "Low",
    },
    "market_stress": {
        "name": "Broad Market Stress Event",
        "description": "Equity market drop >5% triggers margin calls, flight-to-safety flows, "
                       "and elevated Treasury settlement volumes",
        "inflow_shock_pct": -15,
        "outflow_shock_pct": 30,
        "affected_channels": ["CCP_MARGIN", "FED_SECURITIES", "FEDWIRE"],
        "duration_hours": 8,
        "probability": "Low-Medium",
    },
    "month_end_surge": {
        "name": "Month-End Settlement Surge",
        "description": "Elevated month-end mortgage, payroll, and bond settlement flows "
                       "exceed forecast by 50%",
        "inflow_shock_pct": 15,
        "outflow_shock_pct": 45,
        "affected_channels": ["FEDWIRE", "ACH", "FED_SECURITIES"],
        "duration_hours": 10,
        "probability": "High (monthly)",
    },
    "correspondent_default": {
        "name": "Correspondent Bank Stress",
        "description": "A correspondent bank faces liquidity issues, delaying nostro settlements",
        "inflow_shock_pct": -25,
        "outflow_shock_pct": 5,
        "affected_channels": ["FEDWIRE", "CHIPS"],
        "duration_hours": 24,
        "probability": "Low",
    },
}


def classify_severity(current_balance: float, target_balance: float) -> str:
    """Classify severity based on how far balance has fallen below target.
    If balance >= target: no issue (return ADVISORY for monitoring).
    If balance < target but > 60% of target: ELEVATED.
    If balance < 60% of target: CRITICAL.
    """
    if target_balance <= 0:
        return "CRITICAL"
    ratio = current_balance / target_balance * 100
    if ratio >= 100:
        return "ADVISORY"  # above threshold, just monitoring
    elif ratio >= 60:
        return "ELEVATED"
    else:
        return "CRITICAL"


def apply_scenario_stress(
    forecast_df: pd.DataFrame,
    scenario_key: str,
    start_step: int = 0,
) -> pd.DataFrame:
    """
    Apply a pre-defined stress scenario to the forecast.
    """
    scenario = STRESS_SCENARIOS[scenario_key]
    out = forecast_df.copy()

    inflow_mult = 1.0 + (scenario["inflow_shock_pct"] / 100.0)
    outflow_mult = 1.0 + (scenario["outflow_shock_pct"] / 100.0)

    duration_steps = scenario["duration_hours"] * 4  # 15-min intervals
    end_step = min(start_step + duration_steps, len(out))

    # Apply stress to the affected window
    for i in range(start_step, end_step):
        nf = out.loc[out.index[i], "forecast"]
        if nf >= 0:
            out.loc[out.index[i], "forecast_stressed"] = nf * inflow_mult
        else:
            out.loc[out.index[i], "forecast_stressed"] = nf * outflow_mult

    # Outside stress window, stressed = baseline
    mask_outside = (out.index < start_step) | (out.index >= end_step)
    out.loc[mask_outside, "forecast_stressed"] = out.loc[mask_outside, "forecast"]

    return out


def generate_escalation_timeline(
    breach_time: pd.Timestamp,
    severity: str,
    intercompany_amount: float = 2_000_000_000,
    repo_amount: float = 3_000_000_000,
    breach_balance: float = 0,
    threshold: float = 0,
) -> pd.DataFrame:
    """
    Generate escalation timeline based on severity.
    """
    config = SEVERITY_LEVELS[severity]
    t0 = breach_time

    steps = []

    # Step 1: Detection (always)
    steps.append({
        "time": t0,
        "step": 1,
        "action": "Detect anomaly / breach trigger",
        "owner": "IDL Manager (1st Line)",
        "detail": f"Severity: {severity}. Balance ${breach_balance:,.0f} vs threshold ${threshold:,.0f}.",
        "status": "Auto-detected",
    })

    # Step 2: Confirm (not a data error)
    steps.append({
        "time": t0 + pd.Timedelta(minutes=2),
        "step": 2,
        "action": "Validate — confirm not data/system error",
        "owner": "IDL Manager",
        "detail": "Cross-check IDL application, Fedwire, CHIPS dashboards.",
        "status": "Manual check",
    })

    if severity in ("ELEVATED", "CRITICAL"):
        # Step 3: Escalate
        steps.append({
            "time": t0 + pd.Timedelta(minutes=5),
            "step": 3,
            "action": "Escalate to stakeholders",
            "owner": "IDL Manager",
            "detail": f"Notify: {', '.join(config['escalation'])}. Share real-time snapshot.",
            "status": "Notification",
        })

        # Step 4: Throttle payments
        steps.append({
            "time": t0 + pd.Timedelta(minutes=8),
            "step": 4,
            "action": "Throttle non-critical outgoing payments",
            "owner": "Operations / LOB Ops",
            "detail": "Prioritize: CCP margin > FMI settlement > Client-facing > Intercompany > Discretionary.",
            "status": "Action",
        })

        # Step 5: Intercompany funding
        steps.append({
            "time": t0 + pd.Timedelta(minutes=15),
            "step": 5,
            "action": "Execute intercompany cash movement",
            "owner": "Treasury Funding Desk",
            "detail": f"Move ${intercompany_amount:,.0f} from broker-dealer entity to bank entity. Arm's length pricing.",
            "status": "Funding",
        })

    if severity == "CRITICAL":
        # Step 6: Repo facility
        steps.append({
            "time": t0 + pd.Timedelta(minutes=25),
            "step": 6,
            "action": "Activate intraday repo facility",
            "owner": "Treasury Funding Desk",
            "detail": f"Raise ${repo_amount:,.0f} via intraday repo against HQLA collateral.",
            "status": "Contingency",
        })

        # Step 7: Fed notification
        steps.append({
            "time": t0 + pd.Timedelta(minutes=30),
            "step": 7,
            "action": "Notify Federal Reserve (if daylight overdraft cap approached)",
            "owner": "Treasury / Fed Liaison",
            "detail": "Provide ETA for balance recovery. Request temporary cap flexibility if needed.",
            "status": "Regulatory",
        })

    # Final step: Document
    steps.append({
        "time": t0 + pd.Timedelta(minutes=60),
        "step": len(steps) + 1,
        "action": "Post-mortem & incident documentation",
        "owner": "IDL Manager",
        "detail": "Root cause analysis. Update playbook if needed. File with 2nd Line (LRM).",
        "status": "Close-out",
    })

    return pd.DataFrame(steps)


def project_balance_with_remediation(
    forecast_df: pd.DataFrame,
    start_balance: float,
    intercompany_amount: float,
    repo_amount: float,
    intercompany_step: int,
    repo_step: Optional[int],
    severity: str,
) -> pd.DataFrame:
    """
    Project balance under stress with playbook remediation actions applied.
    """
    out = forecast_df.copy()
    netflows = out["forecast_stressed"].values if "forecast_stressed" in out.columns else out["forecast"].values

    # No-action path
    b_no_action = start_balance
    no_action_path = []
    for nf in netflows:
        b_no_action += nf
        no_action_path.append(b_no_action)
    out["balance_no_action"] = no_action_path

    # With-action path
    b_action = start_balance
    action_path = []
    for i, nf in enumerate(netflows):
        b_action += nf
        if i == intercompany_step and severity in ("ELEVATED", "CRITICAL"):
            b_action += intercompany_amount
        if repo_step is not None and i == repo_step and severity == "CRITICAL":
            b_action += repo_amount
        action_path.append(b_action)
    out["balance_with_actions"] = action_path

    return out
