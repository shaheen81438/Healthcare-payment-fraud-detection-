"""Monitoring status: drift reports, data quality, delayed performance."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import json

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from app.dashboard.common import (
    STATUS,
    load_decisions,
    load_predictions,
    page_setup,
    synthetic_banner,
)

page_setup("Monitoring")
synthetic_banner()

REPORTS = Path("monitoring/reports")
status_path = REPORTS / "monitoring_status.json"

# ---- drift status --------------------------------------------------------
st.subheader("Drift status")
if status_path.exists():
    status = json.loads(status_path.read_text())
    level = str(status.get("status", "unknown")).lower()
    badge = {"ok": "good", "warn": "warning", "critical": "critical"}.get(level, "serious")
    st.markdown(
        f"<span style='background:{STATUS[badge]};color:white;padding:4px 12px;"
        f"border-radius:12px;font-weight:600'>{'✔' if level == 'ok' else '⚠'} {level.upper()}</span>"
        f"&nbsp;&nbsp;run {status.get('run_at', '')} · reference {status.get('reference_window', '')} "
        f"vs current {status.get('current_window', '')}",
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    cols[0].metric(
        "Drifted features",
        f"{status.get('n_drifted_features', 0)}/{status.get('n_features_checked', 0)}",
    )
    cols[1].metric("Risk-score PSI", f"{status.get('score_psi', 0):.3f}")
    cols[2].metric("Provider-mix divergence", f"{status.get('provider_mix_jsd', 0):.3f}")
    cols[3].metric("Missing-value alerts", str(status.get("missing_value_alerts", 0)))
    if status.get("alerts"):
        for alert in status["alerts"]:
            st.warning(alert, icon="🚨")
    html_report = status.get("html_report")
    if html_report and Path(html_report).exists():
        st.caption(f"Full Evidently report: `{html_report}`")
        with st.expander("Open Evidently report inline"):
            components.html(Path(html_report).read_text(), height=650, scrolling=True)
else:
    st.info(
        "No monitoring run found. Generate one with `make monitor` "
        "(or `python scripts/run_monitoring.py`, add `--simulate-drift` to demo alerts).",
        icon="🛰️",
    )

# ---- data quality (latest validation run) -------------------------------
st.subheader("Latest data-validation run")
validation_path = REPORTS / "validation_report.json"
if validation_path.exists():
    report = json.loads(validation_path.read_text())
    v1, v2, v3 = st.columns(3)
    v1.metric("Rows in", f"{report['rows_in']:,}")
    v2.metric("Quarantined", f"{report['rows_quarantined']:,} ({report['quarantine_share']:.2%})")
    v3.metric("Within tolerance", "yes" if report["within_tolerance"] else "NO")
    checks = pd.DataFrame([{"check": k, **v} for k, v in report["checks"].items()]).sort_values(
        ["severity", "failed"], ascending=[True, False]
    )
    st.dataframe(checks, use_container_width=True, hide_index=True)
else:
    st.caption("No validation report found — run `make data`.")

# ---- delayed performance from investigator decisions --------------------
st.subheader("Delayed performance (investigator outcomes)")
decisions = load_decisions()
if decisions.empty:
    st.caption(
        "No investigator decisions yet — approve/dismiss claims in the Review Queue and this "
        "section becomes the realized-precision report."
    )
else:
    preds = load_predictions()[["claim_id", "risk_score", "estimated_payment_at_risk"]]
    joined = decisions.merge(preds, on="claim_id", how="left")
    decided = joined[joined["status"].isin(["approved", "dismissed"])]
    if decided.empty:
        st.caption("Only pending decisions so far.")
    else:
        precision = float((decided["status"] == "approved").mean())
        p1, p2, p3 = st.columns(3)
        p1.metric("Decisions", f"{len(decided):,}")
        p2.metric("Realized flag precision", f"{precision:.1%}")
        p3.metric(
            "Confirmed $ at risk",
            f"${decided.loc[decided['status'] == 'approved', 'estimated_payment_at_risk'].sum():,.0f}",
        )
        st.dataframe(
            joined.sort_values("decided_at", ascending=False).head(25),
            use_container_width=True,
            hide_index=True,
        )
