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


def read_report(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        st.warning(f"Report `{path}` could not be read: {exc}")
        return {}


# ---- drift status --------------------------------------------------------
st.subheader("Drift status")
if status_path.exists() and (status := read_report(status_path)):
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
    n_drifted = status.get("n_drifted_features")
    n_checked = status.get("n_features_checked")
    score_psi = status.get("score_psi")
    provider_jsd = status.get("provider_mix_jsd")
    missing_alerts = status.get("missing_value_alerts")
    cols[0].metric(
        "Drifted features",
        f"{n_drifted}/{n_checked}"
        if n_drifted is not None and n_checked is not None
        else "Unavailable",
    )
    cols[1].metric(
        "Risk-score PSI",
        f"{score_psi:.3f}" if isinstance(score_psi, (int, float)) else "Unavailable",
    )
    cols[2].metric(
        "Provider-mix divergence",
        f"{provider_jsd:.3f}" if isinstance(provider_jsd, (int, float)) else "Unavailable",
    )
    cols[3].metric(
        "Missing-value alerts", str(missing_alerts) if missing_alerts is not None else "Unavailable"
    )
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
report = read_report(validation_path) if validation_path.exists() else {}
if report and not {
    "rows_in",
    "rows_quarantined",
    "quarantine_share",
    "within_tolerance",
    "checks",
} <= set(report):
    st.warning("The validation report is incomplete or incompatible.")
    report = {}
if report:
    v1, v2, v3 = st.columns(3)
    v1.metric("Rows in", f"{report['rows_in']:,}")
    v2.metric("Quarantined", f"{report['rows_quarantined']:,} ({report['quarantine_share']:.2%})")
    v3.metric("Within tolerance", "yes" if report["within_tolerance"] else "NO")
    checks = pd.DataFrame([{"check": k, **v} for k, v in report["checks"].items()]).sort_values(
        ["severity", "failed"], ascending=[True, False]
    )
    st.dataframe(checks, use_container_width=True, hide_index=True)
else:
    st.caption("No validation report found. Run `make data`.")

# ---- delayed performance from investigator decisions --------------------
st.subheader("Delayed performance (investigator outcomes)")
decisions = load_decisions()
if decisions.empty:
    st.caption(
        "No investigator decisions yet. Approve or dismiss claims in the Review Queue and this "
        "section becomes the realized-precision report."
    )
else:
    predictions = load_predictions()
    if predictions.empty or not {"claim_id", "risk_score", "estimated_payment_at_risk"} <= set(
        predictions
    ):
        st.info("Scored claim data is unavailable for joining investigator outcomes.")
        st.stop()
    preds = predictions[["claim_id", "risk_score", "estimated_payment_at_risk"]]
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
