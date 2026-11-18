"""Availability-aware investigation-priority aggregation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import plotly.express as px
import streamlit as st

from app.dashboard.common import load_predictions, page_setup, synthetic_banner
from src.payment_integrity.improper_payment import evaluate_payment_row
from src.payment_integrity.risk_assessment import aggregate_risk

page_setup("Risk Assessment")
synthetic_banner()
st.info(
    "The combined result is an investigation priority. Component scores remain visible, "
    "and unavailable signals are reported rather than treated as zero."
)

predictions = load_predictions()
if predictions.empty:
    st.warning("No scored claims are available.")
    st.stop()

claim_id = st.selectbox("Claim", predictions["claim_id"].astype(str).tolist())
row = predictions[predictions["claim_id"].astype(str) == claim_id].iloc[0]
payment = evaluate_payment_row(row.to_dict())
assessment = aggregate_risk(
    improper_payment_score=payment["issue_score"],
    supervised_model_score=float(row["risk_score"]) if "risk_score" in row else None,
)

overall = assessment["overall_investigation_priority"]
st.metric("Overall investigation priority", "Unavailable" if overall is None else f"{overall:.3f}")
st.caption(assessment["interpretation"])

component_rows = [
    {
        "component": name.replace("_", " ").title(),
        "available": detail["available"],
        "score": detail["score"],
    }
    for name, detail in assessment["components"].items()
]
st.dataframe(component_rows, use_container_width=True, hide_index=True)
available = [item for item in component_rows if item["available"]]
if available:
    st.plotly_chart(
        px.bar(
            available,
            x="component",
            y="score",
            range_y=[0, 1],
            title="Normalized evidence components",
        ),
        use_container_width=True,
    )
if assessment["reason_codes"]:
    st.write("Reason components:", ", ".join(assessment["reason_codes"]))
if assessment["data_quality_warnings"]:
    st.warning("; ".join(assessment["data_quality_warnings"]))
