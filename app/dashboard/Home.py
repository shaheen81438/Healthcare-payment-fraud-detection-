"""Investigator dashboard — portfolio overview."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import altair as alt
import pandas as pd
import streamlit as st

from app.dashboard.common import (
    ACTION_COLORS,
    BLUE,
    load_decisions,
    load_policy,
    load_predictions,
    money,
    page_setup,
    synthetic_banner,
)

page_setup("Payment-Integrity Portfolio")
synthetic_banner()

preds = load_predictions()
policy = load_policy()
decisions = load_decisions()

flagged = preds[preds["recommended_action"] == "manual_review"]
monitor = preds[preds["recommended_action"] == "monitor"]

# ---- KPI row -------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Claims scored", f"{len(preds):,}")
c2.metric("Flagged for review", f"{len(flagged):,}")
c3.metric("Estimated payment at risk", money(float(flagged["estimated_payment_at_risk"].sum())))
c4.metric("Review threshold τ", f"{policy.get('review_threshold', '—')}")
decided = len(decisions)
c5.metric("Decisions captured", f"{decided:,}")

st.divider()

left, right = st.columns([3, 2])

with left:
    st.subheader("Risk-score distribution")
    hist = (
        alt.Chart(preds.sample(min(len(preds), 30_000), random_state=7))
        .mark_bar(color=BLUE, binSpacing=1)
        .encode(
            x=alt.X("risk_score:Q", bin=alt.Bin(maxbins=40), title="Ensemble risk score"),
            y=alt.Y("count()", title="Claims"),
            tooltip=[alt.Tooltip("count()", title="Claims")],
        )
        .properties(height=260)
    )
    st.altair_chart(hist, use_container_width=True)

with right:
    st.subheader("Recommended actions")
    actions = (
        preds["recommended_action"].value_counts().rename_axis("action").reset_index(name="claims")
    )
    bar = (
        alt.Chart(actions)
        .mark_bar()
        .encode(
            x=alt.X("claims:Q", title="Claims"),
            y=alt.Y("action:N", sort="-x", title=None),
            color=alt.Color(
                "action:N",
                scale=alt.Scale(domain=list(ACTION_COLORS), range=list(ACTION_COLORS.values())),
                legend=None,
            ),
            tooltip=["action", "claims"],
        )
        .properties(height=180)
    )
    st.altair_chart(bar, use_container_width=True)
    st.caption("Colors: orange = manual review, blue = monitor, aqua = auto-clear.")

st.subheader("Flagged dollars by month")
monthly = (
    flagged.assign(month=pd.to_datetime(flagged["service_date"]).dt.to_period("M").astype(str))
    .groupby("month", as_index=False)["estimated_payment_at_risk"]
    .sum()
)
line = (
    alt.Chart(monthly)
    .mark_line(color=BLUE, point=alt.OverlayMarkDef(color=BLUE, size=45))
    .encode(
        x=alt.X("month:N", title=None),
        y=alt.Y("estimated_payment_at_risk:Q", title="Estimated $ at risk"),
        tooltip=["month", alt.Tooltip("estimated_payment_at_risk:Q", format="$,.0f")],
    )
    .properties(height=240)
)
st.altair_chart(line, use_container_width=True)

st.info(
    "Work the **Review Queue** page to triage flagged claims; open **Claim Detail** for SHAP "
    "explanations; use the **Threshold Simulator** to trade capacity against net value.",
    icon="🧭",
)
