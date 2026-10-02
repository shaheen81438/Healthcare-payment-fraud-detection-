"""Threshold simulator: trade investigator capacity against expected net value."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import altair as alt
import pandas as pd
import streamlit as st

from app.dashboard.common import (
    BLUE,
    INK_2,
    ORANGE,
    load_labels_research_mode,
    load_policy,
    load_predictions,
    money,
    page_setup,
    synthetic_banner,
)
from src.evaluation.financial import CostParams, capacity_sweep, threshold_sweep

page_setup("Threshold Simulator")
synthetic_banner()
st.caption(
    "All levers below are scenario assumptions. Realized values use the synthetic ground-truth "
    "labels (research mode). A production deployment would use investigator outcomes instead."
)

preds = load_predictions()
policy = load_policy()
labels = load_labels_research_mode()
frame = preds.merge(labels, on="claim_id", how="left")
frame["is_improper"] = frame["is_improper"].fillna(False)
frame["improper_amount"] = frame["improper_amount"].fillna(0.0)

# ---- levers --------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
capacity_pct = c1.slider("Investigator capacity (% of claims)", 0.1, 10.0, 1.0, 0.1)
review_cost = c2.number_input("Review cost per claim ($)", 5.0, 200.0, 30.0, 5.0)
recovery_rate = c3.slider("Recovery rate", 0.1, 1.0, 0.62, 0.02)
fp_cost = c4.number_input("False-positive extra cost ($)", 0.0, 100.0, 12.0, 2.0)
min_threshold = c5.slider("Minimum risk threshold", 0.0, 0.99, 0.20, 0.01)

params = CostParams(
    review_cost=float(review_cost),
    recovery_rate=float(recovery_rate),
    fp_extra_cost=float(fp_cost),
    min_risk_threshold=float(min_threshold),
    monitor_fraction=0.5,
)
args = (
    frame["risk_score"].to_numpy(),
    frame["supervised_probability"].to_numpy(),
    frame["paid_amount"].to_numpy(dtype=float),
    frame["is_improper"].to_numpy(dtype=int),
    frame["improper_amount"].to_numpy(dtype=float),
    params,
)

# ---- chosen operating point ---------------------------------------------
cap_row = capacity_sweep(*args, capacities=[capacity_pct / 100]).iloc[0]
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Claims reviewed", f"{int(cap_row['n_reviewed']):,}")
m2.metric("Precision", f"{cap_row['precision']:.1%}")
m3.metric("Recall of improper claims", f"{cap_row['recall']:.1%}")
m4.metric("Net value (realized)", money(float(cap_row["net_value"])))
m5.metric(
    "$ identified / 1,000 reviews", money(float(cap_row["dollars_identified_per_1000_reviewed"]))
)
st.caption(
    f"Implied review threshold at this capacity: **τ = {cap_row['review_threshold']:.3f}** "
    f"(production policy currently τ = {policy.get('review_threshold')})."
)

# ---- ENV curve -----------------------------------------------------------
sweep = threshold_sweep(*args, n_grid=60)
base = alt.Chart(sweep).encode(x=alt.X("threshold:Q", title="Review threshold τ"))
env_line = base.mark_line(color=BLUE).encode(
    y=alt.Y("net_value:Q", title="Net value ($)"),
    tooltip=[
        alt.Tooltip("threshold:Q", format=".2f"),
        alt.Tooltip("net_value:Q", format="$,.0f"),
        "n_reviewed:Q",
    ],
)
exp_line = base.mark_line(color=INK_2, strokeDash=[4, 3]).encode(y="expected_net_value:Q")
tau_rule = (
    alt.Chart(pd.DataFrame({"tau": [float(cap_row["review_threshold"])]}))
    .mark_rule(color=ORANGE, strokeWidth=2)
    .encode(x="tau:Q")
)
st.subheader("Net value vs threshold")
st.altair_chart((env_line + exp_line + tau_rule).properties(height=300), use_container_width=True)
st.caption(
    "Solid blue: realized net value · dashed gray: expected (probability-based) · orange rule: your capacity's threshold."
)

# ---- capacity table ------------------------------------------------------
st.subheader("Capacity scenarios")
cap_grid = capacity_sweep(*args, capacities=[0.005, 0.01, 0.02, 0.05, 0.10])
display = cap_grid[
    ["capacity_share", "n_reviewed", "precision", "recall", "dollars_recovered", "net_value"]
].copy()
display["capacity_share"] = (display["capacity_share"] * 100).map("{:.1f}%".format)
st.dataframe(
    display.rename(
        columns={
            "capacity_share": "Capacity",
            "n_reviewed": "Reviewed",
            "precision": "Precision",
            "recall": "Recall",
            "dollars_recovered": "$ recovered",
            "net_value": "Net value",
        }
    ),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Precision": st.column_config.NumberColumn(format="%.3f"),
        "Recall": st.column_config.NumberColumn(format="%.3f"),
        "$ recovered": st.column_config.NumberColumn(format="$%.0f"),
        "Net value": st.column_config.NumberColumn(format="$%.0f"),
    },
)

# ---- marginal-value intuition -------------------------------------------
with st.expander("How the recommended threshold is chosen"):
    st.markdown(
        """
The production policy takes the **highest** of three cutoffs:

1. **ENV-optimal τ**: where the expected marginal review value turns negative
   (`p·recovery·paid − review_cost − (1−p)·fp_cost < 0`);
2. **capacity-implied τ**: the score of the last claim that fits the review budget;
3. the configured **risk floor**, which protects precision even when capacity is left over.

Raising recovery rate or paid amounts pushes the ENV-optimal τ down (more claims are
worth reviewing); tightening capacity pushes the operating τ up.
"""
    )
    top_frame = frame.nlargest(int(len(frame) * capacity_pct / 100), "risk_score")
    marginal = (
        params.recovery_rate
        * top_frame["supervised_probability"].to_numpy()
        * top_frame["paid_amount"].to_numpy()
        - params.review_cost
        - (1 - top_frame["supervised_probability"].to_numpy()) * params.fp_extra_cost
    )
    st.caption(
        f"At your capacity, the marginal (last) reviewed claim has expected value "
        f"{money(float(marginal[-1] if len(marginal) else 0))}. A negative value means capacity "
        "is larger than the queue the economics justify."
    )
