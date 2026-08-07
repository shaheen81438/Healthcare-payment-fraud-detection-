"""Capacity-aware review queue with decision capture."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import json

import streamlit as st

from app.dashboard.common import (
    decision_status_map,
    load_policy,
    load_predictions,
    money,
    page_setup,
    record_decision,
    synthetic_banner,
)

page_setup("Review Queue")
synthetic_banner()

preds = load_predictions()
policy = load_policy()
status_map = decision_status_map()

# ---- filters -------------------------------------------------------------
f1, f2, f3, f4 = st.columns([1.2, 1.2, 1, 1])
action = f1.multiselect(
    "Action", ["manual_review", "monitor", "auto_clear"], default=["manual_review"]
)
specialties = f2.multiselect("Specialty", sorted(preds["specialty"].unique().tolist()))
min_score = f3.slider("Min risk score", 0.0, 1.0, 0.0, 0.05)
capacity = f4.number_input("Queue size (capacity)", 50, 5000, 500, step=50)

queue = preds[preds["recommended_action"].isin(action)] if action else preds
if specialties:
    queue = queue[queue["specialty"].isin(specialties)]
queue = queue[queue["risk_score"] >= min_score]
queue = queue.sort_values("risk_score", ascending=False).head(int(capacity)).copy()
queue["status"] = queue["claim_id"].map(status_map).fillna("pending")
queue["reasons"] = queue["reason_codes"].map(
    lambda raw: ", ".join(json.loads(raw)) if isinstance(raw, str) else ""
)

st.caption(
    f"{len(queue):,} claims in queue · estimated payment at risk "
    f"**{money(float(queue['estimated_payment_at_risk'].sum()))}** · review threshold "
    f"τ={policy.get('review_threshold')}"
)

display_cols = {
    "claim_id": "Claim",
    "provider_id": "Provider",
    "specialty": "Specialty",
    "hcpcs": "Code",
    "service_date": "Service date",
    "paid_amount": "Paid",
    "risk_score": "Risk",
    "estimated_payment_at_risk": "Est. $ at risk",
    "reasons": "Reasons",
    "status": "Status",
}
selection = st.dataframe(
    queue[list(display_cols)].rename(columns=display_cols),
    use_container_width=True,
    hide_index=True,
    height=430,
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "Paid": st.column_config.NumberColumn(format="$%.2f"),
        "Est. $ at risk": st.column_config.NumberColumn(format="$%.2f"),
        "Risk": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.3f"),
        "Service date": st.column_config.DateColumn(),
    },
)

rows = selection.selection.rows if hasattr(selection, "selection") else []
if rows:
    picked = queue.iloc[rows[0]]
    st.session_state["selected_claim_id"] = str(picked["claim_id"])
    st.subheader(f"Decision — {picked['claim_id']}")
    d1, d2, d3 = st.columns([1, 1, 2])
    investigator = d1.text_input("Investigator", value=st.session_state.get("investigator", ""))
    note = d2.text_input("Note", value="")
    with d3:
        b1, b2, b3, b4 = st.columns(4)
        decided = None
        if b1.button("✅ Approve (improper)", use_container_width=True):
            decided = "approved"
        if b2.button("🚫 Dismiss", use_container_width=True):
            decided = "dismissed"
        if b3.button("⏳ Keep pending", use_container_width=True):
            decided = "pending"
        if b4.button("🔎 Open detail", use_container_width=True):
            st.switch_page("pages/2_Claim_Detail.py")
    if decided:
        st.session_state["investigator"] = investigator
        record_decision(
            str(picked["claim_id"]), decided, investigator, note, str(picked["model_version"])
        )
        st.success(f"{picked['claim_id']} marked **{decided}**.")
        st.rerun()
else:
    st.caption("Select a row to capture a decision or open the claim detail page.")

with st.expander("Decision log"):
    from app.dashboard.common import load_decisions

    log = load_decisions()
    if log.empty:
        st.write("No decisions captured yet.")
    else:
        st.dataframe(
            log.sort_values("decided_at", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
    if not log.empty:
        counts = log["status"].value_counts()
        st.caption(
            " · ".join(f"{status}: {count}" for status, count in counts.items())
            + " — these outcomes feed the delayed-performance monitoring report."
        )
