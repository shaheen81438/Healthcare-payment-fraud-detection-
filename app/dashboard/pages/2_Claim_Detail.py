"""Claim detail: SHAP explanation, reason codes, similar claims, peer comparison."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import json

import altair as alt
import pandas as pd
import streamlit as st

from app.dashboard.common import (
    BLUE,
    ORANGE,
    load_claims,
    load_explanations,
    load_peer_stats,
    load_predictions,
    page_setup,
    record_decision,
    synthetic_banner,
)

page_setup("Claim Detail")
synthetic_banner()

preds = load_predictions()
default_id = st.session_state.get("selected_claim_id", preds.iloc[0]["claim_id"])
claim_id = st.text_input("Claim ID", value=str(default_id))
row = preds[preds["claim_id"] == claim_id]
if row.empty:
    st.warning("Claim not found in the scored window. Pick one from the Review Queue.")
    st.stop()
claim = row.iloc[0]

# ---- headline tiles ------------------------------------------------------
t1, t2, t3, t4, t5 = st.columns(5)
t1.metric("Risk score", f"{claim['risk_score']:.3f}")
t2.metric("Action", claim["recommended_action"].replace("_", " "))
t3.metric("Est. payment at risk", f"${claim['estimated_payment_at_risk']:,.2f}")
t4.metric("Paid amount", f"${claim['paid_amount']:,.2f}")
t5.metric("Model", str(claim["model_version"]).removeprefix("claims-risk-"))

reasons = json.loads(claim["reason_codes"]) if isinstance(claim["reason_codes"], str) else []
if reasons:
    st.markdown("**Reason codes:** " + " · ".join(f"`{code}`" for code in reasons))
rule_hits = [r for r in str(claim.get("rule_hits", "")).split(";") if r]
if rule_hits:
    st.caption("Rule hits: " + ", ".join(rule_hits))

left, right = st.columns([3, 2])

# ---- SHAP waterfall-style bar -------------------------------------------
with left:
    st.subheader("Why this score (SHAP)")
    ex = load_explanations()
    ex_row = ex[ex["claim_id"] == claim_id]
    if ex_row.empty:
        st.info("No stored explanation; use POST /explain for on-demand explanations.")
    else:
        shap_top = pd.DataFrame(json.loads(ex_row.iloc[0]["shap_top"]))
        shap_top["direction"] = shap_top["contribution"].map(
            lambda v: "raises risk" if v > 0 else "lowers risk"
        )
        shap_top["display"] = shap_top.apply(
            lambda r: f"{r['feature']} = {r['value']}" if r["value"] is not None else r["feature"],
            axis=1,
        )
        chart = (
            alt.Chart(shap_top)
            .mark_bar()
            .encode(
                x=alt.X("contribution:Q", title="Contribution to risk (log-odds)"),
                y=alt.Y("display:N", sort=None, title=None),
                color=alt.Color(
                    "direction:N",
                    scale=alt.Scale(domain=["raises risk", "lowers risk"], range=[ORANGE, BLUE]),
                    legend=alt.Legend(title=None, orient="bottom"),
                ),
                tooltip=["feature", alt.Tooltip("contribution:Q", format=".4f"), "value"],
            )
            .properties(height=250)
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption(f"Base value {float(ex_row.iloc[0]['base_value']):.3f} (log-odds).")

# ---- provider peer comparison -------------------------------------------
with right:
    st.subheader("Provider vs peers")
    peer = load_peer_stats()
    spec_code = peer["spec_code_stats"]
    match = spec_code[
        (spec_code["specialty"] == claim["specialty"]) & (spec_code["hcpcs"] == claim["hcpcs"])
    ]
    code_norm = (
        float(match["spec_code_paid_unit_median"].iloc[0])
        if len(match)
        else float(
            peer["code_stats"]
            .loc[peer["code_stats"]["hcpcs"] == claim["hcpcs"], "code_paid_unit_median"]
            .squeeze()
            if (peer["code_stats"]["hcpcs"] == claim["hcpcs"]).any()
            else 0.0
        )
    )
    comparison = pd.DataFrame(
        {
            "who": ["This claim (per unit)", f"{claim['specialty']} median"],
            "paid": [float(claim["paid_amount"]), code_norm],
        }
    )
    bar = (
        alt.Chart(comparison)
        .mark_bar(color=BLUE)
        .encode(
            x=alt.X("paid:Q", title=f"Paid for {claim['hcpcs']}"),
            y=alt.Y("who:N", title=None),
            color=alt.Color("who:N", legend=None, scale=alt.Scale(range=[ORANGE, BLUE])),
            tooltip=[alt.Tooltip("paid:Q", format="$,.2f")],
        )
        .properties(height=130)
    )
    st.altair_chart(bar, use_container_width=True)

    st.subheader("Decision")
    investigator = st.text_input(
        "Investigator", value=st.session_state.get("investigator", ""), key="det_inv"
    )
    note = st.text_input("Note", key="det_note")
    b1, b2, b3 = st.columns(3)
    decided = None
    if b1.button("✅ Approve", use_container_width=True):
        decided = "approved"
    if b2.button("🚫 Dismiss", use_container_width=True):
        decided = "dismissed"
    if b3.button("⏳ Pending", use_container_width=True):
        decided = "pending"
    if decided:
        st.session_state["investigator"] = investigator
        record_decision(claim_id, decided, investigator, note, str(claim["model_version"]))
        st.success(f"Recorded **{decided}**.")

# ---- similar claims ------------------------------------------------------
st.subheader("Similar claims (same beneficiary + code, ±45 days)")
claims = load_claims()
window = pd.Timedelta(days=45)
service_date = pd.Timestamp(claim["service_date"])
similar = claims[
    (claims["bene_id"] == claim["bene_id"])
    & (claims["hcpcs"] == claim["hcpcs"])
    & (pd.to_datetime(claims["service_date"]).between(service_date - window, service_date + window))
].copy()
if len(similar) <= 1:
    st.caption("No other claims for this beneficiary + code in the window.")
else:
    similar["this_claim"] = similar["claim_id"] == claim_id
    st.dataframe(
        similar[
            [
                "claim_id",
                "provider_id",
                "service_date",
                "submit_date",
                "units",
                "modifier",
                "pos",
                "paid_amount",
                "this_claim",
            ]
        ].sort_values("service_date"),
        use_container_width=True,
        hide_index=True,
    )

# ---- research mode -------------------------------------------------------
with st.expander("🔬 Research mode — synthetic ground truth (demo only)"):
    st.caption(
        "Labels exist because this dataset is synthetic; a production system would not have this."
    )
    from app.dashboard.common import load_labels_research_mode

    labels = load_labels_research_mode()
    truth = labels[labels["claim_id"] == claim_id]
    if truth.empty or not bool(truth.iloc[0]["is_improper"]):
        st.write("Ground truth: **not injected as improper**.")
    else:
        st.write(
            f"Ground truth: **improper** — patterns `{truth.iloc[0]['patterns']}`, "
            f"injected improper amount ${float(truth.iloc[0]['improper_amount']):,.2f}."
        )
    st.caption(f"Anomaly percentile: {claim['anomaly_percentile']:.3f}")
