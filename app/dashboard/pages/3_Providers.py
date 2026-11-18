"""Provider risk ranking and peer comparison."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import altair as alt
import pandas as pd
import streamlit as st

from app.dashboard.common import (
    BLUE,
    ORANGE,
    load_peer_stats,
    load_predictions,
    money,
    page_setup,
    synthetic_banner,
)


@st.cache_data(ttl=600, show_spinner="Summarizing provider activity…")
def summarize_providers(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby(["provider_id", "specialty"], as_index=False)
        .agg(
            claims=("claim_id", "size"),
            mean_risk=("risk_score", "mean"),
            p95_risk=("risk_score", lambda values: values.quantile(0.95)),
            flagged=("recommended_action", lambda values: int((values == "manual_review").sum())),
            est_at_risk=("estimated_payment_at_risk", "sum"),
            paid=("paid_amount", "sum"),
        )
        .assign(flag_rate=lambda frame: frame["flagged"] / frame["claims"])
    )


page_setup("Provider Risk")
synthetic_banner()

preds = load_predictions()
if preds.empty or not {
    "provider_id",
    "specialty",
    "risk_score",
    "claim_id",
    "recommended_action",
    "estimated_payment_at_risk",
    "paid_amount",
    "month",
    "service_date",
    "hcpcs",
}.issubset(preds.columns):
    st.info(
        "Scored provider data is unavailable. Build the dataset and run scoring to view provider analytics."
    )
    st.stop()

providers = summarize_providers(preds)

f1, f2 = st.columns([1.4, 1])
specialty = f1.multiselect("Specialty", sorted(providers["specialty"].unique().tolist()))
min_claims = f2.slider("Min claims in window", 10, 500, 50, 10)
view = providers[providers["claims"] >= min_claims]
if specialty:
    view = view[view["specialty"].isin(specialty)]

st.caption(
    f"{len(view):,} providers · {int(view['flagged'].sum()):,} flagged claims · "
    f"{money(float(view['est_at_risk'].sum()))} estimated at risk"
)

ranked = view.sort_values(["est_at_risk", "mean_risk"], ascending=False).head(25)
if ranked.empty:
    st.info("No providers match the selected specialty and minimum-claim filters.")
    st.stop()
st.subheader("Top providers by estimated payment at risk")
bar = (
    alt.Chart(ranked)
    .mark_bar(color=BLUE)
    .encode(
        x=alt.X("est_at_risk:Q", title="Estimated $ at risk"),
        y=alt.Y("provider_id:N", sort="-x", title=None),
        tooltip=[
            "provider_id",
            "specialty",
            "claims",
            alt.Tooltip("mean_risk:Q", format=".3f"),
            alt.Tooltip("flag_rate:Q", format=".1%"),
            alt.Tooltip("est_at_risk:Q", format="$,.0f"),
        ],
    )
    .properties(height=520)
)
st.altair_chart(bar, use_container_width=True)

st.subheader("Volume vs risk (each dot = a provider)")
scatter = (
    alt.Chart(view)
    .mark_circle(size=70, color=BLUE, opacity=0.55, stroke="#fcfcfb", strokeWidth=1)
    .encode(
        x=alt.X("claims:Q", title="Claims in scored window", scale=alt.Scale(type="log")),
        y=alt.Y("mean_risk:Q", title="Mean risk score"),
        tooltip=[
            "provider_id",
            "specialty",
            "claims",
            alt.Tooltip("mean_risk:Q", format=".3f"),
            alt.Tooltip("est_at_risk:Q", format="$,.0f"),
        ],
    )
    .properties(height=340)
)
flag_layer = (
    alt.Chart(view[view["flag_rate"] >= 0.05])
    .mark_circle(size=90, color=ORANGE, stroke="#fcfcfb", strokeWidth=1)
    .encode(
        x="claims:Q",
        y="mean_risk:Q",
        tooltip=["provider_id", "specialty", alt.Tooltip("flag_rate:Q", format=".1%")],
    )
)
st.altair_chart(scatter + flag_layer, use_container_width=True)
st.caption("Orange dots: providers with ≥5% of claims flagged for manual review.")

st.subheader("Provider drill-down")
pid = st.selectbox("Provider", ranked["provider_id"].tolist())
prow = providers[providers["provider_id"] == pid].iloc[0]
d1, d2, d3, d4 = st.columns(4)
d1.metric("Claims", f"{int(prow['claims']):,}")
d2.metric("Mean risk", f"{prow['mean_risk']:.3f}")
d3.metric("Flag rate", f"{prow['flag_rate']:.1%}")
d4.metric("Est. $ at risk", money(float(prow["est_at_risk"])))

peer_stats = load_peer_stats()
peer = peer_stats.get("provider_month_peer")
peer_row = peer[peer["specialty"] == prow["specialty"]] if isinstance(peer, pd.DataFrame) else []
if len(peer_row):
    monthly_paid = float(prow["paid"]) / max(1, preds["month"].nunique())
    st.caption(
        f"Avg monthly paid {money(monthly_paid)} vs specialty median "
        f"{money(float(peer_row['peer_monthly_paid_median'].iloc[0]))} "
        "(frozen train-window benchmark)."
    )
prov_claims = preds[preds["provider_id"] == pid].sort_values("risk_score", ascending=False)
st.dataframe(
    prov_claims[
        ["claim_id", "hcpcs", "service_date", "paid_amount", "risk_score", "recommended_action"]
    ].head(15),
    use_container_width=True,
    hide_index=True,
)
