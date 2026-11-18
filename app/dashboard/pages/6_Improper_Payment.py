"""Improper-payment indicators, separate from fraud-risk predictions."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import plotly.express as px
import streamlit as st

from app.dashboard.common import load_claims, page_setup, synthetic_banner
from src.payment_integrity.improper_payment import analyze_payments


@st.cache_data(ttl=600, show_spinner="Checking payment indicators…")
def load_payment_analysis():
    return analyze_payments(load_claims())


page_setup("Improper Payment Analysis")
synthetic_banner()
st.info(
    "These are transparent payment-error indicators based on available fields. "
    "They are not fraud findings or adjudicated CMS outcomes."
)

try:
    claims = load_payment_analysis()
except Exception as exc:
    st.warning(f"Claims data is unavailable: {exc}")
    st.stop()

flagged = claims[claims["payment_issue_count"] > 0]
c1, c2, c3 = st.columns(3)
c1.metric("Claims analyzed", f"{len(claims):,}")
c2.metric("Potential payment issues", f"{len(flagged):,}")
c3.metric("Potential overpayment rows", f"{int(claims['potential_overpayment'].sum()):,}")

if flagged.empty:
    st.success("No configured payment-error indicators fired for this dataset.")
else:
    categories = flagged[["potential_overpayment", "potential_underpayment"]].sum()
    chart = px.bar(
        x=["Potential overpayment", "Potential underpayment"],
        y=[int(categories.iloc[0]), int(categories.iloc[1])],
        labels={"x": "Indicator", "y": "Claims"},
        title="Payment direction indicators",
    )
    st.plotly_chart(chart, use_container_width=True)
    columns = [
        c
        for c in [
            "claim_id",
            "paid_amount",
            "allowed_amount",
            "payment_issue_score",
            "payment_reason_codes",
        ]
        if c in flagged
    ]
    table = flagged[columns].sort_values("payment_issue_score", ascending=False)
    if len(table) > 500:
        st.caption(
            f"Showing the 500 highest-scoring rows of {len(table):,} flagged claims; the download includes all flagged rows."
        )
    st.dataframe(table.head(500), use_container_width=True, hide_index=True)
    st.download_button(
        "Download payment indicators",
        flagged.to_csv(index=False),
        "improper_payment_indicators.csv",
        "text/csv",
    )
