"""Specialty-aware anomaly screening for CMS provider-service aggregates."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import plotly.express as px
import streamlit as st

from app.dashboard.common import page_setup, synthetic_banner
from src.data.ingest_cms import load_snapshot
from src.payment_integrity.provider_anomaly import detect_provider_anomalies


@st.cache_data(ttl=600, show_spinner="Scoring provider peers…")
def load_provider_anomalies():
    return detect_provider_anomalies(load_snapshot())


page_setup("Provider Anomaly Analysis")
synthetic_banner()
st.info(
    "This page screens provider-level CMS utilization aggregates against peers in the same "
    "provider type. It identifies unusual patterns for review; it does not establish improper "
    "billing or fraud."
)

try:
    anomalies = load_provider_anomalies()
except FileNotFoundError:
    st.warning(
        "CMS provider-service data is unavailable. Add the snapshot or ingest CMS data first."
    )
    st.stop()
except ValueError as exc:
    st.warning(f"CMS provider-service data cannot be analyzed: {exc}")
    st.stop()

if anomalies.empty:
    st.warning("No provider records are available for anomaly analysis.")
    st.stop()

anomalies = anomalies.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
anomalies["provider_label"] = [f"Provider {index + 1:03d}" for index in range(len(anomalies))]
anomalies = anomalies.drop(columns=["provider_id"])
st.caption(f"Analyzed {len(anomalies):,} provider and specialty groups.")
left, right = st.columns(2)
with left:
    st.plotly_chart(
        px.scatter(
            anomalies,
            x="service_volume",
            y="average_paid",
            color="peer_group",
            size="anomaly_score",
            hover_name="provider_label",
            hover_data=["anomaly_score", "investigation_priority", "procedure_concentration"],
            title="Service volume and average paid amount",
        ),
        use_container_width=True,
    )
with right:
    st.plotly_chart(
        px.histogram(
            anomalies,
            x="anomaly_score",
            color="peer_group",
            nbins=20,
            title="Anomaly score by provider type",
        ),
        use_container_width=True,
    )

st.subheader("Providers for review")
priority = st.multiselect(
    "Investigation priority",
    ["high", "medium", "routine"],
    default=["high", "medium"],
)
view = anomalies[anomalies["investigation_priority"].isin(priority)].sort_values(
    "anomaly_score", ascending=False
)
st.dataframe(view, use_container_width=True, hide_index=True)
st.download_button(
    "Download provider anomaly results",
    view.to_csv(index=False),
    "provider_anomaly_results.csv",
    "text/csv",
)
st.caption(
    "CMS data is provider-service aggregate data, not adjudicated claims. Scores are relative "
    "to the available peer sample and should be reviewed with source records."
)
