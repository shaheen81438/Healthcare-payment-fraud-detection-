"""OIG LEIE exclusion screening with manual-verification warnings."""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import streamlit as st

from app.dashboard.common import page_setup, synthetic_banner
from src.data.ingest_leie import load_fixture
from src.payment_integrity.leie_screening import normalize_leie, screen_provider


@st.cache_data(ttl=600, show_spinner=False)
def load_default_leie():
    return normalize_leie(load_fixture())


@st.cache_data(ttl=600, show_spinner=False)
def normalize_uploaded_leie(content: bytes):
    return normalize_leie(pd.read_csv(io.BytesIO(content), dtype=str))


page_setup("OIG Exclusion Screening")
synthetic_banner()
st.warning(
    "An exclusion match is a compliance-screening result, not proof that a claim or provider is fraudulent. "
    "Verify identity, exclusion dates, and reinstatement status manually."
)

uploaded = st.file_uploader("Optional official OIG LEIE CSV", type=["csv"])
try:
    leie = normalize_uploaded_leie(uploaded.getvalue()) if uploaded else load_default_leie()
except Exception as exc:
    st.error(f"Unable to load LEIE data: {exc}")
    st.stop()

left, right = st.columns(2)
with left:
    npi = st.text_input("Provider NPI", max_chars=10)
with right:
    name = st.text_input("Provider name (secondary matching)")

if st.button("Screen provider", type="primary"):
    matches = screen_provider({"npi": npi, "provider_name": name}, leie)
    if not matches:
        st.success("No possible LEIE match found in the loaded dataset.")
    else:
        st.error(f"{len(matches)} possible match(es) require review.")
        display_matches = []
        for match in matches:
            visible = dict(match)
            provider_identifier = str(visible.get("provider_id", ""))
            if provider_identifier.isdigit() and len(provider_identifier) == 10:
                visible["provider_id"] = f"******{provider_identifier[-4:]}"
            evidence = dict(visible.get("match_evidence") or {})
            for key in evidence:
                if key.lower() == "npi" and evidence[key] is not None:
                    identifier = str(evidence[key])
                    evidence[key] = (
                        f"******{identifier[-4:]}" if len(identifier) == 10 else "masked"
                    )
            visible["match_evidence"] = evidence
            display_matches.append(visible)
        if len(display_matches) > 100:
            st.warning(
                f"Showing 100 of {len(display_matches):,} possible matches; narrow the search before manual review."
            )
        st.dataframe(
            pd.DataFrame(display_matches).head(100), use_container_width=True, hide_index=True
        )

st.caption(
    f"Loaded {len(leie):,} LEIE rows. SSNs are never displayed; matched NPIs are masked in results."
)
