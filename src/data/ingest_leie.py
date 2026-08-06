"""Ingestion of the OIG List of Excluded Individuals/Entities (LEIE).

The LEIE is the real-world registry of providers excluded from federal health
programs (fraud convictions, license actions, etc.). It provides *provider-level*
context and a possible weak-label extension. Known limitations, documented in
``docs/labeling.md``: few historical rows carry NPIs, excluded providers largely
stop appearing in utilization data (survivorship bias), and exclusion dates
rarely align with observed billing windows — which is why LEIE is **not** the
primary supervision source in this project.

Modes: real download (open-egress environments) or a clearly-synthetic fixture
(sandboxes/tests). Fixture NPIs use a reserved fake prefix and match nothing real.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
import requests

from src.data.db import write_df

logger = logging.getLogger(__name__)

LEIE_URL = "https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv"
FIXTURE_PATH = Path("tests/fixtures/leie_fixture.csv")

#: Exclusion authorities most associated with program-related fraud convictions.
#: (See OIG exclusion authorities; e.g., 1128(a)(1) program-related crimes,
#: 1128(a)(2) patient abuse, 1128(a)(3) health-care fraud felony,
#: 1128(b)(7) fraud/kickbacks.)
FRAUD_RELATED_TYPES = {"1128a1", "1128a2", "1128a3", "1128b7"}

KEEP_COLS = {
    "NPI": "npi",
    "EXCLTYPE": "exclusion_type",
    "EXCLDATE": "exclusion_date",
    "SPECIALTY": "specialty",
    "STATE": "state",
}


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    present = {k: v for k, v in KEEP_COLS.items() if k in df.columns}
    out = df[list(present)].rename(columns=present)
    out["npi"] = out["npi"].astype(str).str.strip()
    out = out[out["npi"].str.fullmatch(r"\d{10}") & (out["npi"] != "0000000000")]
    out["exclusion_type"] = out["exclusion_type"].astype(str).str.strip().str.lower()
    out["exclusion_date"] = pd.to_datetime(
        out["exclusion_date"].astype(str), format="%Y%m%d", errors="coerce"
    ).dt.date
    out["fraud_related"] = out["exclusion_type"].isin(FRAUD_RELATED_TYPES)
    return out.reset_index(drop=True)


def fetch_from_source(timeout: int = 120) -> pd.DataFrame:
    """Download the full LEIE CSV from oig.hhs.gov."""
    resp = requests.get(LEIE_URL, timeout=timeout)
    resp.raise_for_status()
    return _normalize(pd.read_csv(io.BytesIO(resp.content), dtype=str))


def load_fixture(path: Path = FIXTURE_PATH) -> pd.DataFrame:
    """Load the synthetic test fixture (NOT real exclusion records)."""
    return _normalize(pd.read_csv(path, dtype=str))


def ingest_leie(use_fixture: bool = True) -> pd.DataFrame:
    """Load LEIE data into ``raw.leie_exclusions``."""
    df = load_fixture() if use_fixture else fetch_from_source()
    write_df(df, "leie_exclusions", schema="raw")
    logger.info(
        "leie_ingested", extra={"rows": len(df), "mode": "fixture" if use_fixture else "download"}
    )
    return df
