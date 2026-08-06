"""Ingestion of CMS Medicare Physician & Other Practitioners (by Provider and Service).

Two modes:

* **API mode** — pages the public ``data.cms.gov`` data-api (works anywhere with
  open egress: your laptop, GitHub Actions). Handles paging, retries, and column
  normalization.
* **Snapshot mode** (default) — loads the committed calibration sample in
  ``data/reference/`` so the pipeline is reproducible offline. Provenance for the
  snapshot lives in ``data/reference/cms_snapshot_metadata.json``.

Source dataset (public, provider-level aggregates — no beneficiary data):
https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners/medicare-physician-other-practitioners-by-provider-and-service
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

from src.data.db import write_df

logger = logging.getLogger(__name__)

DATASET_UUID = "92396110-2aed-4d63-a6a2-5d6207d46a29"
API_URL = f"https://data.cms.gov/data-api/v1/dataset/{DATASET_UUID}/data"
SNAPSHOT_PATH = Path("data/reference/cms_provider_service_sample.csv")

#: source column -> normalized column
COLUMN_MAP = {
    "Rndrng_NPI": "npi",
    "Rndrng_Prvdr_Type": "provider_type",
    "Rndrng_Prvdr_State_Abrvtn": "state",
    "Rndrng_Prvdr_RUCA": "ruca",
    "HCPCS_Cd": "hcpcs",
    "HCPCS_Desc": "hcpcs_desc",
    "HCPCS_Drug_Ind": "drug_ind",
    "Place_Of_Srvc": "place_of_service",
    "Tot_Benes": "total_benes",
    "Tot_Srvcs": "total_services",
    "Tot_Bene_Day_Srvcs": "total_bene_day_services",
    "Avg_Sbmtd_Chrg": "avg_submitted",
    "Avg_Mdcr_Alowd_Amt": "avg_allowed",
    "Avg_Mdcr_Pymt_Amt": "avg_paid",
}
NUMERIC_COLS = [
    "total_benes",
    "total_services",
    "total_bene_day_services",
    "avg_submitted",
    "avg_allowed",
    "avg_paid",
]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Rename to snake_case, coerce numerics, drop rows without a usable payment."""
    present = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    out = df[list(present)].rename(columns=present)
    for col in NUMERIC_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["avg_paid", "hcpcs", "provider_type"])
    out["npi"] = out["npi"].astype(str)
    return out.reset_index(drop=True)


def fetch_from_api(
    max_rows: int = 100_000,
    page_size: int = 5_000,
    provider_types: list[str] | None = None,
    timeout: int = 60,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Page the CMS data-api into a DataFrame (optionally filtered by provider type)."""
    frames: list[pd.DataFrame] = []
    filters: list[str | None] = list(provider_types) if provider_types else [None]
    for ptype in filters:
        offset = 0
        while offset < max_rows // len(filters):
            params: dict[str, str | int] = {"size": page_size, "offset": offset}
            if ptype is not None:
                params["filter[Rndrng_Prvdr_Type]"] = ptype
            for attempt in range(1, max_retries + 1):
                try:
                    resp = requests.get(API_URL, params=params, timeout=timeout)
                    resp.raise_for_status()
                    rows = resp.json()
                    break
                except (requests.RequestException, ValueError):
                    if attempt == max_retries:
                        raise
                    time.sleep(2**attempt)
            if not rows:
                break
            frames.append(pd.DataFrame(rows))
            logger.info(
                "cms_api_page", extra={"provider_type": ptype, "offset": offset, "rows": len(rows)}
            )
            if len(rows) < page_size:
                break
            offset += page_size
    if not frames:
        return pd.DataFrame(columns=list(COLUMN_MAP.values()))
    return _normalize(pd.concat(frames, ignore_index=True))


def load_snapshot(path: Path = SNAPSHOT_PATH) -> pd.DataFrame:
    """Load the committed calibration snapshot."""
    df = pd.read_csv(path, dtype={"Rndrng_NPI": str, "HCPCS_Cd": str})
    return _normalize(df)


def ingest_cms(use_snapshot: bool = True, max_rows: int = 100_000) -> pd.DataFrame:
    """Load CMS provider-service data (snapshot or API) into ``raw.cms_provider_service``."""
    df = load_snapshot() if use_snapshot else fetch_from_api(max_rows=max_rows)
    write_df(df, "cms_provider_service", schema="raw")
    logger.info(
        "cms_ingested", extra={"rows": len(df), "mode": "snapshot" if use_snapshot else "api"}
    )
    return df
