"""Ingestion of the OIG List of Excluded Individuals/Entities (LEIE).

The LEIE is the real-world registry of providers excluded from federal health
programs (fraud convictions, license actions, etc.). It adds provider-level
context and could extend the weak labels. Known limitations, documented in
``docs/labeling.md``: few historical rows carry NPIs, excluded providers mostly
stop appearing in utilization data (survivorship bias), and exclusion dates
rarely line up with observed billing windows. For these reasons LEIE is not the
primary supervision source in this project.

Modes: real download (open-egress environments) or a clearly synthetic fixture
(sandboxes/tests). Fixture NPIs use a reserved fake prefix and match nothing real.
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from src.data.db import write_df
from src.data.provenance import record_dataset_provenance

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
    "REINDATE": "reinstatement_date",
    "LASTNAME": "last_name",
    "FIRSTNAME": "first_name",
    "MIDNAME": "middle_name",
    "BUSNAME": "provider_name",
    "SPECIALTY": "specialty",
    "STATE": "state",
}
REQUIRED_SOURCE_COLUMNS = {"NPI", "EXCLTYPE", "EXCLDATE"}
KNOWN_SOURCE_COLUMNS = set(KEEP_COLS) | {
    "GENERAL",
    "UPIN",
    "DOB",
    "ADDRESS",
    "CITY",
    "ZIP",
    "WVRSTATE",
}


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the LEIE source schema, then keep only fields used downstream."""
    source_columns = {str(column).strip().upper() for column in df.columns}
    missing = REQUIRED_SOURCE_COLUMNS - source_columns
    unexpected = source_columns - KNOWN_SOURCE_COLUMNS
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing required columns: {sorted(missing)}")
        if unexpected:
            details.append(f"unexpected columns: {sorted(unexpected)}")
        raise ValueError("Invalid LEIE schema (" + "; ".join(details) + ")")

    source_names = {str(column).strip().upper(): column for column in df.columns}
    present = {source_names[key]: value for key, value in KEEP_COLS.items() if key in source_names}
    out = df[list(present)].rename(columns=present)
    out["npi"] = out["npi"].astype(str).str.strip()
    valid_npi = out["npi"].str.fullmatch(r"\d{10}") & (out["npi"] != "0000000000")
    dropped_npi = int((~valid_npi).sum())
    if dropped_npi:
        logger.warning(
            "leie_rows_without_valid_npi_dropped",
            extra={"rows": dropped_npi, "total_rows": len(out)},
        )
    out = out[valid_npi].copy()
    if out.empty:
        raise ValueError("LEIE schema was valid but no rows had usable NPIs")
    out["exclusion_type"] = out["exclusion_type"].astype("string").str.strip().str.lower()
    if out["exclusion_type"].isna().any() or out["exclusion_type"].eq("").any():
        raise ValueError("LEIE rows contain missing exclusion types")
    out["exclusion_date"] = pd.to_datetime(
        out["exclusion_date"], format="%Y%m%d", errors="coerce"
    ).dt.date
    if out["exclusion_date"].isna().any():
        raise ValueError("LEIE rows contain invalid exclusion dates")
    if "reinstatement_date" in out:
        out["reinstatement_date"] = pd.to_datetime(
            out["reinstatement_date"], format="%Y%m%d", errors="coerce"
        ).dt.date
    out["fraud_related"] = out["exclusion_type"].isin(FRAUD_RELATED_TYPES)
    return out.reset_index(drop=True)


def fetch_from_source(timeout: int = 120, max_retries: int = 3) -> pd.DataFrame:
    """Download and validate the full LEIE CSV with bounded transient retries."""
    if timeout <= 0 or max_retries < 1:
        raise ValueError("timeout and max_retries must be positive")
    response = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(LEIE_URL, timeout=(10, timeout))
        except (requests.Timeout, requests.ConnectionError):
            if attempt == max_retries:
                raise
            time.sleep(min(2 ** (attempt - 1), 8))
            continue
        if response.status_code in {408, 429} or response.status_code >= 500:
            if attempt == max_retries:
                response.raise_for_status()
            time.sleep(min(2 ** (attempt - 1), 8))
            continue
        response.raise_for_status()
        break
    if response is None:
        raise RuntimeError("LEIE download did not return a response")

    content = response.content
    prefix = content.lstrip()[:256].lower()
    if not content or content.startswith((b"PK\x03\x04", b"MZ")):
        raise ValueError("LEIE response is empty or is not a plain CSV file")
    if b"<html" in prefix or prefix.startswith((b"<!doctype", b"<?xml")):
        raise ValueError("LEIE response is HTML/XML, not the expected CSV file")
    try:
        raw = pd.read_csv(io.BytesIO(content), dtype=str, encoding="utf-8-sig")
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError("LEIE response could not be parsed as CSV") from exc
    return _normalize(raw)


def load_fixture(path: Path = FIXTURE_PATH) -> pd.DataFrame:
    """Load the synthetic test fixture. These are not real exclusion records."""
    return _normalize(pd.read_csv(path, dtype=str))


def ingest_leie(
    use_fixture: bool = True,
    manifest_path: str | Path = "monitoring/reports/data_provenance.json",
) -> pd.DataFrame:
    """Load LEIE data into ``raw.leie_exclusions``."""
    source_identifier = str(FIXTURE_PATH) if use_fixture else LEIE_URL
    file_name = FIXTURE_PATH.name if use_fixture else "OIG LEIE download"
    classification = "synthetic_fixture" if use_fixture else "real_public"
    try:
        df = load_fixture() if use_fixture else fetch_from_source()
        write_df(df, "leie_exclusions", schema="raw")
    except Exception as exc:
        logger.exception(
            "leie_ingestion_failed",
            extra={
                "mode": "fixture" if use_fixture else "download",
                "error_type": type(exc).__name__,
            },
        )
        try:
            record_dataset_provenance(
                pd.DataFrame(),
                source_name="OIG List of Excluded Individuals/Entities",
                source_url=None if use_fixture else LEIE_URL,
                source_identifier=source_identifier,
                file_name=file_name,
                file_format="CSV",
                classification=classification,
                validation_status=f"failed:{type(exc).__name__}",
                manifest_path=manifest_path,
            )
        except Exception:
            logger.exception("leie_failure_provenance_write_failed")
        raise
    record_dataset_provenance(
        df,
        source_name="OIG List of Excluded Individuals/Entities",
        source_url=None if use_fixture else LEIE_URL,
        source_identifier=source_identifier,
        file_name=file_name,
        file_format="CSV",
        classification=classification,
        validation_status="normalized",
        manifest_path=manifest_path,
    )
    logger.info(
        "leie_ingested", extra={"rows": len(df), "mode": "fixture" if use_fixture else "download"}
    )
    return df
