"""Cleaning of validated claims into typed staging form.

Cleaning fixes data-quality problems only. It keeps suspected fraud duplicates
(different claim IDs, same content) because the duplicate detectors need them.
It only collapses rows that repeat the same ``claim_id``, which are ingestion
artifacts.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.data.validate import parse_amount

logger = logging.getLogger(__name__)

AMOUNT_COLS = ["submitted_amount", "allowed_amount", "paid_amount"]
DATE_COLS = ["service_date", "service_end_date", "submit_date"]


def clean_claims(passed: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Type, normalize, and de-artifact validated claims; returns (clean, report)."""
    df = passed.copy()
    n_in = len(df)

    # Collapse ingestion duplicates: same claim_id appearing multiple times.
    before = len(df)
    df = df.drop_duplicates(subset=["claim_id"], keep="first")
    ingestion_dups_removed = before - len(df)

    # Types and normalization.
    for col in AMOUNT_COLS:
        df[col] = parse_amount(df[col]).round(2)
    for col in DATE_COLS:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()
    df["service_end_date"] = df["service_end_date"].fillna(df["service_date"])
    df["hcpcs"] = df["hcpcs"].astype(str).str.strip().str.upper()
    df["modifier"] = df["modifier"].fillna("").astype(str).str.strip().str.upper()
    df["units"] = pd.to_numeric(df["units"], errors="coerce").fillna(1).astype(int)

    # Missing POS gets an explicit "UNK" category instead of staying null.
    pos = df["pos"].astype(str).str.strip()
    missing_pos_filled = int((pos == "").sum())
    df["pos"] = pos.where(pos != "", "UNK")

    # Paid can never exceed allowed/submitted after typing; clip tiny float drift.
    df["allowed_amount"] = df[["allowed_amount", "submitted_amount"]].min(axis=1)
    df["paid_amount"] = df[["paid_amount", "allowed_amount"]].min(axis=1)

    df = df.sort_values(["service_date", "provider_id", "claim_id"]).reset_index(drop=True)
    report = {
        "rows_in": n_in,
        "rows_out": len(df),
        "ingestion_duplicates_removed": ingestion_dups_removed,
        "missing_pos_filled": missing_pos_filled,
    }
    logger.info("cleaning_complete", extra=dict(report))
    return df, report
