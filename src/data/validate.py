"""Data-contract validation for raw claims.

Checks are explicit and versioned (V001–V010). Rows failing **error**-severity
checks are quarantined (excluded from staging, kept for inspection); rows with
**warning**-level issues pass through to cleaning, which repairs them; **info**
checks are counted only. The full report is persisted as JSON so every pipeline
run leaves an auditable data-quality trail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

HCPCS_PATTERN = r"^[A-Z0-9]\d{3}[A-Z0-9]$"
REQUIRED_FIELDS = ["claim_id", "bene_id", "provider_id", "hcpcs", "service_date", "paid_amount"]


@dataclass
class ValidationResult:
    passed: pd.DataFrame
    quarantined: pd.DataFrame
    report: dict[str, Any] = field(default_factory=dict)


def load_validation_config(path: str = "configs/validation.yaml") -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)
    return cfg


def parse_amount(series: pd.Series) -> pd.Series:
    """Parse amounts that may arrive as floats or as strings like ``$1,234.56``."""
    as_str = series.astype(str).str.replace(r"[$,]", "", regex=True).str.strip()
    return pd.to_numeric(as_str, errors="coerce")


def validate_claims(
    raw: pd.DataFrame,
    pos_codes: pd.DataFrame,
    dx_groups: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> ValidationResult:
    """Run the claim data contract; split rows into passed vs quarantined."""
    cfg = config or load_validation_config()
    df = raw.copy()
    n = len(df)

    paid = parse_amount(df["paid_amount"])
    service_dt = pd.to_datetime(df["service_date"], errors="coerce")
    submit_dt = pd.to_datetime(df["submit_date"], errors="coerce")
    hcpcs = df["hcpcs"].astype(str).str.strip().str.upper()
    pos = df["pos"].fillna("").astype(str).str.strip().replace("nan", "")
    known_pos = set(pos_codes["pos_code"].astype(str))
    known_dx = set(dx_groups["dx_group"].astype(str))
    units = pd.to_numeric(df["units"], errors="coerce")

    min_date = pd.Timestamp(cfg["service_date_min"])
    max_date = pd.Timestamp(cfg["service_date_max"])

    error_masks: dict[str, pd.Series] = {
        "V001_missing_required": df[REQUIRED_FIELDS].isna().any(axis=1)
        | (df["claim_id"].astype(str).str.len() == 0),
        "V002_unparseable_amount": paid.isna(),
        "V003_negative_amount": paid < 0,
        "V004_impossible_dates": submit_dt.notna() & service_dt.notna() & (submit_dt < service_dt),
        "V005_invalid_hcpcs": ~hcpcs.str.match(HCPCS_PATTERN),
        "V006_unknown_pos": (pos != "") & ~pos.isin(known_pos),
        "V008_date_out_of_window": service_dt.isna()
        | (service_dt < min_date)
        | (service_dt > max_date),
        "V009_units_out_of_range": units.isna() | (units < 1) | (units > int(cfg["units_max"])),
    }
    warning_masks: dict[str, pd.Series] = {
        "V006w_missing_pos": pos == "",
        "V010_unknown_dx_group": ~df["dx_group"].astype(str).isin(known_dx),
    }
    dup_mask = df.duplicated(subset=["claim_id"], keep=False)
    info_counts = {"V007_duplicate_claim_id_rows": int(dup_mask.sum())}

    quarantine_mask = pd.Series(False, index=df.index)
    reasons = pd.Series("", index=df.index, dtype=object)
    for code, mask in error_masks.items():
        mask = mask.fillna(True) if mask.isna().any() else mask
        reasons = np.where(mask & (reasons == ""), code, reasons)  # first failing check wins
        reasons = pd.Series(reasons, index=df.index, dtype=object)
        quarantine_mask |= mask

    quarantined = df[quarantine_mask].copy()
    quarantined["quarantine_reason"] = reasons[quarantine_mask]
    passed = df[~quarantine_mask].copy()

    report: dict[str, Any] = {
        "rows_in": n,
        "rows_passed": len(passed),
        "rows_quarantined": len(quarantined),
        "quarantine_share": round(float(quarantine_mask.mean()), 5),
        "checks": {
            **{c: {"severity": "error", "failed": int(m.sum())} for c, m in error_masks.items()},
            **{
                c: {"severity": "warning", "failed": int(m.sum())} for c, m in warning_masks.items()
            },
            **{c: {"severity": "info", "failed": v} for c, v in info_counts.items()},
        },
    }
    max_share = float(cfg["max_quarantine_share"])
    report["within_tolerance"] = report["quarantine_share"] <= max_share
    logger.info(
        "validation_complete",
        extra={
            "rows_in": n,
            "quarantined": report["rows_quarantined"],
            "within_tolerance": report["within_tolerance"],
        },
    )
    return ValidationResult(passed=passed, quarantined=quarantined, report=report)
