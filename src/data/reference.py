"""Reference code sets: loading and HCPCS categorization."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data.db import write_df

logger = logging.getLogger(__name__)

REFERENCE_DIR = Path("data/reference")

REFERENCE_TABLES = {
    "pos_codes": "pos_codes.csv",
    "dx_groups": "dx_groups.csv",
    "dx_procedure_compat": "dx_procedure_compat.csv",
    "ptp_pairs": "ptp_pairs_illustrative.csv",
    "em_levels": "em_levels.csv",
}

EM_OFFICE_CODES = {"99202", "99203", "99204", "99205", "99211", "99212", "99213", "99214", "99215"}
EM_FACILITY_CODES = {"99221", "99222", "99223", "99231", "99232", "99233", "99238", "99239"}


def hcpcs_category(code: str) -> str:
    """Map an HCPCS/CPT code to a coarse service category used for compatibility rules."""
    code = str(code).strip().upper()
    if code in EM_OFFICE_CODES:
        return "EM_OFFICE"
    if code in EM_FACILITY_CODES or code.startswith("992"):
        return "EM_FACILITY"
    if code[:1] == "J":
        return "DRUG"
    if not code[:1].isdigit():
        return "OTHER"  # G/Q/A codes etc.
    first = int(code[0])
    if code.startswith("97"):
        return "THERAPY"
    if first in (7,):
        return "IMAGING"
    if first in (8,):
        return "LAB"
    if first in (1, 2, 3, 4, 5, 6):
        return "PROCEDURE"
    return "OTHER"


def load_reference_frames(ref_dir: Path = REFERENCE_DIR) -> dict[str, pd.DataFrame]:
    """Read all reference CSVs into DataFrames."""
    return {
        name: pd.read_csv(ref_dir / fname, dtype=str) for name, fname in REFERENCE_TABLES.items()
    }


def load_reference_tables(ref_dir: Path = REFERENCE_DIR) -> dict[str, int]:
    """Load every reference CSV into the ``ref`` schema; returns row counts."""
    counts: dict[str, int] = {}
    frames = load_reference_frames(ref_dir)
    frames["dx_procedure_compat"]["compat"] = frames["dx_procedure_compat"]["compat"].astype(float)
    frames["ptp_pairs"]["component_amount_ratio"] = frames["ptp_pairs"][
        "component_amount_ratio"
    ].astype(float)
    frames["em_levels"]["level"] = frames["em_levels"]["level"].astype(int)
    frames["em_levels"]["relative_payment"] = frames["em_levels"]["relative_payment"].astype(float)
    for name, frame in frames.items():
        counts[name] = write_df(frame, name, schema="ref")
    logger.info("reference_loaded", extra=dict(counts))
    return counts
