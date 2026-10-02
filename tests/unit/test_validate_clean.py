"""Validation and cleaning behavior on seeded defects."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.clean import clean_claims
from src.data.reference import load_reference_frames
from src.data.validate import parse_amount, validate_claims


@pytest.fixture(scope="module")
def ref() -> dict[str, pd.DataFrame]:
    return load_reference_frames()


def _base_claims(n: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "claim_id": [f"CLM-{i:08d}" for i in range(1, n + 1)],
            "bene_id": ["BEN-000001"] * n,
            "provider_id": ["PRV-00001"] * n,
            "specialty": ["Internal Medicine"] * n,
            "hcpcs": ["99213"] * n,
            "modifier": [""] * n,
            "units": [1] * n,
            "dx_group": ["DXG01"] * n,
            "pos": ["11"] * n,
            "service_date": ["2024-03-04"] * n,
            "service_end_date": ["2024-03-04"] * n,
            "submit_date": ["2024-03-14"] * n,
            "submitted_amount": [150.0] * n,
            "allowed_amount": [100.0] * n,
            "paid_amount": [80.0] * n,
        }
    )


def test_parse_amount_handles_currency_strings() -> None:
    parsed = parse_amount(pd.Series(["$1,234.56", 80.0, "72.5", "garbage"]))
    assert parsed.iloc[0] == 1234.56
    assert parsed.iloc[1] == 80.0
    assert parsed.iloc[2] == 72.5
    assert np.isnan(parsed.iloc[3])


def test_validation_quarantines_each_error_class(ref: dict[str, pd.DataFrame]) -> None:
    df = _base_claims(12)
    df["paid_amount"] = df["paid_amount"].astype(object)
    df.loc[0, "paid_amount"] = "not-a-number"  # V002
    df.loc[1, "paid_amount"] = -50.0  # V003
    df.loc[2, "submit_date"] = "2024-02-20"  # V004 submit before service
    df.loc[3, "hcpcs"] = "9921"  # V005
    df.loc[4, "pos"] = "99"  # V006 unknown pos
    df.loc[5, "service_date"] = "1994-01-01"  # V008
    df.loc[6, "units"] = 9999  # V009
    result = validate_claims(df, ref["pos_codes"], ref["dx_groups"])
    assert result.report["rows_quarantined"] == 7
    reasons = set(result.quarantined["quarantine_reason"])
    assert {
        "V002_unparseable_amount",
        "V003_negative_amount",
        "V004_impossible_dates",
        "V005_invalid_hcpcs",
        "V006_unknown_pos",
        "V008_date_out_of_window",
        "V009_units_out_of_range",
    } == reasons
    assert result.report["checks"]["V002_unparseable_amount"]["failed"] == 1


def test_missing_pos_is_warning_not_quarantine(ref: dict[str, pd.DataFrame]) -> None:
    df = _base_claims(5)
    df.loc[2, "pos"] = ""
    result = validate_claims(df, ref["pos_codes"], ref["dx_groups"])
    assert result.report["rows_quarantined"] == 0
    assert result.report["checks"]["V006w_missing_pos"]["failed"] == 1


def test_cleaning_preserves_fraud_dups_removes_ingestion_dups(
    ref: dict[str, pd.DataFrame],
) -> None:
    df = _base_claims(6)
    # Fraud-style duplicate: different claim_id, same content. It must survive.
    fraud_dup = df.iloc[[0]].copy()
    fraud_dup["claim_id"] = "CLM-00009999"
    # Ingestion artifact: the same claim_id repeated. It must be collapsed.
    ingest_dup = df.iloc[[1]].copy()
    df = pd.concat([df, fraud_dup, ingest_dup], ignore_index=True)

    result = validate_claims(df, ref["pos_codes"], ref["dx_groups"])
    clean, report = clean_claims(result.passed)
    assert report["ingestion_duplicates_removed"] == 1
    assert "CLM-00009999" in set(clean["claim_id"])  # fraud dup kept
    assert clean["claim_id"].is_unique


def test_cleaning_fixes_types_and_pos(ref: dict[str, pd.DataFrame]) -> None:
    df = _base_claims(4)
    df["paid_amount"] = df["paid_amount"].astype(object)
    df.loc[0, "paid_amount"] = "$80.00"
    df.loc[1, "pos"] = ""
    result = validate_claims(df, ref["pos_codes"], ref["dx_groups"])
    clean, report = clean_claims(result.passed)
    assert clean["paid_amount"].dtype == float
    assert float(clean.loc[clean.index[0], "paid_amount"]) == 80.0
    assert report["missing_pos_filled"] == 1
    assert (clean.loc[clean["pos"] == "UNK"].shape[0]) == 1
    assert str(clean["service_date"].dtype).startswith("datetime64")
