"""Feature-calculation unit tests on hand-built mini frames."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.data.reference import load_reference_frames
from src.features.claim_features import (
    amount_and_coding_features,
    beneficiary_history_features,
)
from src.features.dup_similarity import duplicate_features
from src.features.peer_stats import fit_peer_stats
from src.features.splits import assign_split, load_features_config


@pytest.fixture(scope="module")
def cfg() -> dict[str, Any]:
    return load_features_config()


@pytest.fixture(scope="module")
def ref() -> dict[str, pd.DataFrame]:
    return load_reference_frames()


def _claims(rows: list[dict[str, Any]]) -> pd.DataFrame:
    defaults: dict[str, Any] = {
        "bene_id": "BEN-000001",
        "provider_id": "PRV-00001",
        "specialty": "Internal Medicine",
        "hcpcs": "99213",
        "modifier": "",
        "units": 1,
        "dx_group": "DXG01",
        "pos": "11",
        "submitted_amount": 150.0,
        "allowed_amount": 100.0,
        "paid_amount": 80.0,
    }
    out = []
    for i, row in enumerate(rows):
        d = {**defaults, "claim_id": f"CLM-{i + 1:08d}", **row}
        d.setdefault("service_end_date", d["service_date"])
        d.setdefault("submit_date", d["service_date"])
        out.append(d)
    df = pd.DataFrame(out)
    for col in ["service_date", "service_end_date", "submit_date"]:
        df[col] = pd.to_datetime(df[col])
    return df


def test_split_assignment(cfg: dict[str, Any]) -> None:
    df = _claims(
        [
            {"service_date": "2024-02-10"},
            {"service_date": "2024-07-01"},
            {"service_date": "2025-05-20"},
            {"service_date": "2025-11-03"},
        ]
    )
    out = assign_split(df, cfg)
    assert list(out["split"]) == ["burn_in", "train", "val", "test"]


def test_duplicate_features_backward_only(cfg: dict[str, Any]) -> None:
    df = _claims(
        [
            {"service_date": "2024-05-01", "submit_date": "2024-05-05"},
            {"service_date": "2024-05-01", "submit_date": "2024-05-20"},  # exact dup, later submit
            {"service_date": "2024-08-01", "submit_date": "2024-08-05", "hcpcs": "97110"},
        ]
    )
    out = duplicate_features(df, cfg["duplicates"]).set_index("claim_id")
    # The EARLIER claim is never flagged; the LATER one is.
    assert out.loc["CLM-00000001", "exact_dup_prior"] == 0
    assert out.loc["CLM-00000001", "dup_cand_count"] == 0
    assert out.loc["CLM-00000002", "exact_dup_prior"] == 1
    assert out.loc["CLM-00000002", "dup_max_sim"] > 0.9
    # Different code, no candidates.
    assert out.loc["CLM-00000003", "dup_cand_count"] == 0


def test_duplicate_window_excludes_distant_claims(cfg: dict[str, Any]) -> None:
    df = _claims(
        [
            {"service_date": "2024-01-10"},
            {"service_date": "2024-06-10"},  # 5 months later, outside 45d window
        ]
    )
    out = duplicate_features(df, cfg["duplicates"]).set_index("claim_id")
    assert out.loc["CLM-00000002", "dup_cand_count"] == 0


def test_bene_history_rolling_is_strictly_prior(cfg: dict[str, Any]) -> None:
    df = _claims(
        [
            {"service_date": "2024-05-01", "paid_amount": 100.0},
            {"service_date": "2024-05-10", "paid_amount": 50.0},
            {"service_date": "2024-05-10", "paid_amount": 25.0, "hcpcs": "97110"},
            {"service_date": "2024-07-30", "paid_amount": 10.0},
        ]
    )
    out = beneficiary_history_features(df).set_index("claim_id")
    assert out.loc["CLM-00000001", "bene_claims_30d"] == 0
    # Same-day claims are excluded by closed='left'.
    assert out.loc["CLM-00000002", "bene_claims_30d"] == 1
    assert out.loc["CLM-00000002", "bene_paid_30d"] == 100.0
    assert out.loc["CLM-00000003", "bene_claims_30d"] == 1
    # 81 days later: outside the 30d window.
    assert out.loc["CLM-00000004", "bene_claims_30d"] == 0
    assert out.loc["CLM-00000002", "bene_same_day_claims"] == 1
    assert out.loc["CLM-00000002", "days_since_bene_last_claim"] == 9


def test_unbundle_flag_requires_missing_modifier(
    cfg: dict[str, Any], ref: dict[str, pd.DataFrame]
) -> None:
    rows = [
        {"service_date": "2024-06-03", "hcpcs": "93000", "paid_amount": 40.0},
        {"service_date": "2024-06-03", "hcpcs": "93010", "paid_amount": 15.0},  # component, no mod
        {"service_date": "2024-06-03", "hcpcs": "93005", "paid_amount": 20.0, "modifier": "59"},
    ]
    df = assign_split(_claims(rows), cfg)
    peer = fit_peer_stats(df, ref["em_levels"], cfg)
    out = amount_and_coding_features(df, ref, peer).set_index("claim_id")
    assert out.loc["CLM-00000002", "unbundle_component_no_mod"] == 1
    assert out.loc["CLM-00000003", "unbundle_component_no_mod"] == 0  # modifier 59 bypass
    assert out.loc["CLM-00000001", "unbundle_component_no_mod"] == 0  # comprehensive itself


def test_em_gap_and_amount_ratios(cfg: dict[str, Any], ref: dict[str, pd.DataFrame]) -> None:
    rows = [
        {"service_date": "2024-06-03", "hcpcs": "99213", "paid_amount": 80.0},
        {"service_date": "2024-06-04", "hcpcs": "99213", "paid_amount": 80.0},
        {"service_date": "2024-06-05", "hcpcs": "99213", "paid_amount": 80.0},
        {"service_date": "2024-06-06", "hcpcs": "99215", "paid_amount": 160.0},
        {"service_date": "2024-06-07", "hcpcs": "99213", "paid_amount": 240.0},  # 3x the median
    ]
    df = assign_split(_claims(rows), cfg)
    peer = fit_peer_stats(df, ref["em_levels"], cfg)
    out = amount_and_coding_features(df, ref, peer).set_index("claim_id")
    assert out.loc["CLM-00000004", "em_level"] == 5
    assert out.loc["CLM-00000004", "em_gap_vs_specialty"] > 0
    assert out.loc["CLM-00000005", "paid_vs_code_median"] == pytest.approx(3.0)
    # Facility-typical code billed in the office scores high POS atypicality.
    assert out.loc["CLM-00000001", "pos_atypicality"] < 0.5
