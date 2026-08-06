"""Generator invariants: determinism, label separation, pattern injection."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.data.ingest_cms import load_snapshot
from src.data.reference import load_reference_frames
from src.data.synthetic.assemble import DatasetBundle, build_dataset
from src.data.synthetic.generate import load_generation_config

TINY_OVERRIDES: dict[str, Any] = {
    "n_providers": 30,
    "n_beneficiaries": 900,
    "n_months": 6,
    "seed": 11,
}


def _tiny_cfg() -> dict[str, Any]:
    cfg = load_generation_config()
    cfg.update(TINY_OVERRIDES)
    return cfg


@pytest.fixture(scope="module")
def bundle() -> DatasetBundle:
    return build_dataset(_tiny_cfg(), load_snapshot(), load_reference_frames())


def test_labels_are_separated_from_claims(bundle: DatasetBundle) -> None:
    label_cols = {"is_improper", "patterns", "improper_amount"}
    assert not label_cols & set(bundle.claims.columns)
    assert not label_cols & set(bundle.raw_claims.columns)
    assert {"claim_id", *label_cols} == set(bundle.labels.columns)
    assert len(bundle.labels) == len(bundle.claims)
    assert bundle.labels["claim_id"].is_unique


def test_positive_rate_in_sane_band(bundle: DatasetBundle) -> None:
    rate = bundle.labels["is_improper"].mean()
    assert 0.01 < rate < 0.20


def test_improper_amount_never_exceeds_paid(bundle: DatasetBundle) -> None:
    merged = bundle.labels.merge(bundle.claims[["claim_id", "paid_amount"]], on="claim_id")
    assert (merged["improper_amount"] <= merged["paid_amount"] + 0.01).all()
    positives = merged[merged["is_improper"]]
    assert (positives["improper_amount"] > 0).all()


def test_exact_duplicates_exist_with_distinct_claim_ids(bundle: DatasetBundle) -> None:
    dup_ids = bundle.labels.loc[
        bundle.labels["patterns"].str.contains("F01_exact_duplicate"), "claim_id"
    ]
    assert len(dup_ids) > 0
    dups = bundle.claims[bundle.claims["claim_id"].isin(dup_ids)]
    keys = ["bene_id", "provider_id", "hcpcs", "service_date", "paid_amount"]
    originals = bundle.claims.merge(dups[keys], on=keys)
    # Each injected duplicate matches at least one other claim (the original) on content.
    assert (originals.groupby(keys).size() >= 2).all()


def test_provider_and_bene_ids_are_synthetic(bundle: DatasetBundle) -> None:
    assert bundle.providers["provider_id"].str.match(r"PRV-\d{5}").all()
    assert bundle.beneficiaries["bene_id"].str.match(r"BEN-\d{6}").all()
    assert bundle.claims["claim_id"].str.match(r"CLM-\d{8}").all()
    # No real-NPI column anywhere in scored entities.
    for frame in (bundle.claims, bundle.providers, bundle.beneficiaries):
        assert "npi" not in frame.columns


def test_generation_is_deterministic() -> None:
    cms, ref = load_snapshot(), load_reference_frames()
    a = build_dataset(_tiny_cfg(), cms, ref)
    b = build_dataset(_tiny_cfg(), cms, ref)
    pd.testing.assert_frame_equal(a.claims, b.claims)
    pd.testing.assert_frame_equal(a.labels, b.labels)
    assert a.dataset_version == b.dataset_version
