"""Leakage tests: features must never see the future."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.data.ingest_cms import load_snapshot
from src.data.reference import load_reference_frames
from src.data.synthetic.assemble import build_dataset
from src.data.synthetic.generate import load_generation_config
from src.features.builder import LABEL_COLUMNS, build_features
from src.features.peer_stats import fit_peer_stats, peer_stats_fingerprint
from src.features.splits import load_features_config


@pytest.fixture(scope="module")
def small_world() -> dict[str, Any]:
    gen_cfg = load_generation_config()
    gen_cfg.update(n_providers=25, n_beneficiaries=700, n_months=10, seed=23)
    bundle = build_dataset(gen_cfg, load_snapshot(), load_reference_frames())
    claims = bundle.claims.copy()
    for col in ["service_date", "service_end_date", "submit_date"]:
        claims[col] = pd.to_datetime(claims[col])
    fcfg = load_features_config()
    # Align split boundaries to the 10-month window (2024-01..2024-10).
    fcfg["splits"] = {"burn_in_end": "2024-02", "train_end": "2024-06", "val_end": "2024-08"}
    return {
        "claims": claims,
        "providers": bundle.providers,
        "labels": bundle.labels,
        "ref": load_reference_frames(),
        "cfg": fcfg,
    }


def test_label_columns_rejected_and_absent(small_world: dict[str, Any]) -> None:
    w = small_world
    poisoned = w["claims"].merge(w["labels"], on="claim_id")
    with pytest.raises(ValueError, match="label columns"):
        build_features(poisoned, w["providers"], w["ref"], w["cfg"])
    features, _, _ = build_features(w["claims"], w["providers"], w["ref"], w["cfg"])
    assert not (LABEL_COLUMNS & set(features.columns))


def test_features_invariant_when_future_appended(small_world: dict[str, Any]) -> None:
    """Features of past claims must not change when later months of data arrive."""
    w = small_world
    cutoff = pd.Timestamp("2024-08-01")
    past = w["claims"][w["claims"]["service_date"] < cutoff].copy()

    full_features, _, _ = build_features(w["claims"], w["providers"], w["ref"], w["cfg"])
    past_features, _, _ = build_features(past, w["providers"], w["ref"], w["cfg"])

    on_past = (
        full_features[full_features["claim_id"].isin(past["claim_id"])]
        .sort_values("claim_id")
        .reset_index(drop=True)
    )
    past_only = past_features.sort_values("claim_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(on_past, past_only, check_like=True)


def test_peer_stats_fit_on_train_window_only(small_world: dict[str, Any]) -> None:
    """Corrupting val/test amounts must not move the frozen peer stats."""
    w = small_world
    from src.features.splits import assign_split

    df = assign_split(w["claims"], w["cfg"])
    stats_a = fit_peer_stats(df[df["split"] == "train"], w["ref"]["em_levels"], w["cfg"])

    corrupted = df.copy()
    mask = corrupted["split"].isin(["val", "test"])
    corrupted.loc[mask, "paid_amount"] = corrupted.loc[mask, "paid_amount"] * 100
    stats_b = fit_peer_stats(
        corrupted[corrupted["split"] == "train"], w["ref"]["em_levels"], w["cfg"]
    )
    assert peer_stats_fingerprint(stats_a) == peer_stats_fingerprint(stats_b)


def test_provider_features_lag_one_month(small_world: dict[str, Any]) -> None:
    """A provider's month-m claims must not contribute to month-m provider features."""
    w = small_world
    features, _, _ = build_features(w["claims"], w["providers"], w["ref"], w["cfg"])
    claims = w["claims"]
    pid = claims["provider_id"].value_counts().index[0]
    months = sorted(
        claims.loc[claims["provider_id"] == pid, "service_date"].dt.to_period("M").unique()
    )
    m = str(months[len(months) // 2])
    prev = str(months[len(months) // 2 - 1])
    actual_prev_1m = int(
        (
            (claims["provider_id"] == pid)
            & (claims["service_date"].dt.to_period("M").astype(str) == prev)
        ).sum()
    )
    row = features[(features["provider_id"] == pid) & (features["month"] == m)].iloc[0]
    assert row["prov_claims_1m"] == actual_prev_1m
