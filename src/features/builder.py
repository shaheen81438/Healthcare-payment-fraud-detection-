"""Build the feature matrix: assign splits, fit frozen peer stats, then add claim and provider features."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from src.features.claim_features import (
    amount_and_coding_features,
    beneficiary_history_features,
)
from src.features.dup_similarity import duplicate_features
from src.features.peer_stats import fit_peer_stats
from src.features.provider_features import PROVIDER_FEATURES, build_provider_month_features
from src.features.splits import assign_split

logger = logging.getLogger(__name__)

ID_COLUMNS = ["claim_id", "bene_id", "provider_id", "service_date", "month", "split"]
PASSTHROUGH_COLUMNS = ["hcpcs", "modifier"]

NUMERIC_FEATURES = [
    "paid_amount",
    "submitted_amount",
    "allowed_amount",
    "units",
    "paid_per_unit",
    "paid_vs_code_median",
    "paid_vs_spec_code_median",
    "units_vs_code_p95",
    "submit_lag_days",
    "service_days",
    "dow",
    "month_num",
    "mod_25",
    "mod_59",
    "mod_ltrt",
    "em_level",
    "em_gap_vs_specialty",
    "dxproc_compat",
    "pos_office",
    "pos_atypicality",
    "unbundle_component_no_mod",
    "bene_same_day_claims",
    "bene_provider_same_day_claims",
    "days_since_bene_last_claim",
    "days_since_bene_same_hcpcs",
    "bene_claims_30d",
    "bene_paid_30d",
    "bene_distinct_providers_prior",
    "dup_max_sim",
    "dup_cand_count",
    "exact_dup_prior",
    *PROVIDER_FEATURES,
]
CATEGORICAL_FEATURES = ["specialty", "pos", "dx_group"]
LABEL_COLUMNS = {"is_improper", "patterns", "improper_amount"}


def build_features(
    claims: pd.DataFrame,
    providers: pd.DataFrame,
    ref: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    """Build the full point-in-time feature matrix.

    Returns (features, frozen peer stats, feature metadata). ``claims`` must be
    the cleaned staging frame. Any label column in it raises an error.
    """
    forbidden = LABEL_COLUMNS & set(claims.columns)
    if forbidden:
        raise ValueError(f"label columns must never enter the feature builder: {forbidden}")

    df = assign_split(claims, cfg)
    train = df[df["split"] == "train"]
    peer = fit_peer_stats(train, ref["em_levels"], cfg)

    df = amount_and_coding_features(df, ref, peer)
    df = beneficiary_history_features(df)

    dup = duplicate_features(df, cfg["duplicates"])
    df = df.merge(dup, on="claim_id", how="left")

    df["month"] = df["service_date"].dt.to_period("M").astype(str)
    prov_month = build_provider_month_features(
        claims, ref["em_levels"], peer["code_stats"], peer["provider_month_peer"], providers, cfg
    )
    df = df.merge(prov_month, on=["provider_id", "month"], how="left")
    for col in PROVIDER_FEATURES:
        df[col] = df[col].fillna(0.0)

    features = df[ID_COLUMNS + PASSTHROUGH_COLUMNS + CATEGORICAL_FEATURES + NUMERIC_FEATURES].copy()
    assert not (LABEL_COLUMNS & set(features.columns))

    meta: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows": len(features),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "split_counts": features["split"].value_counts().to_dict(),
        "splits_config": cfg["splits"],
    }
    logger.info(
        "features_built",
        extra={
            "rows": len(features),
            "n_features": len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES),
        },
    )
    return features, peer, meta
