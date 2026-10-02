#!/usr/bin/env python
"""Build the point-in-time feature matrix and frozen peer-stat artifacts.

Usage: python scripts/build_features.py [--skip-db]
Inputs: data/processed/claims.parquet, providers.parquet, data/reference/*
Outputs: data/processed/features.parquet, models/feature_store/*, features.claim_features (PG)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.reference import load_reference_frames
from src.features.builder import build_features
from src.features.peer_stats import save_peer_stats
from src.features.splits import load_features_config
from src.utils.logging import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-db", action="store_true")
    args = parser.parse_args()
    configure_logging("INFO")

    cfg = load_features_config()
    claims = pd.read_parquet("data/processed/claims.parquet")
    providers = pd.read_parquet("data/processed/providers.parquet")
    ref = load_reference_frames()

    features, peer, meta = build_features(claims, providers, ref, cfg)

    features.to_parquet("data/processed/features.parquet", index=False)
    store = Path("models/feature_store")
    save_peer_stats(peer, store)
    (store / "feature_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Provider-month table for online serving (the API's point-in-time provider lookup).
    from src.features.provider_features import build_provider_month_features

    provider_month = build_provider_month_features(
        claims, ref["em_levels"], peer["code_stats"], peer["provider_month_peer"], providers, cfg
    )
    provider_month.to_parquet("data/processed/provider_month_features.parquet", index=False)

    if not args.skip_db:
        from src.data.db import write_df

        write_df(features, "claim_features", schema="features")

    print(
        json.dumps(
            {
                "rows": meta["rows"],
                "numeric_features": len(meta["numeric_features"]),
                "categorical_features": len(meta["categorical_features"]),
                "split_counts": meta["split_counts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
