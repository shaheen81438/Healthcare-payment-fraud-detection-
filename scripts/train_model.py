#!/usr/bin/env python
"""Train all models, save the scoring bundle, and log/register with MLflow.

Usage: python scripts/train_model.py [--no-register] [--config configs/model.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modeling.registry import log_and_register
from src.modeling.train import load_model_config, prepare_training_frame, train_all
from src.utils.logging import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    configure_logging("INFO")

    cfg = load_model_config(args.config)
    features = pd.read_parquet("data/processed/features.parquet")
    labels = pd.read_parquet("data/processed/labels.parquet")
    feature_meta = json.loads(Path("models/feature_store/feature_meta.json").read_text())

    frame, numeric, rule_hits = prepare_training_frame(
        features[features["split"] != "burn_in"], labels, bool(cfg["include_rule_features"])
    )
    result = train_all(frame, numeric, cfg, feature_meta)
    result.rule_hit_counts = rule_hits

    # Attach the dataset version and save the bundle.
    stats = json.loads(Path("data/interim/generation_stats.json").read_text())
    dataset_version = "unknown"
    meta_path = Path("data/interim/dataset_version.txt")
    if meta_path.exists():
        dataset_version = meta_path.read_text().strip()
    result.bundle.metadata.update(
        {"dataset_version": dataset_version, "generation_stats_rows": stats.get("rows_claims")}
    )
    bundle_dir = result.bundle.save(Path("models") / str(cfg["model_version"]))
    (bundle_dir / "metrics.json").write_text(
        json.dumps(result.metrics, indent=2, default=str), encoding="utf-8"
    )
    (bundle_dir / "rule_hit_counts.json").write_text(
        json.dumps(rule_hits, indent=2), encoding="utf-8"
    )

    run_id = log_and_register(
        bundle_dir, result.metrics, cfg, dataset_version, register=not args.no_register
    )

    summary = {
        "model_version": result.bundle.model_version,
        "bundle_dir": str(bundle_dir),
        "mlflow_run_id": run_id,
        "ensemble_weight": result.metrics["ensemble"]["weight"],
        "val": {
            m: round(result.metrics[m]["val"]["pr_auc"], 4)
            for m in [
                "rules_baseline",
                "logistic_regression",
                "xgboost",
                "isolation_forest",
                "ensemble",
            ]
        },
        "test": {
            m: round(result.metrics[m]["test"]["pr_auc"], 4)
            for m in [
                "rules_baseline",
                "logistic_regression",
                "xgboost",
                "isolation_forest",
                "ensemble",
            ]
        },
        "test_recall_at_1pct": round(result.metrics["ensemble"]["test"]["recall_at_1pct"], 4),
        "test_recall_at_5pct": round(result.metrics["ensemble"]["test"]["recall_at_5pct"], 4),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
