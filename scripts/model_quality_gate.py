#!/usr/bin/env python
"""CI model-quality regression gate.

Generates a small fixture dataset, runs the full feature + training pipeline,
and fails (exit 1) if quality drops below floors — catching silent regressions
in features, injection, or training code before they merge.

Usage: python scripts/model_quality_gate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ingest_cms import load_snapshot
from src.data.reference import load_reference_frames
from src.data.synthetic.assemble import build_dataset
from src.data.synthetic.generate import load_generation_config
from src.features.builder import build_features
from src.features.splits import load_features_config
from src.modeling.train import load_model_config, prepare_training_frame, train_all
from src.utils.logging import configure_logging

FLOORS = {
    "xgboost_test_pr_auc": 0.45,
    "ensemble_test_pr_auc": 0.45,
    "ensemble_test_recall_at_5pct": 0.35,
}


def main() -> int:
    configure_logging("WARNING")
    gen_cfg = load_generation_config()
    gen_cfg.update(n_providers=40, n_beneficiaries=1200, n_months=10, seed=5)
    bundle_ds = build_dataset(gen_cfg, load_snapshot(), load_reference_frames())
    claims = bundle_ds.claims.copy()
    for col in ["service_date", "service_end_date", "submit_date"]:
        claims[col] = pd.to_datetime(claims[col])

    fcfg = load_features_config()
    fcfg["splits"] = {"burn_in_end": "2024-02", "train_end": "2024-06", "val_end": "2024-08"}
    features, _, meta = build_features(claims, bundle_ds.providers, load_reference_frames(), fcfg)

    mcfg = load_model_config()
    mcfg["xgboost"] = {**mcfg["xgboost"], "n_estimators": 150, "early_stopping_rounds": 25}
    frame, numeric, _ = prepare_training_frame(
        features[features["split"] != "burn_in"], bundle_ds.labels, True
    )
    result = train_all(frame, numeric, mcfg, meta)

    observed = {
        "xgboost_test_pr_auc": result.metrics["xgboost"]["test"]["pr_auc"],
        "ensemble_test_pr_auc": result.metrics["ensemble"]["test"]["pr_auc"],
        "ensemble_test_recall_at_5pct": result.metrics["ensemble"]["test"]["recall_at_5pct"],
        "rules_test_pr_auc": result.metrics["rules_baseline"]["test"]["pr_auc"],
    }
    failures = [
        f"{name}: {observed[name]:.4f} < floor {floor}"
        for name, floor in FLOORS.items()
        if observed[name] < floor
    ]
    if result.metrics["ensemble"]["test"]["pr_auc"] <= observed["rules_test_pr_auc"]:
        failures.append("ensemble does not beat the rules baseline")

    print(
        json.dumps(
            {
                "observed": {k: round(v, 4) for k, v in observed.items()},
                "floors": FLOORS,
                "failures": failures,
            },
            indent=2,
        )
    )
    if failures:
        print("MODEL QUALITY GATE FAILED", file=sys.stderr)
        return 1
    print("model quality gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
