#!/usr/bin/env python
"""Novel-scheme holdout experiment.

Question: if a scheme family was NEVER in the training labels, does the
unsupervised leg still surface it? We drop all claims whose only patterns are
F09 (dx mismatch) and/or F11 (collusion clusters) from the TRAIN split, retrain
the full stack, and measure recall on exactly those claims in the TEST split —
supervised-only vs the ensemble with its anomaly component.

Output: docs/assets/novel_scheme_results.json (consumed by the evaluation report).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modeling.train import load_model_config, prepare_training_frame, train_all
from src.utils.logging import configure_logging

HOLDOUT_PATTERNS = ("F09_dx_mismatch", "F11_collusion_cluster")


def _only_holdout(patterns: str) -> bool:
    parts = [p for p in patterns.split(";") if p]
    return bool(parts) and all(p in HOLDOUT_PATTERNS for p in parts)


def main() -> int:
    configure_logging("INFO")
    cfg = load_model_config()
    features = pd.read_parquet("data/processed/features.parquet")
    labels = pd.read_parquet("data/processed/labels.parquet")

    frame, numeric, _ = prepare_training_frame(
        features[features["split"] != "burn_in"], labels, bool(cfg["include_rule_features"])
    )
    holdout_mask = frame["patterns"].map(_only_holdout)

    # Remove holdout-scheme positives from TRAIN only (they simply never happened).
    train_drop = holdout_mask & (frame["split"] == "train") & frame["is_improper"]
    ablated = frame[~train_drop].copy()
    result = train_all(ablated, numeric, cfg)

    test = frame[frame["split"] == "test"].copy()
    X_test = test[result.bundle.numeric_features + result.bundle.categorical_features]
    scored = result.bundle.risk_score(X_test)
    test = test.assign(
        supervised=scored["supervised_probability"].to_numpy(),
        ensemble=scored["risk_score"].to_numpy(),
        anomaly=scored["anomaly_percentile"].to_numpy(),
    )

    novel = test[holdout_mask.reindex(test.index, fill_value=False) & test["is_improper"]]
    out: dict[str, object] = {
        "holdout_patterns": list(HOLDOUT_PATTERNS),
        "train_positives_removed": int(train_drop.sum()),
        "novel_test_claims": len(novel),
    }
    for pct in (0.01, 0.05, 0.10):
        k = max(1, int(len(test) * pct))
        for score_col in ("supervised", "ensemble", "anomaly"):
            top = test.nlargest(k, score_col)
            captured = int(top["claim_id"].isin(novel["claim_id"]).sum())
            out[f"novel_recall_at_{int(pct * 100)}pct_{score_col}"] = round(
                captured / max(1, len(novel)), 4
            )
    out["mean_anomaly_pct_novel"] = round(float(novel["anomaly"].mean()), 4)
    out["mean_anomaly_pct_all_test"] = round(float(test["anomaly"].mean()), 4)

    dest = Path("docs/assets")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "novel_scheme_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
