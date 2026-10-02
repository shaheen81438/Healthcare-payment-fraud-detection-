#!/usr/bin/env python
"""Batch scoring with explanations: writes scoring.predictions and scoring.explanations.

Scores the test window (the "current production traffic" in this project's
timeline), computes SHAP-based reasons for every claim, saves them to Postgres
and parquet, and draws the global importance chart for the model card.

Usage: python scripts/score_claims.py [--skip-db] [--top-k 6]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.financial import (
    estimated_payment_at_risk,
    load_cost_params,
    recommended_action,
)
from src.modeling.bundle import ScoringBundle
from src.modeling.explain import ClaimExplainer, plot_global_importance
from src.modeling.train import load_model_config, prepare_training_frame
from src.utils.logging import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-db", action="store_true")
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()
    configure_logging("INFO")

    cfg = load_model_config()
    bundle = ScoringBundle.load_latest()
    explainer = ClaimExplainer(bundle)
    policy = bundle.threshold_policy
    if not policy:
        raise SystemExit("bundle has no threshold policy; run scripts/evaluate_model.py first")

    features = pd.read_parquet("data/processed/features.parquet")
    labels = pd.read_parquet("data/processed/labels.parquet")
    frame, _, _ = prepare_training_frame(
        features[features["split"] != "burn_in"], labels, bool(cfg["include_rule_features"])
    )
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    X = test[bundle.numeric_features + bundle.categorical_features]

    scored = bundle.risk_score(X)
    contribs, base_value = explainer.top_contributions(X, top_k=args.top_k)
    scored_at = datetime.now(UTC)

    reasons = [
        explainer.reason_codes(
            c,
            rule_reasons=str(test.loc[i, "rule_reasons"]),
            anomaly_percentile=float(scored.loc[i, "anomaly_percentile"]),
        )
        for i, c in zip(test.index, contribs, strict=True)
    ]

    predictions = pd.DataFrame(
        {
            "claim_id": test["claim_id"],
            "provider_id": test["provider_id"],
            "bene_id": test["bene_id"],
            "specialty": test["specialty"],
            "hcpcs": test["hcpcs"],
            "service_date": test["service_date"],
            "month": test["month"],
            "paid_amount": test["paid_amount"],
            "risk_score": scored["risk_score"],
            "supervised_probability": scored["supervised_probability"],
            "anomaly_percentile": scored["anomaly_percentile"],
            "rule_score": test["rule_score"],
            "rule_hits": test["rule_hits"],
            "recommended_action": [
                recommended_action(float(s), policy) for s in scored["risk_score"]
            ],
            "estimated_payment_at_risk": estimated_payment_at_risk(
                scored["supervised_probability"].to_numpy(),
                test["paid_amount"].to_numpy(dtype=float),
            ),
            "reason_codes": [json.dumps(r) for r in reasons],
            "model_version": bundle.model_version,
            "scored_at": scored_at,
        }
    )
    explanations = pd.DataFrame(
        {
            "claim_id": test["claim_id"],
            "model_version": bundle.model_version,
            "base_value": base_value,
            "shap_top": [json.dumps(c) for c in contribs],
            "scored_at": scored_at,
        }
    )

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    predictions.to_parquet("data/processed/predictions.parquet", index=False)
    explanations.to_parquet("data/processed/explanations.parquet", index=False)

    if not args.skip_db:
        from src.data.db import write_df

        write_df(predictions, "predictions", schema="scoring")
        write_df(explanations, "explanations", schema="scoring")

    importance = explainer.global_importance(X)
    Path("docs/assets").mkdir(parents=True, exist_ok=True)
    importance.to_csv("docs/assets/global_importance.csv", index=False)
    plot_global_importance(importance, "docs/assets/shap_summary.png")

    _, raw_costs = load_cost_params()
    flagged = predictions[predictions["recommended_action"] == "manual_review"]
    print(
        json.dumps(
            {
                "scored": len(predictions),
                "manual_review": int((predictions["recommended_action"] == "manual_review").sum()),
                "monitor": int((predictions["recommended_action"] == "monitor").sum()),
                "estimated_payment_at_risk_flagged": round(
                    float(flagged["estimated_payment_at_risk"].sum()), 2
                ),
                "top_global_features": importance.head(5)["feature"].tolist(),
                "review_threshold": policy["review_threshold"],
                "default_capacity": raw_costs["default_capacity"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
