#!/usr/bin/env python
"""Full evaluation: score val/test, select the operating policy, build the report.

Usage: python scripts/evaluate_model.py
Outputs:
  - docs/evaluation_report.md + docs/assets/*.png
  - data/processed/scored_test.parquet (scores + actions for downstream phases)
  - threshold policy persisted into the model bundle (models/<version>/metadata.json)
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.financial import (
    capacity_sweep,
    estimated_payment_at_risk,
    load_cost_params,
    recommended_action,
    select_policy,
    threshold_sweep,
)
from src.evaluation.metrics import calibration_report, threshold_metrics
from src.evaluation.report import (
    build_report,
    plot_capacity_recall,
    plot_env_curve,
    plot_novel_scheme,
    plot_pr_curves,
    plot_reliability,
    plot_score_distribution,
)
from src.modeling.bundle import ScoringBundle
from src.modeling.train import load_model_config, prepare_training_frame
from src.utils.logging import configure_logging


def main() -> int:
    configure_logging("INFO")
    cfg = load_model_config()
    params, raw_costs = load_cost_params()
    bundle = ScoringBundle.load_latest()

    features = pd.read_parquet("data/processed/features.parquet")
    labels = pd.read_parquet("data/processed/labels.parquet")
    frame, _, _ = prepare_training_frame(
        features[features["split"] != "burn_in"], labels, bool(cfg["include_rule_features"])
    )
    X_cols = bundle.numeric_features + bundle.categorical_features

    scored = {}
    for split in ["val", "test"]:
        part = frame[frame["split"] == split].reset_index(drop=True)
        s = bundle.risk_score(part[X_cols])
        scored[split] = part.assign(
            risk_score=s["risk_score"].to_numpy(),
            supervised_probability=s["supervised_probability"].to_numpy(),
            anomaly_percentile=s["anomaly_percentile"].to_numpy(),
        )

    val, test = scored["val"], scored["test"]
    y_val = val["is_improper"].to_numpy(dtype=int)
    y_test = test["is_improper"].to_numpy(dtype=int)

    # ---- policy selection on validation
    policy = select_policy(
        val["risk_score"].to_numpy(),
        val["supervised_probability"].to_numpy(),
        val["paid_amount"].to_numpy(dtype=float),
        y_val,
        val["improper_amount"].to_numpy(dtype=float),
        params,
        float(raw_costs["default_capacity"]),
    )
    bundle.threshold_policy = policy
    bundle_dir = bundle.save(Path("models") / bundle.model_version)
    # Re-register so the MLflow production alias serves the bundle WITH its policy.
    train_metrics_for_log = json.loads((bundle_dir / "metrics.json").read_text())
    from src.modeling.registry import log_and_register

    log_and_register(
        bundle_dir,
        train_metrics_for_log,
        cfg,
        str(bundle.metadata.get("dataset_version", "unknown")),
        register=True,
    )

    # ---- test-window sweeps + metrics
    args_test = (
        test["risk_score"].to_numpy(),
        test["supervised_probability"].to_numpy(),
        test["paid_amount"].to_numpy(dtype=float),
        y_test,
        test["improper_amount"].to_numpy(dtype=float),
        params,
    )
    cap_table = capacity_sweep(*args_test, capacities=list(raw_costs["capacity_grid"]))
    tau_sweep = threshold_sweep(*args_test)
    at_tau = threshold_metrics(y_test, test["risk_score"].to_numpy(), policy["review_threshold"])
    calib = calibration_report(y_test, test["supervised_probability"].to_numpy())

    # ---- charts
    assets = Path("docs/assets")
    assets.mkdir(parents=True, exist_ok=True)
    scores_by_model = {
        "ensemble": test["risk_score"].to_numpy(),
        "xgboost": test["supervised_probability"].to_numpy(),
        "rules_baseline": test["rule_score"].to_numpy(),
        "isolation_forest": test["anomaly_percentile"].to_numpy(),
    }
    plot_pr_curves(y_test, scores_by_model, assets / "pr_curves.png")
    plot_capacity_recall(y_test, scores_by_model, assets / "capacity_recall.png")
    plot_reliability(y_test, test["supervised_probability"].to_numpy(), assets / "reliability.png")
    plot_env_curve(tau_sweep, policy, assets / "env_curve.png")
    plot_score_distribution(
        y_test, test["risk_score"].to_numpy(), assets / "score_distribution.png"
    )
    novel = json.loads(Path("docs/assets/novel_scheme_results.json").read_text())
    plot_novel_scheme(novel, assets / "novel_scheme.png")

    # ---- tables
    train_metrics = json.loads((Path("models") / bundle.model_version / "metrics.json").read_text())
    rows = []
    for model in [
        "rules_baseline",
        "logistic_regression",
        "xgboost",
        "isolation_forest",
        "ensemble",
    ]:
        t = train_metrics[model]["test"]
        rows.append(
            {
                "Model": model.replace("_", " "),
                "PR-AUC": round(t["pr_auc"], 4),
                "ROC-AUC": round(t["roc_auc"], 4),
                "Recall@1%": round(t["recall_at_1pct"], 3),
                "Recall@5%": round(t["recall_at_5pct"], 3),
                "Recall@10%": round(t["recall_at_10pct"], 3),
            }
        )
    metrics_table = pd.DataFrame(rows)

    cap_display = cap_table[
        [
            "capacity_share",
            "n_reviewed",
            "precision",
            "recall",
            "dollars_recovered",
            "net_value",
            "dollars_identified_per_1000_reviewed",
            "expected_net_value",
        ]
    ].copy()
    cap_display["capacity_share"] = (cap_display["capacity_share"] * 100).map("{:.1f}%".format)
    for col in ["precision", "recall"]:
        cap_display[col] = cap_display[col].map("{:.3f}".format)
    for col in [
        "dollars_recovered",
        "net_value",
        "dollars_identified_per_1000_reviewed",
        "expected_net_value",
    ]:
        cap_display[col] = cap_display[col].map("${:,.0f}".format)

    novel_table = pd.DataFrame(
        [
            {
                "Review capacity": f"top {p}%",
                "Supervised only": f"{novel[f'novel_recall_at_{p}pct_supervised']:.1%}",
                "Ensemble": f"{novel[f'novel_recall_at_{p}pct_ensemble']:.1%}",
                "Anomaly only": f"{novel[f'novel_recall_at_{p}pct_anomaly']:.1%}",
            }
            for p in [1, 5, 10]
        ]
    )

    # Per-pattern recall at the operating point.
    flagged = test["risk_score"].to_numpy() >= policy["review_threshold"]
    pattern_rows = []
    exploded = test.loc[test["is_improper"], ["patterns"]].copy()
    exploded["flagged"] = flagged[test["is_improper"].to_numpy()]
    exploded = exploded.assign(pattern=exploded["patterns"].str.split(";")).explode("pattern")
    for pattern, grp in exploded.groupby("pattern"):
        pattern_rows.append(
            {
                "Pattern": pattern,
                "Test claims": len(grp),
                "Flagged at τ": int(grp["flagged"].sum()),
                "Recall": f"{grp['flagged'].mean():.1%}",
            }
        )
    pattern_table = pd.DataFrame(pattern_rows).sort_values("Recall", ascending=False)

    context = {
        "model_version": bundle.model_version,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "dataset_version": bundle.metadata.get("dataset_version", "unknown"),
        "test_window": f"{test['service_date'].min():%Y-%m} → {test['service_date'].max():%Y-%m}",
        "n_test": len(test),
        "test_positive_rate": float(y_test.mean()),
        "metrics_table": metrics_table,
        "policy": policy,
        "confusion": at_tau["confusion_matrix"],
        "at_threshold": at_tau,
        "capacity_table": cap_display,
        "calibration": calib,
        "novel_table": novel_table,
        "pattern_table": pattern_table,
    }
    build_report(context)

    # ---- persist scored test for downstream phases (explanations, dashboard, monitoring)
    out_cols = [
        "claim_id",
        "provider_id",
        "bene_id",
        "specialty",
        "hcpcs",
        "service_date",
        "month",
        "split",
        "paid_amount",
        "risk_score",
        "supervised_probability",
        "anomaly_percentile",
        "rule_score",
        "rule_hits",
        "rule_reasons",
        "is_improper",
        "patterns",
        "improper_amount",
    ]
    scored_test = test[out_cols].copy()
    scored_test["estimated_payment_at_risk"] = estimated_payment_at_risk(
        test["supervised_probability"].to_numpy(), test["paid_amount"].to_numpy(dtype=float)
    )
    scored_test["recommended_action"] = [
        recommended_action(s, policy) for s in scored_test["risk_score"]
    ]
    scored_test["model_version"] = bundle.model_version
    scored_test.to_parquet("data/processed/scored_test.parquet", index=False)

    print(
        json.dumps(
            {
                "review_threshold": policy["review_threshold"],
                "test_precision_at_tau": round(at_tau["precision"], 4),
                "test_recall_at_tau": round(at_tau["recall"], 4),
                "net_value_at_1pct_capacity": float(
                    cap_table.loc[cap_table["capacity_share"] == 0.01, "net_value"].iloc[0]
                ),
                "brier": calib["brier"],
                "ece": calib["ece"],
                "report": "docs/evaluation_report.md",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
