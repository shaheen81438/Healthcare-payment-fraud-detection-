"""Training: rules baseline → logistic regression → XGBoost → IsolationForest → ensemble.

All models share one temporal split (train / val / test) and one preprocessing
contract. The validation window is used for early stopping, isotonic
calibration, and ensemble weighting; the test window is touched only for final
reporting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.evaluation.metrics import ranking_metrics, recall_at_top_pct
from src.features.builder import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.modeling.bundle import ScoringBundle
from src.modeling.preprocess import make_preprocessor
from src.modeling.rules import RuleEngine

logger = logging.getLogger(__name__)

RULE_FEATURES = ["rule_score", "n_rules_hit"]

#: Behavioral numeric subset for the unsupervised anomaly detector.
ISO_FEATURES = [
    "paid_amount",
    "paid_per_unit",
    "paid_vs_code_median",
    "paid_vs_spec_code_median",
    "units",
    "units_vs_code_p95",
    "dup_max_sim",
    "dup_cand_count",
    "bene_claims_30d",
    "bene_paid_30d",
    "bene_provider_same_day_claims",
    "days_since_bene_same_hcpcs",
    "prov_velocity_1v3",
    "prov_claims_per_bene_3m",
    "prov_top_code_share_3m",
    "prov_paid_vs_peer_3m",
    "prov_new_bene_share_1m",
    "pos_atypicality",
    "dxproc_compat",
    "em_gap_vs_specialty",
]


def load_model_config(path: str = "configs/model.yaml") -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)
    return cfg


@dataclass
class TrainResult:
    bundle: ScoringBundle
    metrics: dict[str, Any]
    rule_hit_counts: dict[str, int]
    lr_model: Any = None
    extras: dict[str, Any] = field(default_factory=dict)


def prepare_training_frame(
    features: pd.DataFrame, labels: pd.DataFrame, include_rules: bool = True
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """Join features+labels, evaluate rules, return (frame, numeric feature list, rule hits)."""
    frame = features.merge(labels, on="claim_id", how="inner", validate="one_to_one")
    numeric = list(NUMERIC_FEATURES)
    hit_counts: dict[str, int] = {}
    if include_rules:
        engine = RuleEngine.from_yaml()
        evaluated = engine.evaluate(frame)
        frame = frame.merge(
            evaluated[["claim_id", "rule_score", "n_rules_hit", "rule_hits", "rule_reasons"]],
            on="claim_id",
        )
        hit_counts = engine.hit_counts(evaluated)
        numeric += RULE_FEATURES
    return frame, numeric, hit_counts


def _split(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {name: frame[frame["split"] == name] for name in ["train", "val", "test"]}


def train_all(
    frame: pd.DataFrame,
    numeric: list[str],
    cfg: dict[str, Any],
    feature_meta: dict[str, Any] | None = None,
) -> TrainResult:
    """Train every model family and assemble the scoring bundle."""
    parts = _split(frame)
    cats = list(CATEGORICAL_FEATURES)
    X = {k: v[numeric + cats] for k, v in parts.items()}
    y = {k: v["is_improper"].to_numpy(dtype=int) for k, v in parts.items()}
    metrics: dict[str, Any] = {"n": {k: len(v) for k, v in parts.items()}}

    # ---------------- rules baseline
    for split_name in ["val", "test"]:
        scores = parts[split_name]["rule_score"].to_numpy()
        metrics.setdefault("rules_baseline", {})[split_name] = {
            **ranking_metrics(y[split_name], scores),
            **recall_at_top_pct(y[split_name], scores),
        }

    # ---------------- logistic regression baseline
    pre_lr = make_preprocessor(numeric, cats, scale_numeric=True)
    Xt_lr_train = pre_lr.fit_transform(X["train"])
    lr_cfg = cfg["logistic_regression"]
    lr = LogisticRegression(
        C=float(lr_cfg["C"]),
        max_iter=int(lr_cfg["max_iter"]),
        class_weight=lr_cfg["class_weight"],
    )
    lr.fit(Xt_lr_train, y["train"])
    for split_name in ["val", "test"]:
        p = lr.predict_proba(pre_lr.transform(X[split_name]))[:, 1]
        metrics.setdefault("logistic_regression", {})[split_name] = {
            **ranking_metrics(y[split_name], p),
            **recall_at_top_pct(y[split_name], p),
        }

    # ---------------- XGBoost (primary supervised model)
    pre_tree = make_preprocessor(numeric, cats, scale_numeric=False)
    Xt_train = pre_tree.fit_transform(X["train"])
    Xt_val = pre_tree.transform(X["val"])
    xgb_cfg = dict(cfg["xgboost"])
    early = int(xgb_cfg.pop("early_stopping_rounds"))
    pos = max(1, int(y["train"].sum()))
    scale_pos_weight = float((len(y["train"]) - pos) / pos)
    xgb = XGBClassifier(
        **xgb_cfg,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=early,
        n_jobs=-1,
        random_state=42,
    )
    xgb.fit(Xt_train, y["train"], eval_set=[(Xt_val, y["val"])], verbose=False)
    raw = {s: xgb.predict_proba(pre_tree.transform(X[s]))[:, 1] for s in ["train", "val", "test"]}
    for split_name in ["val", "test"]:
        metrics.setdefault("xgboost", {})[split_name] = {
            **ranking_metrics(y[split_name], raw[split_name]),
            **recall_at_top_pct(y[split_name], raw[split_name]),
        }
    metrics["xgboost"]["best_iteration"] = int(getattr(xgb, "best_iteration", 0) or 0)
    metrics["xgboost"]["scale_pos_weight"] = round(scale_pos_weight, 2)

    # ---------------- calibration (isotonic on validation)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw["val"], y["val"])
    calibrated = {s: np.clip(calibrator.predict(raw[s]), 0.0, 1.0) for s in raw}

    # ---------------- Isolation Forest (unsupervised coverage)
    iso_cfg = cfg["isolation_forest"]
    iso = IsolationForest(
        n_estimators=int(iso_cfg["n_estimators"]),
        max_samples=float(iso_cfg["max_samples"]),
        contamination=iso_cfg["contamination"],
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X["train"][ISO_FEATURES].fillna(0.0))
    anomaly_train = -iso.score_samples(X["train"][ISO_FEATURES].fillna(0.0))
    ref = np.sort(anomaly_train)
    anomaly_pct = {
        s: np.searchsorted(ref, -iso.score_samples(X[s][ISO_FEATURES].fillna(0.0)), side="right")
        / len(ref)
        for s in ["val", "test"]
    }
    for split_name in ["val", "test"]:
        metrics.setdefault("isolation_forest", {})[split_name] = {
            **ranking_metrics(y[split_name], anomaly_pct[split_name]),
            **recall_at_top_pct(y[split_name], anomaly_pct[split_name]),
        }

    # ---------------- ensemble weight selection on validation
    best_w, best_score = 1.0, -1.0
    for w in cfg["ensemble"]["weight_grid"]:
        blended = w * calibrated["val"] + (1 - w) * anomaly_pct["val"]
        score = ranking_metrics(y["val"], blended)[cfg["ensemble"]["selection_metric"]]
        if score > best_score:
            best_w, best_score = float(w), float(score)
    for split_name in ["val", "test"]:
        blended = best_w * calibrated[split_name] + (1 - best_w) * anomaly_pct[split_name]
        metrics.setdefault("ensemble", {})[split_name] = {
            **ranking_metrics(y[split_name], blended),
            **recall_at_top_pct(y[split_name], blended),
        }
    metrics["ensemble"]["weight"] = best_w

    bundle = ScoringBundle(
        model_version=str(cfg["model_version"]),
        preprocessor=pre_tree,
        xgb_model=xgb,
        calibrator=calibrator,
        iso_forest=iso,
        iso_features=list(ISO_FEATURES),
        anomaly_ref_scores=ref,
        ensemble_weight=best_w,
        numeric_features=list(numeric),
        categorical_features=cats,
        feature_names_out=list(pre_tree.get_feature_names_out()),
        metadata={
            "feature_meta": feature_meta or {},
            "train_rows": len(parts["train"]),
            "positive_rate_train": float(y["train"].mean()),
        },
        shap_background=X["train"].sample(min(500, len(X["train"])), random_state=42),
    )
    logger.info(
        "training_complete",
        extra={
            "ensemble_weight": best_w,
            "val_pr_auc": metrics["ensemble"]["val"]["pr_auc"],
            "test_pr_auc": metrics["ensemble"]["test"]["pr_auc"],
        },
    )
    return TrainResult(bundle=bundle, metrics=metrics, rule_hit_counts={}, lr_model=lr)
