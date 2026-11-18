"""Artifact-backed evaluation calculations for the dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

SCORE_COLUMNS = {
    "risk_score": "Ensemble",
    "supervised_probability": "Supervised model",
    "anomaly_percentile": "Isolation Forest",
    "rule_score": "Rules baseline",
}


def calculate_threshold_metrics(
    y_true: np.ndarray | list[int],
    scores: np.ndarray | list[float],
    threshold: float,
) -> dict[str, float | int]:
    """Calculate binary metrics for one cutoff without changing outcomes or scores."""
    try:
        raw_labels = np.asarray(y_true, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Metrics require binary ground-truth labels") from exc
    values = np.asarray(scores, dtype=float)
    if raw_labels.ndim != 1 or values.ndim != 1 or len(raw_labels) != len(values):
        raise ValueError("Ground-truth labels and scores must be one-dimensional and equally sized")
    if (
        not len(raw_labels)
        or not np.isfinite(raw_labels).all()
        or not np.isfinite(values).all()
        or not np.isin(raw_labels, [0, 1]).all()
    ):
        raise ValueError("Metrics require finite scores and binary ground-truth labels")
    labels = raw_labels.astype(int)
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    predicted = (values >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    recall = float(recall_score(labels, predicted, zero_division=0))
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": recall,
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "review_rate": float(predicted.mean()),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def threshold_analysis(
    y_true: np.ndarray | list[int],
    scores: np.ndarray | list[float],
    thresholds: list[float] | np.ndarray,
) -> pd.DataFrame:
    """Compare the same evaluation rows over caller-selected score cutoffs."""
    return pd.DataFrame(
        [calculate_threshold_metrics(y_true, scores, float(threshold)) for threshold in thresholds]
    )


def _read_policy(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


@st.cache_data(ttl=600, show_spinner="Loading evaluation results…")
def load_model_performance(
    predictions_path: str | Path | None = None,
    labels_path: str | Path | None = None,
    *,
    importance_path: str | Path = "docs/assets/global_importance.csv",
    policy_path: str | Path | None = None,
) -> dict[str, Any]:
    """Calculate metrics only when scored rows and their synthetic outcomes exist."""
    predictions_file = (
        Path(predictions_path) if predictions_path else Path("data/processed/scored_test.parquet")
    )
    labels_file = Path(labels_path) if labels_path else Path("data/processed/labels.parquet")
    if predictions_path is None and not predictions_file.is_file():
        predictions_file = Path("demo_data/predictions.parquet")
    if labels_path is None and not labels_file.is_file():
        labels_file = Path("demo_data/labels.parquet")
    missing = [str(path) for path in (predictions_file, labels_file) if not path.is_file()]
    if missing:
        return {
            "available": False,
            "status": "Not evaluated",
            "message": "Evaluation scores or matching labels are unavailable.",
            "missing_artifacts": missing,
        }
    try:
        predictions = pd.read_parquet(predictions_file)
        labels = pd.read_parquet(labels_file)
    except Exception as exc:
        return {"available": False, "status": "Not evaluated", "message": str(exc)}
    if not {"claim_id", "is_improper"} <= set(labels.columns) or "claim_id" not in predictions:
        return {
            "available": False,
            "status": "Not evaluated",
            "message": "Evaluation artifacts do not contain claim IDs and outcome labels.",
        }
    if labels["claim_id"].duplicated().any() or predictions["claim_id"].duplicated().any():
        return {
            "available": False,
            "status": "Not evaluated",
            "message": "Duplicate claim IDs prevent a one-to-one evaluation join.",
        }
    evaluation_labels = labels[["claim_id", "is_improper"]].rename(
        columns={"is_improper": "ground_truth_label"}
    )
    frame = predictions.merge(evaluation_labels, on="claim_id", how="inner", validate="one_to_one")
    if frame.empty:
        return {
            "available": False,
            "status": "Not evaluated",
            "message": "No scored claims match the available outcome labels.",
        }
    if len(frame) != len(predictions):
        return {
            "available": False,
            "status": "Not evaluated",
            "message": (
                f"Only {len(frame):,} of {len(predictions):,} scored claims have matching "
                "ground-truth labels."
            ),
        }
    y = pd.to_numeric(frame["ground_truth_label"], errors="coerce")
    valid = y.isin([0, 1])
    if not valid.all():
        return {
            "available": False,
            "status": "Not evaluated",
            "message": "Ground-truth labels contain missing or non-binary outcomes.",
        }
    if "is_improper" in frame:
        embedded_labels = pd.to_numeric(frame["is_improper"], errors="coerce")
        if not embedded_labels.isin([0, 1]).all() or not embedded_labels.equals(y):
            return {
                "available": False,
                "status": "Not evaluated",
                "message": "Embedded and separate ground-truth labels do not match.",
            }
    frame, y = frame.reset_index(drop=True), y.astype(int).to_numpy()
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {
            "available": False,
            "status": "Not evaluated",
            "message": "Evaluation needs labeled examples from both outcome classes.",
        }

    comparison: list[dict[str, Any]] = []
    available_scores: dict[str, np.ndarray] = {}
    for column, label in SCORE_COLUMNS.items():
        if column not in frame:
            continue
        scores = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(scores)
        if not finite.all() or finite.sum() != len(y):
            continue
        available_scores[column] = scores
        predicted = scores >= 0.5
        comparison.append(
            {
                "Model": label,
                "Accuracy": float(accuracy_score(y, predicted)),
                "Precision": float(precision_score(y, predicted, zero_division=0)),
                "Recall": float(recall_score(y, predicted, zero_division=0)),
                "F1-score": float(f1_score(y, predicted, zero_division=0)),
                "ROC-AUC": float(roc_auc_score(y, scores)),
                "PR-AUC": float(average_precision_score(y, scores)),
            }
        )
    if not available_scores:
        return {
            "available": False,
            "status": "Not evaluated",
            "message": "No supported numeric model score columns are available.",
        }

    selected = "risk_score" if "risk_score" in available_scores else next(iter(available_scores))
    version = (
        str(frame["model_version"].dropna().iloc[0])
        if "model_version" in frame and frame["model_version"].notna().any()
        else "Unknown"
    )
    if policy_path is not None:
        policy = _read_policy(Path(policy_path))
    else:
        model_metadata = Path("models") / version / "metadata.json"
        policy = _read_policy(model_metadata).get("threshold_policy", {})
        if not policy and predictions_file.parts[:1] == ("demo_data",):
            policy = _read_policy(Path("demo_data/policy.json"))
    threshold = float(policy.get("review_threshold", 0.5)) if selected == "risk_score" else 0.5
    selected_scores = available_scores[selected]
    operating_metrics = calculate_threshold_metrics(y, selected_scores, threshold)
    precision, recall, _ = precision_recall_curve(y, selected_scores)
    fpr, tpr, _ = roc_curve(y, selected_scores)
    precision_recall = pd.DataFrame({"recall": recall[:-1], "precision": precision[:-1]})
    roc_data = pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr})

    importance_file = Path(importance_path)
    importance = pd.DataFrame()
    if importance_file.is_file():
        try:
            candidate = pd.read_csv(importance_file)
            if {"feature", "mean_abs_shap"} <= set(candidate.columns):
                importance = candidate.sort_values("mean_abs_shap", ascending=False).head(20)
        except (OSError, ValueError, pd.errors.ParserError):
            pass

    return {
        "available": True,
        "status": "Evaluated on synthetic claim outcomes",
        "evaluation_kind": "Synthetic claim evaluation; not public CMS outcomes",
        "model_version": version,
        "dataset_size": len(y),
        "positive_count": int(y.sum()),
        "negative_count": int(len(y) - y.sum()),
        "positive_rate": float(y.mean()),
        "score_label": SCORE_COLUMNS[selected],
        "threshold": threshold,
        "metrics": {
            "accuracy": operating_metrics["accuracy"],
            "precision": operating_metrics["precision"],
            "recall": operating_metrics["recall"],
            "f1": operating_metrics["f1"],
            "roc_auc": float(roc_auc_score(y, selected_scores)),
            "pr_auc": float(average_precision_score(y, selected_scores)),
            "false_negative_rate": operating_metrics["false_negative_rate"],
        },
        "confusion_matrix": {
            "true_negative": operating_metrics["true_negative"],
            "false_positive": operating_metrics["false_positive"],
            "false_negative": operating_metrics["false_negative"],
            "true_positive": operating_metrics["true_positive"],
        },
        "y_true": y,
        "selected_scores": selected_scores,
        "model_comparison": pd.DataFrame(comparison),
        "precision_recall_curve": precision_recall,
        "roc_curve": roc_data,
        "feature_importance": importance,
        "feature_importance_available": not importance.empty,
    }
