"""Evaluation metrics for imbalanced ranking problems.

Accuracy is deliberately absent: with ~4–5% positives it is uninformative.
Primary metrics are PR-AUC and capacity-style ranking metrics (recall@top-k%),
because investigators review a *budgeted queue*, not a threshold in the
abstract.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def ranking_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Threshold-free metrics: PR-AUC (primary) and ROC-AUC (secondary)."""
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }


def threshold_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    """Precision / recall / F1 / confusion matrix at a fixed score threshold."""
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def recall_at_top_pct(
    y_true: np.ndarray, scores: np.ndarray, pcts: tuple[float, ...] = (0.01, 0.05, 0.10)
) -> dict[str, float]:
    """Share of all positives captured when reviewing only the top X% of claims."""
    order = np.argsort(-scores, kind="stable")
    y_sorted = np.asarray(y_true)[order]
    total_pos = max(1, int(y_sorted.sum()))
    out: dict[str, float] = {}
    for pct in pcts:
        k = max(1, int(len(y_sorted) * pct))
        out[f"recall_at_{int(pct * 100)}pct"] = float(y_sorted[:k].sum() / total_pos)
    return out


def precision_at_top_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Precision within a review budget of k claims."""
    k = min(max(1, k), len(scores))
    order = np.argsort(-scores, kind="stable")[:k]
    return float(np.asarray(y_true)[order].mean())


def calibration_report(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    """Brier score + reliability table (equal-width probability bins)."""
    bins = np.clip((probs * n_bins).astype(int), 0, n_bins - 1)
    table = (
        pd.DataFrame({"bin": bins, "prob": probs, "actual": y_true})
        .groupby("bin")
        .agg(
            mean_predicted=("prob", "mean"), observed_rate=("actual", "mean"), n=("actual", "size")
        )
        .reset_index()
    )
    # Expected calibration error, weighted by bin size.
    ece = float(
        ((table["mean_predicted"] - table["observed_rate"]).abs() * table["n"]).sum()
        / max(1, table["n"].sum())
    )
    return {
        "brier": float(brier_score_loss(y_true, probs)),
        "ece": round(ece, 5),
        "reliability": table.round(5).to_dict(orient="records"),
    }


def full_evaluation(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float | None = None,
    probs: np.ndarray | None = None,
) -> dict[str, Any]:
    """Convenience wrapper combining ranking, capacity, threshold, and calibration metrics."""
    out: dict[str, Any] = {}
    out.update(ranking_metrics(y_true, scores))
    out.update(recall_at_top_pct(y_true, scores))
    if threshold is not None:
        out.update(threshold_metrics(y_true, scores, threshold))
    if probs is not None:
        out["calibration"] = calibration_report(y_true, probs)
    return out
