"""Evaluation-metric correctness on hand-computed cases."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    calibration_report,
    precision_at_top_k,
    ranking_metrics,
    recall_at_top_pct,
    threshold_metrics,
)


def test_ranking_metrics_perfect_and_random() -> None:
    y = np.array([0, 0, 0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    m = ranking_metrics(y, perfect)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)


def test_recall_at_top_pct_hand_case() -> None:
    # 100 claims, 10 positives all ranked in the top 10.
    y = np.array([1] * 10 + [0] * 90)
    scores = np.linspace(1.0, 0.0, 100)
    out = recall_at_top_pct(y, scores)
    assert out["recall_at_1pct"] == pytest.approx(0.1)  # top-1 claim catches 1 of 10 positives
    assert out["recall_at_5pct"] == pytest.approx(0.5)
    assert out["recall_at_10pct"] == pytest.approx(1.0)


def test_precision_at_top_k() -> None:
    y = np.array([1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    assert precision_at_top_k(y, scores, 2) == pytest.approx(1.0)
    assert precision_at_top_k(y, scores, 4) == pytest.approx(0.5)


def test_threshold_metrics_confusion_matrix() -> None:
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.3, 0.8, 0.1])
    m = threshold_metrics(y, scores, threshold=0.5)
    cm = m["confusion_matrix"]
    assert cm == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)


def test_calibration_report_well_calibrated() -> None:
    rng = np.random.default_rng(3)
    probs = rng.uniform(0, 1, 20_000)
    y = (rng.uniform(0, 1, 20_000) < probs).astype(int)
    report = calibration_report(y, probs)
    assert report["ece"] < 0.02
    assert 0.15 < report["brier"] < 0.19  # ~E[p(1-p)] for uniform p
