"""Tests for dashboard artifact readers and quality summaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from app.dashboard import common as dashboard_common
from app.dashboard.common import _read_parquet_or_empty
from app.dashboard.data_quality import discover_datasets, inspect_frame
from app.dashboard.model_performance import (
    calculate_threshold_metrics,
    load_model_performance,
    threshold_analysis,
)
from src.data.provenance import record_dataset_provenance


def test_model_performance_calculates_from_matched_demo_outcomes() -> None:
    result = load_model_performance()
    assert result["available"]
    assert result["dataset_size"] == 96_435
    assert result["evaluation_kind"].startswith("Synthetic")
    assert result["metrics"]["pr_auc"] > 0.8
    assert not result["model_comparison"].empty
    assert not result["precision_recall_curve"].empty


def test_model_performance_missing_artifacts_is_not_evaluated(tmp_path) -> None:
    result = load_model_performance(
        tmp_path / "missing_predictions.parquet",
        tmp_path / "missing_labels.parquet",
    )
    assert not result["available"]
    assert result["status"] == "Not evaluated"


def test_corrupt_or_missing_dashboard_parquet_returns_empty_frame(tmp_path) -> None:
    missing = _read_parquet_or_empty(tmp_path / "absent.parquet", "test")
    corrupt_path = tmp_path / "broken.parquet"
    corrupt_path.write_text("not parquet", encoding="utf-8")
    corrupt = _read_parquet_or_empty(corrupt_path, "test")
    assert missing.empty
    assert corrupt.empty


def test_model_performance_rejects_unmatched_score_rows(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    labels_path = tmp_path / "labels.parquet"
    pd.DataFrame({"claim_id": ["C1", "C2"], "risk_score": [0.9, 0.1]}).to_parquet(predictions_path)
    pd.DataFrame({"claim_id": ["C1"], "is_improper": [1]}).to_parquet(labels_path)
    result = load_model_performance(predictions_path, labels_path)
    assert not result["available"]
    assert "Only 1 of 2" in result["message"]


def test_model_performance_uses_matching_labels_when_scores_embed_outcomes(tmp_path) -> None:
    predictions_path = tmp_path / "scored_test.parquet"
    labels_path = tmp_path / "labels.parquet"
    pd.DataFrame(
        {
            "claim_id": ["C1", "C2"],
            "risk_score": [0.9, 0.1],
            "is_improper": [1, 0],
        }
    ).to_parquet(predictions_path)
    pd.DataFrame({"claim_id": ["C1", "C2"], "is_improper": [1, 0]}).to_parquet(labels_path)

    result = load_model_performance(predictions_path, labels_path)

    assert result["available"]
    assert result["positive_count"] == 1


def test_model_performance_rejects_conflicting_embedded_outcomes(tmp_path) -> None:
    predictions_path = tmp_path / "scored_test.parquet"
    labels_path = tmp_path / "labels.parquet"
    pd.DataFrame(
        {
            "claim_id": ["C1", "C2"],
            "risk_score": [0.9, 0.1],
            "is_improper": [0, 0],
        }
    ).to_parquet(predictions_path)
    pd.DataFrame({"claim_id": ["C1", "C2"], "is_improper": [1, 0]}).to_parquet(labels_path)

    result = load_model_performance(predictions_path, labels_path)

    assert not result["available"]
    assert "do not match" in result["message"]


def test_threshold_changes_confusion_metrics_without_changing_ground_truth() -> None:
    labels = [1, 1, 0, 0]
    original_labels = labels.copy()
    scores = [0.9, 0.4, 0.6, 0.1]
    standard = calculate_threshold_metrics(labels, scores, 0.5)
    lower_cutoff = calculate_threshold_metrics(labels, scores, 0.3)

    assert standard["true_positive"] == 1
    assert standard["false_positive"] == 1
    assert standard["false_negative"] == 1
    assert standard["true_negative"] == 1
    assert standard["false_negative_rate"] == 0.5
    assert lower_cutoff["false_negative"] == 0
    assert lower_cutoff["recall"] == 1.0
    assert labels == original_labels

    sweep = threshold_analysis(labels, scores, [0.3, 0.5, 0.7])
    assert list(sweep["threshold"]) == [0.3, 0.5, 0.7]
    assert list(sweep["false_negative"]) == [0, 1, 1]


def test_data_quality_calculations_and_missing_dataset_discovery(tmp_path) -> None:
    data = pd.DataFrame(
        [
            {"claim_id": "C1", "provider_id": "P1", "paid_amount": -1, "service_date": "bad"},
            {"claim_id": "C1", "provider_id": "P1", "paid_amount": -1, "service_date": "bad"},
        ]
    )
    inspected = inspect_frame(
        data,
        dataset_name="claims",
        source_path=tmp_path / "not_saved.parquet",
        classification="synthetic",
    )
    assert inspected["record_count"] == 2
    assert inspected["duplicate_rows"] == 1
    assert inspected["duplicate_percent"] == 50.0
    assert inspected["completeness_percent"] == 100.0
    assert inspected["invalid_values"]["invalid_paid_amount"] == 2
    assert inspected["invalid_values"]["invalid_service_date"] == 2
    assert inspected["schema_status"] == "passed"
    assert inspected["dtype_validation_status"] == "warning"
    assert set(inspected["column_completeness"].columns) >= {
        "column",
        "missing_percent",
        "completeness_percent",
    }
    unavailable = discover_datasets(tmp_path)
    assert unavailable
    assert all(not dataset["available"] for dataset in unavailable)


def test_data_quality_completeness_types_and_iqr_outliers(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "claim_id": [f"C{index}" for index in range(10)],
            "provider_id": [f"P{index}" for index in range(10)],
            "paid_amount": [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000],
            "service_date": ["2025-01-01"] * 9 + [None],
        }
    )
    result = inspect_frame(
        frame,
        dataset_name="claims",
        source_path=tmp_path / "claims.parquet",
        classification="synthetic",
    )
    assert result["completeness_percent"] == 97.5
    assert result["missing_by_column"].iloc[0]["missing_percent"] == 10.0
    assert result["dtype_validation_status"] == "passed"
    assert result["outlier_counts"]["paid_amount"] == 1


def test_provenance_writer_records_metadata_only(tmp_path) -> None:
    manifest = tmp_path / "provenance.json"
    frame = pd.DataFrame({"secret_value": ["not recorded"], "amount": [5]})
    record_dataset_provenance(
        frame,
        source_name="test source",
        source_identifier="fixture-id",
        file_name="fixture.csv",
        file_format="CSV",
        classification="synthetic",
        validation_status="passed",
        manifest_path=manifest,
    )
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    assert saved[0]["record_count"] == 1
    assert saved[0]["column_count"] == 2
    assert saved[0]["classification"] == "synthetic"
    assert "not recorded" not in manifest.read_text(encoding="utf-8")
    assert "secret_value" not in manifest.read_text(encoding="utf-8")


def test_dashboard_pages_load_without_streamlit_errors(monkeypatch) -> None:
    from app.dashboard import model_performance

    monkeypatch.setattr(
        model_performance,
        "load_model_performance",
        lambda: {
            "available": True,
            "evaluation_kind": "Synthetic test fixture",
            "model_version": "test",
            "dataset_size": 4,
            "threshold": 0.5,
            "metrics": {
                "accuracy": 0.75,
                "precision": 0.5,
                "recall": 0.5,
                "f1": 0.5,
                "roc_auc": 0.75,
                "pr_auc": 0.75,
            },
            "y_true": np.array([1, 1, 0, 0]),
            "selected_scores": np.array([0.9, 0.4, 0.6, 0.1]),
            "positive_count": 2,
            "negative_count": 2,
            "positive_rate": 0.5,
            "score_label": "Ensemble",
            "confusion_matrix": {
                "true_negative": 1,
                "false_positive": 1,
                "false_negative": 1,
                "true_positive": 1,
            },
            "model_comparison": pd.DataFrame(
                [
                    {
                        "Model": "Test model",
                        "Accuracy": 0.75,
                        "Precision": 0.5,
                        "Recall": 0.5,
                        "F1-score": 0.5,
                        "ROC-AUC": 0.75,
                        "PR-AUC": 0.75,
                    }
                ]
            ),
            "precision_recall_curve": pd.DataFrame({"recall": [0.0, 1.0], "precision": [1.0, 0.5]}),
            "roc_curve": pd.DataFrame(
                {"false_positive_rate": [0.0, 1.0], "true_positive_rate": [0.0, 1.0]}
            ),
            "feature_importance": pd.DataFrame({"feature": ["score"], "mean_abs_shap": [0.2]}),
            "feature_importance_available": True,
        },
    )
    root = Path(__file__).resolve().parents[2]
    pages = [
        root / "app" / "dashboard" / "Home.py",
        *sorted((root / "app" / "dashboard" / "pages").glob("*.py")),
    ]
    for page in pages:
        app = AppTest.from_file(str(page)).run(timeout=20)
        assert not app.exception, [str(exception.message) for exception in app.exception]


def test_scored_claim_pages_show_empty_state_without_crashing(monkeypatch) -> None:
    monkeypatch.setattr(dashboard_common, "load_predictions", lambda: pd.DataFrame())
    root = Path(__file__).resolve().parents[2]
    pages = [
        "Home.py",
        "pages/1_Review_Queue.py",
        "pages/2_Claim_Detail.py",
        "pages/3_Providers.py",
        "pages/4_Threshold_Simulator.py",
        "pages/8_Risk_Assessment.py",
    ]
    for relative in pages:
        app = AppTest.from_file(str(root / "app" / "dashboard" / relative)).run(timeout=10)
        assert not app.exception, [str(exception.message) for exception in app.exception]
