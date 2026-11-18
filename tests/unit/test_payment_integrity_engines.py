"""Focused tests for the independent payment-integrity engines."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.provenance import file_provenance
from src.payment_integrity.improper_payment import analyze_payments, evaluate_payment_row
from src.payment_integrity.leie_screening import normalize_leie, screen_provider
from src.payment_integrity.provider_anomaly import detect_provider_anomalies
from src.payment_integrity.risk_assessment import aggregate_risk


def test_payment_rules_separate_overpayment_underpayment_and_documentation() -> None:
    result = evaluate_payment_row(
        {
            "submitted_amount": 100,
            "allowed_amount": 80,
            "paid_amount": 90,
            "expected_payment_amount": 95,
            "documentation_status": "insufficient",
        }
    )
    assert result["has_potential_overpayment"]
    assert result["has_potential_underpayment"]
    assert {"PAY_PAID_GT_ALLOWED", "PAY_PAID_LT_EXPECTED", "DOC_INSUFFICIENT"} <= set(
        result["reason_codes"]
    )


def test_payment_analysis_preserves_input_and_handles_missing_optional_fields() -> None:
    claims = pd.DataFrame([{"claim_id": "C1", "paid_amount": 10}])
    output = analyze_payments(claims)
    assert list(output["claim_id"]) == ["C1"]
    assert output.loc[0, "payment_issue_count"] == 0


def test_provider_anomalies_are_specialty_aware_and_explainable() -> None:
    rows = []
    for provider, specialty, volume in [
        ("1", "Cardiology", 100),
        ("2", "Cardiology", 110),
        ("3", "Cardiology", 1000),
        ("4", "Dermatology", 1000),
        ("5", "Dermatology", 1050),
    ]:
        rows.append(
            {
                "npi": provider,
                "provider_type": specialty,
                "total_services": volume,
                "avg_paid": 50,
                "avg_allowed": 60,
                "avg_submitted": 70,
                "hcpcs": "99213",
            }
        )
    output = detect_provider_anomalies(pd.DataFrame(rows))
    assert {
        "provider_id",
        "peer_group",
        "anomaly_score",
        "triggered_indicators",
        "explanation",
    } <= set(output.columns)
    assert output.loc[output["provider_id"] == "3", "peer_group"].item() == "Cardiology"
    assert output["anomaly_score"].between(0, 1).all()


def test_provider_anomaly_reports_missing_dataset_fields() -> None:
    with pytest.raises(ValueError, match="provider-service data missing required columns"):
        detect_provider_anomalies(pd.DataFrame())


def test_leie_prefers_npi_and_never_returns_ssn() -> None:
    leie = pd.DataFrame(
        [
            {
                "npi": "1234567890",
                "provider_name": "Ada Lovelace",
                "exclusion_date": "20240101",
                "ssn": "secret",
            }
        ]
    )
    matches = screen_provider(
        {"provider_id": "P1", "npi": "1234567890", "provider_name": "Wrong Name"}, leie
    )
    assert len(matches) == 1
    assert matches[0]["match_type"] == "npi"
    assert "ssn" not in str(matches[0]).lower()


def test_leie_loader_fields_include_reinstatement_and_drop_ssn() -> None:
    normalized = normalize_leie(
        pd.DataFrame(
            [
                {
                    "NPI": "1234567890",
                    "FIRSTNAME": "Ada",
                    "LASTNAME": "Lovelace",
                    "EXCLDATE": "20240101",
                    "REINDATE": "20250101",
                    "SSN": "secret",
                }
            ]
        )
    )
    match = screen_provider({"npi": "1234567890"}, normalized)[0]
    assert match["exclusion_date"].isoformat() == "2024-01-01"
    assert match["reinstatement_date"].isoformat() == "2025-01-01"
    assert "ssn" not in str(normalized.columns).lower()
    assert "Ada" in str(normalized.iloc[0]["first_name"])


def test_risk_aggregation_marks_missing_components_without_zero_substitution() -> None:
    result = aggregate_risk(improper_payment_score=0.8, provider_anomaly_score=None)
    assert result["overall_investigation_priority"] == 0.8
    assert "provider_anomaly signal unavailable" in result["data_quality_warnings"]
    assert result["components"]["provider_anomaly"]["score"] is None


def test_file_provenance_records_hash_and_rejects_missing_source(tmp_path) -> None:
    source = tmp_path / "sample.csv"
    source.write_bytes(b"id,value\n1,2\n")
    record = file_provenance(source, source_name="test", data_kind="synthetic")
    assert record["file_name"] == "sample.csv"
    assert record["sha256"] == "c15097c46e6dbfe67d8106c18b850ff4312898c146771669ae50324f910ed93c"
    assert record["file_size_bytes"] == source.stat().st_size
    with pytest.raises(FileNotFoundError, match="Data source file not found"):
        file_provenance(tmp_path / "missing.csv", source_name="test", data_kind="synthetic")
