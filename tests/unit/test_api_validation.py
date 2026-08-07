"""API validation + error-path tests (no model artifacts required)."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient


def _claim(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "claim_id": "CLM-99000001",
        "bene_id": "BEN-000101",
        "provider_id": "PRV-00001",
        "specialty": "Internal Medicine",
        "hcpcs": "99214",
        "modifier": "",
        "units": 1,
        "dx_group": "DXG02",
        "pos": "11",
        "service_date": "2025-09-10",
        "submit_date": "2025-09-18",
        "submitted_amount": 220.0,
        "allowed_amount": 130.0,
        "paid_amount": 104.0,
    }
    base.update(overrides)
    return base


def test_predict_without_model_returns_helpful_503(client: TestClient) -> None:
    resp = client.post("/predict", json=_claim())
    assert resp.status_code == 503
    assert "make train" in resp.json()["detail"]


def test_model_info_without_model_returns_503(client: TestClient) -> None:
    assert client.get("/model-info").status_code == 503


def test_invalid_hcpcs_rejected_with_details(client: TestClient) -> None:
    resp = client.post("/predict", json=_claim(hcpcs="99"))
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "validation_error"
    assert any("hcpcs" in d["loc"] for d in body["details"])
    assert body["request_id"].startswith("req-")


def test_amount_hierarchy_enforced(client: TestClient) -> None:
    resp = client.post("/predict", json=_claim(paid_amount=500.0))  # paid > allowed
    assert resp.status_code == 422
    assert any("paid_amount cannot exceed" in d["message"] for d in resp.json()["details"])


def test_negative_amount_rejected(client: TestClient) -> None:
    assert client.post("/predict", json=_claim(paid_amount=-5)).status_code == 422


def test_date_order_enforced(client: TestClient) -> None:
    resp = client.post("/predict", json=_claim(submit_date="2025-09-01"))
    assert resp.status_code == 422
    assert any("submit_date" in d["message"] for d in resp.json()["details"])


def test_unknown_pos_and_modifier_rejected(client: TestClient) -> None:
    assert client.post("/predict", json=_claim(pos="99")).status_code == 422
    assert client.post("/predict", json=_claim(modifier="XX")).status_code == 422


def test_units_range_enforced(client: TestClient) -> None:
    assert client.post("/predict", json=_claim(units=0)).status_code == 422
    assert client.post("/predict", json=_claim(units=9999)).status_code == 422


def test_batch_size_limit_returns_413(client: TestClient) -> None:
    claims = [_claim(claim_id=f"CLM-99{i:06d}") for i in range(1001)]
    resp = client.post("/batch-predict", json={"claims": claims})
    assert resp.status_code == 413
    assert "limit of 1000" in resp.json()["detail"]


def test_empty_batch_rejected(client: TestClient) -> None:
    assert client.post("/batch-predict", json={"claims": []}).status_code == 422


def test_metrics_endpoint_prometheus_format(client: TestClient) -> None:
    client.get("/health")
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "api_requests_total" in resp.text
    assert "model_loaded 0" in resp.text
