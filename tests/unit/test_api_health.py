"""Health endpoint and request-ID middleware tests."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "healthcare-payment-integrity-api"
    assert body["environment"] == "dev"
    assert isinstance(body["model_loaded"], bool)
    assert body["version"]
    assert body["timestamp"]


def test_request_id_generated(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.headers["X-Request-ID"].startswith("req-")


def test_request_id_echoed_when_provided(client: TestClient) -> None:
    resp = client.get("/health", headers={"X-Request-ID": "req-from-gateway-42"})
    assert resp.headers["X-Request-ID"] == "req-from-gateway-42"


def test_openapi_docs_available(client: TestClient) -> None:
    assert client.get("/openapi.json").status_code == 200
