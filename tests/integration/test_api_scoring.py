"""End-to-end API tests against the real trained bundle + feature stores.

Requires artifacts from `make data`, `make features`, `make train`, `make evaluate`
(skipped automatically when absent). Marked `integration`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

REQUIRED_ARTIFACTS = [
    "models/latest.json",
    "models/feature_store/code_stats.parquet",
    "data/processed/claims.parquet",
    "data/processed/provider_month_features.parquet",
]


@pytest.fixture(scope="module")
def live_client() -> TestClient:
    missing = [p for p in REQUIRED_ARTIFACTS if not Path(p).exists()]
    if missing:
        pytest.skip(f"artifacts missing: {missing}")
    import os

    os.environ["SKIP_MODEL_LOAD"] = "1"  # we initialize explicitly below
    from app.api.main import create_app
    from app.services.scoring_service import initialize

    app = create_app()
    service = initialize()
    assert service is not None, "scoring service failed to initialize"
    return TestClient(app)


def _claim(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "claim_id": "CLM-API-TEST-01",
        "bene_id": "BEN-000101",
        "provider_id": "PRV-00001",
        "specialty": "Internal Medicine",
        "hcpcs": "99214",
        "modifier": "",
        "units": 1,
        "dx_group": "DXG02",
        "pos": "11",
        "service_date": "2025-11-10",
        "submit_date": "2025-11-18",
        "submitted_amount": 220.0,
        "allowed_amount": 130.0,
        "paid_amount": 104.0,
    }
    base.update(overrides)
    return base


def test_health_reports_model_loaded(live_client: TestClient) -> None:
    body = live_client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_contract(live_client: TestClient) -> None:
    resp = live_client.post("/predict", json=_claim())
    assert resp.status_code == 200
    body = resp.json()
    assert body["claim_id"] == "CLM-API-TEST-01"
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["recommended_action"] in {"manual_review", "monitor", "auto_clear"}
    assert body["estimated_payment_at_risk"] >= 0
    assert isinstance(body["reason_codes"], list)
    assert body["model_version"].startswith("claims-risk")
    assert resp.headers["X-Request-ID"]


def test_batch_predict_preserves_order(live_client: TestClient) -> None:
    claims = [
        _claim(claim_id="CLM-API-B-01", paid_amount=104.0),
        _claim(claim_id="CLM-API-B-02", hcpcs="99215", paid_amount=125.0, allowed_amount=156.0),
        _claim(claim_id="CLM-API-B-03", provider_id="PRV-01999"),
    ]
    resp = live_client.post("/batch-predict", json={"claims": claims})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 3
    assert [r["claim_id"] for r in body["results"]] == [c["claim_id"] for c in claims]


def test_duplicate_of_history_fires_dup_signals_and_raises_risk(live_client: TestClient) -> None:
    """Copy a real historical claim under a new ID → duplicate signals must fire,
    and risk must exceed the identical claim WITHOUT duplicate context.

    (The absolute score stays moderate by design: an exact resubmission at an
    otherwise-clean provider resembles the benign corrected-claim confounder.)
    """
    import pandas as pd

    hist = pd.read_parquet("data/processed/claims.parquet").iloc[-500]
    fields: dict[str, Any] = dict(  # noqa: C408 - kwargs style mirrors the payload builder
        provider_id=str(hist["provider_id"]),
        specialty=str(hist["specialty"]),
        hcpcs=str(hist["hcpcs"]),
        modifier="" if pd.isna(hist["modifier"]) else str(hist["modifier"]),
        units=int(hist["units"]),
        dx_group=str(hist["dx_group"]),
        pos=str(hist["pos"]),
        service_date=str(pd.Timestamp(hist["service_date"]).date()),
        submit_date=str((pd.Timestamp(hist["service_date"]) + pd.Timedelta(days=12)).date()),
        submitted_amount=float(hist["submitted_amount"]),
        allowed_amount=float(hist["allowed_amount"]),
        paid_amount=float(hist["paid_amount"]),
    )
    dup = _claim(claim_id="CLM-API-DUP-01", bene_id=str(hist["bene_id"]), **fields)
    fresh = _claim(claim_id="CLM-API-DUP-02", bene_id="BEN-NOHISTORY-9", **fields)

    dup_body = live_client.post("/explain", json=dup).json()
    fresh_body = live_client.post("/explain", json=fresh).json()

    assert "R001_exact_duplicate" in dup_body["rule_hits"]
    assert any("duplicate" in code for code in dup_body["reason_codes"])
    assert "R001_exact_duplicate" not in fresh_body["rule_hits"]
    assert dup_body["risk_score"] > fresh_body["risk_score"]


def test_explain_returns_shap_contributions(live_client: TestClient) -> None:
    resp = live_client.post("/explain", json=_claim(claim_id="CLM-API-EXP-01"))
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["shap_contributions"]) >= 3
    top = body["shap_contributions"][0]
    assert {"feature", "contribution", "value"} <= set(top)
    assert isinstance(body["base_value"], float)
    assert set(body["reason_labels"]) == set(body["reason_codes"])


def test_model_info_contract(live_client: TestClient) -> None:
    body = live_client.get("/model-info").json()
    assert body["model_version"].startswith("claims-risk")
    assert body["source"] in {"mlflow_registry", "local_bundle"}
    assert body["numeric_features"] >= 40
    assert body["threshold_policy"]["review_threshold"] > 0
    assert body["ensemble_weight"] > 0.5


def test_metrics_counts_predictions(live_client: TestClient) -> None:
    text = live_client.get("/metrics").text
    assert "claims_scored_total" in text
    assert "model_loaded 1" in text
