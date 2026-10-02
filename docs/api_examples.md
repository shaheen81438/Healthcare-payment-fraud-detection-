# API usage examples

Interactive docs at `http://localhost:8000/docs`. All responses carry an `X-Request-ID` header
(echoed from the request when supplied) that matches the structured server logs.

## Score one claim: `POST /predict`

```bash
curl -s localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "claim_id": "CLM-102849", "bene_id": "BEN-004521", "provider_id": "PRV-00042",
  "specialty": "Internal Medicine", "hcpcs": "99215", "modifier": "", "units": 1,
  "dx_group": "DXG02", "pos": "11",
  "service_date": "2025-11-10", "submit_date": "2025-11-18",
  "submitted_amount": 420.0, "allowed_amount": 210.0, "paid_amount": 168.0
}'
```

```json
{
  "claim_id": "CLM-102849",
  "risk_score": 0.2263,
  "recommended_action": "auto_clear",
  "estimated_payment_at_risk": 32.94,
  "reason_codes": ["potential_upcoding_pattern"],
  "supervised_probability": 0.1961,
  "anomaly_percentile": 0.8007,
  "rule_hits": ["R010_upcoding_indicator"],
  "insufficient_provider_history": false,
  "model_version": "claims-risk-v1.0.0",
  "scored_at": "2026-08-07T06:53:27.102760Z"
}
```

A level-5 E/M paid at ~2× the specialty norm trips the upcoding rule. The calibrated model still
reads the overall claim as low risk, and the rule signal appears in `reason_codes` either way.

## Explain a claim: `POST /explain`

Same payload as `/predict`; adds signed SHAP contributions and labeled reasons:

```json
{
  "risk_score": 0.2263,
  "reason_codes": ["potential_upcoding_pattern", "provider_above_peer_group", "amount_above_peer_norm"],
  "base_value": 0.13273,
  "shap_contributions": [
    {"feature": "prov_paid_vs_peer_3m", "contribution": 1.08728, "value": 2.5691},
    {"feature": "submitted_amount",     "contribution": -0.91036, "value": 420.0},
    {"feature": "specialty",            "contribution": -0.86349, "value": "Internal Medicine"},
    {"feature": "paid_amount",          "contribution": 0.83596,  "value": 168.0}
  ],
  "reason_labels": {
    "potential_upcoding_pattern": "Provider pattern suggests possible upcoding",
    "provider_above_peer_group": "Provider billing above peer group",
    "amount_above_peer_norm": "High paid amount"
  }
}
```

## Batch scoring: `POST /batch-predict`

```bash
curl -s localhost:8000/batch-predict -H 'Content-Type: application/json' \
  -d '{"claims": [ {...}, {...} ]}'          # up to BATCH_PREDICT_LIMIT (default 1000)
```

Returns `{count, model_version, results: [PredictionOut…]}` in request order. Oversized batches
get a 413 that states the limit; an empty list gets a 422.

## Validation errors: `422`

```json
{
  "error": "validation_error",
  "detail": "One or more fields failed validation. See `details`.",
  "details": [{"loc": "body", "message": "Value error, paid_amount cannot exceed allowed_amount"}],
  "request_id": "req-f9093fdff2f94b44"
}
```

The API checks HCPCS format, POS/modifier enums, units 1-200, amounts ≥0 with
`paid ≤ allowed ≤ submitted`, and sane date ordering and windows.

## Model metadata: `GET /model-info`

```json
{
  "model_version": "claims-risk-v1.0.0",
  "source": "mlflow_registry",
  "dataset_version": "ds-7996bf5a94f9",
  "trained_rows": 187571,
  "numeric_features": 45,
  "categorical_features": 3,
  "ensemble_weight": 0.95,
  "threshold_policy": {
    "review_threshold": 0.9945,
    "monitor_threshold": 0.4973,
    "capacity_share": 0.01,
    "cost_params": {"review_cost": 30.0, "recovery_rate": 0.62,
                    "fp_extra_cost": 12.0, "min_risk_threshold": 0.2}
  }
}
```

## Python client sketch

```python
import requests

API = "http://localhost:8000"
resp = requests.post(f"{API}/predict", json=claim_payload, timeout=10)
resp.raise_for_status()
result = resp.json()
if result["recommended_action"] == "manual_review":
    queue_for_investigation(
        result["claim_id"], result["reason_codes"], result["estimated_payment_at_risk"]
    )
```

## Operational metrics: `GET /metrics` (Prometheus text)

`api_requests_total{method,path,status}`, `api_request_latency_ms_sum/count{path}`,
`claims_scored_total`, `predictions_by_action_total{action}`, `model_loaded`,
`model_info{version}`.
