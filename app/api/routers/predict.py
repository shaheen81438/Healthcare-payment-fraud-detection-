"""Prediction, batch prediction, and explanation endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from app.schemas.claims import (
    BatchPredictIn,
    BatchPredictOut,
    ClaimIn,
    ErrorOut,
    ExplainOut,
    PredictionOut,
)
from app.services.metrics_store import metrics_store
from app.services.scoring_service import ScoringService, get_service
from src.utils.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["scoring"])

RESPONSES: dict[int | str, dict[str, Any]] = {
    503: {"model": ErrorOut, "description": "Model not loaded"},
    413: {"model": ErrorOut, "description": "Batch exceeds the configured limit"},
}


def _service_or_503() -> ScoringService:
    service = get_service()
    if service is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No scoring model is loaded. Train one with `make train` "
                "(or restore models/latest.json), then restart the API."
            ),
        )
    return service


@router.post("/predict", response_model=PredictionOut, responses=RESPONSES)
def predict(claim: ClaimIn) -> PredictionOut:
    """Score a single claim: risk, recommended action, payment at risk, reasons."""
    service = _service_or_503()
    result = service.predict([claim])[0]
    metrics_store.observe_predictions([result.recommended_action])
    return result


@router.post("/batch-predict", response_model=BatchPredictOut, responses=RESPONSES)
def batch_predict(batch: BatchPredictIn) -> BatchPredictOut:
    """Score up to BATCH_PREDICT_LIMIT claims in one call (order preserved)."""
    settings = get_settings()
    if len(batch.claims) > settings.batch_predict_limit:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Batch of {len(batch.claims)} exceeds the limit of "
                f"{settings.batch_predict_limit}. Split the request or raise "
                "BATCH_PREDICT_LIMIT."
            ),
        )
    service = _service_or_503()
    results = service.predict(batch.claims)
    metrics_store.observe_predictions([r.recommended_action for r in results])
    return BatchPredictOut(
        count=len(results), model_version=service.bundle.model_version, results=results
    )


@router.post("/explain", response_model=ExplainOut, responses=RESPONSES)
def explain(claim: ClaimIn) -> ExplainOut:
    """Score a claim and also return SHAP contributions and labeled reason codes."""
    service = _service_or_503()
    result = service.explain([claim])[0]
    metrics_store.observe_predictions([result.recommended_action])
    return result
