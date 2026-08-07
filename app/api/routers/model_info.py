"""Model metadata + operational metrics endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from app.schemas.claims import ErrorOut, ModelInfoOut
from app.services.metrics_store import metrics_store
from app.services.scoring_service import get_service

router = APIRouter(tags=["system"])


@router.get(
    "/model-info",
    response_model=ModelInfoOut,
    responses={503: {"model": ErrorOut, "description": "Model not loaded"}},
)
def model_info() -> ModelInfoOut:
    """Version, provenance, feature counts, threshold policy, headline metrics."""
    service = get_service()
    if service is None:
        raise HTTPException(status_code=503, detail="No scoring model is loaded.")
    return service.model_info()


@router.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    """Prometheus text-format operational metrics."""
    service = get_service()
    return metrics_store.render(
        model_version=service.bundle.model_version if service else "none",
        model_loaded=service is not None,
    )
