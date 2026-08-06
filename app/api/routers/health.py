"""Liveness/readiness endpoint."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter
from pydantic import BaseModel

from src.utils.config import get_settings
from src.utils.version import get_version

router = APIRouter(tags=["system"])


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str
    model_loaded: bool
    timestamp: datetime


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Service liveness probe. ``model_loaded`` flips true once a scoring bundle is attached."""
    settings = get_settings()
    # Imported lazily to avoid a hard dependency before the model service exists
    # (wired in the prediction phase); the probe must never fail on import.
    try:
        from app.services.model_service import model_is_loaded

        loaded = model_is_loaded()
    except ImportError:
        loaded = False
    return HealthResponse(
        status="ok",
        service="healthcare-payment-integrity-api",
        version=get_version(),
        environment=settings.env,
        model_loaded=loaded,
        timestamp=datetime.now(UTC),
    )
