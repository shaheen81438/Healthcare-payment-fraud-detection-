"""FastAPI application factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from app.api.middleware import RequestContextMiddleware
from app.api.routers import health
from src.utils.config import get_settings
from src.utils.logging import configure_logging
from src.utils.version import get_version

logger = logging.getLogger(__name__)

API_DESCRIPTION = """
Claim-level payment-integrity risk scoring for healthcare claims.

Returns a calibrated risk score, a recommended action, estimated payment at
risk, and explainable reason codes for every claim. Scores prioritize claims
for human investigation — they are **not** determinations of fraud.
"""


def create_app() -> FastAPI:
    """Build the FastAPI app with middleware, routers, and logging configured."""
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="Healthcare Payment Integrity API",
        description=API_DESCRIPTION,
        version=get_version(),
        contact={"name": "Faridoon Farahi", "url": "https://github.com/FaridoonFarahi"},
        license_info={"name": "MIT"},
    )
    app.add_middleware(RequestContextMiddleware)
    app.include_router(health.router)
    logger.info("app_created", extra={"environment": settings.env, "version": get_version()})
    return app


app = create_app()
