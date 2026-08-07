"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.middleware import RequestContextMiddleware
from app.api.routers import health, model_info, predict
from src.utils.config import get_settings
from src.utils.logging import configure_logging, request_id_ctx
from src.utils.version import get_version

logger = logging.getLogger(__name__)

API_DESCRIPTION = """
Claim-level payment-integrity risk scoring for healthcare claims.

Returns a calibrated risk score, a recommended action, estimated payment at
risk, and explainable reason codes for every claim. Scores prioritize claims
for human investigation — they are **not** determinations of fraud.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the scoring bundle at startup (skippable for lightweight tests)."""
    import os

    if os.environ.get("SKIP_MODEL_LOAD", "").lower() not in {"1", "true"}:
        from app.services.scoring_service import initialize

        initialize()
    yield


def create_app(load_model: bool = True) -> FastAPI:
    """Build the FastAPI app with middleware, routers, and logging configured."""
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="Healthcare Payment Integrity API",
        description=API_DESCRIPTION,
        version=get_version(),
        contact={"name": "Faridoon Farahi", "url": "https://github.com/FaridoonFarahi"},
        license_info={"name": "MIT"},
        lifespan=lifespan if load_model else None,
    )
    app.add_middleware(RequestContextMiddleware)
    app.include_router(health.router)
    app.include_router(model_info.router)
    app.include_router(predict.router)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {
                "loc": ".".join(str(part) for part in err.get("loc", [])),
                "message": err.get("msg", "invalid value"),
            }
            for err in exc.errors()
        ]
        logger.info("validation_error", extra={"path": request.url.path, "errors": len(details)})
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "detail": "One or more fields failed validation. See `details`.",
                "details": details,
                "request_id": request_id_ctx.get(),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_exception", extra={"path": request.url.path})
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "detail": "Unexpected server error. Correlate with logs via request_id.",
                "request_id": request_id_ctx.get(),
            },
        )

    logger.info("app_created", extra={"environment": settings.env, "version": get_version()})
    return app


app = create_app()
