# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Base: shared Python environment for API and dashboard targets
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# libgomp1: OpenMP runtime required by XGBoost / SHAP; curl: container healthchecks
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for layer caching
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Install the application
COPY pyproject.toml README.md ./
COPY src/ src/
COPY app/ app/
COPY configs/ configs/
RUN pip install --no-deps .

# Non-root user
RUN useradd --create-home --uid 1000 appuser
USER appuser

# ---------------------------------------------------------------------------
# API target
# ---------------------------------------------------------------------------
FROM base AS api
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
