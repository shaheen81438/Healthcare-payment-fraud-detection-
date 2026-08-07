.DEFAULT_GOAL := help
SHELL := /bin/bash

-include .env
export

.PHONY: help install format lint typecheck test test-all compose-up compose-down compose-serve api clean data dashboard features train evaluate score monitor pipeline

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install the package with dev dependencies
	pip install -e ".[dev]"

format: ## Auto-format code with Ruff
	ruff format .
	ruff check --fix .

lint: ## Check formatting and lint rules
	ruff format --check .
	ruff check .

typecheck: ## Run MyPy on src/ and app/
	mypy src app

test: ## Run fast unit tests with coverage
	pytest -m "not integration and not slow" --cov=src --cov=app --cov-report=term-missing

test-all: ## Run the full test suite (requires docker compose services)
	pytest --cov=src --cov=app --cov-report=term-missing

compose-up: ## Start Postgres + MLflow locally
	docker compose up -d postgres mlflow

compose-down: ## Stop all compose services
	docker compose down

compose-serve: ## Build and start the full stack (API + dashboard included)
	docker compose --profile serve up -d --build

data: ## Build the dataset: ingest, generate, validate, clean, load Postgres
	python scripts/build_dataset.py

api: ## Run the FastAPI app locally with reload
	uvicorn app.api.main:app --reload --port $${API_PORT:-8000}

dashboard: ## Run the Streamlit investigator dashboard
	streamlit run app/dashboard/Home.py --server.port $${DASHBOARD_PORT:-8501} --server.headless true

features: ## Build the feature matrix + frozen peer stats
	python scripts/build_features.py

train: ## Train models, log + register with MLflow
	python scripts/train_model.py

evaluate: ## Evaluate, select thresholds, build the report
	python scripts/evaluate_model.py

score: ## Batch-score the test window with SHAP reasons -> Postgres
	python scripts/score_claims.py

monitor: ## Run Evidently drift monitoring
	python scripts/run_monitoring.py

pipeline: data features train evaluate score ## Full pipeline end-to-end

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
