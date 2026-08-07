"""MLflow experiment tracking + model registry, with offline-tolerant fallbacks.

Every training run logs params, metrics, dataset/git provenance, and the full
scoring-bundle directory. The chosen model is registered as ``claims-risk``
with the ``production`` alias. Serving loads from the registry when reachable,
otherwise from the local bundle pointer — the API never depends on MLflow
being up.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from src.modeling.bundle import ScoringBundle
from src.utils.config import get_settings

logger = logging.getLogger(__name__)


def _git_commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout.strip()
            or "unknown"
        )
    except OSError:  # pragma: no cover
        return "unknown"


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for model_name, per_split in metrics.items():
        if not isinstance(per_split, dict):
            continue
        for split, values in per_split.items():
            if isinstance(values, dict):
                for metric, value in values.items():
                    if isinstance(value, int | float):
                        flat[f"{model_name}_{split}_{metric}"] = float(value)
            elif isinstance(values, int | float):
                flat[f"{model_name}_{split}"] = float(values)
    return flat


def log_and_register(
    bundle_dir: Path,
    metrics: dict[str, Any],
    cfg: dict[str, Any],
    dataset_version: str,
    register: bool = True,
) -> str | None:
    """Log the run to MLflow and register the model; returns run_id (None if offline)."""
    settings = get_settings()
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment)
        with mlflow.start_run(run_name=str(cfg["model_version"])) as run:
            mlflow.log_params(
                {
                    "model_version": cfg["model_version"],
                    "dataset_version": dataset_version,
                    "git_commit": _git_commit(),
                    **{f"xgb_{k}": v for k, v in cfg["xgboost"].items()},
                    **{f"iso_{k}": v for k, v in cfg["isolation_forest"].items()},
                    "ensemble_weight": metrics.get("ensemble", {}).get("weight"),
                    "calibration": cfg["calibration"]["method"],
                }
            )
            mlflow.log_metrics(_flatten_metrics(metrics))
            mlflow.log_artifacts(str(bundle_dir), artifact_path="bundle")
            mlflow.set_tags(
                {
                    "stage": "training",
                    "dataset_version": dataset_version,
                    "git_commit": _git_commit(),
                }
            )
            run_id = run.info.run_id
        if register:
            from mlflow import MlflowClient

            client = MlflowClient()
            model_uri = f"runs:/{run_id}/bundle"
            registered = mlflow.register_model(model_uri, "claims-risk")
            client.set_registered_model_alias("claims-risk", "production", registered.version)
            logger.info(
                "model_registered",
                extra={
                    "model_name": "claims-risk",
                    "registry_version": registered.version,
                    "alias": "production",
                },
            )
        return str(run_id)
    except Exception as exc:  # MLflow unreachable → training still succeeds
        logger.warning("mlflow_unavailable", extra={"error": str(exc)[:200]})
        return None


def load_production_bundle() -> ScoringBundle:
    """Load the production bundle: MLflow registry first, local pointer as fallback."""
    settings = get_settings()
    try:
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = MlflowClient()
        version = client.get_model_version_by_alias("claims-risk", "production")
        local = mlflow.artifacts.download_artifacts(
            "models:/claims-risk@production", dst_path="models/_registry_cache"
        )
        logger.info("bundle_loaded_from_registry", extra={"registry_version": version.version})
        bundle = ScoringBundle.load(local)
        bundle.metadata["source"] = "mlflow_registry"
        return bundle
    except Exception as exc:
        logger.warning("registry_fallback_to_local", extra={"error": str(exc)[:200]})
        bundle = ScoringBundle.load_latest()
        bundle.metadata["source"] = "local_bundle"
        return bundle


def local_bundle_info() -> dict[str, Any]:
    pointer = Path("models/latest.json")
    if pointer.exists():
        info: dict[str, Any] = json.loads(pointer.read_text())
        return info
    return {}
