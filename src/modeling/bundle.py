"""The scoring bundle: everything needed to score a claim, versioned as one artifact.

A bundle directory contains joblib artifacts + metadata.json + frozen peer stats,
so the API and batch scorer can run with no database and no MLflow server.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LATEST_POINTER = Path("models/latest.json")


@dataclass
class ScoringBundle:
    """Fitted artifacts + config required to produce a risk score with reasons."""

    model_version: str
    preprocessor: Any  # fitted ColumnTransformer (unscaled variant, for trees/SHAP)
    xgb_model: Any
    calibrator: Any  # isotonic regression: raw xgb prob -> calibrated prob
    iso_forest: Any
    iso_features: list[str]
    anomaly_ref_scores: np.ndarray  # sorted train anomaly scores for percentile mapping
    ensemble_weight: float
    numeric_features: list[str]
    categorical_features: list[str]
    feature_names_out: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    shap_background: pd.DataFrame | None = None
    peer_stats: dict[str, pd.DataFrame] = field(default_factory=dict)
    threshold_policy: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ scoring
    def supervised_probability(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated probability from the supervised path."""
        matrix = self.preprocessor.transform(X)
        raw = self.xgb_model.predict_proba(matrix)[:, 1]
        return np.clip(self.calibrator.predict(raw), 0.0, 1.0)

    def anomaly_percentile(self, X: pd.DataFrame) -> np.ndarray:
        """IsolationForest anomaly score mapped to a train-referenced percentile."""
        anomaly = -self.iso_forest.score_samples(X[self.iso_features].fillna(0.0))
        ranks = np.searchsorted(self.anomaly_ref_scores, anomaly, side="right")
        return ranks / max(1, len(self.anomaly_ref_scores))

    def risk_score(self, X: pd.DataFrame) -> pd.DataFrame:
        """Final ensemble risk with components; index-aligned with X."""
        p = self.supervised_probability(X)
        a = self.anomaly_percentile(X)
        w = self.ensemble_weight
        return pd.DataFrame(
            {
                "supervised_probability": np.round(p, 6),
                "anomaly_percentile": np.round(a, 6),
                "risk_score": np.round(w * p + (1 - w) * a, 6),
            },
            index=X.index,
        )

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> Path:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "preprocessor": self.preprocessor,
                "xgb_model": self.xgb_model,
                "calibrator": self.calibrator,
                "iso_forest": self.iso_forest,
                "anomaly_ref_scores": self.anomaly_ref_scores,
            },
            path / "artifacts.joblib",
            compress=3,
        )
        if self.shap_background is not None:
            self.shap_background.to_parquet(path / "shap_background.parquet", index=False)
        for name, frame in self.peer_stats.items():
            frame.to_parquet(path / f"peer_{name}.parquet", index=False)
        config = {
            "model_version": self.model_version,
            "ensemble_weight": self.ensemble_weight,
            "iso_features": self.iso_features,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "feature_names_out": self.feature_names_out,
            "threshold_policy": self.threshold_policy,
            "peer_stats_tables": sorted(self.peer_stats),
            "metadata": self.metadata,
        }
        (path / "metadata.json").write_text(json.dumps(config, indent=2, default=str))
        LATEST_POINTER.parent.mkdir(parents=True, exist_ok=True)
        LATEST_POINTER.write_text(
            json.dumps({"path": str(path), "model_version": self.model_version})
        )
        logger.info("bundle_saved", extra={"path": str(path), "version": self.model_version})
        return path

    @classmethod
    def load(cls, directory: str | Path) -> ScoringBundle:
        path = Path(directory)
        arts = joblib.load(path / "artifacts.joblib")
        config = json.loads((path / "metadata.json").read_text())
        background = None
        if (path / "shap_background.parquet").exists():
            background = pd.read_parquet(path / "shap_background.parquet")
        peer = {
            name: pd.read_parquet(path / f"peer_{name}.parquet")
            for name in config.get("peer_stats_tables", [])
        }
        return cls(
            model_version=config["model_version"],
            preprocessor=arts["preprocessor"],
            xgb_model=arts["xgb_model"],
            calibrator=arts["calibrator"],
            iso_forest=arts["iso_forest"],
            iso_features=config["iso_features"],
            anomaly_ref_scores=np.asarray(arts["anomaly_ref_scores"]),
            ensemble_weight=float(config["ensemble_weight"]),
            numeric_features=config["numeric_features"],
            categorical_features=config["categorical_features"],
            feature_names_out=config["feature_names_out"],
            metadata=config.get("metadata", {}),
            shap_background=background,
            peer_stats=peer,
            threshold_policy=config.get("threshold_policy", {}),
        )

    @classmethod
    def load_latest(cls) -> ScoringBundle:
        pointer = json.loads(LATEST_POINTER.read_text())
        return cls.load(pointer["path"])
