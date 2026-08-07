"""Scoring orchestration for the API: features → rules → ensemble → reasons."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from app.schemas.claims import (
    ClaimIn,
    ExplainOut,
    ModelInfoOut,
    PredictionOut,
    ShapContribution,
    ThresholdPolicyOut,
)
from app.services import model_service
from app.services.feature_service import FeatureService
from src.evaluation.financial import recommended_action
from src.modeling.bundle import ScoringBundle
from src.modeling.explain import ClaimExplainer
from src.modeling.rules import RuleEngine

logger = logging.getLogger(__name__)


class ScoringService:
    """Process-wide singleton wrapping the bundle + explainer + feature assembly."""

    def __init__(self, bundle: ScoringBundle, source: str) -> None:
        self.bundle = bundle
        self.source = source
        self.features = FeatureService()
        self.rules = RuleEngine.from_yaml()
        self.explainer = ClaimExplainer(bundle)
        self.policy: dict[str, Any] = bundle.threshold_policy or {
            "review_threshold": 0.9,
            "monitor_threshold": 0.45,
            "capacity_share": 0.01,
            "cost_params": {},
        }

    # ------------------------------------------------------------------ scoring
    def _assemble(self, claims: list[ClaimIn]) -> Any:
        df = self.features.build(claims)
        evaluated = self.rules.evaluate(df)
        df = df.merge(
            evaluated[["claim_id", "rule_score", "n_rules_hit", "rule_hits", "rule_reasons"]],
            on="claim_id",
            how="left",
        )
        missing = [
            c
            for c in self.bundle.numeric_features + self.bundle.categorical_features
            if c not in df.columns
        ]
        if missing:  # defensive: feature drift between bundle and service code
            raise RuntimeError(f"online features missing columns: {missing}")
        return df

    def predict(self, claims: list[ClaimIn]) -> list[PredictionOut]:
        df = self._assemble(claims)
        X = df[self.bundle.numeric_features + self.bundle.categorical_features]
        scored = self.bundle.risk_score(X)
        now = datetime.now(UTC)
        results: list[PredictionOut] = []
        for i in df.index:
            risk = float(scored.at[i, "risk_score"])
            prob = float(scored.at[i, "supervised_probability"])
            paid = float(df.at[i, "paid_amount"])
            rule_reasons = str(df.at[i, "rule_reasons"] or "")
            reason_codes = [r for r in rule_reasons.split(";") if r][:5] or (
                ["no_specific_reason"] if risk < 0.2 else []
            )
            results.append(
                PredictionOut(
                    claim_id=str(df.at[i, "claim_id"]),
                    risk_score=round(risk, 4),
                    recommended_action=recommended_action(risk, self.policy),  # type: ignore[arg-type]
                    estimated_payment_at_risk=round(prob * paid, 2),
                    reason_codes=reason_codes,
                    supervised_probability=round(prob, 4),
                    anomaly_percentile=round(float(scored.at[i, "anomaly_percentile"]), 4),
                    rule_hits=[r for r in str(df.at[i, "rule_hits"] or "").split(";") if r],
                    insufficient_provider_history=bool(df.at[i, "insufficient_provider_history"]),
                    model_version=self.bundle.model_version,
                    scored_at=now,
                )
            )
        logger.info(
            "claims_scored",
            extra={
                "n": len(results),
                "manual_review": sum(r.recommended_action == "manual_review" for r in results),
                "model_version": self.bundle.model_version,
            },
        )
        return results

    def explain(self, claims: list[ClaimIn]) -> list[ExplainOut]:
        df = self._assemble(claims)
        X = df[self.bundle.numeric_features + self.bundle.categorical_features]
        scored = self.bundle.risk_score(X)
        contribs, base_value = self.explainer.top_contributions(X)
        labels = self.explainer.reason_labels()
        now = datetime.now(UTC)
        results: list[ExplainOut] = []
        for pos, i in enumerate(df.index):
            risk = float(scored.at[i, "risk_score"])
            prob = float(scored.at[i, "supervised_probability"])
            anomaly = float(scored.at[i, "anomaly_percentile"])
            rule_reasons = str(df.at[i, "rule_reasons"] or "")
            codes = self.explainer.reason_codes(
                contribs[pos], rule_reasons=rule_reasons, anomaly_percentile=anomaly
            )
            results.append(
                ExplainOut(
                    claim_id=str(df.at[i, "claim_id"]),
                    risk_score=round(risk, 4),
                    recommended_action=recommended_action(risk, self.policy),  # type: ignore[arg-type]
                    estimated_payment_at_risk=round(prob * float(df.at[i, "paid_amount"]), 2),
                    reason_codes=codes,
                    supervised_probability=round(prob, 4),
                    anomaly_percentile=round(anomaly, 4),
                    rule_hits=[r for r in str(df.at[i, "rule_hits"] or "").split(";") if r],
                    insufficient_provider_history=bool(df.at[i, "insufficient_provider_history"]),
                    model_version=self.bundle.model_version,
                    scored_at=now,
                    base_value=round(base_value, 5),
                    shap_contributions=[ShapContribution(**c) for c in contribs[pos]],
                    reason_labels={c: labels.get(c, c) for c in codes},
                )
            )
        return results

    # ------------------------------------------------------------------ info
    def model_info(self) -> ModelInfoOut:
        meta = self.bundle.metadata
        metrics: dict[str, float] = {}
        for name, values in meta.get("headline_metrics", {}).items():
            metrics[name] = float(values)
        return ModelInfoOut(
            model_version=self.bundle.model_version,
            source=self.source,  # type: ignore[arg-type]
            dataset_version=str(meta.get("dataset_version", "unknown")),
            trained_rows=meta.get("train_rows"),
            numeric_features=len(self.bundle.numeric_features),
            categorical_features=len(self.bundle.categorical_features),
            ensemble_weight=self.bundle.ensemble_weight,
            threshold_policy=ThresholdPolicyOut(
                review_threshold=float(self.policy["review_threshold"]),
                monitor_threshold=float(self.policy["monitor_threshold"]),
                capacity_share=float(self.policy.get("capacity_share", 0.01)),
                cost_params={k: float(v) for k, v in self.policy.get("cost_params", {}).items()},
            ),
            metrics=metrics,
        )


def initialize() -> ScoringService | None:
    """Load the production bundle and attach the service singleton (startup)."""
    try:
        from src.modeling.registry import load_production_bundle

        bundle = load_production_bundle()
        source = str(bundle.metadata.get("source", "local_bundle"))
        service = ScoringService(bundle, source=source)
        model_service.set_bundle(service)
        logger.info("scoring_service_initialized", extra={"model_version": bundle.model_version})
        return service
    except Exception as exc:
        logger.warning("scoring_service_unavailable", extra={"error": str(exc)[:300]})
        model_service.set_bundle(None)
        return None


def get_service() -> ScoringService | None:
    svc = model_service.get_bundle()
    return svc if isinstance(svc, ScoringService) else None
