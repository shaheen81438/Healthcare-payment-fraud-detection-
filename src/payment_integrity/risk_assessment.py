"""Central, availability-aware investigation-priority aggregation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

DEFAULT_WEIGHTS = {
    "improper_payment": 0.35,
    "provider_anomaly": 0.30,
    "leie_screening": 0.20,
    "supervised_model": 0.15,
}


def _clamp(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))


def aggregate_risk(
    *,
    improper_payment_score: float | None = None,
    provider_anomaly_score: float | None = None,
    leie_matches: Sequence[Mapping[str, Any]] | None = None,
    supervised_model_score: float | None = None,
    weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Aggregate normalized signals without treating missing signals as zero.

    The output is an investigation priority. It is not a fraud probability.
    Available weights are renormalized so an unavailable component is explicit
    and does not silently lower the score.
    """
    configured = dict(DEFAULT_WEIGHTS)
    configured.update(weights or {})
    components: dict[str, dict[str, Any]] = {
        "improper_payment": {
            "available": improper_payment_score is not None,
            "score": None if improper_payment_score is None else _clamp(improper_payment_score),
        },
        "provider_anomaly": {
            "available": provider_anomaly_score is not None,
            "score": None if provider_anomaly_score is None else _clamp(provider_anomaly_score),
        },
        "leie_screening": {
            "available": leie_matches is not None,
            "score": None,
        },
        "supervised_model": {
            "available": supervised_model_score is not None,
            "score": None if supervised_model_score is None else _clamp(supervised_model_score),
        },
    }
    if leie_matches is not None:
        components["leie_screening"]["score"] = max(
            (_clamp(match.get("match_confidence", 0.0)) for match in leie_matches),
            default=0.0,
        )
        components["leie_screening"]["match_count"] = len(leie_matches)

    available = {name: item for name, item in components.items() if item["available"]}
    total_weight = sum(float(configured[name]) for name in available)
    if total_weight <= 0:
        overall = None
    else:
        overall = sum(
            float(configured[name]) / total_weight * float(item["score"])
            for name, item in available.items()
        )
    reasons = [
        name
        for name, item in components.items()
        if item["available"] and item["score"] is not None and float(item["score"]) >= 0.5
    ]
    warnings = [
        f"{name} signal unavailable" for name, item in components.items() if not item["available"]
    ]
    priority = (
        "unavailable"
        if overall is None
        else ("high" if overall >= 0.75 else "medium" if overall >= 0.5 else "routine")
    )
    return {
        "overall_investigation_priority": overall,
        "priority_band": priority,
        "components": components,
        "weights_used": {name: float(configured[name]) / total_weight for name in available}
        if total_weight
        else {},
        "reason_codes": reasons,
        "data_quality_warnings": warnings,
        "interpretation": "Investigation priority only; not a calibrated fraud probability.",
    }
