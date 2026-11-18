"""Specialty-aware provider anomaly detection over CMS provider-service data."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def _robust_z(values: pd.Series, median: float, mad: float) -> pd.Series:
    scale = max(1.4826 * mad, 1e-9)
    return (values - median).abs() / scale


def _priority(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.5:
        return "medium"
    return "routine"


def detect_provider_anomalies(
    provider_service: pd.DataFrame,
    provider_id_column: str = "npi",
    peer_column: str = "provider_type",
    random_state: int = 42,
    contamination: float = 0.05,
) -> pd.DataFrame:
    """Score providers only against peers in the same specialty/provider type.

    The result is a review-priority ranking, not a fraud label. Isolation Forest
    is blended with a transparent robust-deviation baseline.
    """
    required = {provider_id_column, peer_column, "total_services", "avg_paid"}
    missing = required - set(provider_service.columns)
    if missing:
        raise ValueError(f"provider-service data missing required columns: {sorted(missing)}")

    frame = provider_service.copy()
    frame["total_services"] = pd.to_numeric(frame["total_services"], errors="coerce").fillna(0)
    for col in ("avg_paid", "avg_allowed", "avg_submitted"):
        if col not in frame:
            frame[col] = np.nan
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "hcpcs" not in frame:
        frame["hcpcs"] = "unknown"
    frame["hcpcs"] = frame["hcpcs"].astype(str)
    group_cols = [provider_id_column, peer_column]
    rows: list[dict[str, object]] = []
    for (provider_id, peer), group in frame.groupby(group_cols, dropna=False):
        weights = group["total_services"].clip(lower=0)
        total = float(weights.sum())
        if total <= 0:
            total = float(len(group))
            weights = pd.Series(1.0, index=group.index)
        paid = float(np.average(group["avg_paid"].fillna(0), weights=weights))
        allowed = float(np.average(group["avg_allowed"].fillna(0), weights=weights))
        submitted = float(np.average(group["avg_submitted"].fillna(0), weights=weights))
        code_share = group.groupby("hcpcs")["total_services"].sum() / max(total, 1.0)
        rows.append(
            {
                "provider_id": str(provider_id),
                "peer_group": str(peer),
                "service_volume": total,
                "average_paid": paid,
                "average_allowed": allowed,
                "average_submitted": submitted,
                "procedure_concentration": float(code_share.max()) if len(code_share) else 0.0,
            }
        )
    features = pd.DataFrame(rows)
    if features.empty:
        return pd.DataFrame(
            columns=[
                "provider_id",
                "peer_group",
                "anomaly_score",
                "statistical_score",
                "isolation_score",
                "triggered_indicators",
                "explanation",
                "investigation_priority",
            ]
        )

    numeric = ["service_volume", "average_paid", "average_allowed", "procedure_concentration"]
    features["log_service_volume"] = np.log1p(features["service_volume"])
    baseline = pd.Series(0.0, index=features.index)
    trigger_sets: list[set[str]] = [set() for _ in range(len(features))]
    for _peer, indices in features.groupby("peer_group").groups.items():
        peer_frame = features.loc[indices]
        for col in numeric:
            values = peer_frame[col]
            median = float(values.median())
            mad = float((values - median).abs().median())
            z = _robust_z(values, median, mad)
            baseline.loc[indices] += z.clip(upper=5.0) / max(len(numeric), 1)
            for idx, value in z.items():
                if value >= 3:
                    trigger_sets[features.index.get_loc(idx)].add(f"{col}_peer_outlier")
        if len(peer_frame) >= 3:
            concentration_q = float(peer_frame["procedure_concentration"].quantile(0.95))
            for idx in peer_frame.index[peer_frame["procedure_concentration"] >= concentration_q]:
                trigger_sets[features.index.get_loc(idx)].add("procedure_concentration")

    features["statistical_score"] = (1 - np.exp(-baseline / 2.0)).clip(0, 1)
    iso_values = np.zeros(len(features), dtype=float)
    if len(features) >= 5:
        model = IsolationForest(
            n_estimators=200,
            contamination=min(max(contamination, 0.01), 0.5),
            random_state=random_state,
        )
        raw = -model.fit(features[["log_service_volume", *numeric[1:]]]).score_samples(
            features[["log_service_volume", *numeric[1:]]]
        )
        ranks = pd.Series(raw).rank(pct=True).to_numpy()
        iso_values = ranks
    features["isolation_score"] = iso_values
    features["anomaly_score"] = (
        0.5 * features["statistical_score"] + 0.5 * features["isolation_score"]
    ).clip(0, 1)
    features["triggered_indicators"] = [json.dumps(sorted(values)) for values in trigger_sets]
    features["explanation"] = features.apply(
        lambda row: (
            f"Compared with {row['peer_group']} peers: "
            f"volume={row['service_volume']:.0f}, average paid=${row['average_paid']:.2f}, "
            f"procedure concentration={row['procedure_concentration']:.1%}."
        ),
        axis=1,
    )
    features["investigation_priority"] = features["anomaly_score"].map(_priority)
    return features.drop(columns=["log_service_volume"])
