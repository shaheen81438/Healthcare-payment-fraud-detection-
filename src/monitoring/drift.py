"""Drift monitoring core: Evidently report, PSI/JSD statistics, and alert evaluation.

Reference window = validation-period scored claims, frozen as an artifact.
Current window = the latest scored claims. Thresholds live in
``configs/monitoring.yaml``; every run emits an HTML report, a JSON metric
dump, and a compact ``monitoring_status.json`` consumed by the dashboard.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_monitoring_config(path: str = "configs/monitoring.yaml") -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)
    return cfg


# ------------------------------------------------------------------ statistics
def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with reference-quantile binning."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # degenerate distribution
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(reference, bins=edges)[0] / max(1, len(reference))
    cur_share = np.histogram(current, bins=edges)[0] / max(1, len(current))
    ref_share = np.clip(ref_share, 1e-6, None)
    cur_share = np.clip(cur_share, 1e-6, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def jsd(reference: pd.Series, current: pd.Series, top_k: int | None = None) -> float:
    """Jensen-Shannon divergence between two categorical distributions (base 2)."""
    categories = pd.Index(reference.astype(str)).union(pd.Index(current.astype(str))).unique()
    p = reference.astype(str).value_counts(normalize=True).reindex(categories, fill_value=0.0)
    q = current.astype(str).value_counts(normalize=True).reindex(categories, fill_value=0.0)
    if top_k:
        keep = p.add(q, fill_value=0).nlargest(top_k).index
        p, q = p.reindex(keep, fill_value=0.0), q.reindex(keep, fill_value=0.0)
        p, q = p / max(p.sum(), 1e-9), q / max(q.sum(), 1e-9)
    m = (p + q) / 2

    def _kl(a: pd.Series, b: pd.Series) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.clip(b[mask], 1e-12, None))))

    return float((_kl(p, m) + _kl(q, m)) / 2)


# ------------------------------------------------------------------ evidently
def evidently_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    html_out: Path,
    json_out: Path,
) -> dict[str, Any]:
    """Run Evidently's DataDriftPreset; returns {share, count, per_column}."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    cols = [c for c in numeric + categorical if c in reference.columns and c in current.columns]
    numeric_present = [c for c in numeric if c in cols]
    categorical_present = [c for c in categorical if c in cols]
    definition = DataDefinition(
        numerical_columns=numeric_present, categorical_columns=categorical_present
    )
    report = Report([DataDriftPreset()])
    snapshot = report.run(
        Dataset.from_pandas(reference[cols], data_definition=definition),
        Dataset.from_pandas(current[cols], data_definition=definition),
    )
    snapshot.save_html(str(html_out))
    payload = json.loads(snapshot.json())
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    drifted_share, drifted_count = 0.0, 0
    per_column: dict[str, float] = {}
    for metric in payload.get("metrics", []):
        name = str(metric.get("metric_name", ""))
        value = metric.get("value")
        if name.startswith("DriftedColumnsCount") and isinstance(value, dict):
            drifted_share = float(value.get("share", 0.0))
            drifted_count = int(value.get("count", 0))
        elif name.startswith("ValueDrift(") and isinstance(value, int | float):
            column = name.split("column=")[1].split(",")[0].rstrip(")")
            per_column[column] = float(value)  # p-value (low = drifted)
    return {
        "share": drifted_share,
        "count": drifted_count,
        "n_checked": len(cols),
        "per_column_p_values": per_column,
    }


# ------------------------------------------------------------------ run
def run_monitoring(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    cfg: dict[str, Any],
    reports_dir: Path = Path("monitoring/reports"),
    decisions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Full monitoring pass; writes reports and returns the status document."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    thresholds = cfg["thresholds"]
    numeric = list(cfg["monitored_numeric"])
    categorical = list(cfg["monitored_categorical"])

    drift = evidently_drift_report(
        reference,
        current,
        numeric,
        categorical,
        reports_dir / "drift_report.html",
        reports_dir / "drift_report.json",
    )
    score_psi = psi(reference["risk_score"].to_numpy(), current["risk_score"].to_numpy())
    provider_mix = jsd(reference["specialty"], current["specialty"])
    procedure_mix = jsd(reference["hcpcs"], current["hcpcs"], top_k=20)
    # Compare monthly volumes, since the windows can span different numbers of months.
    ref_months = max(1, pd.to_datetime(reference["service_date"]).dt.to_period("M").nunique())
    cur_months = max(1, pd.to_datetime(current["service_date"]).dt.to_period("M").nunique())
    volume_change = (len(current) / cur_months - len(reference) / ref_months) / max(
        1.0, len(reference) / ref_months
    )

    missing_alerts: list[str] = []
    for col in numeric + categorical:
        if col in reference.columns and col in current.columns:
            delta = float(current[col].isna().mean() - reference[col].isna().mean())
            if delta > float(thresholds["missing_share_increase_warn"]):
                missing_alerts.append(f"missing-share of `{col}` up {delta:.1%} vs reference")

    alerts: list[str] = list(missing_alerts)
    status = "ok"
    if (
        drift["share"] >= thresholds["drift_share_critical"]
        or score_psi >= thresholds["score_psi_critical"]
    ):
        status = "critical"
    elif (
        drift["share"] >= thresholds["drift_share_warn"]
        or score_psi >= thresholds["score_psi_warn"]
        or provider_mix >= thresholds["provider_mix_jsd_warn"]
        or procedure_mix >= thresholds["procedure_mix_jsd_warn"]
        or abs(volume_change) >= thresholds["volume_change_warn"]
        or missing_alerts
    ):
        status = "warn"
    if drift["share"] >= thresholds["drift_share_warn"]:
        alerts.append(
            f"{drift['count']}/{drift['n_checked']} monitored features drifted "
            f"(share {drift['share']:.0%})"
        )
    if score_psi >= thresholds["score_psi_warn"]:
        alerts.append(f"risk-score PSI {score_psi:.3f} above {thresholds['score_psi_warn']}")
    if provider_mix >= thresholds["provider_mix_jsd_warn"]:
        alerts.append(f"provider-mix JSD {provider_mix:.3f} above threshold")
    if procedure_mix >= thresholds["procedure_mix_jsd_warn"]:
        alerts.append(f"procedure-mix JSD {procedure_mix:.3f} above threshold")
    if abs(volume_change) >= thresholds["volume_change_warn"]:
        alerts.append(f"claim volume changed {volume_change:+.0%} vs reference window")

    delayed = _delayed_performance(decisions, current, cfg)
    if delayed.get("alert"):
        alerts.append(str(delayed["alert"]))
        status = "warn" if status == "ok" else status

    doc: dict[str, Any] = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": status,
        "reference_window": _window(reference),
        "current_window": _window(current),
        "reference_rows": len(reference),
        "current_rows": len(current),
        "n_drifted_features": drift["count"],
        "n_features_checked": drift["n_checked"],
        "drift_share": round(drift["share"], 4),
        "score_psi": round(score_psi, 4),
        "provider_mix_jsd": round(provider_mix, 4),
        "procedure_mix_jsd": round(procedure_mix, 4),
        "volume_change": round(float(volume_change), 4),
        "missing_value_alerts": len(missing_alerts),
        "delayed_performance": delayed,
        "alerts": alerts,
        "html_report": str(reports_dir / "drift_report.html"),
    }
    (reports_dir / "monitoring_status.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    logger.info(
        "monitoring_run_complete",
        extra={"status": status, "alerts": len(alerts), "score_psi": doc["score_psi"]},
    )
    return doc


def _window(df: pd.DataFrame) -> str:
    dates = pd.to_datetime(df["service_date"])
    return f"{dates.min():%Y-%m} → {dates.max():%Y-%m}"


def _delayed_performance(
    decisions: pd.DataFrame | None, current: pd.DataFrame, cfg: dict[str, Any]
) -> dict[str, Any]:
    if decisions is None or decisions.empty:
        return {"n_decisions": 0}
    decided = decisions[decisions["status"].isin(["approved", "dismissed"])]
    n = len(decided)
    if n < int(cfg["delayed_labels"]["min_decisions"]):
        return {"n_decisions": n, "note": "below min_decisions; not evaluated"}
    precision = float((decided["status"] == "approved").mean())
    out: dict[str, Any] = {"n_decisions": n, "realized_flag_precision": round(precision, 4)}
    if precision < float(cfg["delayed_labels"]["precision_warn"]):
        out["alert"] = (
            f"realized flag precision {precision:.0%} below "
            f"{cfg['delayed_labels']['precision_warn']:.0%} across {n} decisions"
        )
    return out


# ------------------------------------------------------------------ simulation
def simulate_drift(current: pd.DataFrame, rng_seed: int = 13) -> pd.DataFrame:
    """Inject drift for demos and tests: a pricing shift, a mix shift, and new missing values."""
    rng = np.random.default_rng(rng_seed)
    df = current.copy()
    # Payment inflation concentrated in one large specialty.
    target = df["specialty"].value_counts().index[0]
    mask = df["specialty"] == target
    for col in ["paid_amount", "paid_per_unit", "paid_vs_code_median"]:
        if col in df.columns:
            df.loc[mask, col] = df.loc[mask, col] * 1.6
    # Provider-mix shift: drop most claims from two specialties.
    drop_specs = df["specialty"].value_counts().index[-2:]
    drop_mask = df["specialty"].isin(drop_specs) & (rng.random(len(df)) < 0.8)
    df = df[~drop_mask].copy()
    # New missingness in POS.
    df.loc[rng.random(len(df)) < 0.05, "pos"] = np.nan
    # Scores drift with the features. As a stand-in for rescoring, raise them slightly.
    df["risk_score"] = np.clip(df["risk_score"] * 1.25 + 0.02, 0, 1)
    return df
