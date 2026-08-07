#!/usr/bin/env python
"""Run drift monitoring: reference (validation window) vs current (latest scored months).

Usage:
    python scripts/run_monitoring.py [--simulate-drift] [--current-months 2]

Outputs to monitoring/reports/: drift_report.html, drift_report.json,
monitoring_status.json; appends a row to ops.monitoring_runs when Postgres is up.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modeling.bundle import ScoringBundle
from src.modeling.train import load_model_config, prepare_training_frame
from src.monitoring.drift import load_monitoring_config, run_monitoring, simulate_drift
from src.utils.logging import configure_logging

REFERENCE_PATH = Path("models/feature_store/monitoring_reference.parquet")

MONITOR_COLUMNS = [
    "claim_id",
    "specialty",
    "hcpcs",
    "pos",
    "dx_group",
    "service_date",
    "paid_amount",
    "paid_per_unit",
    "paid_vs_code_median",
    "units",
    "dup_max_sim",
    "bene_claims_30d",
    "prov_claims_3m",
    "prov_paid_vs_peer_3m",
    "prov_top_code_share_3m",
    "prov_new_bene_share_1m",
    "pos_atypicality",
    "dxproc_compat",
    "submit_lag_days",
    "risk_score",
]


def _scored_frame(split: str) -> pd.DataFrame:
    """Features + risk scores for a split (the monitoring view of traffic)."""
    cfg = load_model_config()
    bundle = ScoringBundle.load_latest()
    features = pd.read_parquet("data/processed/features.parquet")
    labels = pd.read_parquet("data/processed/labels.parquet")
    frame, _, _ = prepare_training_frame(
        features[features["split"] == split], labels, bool(cfg["include_rule_features"])
    )
    scored = bundle.risk_score(frame[bundle.numeric_features + bundle.categorical_features])
    frame = frame.assign(risk_score=scored["risk_score"].to_numpy())
    return frame[MONITOR_COLUMNS].reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulate-drift", action="store_true")
    parser.add_argument("--current-months", type=int, default=2)
    args = parser.parse_args()
    configure_logging("INFO")
    cfg = load_monitoring_config()

    # Reference: frozen validation-window snapshot (built once).
    if REFERENCE_PATH.exists():
        reference = pd.read_parquet(REFERENCE_PATH)
    else:
        reference = _scored_frame("val")
        REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        reference.to_parquet(REFERENCE_PATH, index=False)

    # Current: latest N months of scored test traffic.
    current = _scored_frame("test")
    months = sorted(pd.to_datetime(current["service_date"]).dt.to_period("M").unique())
    keep = [str(m) for m in months[-args.current_months :]]
    current = current[
        pd.to_datetime(current["service_date"]).dt.to_period("M").astype(str).isin(keep)
    ].reset_index(drop=True)
    if args.simulate_drift:
        current = simulate_drift(current)

    # Investigator decisions (delayed labels), best effort.
    decisions = None
    try:
        from src.data.db import read_df

        decisions = read_df("SELECT * FROM ops.review_decisions")
    except Exception:
        fallback = Path("data/processed/review_decisions.parquet")
        if fallback.exists():
            decisions = pd.read_parquet(fallback)

    doc = run_monitoring(reference, current, cfg, decisions=decisions)

    # Best-effort ops log.
    try:
        from src.data.db import write_df

        write_df(
            pd.DataFrame(
                [
                    {
                        "run_at": doc["run_at"],
                        "status": doc["status"],
                        "drift_share": doc["drift_share"],
                        "score_psi": doc["score_psi"],
                        "alerts": json.dumps(doc["alerts"]),
                        "html_report": doc["html_report"],
                    }
                ]
            ),
            "monitoring_runs",
            schema="ops",
            if_exists="append",
        )
    except Exception:
        pass

    print(
        json.dumps(
            {
                k: doc[k]
                for k in ["status", "drift_share", "score_psi", "provider_mix_jsd", "alerts"]
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
