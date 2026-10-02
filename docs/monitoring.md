# Monitoring guide

## What runs

`make monitor` (which calls `scripts/run_monitoring.py`) compares a **frozen reference window** (the
validation months, snapshotted to `models/feature_store/monitoring_reference.parquet`) against the
**current window** (latest scored months) and writes to `monitoring/reports/`:

| Artifact | Contents |
|---|---|
| `drift_report.html` | Full Evidently report (also viewable inline on the dashboard's Monitoring page) |
| `drift_report.json` | Raw Evidently metrics (per-column drift p-values) |
| `monitoring_status.json` | Compact status doc: `ok / warn / critical`, key stats, alert strings |
| `ops.monitoring_runs` (Postgres) | Append-only run log |

## What is measured

| Signal | Method | Warn | Critical |
|---|---|---|---|
| Input feature drift | Evidently `DataDriftPreset` over 13 numeric + 3 categorical monitored columns | drifted share ≥ 0.30 | ≥ 0.50 |
| Prediction drift | PSI on the risk-score distribution (reference-quantile bins) | ≥ 0.20 | ≥ 0.30 |
| Provider mix | Jensen-Shannon divergence on specialty shares | ≥ 0.10 | n/a |
| Procedure mix | JSD on top-20 HCPCS shares | ≥ 0.10 | n/a |
| Volume | Monthly claim-volume delta vs reference | \|Δ\| ≥ 50% | n/a |
| Data quality | Missing-share increase per monitored column | +2 pp | n/a |
| Delayed performance | Realized precision of flags from investigator decisions (≥25 decided) | < 50% | n/a |

Thresholds live in `configs/monitoring.yaml`; the status escalates to the worst triggered level
and every triggered condition becomes a human-readable alert string.

## Demo and fire drill

```bash
python scripts/run_monitoring.py --simulate-drift
```

This injects a realistic scenario: payment inflation in the largest specialty, a specialty-mix
shift, new POS missingness, and score inflation. Verified behavior: the baseline run returns
**ok**, and the simulated run returns **critical** with `risk-score PSI 3.813` and a missing-POS
alert. Re-run without the flag to
restore the truthful status.

## Delayed labels

Investigator decisions captured in the dashboard (`approved` / `dismissed`) are the platform's
eventual ground truth. Once ≥25 decisions exist, each monitoring run reports realized flag
precision and alerts when it falls under 50%. This is the early warning that the model's flags are
losing investigator trust, and it usually comes before measurable drift.

## Runbook

- warn (input drift only): inspect `drift_report.html` per-column results; if the drivers are
  amount features, check for fee-schedule updates before suspecting the model.
- warn (mix/volume): usually a population change (new provider cohort, seasonal). Consider
  refreshing peer stats and retraining (`make pipeline`) if it holds for two consecutive runs.
- critical (score PSI): stop trusting the queue ordering; retrain and re-evaluate before
  resuming automated prioritization; keep rules-only triage in the interim.
- delayed-precision alert: sample recent dismissed claims with investigators. The cause is often
  a threshold/capacity mismatch, so revisit the Threshold Simulator before touching the model.
- After any retrain: the evaluation step re-freezes the reference and re-registers the bundle;
  delete `models/feature_store/monitoring_reference.parquet` to rebase the reference window.

## Production notes

Single-process API metrics are in-process counters (`/metrics`); a multi-worker deployment should
swap in `prometheus-client` with multiprocess mode. Schedule `make monitor` (cron / GitHub
Actions schedule / Azure Container Apps job) at your scoring cadence, which is monthly here.
