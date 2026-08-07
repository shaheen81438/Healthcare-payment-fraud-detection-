# Resume Bullets

Pick 3–5; each is truthful to what the repository demonstrates. Quantified claims reference the
committed evaluation report.

**Project line**

> **Healthcare Claims Payment-Integrity & FWA Detection Platform** — Python, XGBoost, FastAPI,
> PostgreSQL, MLflow, Evidently, Streamlit, Docker, GitHub Actions, Azure ·
> github.com/FaridoonFarahi/healthcare-payment-integrity

**Bullets**

- Built an end-to-end payment-integrity ML platform that scores healthcare claims for
  fraud/waste/abuse risk — rule engine + calibrated XGBoost + Isolation Forest ensemble reaching
  **0.88 PR-AUC (19× the 4.7% base rate)** and capturing **84% of improper claims in a top-5%
  review queue** on a temporally held-out test window.
- Engineered leakage-safe, point-in-time features (backward-only rolling windows, month-lagged
  provider aggregates, frozen peer-group statistics) over **378k synthetic claims calibrated to
  real CMS Medicare utilization data**, with automated leakage tests (future-append invariance,
  label isolation) enforcing the contract in CI.
- Designed a **cost-based review policy**: expected-net-value optimization over investigator
  capacity, review cost, recovery rate, and false-positive cost — achieving 99% precision at 1%
  review capacity (~$674k identified per 1,000 reviews on the test window) with a live what-if
  simulator for operations.
- Shipped a production-style serving layer: FastAPI scoring API (Pydantic-validated, SHAP reason
  codes on every prediction, request-ID structured logging, Prometheus metrics) sharing one
  feature implementation with batch pipelines to eliminate train/serve skew, plus a 6-page
  Streamlit investigator dashboard with decision capture.
- Implemented full MLOps: MLflow experiment tracking + model registry with production aliases,
  Evidently drift monitoring with PSI/mix-divergence alerting and delayed-label precision
  tracking, and a GitHub Actions pipeline whose **model-quality regression gate fails CI** when
  PR-AUC/recall floors regress — deployed to Azure Container Apps via OIDC.
- Documented transparent weak-supervision methodology (12 injected FWA pattern families with
  per-pattern detection analysis and a novel-scheme holdout experiment), separating real CMS
  source data from generated labels end to end.

**Shorter variants (space-constrained)**

- Payment-integrity ML platform (XGBoost + anomaly ensemble, 0.88 PR-AUC): capacity-optimized
  review queue capturing 84% of improper claims at 5% review rate; FastAPI + SHAP explanations;
  MLflow registry; Evidently drift alerts; CI model-quality gate; Azure deployment.
- End-to-end healthcare FWA detection: CMS-calibrated synthetic claims, leakage-tested
  point-in-time features, expected-net-value thresholding, investigator dashboard with decision
  capture feeding delayed-label monitoring.
