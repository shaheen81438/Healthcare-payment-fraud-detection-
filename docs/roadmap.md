# Implementation roadmap

Thirteen sequential phases, each shipping runnable code, tests, and docs. Exit criteria are the
definition of done. Live status was tracked in the build session; this file records the plan.

| # | Phase | Deliverables | Exit criteria |
|---|-------|------------------|---------------|
| 1 | Architecture & decisions | `docs/architecture.md`, `docs/roadmap.md` | Dataset strategy, stack, and deploy target locked |
| 2 | Scaffold | Tooling, settings, JSON logging, health API, compose (Postgres+MLflow), CI skeleton | `ruff` + `mypy` + `pytest` green; `/health` serves |
| 3 | Data layer | CMS/LEIE ingestion + committed reference snapshot, reference code sets, synthetic generator with injected F01-F12 patterns, separated `labels` schema, validation+quarantine, cleaning, Postgres load, `data_dictionary.md`, `labeling.md` | One command rebuilds the dataset deterministically; validation report emitted; unit tests pass |
| 4 | Features | Claim + provider point-in-time features, duplicate similarity, frozen peer stats, temporal splits | Feature matrix built; leakage tests pass |
| 5 | Models | Rule engine, LogReg, XGBoost, IsolationForest, calibrated ensemble; MLflow tracking; novel-scheme holdout | Models trained/logged/compared on identical splits |
| 6 | Evaluation & thresholds | Metrics suite, financial ENV framework, capacity optimizer, evaluation report, CI quality gate | Report generated with ranking + financial metrics; gate wired |
| 7 | Explainability | SHAP pipeline, reason-code mapping, per-claim explanations to Postgres, global plots | Every scored claim has reasons + contributions |
| 8 | API | All six endpoints, validation, errors, batch limits, model loading w/ registry fallback | Unit + integration tests green; OpenAPI docs complete |
| 9 | Dashboard | Six investigator pages incl. queue, claim detail, simulator, decision capture | End-to-end review workflow usable against scored data |
| 10 | Registry & monitoring | Production alias + loader, Evidently reports, alert thresholds, drift simulation, delayed-label report | Drift run produces HTML/JSON + status row; alerts trigger on simulated drift |
| 11 | Docker, CI/CD, Azure | Dashboard image, full pipeline (incl. integration + gate + smoke), ACR/ACA deploy scripts + guide | CI pipeline complete; deploy job gated on secrets |
| 12 | Tests & docs | Coverage pass; README, model card, eval report, deployment/API/monitoring guides, ethics, demo script, resume bullets, interview talking points, screenshots | All 13 documentation deliverables present |
| 13 | Production-readiness review | Fresh-clone verification, security/PII pass, debt register, final bundle | Clean-machine quickstart works as documented |

## Risk register

| Risk | Mitigation |
|---|---|
| Public CMS data lacks claim-level granularity | Calibrated synthetic claims + explicit labeling docs (accepted, by design) |
| Synthetic labels overstate separability | Benign confounders injected; results framed as pattern-detection on documented injections; novel-scheme holdout quantifies generalization |
| Sandbox build env blocks container registries | Docker builds verified in GitHub Actions CI; local pipeline verified against native Postgres |
| Library API churn (Evidently 0.7, MLflow 2.x, XGBoost 3.x) | Exact working versions captured in `requirements.txt` bounds + CI |
| LEIE↔utilization joins are sparse/biased | Positioned as context + extension, not primary labels |
| SHAP latency on large batches | Precompute during batch scoring; on-demand only for single claims |

## Deferred / technical debt (living list)

- Alembic migrations (schema managed by bootstrap SQL + SQLAlchemy metadata)
- AuthN/AuthZ on API + dashboard (out of scope for portfolio demo; noted in deployment guide)
- Real NCCI PTP full-file loader is provided, but the committed pair list is illustrative
- Provider-level LEIE weak-label study (loader + docs shipped; full study future work)
- Streaming/near-real-time feature refresh (batch cadence assumed)
