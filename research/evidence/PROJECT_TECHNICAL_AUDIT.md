# Project technical audit

Audit generated 2026-11-18 from the checked-out repository and locally available files. This is a source/artifact audit, not a live production audit.

## Implemented capabilities

| Area | Verified implementation | Evidence and boundary |
|---|---|---|
| CMS data | CMS Physician & Other Practitioners provider-service ingestion; a 480-row local calibration snapshot; metadata and source documentation | Aggregated provider × HCPCS/payment records, not claims or adjudicated outcomes. The committed snapshot's publication year is not recorded locally. |
| OIG LEIE | Download/schema validation in `src/data/ingest_leie.py`; NPI-first screening and secondary name matching in `src/payment_integrity/leie_screening.py`; SSN columns removed from normalized evidence | No official LEIE download exists in the repository. The only local LEIE CSV is a six-row synthetic test fixture. Operational match quality was not evaluated. |
| Synthetic claims | Deterministic generator with documented F01–F12 injected patterns, data-quality defects, separate labels, and temporal feature-split configuration | The frozen demo claims/labels are synthetic. Positive means generated pattern label, not verified improper payment or fraud. |
| Claim models | Rule score, supervised probability, anomaly percentile, ensemble score, model version and reason-code fields in frozen prediction table; training code includes logistic regression, XGBoost, Isolation Forest, validation calibration and ensemble selection | Current repository has no usable trained bundle; `models/latest.json` points to an external temporary test path. Frozen scores can be re-evaluated against matching demo labels, but models cannot be reloaded/retrained from committed artifacts. |
| Threshold evaluation | `app/dashboard/model_performance.py` joins prediction and ground-truth labels one-to-one, validates coverage, and computes metrics; threshold simulator changes score cutoffs | The independently recomputed results are in `MODEL_EVALUATION.csv`. |
| Explainability | SHAP implementation, 8,700 saved per-claim top-contribution records, and a saved global mean-absolute-SHAP table/plot | Evidence is an existing saved artifact; no current model bundle is available to reproduce SHAP. Rule reasons remain distinct from SHAP contributions. |
| Payment indicators | Configurable evidence-backed checks for amounts and documentation fields | No detections on the available demo claims because these fields are absent. This does not establish rule effectiveness. |
| Provider anomaly | Specialty-aware peer aggregation, robust deviations and seeded Isolation Forest; deterministic seed default 42 | Run on 480 CMS aggregate rows to produce 73 provider-specialty groups and priority bands. There are no truth labels; bands are not fraud findings. |
| Risk aggregation | Availability-aware normalized aggregation with missing-component warnings; documented as investigation priority rather than fraud probability | Not evaluated against adjudicated outcomes. |
| Dashboard/API | Streamlit multipage dashboard and FastAPI application are present in source; dashboard has 11 pages | Repository inspection confirms page files, not a deployed or browser-verified production instance. |
| Deployment | Docker/Compose, Azure provisioning, Entra Easy Auth setup script and CI deployment guard are present | No live Azure deployment/configuration verification was performed. See remaining Phase 3 limitations in the handoff/final report. |

## Artifact integrity checks

- The 96,435 prediction claim IDs join one-to-one to separate label rows; no predictions are lost in the join. Duplicate IDs in either side would abort generation.
- Scores are finite, labels are binary and both classes are present.
- The synthetic evaluation has 4,550 positive and 91,885 negative rows (4.7183% positive).
- The CMS snapshot contains 480 rows and 14 columns and has zero exact duplicate rows. Provider identifiers are not exported into research tables or figures.
- The complete per-column dtype/missingness/unique-count audit is `TABLES/DATASET_COLUMN_QUALITY.csv`; it records schema statistics only, not row values.
- The local `models/latest.json` references a temporary path outside the project. The project model directory contains no trained model bundle or feature-store artifacts.

## Security / data handling scope

Research outputs contain only aggregate counts, metric results, schemas, and plots. No provider identifiers, beneficiary/claim identifiers, names, SSNs, credentials, or source row values are exported. Local source tables remain unmodified. The official LEIE dataset is absent, so no sensitive LEIE source was processed. The hash-locked dependency set was audited with pip-audit after updating the vulnerable MLflow, PyArrow and virtualenv versions; see `DEPENDENCY_AUDIT.md`. This does not replace a secret-scanning, live identity-control, or deployment penetration test.
