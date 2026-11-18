# MedIntegrity AI Project Audit

Audit date: 2026-11-18  
Repository: `healthcare-payment-integrity`  
Audited commit: `8ee6ec359cbbcc104d7d8944526c620f334337b5`

## Scope and approach

This is the Phase 1 audit requested before modernization work. The audit inspected the repository tree, source packages, scripts, configuration, documentation, committed reference/demo data, tests, Docker/CI entry points, and Git metadata. No existing working component was deleted or reorganized.

The working tree was clean at the start of the audit. The repository currently points to the upstream remote `https://github.com/FaridoonFarahi/healthcare-payment-integrity.git`.

## Existing functionality

The repository is already an end-to-end payment-integrity prototype with these layers:

- `src/data/`: CMS Physician & Other Practitioners ingestion, OIG LEIE ingestion, reference-table loading, synthetic claim generation, validation, cleaning, and Postgres/parquet persistence.
- `src/features/`: temporal splits, backward-looking duplicate similarity, claim features, provider rolling features, and specialty/code peer statistics.
- `src/modeling/`: rule engine, preprocessing, Logistic Regression, XGBoost, Isolation Forest, calibration/ensemble bundle, model registry integration, and SHAP explanation support.
- `src/evaluation/`: ranking/threshold/calibration metrics, financial thresholding, plots, and Markdown report generation.
- `src/monitoring/`: Evidently-based drift reporting plus PSI/Jensen-Shannon/missingness/volume checks and delayed investigator-outcome metrics.
- `app/api/`: FastAPI health, model-info, metrics, prediction, batch prediction, and explanation endpoints with Pydantic validation and request logging.
- `app/dashboard/`: Streamlit multipage investigator workflow with portfolio overview, review queue, claim detail, provider risk, threshold simulation, and monitoring.
- `scripts/`: dataset build, feature build, training, evaluation, scoring, monitoring, model-quality gate, novel-scheme experiment, and demo-bundle generation.
- Infrastructure: Dockerfiles, Docker Compose with Postgres/MLflow/API/dashboard services, Azure deployment scripts, Make targets, GitHub CI/deploy workflows, and a committed Streamlit demo bundle.

## Existing models and analysis

The current training module implements:

1. A deterministic rule baseline with 13 configured rules (`R001`-`R013`).
2. Balanced Logistic Regression.
3. XGBoost classification with class-imbalance weighting and early stopping.
4. Isolation Forest over behavioral claim features.
5. Isotonic calibration and a validation-selected supervised/anomaly ensemble.
6. TreeExplainer-based SHAP contributions, aggregated reason codes, and global importance output.

The feature pipeline has explicit temporal protections: a burn-in period, train/validation/test temporal windows, frozen peer statistics, lagged provider features, backward-only duplicate matching, and rejection of label columns from the feature matrix. These protections are covered by dedicated leakage tests.

The current implementation is primarily a claim-level FWA/risk-ranking system. It does not yet provide the requested separate improper-payment category engine, separate LEIE matching/screening service, or a provider-level Isolation Forest/anomaly model as distinct modules. Provider-level peer and rolling features do exist and are used in claim scoring and dashboard aggregation.

## Existing datasets and provenance

### Real public context

- A committed 480-row CMS Medicare Physician & Other Practitioners provider-service calibration snapshot covers 12 specialties. Its source URL, dataset UUID, retrieval date (`2026-08-06`), filters, columns, and limitations are recorded in `data/reference/cms_snapshot_metadata.json`.
- `src/data/ingest_cms.py` supports the CMS data API and explicitly describes the data as provider-level aggregates, not adjudicated claims.
- `src/data/ingest_leie.py` supports downloading the official OIG LEIE CSV, but the repository’s default and committed test path is a clearly synthetic fixture at `tests/fixtures/leie_fixture.csv`. No official LEIE data file is committed.
- Reference tables include POS codes, E/M levels, diagnosis groups, diagnosis/procedure compatibility, and an explicitly illustrative PTP-pair file.

### Synthetic data

- The generator creates synthetic claims, providers, beneficiaries, and labels using configuration in `configs/data_generation.yaml`.
- Twelve documented injected pattern families (`F01`-`F12`) produce synthetic supervision; labels are kept separately from claim facts.
- The committed `demo_data/` bundle contains sampled claims, predictions, explanations, labels, peer stats, and policy artifacts for the dashboard fallback.
- Documentation consistently states that claim-level metrics measure detection of documented synthetic injections, not real-world fraud or adjudicated improper payments.

### Privacy and sensitive data

The committed synthetic identifiers use reserved project prefixes. The checked-in documentation states that no names, addresses, SSNs, PHI, or real beneficiaries are included. Raw/interim/processed data and model artifacts are ignored by Git, while small reference/demo artifacts are intentionally committed.

## Existing evaluation results

`docs/evaluation_report.md`, `docs/model_card.md`, `docs/assets/`, and the README contain a generated evaluation for a six-month synthetic test window of 96,435 claims with a 4.72% injected-improper rate. Reported test results include:

| Model | PR-AUC | ROC-AUC | Recall@1% | Recall@5% | Recall@10% |
|---|---:|---:|---:|---:|---:|
| Rules baseline | 0.273 | 0.722 | 0.123 | 0.409 | 0.500 |
| Logistic Regression | 0.571 | 0.910 | 0.170 | 0.607 | 0.755 |
| XGBoost | 0.881 | 0.963 | 0.212 | 0.836 | 0.907 |
| Isolation Forest | 0.316 | 0.837 | 0.125 | 0.372 | 0.550 |
| Ensemble | 0.871 | 0.948 | 0.211 | 0.835 | 0.901 |

The report also contains calibration, capacity/net-value, precision-at-capacity, confusion-matrix, PR-curve, reliability, score-distribution, SHAP, and novel-scheme experiment artifacts. These figures are existing repository outputs; they were not regenerated during this audit because the local environment lacks the project dependencies.

## Baseline verification

### Passed

- `py -m compileall -q src app scripts tests`: passed.
- `py -m pytest tests/unit/test_config.py -vv`: 4 tests passed, 1 deprecation warning.

### Blocked or incomplete

- The project-local `.venv` contains only `pip`; `\.venv\Scripts\python.exe -m pytest` fails with `No module named pytest`.
- Running `py -m pytest tests/unit -m "not slow"` under the available system Python 3.14.7 fails during collection for seven modules because `sqlalchemy` is not installed. The remaining tests were not a valid full baseline because collection stopped.
- A top-level `py -m pytest` attempt did not produce a usable completion result in the available environment and was stopped after it failed to progress. This is recorded as an environment/test-execution issue, not as a code failure.
- Streamlit skill discovery confirms Streamlit is not installed in the detected project environment. The declared dependency is present in `pyproject.toml` and `requirements.txt`, but the environment must be provisioned before dashboard execution or rendering validation.

## Technical problems and modernization gaps

### High priority

1. **Environment reproducibility is not currently usable from the checked-in `.venv`.** Dependencies are declared, but the local environment is empty and Python 3.14 is the only registered interpreter. The project declares Python `>=3.11`; compatibility with 3.14 and a supported 3.11/3.12 environment should be validated after installation.
2. **The requested new identity and attribution layer is absent.** There is no `ATTRIBUTION.md`, and the current README/project metadata retain the upstream name and branding. The MIT license is present and must be preserved while documenting retained/adapted upstream work and original contributions.
3. **The current dashboard does not match the requested target.** It has five pages, uses Altair rather than Plotly, and lacks dedicated Executive Overview, Improper Payment Analysis, OIG Exclusion Screening, Explainability, Model Performance, and Data Quality pages as separately specified. Some equivalent information is currently combined into existing pages.
4. **LEIE is ingestion-only.** There is no implemented NPI/name normalization and cautious fuzzy matching service with confidence, ambiguity, exclusion/reinstatement detail, and manual-verification status.
5. **Improper payment is not modeled as a separate analysis domain.** Current rules and labels represent synthetic FWA patterns and risk ranking. The requested configurable categories (documentation, coding, medical necessity, amount error, overpayment/underpayment) and rule evidence model are not present as a dedicated module.

### Medium priority

6. **Provider analytics are mostly aggregations of claim predictions.** Peer-group and rolling features are present, but the requested independent provider anomaly engine, provider-level Isolation Forest comparison, provider anomaly evidence contract, and review-priority output should be made explicit.
7. **The current evaluation is documented and appears generated, but the audit could not reproduce it without dependencies and supporting services.** A clean-environment rerun should produce reports from the current code and update documentation only from generated outputs.
8. **The project has no dedicated `reports/` output directory matching the requested structure.** Evaluation charts are under `docs/assets/`, monitoring output is under `monitoring/reports/`, and scored artifacts are under ignored `data/processed/`.
9. **The current architecture is coherent but differs from the proposed target tree.** It uses `app/`, `src/`, `configs/`, `scripts/`, and `demo_data/`; any reorganization should be incremental and preserve API, dashboard, and pipeline entry points.
10. **Data validation/provenance is stronger for synthetic/CMS paths than for a full source registry.** Source URLs and CMS snapshot metadata exist, but a unified provenance record for CMS, LEIE downloads, synthetic generation, versions, hashes, and validation outcomes should be added.

## Proposed improvements

1. Establish a supported, reproducible environment and CI matrix before changing behavior; install dependencies in a fresh environment and obtain a complete baseline test/lint/type-check result.
2. Add the independent MedIntegrity AI identity, README, `ATTRIBUTION.md`, source registry, limitations, and responsible-use language while retaining the MIT license and upstream notices.
3. Add a validated data-contract/provenance layer shared by CMS, LEIE, and synthetic loaders, including schema checks, duplicate/invalid-value reports, source metadata, and clear unavailable-source errors.
4. Implement a configurable improper-payment rules package with category, direction (potential overpayment/underpayment), trigger evidence, and synthetic/real provenance fields.
5. Implement a dedicated LEIE screening package with identifier-first matching, normalized names, secondary fuzzy matching, ambiguity handling, date details, and manual verification states; keep exclusion matches separate from fraud risk.
6. Formalize provider anomaly detection with specialty-aware peer groups, transparent statistical baselines, Isolation Forest comparison, evidence payloads, and a provider-level evaluation protocol.
7. Preserve the existing leakage-safe training design while making model protocol, threshold selection, calibration, feature importance, and generated evaluation outputs reproducible and machine-readable.
8. Build the requested dashboard pages around shared data services, Plotly charts, actual-data KPIs, readable tables, downloads, optional-data empty states, and a clearly labeled synthetic demo mode.
9. Add focused tests for the new source validation, improper-payment categories, LEIE matching, provider anomaly outputs, risk aggregation, dashboard loaders, and provenance contracts before reorganizing files.
10. Update README metrics only from a verified current evaluation run; use “Not yet evaluated” when an experiment has not been rerun.

## Components retained, modified, and newly developed

### Retain with minimal changes

- MIT `LICENSE` and existing responsible-use principles.
- CMS snapshot/API loader and provenance metadata.
- Synthetic generator, documented pattern injectors, separated labels, and leakage tests.
- Temporal feature engineering, peer statistics, validation/cleaning, rules baseline, model bundle, SHAP machinery, monitoring utilities, API contracts, Docker/Make entry points, and existing demo artifacts.

### Modify

- Project branding and package metadata.
- README and documentation index.
- Provenance and validation contracts.
- CMS/LEIE loader error handling and source metadata.
- Provider analytics into a first-class provider anomaly API.
- Evaluation/report outputs and generated metric consumption.
- Streamlit navigation, visual system, charting, page coverage, empty states, downloads, and synthetic-data disclosures.
- Test environment and CI compatibility.

### Newly develop

- `ATTRIBUTION.md` and independent-contribution record.
- Dedicated improper-payment analysis/rules module and configurable category definitions.
- Dedicated LEIE exclusion-screening/matching module.
- Transparent risk aggregation service with component availability and data-quality warnings.
- Provider anomaly result contract and comparison evaluation.
- Unified provenance/source registry and validation report model.
- Requested dashboard pages and dashboard data-loading tests.
- Final generated `reports/` evaluation artifacts if retained as the project’s canonical report location.

## Phase 2 handoff

Before implementation begins, the concrete modification plan should preserve the current pipeline/API/dashboard entry points and sequence changes as follows: environment baseline; attribution/identity; source contracts and provenance; improper-payment and LEIE modules; provider anomaly/risk aggregation; model/evaluation updates; dashboard redesign; tests; clean pipeline/evaluation; final documentation.

No destructive changes were made during this audit.
