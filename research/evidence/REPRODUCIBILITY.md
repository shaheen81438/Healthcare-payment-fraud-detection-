# Reproducibility guide

## Frozen evidence regeneration

The final package was regenerated with Python 3.11.9, pandas 2.3.3, NumPy 2.4.6, scikit-learn 1.9.1 and Pillow 13.0.0. An earlier cross-check under Python 3.14.7 gave the same evaluation metrics. These are evidence-generation versions, not the versions used to train the saved model.

From the repository root, install the project's Python 3.11 environment from its checked-in hash locks, then run the generator with an interpreter that has pandas, NumPy, scikit-learn, Pillow and PyArrow installed:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install --require-hashes -r requirements-dev.lock
.\.venv311\Scripts\python.exe -m pip install --no-deps -e .
.\.venv311\Scripts\python.exe scripts\generate_research_evidence.py
```

The recorded evidence run used:

```powershell
.\.venv311\Scripts\python.exe scripts\generate_research_evidence.py
```

The script reads frozen Parquet/CSV artifacts and source modules; it only writes under `research/evidence/`. It does not write source data, model artifacts, or external services. It fails closed if prediction-label IDs do not join one-to-one, if labels are non-binary, or if scores are non-finite.

## Full pipeline reproduction (not run for this package)

The project's documented commands are `make data`, `make features`, `make train`, `make evaluate`, and `make score` (or `make pipeline`). They require external/large source inputs and services/artifacts and may write data/models. They were not run in creating this package. The seven API integration tests skip because `models/feature_store/code_stats.parquet`, `data/processed/claims.parquet`, and `data/processed/provider_month_features.parquet` are absent; no substitute artifacts were created. Do not run the full pipeline when preserving frozen project artifacts is required without first selecting an isolated workspace and confirming source availability.

## Seeds and splits

- Synthetic data generation seed: 42 (`configs/data_generation.yaml`).
- Temporal split: burn-in through 2024-03, train through 2025-03, validation through 2025-06, test thereafter (`configs/features.yaml`).
- Isolation Forest and model training seeds are configured in model/source code; provider anomaly function default seed is 42.
- Training runtime, hardware, exact fitted model files and exact training environment for the frozen scores are unavailable.

## Expected outputs

`MODEL_EVALUATION.csv`, `DATASET_SUMMARY.csv`, `EVALUATION_METADATA.json`, `GENERATION_METADATA.json`, aggregate publication tables under `TABLES/`, and 300-DPI PNG images under `FIGURES/`. The script contains no network calls.
