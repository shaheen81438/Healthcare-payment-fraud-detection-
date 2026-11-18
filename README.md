# Healthcare Payment Integrity Using Explainable AI

An Explainable Multi-Engine Framework for Healthcare Payment Integrity: Improper Payment Rules, Provider Anomaly Detection, and OIG Exclusion Screening.

## Overview

Healthcare payment integrity is an important challenge involving the identification of potentially improper payments, unusual provider billing patterns, and compliance risks.

This project presents an AI-powered healthcare payment integrity framework that combines multiple analytical engines to support payment review and risk prioritization.

The framework integrates rule-based detection, machine learning-based anomaly detection, exclusion screening, and explainable risk aggregation into a unified application.

**Important:** This system is designed to support healthcare payment review and compliance analysis. Its outputs are risk indicators and should not be interpreted as confirmed fraud findings.

## Key Features

- **Improper Payment Detection:** Rule-based analysis to identify potentially improper payment patterns.
- **Provider Anomaly Detection:** Identifies unusual provider-specialty patterns using anomaly detection techniques.
- **OIG LEIE Screening:** Supports screening against exclusion-related records.
- **Risk Aggregation:** Combines analytical signals to prioritize cases for human review.
- **Explainable AI:** Provides interpretable risk indicators and supporting explanations.
- **Interactive Dashboard:** Streamlit-based interface for exploring payment integrity results.
- **Data Quality and Provenance:** Supports dataset auditing, missing-data assessment, and source documentation.

## System Architecture

The framework consists of four primary analytical engines:

1. Improper Payment Detection Engine
2. Provider Anomaly Detection Engine
3. OIG LEIE Screening Engine
4. Risk Aggregation and Explainability Engine

These components contribute analytical signals to a unified review workflow.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Anomaly Detection | Isolation Forest |
| Explainable AI | SHAP |
| Database and Storage | SQLAlchemy, DuckDB |
| Dashboard | Streamlit, Plotly |
| API | FastAPI |
| Testing | Pytest |
| Code Quality | Ruff |

## Model Evaluation

The ensemble was evaluated using a frozen demonstration dataset containing generated synthetic labels.

| Metric | Result |
|---|---:|
| Evaluation records | 96,435 |
| Synthetic positive labels | 4,550 |
| ROC-AUC | 0.948148 |
| Average Precision | 0.871349 |

### Threshold Analysis

The following results demonstrate the effect of changing the decision threshold.

| Metric | Threshold 0.9945 | Threshold 0.2 |
|---|---:|---:|
| Recall | 0.231868 | 0.837143 |
| False Negatives | 3,495 | 741 |
| Review Rate | Not reported | 5.0397% |
| Precision | Not reported | 0.783745 |

The lower threshold improves recall on the synthetic benchmark while increasing the number of records requiring review.

**Research interpretation:** These results represent performance against generated synthetic labels. They do not demonstrate validated performance on real Medicare claims or confirmed fraud cases.

## CMS Provider Anomaly Analysis

The project also includes an analysis using a 480-row CMS provider-service aggregate dataset.

The anomaly detection run generated:

- 73 provider-specialty review groups.
- No verified fraud outcome labels.
- Provider-level aggregate analysis rather than claim-level fraud classification.

CMS aggregate data should not be treated as claim-level ground truth.

## Dataset Information

The research implementation uses different types of data:

- Synthetic demonstration data for model development and evaluation.
- CMS provider-service aggregate data for provider anomaly analysis.
- A six-row synthetic LEIE fixture for testing exclusion-screening functionality.

The repository does not currently contain an official OIG LEIE download.

For operational screening, obtain the latest official exclusion data and account for exclusion dates, reinstatements, and matching ambiguity.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install project dependencies using the repository's dependency files:

```bash
pip install -e .
```

If development and testing dependencies are required:

```bash
pip install -r requirements-dev.txt
```

## Run the Streamlit Application

Start the dashboard:

```bash
streamlit run app/dashboard/Home.py --server.port 8501
```

Open:

```text
http://localhost:8501
```

## Run the API

In a separate terminal:

```bash
uvicorn app.api.main:app --reload --port 8000
```

## Run Tests

Execute the automated test suite:

```bash
python -m pytest
```

Run code quality checks:

```bash
ruff check .
ruff format --check .
```

## Research Evidence

The project includes a dedicated research evidence package.

```text
research/
│
├── evidence/
│   ├── RESEARCH_EVIDENCE_SUMMARY.md
│   ├── PROJECT_TECHNICAL_AUDIT.md
│   ├── DATASET_AUDIT.md
│   ├── EXPERIMENTAL_RESULTS.md
│   ├── MODEL_EVALUATION.csv
│   ├── DATASET_SUMMARY.csv
│   ├── REPRODUCIBILITY.md
│   ├── LIMITATIONS_AND_THREATS.md
│   ├── DEPENDENCY_AUDIT.md
│   ├── FIGURES/
│   └── TABLES/
│
└── RESEARCH_MANUSCRIPT_NOTES.md
```

The evidence package documents the experimental results, dataset characteristics, technical implementation, reproducibility information, and research limitations.

## Verification

The reported verification results include:

- 119 tests passed.
- 7 tests skipped because required feature-store and processed-claim artifacts were unavailable.
- Ruff checks and formatting passed.
- `git diff --check` passed.
- Dependency audit reported no known vulnerabilities in the regenerated development dependencies.
- Streamlit started locally and returned HTTP 200.

Live Azure protection, Docker image building, and browser-based narrow-viewport verification have not been confirmed.

## Limitations

This project has several important research limitations:

1. Synthetic labels are not verified Medicare payment outcomes.
2. CMS provider-service data is aggregated and does not provide claim-level fraud ground truth.
3. The current LEIE fixture is synthetic and is not a substitute for the official OIG exclusion database.
4. Provider anomalies are indicators for further investigation, not proof of fraud.
5. Improper payments are not necessarily fraudulent payments.
6. Real-world operational performance requires validation using appropriate verified data and independent evaluation.

## Future Enhancements

- Integration of regularly updated official OIG LEIE data.
- Validation using suitable real-world healthcare payment datasets.
- Controlled baseline and ablation experiments.
- Improved threshold selection and review prioritization.
- Further explainability and model monitoring.
- Production deployment and security validation.
- Expanded data provenance and audit capabilities.

## Ethical Considerations

The system is intended to assist analysts and compliance teams. Automated risk scores should not independently determine fraud, deny payments, or penalize healthcare providers.

Human review, appropriate data governance, privacy protection, and independent validation are essential before operational deployment.

## License and Attribution

This project builds upon an upstream MIT-licensed repository.

Please retain the original MIT license and appropriate attribution when distributing modified versions. Refer to the repository's `LICENSE` file for licensing details.

## Disclaimer

This project is developed for academic research, experimentation, and educational purposes. It is not an officially validated Medicare fraud detection system and should not be used as the sole basis for payment, compliance, or enforcement decisions.

---

**Research Focus:** Explainable AI | Healthcare Analytics | Payment Integrity | Anomaly Detection | Machine Learning | Risk Prioritization
