# Healthcare Claims Payment-Integrity & FWA Detection Platform

[![CI](https://github.com/FaridoonFarahi/healthcare-payment-integrity/actions/workflows/ci.yml/badge.svg)](https://github.com/FaridoonFarahi/healthcare-payment-integrity/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

An end-to-end, production-style platform that scores healthcare claims for fraud, waste, and
abuse (FWA) risk — combining a rule engine, supervised ML (XGBoost), and unsupervised anomaly
detection, with SHAP-based reason codes, a financial thresholding framework, a FastAPI scoring
service, a Streamlit investigator dashboard, MLflow experiment tracking, and Evidently drift
monitoring. Fully containerized with CI/CD and Azure deployment configs.

> **Status: under active construction.** See [docs/architecture.md](docs/architecture.md) for the
> system design and [docs/roadmap.md](docs/roadmap.md) for the phased build plan. This README is
> finalized in the documentation phase.

> **Responsible-use note:** model outputs are *review priorities*, not fraud determinations.
> Supervision labels are transparently synthetic (documented in `docs/labeling.md`); no real
> provider is ever labeled or displayed as fraudulent.

## Quickstart (current state)

```bash
git clone https://github.com/FaridoonFarahi/healthcare-payment-integrity.git
cd healthcare-payment-integrity
python -m venv .venv && source .venv/bin/activate
make install        # pip install -e ".[dev]"
make test           # unit tests
make compose-up     # Postgres + MLflow via docker compose
make api            # FastAPI on :8000  ->  GET /health
```

## License

MIT — see [LICENSE](LICENSE).
