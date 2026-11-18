# MedIntegrity AI Architecture

MedIntegrity AI is an investigative decision-support prototype. Its outputs are
review priorities and screening evidence, not fraud determinations.

## Core flow

```text
CMS provider-service aggregates ─┐
OIG LEIE records ────────────────┼─> validation/provenance
Synthetic claim development ────┘          │
                                          ├─> payment-integrity indicators
                                          ├─> provider peer/anomaly engine
                                          ├─> LEIE identifier/name screening
                                          ├─> supervised/anomaly claim model
                                          └─> availability-aware risk assessment
                                                       │
                                      FastAPI + Streamlit investigator views
```

The existing claim-model path remains under `src/features/`, `src/modeling/`,
and `src/evaluation/`. New independent engines are under
`src/payment_integrity/`:

- `improper_payment.py`: payment-error indicators.
- `provider_anomaly.py`: peer comparisons, robust deviation, and Isolation Forest.
- `leie_screening.py`: NPI-first and secondary name matching.
- `risk_assessment.py`: normalized aggregation with unavailable-signal warnings.

The combined score is an investigation priority. It is never presented as a
calibrated fraud probability.

Streamlit loaders cache source data for bounded intervals. Reusable evaluation,
data-quality, provider-anomaly, payment-indicator, and LEIE results use separate
caches. Investigator tables cap visible rows and preserve full downloads where
provided. Missing or corrupt artifacts produce explicit empty states rather
than placeholder metrics.

Local file provenance is available through `src/data/provenance.py`; CMS
snapshot dataset and retrieval metadata remain alongside the snapshot. Missing
input files raise an explicit error, and optional dashboard pages display a
source-unavailable state.

Successful CMS and LEIE loader calls and validated synthetic-claim builds append
metadata-only events to `monitoring/reports/data_provenance.json`. Events include
source classification, identifiers, format, counts, timestamp, and validation
status. They never contain row values or column names. The dashboard data-quality
page checks aggregates from available claim, CMS, LEIE fixture, and scored-data
artifacts. Its model-performance page calculates metrics only after joining
scored rows to matching labels; the committed demo bundle is explicitly labeled
as synthetic evaluation data.
