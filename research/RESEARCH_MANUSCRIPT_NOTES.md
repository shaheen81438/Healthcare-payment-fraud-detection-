# Manuscript notes

## Proposed title

**An Explainable Multi-Engine Framework for Healthcare Payment Integrity: Improper Payment Rules, Provider Anomaly Detection, and OIG Exclusion Screening**

## Abstract (evidence-scoped draft)

This project implements a healthcare payment-integrity decision-support prototype combining claim risk scores and SHAP explanations with transparent payment indicators, provider-peer anomaly ranking, OIG LEIE screening, and availability-aware risk aggregation. A local frozen synthetic evaluation bundle provides 96,435 matched risk-score/label rows; the labels encode injected synthetic patterns and are not verified Medicare fraud or adjudicated improper-payment outcomes. On that synthetic sample, the ensemble score achieved ROC-AUC 0.948148 and average precision 0.871349. At the saved review cutoff of 0.9945, recall was 0.231868 and the false-negative rate was 0.768132 (3,495 false negatives among 4,550 synthetic positives), illustrating the tradeoff between review volume and missed synthetic positives. A separate 480-row public CMS provider-service aggregate sample supported 73 peer anomaly review groups but had no outcome labels. No official LEIE dataset or real adjudicated claim outcomes were locally available. These results establish software capability and synthetic benchmark behavior only; external validity remains untested.

## 1. Introduction

Describe payment-integrity review as a prioritization problem and distinguish improper payment from fraud. CMS CERT is an improper-payment measurement program, not a fraud label source ([CMS CERT](https://www.cms.gov/data-research/monitoring-programs/improper-payment-measurement-programs/comprehensive-error-rate-testing-cert), [CERT background](https://www.cms.gov/data-research/monitoring-programs/improper-payment-measurement-programs/comprehensive-error-rate-testing-cert/background)). Do not claim research novelty until a systematic comparison with related work is completed. Motivation and gap: [NOT AVAILABLE — independent literature review not performed].

## 2. System design

Use `FIGURES/SYSTEM_ARCHITECTURE_EVIDENCE.png` and `docs/architecture.md`. Describe four engines: evidence-backed payment checks, peer-based provider anomaly scores, identifier-first LEIE screening, and synthetic claim scoring with rule/ML/anomaly components. Risk aggregation is a normalized investigation priority, not a calibrated probability. Note human review and availability warnings.

## 3. Data and labels

CMS public Physician & Other Practitioners provider-service data is aggregated by provider/service and used as context/calibration only. The local sample has 480 rows and 14 columns, retrieved 2026-08-06; release year is [NOT AVAILABLE] in local metadata. Synthetic claim development data has 118,229 rows and 5,537 injected-pattern positives. The frozen evaluated subset has 96,435 rows and 4,550 positives, from July 2025 through January 2026. Official OIG LEIE records used for evaluation: [NOT AVAILABLE]; only a six-record synthetic fixture is committed. Describe the F01–F12 label-generation assumptions and quality defects from `docs/labeling.md`.

## 4. Methods

Report temporal split configuration: burn-in through 2024-03, train through 2025-03, validation through 2025-06, then test. Describe preprocessing fit on training data and validation use for calibration/model/policy selection per source/config. However, current feature/model bundle is [NOT AVAILABLE], so independent reproduction of training and exact frozen predictions is not established. The current package recomputes metrics from stored scores matched one-to-one to labels; document it as artifact evaluation, not retraining.

## 5. Results

Use `MODEL_EVALUATION.csv` and `EXPERIMENTAL_RESULTS.md`. At risk threshold 0.9945: ROC-AUC 0.948148; average precision 0.871349; accuracy 0.963696; precision 0.994345; recall 0.231868; F1 0.376047; MCC 0.471166; TP 1,055; TN 91,879; FP 6; FN 3,495. Label the 76.8132% FNR explicitly as synthetic. Explain that the 0.2 threshold reduced synthetic FNR to 16.2857% at 5.0397% review rate. Do not present top-score comparisons as independently retrained model baselines.

## 6. Explainability and component outputs

The repository contains saved global mean-absolute-SHAP importance and per-claim top-contribution records (8,700 rows); these were not regenerated because model artifacts are unavailable. Rule reasons are deterministic evidence and must not be labeled SHAP. CMS anomaly run: 6 high, 21 medium and 46 routine groups, without fraud labels. Payment-rule demo run has no eligible expected-payment/documentation fields and no findings. LEIE matching tests use a fake fixture only.

## 7. Discussion

Discuss the cutoff/recall tradeoff in `EXPERIMENTAL_RESULTS.md`; this illustrates policy sensitivity on synthetic labels, not a real-world optimum. High precision at strict cutoff comes with many missed generated positives. Emphasize review-priority outputs, false-positive/negative costs, the lack of claims-level public labels, CMS aggregation and the lack of official LEIE evaluation.

## 8. Limitations, ethics, and deployment

Use `LIMITATIONS_AND_THREATS.md`. Do not accuse providers, deny claims, or quantify real Medicare fraud from these results. Validate identity and dates for exclusion matches. Production authentication/deployment status: [NOT AVAILABLE — no live Azure verification]. Fairness, prospective outcomes, operational workflow and security penetration study: [NOT AVAILABLE].

## 9. Conclusion

The repository demonstrates an implemented multi-engine software framework and a synthetic score evaluation. Evidence does not establish external validity or real-world fraud-detection efficacy. Future work needs authorized outcome data, independently reproducible fit artifacts, external/temporal validation, official LEIE data lifecycle tests, and a literature review.

## Required work before journal submission

- [NOT AVAILABLE] Independent related-work review and defensible novelty positioning.
- [NOT AVAILABLE] Authorized adjudicated claim-level outcome set and governance/ethics approvals.
- [NOT AVAILABLE] Reproducible trained model and feature artifacts, exact split manifest, and independent fit rerun.
- [NOT AVAILABLE] Prospective/external temporal and provider-level validation with uncertainty intervals and subgroup analysis.
- [NOT AVAILABLE] Operational LEIE dataset version/date handling and verified identity-match study.
- [NOT AVAILABLE] Empirically supported workflow capacity, recovery and review-cost assumptions.
