# MedIntegrity AI Model Card

## Intended use

Rank synthetic or authorized claim populations for human investigation and show
supporting evidence. Do not deny claims, accuse providers, or treat an exclusion
match as claim-level fraud proof.

## Existing claim models

The retained model pipeline evaluates Logistic Regression, XGBoost, Isolation
Forest, a rules baseline, and a calibrated supervised/anomaly ensemble. Existing
metrics are documented in [evaluation_report.md](evaluation_report.md) and are
based on synthetic injected patterns. The last verified repository report lists
XGBoost PR-AUC 0.881 and ROC-AUC 0.963 on its synthetic test window; these are
not real-world fraud-performance estimates.

## New evidence engines

- Payment rules return category, direction, reason code, message, and field-level
  evidence. Missing optional fields do not create a synthetic error.
- Provider anomalies compare providers only within the supplied peer group and
  combine a robust statistical baseline with Isolation Forest.
- LEIE screening exposes NPI/name match type, confidence, dates, ambiguity, and
  manual-verification status without exposing SSNs.
- Risk aggregation keeps each component score separate and reports unavailable
  signals. It returns an investigation priority, not a fraud probability.

## Limitations

Synthetic labels, aggregate CMS source data, optional LEIE availability, and
batch-oriented provider features limit external validity. Reviewers must verify
the underlying records and applicable policy before acting.

The dashboard calculates accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and
confusion counts from scored rows joined to matching labels. If those artifacts
are missing or cannot be joined, it displays “Not evaluated.” Public CMS
provider aggregates are never treated as claim-level outcomes.

The dashboard threshold simulator changes only the score cutoff used to assign
review candidates. It shows confusion counts, precision, recall, F1, review
rate, and false-negative rate at that cutoff, alongside ROC and precision-recall
curves. A false negative is a labeled synthetic positive that falls below the
selected review cutoff; the resulting rate does not estimate missed real-world
payment errors or fraud.
