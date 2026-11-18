# Experimental results

## Protocol and evidence

This package recomputes metrics from the existing frozen `demo_data/predictions.parquet` scores and `demo_data/labels.parquet` labels. It does not train, refit, retune or replace a model. The join is one-to-one on claim ID and includes all 96,435 prediction rows. The result labels represent synthetic injected patterns, not confirmed improper payments, fraud, Medicare enforcement outcomes or CMS CERT findings.

The documented training protocol is chronological: train through 2025-03, validation through 2025-06, and later test rows; preprocessing/peer statistics are fit on train, while validation is used for calibration/selection. The frozen scored window is 2025-07-01 through 2026-01-18. The local repository lacks model/feature-store artifacts needed to reproduce training, so protocol details rely on code/configuration/docs and are not a fresh train/test execution.

## Ensemble threshold analysis

Threshold-independent metrics on the matched evaluation rows: ROC-AUC 0.948148; average precision (PR-AUC) 0.871349. Accuracy is included for completeness but is not the selection objective on this imbalanced synthetic dataset.

| Cutoff | Accuracy | Precision | Recall | F1 | TP | TN | FP | FN | False-negative rate | Review rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2000 | 0.981418 | 0.783745 | 0.837143 | 0.809564 | 3,809 | 90,834 | 1,051 | 741 | 16.2857% | 5.0397% |
| 0.5000 | 0.985451 | 0.925615 | 0.752088 | 0.829878 | 3,422 | 91,610 | 275 | 1,128 | 24.7912% | 3.8337% |
| 0.7300 | 0.984124 | 0.968643 | 0.685714 | 0.802985 | 3,120 | 91,784 | 101 | 1,430 | 31.4286% | 3.3401% |
| 0.9000 | 0.982071 | 0.981235 | 0.632088 | 0.768881 | 2,876 | 91,830 | 55 | 1,674 | 36.7912% | 3.0394% |
| 0.9945 (saved policy) | 0.963696 | 0.994345 | 0.231868 | 0.376047 | 1,055 | 91,879 | 6 | 3,495 | 76.8132% | 1.1002% |

The false-negative rate is `FN / (FN + TP)`: the share of synthetic positive examples below the selected review cutoff. In payment integrity, missed positive cases can mean potentially improper payments are not prioritized for review. The rate does not estimate real-world missed fraud or financial loss. The saved high cutoff yields very high synthetic precision at the cost of missing most labeled positives. Threshold selection is therefore an operational capacity/recall tradeoff, not accuracy maximization.

## Score-column comparison

`TABLES/SCORE_COLUMN_COMPARISON.csv` reports ranking and cutoff metrics for the saved ensemble, supervised probability, anomaly percentile and rule score, evaluated against the same matched synthetic labels. This is a score comparison on an existing frozen bundle; it is not a retrained model baseline or a real-world performance estimate. The anomaly score becomes a synthetic-label ranking evaluation here only because matching synthetic labels exist. There are no independently labeled CMS provider outcomes.

## Component checks and unavailable experiments

- Payment-indicator rules were executed on the frozen demo claims. No rules fired because the sample lacks the expected payment amount and documentation evidence fields. This is an input coverage limitation, not evidence that the rules are accurate or effective.
- Provider anomaly detection was run on the 480-row CMS aggregate snapshot and returned 73 provider-specialty groups: 6 high, 21 medium and 46 routine priority. This is unsupervised triage only; no fraud classification metrics are valid for those provider groups.
- LEIE matching accuracy was not calculated: only a six-row synthetic test fixture exists, and no official OIG file or independently verified provider identity outcomes are available.
- No genuine ablation results are reported. The saved score columns allow a same-row component score comparison, but an ablation requires controlled retraining/removal of components, which cannot be performed/reproduced without model and feature artifacts.
- No new SHAP values were calculated. The package summarizes the saved global mean-absolute-SHAP artifact separately from rule explanations.
- No real-world Medicare fraud outcome, improper-payment recovery, or external validation experiment was possible.

Machine-readable metrics and tables are in `MODEL_EVALUATION.csv`, `TABLES/`, and `EVALUATION_METADATA.json`. Figures use the same matched score/label set.
