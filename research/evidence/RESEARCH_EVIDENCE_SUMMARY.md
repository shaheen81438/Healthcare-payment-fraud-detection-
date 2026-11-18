# Research evidence summary

Generated 2026-11-18 from local repository artifacts. This is a reproducible technical evidence package, not a claim of real-world model validation.

## Findings

- **Data evaluated:** 96,435 frozen claim risk scores joined one-to-one to generated synthetic labels (4,550 positive, 91,885 negative); 480 rows of real public CMS provider-service aggregates were separately used for an unlabeled anomaly-priority check. No official OIG LEIE download or adjudicated real-world claim labels were present.
- **Claim score:** frozen ensemble score column, version `claims-risk-v1.0.0`. ROC-AUC 0.948148 and average precision 0.871349 on the matched synthetic labels.
- **Saved review cutoff 0.9945:** accuracy 0.963696; precision 0.994345; recall 0.231868; F1 0.376047; TP 1,055; TN 91,879; FP 6; FN 3,495; synthetic false-negative rate 76.8132%.
- **Threshold lesson:** reducing the cutoff to 0.2 raised synthetic recall to 0.837143 and lowered FNR to 16.2857%, while increasing review rate to 5.0397% and lowering precision to 0.783745. The operating decision requires real capacity and outcome validation.
- **Other components:** the CMS aggregate anomaly run created 73 provider-specialty review groups (6 high, 21 medium, 46 routine) without outcome labels. Payment rule checks had no eligible input evidence fields in demo claims. LEIE actual-source screening was unavailable.

## Reproducible outputs

- Metric detail: `MODEL_EVALUATION.csv`
- Dataset rows/files: `DATASET_SUMMARY.csv`
- Per-column audit: `TABLES/DATASET_COLUMN_QUALITY.csv`
- Figures: `FIGURES/` (high-resolution PNG at 300 DPI)
- Component outputs: `TABLES/IMPLEMENTED_COMPONENTS_AND_RESULTS.csv`, `TABLES/CMS_PROVIDER_ANOMALY_DISTRIBUTION.csv`, `TABLES/SCORE_COLUMN_COMPARISON.csv`
- Run command and versions: `REPRODUCIBILITY.md`, `GENERATION_METADATA.json`

No `ABLATION_RESULTS.csv` is provided because no controlled component-removal retraining could be reproduced. No models or datasets were regenerated or altered.

## Submission assessment

These materials support an engineering/system description and a clearly scoped synthetic benchmark section. They are **not sufficient for a journal paper claiming validated Medicare fraud/improper-payment detection**. Before that claim, the study needs authorized adjudicated outcomes, independently reproducible model training artifacts, prospective or external temporal/provider validation, official LEIE source validation, and documented operational review costs/capacity. Novelty also requires a literature review; none is claimed here.
