# Ethics, Responsible Use & Limitations

## The core framing

This system produces **review priorities, not accusations**. Fraud is a legal determination made
by adjudicators and courts; waste and abuse determinations belong to trained investigators.
Every design choice follows from that: scores route human attention, decisions are captured from
humans, and no output is worded as a finding. The API description, dashboard banner, and reports
all repeat this framing on purpose.

## Synthetic labels — the honest boundary

Public, adjudicated, claim-level fraud labels do not exist. This project **generates** its labels
by injecting documented FWA patterns into synthetic claims calibrated to real CMS distributions.
Consequences, stated plainly:

- Reported metrics (PR-AUC 0.88, recall@5% 0.84) measure detection of *documented injections*
  under *known prevalence*. Real-world performance would be lower and unknowable in advance.
- Injected schemes are cleaner than real ones, which drift, blend, and adapt adversarially. The
  novel-scheme holdout quantifies part of this gap (unseen-scheme recall drops to 2–18%).
- Financial figures (net value, $/1,000 reviews) are **scenario arithmetic** on configurable
  assumptions, not projections.
- Nothing here should ever be quoted as evidence about real providers, real specialties, or real
  fraud rates. Real CMS data appears only as aggregate benchmarks; no real NPI is ever scored.

## Privacy

There is no PHI anywhere in the platform: beneficiaries, providers, and claims are synthetic with
synthetic identifiers (`BEN-…`, `PRV-…`, `CLM-…`). The real CMS file used for calibration is a
public, provider-level aggregate with CMS's own small-cell suppression; provider names/addresses
are dropped at ingestion. The LEIE test fixture uses reserved fake NPIs. A production deployment
would add: PHI-scoped access controls, audit logging, minimum-necessary views for investigators,
and de-identification in monitoring artifacts.

## Fairness & disparate impact

- **No protected attributes** (race, ethnicity, sex, age beyond coarse bands in the synthetic dim
  tables) are features.
- Specialty and geography ARE model-relevant (billing norms genuinely differ) and enter through
  peer normalization — comparing providers to *their own* peer group rather than a global norm is
  itself the primary fairness control against penalizing specialties with legitimately high costs.
- Residual risks a production system must audit: peer groups with small n produce noisier
  deviations (mitigated by support floors in peer stats); newly-enrolled providers score with
  cold-start defaults (surfaced via `insufficient_provider_history`, never auto-flagged);
  review-rate parity across specialties/geography should be monitored the same way drift is.
- False positives carry real costs — provider abrasion, payment delays, appeals — which is why
  the ENV framework prices them explicitly and the risk floor exists even under spare capacity.

## Feedback-loop risk

Training tomorrow's model on today's investigator decisions imports today's targeting bias
(claims never reviewed never generate labels). The platform keeps injected labels and decision
outcomes in separate stores; a production retraining design would need randomized audit samples
(reviewing some low-score claims) to keep the label space unbiased. Documented as future work.

## Technical limitations

Single-line professional claims (no institutional claims, revenue codes, DRGs, or claim-line
hierarchies) · monthly provider-feature lag means intra-month behavior shifts surface late (see
F08 recall) · illustrative NCCI pair list (real full-file loader provided but not committed) ·
no authN/authZ on the demo services · in-process API metrics · batch feature refresh only ·
Alembic-less schema management. Each is tracked in the roadmap's debt register.
