# Ethics, responsible use, and limitations

## Review priorities, not accusations

This system produces review priorities. Fraud is a legal determination made by adjudicators and
courts, and waste and abuse determinations belong to trained investigators. So scores route human
attention, decisions are captured from humans, and no output is worded as a finding. The API
description, dashboard banner, and reports all repeat this on purpose.

## Synthetic labels

Public, adjudicated, claim-level fraud labels do not exist. This project **generates** its labels
by injecting documented FWA patterns into synthetic claims calibrated to real CMS distributions.
That means:

- Reported metrics (PR-AUC 0.88, recall@5% 0.84) measure detection of *documented injections*
  under *known prevalence*. Real-world performance would be lower and cannot be known in advance.
- Injected schemes are cleaner than real ones, which drift, blend, and adapt adversarially. The
  novel-scheme holdout quantifies part of this gap (unseen-scheme recall drops to 2-18%).
- Financial figures (net value, $/1,000 reviews) are scenario arithmetic on configurable
  assumptions, not projections.
- Nothing here should ever be quoted as evidence about real providers, real specialties, or real
  fraud rates. Real CMS data appears only as aggregate benchmarks; no real NPI is ever scored.

## Privacy

There is no PHI anywhere in the platform: beneficiaries, providers, and claims are synthetic with
synthetic identifiers (`BEN-…`, `PRV-…`, `CLM-…`). The real CMS file used for calibration is a
public, provider-level aggregate with CMS's own small-cell suppression; provider names/addresses
are dropped at ingestion. The LEIE test fixture uses reserved fake NPIs. A production deployment
would add PHI-scoped access controls, audit logging, minimum-necessary views for investigators,
and de-identification in monitoring artifacts.

## Fairness and disparate impact

- No protected attributes (race, ethnicity, sex, age beyond coarse bands in the synthetic dim
  tables) are features.
- Specialty and geography do matter to the model (billing norms differ) and enter through peer
  normalization. Comparing providers to *their own* peer group instead of a global norm is the main
  fairness control against penalizing specialties with legitimately high costs.
- Residual risks a production system must audit: peer groups with small n produce noisier
  deviations (mitigated by support floors in peer stats); newly-enrolled providers score with
  cold-start defaults (surfaced via `insufficient_provider_history`, never auto-flagged);
  review-rate parity across specialties/geography should be monitored the same way drift is.
- False positives carry real costs (provider abrasion, payment delays, appeals). That is why the
  ENV framework prices them explicitly and the risk floor applies even with spare capacity.

## Feedback-loop risk

Training tomorrow's model on today's investigator decisions imports today's targeting bias
(claims never reviewed never generate labels). The platform keeps injected labels and decision
outcomes in separate stores; a production retraining design would need randomized audit samples
(reviewing some low-score claims) to keep the label space unbiased. Documented as future work.

## Technical limitations

Only single-line professional claims are covered (no institutional claims, revenue codes, DRGs, or claim-line
hierarchies). The monthly provider-feature lag means intra-month behavior shifts surface late (see
F08 recall). The NCCI pair list is illustrative (a real full-file loader is provided but not
committed). The demo services have no authN/authZ, API metrics are in-process, features refresh in
batch only, and schema management has no Alembic. Each item is tracked in the roadmap's debt
register.
