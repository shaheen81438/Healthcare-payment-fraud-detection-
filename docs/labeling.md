# Labeling methodology and assumptions

## Why labels are synthetic

Adjudicated claim-level fraud/overpayment outcomes are not publicly available: CMS claim-level
research files are access-restricted, and enforcement outcomes (DOJ/OIG cases) are not linkable to
claim lines. **Every supervision label in this project is therefore generated**, by injecting
documented FWA patterns into synthetic claims that are calibrated to real CMS payment and
utilization distributions. This has three consequences:

1. Reported metrics measure **detection of documented synthetic patterns**, not real-world fraud
   discovery. Real-world performance would require adjudicated outcomes and would be lower.
2. No real provider or beneficiary is, or can be, implicated. The scored population is entirely
   synthetic (`PRV-…`, `BEN-…`, `CLM-…`), and real NPIs appear only in aggregate benchmark tables.
3. Labels live in a separate schema (`labels.claim_labels`) and never enter feature space.

The OIG LEIE exclusion list is the closest real label source (provider-level). It ships with a
loader and a derived `fraud_related` flag, but is used as context, not supervision, because:
NPIs are sparsely populated historically, excluded providers largely vanish from utilization data
(survivorship bias), and exclusion dates rarely overlap observed billing windows.

## Generation principles

- Behavior-only injection. Injectors change only fields a bad actor could influence (codes,
  units, amounts, dates, POS, diagnosis). No marker columns, no systematic artifacts (e.g., all
  amounts still carry the same noise process), so labels are recoverable only through behavior.
- Benign confounders are injected with label 0 to create realistic false-positive pressure:
  corrected-claim resubmissions (~0.6%), legitimately expensive claims (~0.3%), and PTP component
  billing **with** modifier 59 (~0.2%), which legitimately bypasses bundling edits.
- Data-quality defects (missing POS, `$`-string amounts, impossible dates, same-ID ingestion
  duplicates, negative amounts, truncated codes) are injected only into the raw file, so the
  validation and cleaning layers face real work. Same-ID duplicates are *artifacts*; F01 fraud
  duplicates carry **different** claim IDs and must survive cleaning.
- Determinism. The same config and seed produce a byte-identical dataset (tested).

## Pattern catalog (v1 defaults; all rates configurable in `configs/data_generation.yaml`)

| ID | Behavior injected | Improper amount definition | Approx. share of claims |
|---|---|---|---|
| F01 exact duplicate | Resubmission of an existing claim under a new claim ID, submitted 3-30 days later | Full paid amount of the duplicate | 0.5% |
| F02 near duplicate | Duplicate with evasion tweaks: ±2% amounts, 50% dx swap, 30% modifier-59 added | Full paid amount | 0.5% |
| F03 rapid repeat | 4-8 repeats of a service at 1-5-day gaps per episode, 2-4 episodes per flagged provider | Paid amount of each inserted repeat | ~0.2% |
| F04 excessive units | Units ×3-8 on unit-scalable codes (therapy/drug/lab) | Paid delta vs original units | 0.3% |
| F05 amount inflation | 40% of a flagged provider's claims re-priced ×1.8-2.8 | Paid − paid/multiplier | ~1.0% |
| F06 upcoding | 99212/99213 shifted 1-2 E/M levels up (70/30), re-priced by real relative payments | Paid delta vs original level | ~0.1% |
| F07 unbundling | PTP component added same day **without** modifier 59 | Component paid amount | 0.3% |
| F08 volume spike | 1-2 months at ×2-4 volume; 60% of spike claims to out-of-panel beneficiaries | Paid amount of spike claims | ~0.2% |
| F09 dx mismatch | Diagnosis group replaced with clinically incompatible group (compat ≤ 0.1), non-E/M services | Full paid (medical-necessity proxy) | ~0.2% |
| F10 POS anomaly | Facility-typical claims re-sited to office/home POS | 30% of paid (site differential proxy) | 0.25% |
| F11 collusion cluster | 3 clusters of 3-4 providers reciprocally billing a shared bene pool | Paid amount of cluster claims | ~0.4% |
| F12 code concentration | 50-70% of a provider's claims shifted to their most lucrative code | 25% of paid (excess-mix proxy) | ~0.7% |

Total positive rate ≈ **4.5%** of claims (~10% of providers exhibit ≥1 pattern). That is the same
order of magnitude as measured improper-payment rates in Medicare FFS (CMS CERT program), though
CERT "improper" also includes documentation errors, not only abuse.

## Known limitations of the labeling scheme

- Pattern prevalences and improper-amount proxies (F10's 30%, F12's 25%) are assumptions, not
  measurements; they are config-exposed so sensitivity can be tested.
- Injected behaviors are cleaner than reality, since real schemes drift, mix, and adapt. The
  novel-scheme holdout experiment (Phase 5) measures part of this gap but does not close it.
- Claim lines are single-procedure; institutional claim structures (revenue codes, DRGs) are out
  of scope for v1.
- Because labels are synthetic, **calibration to real-world base rates is impossible**. The
  financial framework therefore treats recovery rates and costs as user-adjustable scenario
  parameters, never as validated facts.
