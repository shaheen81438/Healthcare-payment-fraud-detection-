# Interview Talking Points

Organized by the question you'll actually get. Each point names the artifact that backs it up.

## "Walk me through the project."

Frame in 4 beats: **(1)** business problem — payers lose ~$29B/yr to improper Medicare FFS
payments and review capacity is the scarce resource; **(2)** data reality — no public claim-level
fraud labels exist, so I calibrated a synthetic claim population to real CMS utilization data and
injected 12 documented FWA patterns, keeping labels in a separate schema; **(3)** the system —
rules + calibrated XGBoost + IsolationForest behind one score, SHAP reason codes, an
expected-net-value review policy, FastAPI + Streamlit serving, MLflow + Evidently MLOps;
**(4)** the results — 0.88 PR-AUC, 84% recall in a top-5% queue, 99% precision at the 1%-capacity
operating point, all reproducible with `make pipeline`.

## "Why should I believe your metrics?" (the leakage question)

- Temporal split with burn-in; no random row splits — duplicates and provider behavior would
  straddle folds (`configs/features.yaml`).
- Every rolling feature is strictly prior (`closed='left'`); provider features are month-lagged;
  duplicate similarity only looks backward; peer stats are fit on train and frozen as artifacts.
- The kicker: `tests/unit/test_leakage.py` *proves* it — append future months and past-claim
  features are byte-identical; corrupt val/test amounts and peer stats don't move; label columns
  are structurally rejected by the builder.
- And the metrics still aren't real-world claims — say that before they ask (synthetic-label
  ceiling, `docs/labeling.md`).

## "Why these models?"

Rules first (investigator trust + reason codes + they become model features); logistic regression
as the calibrated linear bar; XGBoost as the workhorse (scale_pos_weight ≈ 21, early stopping on
PR-AUC, isotonic calibration on a separate window); IsolationForest for schemes supervision can't
know. The ensemble weight (0.95) was chosen on validation — and I kept the anomaly leg despite it
*lowering* test PR-AUC slightly, because the **novel-scheme holdout** (drop F09/F11 from training)
shows the anomaly path recovering ~5× more unseen-scheme claims. That trade-off discussion is the
interesting part.

## "How do you handle class imbalance?"

scale_pos_weight + class weights, threshold/capacity optimization instead of resampling; accuracy
is banned as a metric — PR-AUC and recall@capacity are primary because investigators review a
budgeted queue. Calibration matters more than usual here because `P(improper) × paid` prices the
queue (Brier 0.012, ECE 0.001).

## "How does the business use the score?"

Expected net value = `p·recovery·paid − review_cost − (1−p)·fp_cost`. Policy = max of ENV-optimal
τ, capacity-implied τ, and a precision floor. All levers are configs + dashboard sliders; the
simulator shows marginal review value going negative past the justified queue size. Per-1,000
reviews framing translates directly to staffing conversations.

## "What about drift / production health?"

Frozen reference window; Evidently drift preset + my own PSI (score), JSD (provider/procedure
mix), monthly-volume deltas, missing-value tracking → ok/warn/critical with alert strings; a
simulate-drift mode proves the alarms fire (PSI 3.81 → critical). Investigator decisions feed a
delayed realized-precision report — the metric that actually erodes first in production.

## "What failed / what would you do differently?"

- F08 volume spikes: 0% claim-level recall at τ — by construction, spike claims look individually
  normal and my provider features lag a month. Fix: provider-level alerting + intra-month
  velocity features. I kept the honest number in the report instead of tuning it away.
- An exact duplicate at a clean provider scores ~0.23, not 0.9 — because I injected benign
  corrected-resubmission confounders and the model learned the difference. My first integration
  test asserted the wrong thing; the fix was testing *relative* risk (with vs without duplicate
  context). Good story about tests encoding assumptions.
- Evidently/MLflow/XGBoost major-version churn cost real time; pinning + a CI gate was the answer.
- With more time: institutional claims, graph features for collusion, randomized audit sampling
  to de-bias the future label stream, champion-challenger serving.

## "Is this ethical?"

Review-prioritization framing everywhere (never "fraud found"); human-in-the-loop with decision
capture; no protected attributes; peer-group normalization as the fairness control; FP costs
priced explicitly because provider abrasion is real; all data synthetic so no one is accused by a
demo. `docs/ethics_limitations.md` covers disparate-impact auditing and the feedback-loop risk of
training on investigator decisions.

## Rapid-fire numbers to have memorized

378k claims · 2k providers · 24 months · 4.5% injected positives · 46 features (43+3 numeric/cat
+ 2 rule) · PR-AUC: rules 0.27 / LR 0.57 / XGB 0.88 / ensemble 0.87 · recall@5% = 0.84 ·
τ = 0.9945 → precision 0.994 · Brier 0.012 · net $49k @1% capacity · ~110 tests · 13 rules ·
12 patterns.
