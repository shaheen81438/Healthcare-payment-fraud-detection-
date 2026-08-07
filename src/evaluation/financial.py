"""Financial value framework: expected net value, capacity policies, threshold selection.

    Expected net value =
        expected recoverable payment
        − investigation cost
        − false-positive operational cost

Two views are computed everywhere:

* **Expected** — uses the calibrated probability (available in production).
* **Realized** — uses ground-truth labels + injected improper amounts
  (available here because labels are synthetic; in production this arrives
  later from investigator outcomes).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CostParams:
    review_cost: float
    recovery_rate: float
    fp_extra_cost: float
    min_risk_threshold: float
    monitor_fraction: float


def load_cost_params(path: str = "configs/costs.yaml") -> tuple[CostParams, dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)
    params = CostParams(
        review_cost=float(raw["review_cost"]),
        recovery_rate=float(raw["recovery_rate"]),
        fp_extra_cost=float(raw["fp_extra_cost"]),
        min_risk_threshold=float(raw["min_risk_threshold"]),
        monitor_fraction=float(raw["monitor_fraction"]),
    )
    return params, raw


def expected_value_per_claim(prob: np.ndarray, paid: np.ndarray, params: CostParams) -> np.ndarray:
    """Expected net value of reviewing each claim, from calibrated probabilities."""
    return (
        prob * params.recovery_rate * paid
        - params.review_cost
        - (1.0 - prob) * params.fp_extra_cost
    )


def realized_review_outcome(
    flagged: np.ndarray,
    y_true: np.ndarray,
    improper_amount: np.ndarray,
    params: CostParams,
) -> dict[str, float]:
    """Ground-truth financials for a review policy (which claims were flagged)."""
    flagged = flagged.astype(bool)
    n_reviewed = int(flagged.sum())
    tp_mask = flagged & (y_true == 1)
    fp = int((flagged & (y_true == 0)).sum())
    recovered = float(params.recovery_rate * improper_amount[tp_mask].sum())
    review_cost_total = float(params.review_cost * n_reviewed)
    fp_cost_total = float(params.fp_extra_cost * fp)
    net = recovered - review_cost_total - fp_cost_total
    return {
        "n_reviewed": n_reviewed,
        "true_positives": int(tp_mask.sum()),
        "false_positives": fp,
        "precision": float(tp_mask.sum() / n_reviewed) if n_reviewed else 0.0,
        "recall": float(tp_mask.sum() / max(1, int((y_true == 1).sum()))),
        "dollars_recovered": round(recovered, 2),
        "review_cost_total": round(review_cost_total, 2),
        "fp_cost_total": round(fp_cost_total, 2),
        "net_value": round(net, 2),
        "dollars_identified_per_1000_reviewed": round(
            float(improper_amount[tp_mask].sum()) / n_reviewed * 1000, 2
        )
        if n_reviewed
        else 0.0,
    }


def capacity_sweep(
    scores: np.ndarray,
    prob: np.ndarray,
    paid: np.ndarray,
    y_true: np.ndarray,
    improper_amount: np.ndarray,
    params: CostParams,
    capacities: list[float],
) -> pd.DataFrame:
    """Evaluate top-k review policies at each capacity share."""
    order = np.argsort(-scores, kind="stable")
    rows = []
    for cap in capacities:
        k = max(1, int(len(scores) * cap))
        take = order[:k]
        # Respect the risk floor even with spare capacity.
        take = take[scores[take] >= params.min_risk_threshold]
        flagged = np.zeros(len(scores), dtype=bool)
        flagged[take] = True
        realized = realized_review_outcome(flagged, y_true, improper_amount, params)
        expected = float(expected_value_per_claim(prob[take], paid[take], params).sum())
        rows.append(
            {
                "capacity_share": cap,
                "review_threshold": float(scores[take].min()) if len(take) else float("nan"),
                "expected_net_value": round(expected, 2),
                **realized,
            }
        )
    return pd.DataFrame(rows)


def threshold_sweep(
    scores: np.ndarray,
    prob: np.ndarray,
    paid: np.ndarray,
    y_true: np.ndarray,
    improper_amount: np.ndarray,
    params: CostParams,
    n_grid: int = 99,
) -> pd.DataFrame:
    """Evaluate review-everything-above-τ policies across a threshold grid."""
    rows = []
    for tau in np.linspace(0.01, 0.99, n_grid):
        flagged = scores >= tau
        realized = realized_review_outcome(flagged, y_true, improper_amount, params)
        expected = float(expected_value_per_claim(prob[flagged], paid[flagged], params).sum())
        rows.append(
            {
                "threshold": round(float(tau), 4),
                "expected_net_value": round(expected, 2),
                **realized,
            }
        )
    return pd.DataFrame(rows)


def select_policy(
    val_scores: np.ndarray,
    val_prob: np.ndarray,
    val_paid: np.ndarray,
    val_y: np.ndarray,
    val_improper: np.ndarray,
    params: CostParams,
    capacity_share: float,
) -> dict[str, Any]:
    """Choose the operating point on validation data.

    The review threshold is whichever is HIGHER: the score cutoff implied by
    investigator capacity, the configured risk floor, or the τ that maximizes
    expected net value — capacity binds spending, the floor binds precision,
    and the ENV argmax stops reviewing where marginal value goes negative.
    """
    sweep = threshold_sweep(val_scores, val_prob, val_paid, val_y, val_improper, params)
    best = sweep.loc[sweep["expected_net_value"].idxmax()]
    k = max(1, int(len(val_scores) * capacity_share))
    capacity_threshold = float(np.sort(val_scores)[::-1][min(k, len(val_scores)) - 1])
    review_threshold = float(max(best["threshold"], capacity_threshold, params.min_risk_threshold))
    policy = {
        "review_threshold": round(review_threshold, 4),
        "monitor_threshold": round(review_threshold * params.monitor_fraction, 4),
        "capacity_share": capacity_share,
        "capacity_implied_threshold": round(capacity_threshold, 4),
        "env_optimal_threshold": round(float(best["threshold"]), 4),
        "expected_net_value_at_env_optimum": float(best["expected_net_value"]),
        "cost_params": {
            "review_cost": params.review_cost,
            "recovery_rate": params.recovery_rate,
            "fp_extra_cost": params.fp_extra_cost,
            "min_risk_threshold": params.min_risk_threshold,
        },
    }
    logger.info("policy_selected", extra=dict(policy))
    return policy


def recommended_action(score: float, policy: dict[str, Any]) -> str:
    """Map a risk score to an operational disposition."""
    if score >= float(policy["review_threshold"]):
        return "manual_review"
    if score >= float(policy["monitor_threshold"]):
        return "monitor"
    return "auto_clear"


def estimated_payment_at_risk(prob: np.ndarray, paid: np.ndarray) -> np.ndarray:
    """Per-claim expected improper payment (calibrated probability × paid)."""
    return np.round(prob * paid, 2)
