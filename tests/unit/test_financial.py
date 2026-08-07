"""Financial framework tests with hand-computed cases."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.financial import (
    CostParams,
    capacity_sweep,
    estimated_payment_at_risk,
    expected_value_per_claim,
    realized_review_outcome,
    recommended_action,
    select_policy,
)

PARAMS = CostParams(
    review_cost=30.0,
    recovery_rate=0.5,
    fp_extra_cost=10.0,
    min_risk_threshold=0.2,
    monitor_fraction=0.5,
)


def test_expected_value_per_claim_hand_case() -> None:
    # p=0.8, paid=1000: 0.8*0.5*1000 - 30 - 0.2*10 = 400 - 30 - 2 = 368
    out = expected_value_per_claim(np.array([0.8]), np.array([1000.0]), PARAMS)
    assert out[0] == pytest.approx(368.0)
    # p=0.01, paid=100: 0.5 - 30 - 9.9 = -39.4 → reviewing is value-destroying
    out2 = expected_value_per_claim(np.array([0.01]), np.array([100.0]), PARAMS)
    assert out2[0] == pytest.approx(-39.4)


def test_realized_review_outcome_hand_case() -> None:
    flagged = np.array([True, True, True, False])
    y = np.array([1, 0, 1, 1])
    improper = np.array([500.0, 0.0, 300.0, 900.0])
    out = realized_review_outcome(flagged, y, improper, PARAMS)
    assert out["n_reviewed"] == 3
    assert out["true_positives"] == 2
    assert out["false_positives"] == 1
    # recovered = 0.5 * (500+300) = 400; costs = 3*30 + 1*10 = 100 → net 300
    assert out["net_value"] == pytest.approx(300.0)
    assert out["recall"] == pytest.approx(2 / 3)
    # dollars identified per 1000 reviewed = 800/3*1000
    assert out["dollars_identified_per_1000_reviewed"] == pytest.approx(800 / 3 * 1000, rel=1e-4)


def test_capacity_sweep_respects_capacity_and_floor() -> None:
    rng = np.random.default_rng(1)
    n = 1000
    scores = rng.uniform(0, 1, n)
    y = (scores > 0.9).astype(int)  # top decile is truly improper
    out = capacity_sweep(
        scores, scores, np.full(n, 200.0), y, np.where(y == 1, 150.0, 0.0), PARAMS, [0.01, 0.05]
    )
    assert list(out["capacity_share"]) == [0.01, 0.05]
    assert (out["n_reviewed"] <= [10, 50]).all()
    # All reviewed claims sit above the risk floor.
    assert (out["review_threshold"] >= PARAMS.min_risk_threshold).all()
    # Perfectly separable top decile → precision 1 at both capacities.
    assert (out["precision"] == 1.0).all()


def test_select_policy_prefers_binding_constraint() -> None:
    rng = np.random.default_rng(2)
    n = 2000
    prob = rng.uniform(0, 1, n)
    scores = prob
    y = (rng.uniform(0, 1, n) < prob).astype(int)
    improper = np.where(y == 1, 400.0, 0.0)
    paid = np.full(n, 400.0)
    policy = select_policy(scores, prob, paid, y, improper, PARAMS, capacity_share=0.01)
    # 1% capacity on uniform scores implies a ~0.99 cutoff — capacity must bind.
    assert policy["review_threshold"] >= policy["env_optimal_threshold"]
    assert policy["review_threshold"] >= 0.9
    assert policy["monitor_threshold"] == pytest.approx(policy["review_threshold"] * 0.5, rel=1e-3)


def test_recommended_action_bands() -> None:
    policy = {"review_threshold": 0.6, "monitor_threshold": 0.3}
    assert recommended_action(0.75, policy) == "manual_review"
    assert recommended_action(0.45, policy) == "monitor"
    assert recommended_action(0.1, policy) == "auto_clear"


def test_estimated_payment_at_risk() -> None:
    out = estimated_payment_at_risk(np.array([0.5, 0.1]), np.array([1000.0, 200.0]))
    assert out.tolist() == [500.0, 20.0]
