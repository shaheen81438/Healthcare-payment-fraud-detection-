"""Rule-engine tests."""

from __future__ import annotations

import pandas as pd
import pytest

from src.modeling.rules import RuleEngine


@pytest.fixture(scope="module")
def engine() -> RuleEngine:
    return RuleEngine.from_yaml()


def _feature_row(**overrides: object) -> pd.DataFrame:
    base: dict[str, object] = {
        "claim_id": "CLM-00000001",
        "exact_dup_prior": 0,
        "dup_max_sim": 0.0,
        "days_since_bene_same_hcpcs": 999.0,
        "units_vs_code_p95": 1.0,
        "paid_vs_code_median": 1.0,
        "prov_velocity_1v3": 1.0,
        "prov_claims_1m": 10,
        "dxproc_compat": 0.9,
        "pos_atypicality": 0.1,
        "unbundle_component_no_mod": 0,
        "em_level": 3,
        "em_gap_vs_specialty": 0.0,
        "prov_top_code_share_3m": 0.3,
        "prov_claims_3m": 100,
        "bene_provider_same_day_claims": 0,
        "bene_claims_30d": 2,
        "bene_distinct_providers_prior": 2,
        "days_since_bene_last_claim": 30.0,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_clean_claim_hits_no_rules(engine: RuleEngine) -> None:
    out = engine.evaluate(_feature_row())
    assert out.loc[0, "n_rules_hit"] == 0
    assert out.loc[0, "rule_score"] == 0.0
    assert out.loc[0, "rule_hits"] == ""


def test_exact_duplicate_rule(engine: RuleEngine) -> None:
    out = engine.evaluate(_feature_row(exact_dup_prior=1))
    assert "R001_exact_duplicate" in out.loc[0, "rule_hits"]
    assert "possible_exact_duplicate" in out.loc[0, "rule_reasons"]
    assert out.loc[0, "rule_score"] > 0.5


def test_near_duplicate_excluded_when_exact(engine: RuleEngine) -> None:
    out = engine.evaluate(_feature_row(exact_dup_prior=1, dup_max_sim=0.99))
    hits = out.loc[0, "rule_hits"]
    assert "R001_exact_duplicate" in hits
    assert "R002_near_duplicate" not in hits


def test_any_of_rule_fires_on_either_condition(engine: RuleEngine) -> None:
    a = engine.evaluate(_feature_row(bene_provider_same_day_claims=4))
    b = engine.evaluate(_feature_row(bene_claims_30d=15))
    assert "R012_excessive_bene_activity" in a.loc[0, "rule_hits"]
    assert "R012_excessive_bene_activity" in b.loc[0, "rule_hits"]


def test_scores_accumulate_and_stay_below_one(engine: RuleEngine) -> None:
    out = engine.evaluate(
        _feature_row(
            exact_dup_prior=1,
            units_vs_code_p95=5.0,
            paid_vs_code_median=4.0,
            unbundle_component_no_mod=1,
            dxproc_compat=0.05,
        )
    )
    assert out.loc[0, "n_rules_hit"] == 5
    assert 0.9 < out.loc[0, "rule_score"] < 1.0


def test_unknown_feature_raises(engine: RuleEngine) -> None:
    bad = pd.DataFrame([{"claim_id": "CLM-1", "exact_dup_prior": 0}])
    with pytest.raises(KeyError, match="unknown feature"):
        engine.evaluate(bad)
