"""Reason-code mapping and SHAP aggregation logic (no model training needed)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.modeling.explain import ClaimExplainer, load_reason_config


@pytest.fixture(scope="module")
def mapper() -> ClaimExplainer:
    """An explainer with the SHAP machinery stubbed out — mapping logic only."""
    explainer = ClaimExplainer.__new__(ClaimExplainer)
    explainer.reason_cfg = load_reason_config()
    explainer.bundle = SimpleNamespace(  # type: ignore[assignment]
        ensemble_weight=0.95,
        categorical_features=["specialty", "pos", "dx_group"],
        numeric_features=["paid_amount", "dup_max_sim", "pos_office"],
    )
    return explainer


def _contrib(feature: str, contribution: float, value: Any = 1.0) -> dict[str, Any]:
    return {"feature": feature, "contribution": contribution, "value": value}


def test_positive_contributions_map_to_codes(mapper: ClaimExplainer) -> None:
    codes = mapper.reason_codes(
        [_contrib("dup_max_sim", 0.8), _contrib("paid_vs_code_median", 0.5)],
    )
    assert codes == ["possible_duplicate", "amount_above_peer_norm"]


def test_negative_contributions_are_ignored(mapper: ClaimExplainer) -> None:
    codes = mapper.reason_codes(
        [_contrib("dup_max_sim", -0.9), _contrib("units_vs_code_p95", 0.4)],
    )
    assert codes == ["excessive_units"]


def test_rule_reasons_come_first_and_dedupe(mapper: ClaimExplainer) -> None:
    codes = mapper.reason_codes(
        [_contrib("exact_dup_prior", 0.9), _contrib("units_vs_code_p95", 0.4)],
        rule_reasons="possible_exact_duplicate;excessive_units_for_code",
    )
    assert codes[0] == "possible_exact_duplicate"
    assert codes[1] == "excessive_units_for_code"
    # SHAP-derived codes appended after rules, still deduped overall.
    assert "possible_duplicate" in codes
    assert len(codes) == len(set(codes))


def test_anomaly_reason_added_when_anomaly_dominates(mapper: ClaimExplainer) -> None:
    codes = mapper.reason_codes([], anomaly_percentile=0.99)
    assert codes == ["unusual_overall_pattern"]
    none = mapper.reason_codes([], anomaly_percentile=0.5)
    assert none == []


def test_max_codes_cap(mapper: ClaimExplainer) -> None:
    contribs = [
        _contrib("dup_max_sim", 0.9),
        _contrib("paid_vs_code_median", 0.8),
        _contrib("units_vs_code_p95", 0.7),
        _contrib("prov_paid_vs_peer_3m", 0.6),
        _contrib("pos_atypicality", 0.5),
        _contrib("em_gap_vs_specialty", 0.4),
        _contrib("unbundle_component_no_mod", 0.3),
    ]
    codes = mapper.reason_codes(contribs, max_codes=5)
    assert len(codes) == 5


def test_parent_map_folds_ohe_columns(mapper: ClaimExplainer) -> None:
    names_out = [
        "paid_amount",
        "dup_max_sim",
        "pos_office",
        "specialty_Internal Medicine",
        "specialty_Cardiology",
        "pos_11",
        "dx_group_DXG01",
    ]
    parents = mapper._parent_map(names_out)
    assert parents == [
        "paid_amount",
        "dup_max_sim",
        "pos_office",  # numeric feature, NOT folded into 'pos'
        "specialty",
        "specialty",
        "pos",
        "dx_group",
    ]


def test_every_numeric_feature_has_a_reason_mapping() -> None:
    """Guards against adding features that silently produce no reason code."""
    from src.features.builder import NUMERIC_FEATURES
    from src.modeling.train import RULE_FEATURES

    cfg = load_reason_config()["reason_codes"]
    unmapped = [
        f
        for f in [*NUMERIC_FEATURES, *RULE_FEATURES]
        if f not in cfg
        # Purely contextual timing fields are allowed to be unmapped.
        and f
        not in {
            "allowed_amount",
            "submit_lag_days",
            "service_days",
            "dow",
            "month_num",
            "mod_25",
            "mod_59",
            "mod_ltrt",
            "service_end_days",
            "prov_months_active",
            "prov_distinct_benes_3m",
            "prov_paid_3m",
            "prov_em_level_mean_3m",
        }
    ]
    assert not unmapped, f"features without reason-code mapping: {unmapped}"
