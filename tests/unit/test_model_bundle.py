"""End-to-end model training and bundle roundtrip on a small dataset (marked slow)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.data.ingest_cms import load_snapshot
from src.data.reference import load_reference_frames
from src.data.synthetic.assemble import build_dataset
from src.data.synthetic.generate import load_generation_config
from src.features.builder import build_features
from src.features.splits import load_features_config
from src.modeling.bundle import ScoringBundle
from src.modeling.train import load_model_config, prepare_training_frame, train_all

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    gen_cfg = load_generation_config()
    gen_cfg.update(n_providers=40, n_beneficiaries=1200, n_months=10, seed=5)
    bundle_ds = build_dataset(gen_cfg, load_snapshot(), load_reference_frames())
    claims = bundle_ds.claims.copy()
    for col in ["service_date", "service_end_date", "submit_date"]:
        claims[col] = pd.to_datetime(claims[col])
    fcfg = load_features_config()
    fcfg["splits"] = {"burn_in_end": "2024-02", "train_end": "2024-06", "val_end": "2024-08"}
    features, _, meta = build_features(claims, bundle_ds.providers, load_reference_frames(), fcfg)

    mcfg = load_model_config()
    mcfg["xgboost"] = {**mcfg["xgboost"], "n_estimators": 120, "early_stopping_rounds": 20}
    frame, numeric, _ = prepare_training_frame(
        features[features["split"] != "burn_in"], bundle_ds.labels, True
    )
    result = train_all(frame, numeric, mcfg, meta)
    return {"result": result, "frame": frame, "tmp": tmp_path_factory.mktemp("bundle")}


def test_training_beats_random_and_rules(trained: dict[str, Any]) -> None:
    m = trained["result"].metrics
    base_rate = trained["frame"]["is_improper"].mean()
    assert m["xgboost"]["test"]["pr_auc"] > 5 * base_rate
    assert m["xgboost"]["test"]["pr_auc"] > m["rules_baseline"]["test"]["pr_auc"]
    assert 0.6 <= m["ensemble"]["weight"] <= 1.0


def test_bundle_roundtrip_scores_identically(trained: dict[str, Any]) -> None:
    result, frame = trained["result"], trained["frame"]
    bundle: ScoringBundle = result.bundle
    X = frame[bundle.numeric_features + bundle.categorical_features].head(200)
    before = bundle.risk_score(X)

    path = Path(trained["tmp"]) / "claims-risk-test"
    bundle.save(path)
    reloaded = ScoringBundle.load(path)
    after = reloaded.risk_score(X)

    np.testing.assert_allclose(
        before["risk_score"].to_numpy(), after["risk_score"].to_numpy(), atol=1e-9
    )
    assert reloaded.model_version == bundle.model_version
    assert reloaded.feature_names_out == bundle.feature_names_out


def test_risk_scores_are_bounded_and_ordered(trained: dict[str, Any]) -> None:
    result, frame = trained["result"], trained["frame"]
    bundle: ScoringBundle = result.bundle
    test = frame[frame["split"] == "test"]
    X = test[bundle.numeric_features + bundle.categorical_features]
    scored = bundle.risk_score(X)
    assert scored["risk_score"].between(0, 1).all()
    assert scored["supervised_probability"].between(0, 1).all()
    assert scored["anomaly_percentile"].between(0, 1).all()
    # Positives should outrank negatives on average.
    y = test["is_improper"].to_numpy()
    assert scored.loc[y == 1, "risk_score"].mean() > scored.loc[y == 0, "risk_score"].mean()
