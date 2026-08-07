"""Monitoring statistics + alert-threshold tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import jsd, load_monitoring_config, psi, run_monitoring, simulate_drift


def test_psi_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, a) < 0.001


def test_psi_detects_mean_shift() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(1.0, 1, 5000)
    assert psi(a, b) > 0.3


def test_jsd_bounds_and_sensitivity() -> None:
    same = pd.Series(["a"] * 50 + ["b"] * 50)
    assert jsd(same, same) < 1e-9
    shifted = pd.Series(["a"] * 90 + ["b"] * 10)
    value = jsd(same, shifted)
    assert 0.05 < value <= 1.0


def _frame(n: int = 800, seed: int = 1, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "claim_id": [f"CLM-{seed}{i:06d}" for i in range(n)],
            "specialty": rng.choice(["Internal Medicine", "Cardiology", "Podiatry"], n),
            "hcpcs": rng.choice(["99213", "99214", "97110", "11721"], n),
            "pos": rng.choice(["11", "21"], n),
            "dx_group": rng.choice(["DXG01", "DXG02"], n),
            "service_date": pd.Timestamp("2025-10-01")
            + pd.to_timedelta(rng.integers(0, 60, n), "D"),
            "paid_amount": rng.lognormal(4, 0.4, n) + shift * 60,
            "paid_per_unit": rng.lognormal(4, 0.4, n) + shift * 60,
            "paid_vs_code_median": rng.normal(1 + shift, 0.2, n),
            "units": rng.integers(1, 4, n),
            "dup_max_sim": rng.uniform(0, 0.5, n),
            "bene_claims_30d": rng.integers(0, 5, n),
            "prov_claims_3m": rng.integers(20, 200, n),
            "prov_paid_vs_peer_3m": rng.normal(1, 0.2, n),
            "prov_top_code_share_3m": rng.uniform(0.1, 0.5, n),
            "prov_new_bene_share_1m": rng.uniform(0, 0.4, n),
            "pos_atypicality": rng.uniform(0, 0.4, n),
            "dxproc_compat": rng.uniform(0.5, 1.0, n),
            "submit_lag_days": rng.integers(1, 30, n),
            "risk_score": np.clip(rng.beta(1, 12, n) + shift * 0.25, 0, 1),
        }
    )


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_monitoring_config()


def test_no_drift_reports_ok(tmp_path, cfg) -> None:
    doc = run_monitoring(_frame(seed=1), _frame(seed=2), cfg, reports_dir=tmp_path)
    assert doc["status"] == "ok"
    assert doc["alerts"] == []
    assert (tmp_path / "drift_report.html").exists()
    assert (tmp_path / "monitoring_status.json").exists()


def test_shifted_current_triggers_alerts(tmp_path, cfg) -> None:
    doc = run_monitoring(_frame(seed=1), _frame(seed=2, shift=1.2), cfg, reports_dir=tmp_path)
    assert doc["status"] in {"warn", "critical"}
    assert doc["alerts"]
    assert doc["score_psi"] > cfg["thresholds"]["score_psi_warn"]


def test_simulate_drift_is_detected(tmp_path, cfg) -> None:
    reference = _frame(seed=3, n=1500)
    doc = run_monitoring(reference, simulate_drift(reference), cfg, reports_dir=tmp_path)
    assert doc["status"] in {"warn", "critical"}
    assert doc["n_drifted_features"] >= 1


def test_delayed_precision_alert(tmp_path, cfg) -> None:
    decisions = pd.DataFrame(
        {
            "claim_id": [f"CLM-D{i}" for i in range(40)],
            "status": ["dismissed"] * 30 + ["approved"] * 10,
        }
    )
    doc = run_monitoring(
        _frame(seed=1), _frame(seed=2), cfg, reports_dir=tmp_path, decisions=decisions
    )
    delayed = doc["delayed_performance"]
    assert delayed["n_decisions"] == 40
    assert delayed["realized_flag_precision"] == pytest.approx(0.25)
    assert any("precision" in a for a in doc["alerts"])
