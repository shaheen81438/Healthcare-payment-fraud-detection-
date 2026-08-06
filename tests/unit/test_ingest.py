"""Ingestion normalization tests (snapshot + fixture based, no network)."""

from __future__ import annotations

from src.data.ingest_cms import load_snapshot
from src.data.ingest_leie import load_fixture


def test_cms_snapshot_normalized() -> None:
    df = load_snapshot()
    assert len(df) > 300
    assert {"npi", "provider_type", "hcpcs", "avg_paid", "total_services"} <= set(df.columns)
    assert df["avg_paid"].notna().all()
    assert (df["avg_paid"] <= df["avg_allowed"] + 0.01).all()
    assert df["npi"].str.fullmatch(r"\d{10}").all()
    assert df["provider_type"].nunique() >= 10


def test_leie_fixture_normalized() -> None:
    df = load_fixture()
    # The all-zero sentinel NPI row must be dropped.
    assert "0000000000" not in set(df["npi"])
    assert df["npi"].str.fullmatch(r"\d{10}").all()
    # Fraud-related exclusion authorities flagged.
    flagged = dict(zip(df["npi"], df["fraud_related"], strict=True))
    assert flagged["9990000001"] is True  # 1128a1
    assert flagged["9990000003"] is True  # 1128b7
    assert flagged["9990000004"] is False  # 1128b4 (license action, not fraud conviction)
    assert df["exclusion_date"].notna().all()
