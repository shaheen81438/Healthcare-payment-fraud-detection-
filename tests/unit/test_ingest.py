"""Ingestion normalization tests (snapshot and fixture based, no network)."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import requests

from src.data import ingest_leie
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


def test_leie_schema_rejects_missing_and_unexpected_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        ingest_leie._normalize(pd.DataFrame({"NPI": ["1234567890"]}))
    with pytest.raises(ValueError, match="unexpected columns"):
        ingest_leie._normalize(
            pd.DataFrame(
                {
                    "NPI": ["1234567890"],
                    "EXCLTYPE": ["1128a1"],
                    "EXCLDATE": ["20240101"],
                    "UNEXPECTED_FIELD": ["value"],
                }
            )
        )


def test_leie_fetch_retries_temporary_timeout_and_sets_timeout(monkeypatch) -> None:
    response = requests.Response()
    response.status_code = 200
    response._content = b"NPI,EXCLTYPE,EXCLDATE\n1234567890,1128a1,20240101\n"
    calls = []

    def fake_get(_url, *, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            raise requests.Timeout("temporary timeout")
        return response

    monkeypatch.setattr(ingest_leie.requests, "get", fake_get)
    monkeypatch.setattr(ingest_leie.time, "sleep", lambda _seconds: None)

    frame = ingest_leie.fetch_from_source(timeout=7, max_retries=2)

    assert len(frame) == 1
    assert calls == [(10, 7), (10, 7)]


def test_leie_fetch_rejects_html_response(monkeypatch) -> None:
    response = requests.Response()
    response.status_code = 200
    response._content = b"<!doctype html><html><body>maintenance</body></html>"
    monkeypatch.setattr(ingest_leie.requests, "get", lambda *_args, **_kwargs: response)

    with pytest.raises(ValueError, match="not the expected CSV"):
        ingest_leie.fetch_from_source(max_retries=1)


def test_leie_ingestion_failure_is_recorded_and_reraised(tmp_path, monkeypatch) -> None:
    manifest_path = tmp_path / "provenance.json"

    def fail_download():
        raise requests.Timeout("request timed out")

    monkeypatch.setattr(ingest_leie, "fetch_from_source", fail_download)

    with pytest.raises(requests.Timeout):
        ingest_leie.ingest_leie(use_fixture=False, manifest_path=manifest_path)

    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert records[-1]["validation_status"] == "failed:Timeout"
    assert records[-1]["classification"] == "real_public"
