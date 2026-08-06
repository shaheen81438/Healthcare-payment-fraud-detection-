"""Structured logging tests."""

from __future__ import annotations

import json
import logging

import pytest

from src.utils.logging import JsonFormatter, configure_logging, request_id_ctx


def _make_record(msg: str = "hello", **extra: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_formats_as_json_with_core_fields() -> None:
    out = json.loads(JsonFormatter().format(_make_record()))
    assert out["message"] == "hello"
    assert out["level"] == "INFO"
    assert out["logger"] == "test.logger"
    assert "timestamp" in out
    assert "request_id" not in out


def test_includes_request_id_when_set() -> None:
    token = request_id_ctx.set("req-abc123")
    try:
        out = json.loads(JsonFormatter().format(_make_record()))
    finally:
        request_id_ctx.reset(token)
    assert out["request_id"] == "req-abc123"


def test_includes_extra_fields() -> None:
    out = json.loads(JsonFormatter().format(_make_record(duration_ms=12.5, status_code=200)))
    assert out["duration_ms"] == 12.5
    assert out["status_code"] == 200


def test_configure_logging_emits_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    logging.getLogger("smoke").info("configured", extra={"phase": 2})
    captured = capsys.readouterr().out.strip().splitlines()[-1]
    parsed = json.loads(captured)
    assert parsed["message"] == "configured"
    assert parsed["phase"] == 2
