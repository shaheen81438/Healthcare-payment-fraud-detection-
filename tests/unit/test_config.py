"""Settings tests."""

from __future__ import annotations

import pytest

from src.utils.config import Settings, get_settings


def test_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.env == "dev"
    assert settings.postgres_port == 5432
    assert settings.batch_predict_limit == 1000
    assert settings.random_seed == 42


def test_database_url_assembly() -> None:
    settings = Settings(
        _env_file=None,
        postgres_user="u",
        postgres_password="p",
        postgres_host="db.example.com",
        postgres_port=6543,
        postgres_db="claims",
    )
    assert settings.database_url == "postgresql+psycopg2://u:p@db.example.com:6543/claims"


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_HOST", "override-host")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    settings = get_settings()
    assert settings.postgres_host == "override-host"
    assert settings.log_level == "DEBUG"


def test_settings_cached() -> None:
    assert get_settings() is get_settings()
