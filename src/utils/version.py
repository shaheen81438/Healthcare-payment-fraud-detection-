"""Package version helper — single source of truth is pyproject.toml."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def get_version() -> str:
    """Return the installed package version, or a dev marker when not installed."""
    try:
        return version("healthcare-payment-integrity")
    except PackageNotFoundError:  # pragma: no cover - only hit outside an install
        return "0.0.0+dev"
