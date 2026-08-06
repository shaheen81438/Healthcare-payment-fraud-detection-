"""Process-wide scoring bundle holder.

The API loads one immutable model bundle per process (loaded at startup in the
prediction phase). This module owns that singleton so /health can report
readiness without importing heavy modeling code at module import time.
"""

from __future__ import annotations

from typing import Any

_bundle: Any | None = None


def set_bundle(bundle: Any | None) -> None:
    """Attach (or detach) the active scoring bundle."""
    global _bundle
    _bundle = bundle


def get_bundle() -> Any | None:
    """Return the active scoring bundle, if one is loaded."""
    return _bundle


def model_is_loaded() -> bool:
    """True once a scoring bundle has been attached to this process."""
    return _bundle is not None
