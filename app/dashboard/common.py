"""Shared dashboard utilities: data access (Postgres, with parquet as fallback),
the Altair theme (project palette), and saving investigator decisions."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

# ---- palette (same validated colors as src/utils/viz.py) ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
ACTION_COLORS = {"manual_review": ORANGE, "monitor": BLUE, "auto_clear": AQUA}

DECISIONS_PARQUET = Path("data/processed/review_decisions.parquet")

#: Committed sampled artifacts for the hosted demo (see scripts/make_demo_data.py).
#: Loaders fall back here when the full generated artifacts are absent (e.g. on
#: Streamlit Community Cloud, which clones only the git repo).
DEMO_DIR = Path("demo_data")


def _resolve(path: str) -> Path:
    """Prefer the full artifact; fall back to the committed demo bundle."""
    full = Path(path)
    if full.exists():
        return full
    demo = DEMO_DIR / full.name
    return demo if demo.exists() else full


def demo_mode() -> bool:
    """True when running off the committed demo bundle rather than full artifacts."""
    return (
        not Path("data/processed/predictions.parquet").exists()
        and (DEMO_DIR / "predictions.parquet").exists()
    )


def apply_theme() -> None:
    """Register and enable the project Altair theme."""

    def _theme() -> dict[str, Any]:
        return {
            "config": {
                "background": SURFACE,
                "view": {"stroke": None},
                "axis": {
                    "labelColor": MUTED,
                    "titleColor": INK_2,
                    "gridColor": GRID,
                    "domainColor": "#c3c2b7",
                    "tickColor": "#c3c2b7",
                },
                "legend": {"labelColor": INK_2, "titleColor": INK_2},
                "title": {"color": INK, "fontSize": 14},
                "range": {"category": [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]},
            }
        }

    try:
        alt.theme.register("pi", enable=True)(_theme)  # altair >= 5.5
    except AttributeError:  # pragma: no cover - older altair
        alt.themes.register("pi", _theme)
        alt.themes.enable("pi")


def page_setup(title: str) -> None:
    st.set_page_config(page_title=f"{title} · Payment Integrity", layout="wide")
    apply_theme()
    st.title(title)


def synthetic_banner() -> None:
    st.caption(
        "All claims, providers, and beneficiaries are **synthetic** (calibrated to real CMS "
        "distributions). Risk scores are **review priorities, not fraud determinations**."
    )
    if demo_mode():
        st.caption(
            "Hosted demo running on the committed sample bundle. The full artifacts, the "
            "scoring API, and MLflow come with the repo's local and Azure setups "
            "([github.com/FaridoonFarahi/healthcare-payment-integrity]"
            "(https://github.com/FaridoonFarahi/healthcare-payment-integrity))."
        )


# ------------------------------------------------------------------ data access
def _engine() -> Any | None:
    try:
        from src.data.db import get_engine

        engine = get_engine()
        with engine.connect():
            pass
        return engine
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner="Loading scored claims…")
def load_predictions() -> pd.DataFrame:
    engine = _engine()
    if engine is not None:
        try:
            return pd.read_sql("SELECT * FROM scoring.predictions", engine)
        except Exception:
            logger.warning("predictions_db_read_failed; falling back to parquet")
    return pd.read_parquet(_resolve("data/processed/predictions.parquet"))


@st.cache_data(ttl=300, show_spinner=False)
def load_explanations() -> pd.DataFrame:
    engine = _engine()
    if engine is not None:
        try:
            return pd.read_sql(
                "SELECT claim_id, base_value, shap_top FROM scoring.explanations", engine
            )
        except Exception:
            logger.warning("explanations_db_read_failed; falling back to parquet")
    return pd.read_parquet(_resolve("data/processed/explanations.parquet"))


@st.cache_data(ttl=600, show_spinner=False)
def load_claims() -> pd.DataFrame:
    return pd.read_parquet(_resolve("data/processed/claims.parquet"))


@st.cache_data(ttl=600, show_spinner=False)
def load_labels_research_mode() -> pd.DataFrame:
    """Ground-truth injected labels, shown only behind the demo toggle."""
    return pd.read_parquet(_resolve("data/processed/labels.parquet"))


@st.cache_data(ttl=600, show_spinner=False)
def load_peer_stats() -> dict[str, pd.DataFrame]:
    from src.features.peer_stats import load_peer_stats as _load

    base = Path("models/feature_store")
    if not (base / "code_stats.parquet").exists() and (DEMO_DIR / "code_stats.parquet").exists():
        base = DEMO_DIR
    return _load(str(base))


@st.cache_data(ttl=600, show_spinner=False)
def load_policy() -> dict[str, Any]:
    pointer_path = Path("models/latest.json")
    if pointer_path.exists():
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        meta = json.loads((Path(pointer["path"]) / "metadata.json").read_text(encoding="utf-8"))
        policy: dict[str, Any] = meta.get("threshold_policy", {})
        return policy
    demo_policy = DEMO_DIR / "policy.json"
    if demo_policy.exists():
        loaded: dict[str, Any] = json.loads(demo_policy.read_text(encoding="utf-8"))
        return loaded
    return {}


# ------------------------------------------------------------------ decisions
def load_decisions() -> pd.DataFrame:
    engine = _engine()
    if engine is not None:
        try:
            return pd.read_sql("SELECT * FROM ops.review_decisions", engine)
        except Exception:
            pass
    if DECISIONS_PARQUET.exists():
        return pd.read_parquet(DECISIONS_PARQUET)
    return pd.DataFrame(
        columns=["claim_id", "status", "decided_by", "note", "decided_at", "model_version"]
    )


def record_decision(
    claim_id: str, status: str, decided_by: str, note: str, model_version: str
) -> None:
    """Upsert an investigator decision (Postgres when available, parquet otherwise)."""
    row = pd.DataFrame(
        [
            {
                "claim_id": claim_id,
                "status": status,
                "decided_by": decided_by or "investigator",
                "note": note,
                "decided_at": datetime.now(UTC),
                "model_version": model_version,
            }
        ]
    )
    engine = _engine()
    if engine is not None:
        try:
            from sqlalchemy import text

            with engine.begin() as conn:
                conn.execute(
                    text("DELETE FROM ops.review_decisions WHERE claim_id = :cid"),
                    {"cid": claim_id},
                )
            row.to_sql("review_decisions", engine, schema="ops", if_exists="append", index=False)
            return
        except Exception:
            logger.warning("decision_db_write_failed; using parquet fallback")
    existing = pd.read_parquet(DECISIONS_PARQUET) if DECISIONS_PARQUET.exists() else pd.DataFrame()
    combined = pd.concat(
        [existing[existing.get("claim_id") != claim_id] if len(existing) else existing, row]
    )
    DECISIONS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(DECISIONS_PARQUET, index=False)


def decision_status_map() -> dict[str, str]:
    decisions = load_decisions()
    if decisions.empty:
        return {}
    return dict(zip(decisions["claim_id"], decisions["status"], strict=False))


def money(value: float) -> str:
    return f"${value:,.0f}"
