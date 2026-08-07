"""Temporal split assignment (burn_in / train / val / test) by service month."""

from __future__ import annotations

from typing import Any

import pandas as pd
import yaml


def load_features_config(path: str = "configs/features.yaml") -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)
    return cfg


def assign_split(claims: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Add a ``split`` column based on service month boundaries."""
    s = cfg["splits"]
    month = claims["service_date"].dt.to_period("M").astype(str)
    out = claims.copy()
    out["split"] = "test"
    out.loc[month <= s["val_end"], "split"] = "val"
    out.loc[month <= s["train_end"], "split"] = "train"
    out.loc[month <= s["burn_in_end"], "split"] = "burn_in"
    return out
