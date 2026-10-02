"""Vectorized rule engine. Its output is a transparent baseline, reason codes, and model features."""

from __future__ import annotations

import logging
import operator
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_OPS = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
}


class RuleEngine:
    """Evaluates configured threshold rules over the feature matrix."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.rules: dict[str, dict[str, Any]] = config["rules"]

    @classmethod
    def from_yaml(cls, path: str = "configs/rules.yaml") -> RuleEngine:
        with open(path, encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh))

    @staticmethod
    def _mask(features: pd.DataFrame, conditions: list[list[Any]], mode: str) -> pd.Series:
        masks = []
        for feature, op, value in conditions:
            if feature not in features.columns:
                raise KeyError(f"rule references unknown feature: {feature}")
            masks.append(_OPS[op](features[feature], value))
        combined = masks[0]
        for mask in masks[1:]:
            combined = (combined & mask) if mode == "all" else (combined | mask)
        return combined

    def evaluate(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return per-claim rule hits, reasons, weighted rule score, and hit count."""
        hits = pd.DataFrame(index=features.index)
        weights: dict[str, float] = {}
        reasons: dict[str, str] = {}
        for rule_id, spec in self.rules.items():
            mode = "all" if "all" in spec else "any_of"
            hits[rule_id] = self._mask(features, spec[mode], mode).astype(int)
            weights[rule_id] = float(spec["weight"])
            reasons[rule_id] = str(spec["reason"])

        weight_arr = np.array([weights[c] for c in hits.columns])
        raw_score = hits.to_numpy() @ weight_arr
        rule_score = 1.0 - np.exp(-raw_score)  # squash to [0, 1)

        hit_lists = [
            ";".join(rule for rule in hits.columns[row.astype(bool)]) for row in hits.to_numpy()
        ]
        reason_lists = [
            ";".join(dict.fromkeys(reasons[rule] for rule in h.split(";") if rule)) if h else ""
            for h in hit_lists
        ]
        out = pd.DataFrame(
            {
                "claim_id": features["claim_id"].to_numpy(),
                "rule_hits": hit_lists,
                "rule_reasons": reason_lists,
                "rule_score": np.round(rule_score, 4),
                "n_rules_hit": hits.sum(axis=1).to_numpy(),
            }
        )
        logger.info(
            "rules_evaluated",
            extra={
                "claims": len(out),
                "any_hit_share": round(float((out["n_rules_hit"] > 0).mean()), 4),
            },
        )
        return out

    def hit_counts(self, evaluated: pd.DataFrame) -> dict[str, int]:
        """Per-rule hit counts from an evaluate() result."""
        counts: dict[str, int] = dict.fromkeys(self.rules, 0)
        for hit_list in evaluated["rule_hits"]:
            for rule in filter(None, hit_list.split(";")):
                counts[rule] += 1
        return counts
