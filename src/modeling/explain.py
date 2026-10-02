"""SHAP explanations and reason-code mapping.

Per claim we return signed SHAP contributions aggregated back to the raw
feature level (one-hot columns fold into their parent categorical), the top
drivers, and investigator-facing reason codes that merge SHAP drivers with
deterministic rule hits. Global summaries feed the model card.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.modeling.bundle import ScoringBundle

logger = logging.getLogger(__name__)


def load_reason_config(path: str = "configs/reason_codes.yaml") -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)
    return cfg


class ClaimExplainer:
    """SHAP TreeExplainer over the bundle's XGBoost model, with reason mapping."""

    def __init__(self, bundle: ScoringBundle, reason_config: dict[str, Any] | None = None) -> None:
        import shap

        self.bundle = bundle
        self.explainer = shap.TreeExplainer(bundle.xgb_model)
        self.reason_cfg = reason_config or load_reason_config()
        self._parents = self._parent_map(bundle.feature_names_out)

    def _parent_map(self, names_out: list[str]) -> list[str]:
        """Map each post-encoding column to its raw parent feature name."""
        parents: list[str] = []
        cats = sorted(self.bundle.categorical_features, key=len, reverse=True)
        raw = set(self.bundle.numeric_features)
        for name in names_out:
            if name in raw:
                parents.append(name)
                continue
            parent = next((c for c in cats if name.startswith(f"{c}_")), name)
            parents.append(parent)
        return parents

    # ------------------------------------------------------------------ core
    def shap_frame(self, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """Aggregated (parent-level) SHAP contributions for each row of X."""
        matrix = self.bundle.preprocessor.transform(X)
        values = self.explainer.shap_values(matrix)
        base = float(np.ravel(self.explainer.expected_value)[0])
        frame = pd.DataFrame(values, columns=self.bundle.feature_names_out, index=X.index)
        agg = frame.T.groupby(np.array(self._parents)).sum().T  # fold OHE into parents
        return agg, base

    def top_contributions(
        self, X: pd.DataFrame, top_k: int = 6
    ) -> tuple[list[list[dict[str, Any]]], float]:
        """Per-row top-k signed contributions with raw feature values."""
        agg, base = self.shap_frame(X)
        cols = list(agg.columns)
        vals = agg.to_numpy()
        raw_values = {c: X[c].to_numpy() for c in cols if c in X.columns}
        k = min(top_k, vals.shape[1])
        # Pick the top k by |contribution| with argpartition, then sort those k exactly.
        part = np.argpartition(-np.abs(vals), k - 1, axis=1)[:, :k]
        results: list[list[dict[str, Any]]] = []
        for r in range(vals.shape[0]):
            idx_sorted = part[r][np.argsort(-np.abs(vals[r, part[r]]))]
            results.append(
                [
                    {
                        "feature": cols[j],
                        "contribution": round(float(vals[r, j]), 5),
                        "value": _serialize(raw_values[cols[j]][r])
                        if cols[j] in raw_values
                        else None,
                    }
                    for j in idx_sorted
                ]
            )
        return results, base

    # ------------------------------------------------------------------ reasons
    def reason_codes(
        self,
        contributions: list[dict[str, Any]],
        rule_reasons: str = "",
        anomaly_percentile: float = 0.0,
        max_codes: int = 5,
    ) -> list[str]:
        """Merge rule reasons with SHAP-derived codes (rules first, deduped)."""
        codes: list[str] = [r for r in rule_reasons.split(";") if r]
        mapping = self.reason_cfg["reason_codes"]
        for item in contributions:
            if item["contribution"] <= 0:
                continue
            entry = mapping.get(item["feature"])
            if entry:
                codes.append(str(entry["code"]))
        if anomaly_percentile >= 0.95 and self.bundle.ensemble_weight < 1.0:
            codes.append(str(self.reason_cfg["anomaly_reason"]["code"]))
        deduped = list(dict.fromkeys(codes))
        return deduped[:max_codes]

    def reason_labels(self) -> dict[str, str]:
        """Map each reason code to its readable label (for the API and dashboard)."""
        labels: dict[str, str] = {}
        for entry in self.reason_cfg["reason_codes"].values():
            labels.setdefault(str(entry["code"]), str(entry["label"]))
        anomaly = self.reason_cfg["anomaly_reason"]
        labels.setdefault(str(anomaly["code"]), str(anomaly["label"]))
        for code, label in self.reason_cfg.get("rule_reason_labels", {}).items():
            labels.setdefault(str(code), str(label))
        return labels

    # ------------------------------------------------------------------ global
    def global_importance(self, X: pd.DataFrame, max_rows: int = 5000) -> pd.DataFrame:
        """Mean |SHAP| per parent feature over a sample (for the model card)."""
        sample = X.sample(min(max_rows, len(X)), random_state=42) if len(X) > max_rows else X
        agg, _ = self.shap_frame(sample)
        importance = agg.abs().mean().sort_values(ascending=False)
        return importance.rename("mean_abs_shap").rename_axis("feature").reset_index()


def _serialize(value: Any) -> Any:
    if isinstance(value, np.integer | np.floating):
        return round(float(value), 4)
    return value


def plot_global_importance(importance: pd.DataFrame, out_path: str, top_n: int = 15) -> None:
    """Horizontal bar chart of mean |SHAP|. It shows magnitude only, so it uses one sequential hue."""
    import matplotlib.pyplot as plt

    from src.utils.viz import INK_SECONDARY, SEQUENTIAL_BLUES, apply_mpl_style, style_axis

    apply_mpl_style()
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.34 * top_n + 1.2))
    style_axis(ax)
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    ax.barh(top["feature"], top["mean_abs_shap"], color=SEQUENTIAL_BLUES[4], height=0.62)
    for y_pos, mean_abs in enumerate(top["mean_abs_shap"]):
        ax.annotate(
            f"{mean_abs:.3f}",
            (mean_abs, y_pos),
            textcoords="offset points",
            xytext=(4, -3),
            fontsize=8,
            color=INK_SECONDARY,
        )
    ax.set_xlabel("Mean |SHAP| (log-odds)")
    ax.set_title("Global feature importance (claims-risk model)")
    fig.savefig(out_path)
    plt.close(fig)
