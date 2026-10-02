"""Build the evaluation report: charts in the project chart style, plus markdown."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.utils.viz import (
    BASELINE,
    CLASS_COLORS,
    INK_MUTED,
    INK_SECONDARY,
    MODEL_COLORS,
    MODEL_LABELS,
    apply_mpl_style,
    style_axis,
)

logger = logging.getLogger(__name__)


def _fig(width: float = 7.0, height: float = 4.2) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width, height))
    style_axis(ax)
    return fig, ax


def plot_pr_curves(y: np.ndarray, scores_by_model: dict[str, np.ndarray], out: Path) -> None:
    apply_mpl_style()
    fig, ax = _fig()
    for name, scores in scores_by_model.items():
        precision, recall, _ = precision_recall_curve(y, scores)
        ax.plot(recall, precision, color=MODEL_COLORS[name], label=MODEL_LABELS[name])
    base_rate = float(np.mean(y))
    ax.axhline(base_rate, color=BASELINE, linestyle="--", linewidth=1)
    ax.annotate(
        f"base rate {base_rate:.1%}",
        (0.02, base_rate),
        textcoords="offset points",
        xytext=(0, 5),
        fontsize=8,
        color=INK_MUTED,
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves (test window)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right")
    fig.savefig(out)
    import matplotlib.pyplot as plt

    plt.close(fig)


def plot_capacity_recall(
    y: np.ndarray, scores_by_model: dict[str, np.ndarray], out: Path, max_share: float = 0.15
) -> None:
    apply_mpl_style()
    fig, ax = _fig()
    shares = np.linspace(0.002, max_share, 60)
    for name, scores in scores_by_model.items():
        order = np.argsort(-scores, kind="stable")
        y_sorted = y[order]
        total = max(1, int(y.sum()))
        recalls = [y_sorted[: max(1, int(len(y) * s))].sum() / total for s in shares]
        ax.plot(
            shares * 100,
            np.array(recalls) * 100,
            color=MODEL_COLORS[name],
            label=MODEL_LABELS[name],
        )
    ax.set_xlabel("Share of claims reviewed (%)")
    ax.set_ylabel("Improper claims captured (%)")
    ax.set_title("Recall vs investigator capacity")
    ax.legend(loc="lower right")
    fig.savefig(out)
    import matplotlib.pyplot as plt

    plt.close(fig)


def plot_reliability(y: np.ndarray, probs: np.ndarray, out: Path) -> None:
    apply_mpl_style()
    fig, ax = _fig(5.4, 4.6)
    bins = np.clip((probs * 10).astype(int), 0, 9)
    frame = pd.DataFrame({"bin": bins, "p": probs, "y": y})
    grouped = frame.groupby("bin").agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    ax.plot([0, 1], [0, 1], color=BASELINE, linestyle="--", linewidth=1)
    ax.plot(
        grouped["pred"], grouped["obs"], color=MODEL_COLORS["ensemble"], marker="o", markersize=5
    )
    for _, row in grouped.iterrows():
        ax.annotate(
            f"n={int(row['n']):,}",
            (row["pred"], row["obs"]),
            textcoords="offset points",
            xytext=(6, -9),
            fontsize=7,
            color=INK_MUTED,
        )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed improper rate")
    ax.set_title("Calibration (test window)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(out)
    import matplotlib.pyplot as plt

    plt.close(fig)


def plot_env_curve(sweep: pd.DataFrame, policy: dict[str, Any], out: Path) -> None:
    apply_mpl_style()
    fig, ax = _fig()
    ax.plot(
        sweep["threshold"],
        sweep["net_value"],
        color=MODEL_COLORS["ensemble"],
        label="Realized net value",
    )
    ax.plot(
        sweep["threshold"],
        sweep["expected_net_value"],
        color=INK_MUTED,
        linewidth=1.4,
        linestyle=":",
        label="Expected net value",
    )
    ax.axhline(0, color=BASELINE, linewidth=1)
    tau = float(policy["review_threshold"])
    ax.axvline(tau, color=INK_SECONDARY, linestyle="--", linewidth=1)
    ax.annotate(
        f"operating point τ={tau:.2f}",
        (tau, ax.get_ylim()[1] * 0.92),
        textcoords="offset points",
        xytext=(6, 0),
        fontsize=8,
        color=INK_SECONDARY,
    )
    ax.set_xlabel("Review threshold on risk score")
    ax.set_ylabel("Net value (USD, test window)")
    ax.set_title("Net value vs review threshold")
    ax.legend(loc="lower left")
    fig.savefig(out)
    import matplotlib.pyplot as plt

    plt.close(fig)


def plot_score_distribution(y: np.ndarray, scores: np.ndarray, out: Path) -> None:
    apply_mpl_style()
    fig, ax = _fig()
    bins = np.linspace(0, 1, 41)
    ax.hist(
        scores[y == 0],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        color=CLASS_COLORS["clean"],
        label="Not improper",
    )
    ax.hist(
        scores[y == 1],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        color=CLASS_COLORS["improper"],
        label="Improper (injected)",
    )
    ax.set_xlabel("Ensemble risk score")
    ax.set_ylabel("Density")
    ax.set_title("Risk-score distribution by class")
    ax.legend(loc="upper center")
    fig.savefig(out)
    import matplotlib.pyplot as plt

    plt.close(fig)


def plot_novel_scheme(novel: dict[str, Any], out: Path) -> None:
    apply_mpl_style()
    fig, ax = _fig(6.6, 3.8)
    paths = ["supervised", "ensemble", "anomaly"]
    colors = [MODEL_COLORS["xgboost"], MODEL_COLORS["ensemble"], MODEL_COLORS["isolation_forest"]]
    labels = ["Supervised only", "Ensemble", "Anomaly only"]
    values = [novel[f"novel_recall_at_10pct_{p}"] * 100 for p in paths]
    bars = ax.bar(labels, values, color=colors, width=0.55)
    for rect, recall_pct in zip(bars, values, strict=True):
        ax.annotate(
            f"{recall_pct:.1f}%",
            (rect.get_x() + rect.get_width() / 2, recall_pct),
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK_SECONDARY,
        )
    ax.set_ylabel("Recall of held-out schemes @ top 10% (%)")
    ax.set_title("Novel-scheme holdout: what the anomaly path adds")
    fig.savefig(out)
    import matplotlib.pyplot as plt

    plt.close(fig)


def _table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def build_report(
    context: dict[str, Any],
    out_md: Path = Path("docs/evaluation_report.md"),
    assets_rel: str = "assets",
) -> None:
    """Write docs/evaluation_report.md from the computed context and saved charts."""
    metrics_table = context["metrics_table"]
    cm = context["confusion"]
    lines = [
        "# Evaluation report: claims-risk " + context["model_version"],
        "",
        f"Generated {context['generated_at']} · dataset `{context['dataset_version']}` · "
        f"test window {context['test_window']} ({context['n_test']:,} claims, "
        f"{context['test_positive_rate']:.2%} labeled improper).",
        "",
        "> Labels are synthetic injected patterns (see `labeling.md`). Every number below",
        "> measures detection of documented injections, not real-world fraud performance.",
        "",
        "## Model comparison (test window)",
        "",
        _table(metrics_table),
        "",
        f"![PR curves]({assets_rel}/pr_curves.png)",
        "",
        f"![Capacity recall]({assets_rel}/capacity_recall.png)",
        "",
        "## Operating point",
        "",
        f"Review threshold **τ = {context['policy']['review_threshold']}** "
        f"(max of ENV-optimal τ {context['policy']['env_optimal_threshold']}, "
        f"capacity-implied τ {context['policy']['capacity_implied_threshold']} at "
        f"{context['policy']['capacity_share']:.1%} capacity, and the "
        f"{context['policy']['cost_params']['min_risk_threshold']} risk floor).",
        "",
        f"Confusion matrix @ τ: TP {cm['tp']:,} · FP {cm['fp']:,} · FN {cm['fn']:,} · "
        f"TN {cm['tn']:,}; precision {context['at_threshold']['precision']:.3f}, "
        f"recall {context['at_threshold']['recall']:.3f}, F1 {context['at_threshold']['f1']:.3f}.",
        "",
        f"![Net value curve]({assets_rel}/env_curve.png)",
        "",
        "## Financial value under capacity scenarios (test window, realized)",
        "",
        _table(context["capacity_table"]),
        "",
        "Cost assumptions: review ${review_cost}/claim, recovery rate {recovery_rate:.0%}, "
        "false-positive overhead ${fp_extra_cost}/claim (configs/costs.yaml). These are "
        "scenario parameters, not validated actuals.".format(**context["policy"]["cost_params"]),
        "",
        "## Calibration",
        "",
        f"Brier {context['calibration']['brier']:.4f} · ECE {context['calibration']['ece']:.4f} "
        "(isotonic-calibrated supervised probability).",
        "",
        f"![Reliability]({assets_rel}/reliability.png)",
        "",
        f"![Score distribution]({assets_rel}/score_distribution.png)",
        "",
        "## Novel-scheme holdout",
        "",
        "Retrained with F09 (dx mismatch) and F11 (collusion) removed from training labels, "
        "then measured on exactly those claims in test:",
        "",
        _table(context["novel_table"]),
        "",
        f"![Novel scheme]({assets_rel}/novel_scheme.png)",
        "",
        "The unsupervised path catches part of the schemes the supervised model has never "
        "seen, but it still misses most of them. No single detector is enough; this argues "
        "for layered controls and ongoing rule development.",
        "",
        "## Per-pattern detection (ensemble, test window)",
        "",
        _table(context["pattern_table"]),
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    logger.info("evaluation_report_written", extra={"path": str(out_md)})
