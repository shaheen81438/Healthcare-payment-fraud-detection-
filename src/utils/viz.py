"""Shared color tokens and matplotlib style (light mode).

Categorical hues use a fixed slot order, checked so neighboring colors stay
distinct for color-vision deficiency. Sequential scales use one blue ramp. Text
uses ink colors, never series colors. The grid stays light and charts use a
single axis. Charts with hues below 3:1 contrast get direct labels or a table
beside them (the relief rule).
"""

from __future__ import annotations

from typing import Any

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

#: Fixed categorical slots, so each model keeps the same color in every chart.
MODEL_COLORS: dict[str, str] = {
    "ensemble": "#2a78d6",  # slot 1 blue
    "xgboost": "#eb6834",  # slot 2 orange
    "logistic_regression": "#1baf7a",  # slot 3 aqua
    "rules_baseline": "#eda100",  # slot 4 yellow
    "isolation_forest": "#e87ba4",  # slot 5 magenta
}
MODEL_LABELS: dict[str, str] = {
    "ensemble": "Ensemble",
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic regression",
    "rules_baseline": "Rules baseline",
    "isolation_forest": "Isolation Forest",
}
CLASS_COLORS = {"clean": "#2a78d6", "improper": "#eb6834"}
SEQUENTIAL_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def apply_mpl_style() -> None:
    """Apply the light-mode chart style to matplotlib."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.size": 10,
            "text.color": INK_PRIMARY,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 2.0,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )


def style_axis(ax: Any) -> None:
    """Apply the light grid and axis styling to one axes object."""
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(length=0)
