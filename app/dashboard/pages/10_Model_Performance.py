"""Performance views derived from scored claims and matched synthetic outcomes."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.dashboard.common import page_setup, synthetic_banner
from app.dashboard.model_performance import (
    calculate_threshold_metrics,
    load_model_performance,
    threshold_analysis,
)

page_setup("Model Performance")
synthetic_banner()
st.caption(
    "Metrics are calculated from available scored claims joined to their outcome labels. "
    "Public CMS provider aggregates do not contain claim-level outcomes and are not used as labels."
)

result = load_model_performance()
if not result["available"]:
    st.info(f"Not evaluated. {result['message']}")
    st.caption("Run the evaluation pipeline to create scored test data and matching labels.")
    st.stop()

st.warning(
    "These results measure synthetic injected patterns only. They are not estimates of "
    "real-world Medicare fraud or improper-payment performance."
)
st.caption(
    f"{result['evaluation_kind']} · model {result['model_version']} · "
    f"{result['dataset_size']:,} labeled claims · score cutoff {result['threshold']:.4f}"
)

metrics = result["metrics"]
st.subheader("Interactive threshold simulator")
threshold = st.slider(
    "Risk score cutoff",
    min_value=0.0,
    max_value=1.0,
    value=float(result["threshold"]),
    step=0.0005,
    format="%.4f",
    help="Scores at or above this cutoff are treated as review candidates for metric calculation.",
)
threshold_metrics = calculate_threshold_metrics(
    result["y_true"], result["selected_scores"], threshold
)
st.caption(
    "Changing the cutoff changes only the predicted review set. Ground-truth labels and model scores "
    "remain fixed. The cutoff is a review-priority setting, not a fraud-probability threshold."
)
cards = st.columns(6)
for slot, (label, key) in zip(
    cards,
    [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-score", "f1"),
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
    ],
    strict=True,
):
    value = threshold_metrics[key] if key in threshold_metrics else metrics[key]
    slot.metric(label, f"{value:.3f}")

st.caption(
    f"Accuracy, precision, recall, F1, and the confusion matrix use the selected score cutoff "
    f"of {threshold:.4f}. The comparison table uses a 0.50 cutoff for each score. "
    "Precision and recall describe "
    "the flagged set and labeled positive set. ROC-AUC and PR-AUC measure ranking across cutoffs; "
    "PR-AUC is especially informative for rare outcomes."
)
fnr = float(threshold_metrics["false_negative_rate"])
st.error(
    f"False-negative rate: {fnr:.1%} ({threshold_metrics['false_negative']:,} of "
    f"{result['positive_count']:,} labeled positive claims were below this cutoff)."
)
st.markdown(
    "In payment-integrity review, false negatives are labeled positive examples that the selected "
    "cutoff leaves out of the review queue. They represent missed opportunities to investigate "
    "potential improper payments. With rare positives, accuracy can remain high while many positives "
    "are missed; review recall and false-negative rate alongside investigator capacity. These labels "
    "are synthetic injections, so this rate does not estimate missed real-world Medicare fraud."
)

left, right = st.columns([1.25, 1])
with left:
    st.subheader("Model comparison")
    st.caption(
        "Threshold metrics use a 0.50 score cutoff for each model; ranking metrics are threshold-free."
    )
    st.dataframe(
        result["model_comparison"].style.format(
            "{:.3f}", subset=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC"]
        ),
        use_container_width=True,
        hide_index=True,
    )
    comparison = result["model_comparison"].melt(
        id_vars="Model", value_vars=["ROC-AUC", "PR-AUC"], var_name="Metric", value_name="Score"
    )
    st.plotly_chart(
        px.bar(
            comparison,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            range_y=[0, 1],
            title="Ranking metrics by model",
        ),
        use_container_width=True,
    )
with right:
    st.subheader("Confusion matrix")
    matrix = [
        [threshold_metrics["true_negative"], threshold_metrics["false_positive"]],
        [threshold_metrics["false_negative"], threshold_metrics["true_positive"]],
    ]
    st.plotly_chart(
        go.Figure(
            go.Heatmap(
                z=matrix,
                x=["Predicted routine", "Predicted review"],
                y=["Labeled routine", "Labeled positive"],
                text=matrix,
                texttemplate="%{text:,}",
                colorscale=[[0, "#e8f1f6"], [1, "#246b8e"]],
                showscale=False,
            )
        ).update_layout(title=f"{result['score_label']} at cutoff {threshold:.4f}"),
        use_container_width=True,
    )
    counts = st.columns(4)
    for slot, key, label in zip(
        counts,
        ["true_positive", "true_negative", "false_positive", "false_negative"],
        ["TP", "TN", "FP", "FN"],
        strict=True,
    ):
        slot.metric(label, f"{threshold_metrics[key]:,}")
    distribution = pd.DataFrame(
        {
            "Class": ["Labeled routine", "Labeled positive"],
            "Claims": [result["negative_count"], result["positive_count"]],
        }
    )
    st.plotly_chart(
        px.bar(
            distribution,
            x="Class",
            y="Claims",
            title=f"Class distribution · positive rate {result['positive_rate']:.1%}",
        ),
        use_container_width=True,
    )

st.subheader("Performance across thresholds")
threshold_values = np.unique(
    np.append(np.arange(0.0, 1.0001, 0.05), [result["threshold"], threshold])
)
sweep = threshold_analysis(result["y_true"], result["selected_scores"], threshold_values)
st.dataframe(
    sweep.style.format(
        {
            "threshold": "{:.4f}",
            "accuracy": "{:.3f}",
            "precision": "{:.3f}",
            "recall": "{:.3f}",
            "f1": "{:.3f}",
            "false_negative_rate": "{:.3f}",
            "review_rate": "{:.3f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)
threshold_lines = sweep.melt(
    id_vars="threshold",
    value_vars=["recall", "false_negative_rate", "review_rate"],
    var_name="Measure",
    value_name="Rate",
)
st.plotly_chart(
    px.line(
        threshold_lines,
        x="threshold",
        y="Rate",
        color="Measure",
        range_y=[0, 1],
        labels={"threshold": "Score cutoff", "Rate": "Share of claims"},
        title="Review recall, false-negative rate, and review volume",
    ),
    use_container_width=True,
)

curve = result["precision_recall_curve"]
if not curve.empty:
    if len(curve) > 3000:
        curve = curve.iloc[:: max(1, len(curve) // 3000)]
    st.subheader("Precision-recall curve")
    st.plotly_chart(
        px.line(
            curve,
            x="recall",
            y="precision",
            range_x=[0, 1],
            range_y=[0, 1],
            labels={"recall": "Recall", "precision": "Precision"},
            title=f"Precision-recall curve · AUC {metrics['pr_auc']:.3f}",
        ),
        use_container_width=True,
    )

roc = result["roc_curve"]
if not roc.empty:
    st.subheader("ROC curve")
    roc_plot = roc.copy()
    if len(roc_plot) > 3000:
        roc_plot = roc_plot.iloc[:: max(1, len(roc_plot) // 3000)]
    roc_plot = pd.concat(
        [roc_plot, pd.DataFrame([{"false_positive_rate": 1.0, "true_positive_rate": 1.0}])],
        ignore_index=True,
    )
    st.plotly_chart(
        px.line(
            roc_plot,
            x="false_positive_rate",
            y="true_positive_rate",
            range_x=[0, 1],
            range_y=[0, 1],
            labels={
                "false_positive_rate": "False-positive rate",
                "true_positive_rate": "True-positive rate",
            },
            title=f"ROC curve · AUC {metrics['roc_auc']:.3f}",
        ).add_scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Chance reference", line={"dash": "dash"}
        ),
        use_container_width=True,
    )

st.subheader("Feature importance")
if result["feature_importance_available"]:
    importance = result["feature_importance"]
    st.caption(
        "Mean absolute SHAP importance from the saved global explanation summary. "
        "That artifact does not record a model version and may not match this score bundle."
    )
    st.plotly_chart(
        px.bar(
            importance.sort_values("mean_abs_shap"),
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            title="Top features by mean absolute SHAP value",
        ),
        use_container_width=True,
    )
else:
    st.info("Feature importance artifact is unavailable.")
