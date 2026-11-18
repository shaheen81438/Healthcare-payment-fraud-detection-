"""Generate a local, reproducible research evidence package from frozen artifacts.

This script is read-only with respect to source datasets and models. It writes
only aggregate evidence under research/evidence and figures from matched demo
scores and labels. All evaluation labels in the demo bundle are synthetic.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EVIDENCE = ROOT / "research" / "evidence"
FIGURES = EVIDENCE / "FIGURES"
TABLES = EVIDENCE / "TABLES"
SCORE_COLUMNS = {
    "risk_score": "Ensemble",
    "supervised_probability": "Supervised probability",
    "anomaly_percentile": "Anomaly percentile",
    "rule_score": "Rule score",
}


def _font(size: int = 18) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def _save_plot(name: str, title: str, draw_content: object) -> None:
    width, height = 1600, 1000
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((90, 45), title, fill="#16324f", font=_font(38))
    draw.text(
        (90, 96),
        "Frozen synthetic evaluation bundle | aggregate results; no identifiers",
        fill="#536579",
        font=_font(20),
    )
    draw_content(draw, width, height)  # type: ignore[operator]
    image.save(FIGURES / f"{name}.png", dpi=(300, 300), optimize=True)


def _plot_lines(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    x_values: list[np.ndarray],
    y_values: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    x_label: str,
    y_label: str,
    diagonal: bool = False,
) -> None:
    left, top, right, bottom = 150, 190, width - 100, height - 160
    draw.line((left, top, left, bottom, right, bottom), fill="#243b53", width=3)
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        x = left + fraction * (right - left)
        y = bottom - fraction * (bottom - top)
        draw.line((x, top, x, bottom), fill="#e4eaf0", width=1)
        draw.line((left, y, right, y), fill="#e4eaf0", width=1)
        draw.text((x - 12, bottom + 16), f"{fraction:.2g}", fill="#536579", font=_font(17))
        draw.text((left - 52, y - 10), f"{fraction:.2g}", fill="#536579", font=_font(17))
    if diagonal:
        draw.line((left, bottom, right, top), fill="#9aa8b5", width=2)
    for i, (xs, ys, label, color) in enumerate(
        zip(x_values, y_values, labels, colors, strict=True)
    ):
        points = [
            (
                left + int(float(x) * (right - left)),
                bottom - int(float(y) * (bottom - top)),
            )
            for x, y in zip(xs, ys, strict=True)
        ]
        if len(points) > 1:
            draw.line(points, fill=color, width=5)
        draw.text((right - 350, top + 10 + i * 32), label, fill=color, font=_font(17))
    draw.text(((left + right) // 2 - 55, height - 75), x_label, fill="#243b53", font=_font(22))
    draw.text((left, top - 35), y_label, fill="#243b53", font=_font(22))


def _dataset_record(
    name: str, path: Path, classification: str, description: str
) -> dict[str, object]:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        return {
            "dataset": name,
            "classification": classification,
            "path": path.relative_to(ROOT).as_posix(),
            "format": path.suffix.lstrip(".") or "unknown",
            "bytes": path.stat().st_size,
            "rows": "",
            "columns": "",
            "missing_cells": "",
            "duplicate_rows": "",
            "label_column": "",
            "positive_rows": "",
            "negative_rows": "",
            "description": description,
        }
    label_col = "is_improper" if "is_improper" in frame else ""
    return {
        "dataset": name,
        "classification": classification,
        "path": path.relative_to(ROOT).as_posix(),
        "format": path.suffix.lstrip("."),
        "bytes": path.stat().st_size,
        "rows": len(frame),
        "columns": len(frame.columns),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "label_column": label_col,
        "positive_rows": int(frame[label_col].sum()) if label_col else "",
        "negative_rows": int((~frame[label_col]).sum()) if label_col else "",
        "description": description,
    }


def _write_dataset_column_audit(paths: list[tuple[str, Path, str]]) -> None:
    records: list[dict[str, object]] = []
    for name, path, kind in paths:
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        for column in frame.columns:
            records.append(
                {
                    "dataset": name,
                    "column": column,
                    "dtype": str(frame[column].dtype),
                    "missing_count": int(frame[column].isna().sum()),
                    "missing_pct": round(float(frame[column].isna().mean() * 100), 4),
                    "unique_count": int(frame[column].nunique(dropna=True)),
                    "classification": kind,
                }
            )
    pd.DataFrame(records).to_csv(TABLES / "DATASET_COLUMN_QUALITY.csv", index=False)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    predictions_path = ROOT / "demo_data" / "predictions.parquet"
    labels_path = ROOT / "demo_data" / "labels.parquet"
    predictions = pd.read_parquet(predictions_path)
    labels = pd.read_parquet(labels_path)
    if predictions["claim_id"].duplicated().any() or labels["claim_id"].duplicated().any():
        raise ValueError("Duplicate claim IDs make one-to-one evaluation invalid")
    joined = predictions.merge(
        labels[["claim_id", "is_improper"]],
        on="claim_id",
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(predictions):
        raise ValueError("Some scored rows lack a matching ground-truth label")
    y = joined["is_improper"].astype(int).to_numpy()
    if np.unique(y).size != 2:
        raise ValueError("ROC/PR metrics require both label classes")
    policy = json.loads((ROOT / "demo_data" / "policy.json").read_text(encoding="utf-8"))
    selected_threshold = float(policy["review_threshold"])
    thresholds = sorted({0.2, 0.5, 0.73, 0.9, selected_threshold})
    metric_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    for score_column, model_name in SCORE_COLUMNS.items():
        if score_column not in joined:
            continue
        scores = joined[score_column].astype(float).to_numpy()
        if not np.isfinite(scores).all():
            raise ValueError(f"Non-finite values in {score_column}")
        roc_auc = float(roc_auc_score(y, scores))
        pr_auc = float(average_precision_score(y, scores))
        operating_thresholds = thresholds if score_column == "risk_score" else [0.5]
        for threshold in operating_thresholds:
            predicted = (scores >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
            metric_rows.append(
                {
                    "evaluation": "frozen synthetic demo scores matched to synthetic labels",
                    "model_or_score": model_name,
                    "score_column": score_column,
                    "threshold": threshold,
                    "n": len(y),
                    "positive_count": int(y.sum()),
                    "negative_count": int((y == 0).sum()),
                    "accuracy": accuracy_score(y, predicted),
                    "precision": precision_score(y, predicted, zero_division=0),
                    "recall": recall_score(y, predicted, zero_division=0),
                    "f1": f1_score(y, predicted, zero_division=0),
                    "roc_auc": roc_auc,
                    "pr_auc_average_precision": pr_auc,
                    "mcc": matthews_corrcoef(y, predicted),
                    "true_positive": int(tp),
                    "true_negative": int(tn),
                    "false_positive": int(fp),
                    "false_negative": int(fn),
                    "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
                    "review_rate": float(predicted.mean()),
                }
            )
        predicted = (
            scores >= (selected_threshold if score_column == "risk_score" else 0.5)
        ).astype(int)
        comparison_rows.append(
            {
                "score_name": model_name,
                "score_column": score_column,
                "operating_threshold": selected_threshold if score_column == "risk_score" else 0.5,
                "roc_auc": roc_auc,
                "pr_auc_average_precision": pr_auc,
                "precision": precision_score(y, predicted, zero_division=0),
                "recall": recall_score(y, predicted, zero_division=0),
                "f1": f1_score(y, predicted, zero_division=0),
                "evidence_note": "score column evaluated on identical matched synthetic rows; not a fresh retraining or real-world validation",
            }
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(EVIDENCE / "MODEL_EVALUATION.csv", index=False, float_format="%.8f")
    pd.DataFrame(comparison_rows).to_csv(TABLES / "SCORE_COLUMN_COMPARISON.csv", index=False)

    dataset_paths = [
        ("demo claims", ROOT / "demo_data/claims.parquet", "Synthetic development data"),
        ("demo labels", labels_path, "Synthetic injected-pattern labels"),
        ("demo predictions", predictions_path, "Synthetic evaluation scores"),
        (
            "demo explanations",
            ROOT / "demo_data/explanations.parquet",
            "Synthetic-model SHAP explanation samples",
        ),
        (
            "CMS provider-service snapshot",
            ROOT / "data/reference/cms_provider_service_sample.csv",
            "Real public, provider-service aggregate sample",
        ),
        (
            "review decisions",
            ROOT / "data/processed/review_decisions.parquet",
            "Local review feedback; provenance/label semantics not independently established",
        ),
        (
            "LEIE test fixture",
            ROOT / "tests/fixtures/leie_fixture.csv",
            "Synthetic offline fixture; no real OIG records",
        ),
    ]
    dataset_rows = [
        _dataset_record(name, path, kind, kind)
        for name, path, kind in dataset_paths
        if path.is_file()
    ]
    pd.DataFrame(dataset_rows).to_csv(EVIDENCE / "DATASET_SUMMARY.csv", index=False)
    _write_dataset_column_audit(
        [(name, path, kind) for name, path, kind in dataset_paths if path.is_file()]
    )

    # Confusion matrix and ROC/PR curves use only the matching frozen ensemble scores.
    scores = joined["risk_score"].astype(float).to_numpy()
    predicted = (scores >= selected_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    cm = pd.DataFrame(
        [
            {"actual": "positive", "predicted_positive": int(tp), "predicted_negative": int(fn)},
            {"actual": "negative", "predicted_positive": int(fp), "predicted_negative": int(tn)},
        ]
    )
    cm.to_csv(TABLES / "CONFUSION_MATRIX_AT_POLICY_THRESHOLD.csv", index=False)
    (EVIDENCE / "EVALUATION_METADATA.json").write_text(
        json.dumps(
            {
                "evaluation_type": "frozen synthetic demonstration bundle matched to generated ground-truth labels",
                "n_scored_and_matched": len(joined),
                "n_positive": int(y.sum()),
                "n_negative": int((y == 0).sum()),
                "positive_rate": float(y.mean()),
                "score_column": "risk_score",
                "threshold": selected_threshold,
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "false_negative_rate": float(fn / (fn + tp)) if fn + tp else None,
                "service_date_min": str(joined["service_date"].min()),
                "service_date_max": str(joined["service_date"].max()),
                "model_version": ", ".join(sorted(joined["model_version"].astype(str).unique())),
                "synthetic_disclosure": "These results measure detection of generated pattern labels and are not verified Medicare fraud outcomes.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    curve_x_roc: list[np.ndarray] = []
    curve_y_roc: list[np.ndarray] = []
    curve_x_pr: list[np.ndarray] = []
    curve_y_pr: list[np.ndarray] = []
    curve_labels: list[str] = []
    for column, model_name in SCORE_COLUMNS.items():
        if column not in joined:
            continue
        column_scores = joined[column].astype(float).to_numpy()
        fpr, tpr, _ = roc_curve(y, column_scores)
        precision, recall, _ = precision_recall_curve(y, column_scores)
        curve_x_roc.append(fpr)
        curve_y_roc.append(tpr)
        curve_x_pr.append(recall)
        curve_y_pr.append(precision)
        short_name = {
            "Ensemble": "Ensemble",
            "Supervised probability": "Supervised",
            "Anomaly percentile": "Anomaly",
            "Rule score": "Rules",
        }[model_name]
        curve_labels.append(
            f"{short_name} ROC {roc_auc_score(y, column_scores):.3f} / AP {average_precision_score(y, column_scores):.3f}"
        )
    curve_colors = ["#2673b8", "#d45c30", "#4b8b63", "#8a62a9"]
    _save_plot(
        "ROC_CURVE_SYNTHETIC",
        "ROC curve comparison across saved score columns",
        lambda draw, w, h: _plot_lines(
            draw,
            w,
            h,
            curve_x_roc,
            curve_y_roc,
            curve_labels,
            curve_colors,
            "False positive rate",
            "True positive rate",
            True,
        ),
    )
    _save_plot(
        "PRECISION_RECALL_CURVE_SYNTHETIC",
        "Precision-recall comparison across saved score columns",
        lambda draw, w, h: _plot_lines(
            draw,
            w,
            h,
            curve_x_pr,
            curve_y_pr,
            curve_labels,
            curve_colors,
            "Recall",
            "Precision",
        ),
    )

    def draw_cm(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
        cells = [[tp, fn], [fp, tn]]
        row_labels = ["Actual positive", "Actual negative"]
        col_labels = ["Predicted positive", "Predicted negative"]
        x0, y0, cellw, cellh = 400, 300, 440, 210
        for j, label in enumerate(col_labels):
            draw.text((x0 + j * cellw + 25, y0 - 55), label, fill="#243b53", font=_font(20))
        for i, label in enumerate(row_labels):
            draw.text((100, y0 + i * cellh + 70), label, fill="#243b53", font=_font(22))
            for j, value in enumerate(cells[i]):
                x, ycoord = x0 + j * cellw, y0 + i * cellh
                draw.rectangle(
                    (x, ycoord, x + cellw - 10, ycoord + cellh - 10),
                    fill="#e7f0f8",
                    outline="#2673b8",
                    width=3,
                )
                names = [["TP", "FN"], ["FP", "TN"]]
                draw.text(
                    (x + 150, ycoord + 60),
                    f"{names[i][j]}: {value:,}",
                    fill="#16324f",
                    font=_font(34),
                )
        draw.text(
            (380, 790),
            f"Threshold = {selected_threshold:g} | FNR = {fn / (fn + tp):.3%}",
            fill="#9a3412",
            font=_font(28),
        )

    _save_plot(
        "CONFUSION_MATRIX_POLICY_THRESHOLD", "Confusion matrix at configured review cutoff", draw_cm
    )

    shap_path = ROOT / "docs/assets/global_importance.csv"
    if shap_path.is_file():
        shap = pd.read_csv(shap_path).sort_values("mean_abs_shap", ascending=False).head(12)
        shap.to_csv(TABLES / "SAVED_SHAP_GLOBAL_IMPORTANCE_TOP12.csv", index=False)

        # This is a raster bar chart from the saved project artifact, not recomputed SHAP.
        def draw_shap(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
            maximum = float(shap["mean_abs_shap"].max()) or 1.0
            for i, row in enumerate(shap.itertuples(index=False)):
                ycoord = 170 + i * 58
                draw.text((70, ycoord), str(row.feature)[:38], fill="#243b53", font=_font(18))
                bar_start, bar_max = 570, 830
                bar_width = int(float(row.mean_abs_shap) / maximum * bar_max)
                draw.rectangle(
                    (bar_start, ycoord + 3, bar_start + bar_width, ycoord + 34), fill="#3983b7"
                )
                draw.text(
                    (bar_start + bar_width + 8, ycoord + 2),
                    f"{row.mean_abs_shap:.3f}",
                    fill="#243b53",
                    font=_font(17),
                )

        _save_plot(
            "SAVED_SHAP_GLOBAL_IMPORTANCE", "Saved global SHAP mean absolute importance", draw_shap
        )

    cms = pd.read_csv(ROOT / "data/reference/cms_provider_service_sample.csv")
    cms_rows = len(cms)
    cms_dupes = int(cms.duplicated().sum())
    cms_normalized = cms.rename(
        columns={
            "Rndrng_NPI": "npi",
            "Rndrng_Prvdr_Type": "provider_type",
            "Tot_Srvcs": "total_services",
            "Avg_Mdcr_Pymt_Amt": "avg_paid",
            "Avg_Mdcr_Alowd_Amt": "avg_allowed",
            "Avg_Sbmtd_Chrg": "avg_submitted",
            "HCPCS_Cd": "hcpcs",
        }
    )
    from src.payment_integrity.improper_payment import analyze_payments
    from src.payment_integrity.provider_anomaly import detect_provider_anomalies

    payment = analyze_payments(pd.read_parquet(ROOT / "demo_data/claims.parquet"))
    anomaly = detect_provider_anomalies(cms_normalized)
    components = pd.DataFrame(
        [
            {
                "component": "Improper payment indicators",
                "input": "demo_data/claims.parquet synthetic claim development sample",
                "rows": len(payment),
                "result": "0 rows triggered; required expected payment/documentation evidence fields are not present in this synthetic claims sample",
                "interpretation": "No evidence of performance: rule inputs are not available in evaluated artifact.",
            },
            {
                "component": "Provider anomaly ranking",
                "input": "CMS 480-row public provider-service aggregate snapshot",
                "rows": len(anomaly),
                "result": f"{int((anomaly.investigation_priority == 'high').sum())} high, {int((anomaly.investigation_priority == 'medium').sum())} medium, {int((anomaly.investigation_priority == 'routine').sum())} routine",
                "interpretation": "Unsupervised review ranking only; there are no provider fraud labels.",
            },
            {
                "component": "OIG LEIE matching",
                "input": "six-row synthetic offline test fixture",
                "rows": 6,
                "result": "Not evaluated on official OIG records; fixture contains reserved synthetic NPIs only.",
                "interpretation": "Fixture matching tests code paths and cannot estimate operational match accuracy.",
            },
            {
                "component": "Combined risk aggregation",
                "input": "component aggregation function",
                "rows": "Not applicable",
                "result": "Code supports availability-aware normalized signals; no eligible real adjudicated outcome set available.",
                "interpretation": "Combined score is an investigation priority, not a calibrated fraud probability.",
            },
        ]
    )
    components.to_csv(TABLES / "IMPLEMENTED_COMPONENTS_AND_RESULTS.csv", index=False)
    anomaly["priority_band"] = anomaly["investigation_priority"]
    anomaly["priority_band"].value_counts().rename_axis("priority_band").reset_index(
        name="providers"
    ).to_csv(TABLES / "CMS_PROVIDER_ANOMALY_DISTRIBUTION.csv", index=False)

    def draw_anomaly(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
        counts = (
            anomaly["priority_band"]
            .value_counts()
            .reindex(["routine", "medium", "high"], fill_value=0)
        )
        colors = ["#3b82a0", "#e3a53a", "#bd4b40"]
        max_count = max(1, int(counts.max()))
        for idx, (label, value) in enumerate(counts.items()):
            ycoord = 210 + idx * 190
            bar_width = int(int(value) / max_count * 900)
            draw.text((90, ycoord + 30), str(label).title(), fill="#243b53", font=_font(28))
            draw.rectangle((360, ycoord, 360 + bar_width, ycoord + 100), fill=colors[idx])
            draw.text(
                (380 + bar_width, ycoord + 30),
                f"{int(value)} provider groups",
                fill="#243b53",
                font=_font(24),
            )
        draw.text(
            (90, 830),
            "Peer-group anomaly priorities; not confirmed fraud labels",
            fill="#9a3412",
            font=_font(25),
        )

    _save_plot(
        "CMS_PROVIDER_ANOMALY_PRIORITY_DISTRIBUTION",
        "CMS aggregate peer anomaly review priorities",
        draw_anomaly,
    )

    def draw_architecture(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
        boxes = [
            (80, 250, 350, 390, "CMS aggregate data"),
            (80, 460, 350, 600, "OIG LEIE input"),
            (80, 670, 350, 810, "Synthetic claims + labels"),
            (520, 300, 820, 450, "Validation + provenance"),
            (520, 555, 820, 705, "Feature / rule / peer engines"),
            (1010, 250, 1330, 390, "Claim model + SHAP"),
            (1010, 470, 1330, 610, "LEIE + payment screening"),
            (1010, 690, 1330, 830, "Provider anomaly ranking"),
            (1410, 430, 1540, 650, "Streamlit / API\nreview priority"),
        ]
        for x1, y1, x2, y2, label in boxes:
            draw.rounded_rectangle(
                (x1, y1, x2, y2), radius=18, fill="#e8f1f8", outline="#2673b8", width=3
            )
            draw.multiline_text(
                (x1 + 20, y1 + 45),
                label,
                fill="#16324f",
                font=_font(21),
                spacing=10,
                align="center",
            )
        arrows = [
            ((350, 320), (520, 350)),
            ((350, 530), (520, 390)),
            ((350, 740), (520, 650)),
            ((670, 450), (670, 555)),
            ((820, 620), (1010, 320)),
            ((820, 620), (1010, 540)),
            ((820, 390), (1010, 760)),
            ((1330, 320), (1410, 490)),
            ((1330, 540), (1410, 540)),
            ((1330, 760), (1410, 600)),
        ]
        for (x1, y1), (x2, y2) in arrows:
            draw.line((x1, y1, x2, y2), fill="#536579", width=4)
        draw.text(
            (430, 900),
            "Research framing: outputs prioritize human review; no engine establishes fraud.",
            fill="#536579",
            font=_font(22),
        )

    _save_plot(
        "SYSTEM_ARCHITECTURE_EVIDENCE",
        "Implemented payment-integrity system components",
        draw_architecture,
    )
    audit = {
        "evaluation command": f"{sys.executable} scripts/generate_research_evidence.py",
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": __import__("sklearn").__version__,
        "Pillow": __import__("PIL").__version__,
        "evaluation rows": len(joined),
        "evaluation positives": int(y.sum()),
        "CMS snapshot rows": cms_rows,
        "CMS duplicate rows": cms_dupes,
        "model artifacts": "No usable trained model bundle found; models/latest.json points outside repository to a temporary test bundle.",
    }
    (EVIDENCE / "GENERATION_METADATA.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    print(metrics[metrics["score_column"].eq("risk_score")].to_string(index=False))


if __name__ == "__main__":
    main()
