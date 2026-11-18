"""Dataset discovery and aggregate quality checks for the dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.data.provenance import file_provenance

SOURCES = {
    "Synthetic claims": ("data/processed/claims.parquet", "demo_data/claims.parquet", "synthetic"),
    "CMS provider-service sample": (
        "data/reference/cms_provider_service_sample.csv",
        "data/reference/cms_provider_service_sample.csv",
        "real_public_sample",
    ),
    "OIG LEIE fixture": (
        "data/reference/leie.csv",
        "tests/fixtures/leie_fixture.csv",
        "synthetic_fixture",
    ),
    "Scored synthetic claims": (
        "data/processed/scored_test.parquet",
        "demo_data/predictions.parquet",
        "synthetic_evaluation_sample",
    ),
}


def load_provenance_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest = Path(path)
    if not manifest.is_file():
        return []
    try:
        records = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return (
        [record for record in records if isinstance(record, dict)]
        if isinstance(records, list)
        else []
    )


def inspect_frame(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    source_path: str | Path,
    classification: str,
    provenance_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Calculate aggregate quality indicators; never return any source row values."""
    path = Path(source_path)
    missing = frame.isna().sum().astype(int)
    duplicate_rows = int(frame.duplicated().sum())
    duplicate_percent = 100.0 * duplicate_rows / len(frame) if len(frame) else 0.0
    invalid: dict[str, int] = {}
    for column in (
        "submitted_amount",
        "allowed_amount",
        "paid_amount",
        "avg_submitted",
        "avg_allowed",
        "avg_paid",
        "total_services",
    ):
        if column in frame:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            invalid[f"invalid_{column}"] = int(
                (numeric.isna() & frame[column].notna()).sum() + (numeric < 0).sum()
            )
    for column in ("service_date", "submit_date", "exclusion_date", "reinstatement_date"):
        if column in frame:
            parsed = pd.to_datetime(frame[column], errors="coerce", format="mixed")
            invalid[f"invalid_{column}"] = int(
                (parsed.isna() & frame[column].notna() & (frame[column].astype(str) != "")).sum()
            )
    npi_column = "npi" if "npi" in frame else "NPI" if "NPI" in frame else None
    if npi_column:
        invalid["invalid_npi"] = int(
            (~frame[npi_column].astype(str).str.fullmatch(r"\d{10}")).sum()
        )
    if {"paid_amount", "allowed_amount"} <= set(frame.columns):
        paid = pd.to_numeric(frame["paid_amount"], errors="coerce")
        allowed = pd.to_numeric(frame["allowed_amount"], errors="coerce")
        invalid["paid_amount_over_allowed"] = int((paid > allowed + 0.01).sum())

    required_by_kind = {
        "synthetic": {"claim_id", "provider_id", "paid_amount", "service_date"},
        "synthetic_evaluation_sample": {"claim_id", "risk_score"},
        "real_public_sample": {"npi", "provider_type", "hcpcs", "avg_paid", "total_services"},
        "synthetic_fixture": {"NPI", "EXCLTYPE", "EXCLDATE"},
    }
    required = required_by_kind.get(classification, set())
    absent = sorted(required - set(frame.columns))
    schema_status = "passed" if not absent else "failed"

    numeric_columns = (
        "submitted_amount",
        "allowed_amount",
        "paid_amount",
        "avg_submitted",
        "avg_allowed",
        "avg_paid",
        "total_services",
        "risk_score",
    )
    date_columns = ("service_date", "submit_date", "exclusion_date", "reinstatement_date")
    dtype_errors = sum(
        int((pd.to_numeric(frame[column], errors="coerce").isna() & frame[column].notna()).sum())
        for column in numeric_columns
        if column in frame
    )
    dtype_errors += sum(
        int(
            (
                pd.to_datetime(frame[column], errors="coerce", format="mixed").isna()
                & frame[column].notna()
                & (frame[column].astype(str) != "")
            ).sum()
        )
        for column in date_columns
        if column in frame
    )
    outlier_counts: dict[str, int] = {}
    for column in numeric_columns:
        if column not in frame:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
        if len(numeric) < 8:
            continue
        q1, q3 = numeric.quantile([0.25, 0.75])
        iqr = float(q3 - q1)
        if iqr > 0:
            outlier_counts[column] = int(
                ((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).sum()
            )
    total_cells = len(frame) * len(frame.columns)
    completeness_percent = (
        100.0 * (total_cells - int(missing.sum())) / total_cells if total_cells else 0.0
    )
    missing_by_column = missing.rename_axis("column").reset_index(name="missing_values")
    missing_by_column["missing_percent"] = (
        100.0 * missing_by_column["missing_values"] / len(frame) if len(frame) else 0.0
    )
    missing_by_column["completeness_percent"] = 100.0 - missing_by_column["missing_percent"]
    dtype_distribution = (
        frame.dtypes.astype(str).value_counts().rename_axis("data_type").reset_index(name="columns")
    )
    matches = [
        record
        for record in (provenance_records or [])
        if record.get("file_name") == path.name or record.get("source_identifier") == str(path)
    ]
    provenance = matches[-1] if matches else None
    source_metadata = None
    if dataset_name == "CMS provider-service sample":
        metadata_path = path.parent / "cms_snapshot_metadata.json"
        if metadata_path.is_file():
            try:
                source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                source_metadata = None
    try:
        fingerprint = file_provenance(
            path,
            source_name=dataset_name,
            data_kind=classification,
        )
    except FileNotFoundError:
        fingerprint = None
    return {
        "dataset_name": dataset_name,
        "source": str(path),
        "classification": classification,
        "record_count": len(frame),
        "column_count": len(frame.columns),
        "missing_total": int(missing.sum()),
        "missing_by_column": missing_by_column[missing_by_column["missing_values"] > 0].sort_values(
            "missing_values", ascending=False
        ),
        "duplicate_rows": duplicate_rows,
        "duplicate_percent": duplicate_percent,
        "completeness_percent": completeness_percent,
        "dtype_validation_status": "passed" if dtype_errors == 0 else "warning",
        "dtype_validation_errors": dtype_errors,
        "dtype_distribution": dtype_distribution,
        "column_completeness": missing_by_column,
        "outlier_counts": outlier_counts,
        "invalid_values": invalid,
        "invalid_total": int(sum(invalid.values())),
        "schema_status": schema_status,
        "schema_warnings": [f"Required field missing: {column}" for column in absent],
        "provenance": provenance,
        "source_metadata": source_metadata,
        "file_provenance": fingerprint,
        "data_quality_warnings": (
            [f"{int(missing.sum()):,} missing values across {int((missing > 0).sum())} columns"]
            if missing.sum()
            else []
        )
        + ([f"{duplicate_rows:,} duplicate rows"] if duplicate_rows else [])
        + ([f"{sum(invalid.values()):,} invalid-value findings"] if sum(invalid.values()) else [])
        + (
            [f"{dtype_errors:,} values do not match expected numeric/date types"]
            if dtype_errors
            else []
        )
        + (
            [f"Potential outliers flagged in {len(outlier_counts)} numeric fields (IQR screen)"]
            if any(outlier_counts.values())
            else []
        )
        + (["Ingestion timestamp not recorded"] if provenance is None else []),
    }


@st.cache_data(ttl=300, show_spinner="Checking available datasets…")
def discover_datasets(
    root: str | Path = ".",
    manifest_path: str | Path = "monitoring/reports/data_provenance.json",
) -> list[dict[str, Any]]:
    base = Path(root)
    manifest = load_provenance_manifest(base / manifest_path)
    discovered: list[dict[str, Any]] = []
    for name, (primary, fallback, classification) in SOURCES.items():
        selected = base / primary
        if not selected.is_file():
            selected = base / fallback
        if not selected.is_file():
            discovered.append(
                {
                    "dataset_name": name,
                    "source": str(base / primary),
                    "classification": classification,
                    "available": False,
                    "message": "Dataset artifact is unavailable at the configured path.",
                }
            )
            continue
        try:
            frame = (
                pd.read_csv(
                    selected,
                    dtype={"NPI": "string", "Rndrng_NPI": "string", "HCPCS_Cd": "string"},
                )
                if selected.suffix.lower() == ".csv"
                else pd.read_parquet(selected)
            )
            if name == "CMS provider-service sample":
                from src.data.ingest_cms import COLUMN_MAP

                frame = frame.rename(columns=COLUMN_MAP)
        except Exception as exc:
            discovered.append(
                {
                    "dataset_name": name,
                    "source": str(selected),
                    "available": False,
                    "load_error": str(exc),
                }
            )
            continue
        discovered.append(
            inspect_frame(
                frame,
                dataset_name=name,
                source_path=selected,
                classification=classification,
                provenance_records=manifest,
            )
        )
        discovered[-1]["available"] = True
    return discovered
