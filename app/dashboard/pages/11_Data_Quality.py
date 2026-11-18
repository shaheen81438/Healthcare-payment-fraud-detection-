"""Dataset availability, provenance, and aggregate data-quality checks."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import plotly.express as px
import streamlit as st

from app.dashboard.common import page_setup
from app.dashboard.data_quality import discover_datasets

page_setup("Data Quality")
st.caption(
    "Quality checks use loaded project datasets and report aggregate counts only. "
    "No claim or provider row values are displayed."
)

datasets = discover_datasets()
if not datasets:
    st.info("No supported datasets are configured for inspection.")
    st.stop()

labels = [dataset["dataset_name"] for dataset in datasets]
selected_name = st.selectbox("Dataset", labels)
selected = next(item for item in datasets if item["dataset_name"] == selected_name)
if not selected.get("available", True):
    if selected.get("load_error"):
        st.error(f"Dataset could not be loaded: {selected['load_error']}")
    else:
        st.info(f"{selected['message']} Expected location: `{selected['source']}`")
    st.stop()
if "load_error" in selected:
    st.error(f"Dataset could not be loaded: {selected['load_error']}")
    st.stop()

metrics = st.columns(6)
metrics[0].metric("Records", f"{selected['record_count']:,}")
metrics[1].metric("Columns", f"{selected['column_count']:,}")
metrics[2].metric("Missing values", f"{selected['missing_total']:,}")
metrics[3].metric("Completeness", f"{selected['completeness_percent']:.1f}%")
metrics[4].metric(
    "Duplicate rows", f"{selected['duplicate_rows']:,} ({selected['duplicate_percent']:.2f}%)"
)
metrics[5].metric("Invalid findings", f"{selected['invalid_total']:,}")

st.subheader("Source and schema")
left, right = st.columns(2)
left.write(f"**Dataset source:** `{selected['source']}`")
left.write(f"**Classification:** {selected['classification']}")
left.write(f"**Schema validation:** {selected['schema_status'].title()}")
left.write(f"**Data type validation:** {selected['dtype_validation_status'].title()}")
if selected["dtype_validation_errors"]:
    left.warning(
        f"{selected['dtype_validation_errors']:,} values did not parse as expected numeric or date types."
    )
if selected["schema_warnings"]:
    for warning in selected["schema_warnings"]:
        st.error(warning)
else:
    left.success("Expected fields are present.")
if selected["invalid_total"] == 0 and selected["dtype_validation_status"] == "passed":
    st.success("Data quality summary: no configured invalid-value or type issues detected.")
else:
    st.warning(
        f"Data quality summary: {selected['invalid_total']:,} invalid-value findings and "
        f"{selected['dtype_validation_errors']:,} type-validation findings."
    )

provenance = selected["provenance"]
source_metadata = selected.get("source_metadata")
if provenance:
    right.write(f"**Last ingestion:** {provenance.get('ingestion_timestamp_utc', 'Not recorded')}")
    right.write(f"**Recorded source:** {provenance.get('source_name', 'Unknown')}")
    if provenance.get("source_url"):
        right.write(f"**Source URL:** {provenance['source_url']}")
    elif provenance.get("source_identifier"):
        right.write(f"**Source identifier:** {provenance['source_identifier']}")
    right.write(
        f"**File:** {provenance.get('file_name') or 'Not recorded'} ({provenance.get('file_format') or 'format not recorded'})"
    )
    right.write(f"**Ingestion validation:** {provenance.get('validation_status', 'Not recorded')}")
else:
    right.info(
        "No ingestion event is recorded for this artifact. File modification time is not treated as an ingestion date."
    )
if source_metadata:
    right.write(
        f"**Source retrieval date:** {source_metadata.get('retrieved_date', 'Not recorded')}"
    )
    right.write(f"**Dataset ID:** {source_metadata.get('dataset_uuid', 'Not recorded')}")

file_info = selected["file_provenance"]
if file_info:
    with st.expander("Artifact provenance"):
        st.write(f"SHA-256: `{file_info['sha256']}`")
        st.write(f"Size: {file_info['file_size_bytes']:,} bytes")
        st.write(f"Fingerprint recorded: {file_info['recorded_at_utc']}")

st.subheader("Missing values and duplicate records")
left, right = st.columns(2)
with left:
    missing = selected["missing_by_column"]
    if missing.empty:
        st.success("No missing values found in this dataset.")
    else:
        st.plotly_chart(
            px.bar(
                missing,
                x="column",
                y="missing_percent",
                hover_data=["missing_values", "missing_percent"],
                labels={"missing_percent": "Missing rows (%)", "column": "Column"},
                title="Missing values by column",
            ),
            use_container_width=True,
        )
with right:
    duplicate_counts = {
        "Unique rows": selected["record_count"] - selected["duplicate_rows"],
        "Duplicate rows": selected["duplicate_rows"],
    }
    st.plotly_chart(
        px.bar(
            x=list(duplicate_counts),
            y=list(duplicate_counts.values()),
            labels={"x": "Record category", "y": "Rows"},
            title=f"Duplicate records · {selected['duplicate_percent']:.2f}%",
        ),
        use_container_width=True,
    )

st.subheader("Data types and completeness")
left, right = st.columns(2)
with left:
    st.plotly_chart(
        px.bar(
            selected["dtype_distribution"], x="data_type", y="columns", title="Column data types"
        ),
        use_container_width=True,
    )
with right:
    completeness = selected["column_completeness"]
    st.plotly_chart(
        px.bar(
            completeness.sort_values("completeness_percent").head(30),
            x="column",
            y="completeness_percent",
            range_y=[0, 100],
            labels={"completeness_percent": "Present values (%)", "column": "Column"},
            title="Column completeness",
        ),
        use_container_width=True,
    )

if selected["outlier_counts"]:
    st.subheader("Potential numeric outliers")
    st.caption(
        "Exploratory 1.5×IQR flags identify values for review; they are not automatically invalid."
    )
    st.dataframe(
        [
            {"field": key, "potential_outliers": value}
            for key, value in selected["outlier_counts"].items()
        ],
        use_container_width=True,
        hide_index=True,
    )

if selected["invalid_values"]:
    with st.expander("Invalid-value checks"):
        st.dataframe(
            [{"check": key, "records": value} for key, value in selected["invalid_values"].items()],
            use_container_width=True,
            hide_index=True,
        )
if selected["data_quality_warnings"]:
    st.warning("Data-quality warnings: " + "; ".join(selected["data_quality_warnings"]))
else:
    st.caption("No aggregate quality warnings were triggered by the available checks.")
