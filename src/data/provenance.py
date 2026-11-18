"""Small, dependency-free provenance records for local data artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def file_provenance(
    path: str | Path,
    *,
    source_name: str,
    source_url: str | None = None,
    data_kind: str,
) -> dict[str, str | int | None]:
    """Describe an existing source file without reading its contents into memory."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Data source file not found: {file_path}")

    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = file_path.stat()
    return {
        "source_name": source_name,
        "source_url": source_url,
        "data_kind": data_kind,
        "file_name": file_path.name,
        "file_size_bytes": stat.st_size,
        "sha256": digest.hexdigest(),
        "recorded_at_utc": datetime.now(UTC).isoformat(),
    }


def dataset_provenance(
    frame: Any,
    *,
    source_name: str,
    classification: str,
    validation_status: str,
    file_name: str | None = None,
    file_format: str | None = None,
    source_url: str | None = None,
    source_identifier: str | None = None,
    recorded_at: datetime | None = None,
) -> dict[str, str | int | None]:
    """Build metadata only; no source rows, values, or column names are recorded."""
    return {
        "source_name": source_name,
        "source_url": source_url,
        "source_identifier": source_identifier,
        "ingestion_timestamp_utc": (recorded_at or datetime.now(UTC)).isoformat(),
        "file_name": file_name,
        "file_format": file_format,
        "record_count": len(frame),
        "column_count": len(frame.columns),
        "classification": classification,
        "validation_status": validation_status,
    }


def record_dataset_provenance(
    frame: Any,
    *,
    source_name: str,
    classification: str,
    validation_status: str,
    file_name: str | None = None,
    file_format: str | None = None,
    source_url: str | None = None,
    source_identifier: str | None = None,
    manifest_path: str | Path = "monitoring/reports/data_provenance.json",
) -> dict[str, str | int | None]:
    """Append a metadata-only ingestion event to the local provenance manifest."""
    record = dataset_provenance(
        frame,
        source_name=source_name,
        classification=classification,
        validation_status=validation_status,
        file_name=file_name,
        file_format=file_format,
        source_url=source_url,
        source_identifier=source_identifier,
    )
    destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        records = (
            json.loads(destination.read_text(encoding="utf-8")) if destination.exists() else []
        )
        if not isinstance(records, list):
            records = []
    except (OSError, json.JSONDecodeError):
        records = []
    records.append(record)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(records, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return record
