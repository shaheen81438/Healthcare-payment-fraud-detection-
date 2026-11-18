"""Identifier-first OIG LEIE screening with cautious secondary name matching."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

import pandas as pd

_NAME_COLUMNS = ("provider_name", "name", "BUSNAME", "LASTNAME", "FIRSTNAME", "MIDNAME")
_SSN_COLUMNS = {"ssn", "ssn_1", "ssn_2", "ssn_3", "social_security_number"}


def normalize_leie(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize official LEIE or fixture columns without retaining SSNs."""
    aliases = {
        "NPI": "npi",
        "EXCLTYPE": "exclusion_type",
        "EXCLDATE": "exclusion_date",
        "REINDATE": "reinstatement_date",
        "REINSTDATE": "reinstatement_date",
        "REINSTATEMENTDATE": "reinstatement_date",
        "BUSNAME": "provider_name",
        "PROVIDER_NAME": "provider_name",
        "LASTNAME": "last_name",
        "FIRSTNAME": "first_name",
        "MIDNAME": "middle_name",
    }
    normalized_columns = {str(column).upper(): column for column in raw.columns}
    frame = raw.rename(
        columns={
            normalized_columns[key]: value
            for key, value in aliases.items()
            if key in normalized_columns
        }
    ).copy()
    frame = frame.drop(columns=[column for column in frame if column.lower() in _SSN_COLUMNS])
    if "npi" not in frame:
        raise ValueError("LEIE file must contain an NPI column")
    frame["npi"] = frame["npi"].map(normalize_npi)
    frame = frame[frame["npi"].str.fullmatch(r"\d{10}", na=False)]
    for column in ("exclusion_date", "reinstatement_date"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.date
    return frame.reset_index(drop=True)


def normalize_npi(value: Any) -> str:
    return re.sub(r"\D", "", "" if value is None or pd.isna(value) else str(value))


def normalize_name(value: Any) -> str:
    value = "" if value is None or pd.isna(value) else str(value)
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _record_name(row: pd.Series) -> str:
    if "provider_name" in row and pd.notna(row["provider_name"]):
        return str(row["provider_name"])
    if "name" in row and pd.notna(row["name"]):
        return str(row["name"])
    parts = [
        str(row[col])
        for col in (
            "first_name",
            "middle_name",
            "last_name",
            "FIRSTNAME",
            "MIDNAME",
            "LASTNAME",
            "BUSNAME",
        )
        if col in row and pd.notna(row[col])
    ]
    return " ".join(parts)


def _safe_public_columns(row: pd.Series) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        if key.lower() in _SSN_COLUMNS:
            continue
        if key.lower() in {
            "npi",
            "exclusion_type",
            "exclusion_date",
            "reinstatement_date",
            "specialty",
            "state",
            "name",
            "provider_name",
            "last_name",
            "first_name",
            "middle_name",
        }:
            result[key] = None if pd.isna(value) else value
    return result


def screen_provider(
    provider: dict[str, Any] | pd.Series,
    leie: pd.DataFrame,
    fuzzy_threshold: float = 0.92,
) -> list[dict[str, Any]]:
    """Return possible LEIE matches; empty means no match was found."""
    subject = pd.Series(provider)
    work = leie.copy()
    if "npi" not in work:
        raise ValueError("LEIE data must contain normalized 'npi' values")
    work["_npi"] = work["npi"].map(normalize_npi)
    subject_npi = normalize_npi(subject.get("npi"))
    subject_name = normalize_name(subject.get("provider_name", subject.get("name")))
    exact = work[work["_npi"] == subject_npi] if subject_npi else work.iloc[0:0]
    candidates: list[tuple[pd.Series, str, float, bool]] = [
        (row, "npi", 1.0, len(exact) > 1) for _, row in exact.iterrows()
    ]
    if not candidates and subject_name:
        scored: list[tuple[float, pd.Series]] = []
        for _, row in work.iterrows():
            candidate_name = normalize_name(_record_name(row))
            if candidate_name:
                scored.append((SequenceMatcher(None, subject_name, candidate_name).ratio(), row))
        scored.sort(key=lambda item: item[0], reverse=True)
        matches = [(score, row) for score, row in scored if score >= fuzzy_threshold]
        ambiguous = len(matches) > 1 and matches[0][0] - matches[1][0] < 0.03
        candidates = [(row, "name_fuzzy", score, ambiguous) for score, row in matches]

    results = []
    for row, match_type, confidence, ambiguous in candidates:
        results.append(
            {
                "provider_id": str(subject.get("provider_id", subject.get("npi", ""))),
                "match_type": match_type,
                "match_confidence": float(confidence),
                "ambiguous": bool(ambiguous),
                "manual_verification_status": "required" if ambiguous else "pending",
                "warning": "Potential exclusion match; verify identity and dates manually.",
                "exclusion_date": row.get("exclusion_date"),
                "reinstatement_date": row.get("reinstatement_date"),
                "exclusion_type": row.get("exclusion_type"),
                "match_evidence": _safe_public_columns(row),
            }
        )
    return results


def screen_providers(providers: pd.DataFrame, leie: pd.DataFrame) -> pd.DataFrame:
    records = [
        match for row in providers.to_dict("records") for match in screen_provider(row, leie)
    ]
    return pd.DataFrame(records)
