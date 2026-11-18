"""Transparent payment-error indicators, separate from fraud classification.

These rules only identify potential payment issues from fields actually present in
the input. They do not create adjudicated labels and do not determine fraud.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pandas as pd

DEFAULT_CATEGORIES: dict[str, dict[str, Any]] = {
    "potential_overpayment": {"enabled": True},
    "potential_underpayment": {"enabled": True},
    "missing_documentation": {"enabled": True},
    "insufficient_documentation": {"enabled": True},
    "incorrect_payment_amount": {"enabled": True},
}


def _enabled(config: Mapping[str, Mapping[str, Any]] | None, category: str) -> bool:
    selected = config or DEFAULT_CATEGORIES
    return bool(selected.get(category, {}).get("enabled", False))


def _number(row: Mapping[str, Any], name: str) -> float | None:
    value = row.get(name)
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _documentation_state(row: Mapping[str, Any]) -> str | None:
    for name in ("documentation_status", "documentation", "documentation_complete"):
        if name not in row or pd.isna(row[name]):
            continue
        value = row[name]
        if isinstance(value, bool):
            return "complete" if value else "missing"
        normalized = str(value).strip().lower().replace("_", " ")
        if normalized in {"missing", "absent", "no", "false", "0"}:
            return "missing"
        if normalized in {"insufficient", "incomplete", "partial", "limited"}:
            return "insufficient"
        if normalized in {"complete", "adequate", "yes", "true", "1"}:
            return "complete"
    return None


def evaluate_payment_row(
    row: Mapping[str, Any],
    categories: Mapping[str, Mapping[str, Any]] | None = None,
    tolerance: float = 0.01,
) -> dict[str, Any]:
    """Evaluate one claim-like record and return evidence-backed indicators."""
    reasons: list[dict[str, Any]] = []
    submitted = _number(row, "submitted_amount")
    allowed = _number(row, "allowed_amount")
    paid = _number(row, "paid_amount")
    expected = _number(row, "expected_payment_amount")
    if expected is None:
        expected = _number(row, "expected_paid_amount")
    if expected is None:
        expected = _number(row, "expected_allowed_amount")

    def add(category: str, code: str, direction: str, message: str, **evidence: Any) -> None:
        if _enabled(categories, category):
            reasons.append(
                {
                    "category": category,
                    "reason_code": code,
                    "direction": direction,
                    "message": message,
                    "evidence": evidence,
                }
            )

    if paid is not None and allowed is not None and paid > allowed + tolerance:
        add(
            "potential_overpayment",
            "PAY_PAID_GT_ALLOWED",
            "overpayment",
            "Paid amount exceeds the allowed amount.",
            paid_amount=paid,
            allowed_amount=allowed,
        )
    if allowed is not None and submitted is not None and allowed > submitted + tolerance:
        add(
            "incorrect_payment_amount",
            "PAY_ALLOWED_GT_SUBMITTED",
            "overpayment",
            "Allowed amount exceeds the submitted amount.",
            allowed_amount=allowed,
            submitted_amount=submitted,
        )
    if expected is not None and paid is not None and paid < expected - tolerance:
        add(
            "potential_underpayment",
            "PAY_PAID_LT_EXPECTED",
            "underpayment",
            "Paid amount is below the supplied expected payment amount.",
            paid_amount=paid,
            expected_payment_amount=expected,
        )
    if expected is not None and paid is not None and paid > expected + tolerance:
        add(
            "potential_overpayment",
            "PAY_PAID_GT_EXPECTED",
            "overpayment",
            "Paid amount exceeds the supplied expected payment amount.",
            paid_amount=paid,
            expected_payment_amount=expected,
        )

    doc_state = _documentation_state(row)
    if doc_state == "missing":
        add(
            "missing_documentation",
            "DOC_MISSING",
            "documentation",
            "Documentation is marked missing.",
            documentation_status=doc_state,
        )
    elif doc_state == "insufficient":
        add(
            "insufficient_documentation",
            "DOC_INSUFFICIENT",
            "documentation",
            "Documentation is marked incomplete or insufficient.",
            documentation_status=doc_state,
        )

    score = min(1.0, 0.35 * len(reasons))
    return {
        "issue_count": len(reasons),
        "issue_score": score,
        "reason_codes": [item["reason_code"] for item in reasons],
        "issues": reasons,
        "has_potential_overpayment": any(item["direction"] == "overpayment" for item in reasons),
        "has_potential_underpayment": any(item["direction"] == "underpayment" for item in reasons),
    }


def analyze_payments(
    claims: pd.DataFrame,
    categories: Mapping[str, Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Return the input claims with JSON-safe payment-integrity evidence columns."""
    output = claims.copy()
    results = [evaluate_payment_row(row, categories) for row in output.to_dict("records")]
    output["payment_issue_count"] = [item["issue_count"] for item in results]
    output["payment_issue_score"] = [item["issue_score"] for item in results]
    output["payment_reason_codes"] = [json.dumps(item["reason_codes"]) for item in results]
    output["payment_evidence"] = [json.dumps(item["issues"]) for item in results]
    output["potential_overpayment"] = [item["has_potential_overpayment"] for item in results]
    output["potential_underpayment"] = [item["has_potential_underpayment"] for item in results]
    return output
