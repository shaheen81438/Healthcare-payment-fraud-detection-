"""HCPCS categorization tests."""

from __future__ import annotations

import pytest

from src.data.reference import hcpcs_category


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("99213", "EM_OFFICE"),
        ("99205", "EM_OFFICE"),
        ("99221", "EM_FACILITY"),
        ("99232", "EM_FACILITY"),
        ("97110", "THERAPY"),
        ("71046", "IMAGING"),
        ("80053", "LAB"),
        ("88305", "LAB"),
        ("J1885", "DRUG"),
        ("G0439", "OTHER"),
        ("11721", "PROCEDURE"),
        ("29881", "PROCEDURE"),
        ("64483", "PROCEDURE"),
    ],
)
def test_hcpcs_category(code: str, expected: str) -> None:
    assert hcpcs_category(code) == expected
