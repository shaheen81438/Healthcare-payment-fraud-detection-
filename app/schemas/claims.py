"""Pydantic request/response schemas for the scoring API."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

VALID_POS = {"11", "12", "21", "22", "23", "24", "31", "81", "UNK"}
VALID_MODIFIERS = {"", "25", "59", "LT", "RT"}
HCPCS_PATTERN = r"^[A-Za-z0-9]\d{3}[A-Za-z0-9]$"


class ClaimIn(BaseModel):
    """A single claim to score. Field ranges mirror the data contract (V001–V010)."""

    claim_id: str = Field(min_length=3, max_length=40, examples=["CLM-102849"])
    bene_id: str = Field(min_length=3, max_length=40, examples=["BEN-004521"])
    provider_id: str = Field(min_length=3, max_length=40, examples=["PRV-00042"])
    specialty: str = Field(min_length=2, max_length=64, examples=["Internal Medicine"])
    hcpcs: str = Field(pattern=HCPCS_PATTERN, examples=["99214"])
    modifier: str = Field(default="", examples=[""])
    units: int = Field(default=1, ge=1, le=200)
    dx_group: str = Field(default="DXG23", min_length=2, max_length=16, examples=["DXG02"])
    pos: str = Field(default="11", examples=["11"])
    service_date: date
    service_end_date: date | None = None
    submit_date: date
    submitted_amount: float = Field(ge=0, le=1_000_000)
    allowed_amount: float = Field(ge=0, le=1_000_000)
    paid_amount: float = Field(ge=0, le=1_000_000)

    @field_validator("hcpcs")
    @classmethod
    def _upper_hcpcs(cls, v: str) -> str:
        return v.upper()

    @field_validator("modifier")
    @classmethod
    def _valid_modifier(cls, v: str) -> str:
        v = v.upper()
        if v not in VALID_MODIFIERS:
            raise ValueError(f"modifier must be one of {sorted(VALID_MODIFIERS)!r}")
        return v

    @field_validator("pos")
    @classmethod
    def _valid_pos(cls, v: str) -> str:
        if v not in VALID_POS:
            raise ValueError(f"pos must be one of {sorted(VALID_POS)}")
        return v

    @model_validator(mode="after")
    def _date_order(self) -> ClaimIn:
        if self.service_end_date is None:
            self.service_end_date = self.service_date
        if self.service_end_date < self.service_date:
            raise ValueError("service_end_date cannot precede service_date")
        if self.submit_date < self.service_date:
            raise ValueError("submit_date cannot precede service_date")
        if not (date(2020, 1, 1) <= self.service_date <= date(2035, 12, 31)):
            raise ValueError("service_date outside supported window (2020-2035)")
        return self

    @model_validator(mode="after")
    def _amount_order(self) -> ClaimIn:
        if self.paid_amount > self.allowed_amount + 0.01:
            raise ValueError("paid_amount cannot exceed allowed_amount")
        if self.allowed_amount > self.submitted_amount + 0.01:
            raise ValueError("allowed_amount cannot exceed submitted_amount")
        return self


class ShapContribution(BaseModel):
    feature: str
    contribution: float
    value: float | str | int | None = None


class PredictionOut(BaseModel):
    claim_id: str
    risk_score: float = Field(ge=0, le=1, examples=[0.87])
    recommended_action: Literal["manual_review", "monitor", "auto_clear"]
    estimated_payment_at_risk: float
    reason_codes: list[str]
    supervised_probability: float
    anomaly_percentile: float
    rule_hits: list[str]
    insufficient_provider_history: bool = False
    model_version: str
    scored_at: datetime


class ExplainOut(PredictionOut):
    base_value: float
    shap_contributions: list[ShapContribution]
    reason_labels: dict[str, str]


class BatchPredictIn(BaseModel):
    claims: list[ClaimIn] = Field(min_length=1)


class BatchPredictOut(BaseModel):
    count: int
    model_version: str
    results: list[PredictionOut]


class ThresholdPolicyOut(BaseModel):
    review_threshold: float
    monitor_threshold: float
    capacity_share: float
    cost_params: dict[str, float]


class ModelInfoOut(BaseModel):
    model_version: str
    source: Literal["mlflow_registry", "local_bundle"]
    dataset_version: str
    trained_rows: int | None = None
    numeric_features: int
    categorical_features: int
    ensemble_weight: float
    threshold_policy: ThresholdPolicyOut
    metrics: dict[str, float]


class ErrorDetail(BaseModel):
    loc: str
    message: str


class ErrorOut(BaseModel):
    error: str
    detail: str | None = None
    details: list[ErrorDetail] | None = None
    request_id: str | None = None
