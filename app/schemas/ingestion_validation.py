from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.ingestion_validation import (
    IngestionRiskCategory,
    IngestionRiskDetector,
    IngestionRiskSeverity,
    ValidationRecommendedAuthority,
    ValidationReportOutcome,
    ValidationReportStatus,
    ValidationUserDecision,
)
from app.utils.validation import normalize_text


class ValidationReportCreate(BaseModel):
    revision_id: uuid.UUID
    attempt_no: int
    status: ValidationReportStatus = ValidationReportStatus.PENDING
    outcome: ValidationReportOutcome = ValidationReportOutcome.PENDING
    content_kind: str | None = None
    content_confidence: float | None = None
    recommended_authority: ValidationRecommendedAuthority | None = None
    detected_language: str | None = None
    language_confidence: float | None = None
    character_count: int = 0
    printable_ratio: float | None = None
    garbled_ratio: float | None = None
    repetition_ratio: float | None = None
    file_checks: dict[str, Any] = Field(default_factory=dict)
    text_checks: dict[str, Any] = Field(default_factory=dict)
    classification_details: dict[str, Any] = Field(default_factory=dict)
    summary: str | None = None
    failure_code: str | None = None
    failure_message: str | None = None
    user_decision: ValidationUserDecision | None = None
    decided_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    @field_validator("attempt_no")
    @classmethod
    def validate_attempt_no_field(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("attempt_no must be greater than 0")
        return value

    @field_validator("character_count")
    @classmethod
    def validate_character_count_field(cls, value: int) -> int:
        if value < 0:
            raise ValueError("character_count must be greater than or equal to 0")
        return value

    @field_validator(
        "content_confidence",
        "language_confidence",
        "printable_ratio",
        "garbled_ratio",
        "repetition_ratio",
    )
    @classmethod
    def validate_ratio_field(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not 0 <= value <= 1:
            raise ValueError("value must be between 0 and 1")
        return value

    @field_validator("content_kind", "detected_language", "summary", "failure_code", "failure_message", mode="before")
    @classmethod
    def normalize_optional_text_field(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_text(value)
        return normalized or None


class ValidationReportRead(BaseModel):
    id: uuid.UUID
    revision_id: uuid.UUID
    attempt_no: int
    status: ValidationReportStatus
    outcome: ValidationReportOutcome
    content_kind: str | None = None
    content_confidence: float | None = None
    recommended_authority: ValidationRecommendedAuthority | None = None
    detected_language: str | None = None
    language_confidence: float | None = None
    character_count: int
    printable_ratio: float | None = None
    garbled_ratio: float | None = None
    repetition_ratio: float | None = None
    file_checks: dict[str, Any]
    text_checks: dict[str, Any]
    classification_details: dict[str, Any]
    summary: str | None = None
    failure_code: str | None = None
    user_decision: ValidationUserDecision | None = None
    decided_by_id: uuid.UUID | None = None
    decided_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ValidationReportInternalRead(ValidationReportRead):
    failure_message: str | None = None


class RiskFlagCreate(BaseModel):
    report_id: uuid.UUID
    category: IngestionRiskCategory
    kind: str
    severity: IngestionRiskSeverity
    detector: IngestionRiskDetector
    message: str
    confidence: float | None = None
    location: dict[str, Any] | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    evidence_excerpt: str | None = None

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    @field_validator("kind", mode="before")
    @classmethod
    def validate_kind_field(cls, value: str) -> str:
        normalized = normalize_text(value)
        if not normalized or not 1 <= len(normalized) <= 80:
            raise ValueError("kind must be between 1 and 80 characters")
        return normalized

    @field_validator("message", mode="before")
    @classmethod
    def validate_message_field(cls, value: str) -> str:
        normalized = normalize_text(value)
        if not normalized or not 1 <= len(normalized) <= 500:
            raise ValueError("message must be between 1 and 500 characters")
        return normalized

    @field_validator("evidence_excerpt", mode="before")
    @classmethod
    def validate_evidence_excerpt_field(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_text(value)
        if not normalized or len(normalized) > 500:
            raise ValueError("evidence_excerpt must be at most 500 characters")
        return normalized

    @field_validator("confidence")
    @classmethod
    def validate_confidence_field(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not 0 <= value <= 1:
            raise ValueError("confidence must be between 0 and 1")
        return value


class RiskFlagRead(BaseModel):
    id: uuid.UUID
    report_id: uuid.UUID
    category: IngestionRiskCategory
    kind: str
    severity: IngestionRiskSeverity
    detector: IngestionRiskDetector
    message: str
    confidence: float | None = None
    location: dict[str, Any] | None = None
    details: dict[str, Any]
    evidence_excerpt: str | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
