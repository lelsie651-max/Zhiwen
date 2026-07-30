from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.ingestion_validation import (
    IngestionRiskCategory,
    IngestionRiskDetector,
    IngestionRiskSeverity,
)
from app.utils.validation import normalize_text


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class TechnicalPreflightOutcome(StrEnum):
    PASS = "pass"
    REJECT = "reject"
    QUARANTINE = "quarantine"


class DetectedFileFormat(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"


class TempFileReference(BaseModel):
    temp_key: str

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("temp_key", mode="before")
    @classmethod
    def validate_temp_key_field(cls, value: str) -> str:
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("temp_key must not be empty")
        return normalized


class ReceivedUpload(BaseModel):
    temp_file: TempFileReference
    safe_original_filename: str
    filename_sanitized: bool = False
    file_size: int
    sha256: str
    declared_mime: str | None = None
    extension: str

    model_config = ConfigDict(extra="forbid")

    @field_validator("safe_original_filename", mode="before")
    @classmethod
    def validate_safe_original_filename_field(cls, value: str) -> str:
        normalized = normalize_text(value)
        if not normalized or "\x00" in normalized or not 1 <= len(normalized) <= 255:
            raise ValueError("safe_original_filename must be between 1 and 255 characters")
        return normalized

    @field_validator("file_size")
    @classmethod
    def validate_file_size_field(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("file_size must be greater than 0")
        return value

    @field_validator("sha256", mode="before")
    @classmethod
    def validate_sha256_field(cls, value: str) -> str:
        normalized = normalize_text(value).lower()
        if not SHA256_PATTERN.fullmatch(normalized):
            raise ValueError("sha256 must be a 64-character lowercase hexadecimal string")
        return normalized

    @field_validator("declared_mime", mode="before")
    @classmethod
    def normalize_declared_mime_field(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_text(value)
        return normalized or None

    @field_validator("extension", mode="before")
    @classmethod
    def validate_extension_field(cls, value: str) -> str:
        normalized = normalize_text(value).lower()
        if normalized and not normalized.startswith("."):
            raise ValueError("extension must start with '.' when provided")
        return normalized


class TechnicalRiskFlag(BaseModel):
    category: IngestionRiskCategory
    kind: str
    severity: IngestionRiskSeverity
    detector: IngestionRiskDetector
    message: str
    confidence: float | None = None
    location: dict[str, Any] | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    evidence_excerpt: str | None = None

    model_config = ConfigDict(extra="forbid")

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


class TechnicalPreflightResult(BaseModel):
    outcome: TechnicalPreflightOutcome
    detected_format: DetectedFileFormat | None = None
    detected_mime: str | None = None
    detected_encoding: str | None = None
    file_checks: dict[str, Any] = Field(default_factory=dict)
    risk_flags: list[TechnicalRiskFlag] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @field_validator("detected_mime", "detected_encoding", mode="before")
    @classmethod
    def normalize_optional_text_field(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_text(value)
        return normalized or None
