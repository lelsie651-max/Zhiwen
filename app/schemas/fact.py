from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.fact import (
    FactEvidenceRole,
    FactStatus,
    FactValueSourceKind,
    FactValueStatus,
    FactValueType,
)
from app.utils.validation import normalize_text


def _normalize_required_text(value: str, *, field_name: str, max_length: int) -> str:
    normalized = normalize_text(value)
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if len(normalized) > max_length:
        raise ValueError(f"{field_name} must be at most {max_length} characters")
    return normalized


def _normalize_optional_text(value: str | None, *, field_name: str, max_length: int) -> str | None:
    if value is None:
        return None
    normalized = normalize_text(value)
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if len(normalized) > max_length:
        raise ValueError(f"{field_name} must be at most {max_length} characters")
    return normalized


class FactIdentityInput(BaseModel):
    subject_kind: str
    subject_key: str
    predicate_key: str
    scope_key: str | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("subject_kind")
    @classmethod
    def validate_subject_kind(cls, value: str) -> str:
        return _normalize_required_text(value, field_name="subject_kind", max_length=64)

    @field_validator("subject_key", "predicate_key")
    @classmethod
    def validate_fact_key_fields(cls, value: str, info) -> str:
        return _normalize_required_text(value, field_name=info.field_name, max_length=255)

    @field_validator("scope_key")
    @classmethod
    def validate_scope_key(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value, field_name="scope_key", max_length=255)


class FactValueInput(BaseModel):
    value_type: FactValueType
    value_json: Any | None = None
    language_code: str | None = None
    confidence: float | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("language_code")
    @classmethod
    def validate_language_code(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value, field_name="language_code", max_length=32)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if value < 0 or value > 1:
            raise ValueError("confidence must be between 0 and 1")
        return value

    @model_validator(mode="after")
    def validate_value_json_rule(self) -> "FactValueInput":
        if self.value_type == FactValueType.NULL:
            if self.value_json is not None:
                raise ValueError("value_json must be null when value_type is null")
        elif self.value_json is None:
            raise ValueError("value_json must not be null unless value_type is null")
        return self


class FactEvidenceLinkRead(BaseModel):
    id: uuid.UUID
    fact_value_id: uuid.UUID
    evidence_id: uuid.UUID
    role: FactEvidenceRole
    is_primary: bool
    source_order: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, extra="forbid")


class FactValueRead(BaseModel):
    id: uuid.UUID
    fact_id: uuid.UUID
    version_no: int
    value_type: FactValueType
    value_json: Any | None = None
    normalized_value_text: str | None = None
    value_hash: str
    language_code: str | None = None
    status: FactValueStatus
    source_kind: FactValueSourceKind
    extraction_run_id: uuid.UUID | None = None
    confidence: float | None = None
    created_by_id: uuid.UUID | None = None
    decided_by_id: uuid.UUID | None = None
    decided_at: datetime | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, extra="forbid")


class FactRead(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    subject_kind: str
    subject_key: str
    predicate_key: str
    scope_key: str | None = None
    identity_hash: str
    status: FactStatus
    current_value_id: uuid.UUID | None = None
    created_by_id: uuid.UUID | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, extra="forbid")
