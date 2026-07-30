from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.dynamic_schema import (
    DynamicSchemaFieldCardinality,
    DynamicSchemaFieldValueType,
    DynamicSchemaStatus,
    DynamicSchemaVersionSourceKind,
    DynamicSchemaVersionStatus,
)
from app.utils.validation import normalize_text


SCHEMA_KEY_PATTERN = re.compile(r"^[a-z0-9._-]+$")


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


def _validate_key(value: str, *, field_name: str) -> str:
    normalized = _normalize_required_text(value, field_name=field_name, max_length=120)
    if not SCHEMA_KEY_PATTERN.fullmatch(normalized):
        raise ValueError(
            f"{field_name} must contain only lowercase letters, numbers, dots, underscores, or hyphens"
        )
    return normalized


def _validate_optional_key(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = _normalize_optional_text(value, field_name=field_name, max_length=120)
    if normalized is None:
        return None
    if not SCHEMA_KEY_PATTERN.fullmatch(normalized):
        raise ValueError(
            f"{field_name} must contain only lowercase letters, numbers, dots, underscores, or hyphens"
        )
    return normalized


class DynamicSchemaIdentityInput(BaseModel):
    schema_key: str
    name: str
    subject_kind: str
    description: str | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("schema_key")
    @classmethod
    def validate_schema_key(cls, value: str) -> str:
        return _validate_key(value, field_name="schema_key")

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _normalize_required_text(value, field_name="name", max_length=200)

    @field_validator("subject_kind")
    @classmethod
    def validate_subject_kind(cls, value: str) -> str:
        return _normalize_required_text(value, field_name="subject_kind", max_length=64)

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value, field_name="description", max_length=5000)


class DynamicSchemaFieldInput(BaseModel):
    field_key: str
    label: str
    description: str | None = None
    predicate_key: str
    scope_key: str | None = None
    expected_value_type: DynamicSchemaFieldValueType = DynamicSchemaFieldValueType.ANY
    cardinality: DynamicSchemaFieldCardinality = DynamicSchemaFieldCardinality.ONE
    is_required: bool = False
    is_title: bool = False
    is_summary: bool = False
    is_hidden: bool = False
    group_key: str | None = None
    display_order: int
    display_config: dict[str, Any] = Field(default_factory=dict)
    validation_rules: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @field_validator("field_key")
    @classmethod
    def validate_field_key(cls, value: str) -> str:
        return _validate_key(value, field_name="field_key")

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        return _normalize_required_text(value, field_name="label", max_length=200)

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value, field_name="description", max_length=5000)

    @field_validator("predicate_key")
    @classmethod
    def validate_predicate_key(cls, value: str) -> str:
        return _normalize_required_text(value, field_name="predicate_key", max_length=255)

    @field_validator("scope_key")
    @classmethod
    def validate_scope_key(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value, field_name="scope_key", max_length=255)

    @field_validator("group_key")
    @classmethod
    def validate_group_key(cls, value: str | None) -> str | None:
        return _validate_optional_key(value, field_name="group_key")

    @field_validator("display_order")
    @classmethod
    def validate_display_order(cls, value: int) -> int:
        if value < 0:
            raise ValueError("display_order must be non-negative")
        return value

    @model_validator(mode="after")
    def validate_visibility_flags(self) -> "DynamicSchemaFieldInput":
        if self.is_title and self.is_hidden:
            raise ValueError("hidden fields cannot also be title fields")
        return self


class DynamicSchemaVersionInput(BaseModel):
    summary: str | None = None
    layout_config: dict[str, Any] = Field(default_factory=dict)
    fields: list[DynamicSchemaFieldInput] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value, field_name="summary", max_length=5000)

    @model_validator(mode="after")
    def validate_fields(self) -> "DynamicSchemaVersionInput":
        if not self.fields:
            raise ValueError("fields must contain at least one item")

        field_keys = [field.field_key for field in self.fields]
        if len(set(field_keys)) != len(field_keys):
            raise ValueError("field_key values must be unique")

        display_orders = [field.display_order for field in self.fields]
        if len(set(display_orders)) != len(display_orders):
            raise ValueError("display_order values must be unique")

        title_fields = [field for field in self.fields if field.is_title]
        if len(title_fields) > 1:
            raise ValueError("at most one title field is allowed")

        return self


class DynamicSchemaFieldRead(BaseModel):
    id: uuid.UUID
    schema_version_id: uuid.UUID
    field_key: str
    label: str
    description: str | None = None
    predicate_key: str
    scope_key: str | None = None
    expected_value_type: DynamicSchemaFieldValueType
    cardinality: DynamicSchemaFieldCardinality
    is_required: bool
    is_title: bool
    is_summary: bool
    is_hidden: bool
    group_key: str | None = None
    display_order: int
    display_config: dict[str, Any] = Field(default_factory=dict)
    validation_rules: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, extra="forbid")


class DynamicSchemaVersionRead(BaseModel):
    id: uuid.UUID
    schema_id: uuid.UUID
    version_no: int
    status: DynamicSchemaVersionStatus
    source_kind: DynamicSchemaVersionSourceKind
    summary: str | None = None
    layout_config: dict[str, Any] = Field(default_factory=dict)
    created_by_id: uuid.UUID | None = None
    activated_by_id: uuid.UUID | None = None
    activated_at: datetime | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, extra="forbid")


class DynamicSchemaRead(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    schema_key: str
    name: str
    subject_kind: str
    description: str | None = None
    status: DynamicSchemaStatus
    current_version_id: uuid.UUID | None = None
    created_by_id: uuid.UUID | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, extra="forbid")
