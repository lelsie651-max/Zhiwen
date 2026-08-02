from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from typing import Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field

from app.models.dynamic_schema import (
    DynamicSchemaFieldCardinality,
    DynamicSchemaFieldValueType,
    DynamicSchemaStatus,
    DynamicSchemaVersionSourceKind,
    DynamicSchemaVersionStatus,
)
from app.models.fact import FactEvidenceRole, FactValueSourceKind, FactValueType


DynamicSchemaDefinitionSnapshotAlgorithmName = Literal["dynamic_schema_definition_snapshot"]
DynamicSchemaDefinitionSnapshotAlgorithmVersion = Literal["1.0.0"]


class ProjectedEvidence(BaseModel):
    evidence_id: uuid.UUID
    role: FactEvidenceRole
    is_primary: bool
    source_order: int
    excerpt: str
    block_id: uuid.UUID
    location_key: str
    page_no: int | None = None
    start_line: int | None = None
    end_line: int | None = None
    start_offset: int
    end_offset: int

    model_config = ConfigDict(extra="forbid")


class ProjectedValue(BaseModel):
    fact_id: uuid.UUID
    fact_value_id: uuid.UUID
    scope_key: str | None = None
    value_type: FactValueType
    value_json: Any | None = None
    normalized_value_text: str | None = None
    language_code: str | None = None
    source_kind: FactValueSourceKind
    confidence: float | None = None
    type_compatible: bool
    evidences: list[ProjectedEvidence] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class ProjectedField(BaseModel):
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
    values: list[ProjectedValue] = Field(default_factory=list)
    is_missing: bool
    issues: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class ProjectedRecord(BaseModel):
    subject_key: str
    fields: list[ProjectedField] = Field(default_factory=list)
    required_missing_field_keys: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class DynamicSchemaProjection(BaseModel):
    schema_id: uuid.UUID
    schema_key: str
    schema_name: str
    schema_status: DynamicSchemaStatus
    version_id: uuid.UUID
    version_no: int
    version_status: DynamicSchemaVersionStatus
    version_source_kind: DynamicSchemaVersionSourceKind
    subject_kind: str
    records: list[ProjectedRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True, slots=True)
class DynamicSchemaFieldDefinitionSnapshot:
    field_id: uuid.UUID
    schema_version_id: uuid.UUID
    field_key: str
    label: str
    description: str | None
    predicate_key: str
    scope_key: str | None
    expected_value_type: str
    cardinality: str
    is_required: bool
    is_title: bool
    is_summary: bool
    is_hidden: bool
    group_key: str | None
    display_order: int
    display_config: object
    validation_rules: object
    created_at: datetime


@dataclass(frozen=True, slots=True)
class DynamicSchemaDefinitionSnapshot:
    project_id: uuid.UUID
    schema_id: uuid.UUID
    schema_key: str
    name: str
    subject_kind: str
    description: str | None
    schema_status: str
    schema_version_id: uuid.UUID
    version_no: int
    version_status: str
    source_kind: str
    summary: str | None
    layout_config: object
    created_by_id: uuid.UUID | None
    activated_by_id: uuid.UUID | None
    activated_at: datetime | None
    is_current: bool
    algorithm_name: DynamicSchemaDefinitionSnapshotAlgorithmName
    algorithm_version: DynamicSchemaDefinitionSnapshotAlgorithmVersion
    field_count: int
    fields: tuple[DynamicSchemaFieldDefinitionSnapshot, ...]
    definition_manifest_hash: str
