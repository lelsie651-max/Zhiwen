from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal
import uuid

from app.schemas.ufl_fact_snapshot import UFLFactSnapshot


DynamicSchemaUFLProjectionAlgorithmName = Literal["dynamic_schema_ufl_projection"]
DynamicSchemaUFLProjectionAlgorithmVersion = Literal["1.0.0"]
DynamicSchemaUFLProjectionIssueCode = Literal[
    "required_missing",
    "cardinality_one_multiple_facts",
    "cardinality_one_multiple_semantic_values",
    "value_type_mismatch",
]


@dataclass(frozen=True, slots=True)
class DynamicSchemaUFLProjectedField:
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
    matched_facts: tuple[UFLFactSnapshot, ...]
    matched_fact_count: int
    semantic_value_count: int
    is_missing: bool
    type_compatible: bool
    issues: tuple[DynamicSchemaUFLProjectionIssueCode, ...]


@dataclass(frozen=True, slots=True)
class DynamicSchemaUFLProjectedRecord:
    subject_key: str
    fields: tuple[DynamicSchemaUFLProjectedField, ...]
    required_missing_field_keys: tuple[str, ...]
    issue_count: int


@dataclass(frozen=True, slots=True)
class DynamicSchemaUFLProjection:
    project_id: uuid.UUID
    schema_id: uuid.UUID
    schema_version_id: uuid.UUID
    orchestration_id: uuid.UUID
    extraction_run_id: uuid.UUID
    schema_definition_manifest_hash: str
    ufl_source_manifest_hash: str
    comparison_quality: Literal["complete", "partial"]
    subject_kind: str
    algorithm_name: DynamicSchemaUFLProjectionAlgorithmName
    algorithm_version: DynamicSchemaUFLProjectionAlgorithmVersion
    record_count: int
    projected_field_count: int
    required_missing_count: int
    issue_count: int
    records: tuple[DynamicSchemaUFLProjectedRecord, ...]
    projection_manifest_hash: str
