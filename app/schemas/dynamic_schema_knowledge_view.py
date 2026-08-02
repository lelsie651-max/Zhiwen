from __future__ import annotations

from dataclasses import dataclass
import uuid
from typing import Literal

from app.schemas.dynamic_schema_review_projection import DynamicSchemaReviewedFact
from app.schemas.dynamic_schema_ufl_projection import DynamicSchemaUFLProjectedField


DynamicSchemaKnowledgeViewAlgorithmName = Literal["dynamic_schema_knowledge_view"]
DynamicSchemaKnowledgeViewAlgorithmVersion = Literal["1.0.0"]
DynamicSchemaKnowledgeState = Literal[
    "missing",
    "review_required",
    "resolved",
    "observation_only",
    "mixed_reviewed_observation",
]


@dataclass(frozen=True, slots=True)
class DynamicSchemaKnowledgeField:
    source_field: DynamicSchemaUFLProjectedField
    reviewed_facts: tuple[DynamicSchemaReviewedFact, ...]
    knowledge_state: DynamicSchemaKnowledgeState
    effective_fact_value_ids: tuple[uuid.UUID, ...]
    observed_fact_value_count: int
    semantic_value_count: int
    has_schema_issues: bool


@dataclass(frozen=True, slots=True)
class DynamicSchemaKnowledgeSection:
    group_key: str | None
    display_order: int
    fields: tuple[DynamicSchemaKnowledgeField, ...]


@dataclass(frozen=True, slots=True)
class DynamicSchemaKnowledgeRecord:
    subject_key: str
    title_field_key: str | None
    has_review_required: bool
    issue_count: int
    sections: tuple[DynamicSchemaKnowledgeSection, ...]


@dataclass(frozen=True, slots=True)
class DynamicSchemaKnowledgeView:
    project_id: uuid.UUID
    schema_id: uuid.UUID
    schema_version_id: uuid.UUID
    orchestration_id: uuid.UUID
    extraction_run_id: uuid.UUID
    consistency_check_application_id: uuid.UUID
    source_consistency_application_id: uuid.UUID
    schema_definition_manifest_hash: str
    ufl_source_manifest_hash: str
    consistency_result_manifest_hash: str
    raw_projection_manifest_hash: str
    reviewed_projection_manifest_hash: str
    comparison_quality: Literal["complete", "partial"]
    algorithm_name: DynamicSchemaKnowledgeViewAlgorithmName
    algorithm_version: DynamicSchemaKnowledgeViewAlgorithmVersion
    record_count: int
    section_count: int
    field_count: int
    missing_field_count: int
    review_required_field_count: int
    resolved_field_count: int
    observation_only_field_count: int
    mixed_field_count: int
    records: tuple[DynamicSchemaKnowledgeRecord, ...]
    knowledge_view_manifest_hash: str
