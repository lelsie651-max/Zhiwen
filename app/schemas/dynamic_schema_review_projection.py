from __future__ import annotations

from dataclasses import dataclass
import uuid
from typing import Literal

from app.schemas.dynamic_schema_ufl_projection import DynamicSchemaUFLProjectedField
from app.schemas.ufl_fact_snapshot import UFLFactSnapshot


DynamicSchemaReviewProjectionAlgorithmName = Literal[
    "dynamic_schema_review_projection"
]
DynamicSchemaReviewProjectionAlgorithmVersion = Literal["1.0.0"]
DynamicSchemaReviewedFactState = Literal[
    "no_consistency_candidate",
    "resolved",
    "pending_review",
    "deferred",
    "unreviewed_compatible",
]
DynamicSchemaReviewedFactResolutionBasis = Literal[
    "human_selection",
    "human_confirmed_compatibility",
    "none",
]


@dataclass(frozen=True, slots=True)
class DynamicSchemaReviewedFact:
    fact: UFLFactSnapshot
    review_state: DynamicSchemaReviewedFactState
    candidate_id: uuid.UUID | None
    assessment_id: uuid.UUID | None
    resolution_basis: DynamicSchemaReviewedFactResolutionBasis
    current_decision_id: uuid.UUID | None
    current_decision_kind: str | None
    effective_fact_value_ids: tuple[uuid.UUID, ...]
    requires_review: bool


@dataclass(frozen=True, slots=True)
class DynamicSchemaReviewedField:
    source_field: DynamicSchemaUFLProjectedField
    reviewed_facts: tuple[DynamicSchemaReviewedFact, ...]
    review_required: bool
    resolved_fact_count: int
    review_required_fact_count: int
    effective_fact_value_ids: tuple[uuid.UUID, ...]


@dataclass(frozen=True, slots=True)
class DynamicSchemaReviewedRecord:
    subject_key: str
    required_missing_field_keys: tuple[str, ...]
    issue_count: int
    fields: tuple[DynamicSchemaReviewedField, ...]


@dataclass(frozen=True, slots=True)
class DynamicSchemaReviewProjection:
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
    comparison_quality: Literal["complete", "partial"]
    algorithm_name: DynamicSchemaReviewProjectionAlgorithmName
    algorithm_version: DynamicSchemaReviewProjectionAlgorithmVersion
    record_count: int
    unique_matched_fact_count: int
    resolved_fact_count: int
    review_required_fact_count: int
    no_candidate_fact_count: int
    field_review_required_count: int
    records: tuple[DynamicSchemaReviewedRecord, ...]
    reviewed_projection_manifest_hash: str
