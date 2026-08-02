from __future__ import annotations

from dataclasses import dataclass
import uuid
from typing import Any, Literal


UFLFactSnapshotQuality = Literal["complete", "partial"]


@dataclass(frozen=True, slots=True)
class UFLFactEvidenceLocator:
    location_key: str
    page_no: int | None
    start_line: int | None
    end_line: int | None
    table_index: int | None
    row_index: int | None


@dataclass(frozen=True, slots=True)
class UFLFactEvidenceSnapshot:
    evidence_link_id: uuid.UUID
    evidence_id: uuid.UUID
    document_revision_id: uuid.UUID
    document_block_id: uuid.UUID
    locator: UFLFactEvidenceLocator
    excerpt: str
    excerpt_hash: str
    content_hash: str
    role: str
    is_primary: bool
    source_order: int


@dataclass(frozen=True, slots=True)
class UFLFactValueSnapshot:
    fact_value_id: uuid.UUID
    source_batch_id: uuid.UUID
    source_application_id: uuid.UUID
    proposal_index: int
    normalized_value_text: str
    value_hash: str
    language_code: str | None
    confidence: float | None


@dataclass(frozen=True, slots=True)
class UFLFactValueGroupSnapshot:
    semantic_key_hash: str
    value_type: str
    value_json: Any | None
    referenced_entity_id: uuid.UUID | None
    fact_value_ids: tuple[uuid.UUID, ...]
    values: tuple[UFLFactValueSnapshot, ...]
    evidences: tuple[UFLFactEvidenceSnapshot, ...]


@dataclass(frozen=True, slots=True)
class UFLFactSnapshot:
    fact_id: uuid.UUID
    identity_hash: str
    subject_kind: str
    subject_key: str
    subject_entity_id: uuid.UUID | None
    predicate_key: str
    scope_key: str | None
    semantic_group_count: int
    fact_value_count: int
    value_groups: tuple[UFLFactValueGroupSnapshot, ...]


@dataclass(frozen=True, slots=True)
class OrchestrationUFLFactSnapshot:
    project_id: uuid.UUID
    orchestration_id: uuid.UUID
    extraction_run_id: uuid.UUID
    orchestration_status: str
    comparison_quality: UFLFactSnapshotQuality
    source_application_count: int
    fact_count: int
    fact_value_count: int
    evidence_count: int
    algorithm_name: Literal["orchestration_ufl_fact_snapshot"]
    algorithm_version: Literal["1.0.0"]
    facts: tuple[UFLFactSnapshot, ...]
    source_manifest_hash: str
