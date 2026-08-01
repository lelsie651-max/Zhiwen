from __future__ import annotations

from dataclasses import dataclass
import uuid
from typing import Any, Literal


DocumentRevisionFactDiffChangeKind = Literal[
    "unchanged",
    "modified",
    "added",
    "removed",
]

DocumentRevisionFactDiffQuality = Literal["complete", "partial"]
DocumentRevisionFactDiffBlockChangeKind = Literal[
    "unchanged",
    "modified",
    "moved",
    "added",
    "removed",
]


@dataclass(frozen=True, slots=True)
class DocumentRevisionFactDiffFactSnapshot:
    fact_id: uuid.UUID
    identity_hash: str
    subject_kind: str
    subject_key: str
    predicate_key: str
    scope_key: str | None
    subject_entity_id: uuid.UUID | None


@dataclass(frozen=True, slots=True)
class DocumentRevisionFactDiffEvidenceLocator:
    location_key: str
    page_no: int | None
    start_line: int | None
    end_line: int | None
    table_index: int | None
    row_index: int | None


@dataclass(frozen=True, slots=True)
class DocumentRevisionFactDiffEvidenceSnapshot:
    evidence_link_id: uuid.UUID
    evidence_id: uuid.UUID
    document_block_id: uuid.UUID
    locator: DocumentRevisionFactDiffEvidenceLocator
    excerpt: str
    excerpt_hash: str
    block_change_kind: DocumentRevisionFactDiffBlockChangeKind


@dataclass(frozen=True, slots=True)
class DocumentRevisionFactDiffValueGroup:
    semantic_key_hash: str
    value_type: str
    value_json: Any | None
    referenced_entity_id: uuid.UUID | None
    fact_value_ids: tuple[uuid.UUID, ...]
    evidences: tuple[DocumentRevisionFactDiffEvidenceSnapshot, ...]


@dataclass(frozen=True, slots=True)
class DocumentRevisionFactDiffItem:
    change_kind: DocumentRevisionFactDiffChangeKind
    base_fact: DocumentRevisionFactDiffFactSnapshot | None
    target_fact: DocumentRevisionFactDiffFactSnapshot | None
    base_value_groups: tuple[DocumentRevisionFactDiffValueGroup, ...]
    target_value_groups: tuple[DocumentRevisionFactDiffValueGroup, ...]


@dataclass(frozen=True, slots=True)
class DocumentRevisionFactDiff:
    project_id: uuid.UUID
    document_id: uuid.UUID
    base_revision_id: uuid.UUID
    target_revision_id: uuid.UUID
    base_extraction_run_id: uuid.UUID
    target_extraction_run_id: uuid.UUID
    base_orchestration_id: uuid.UUID
    target_orchestration_id: uuid.UUID
    base_revision_no: int
    target_revision_no: int
    base_orchestration_status: str
    target_orchestration_status: str
    block_diff_manifest_hash: str
    fact_diff_algorithm_name: str
    fact_diff_algorithm_version: str
    semantic_fingerprint_algorithm_version: str
    planner_name: str
    planner_version: str
    agent_name: str
    agent_version: str
    prompt_contract_hash: str
    provider: str
    requested_model: str
    executor_name: str
    executor_version: str
    persistence_name: str
    persistence_version: str
    entity_resolution_policy_name: str
    entity_resolution_policy_version: str
    comparison_quality: DocumentRevisionFactDiffQuality
    unchanged_count: int
    modified_count: int
    added_count: int
    removed_count: int
    items: tuple[DocumentRevisionFactDiffItem, ...]
    fact_diff_manifest_hash: str
