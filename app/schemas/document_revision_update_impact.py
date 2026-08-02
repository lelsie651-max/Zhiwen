from __future__ import annotations

from dataclasses import dataclass
import uuid
from typing import Literal

from app.schemas.document_revision_fact_diff import (
    DocumentRevisionFactDiffChangeKind,
    DocumentRevisionFactDiffFactSnapshot,
    DocumentRevisionFactDiffQuality,
    DocumentRevisionFactDiffValueGroup,
)
from app.schemas.effective_fact_value import (
    EffectiveFactValueResolutionBasis,
    EffectiveFactValueResolutionStatus,
)


DocumentRevisionUpdateImpactKind = Literal[
    "unchanged_resolved",
    "unchanged_no_review_context",
    "unchanged_unresolved",
    "modified",
    "added",
    "removed",
]


@dataclass(frozen=True, slots=True)
class DocumentRevisionUpdateImpactItem:
    fact_id: uuid.UUID
    fact_change_kind: DocumentRevisionFactDiffChangeKind
    impact_kind: DocumentRevisionUpdateImpactKind
    requires_review: bool
    base_assessment_id: uuid.UUID | None
    base_review_status: str | None
    base_resolution_status: EffectiveFactValueResolutionStatus | None
    base_resolution_basis: EffectiveFactValueResolutionBasis | None
    base_current_decision_id: uuid.UUID | None
    base_current_decision_kind: str | None
    base_effective_fact_value_ids: tuple[uuid.UUID, ...]
    base_fact: DocumentRevisionFactDiffFactSnapshot | None
    base_value_groups: tuple[DocumentRevisionFactDiffValueGroup, ...]
    target_fact: DocumentRevisionFactDiffFactSnapshot | None
    target_value_groups: tuple[DocumentRevisionFactDiffValueGroup, ...]


@dataclass(frozen=True, slots=True)
class DocumentRevisionUpdateImpact:
    project_id: uuid.UUID
    document_id: uuid.UUID
    base_revision_id: uuid.UUID
    target_revision_id: uuid.UUID
    base_extraction_run_id: uuid.UUID
    target_extraction_run_id: uuid.UUID
    base_orchestration_id: uuid.UUID
    target_orchestration_id: uuid.UUID
    base_consistency_check_application_id: uuid.UUID
    base_source_consistency_application_id: uuid.UUID
    comparison_quality: DocumentRevisionFactDiffQuality
    block_diff_manifest_hash: str
    fact_diff_manifest_hash: str
    base_consistency_result_manifest_hash: str
    impact_algorithm_name: str
    impact_algorithm_version: str
    fact_count: int
    review_required_count: int
    unchanged_resolved_count: int
    unchanged_no_review_context_count: int
    unchanged_unresolved_count: int
    modified_count: int
    added_count: int
    removed_count: int
    items: tuple[DocumentRevisionUpdateImpactItem, ...]
    impact_manifest_hash: str
