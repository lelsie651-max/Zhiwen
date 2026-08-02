from __future__ import annotations

from dataclasses import dataclass
import uuid

from app.schemas.document_revision_fact_diff import DocumentRevisionFactDiffQuality
from app.schemas.fact_extraction_orchestration import FactExtractionOrchestrationStatus


@dataclass(frozen=True, slots=True)
class DocumentRevisionUpdateWorkflowResult:
    project_id: uuid.UUID
    document_id: uuid.UUID
    base_revision_id: uuid.UUID
    target_revision_id: uuid.UUID
    base_extraction_run_id: uuid.UUID
    target_extraction_run_id: uuid.UUID
    base_orchestration_id: uuid.UUID
    target_orchestration_id: uuid.UUID
    target_extraction_status: FactExtractionOrchestrationStatus
    target_grouping_application_id: uuid.UUID | None
    target_consistency_application_id: uuid.UUID | None
    target_consistency_check_application_id: uuid.UUID | None
    target_consistency_plan_manifest_hash: str | None
    target_consistency_execution_manifest_hash: str | None
    target_assessment_count: int | None
    target_consistency_created_new: bool | None
    comparison_quality: DocumentRevisionFactDiffQuality | None
    impact_manifest_hash: str | None
    fact_count: int | None
    review_required_count: int | None
    skipped_reason: str | None
