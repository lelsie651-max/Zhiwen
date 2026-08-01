from __future__ import annotations

from dataclasses import dataclass
import uuid

from app.schemas.fact_extraction_orchestration import FactExtractionOrchestrationStatus


@dataclass(frozen=True, slots=True)
class FactExtractionConsistencyPipelineResult:
    extraction_orchestration_id: uuid.UUID
    extraction_status: FactExtractionOrchestrationStatus
    grouping_application_id: uuid.UUID | None
    consistency_application_id: uuid.UUID | None
    consistency_check_application_id: uuid.UUID | None
    consistency_plan_manifest_hash: str | None
    consistency_execution_result_manifest_hash: str | None
    assessment_count: int | None
    consistency_created_new: bool | None
    skipped_reason: str | None
