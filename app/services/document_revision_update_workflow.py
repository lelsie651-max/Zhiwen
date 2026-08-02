from __future__ import annotations

import re
import uuid
from collections.abc import Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.prompt_registry import PromptDefinition
from app.schemas.consistency_check import ConsistencyCheckPlannerConfig
from app.schemas.consistency_pipeline import FactExtractionConsistencyPipelineResult
from app.schemas.document_revision_fact_diff import DocumentRevisionFactDiffQuality
from app.schemas.document_revision_update_impact import DocumentRevisionUpdateImpact
from app.schemas.document_revision_update_workflow import (
    DocumentRevisionUpdateWorkflowResult,
)
from app.schemas.fact_extraction_orchestration import FactExtractionOrchestrationStatus
from app.schemas.fact_extraction_plan import FactExtractionPlan
from app.services import consistency_pipeline as pipeline_service
from app.services import fact_extraction_orchestration as orchestration_service
from app.services import document_revision_update_impact as impact_service
from app.services.llm import LLMClient


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class DocumentRevisionUpdateWorkflowError(Exception):
    """Base class for update workflow failures."""


class DocumentRevisionUpdateWorkflowStateError(DocumentRevisionUpdateWorkflowError):
    """Raised when a composed sub-result is not admissible."""


class DocumentRevisionUpdateWorkflowInvariantError(DocumentRevisionUpdateWorkflowError):
    """Raised when authenticated workflow sub-results diverge."""


def _require_uuid(value: object, *, field_name: str) -> uuid.UUID:
    if not isinstance(value, uuid.UUID):
        raise DocumentRevisionUpdateWorkflowStateError(
            f"document_revision_update_workflow_{field_name}_invalid"
        )
    return value


def _require_sha256(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise DocumentRevisionUpdateWorkflowInvariantError(
            f"document_revision_update_workflow_{field_name}_invalid"
        )
    return value


def _require_non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DocumentRevisionUpdateWorkflowInvariantError(
            f"document_revision_update_workflow_{field_name}_invalid"
        )
    return value


def _require_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise DocumentRevisionUpdateWorkflowInvariantError(
            f"document_revision_update_workflow_{field_name}_invalid"
        )
    return value


def _require_comparison_quality(value: object) -> DocumentRevisionFactDiffQuality:
    if value not in {"complete", "partial"}:
        raise DocumentRevisionUpdateWorkflowInvariantError(
            "document_revision_update_workflow_comparison_quality_invalid"
        )
    return value


def _validate_pipeline_result(
    pipeline_result: FactExtractionConsistencyPipelineResult,
) -> FactExtractionConsistencyPipelineResult:
    _require_uuid(
        pipeline_result.extraction_orchestration_id,
        field_name="target_orchestration_id",
    )
    status = pipeline_result.extraction_status
    if status == FactExtractionOrchestrationStatus.FAILED:
        if pipeline_result.skipped_reason != "extraction_failed":
            raise DocumentRevisionUpdateWorkflowInvariantError(
                "document_revision_update_workflow_pipeline_failed_shape_invalid"
            )
        if any(
            value is not None
            for value in (
                pipeline_result.grouping_application_id,
                pipeline_result.consistency_application_id,
                pipeline_result.consistency_check_application_id,
                pipeline_result.consistency_plan_manifest_hash,
                pipeline_result.consistency_execution_result_manifest_hash,
                pipeline_result.assessment_count,
                pipeline_result.consistency_created_new,
            )
        ):
            raise DocumentRevisionUpdateWorkflowInvariantError(
                "document_revision_update_workflow_pipeline_failed_shape_invalid"
            )
        return pipeline_result
    if status not in {
        FactExtractionOrchestrationStatus.COMPLETED,
        FactExtractionOrchestrationStatus.PARTIAL,
    }:
        raise DocumentRevisionUpdateWorkflowStateError(
            "document_revision_update_workflow_target_extraction_status_invalid"
        )
    if pipeline_result.skipped_reason is not None:
        raise DocumentRevisionUpdateWorkflowInvariantError(
            "document_revision_update_workflow_pipeline_success_shape_invalid"
        )
    _require_uuid(
        pipeline_result.grouping_application_id,
        field_name="target_grouping_application_id",
    )
    _require_uuid(
        pipeline_result.consistency_application_id,
        field_name="target_consistency_application_id",
    )
    _require_uuid(
        pipeline_result.consistency_check_application_id,
        field_name="target_consistency_check_application_id",
    )
    _require_sha256(
        pipeline_result.consistency_plan_manifest_hash,
        field_name="target_consistency_plan_manifest_hash",
    )
    _require_sha256(
        pipeline_result.consistency_execution_result_manifest_hash,
        field_name="target_consistency_execution_manifest_hash",
    )
    _require_non_negative_int(
        pipeline_result.assessment_count,
        field_name="target_assessment_count",
    )
    _require_bool(
        pipeline_result.consistency_created_new,
        field_name="target_consistency_created_new",
    )
    return pipeline_result


def _validate_authenticated_terminal_orchestration(
    terminal_orchestration,
    *,
    expected_orchestration_id: uuid.UUID,
    expected_status: FactExtractionOrchestrationStatus,
):
    if not isinstance(expected_status, FactExtractionOrchestrationStatus):
        raise DocumentRevisionUpdateWorkflowInvariantError(
            "document_revision_update_workflow_target_extraction_status_invalid"
        )
    if terminal_orchestration.orchestration_id != expected_orchestration_id:
        raise DocumentRevisionUpdateWorkflowInvariantError(
            "document_revision_update_workflow_terminal_orchestration_source_mismatch"
        )
    if not isinstance(terminal_orchestration.status, FactExtractionOrchestrationStatus):
        raise DocumentRevisionUpdateWorkflowInvariantError(
            "document_revision_update_workflow_terminal_orchestration_status_invalid"
        )
    if terminal_orchestration.status != expected_status:
        raise DocumentRevisionUpdateWorkflowInvariantError(
            "document_revision_update_workflow_terminal_orchestration_status_mismatch"
        )
    return terminal_orchestration


def _validate_authenticated_impact_source(
    impact: DocumentRevisionUpdateImpact,
    *,
    project_id: uuid.UUID,
    document_id: uuid.UUID,
    base_revision_id: uuid.UUID,
    target_revision_id: uuid.UUID,
    base_extraction_run_id: uuid.UUID,
    target_extraction_run_id: uuid.UUID,
    base_orchestration_id: uuid.UUID,
    target_orchestration_id: uuid.UUID,
    base_consistency_check_application_id: uuid.UUID,
) -> DocumentRevisionUpdateImpact:
    if (
        impact.project_id != project_id
        or impact.document_id != document_id
        or impact.base_revision_id != base_revision_id
        or impact.target_revision_id != target_revision_id
        or impact.base_extraction_run_id != base_extraction_run_id
        or impact.target_extraction_run_id != target_extraction_run_id
        or impact.base_orchestration_id != base_orchestration_id
        or impact.target_orchestration_id != target_orchestration_id
        or impact.base_consistency_check_application_id
        != base_consistency_check_application_id
    ):
        raise DocumentRevisionUpdateWorkflowInvariantError(
            "document_revision_update_workflow_impact_source_mismatch"
        )
    return impact


async def run_document_revision_update_workflow(
    session_factory: Callable[[], AsyncSession],
    *,
    project_id: uuid.UUID,
    document_id: uuid.UUID,
    base_revision_id: uuid.UUID,
    target_revision_id: uuid.UUID,
    base_extraction_run_id: uuid.UUID,
    target_extraction_run_id: uuid.UUID,
    base_orchestration_id: uuid.UUID,
    base_consistency_check_application_id: uuid.UUID,
    target_fact_extraction_plan: FactExtractionPlan,
    fact_extraction_prompt: PromptDefinition,
    llm_client: LLMClient,
    fact_extraction_provider: str,
    fact_extraction_requested_model: str,
    worker_token: uuid.UUID,
    consistency_config: ConsistencyCheckPlannerConfig,
    consistency_prompt: PromptDefinition,
    consistency_provider: str,
    consistency_requested_model: str,
    max_batch_attempts: int = 2,
    batch_lease_seconds: int = 300,
    stale_inference_seconds: int = 900,
) -> DocumentRevisionUpdateWorkflowResult:
    project_id = _require_uuid(project_id, field_name="project_id")
    document_id = _require_uuid(document_id, field_name="document_id")
    base_revision_id = _require_uuid(base_revision_id, field_name="base_revision_id")
    target_revision_id = _require_uuid(target_revision_id, field_name="target_revision_id")
    base_extraction_run_id = _require_uuid(
        base_extraction_run_id,
        field_name="base_extraction_run_id",
    )
    target_extraction_run_id = _require_uuid(
        target_extraction_run_id,
        field_name="target_extraction_run_id",
    )
    base_orchestration_id = _require_uuid(
        base_orchestration_id,
        field_name="base_orchestration_id",
    )
    base_consistency_check_application_id = _require_uuid(
        base_consistency_check_application_id,
        field_name="base_consistency_check_application_id",
    )
    worker_token = _require_uuid(worker_token, field_name="worker_token")

    pipeline_result = _validate_pipeline_result(
        await pipeline_service.run_fact_extraction_consistency_pipeline(
            session_factory,
            project_id=project_id,
            extraction_run_id=target_extraction_run_id,
            plan=target_fact_extraction_plan,
            prompt=fact_extraction_prompt,
            llm_client=llm_client,
            provider=fact_extraction_provider,
            requested_model=fact_extraction_requested_model,
            worker_token=worker_token,
            max_batch_attempts=max_batch_attempts,
            batch_lease_seconds=batch_lease_seconds,
            stale_inference_seconds=stale_inference_seconds,
            consistency_config=consistency_config,
            consistency_prompt=consistency_prompt,
            consistency_provider=consistency_provider,
            consistency_requested_model=consistency_requested_model,
        )
    )
    _validate_authenticated_terminal_orchestration(
        await orchestration_service.authenticate_terminal_fact_extraction_orchestration(
            session_factory,
            project_id=project_id,
            extraction_run_id=target_extraction_run_id,
            orchestration_id=pipeline_result.extraction_orchestration_id,
        ),
        expected_orchestration_id=pipeline_result.extraction_orchestration_id,
        expected_status=pipeline_result.extraction_status,
    )

    if pipeline_result.extraction_status == FactExtractionOrchestrationStatus.FAILED:
        return DocumentRevisionUpdateWorkflowResult(
            project_id=project_id,
            document_id=document_id,
            base_revision_id=base_revision_id,
            target_revision_id=target_revision_id,
            base_extraction_run_id=base_extraction_run_id,
            target_extraction_run_id=target_extraction_run_id,
            base_orchestration_id=base_orchestration_id,
            target_orchestration_id=pipeline_result.extraction_orchestration_id,
            target_extraction_status=pipeline_result.extraction_status,
            target_grouping_application_id=None,
            target_consistency_application_id=None,
            target_consistency_check_application_id=None,
            target_consistency_plan_manifest_hash=None,
            target_consistency_execution_manifest_hash=None,
            target_assessment_count=None,
            target_consistency_created_new=None,
            comparison_quality=None,
            impact_manifest_hash=None,
            fact_count=None,
            review_required_count=None,
            skipped_reason="target_extraction_failed",
        )

    impact = impact_service.authenticate_document_revision_update_impact_projection(
        await impact_service.get_document_revision_update_impact(
            session_factory,
            project_id=project_id,
            document_id=document_id,
            base_revision_id=base_revision_id,
            target_revision_id=target_revision_id,
            base_extraction_run_id=base_extraction_run_id,
            target_extraction_run_id=target_extraction_run_id,
            base_orchestration_id=base_orchestration_id,
            target_orchestration_id=pipeline_result.extraction_orchestration_id,
            base_consistency_check_application_id=base_consistency_check_application_id,
        )
    )
    impact = _validate_authenticated_impact_source(
        impact,
        project_id=project_id,
        document_id=document_id,
        base_revision_id=base_revision_id,
        target_revision_id=target_revision_id,
        base_extraction_run_id=base_extraction_run_id,
        target_extraction_run_id=target_extraction_run_id,
        base_orchestration_id=base_orchestration_id,
        target_orchestration_id=pipeline_result.extraction_orchestration_id,
        base_consistency_check_application_id=base_consistency_check_application_id,
    )
    return DocumentRevisionUpdateWorkflowResult(
        project_id=project_id,
        document_id=document_id,
        base_revision_id=base_revision_id,
        target_revision_id=target_revision_id,
        base_extraction_run_id=base_extraction_run_id,
        target_extraction_run_id=target_extraction_run_id,
        base_orchestration_id=base_orchestration_id,
        target_orchestration_id=pipeline_result.extraction_orchestration_id,
        target_extraction_status=pipeline_result.extraction_status,
        target_grouping_application_id=pipeline_result.grouping_application_id,
        target_consistency_application_id=pipeline_result.consistency_application_id,
        target_consistency_check_application_id=pipeline_result.consistency_check_application_id,
        target_consistency_plan_manifest_hash=pipeline_result.consistency_plan_manifest_hash,
        target_consistency_execution_manifest_hash=(
            pipeline_result.consistency_execution_result_manifest_hash
        ),
        target_assessment_count=pipeline_result.assessment_count,
        target_consistency_created_new=pipeline_result.consistency_created_new,
        comparison_quality=impact.comparison_quality,
        impact_manifest_hash=impact.impact_manifest_hash,
        fact_count=impact.fact_count,
        review_required_count=impact.review_required_count,
        skipped_reason=None,
    )
