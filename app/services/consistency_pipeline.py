from __future__ import annotations

import uuid
from collections.abc import Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.prompt_registry import PromptDefinition
from app.schemas.consistency_check import ConsistencyCheckPlannerConfig
from app.schemas.consistency_pipeline import FactExtractionConsistencyPipelineResult
from app.schemas.fact_extraction_orchestration import (
    FactExtractionOrchestrationStatus,
)
from app.schemas.fact_extraction_plan import FactExtractionPlan
from app.services import consistency_check_workflow as workflow_service
from app.services import fact_extraction_orchestration as orchestration_service
from app.services import fact_value_duplicate_grouping as duplicate_grouping_service
from app.services.llm import LLMClient


class FactExtractionConsistencyPipelineError(Exception):
    """Base class for fact extraction consistency pipeline failures."""


class FactExtractionConsistencyPipelineStateError(FactExtractionConsistencyPipelineError):
    """Raised when a pipeline stage returns an unexpected terminal state."""


async def run_fact_extraction_consistency_pipeline(
    session_factory: Callable[[], AsyncSession],
    *,
    project_id: uuid.UUID,
    extraction_run_id: uuid.UUID,
    plan: FactExtractionPlan,
    prompt: PromptDefinition,
    llm_client: LLMClient,
    provider: str,
    requested_model: str,
    worker_token: uuid.UUID,
    max_batch_attempts: int = 2,
    batch_lease_seconds: int = 300,
    stale_inference_seconds: int = 900,
    consistency_config: ConsistencyCheckPlannerConfig,
    consistency_prompt: PromptDefinition,
    consistency_provider: str,
    consistency_requested_model: str,
) -> FactExtractionConsistencyPipelineResult:
    orchestration_result = await orchestration_service.execute_fact_extraction_orchestration(
        session_factory,
        project_id=project_id,
        extraction_run_id=extraction_run_id,
        plan=plan,
        prompt=prompt,
        llm_client=llm_client,
        provider=provider,
        requested_model=requested_model,
        worker_token=worker_token,
        max_batch_attempts=max_batch_attempts,
        batch_lease_seconds=batch_lease_seconds,
        stale_inference_seconds=stale_inference_seconds,
    )

    if orchestration_result.status == FactExtractionOrchestrationStatus.FAILED:
        return FactExtractionConsistencyPipelineResult(
            extraction_orchestration_id=orchestration_result.orchestration_id,
            extraction_status=orchestration_result.status,
            grouping_application_id=None,
            consistency_application_id=None,
            consistency_check_application_id=None,
            consistency_plan_manifest_hash=None,
            consistency_execution_result_manifest_hash=None,
            assessment_count=None,
            consistency_created_new=None,
            skipped_reason="extraction_failed",
        )

    if orchestration_result.status not in {
        FactExtractionOrchestrationStatus.COMPLETED,
        FactExtractionOrchestrationStatus.PARTIAL,
    }:
        raise FactExtractionConsistencyPipelineStateError(
            "fact_extraction_consistency_pipeline_extraction_status_invalid"
        )

    grouping_result = await duplicate_grouping_service.ensure_cross_batch_duplicate_grouping(
        session_factory,
        orchestration_id=orchestration_result.orchestration_id,
    )
    candidate_result = (
        await duplicate_grouping_service.ensure_cross_batch_multi_value_consistency_candidates(
            session_factory,
            duplicate_grouping_application_id=grouping_result.grouping_application_id,
        )
    )
    workflow_result = await workflow_service.run_consistency_check_workflow(
        session_factory,
        project_id=project_id,
        consistency_application_id=candidate_result.consistency_application_id,
        config=consistency_config,
        prompt=consistency_prompt,
        llm_client=llm_client,
        provider=consistency_provider,
        requested_model=consistency_requested_model,
    )
    return FactExtractionConsistencyPipelineResult(
        extraction_orchestration_id=orchestration_result.orchestration_id,
        extraction_status=orchestration_result.status,
        grouping_application_id=grouping_result.grouping_application_id,
        consistency_application_id=candidate_result.consistency_application_id,
        consistency_check_application_id=workflow_result.consistency_check_application_id,
        consistency_plan_manifest_hash=workflow_result.plan_manifest_hash,
        consistency_execution_result_manifest_hash=workflow_result.execution_result_manifest_hash,
        assessment_count=workflow_result.assessment_count,
        consistency_created_new=workflow_result.created_new,
        skipped_reason=None,
    )
