from __future__ import annotations

import uuid
from collections.abc import Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.prompt_registry import PromptDefinition
from app.schemas.consistency_check import ConsistencyCheckPlannerConfig
from app.schemas.consistency_check_workflow import ConsistencyCheckWorkflowResult
from app.services import consistency_check as consistency_check_service
from app.services import consistency_check_execution as execution_service
from app.services import consistency_check_persistence as persistence_service
from app.services import inference as inference_service
from app.services.llm import LLMClient


class ConsistencyCheckWorkflowError(Exception):
    """Base class for consistency-check workflow failures."""


class ConsistencyCheckWorkflowStateError(ConsistencyCheckWorkflowError):
    """Raised when workflow inputs are invalid or inconsistent."""


def _normalize_execution_identity_inputs(
    *,
    provider: str,
    requested_model: str,
) -> tuple[str, str]:
    try:
        return (
            inference_service.normalize_inference_identity_text(
                provider,
                field_name="provider",
            ),
            inference_service.normalize_inference_identity_text(
                requested_model,
                field_name="requested_model",
            ),
        )
    except inference_service.InvalidInferenceInputError:
        raise ConsistencyCheckWorkflowStateError(
            "consistency_check_workflow_execution_identity_invalid"
        ) from None


async def run_consistency_check_workflow(
    session_factory: Callable[[], AsyncSession],
    *,
    project_id: uuid.UUID,
    consistency_application_id: uuid.UUID,
    config: ConsistencyCheckPlannerConfig,
    prompt: PromptDefinition,
    llm_client: LLMClient,
    provider: str,
    requested_model: str,
) -> ConsistencyCheckWorkflowResult:
    normalized_provider, normalized_requested_model = _normalize_execution_identity_inputs(
        provider=provider,
        requested_model=requested_model,
    )
    authoritative_plan = await consistency_check_service.build_consistency_check_plan(
        session_factory,
        consistency_application_id=consistency_application_id,
        config=config,
    )
    if project_id != authoritative_plan.project_id:
        raise ConsistencyCheckWorkflowStateError(
            "consistency_check_workflow_project_id_mismatch"
        )

    execution_result = await execution_service.execute_consistency_check_plan(
        session_factory,
        project_id=authoritative_plan.project_id,
        plan=authoritative_plan,
        prompt=prompt,
        llm_client=llm_client,
        provider=normalized_provider,
        requested_model=normalized_requested_model,
    )
    persistence_result = await persistence_service.persist_consistency_check_plan_result(
        session_factory,
        plan=authoritative_plan,
        execution_result=execution_result,
        prompt=prompt,
        provider=normalized_provider,
        requested_model=normalized_requested_model,
    )
    return ConsistencyCheckWorkflowResult(
        project_id=authoritative_plan.project_id,
        consistency_application_id=authoritative_plan.consistency_application_id,
        plan_manifest_hash=authoritative_plan.plan_manifest_hash,
        execution_result_manifest_hash=execution_result.result_manifest_hash,
        consistency_check_application_id=persistence_result.consistency_check_application_id,
        created_new=persistence_result.created_new,
        batch_count=persistence_result.batch_count,
        assessment_count=persistence_result.assessment_count,
    )
