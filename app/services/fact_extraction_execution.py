from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.fact_extraction import (
    AgentContextError,
    AgentResponseError,
    parse_fact_extraction_completion,
    parse_fact_extraction_response_object,
    render_fact_extraction_message_contents,
    render_fact_extraction_messages,
)
from app.agents.fact_extraction_planner import PLANNER_NAME, PLANNER_VERSION
from app.agents.prompt_registry import PromptDefinition
from app.models.inference import InferenceRun, InferenceRunStatus, InferenceTaskType
from app.repositories import inference as inference_repository
from app.schemas.agent_fact_extraction import FactExtractionResponse
from app.schemas.fact_extraction_execution import (
    FactExtractionBatchExecutionResult,
    InferenceInputBlockSnapshot,
    MaterializedFactExtractionBatch,
)
from app.schemas.fact_extraction_plan import FactExtractionBatchPlan, FactExtractionPlan
from app.services.inference import (
    InferenceRunNotFoundError,
    InferenceRunStateError,
    claim_inference_run_for_execution,
    complete_inference_run,
    create_inference_input_batch,
    fail_inference_run,
    prepare_inference_run,
)
from app.services.llm import (
    LLMClient,
    LLMIncompleteResponseError,
    LLMResponseError,
    LLMTransportError,
)


logger = logging.getLogger(__name__)

FACT_EXTRACTION_EXECUTOR_NAME = "agent1_fact_extraction_batch_executor"
FACT_EXTRACTION_EXECUTOR_VERSION = "1.0.0"
_FACT_EXTRACTION_TASK_TYPE = InferenceTaskType.FACT_EXTRACTION.value
_SNAPSHOT_HASH_PLACEHOLDER = "0" * 64


class FactExtractionExecutionError(Exception):
    """Base class for batch execution failures."""


class FactExtractionPlanMaterializationError(FactExtractionExecutionError):
    """Raised when the plan and the materialized input batch diverge."""


class FactExtractionRunAlreadyRunningError(FactExtractionExecutionError):
    """Raised when another executor already claimed the same run."""


class FactExtractionEvidenceBoundsError(FactExtractionExecutionError):
    """Raised when response evidence points outside the current batch."""


@dataclass(frozen=True, slots=True)
class _InferenceRunSnapshot:
    run_id: uuid.UUID
    input_batch_id: uuid.UUID
    status: str
    request_hash: str | None
    response_json: dict[str, Any] | None
    response_model: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_uuid_instance(value: uuid.UUID, *, field_name: str) -> uuid.UUID:
    if not isinstance(value, uuid.UUID):
        raise FactExtractionExecutionError(f"{field_name} must be a UUID")
    return value


def _require_batch_index(value: int, batch_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FactExtractionExecutionError("batch_index must be an integer")
    if not 0 <= value < batch_count:
        raise FactExtractionExecutionError("batch_index is out of range")
    return value


def _require_sha256_hash(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise FactExtractionExecutionError(
            f"{field_name} must be a 64-character lowercase SHA-256 hash"
        )
    return value


def _get_batch_plan(plan: FactExtractionPlan, batch_index: int) -> FactExtractionBatchPlan:
    resolved_index = _require_batch_index(batch_index, len(plan.batches))
    batch_plan = plan.batches[resolved_index]
    if batch_plan.batch_index != resolved_index:
        raise FactExtractionExecutionError("batch plan index does not match the requested batch_index")
    return batch_plan


def _validate_plan_inputs(
    *,
    project_id: uuid.UUID,
    extraction_run_id: uuid.UUID,
    plan: FactExtractionPlan,
    batch_index: int,
    prompt: PromptDefinition,
) -> FactExtractionBatchPlan:
    _require_uuid_instance(project_id, field_name="project_id")
    _require_uuid_instance(extraction_run_id, field_name="extraction_run_id")
    if plan.extraction_run_id != extraction_run_id:
        raise FactExtractionExecutionError("plan extraction_run_id does not match the requested extraction_run_id")
    if plan.prompt_contract_hash != prompt.contract_hash:
        raise FactExtractionExecutionError("plan prompt contract hash does not match the prompt")
    if plan.planner_name != PLANNER_NAME:
        raise FactExtractionExecutionError("unexpected fact extraction planner name")
    if plan.planner_version != PLANNER_VERSION:
        raise FactExtractionExecutionError("unexpected fact extraction planner version")
    if prompt.task_type != _FACT_EXTRACTION_TASK_TYPE:
        raise FactExtractionExecutionError("prompt task_type must be fact_extraction")
    if prompt.response_model is not FactExtractionResponse:
        raise FactExtractionExecutionError("prompt response_model must be FactExtractionResponse")

    _require_sha256_hash(plan.prompt_contract_hash, field_name="plan.prompt_contract_hash")
    _require_sha256_hash(plan.plan_hash, field_name="plan.plan_hash")
    batch_plan = _get_batch_plan(plan, batch_index)
    _require_sha256_hash(batch_plan.message_template_hash, field_name="batch_plan.message_template_hash")
    _require_sha256_hash(batch_plan.plan_hash, field_name="batch_plan.plan_hash")

    if batch_plan.batch_index != batch_index:
        raise FactExtractionExecutionError("batch plan does not belong to the requested batch_index")
    return batch_plan


def _build_selection_metadata(
    *,
    extraction_run_id: uuid.UUID,
    plan: FactExtractionPlan,
    batch_plan: FactExtractionBatchPlan,
    prompt: PromptDefinition,
) -> dict[str, Any]:
    return {
        "executor_name": FACT_EXTRACTION_EXECUTOR_NAME,
        "executor_version": FACT_EXTRACTION_EXECUTOR_VERSION,
        "extraction_run_id": str(extraction_run_id),
        "plan_hash": plan.plan_hash,
        "batch_index": batch_plan.batch_index,
        "batch_plan_hash": batch_plan.plan_hash,
        "planner_name": plan.planner_name,
        "planner_version": plan.planner_version,
        "prompt_contract_hash": prompt.contract_hash,
        "message_template_hash": batch_plan.message_template_hash,
        "primary_block_ids": [str(block_id) for block_id in batch_plan.primary_block_ids],
        "overlap_block_ids": [str(block_id) for block_id in batch_plan.overlap_block_ids],
        "context_block_ids": [str(block_id) for block_id in batch_plan.context_block_ids],
    }


def _materialize_batch_snapshot(
    *,
    batch: Any,
    project_id: uuid.UUID,
    extraction_run_id: uuid.UUID,
    batch_plan: FactExtractionBatchPlan,
) -> MaterializedFactExtractionBatch:
    if batch.project_id != project_id:
        raise FactExtractionPlanMaterializationError("materialized batch project_id mismatch")
    if batch.task_type != _FACT_EXTRACTION_TASK_TYPE:
        raise FactExtractionPlanMaterializationError("materialized batch task_type mismatch")
    if "blocks" not in batch.__dict__:
        raise FactExtractionPlanMaterializationError("materialized batch blocks are not loaded")

    ordered_blocks = tuple(sorted(batch.__dict__["blocks"], key=lambda block: block.source_order))
    if len(ordered_blocks) != batch.block_count:
        raise FactExtractionPlanMaterializationError("materialized batch block_count mismatch")
    if len(ordered_blocks) != len(batch_plan.block_ids):
        raise FactExtractionPlanMaterializationError("materialized batch plan block count mismatch")

    block_ids = tuple(block.source_block_id_snapshot for block in ordered_blocks)
    if block_ids != batch_plan.block_ids:
        raise FactExtractionPlanMaterializationError("materialized batch block_ids do not match the batch plan")

    block_refs = tuple(block.block_ref for block in ordered_blocks)
    if block_refs != batch_plan.block_refs:
        raise FactExtractionPlanMaterializationError("materialized batch block_refs do not match the batch plan")

    extraction_run_snapshots = tuple(block.extraction_run_id_snapshot for block in ordered_blocks)
    if any(run_id != extraction_run_id for run_id in extraction_run_snapshots):
        raise FactExtractionPlanMaterializationError(
            "materialized batch extraction_run_id snapshots do not match the requested extraction_run_id"
        )

    character_count = 0
    snapshots: list[InferenceInputBlockSnapshot] = []
    for block in ordered_blocks:
        content_hash = _sha256_text(block.content_text)
        if content_hash != block.content_hash:
            raise FactExtractionPlanMaterializationError("materialized block content_hash mismatch")
        character_count += len(block.content_text)
        snapshots.append(
            InferenceInputBlockSnapshot(
                id=block.id,
                batch_id=batch.id,
                source_order=block.source_order,
                block_ref=block.block_ref,
                document_block_id=block.document_block_id,
                source_block_id_snapshot=block.source_block_id_snapshot,
                extraction_run_id_snapshot=block.extraction_run_id_snapshot,
                block_type=block.block_type,
                location_key=block.location_key,
                anchor_hash=block.anchor_hash,
                page_no=block.page_no,
                start_line=block.start_line,
                end_line=block.end_line,
                heading_path=tuple(block.heading_path),
                content_text=block.content_text,
                content_hash=block.content_hash,
            )
        )

    if character_count != batch.character_count:
        raise FactExtractionPlanMaterializationError("materialized batch character_count mismatch")
    if character_count != batch_plan.content_character_count:
        raise FactExtractionPlanMaterializationError("materialized batch content_character_count mismatch")

    return MaterializedFactExtractionBatch(
        id=batch.id,
        project_id=batch.project_id,
        task_type=batch.task_type,
        selection_strategy=batch.selection_strategy,
        selection_metadata=copy.deepcopy(batch.selection_metadata),
        block_count=batch.block_count,
        character_count=batch.character_count,
        snapshot_hash=batch.snapshot_hash,
        blocks=tuple(snapshots),
    )


def _build_message_content_hash(messages: Sequence[Any]) -> str:
    return _sha256_text(
        _canonical_json(
            {
                "messages": [
                    {"role": message.role, "content": message.content}
                    for message in messages
                ]
            }
        )
    )


def _validate_plan_message_binding(
    *,
    prompt: PromptDefinition,
    batch: MaterializedFactExtractionBatch,
    batch_plan: FactExtractionBatchPlan,
    messages: Sequence[Any],
) -> None:
    total_characters = sum(len(message.content) for message in messages)
    if total_characters != batch_plan.estimated_message_characters:
        raise FactExtractionPlanMaterializationError(
            "rendered message length does not match the planner estimate"
        )

    system_content, user_content = render_fact_extraction_message_contents(
        prompt=prompt,
        snapshot_hash=_SNAPSHOT_HASH_PLACEHOLDER,
        blocks=list(batch.blocks),
    )
    template_hash = _sha256_text(
        _canonical_json(
            {
                "system_content": system_content,
                "user_content": user_content,
            }
        )
    )
    if template_hash != batch_plan.message_template_hash:
        raise FactExtractionPlanMaterializationError(
            "rendered placeholder template hash does not match the batch plan"
        )


def _build_request_metadata(
    *,
    extraction_run_id: uuid.UUID,
    plan: FactExtractionPlan,
    batch_plan: FactExtractionBatchPlan,
    prompt: PromptDefinition,
    message_content_hash: str,
) -> dict[str, Any]:
    metadata = _build_selection_metadata(
        extraction_run_id=extraction_run_id,
        plan=plan,
        batch_plan=batch_plan,
        prompt=prompt,
    )
    metadata["message_content_hash"] = message_content_hash
    return metadata


def validate_fact_extraction_response_against_batch(
    *,
    response: FactExtractionResponse,
    blocks: Sequence[InferenceInputBlockSnapshot],
) -> FactExtractionResponse:
    blocks_by_ref = {block.block_ref: block for block in blocks}
    for fact in response.facts:
        for evidence in fact.evidence:
            block = blocks_by_ref.get(evidence.block_ref)
            if block is None:
                raise FactExtractionEvidenceBoundsError("evidence block_ref does not belong to this batch")
            if evidence.end_offset > len(block.content_text):
                raise FactExtractionEvidenceBoundsError("evidence offsets exceed the referenced block length")
            if evidence.start_offset < 0 or evidence.start_offset >= evidence.end_offset:
                raise FactExtractionEvidenceBoundsError("evidence offsets are invalid")
    return response


def _capture_run_snapshot(run: InferenceRun) -> _InferenceRunSnapshot:
    return _InferenceRunSnapshot(
        run_id=run.id,
        input_batch_id=run.input_batch_id,
        status=run.status,
        request_hash=run.request_hash,
        response_json=copy.deepcopy(run.response_json),
        response_model=run.response_model,
        prompt_tokens=run.prompt_tokens,
        completion_tokens=run.completion_tokens,
        total_tokens=run.total_tokens,
    )


def _build_execution_result(
    *,
    project_id: uuid.UUID,
    extraction_run_id: uuid.UUID,
    plan: FactExtractionPlan,
    batch_plan: FactExtractionBatchPlan,
    input_batch_id: uuid.UUID,
    run_snapshot: _InferenceRunSnapshot,
    message_content_hash: str,
    reused_completed_run: bool,
    response: FactExtractionResponse,
) -> FactExtractionBatchExecutionResult:
    if run_snapshot.request_hash is None:
        raise FactExtractionExecutionError("inference run request_hash is missing")
    return FactExtractionBatchExecutionResult(
        project_id=project_id,
        extraction_run_id=extraction_run_id,
        plan_hash=plan.plan_hash,
        batch_index=batch_plan.batch_index,
        batch_plan_hash=batch_plan.plan_hash,
        input_batch_id=input_batch_id,
        inference_run_id=run_snapshot.run_id,
        request_hash=run_snapshot.request_hash,
        message_content_hash=message_content_hash,
        reused_completed_run=reused_completed_run,
        response=response,
        response_model=run_snapshot.response_model,
        prompt_tokens=run_snapshot.prompt_tokens,
        completion_tokens=run_snapshot.completion_tokens,
        total_tokens=run_snapshot.total_tokens,
    )


def _map_failure_code(error: BaseException) -> str:
    if isinstance(error, LLMTransportError):
        if error.error_code == "authentication_failed":
            return "llm_authentication_failed"
        if error.error_code == "rate_limited":
            return "llm_rate_limited"
        if error.error_code == "request_timeout":
            return "llm_request_timeout"
        if error.error_code == "network_error":
            return "llm_network_error"
        if error.error_code == "upstream_unavailable":
            return "llm_server_error"
        return "llm_transport_error"
    if isinstance(error, LLMIncompleteResponseError):
        return "llm_incomplete_response"
    if isinstance(error, LLMResponseError):
        return "llm_response_invalid"
    if isinstance(error, (AgentContextError, FactExtractionPlanMaterializationError)):
        return "agent_context_invalid"
    if isinstance(error, AgentResponseError):
        return "agent_response_invalid"
    if isinstance(error, FactExtractionEvidenceBoundsError):
        return "agent_evidence_bounds_invalid"
    return "agent_response_invalid"


async def _record_failed_run(
    session_factory: Callable[[], AsyncSession],
    *,
    run_id: uuid.UUID,
    failure_code: str,
) -> None:
    async with session_factory() as session:
        try:
            await fail_inference_run(
                session,
                run_id=run_id,
                failure_code=failure_code,
                failure_message=None,
            )
        except BaseException:
            await session.rollback()
            raise


async def _safe_record_failed_run(
    session_factory: Callable[[], AsyncSession],
    *,
    run_id: uuid.UUID,
    failure_code: str,
) -> None:
    try:
        await _record_failed_run(
            session_factory,
            run_id=run_id,
            failure_code=failure_code,
        )
    except BaseException:
        logger.warning("Failed to record fact extraction inference run failure for run %s", run_id)


async def _load_run_snapshot(
    session_factory: Callable[[], AsyncSession],
    *,
    run_id: uuid.UUID,
) -> _InferenceRunSnapshot:
    async with session_factory() as session:
        try:
            run = await inference_repository.get_run_for_update(session, run_id)
            if run is None:
                raise InferenceRunNotFoundError("Target run not found.")
            snapshot = _capture_run_snapshot(run)
            await session.commit()
            return snapshot
        except BaseException:
            await session.rollback()
            raise


async def execute_fact_extraction_batch(
    session_factory: Callable[[], AsyncSession],
    *,
    project_id: uuid.UUID,
    extraction_run_id: uuid.UUID,
    plan: FactExtractionPlan,
    batch_index: int,
    prompt: PromptDefinition,
    llm_client: LLMClient,
    provider: str,
    requested_model: str,
) -> FactExtractionBatchExecutionResult:
    batch_plan = _validate_plan_inputs(
        project_id=project_id,
        extraction_run_id=extraction_run_id,
        plan=plan,
        batch_index=batch_index,
        prompt=prompt,
    )
    claimed_run_id: uuid.UUID | None = None

    try:
        selection_metadata = _build_selection_metadata(
            extraction_run_id=extraction_run_id,
            plan=plan,
            batch_plan=batch_plan,
            prompt=prompt,
        )
        async with session_factory() as session:
            try:
                created_batch = await create_inference_input_batch(
                    session,
                    project_id=project_id,
                    task_type=_FACT_EXTRACTION_TASK_TYPE,
                    block_ids=batch_plan.block_ids,
                    selection_strategy=PLANNER_NAME,
                    selection_metadata=selection_metadata,
                )
                loaded_batch = await inference_repository.get_batch_by_identity(
                    session,
                    project_id,
                    _FACT_EXTRACTION_TASK_TYPE,
                    created_batch.snapshot_hash,
                )
                if loaded_batch is None:
                    raise FactExtractionPlanMaterializationError("materialized input batch could not be reloaded")
                materialized_batch = _materialize_batch_snapshot(
                    batch=loaded_batch,
                    project_id=project_id,
                    extraction_run_id=extraction_run_id,
                    batch_plan=batch_plan,
                )
                await session.commit()
            except BaseException:
                await session.rollback()
                raise

        messages = render_fact_extraction_messages(prompt=prompt, batch=materialized_batch)
        _validate_plan_message_binding(
            prompt=prompt,
            batch=materialized_batch,
            batch_plan=batch_plan,
            messages=messages,
        )
        message_content_hash = _build_message_content_hash(messages)
        request_metadata = _build_request_metadata(
            extraction_run_id=extraction_run_id,
            plan=plan,
            batch_plan=batch_plan,
            prompt=prompt,
            message_content_hash=message_content_hash,
        )

        async with session_factory() as session:
            try:
                prepared = await prepare_inference_run(
                    session,
                    project_id=project_id,
                    input_batch_id=materialized_batch.id,
                    task_type=_FACT_EXTRACTION_TASK_TYPE,
                    agent_name=prompt.agent_name,
                    agent_version=prompt.agent_version,
                    prompt_name=prompt.prompt_name,
                    prompt_version=prompt.prompt_version,
                    prompt_contract_hash=prompt.contract_hash,
                    provider=provider,
                    requested_model=requested_model,
                    temperature=prompt.temperature,
                    max_output_tokens=prompt.max_output_tokens,
                    request_metadata=request_metadata,
                )
                prepared_snapshot = _capture_run_snapshot(prepared.run)
            except BaseException:
                await session.rollback()
                raise

        if prepared.reused_completed:
            response = parse_fact_extraction_response_object(prepared_snapshot.response_json or {})
            response = validate_fact_extraction_response_against_batch(
                response=response,
                blocks=materialized_batch.blocks,
            )
            return _build_execution_result(
                project_id=project_id,
                extraction_run_id=extraction_run_id,
                plan=plan,
                batch_plan=batch_plan,
                input_batch_id=materialized_batch.id,
                run_snapshot=prepared_snapshot,
                message_content_hash=message_content_hash,
                reused_completed_run=True,
                response=response,
            )

        async with session_factory() as session:
            try:
                claim = await claim_inference_run_for_execution(
                    session,
                    run_id=prepared_snapshot.run_id,
                )
            except BaseException:
                await session.rollback()
                raise

        if claim.status == InferenceRunStatus.COMPLETED.value:
            completed_snapshot = await _load_run_snapshot(
                session_factory,
                run_id=prepared_snapshot.run_id,
            )
            response = parse_fact_extraction_response_object(completed_snapshot.response_json or {})
            response = validate_fact_extraction_response_against_batch(
                response=response,
                blocks=materialized_batch.blocks,
            )
            return _build_execution_result(
                project_id=project_id,
                extraction_run_id=extraction_run_id,
                plan=plan,
                batch_plan=batch_plan,
                input_batch_id=materialized_batch.id,
                run_snapshot=completed_snapshot,
                message_content_hash=message_content_hash,
                reused_completed_run=True,
                response=response,
            )
        if claim.status == InferenceRunStatus.RUNNING.value and not claim.claimed:
            raise FactExtractionRunAlreadyRunningError("fact extraction inference run is already running")
        if claim.claimed:
            claimed_run_id = prepared_snapshot.run_id

        completion = await llm_client.complete(
            messages,
            temperature=prompt.temperature,
            max_tokens=prompt.max_output_tokens,
            response_format_json=True,
        )
        response = parse_fact_extraction_completion(completion)
        response = validate_fact_extraction_response_against_batch(
            response=response,
            blocks=materialized_batch.blocks,
        )

        async with session_factory() as session:
            try:
                completed_run = await complete_inference_run(
                    session,
                    run_id=prepared_snapshot.run_id,
                    completion=completion,
                )
                completed_snapshot = _capture_run_snapshot(completed_run)
            except BaseException:
                await session.rollback()
                raise

        return _build_execution_result(
            project_id=project_id,
            extraction_run_id=extraction_run_id,
            plan=plan,
            batch_plan=batch_plan,
            input_batch_id=materialized_batch.id,
            run_snapshot=completed_snapshot,
            message_content_hash=message_content_hash,
            reused_completed_run=False,
            response=response,
        )
    except asyncio.CancelledError:
        if claimed_run_id is not None:
            await asyncio.shield(
                _safe_record_failed_run(
                    session_factory,
                    run_id=claimed_run_id,
                    failure_code="fact_extraction_execution_cancelled",
                )
            )
        raise
    except BaseException as error:
        if claimed_run_id is not None:
            await _safe_record_failed_run(
                session_factory,
                run_id=claimed_run_id,
                failure_code=_map_failure_code(error),
            )
        raise
