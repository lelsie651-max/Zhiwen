from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import uuid
from collections.abc import Callable, Sequence
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.consistency_check import (
    AgentConsistencyCheckContextError,
    AgentConsistencyCheckResponseError,
    parse_consistency_check_completion,
    parse_consistency_check_response_object,
    render_consistency_check_messages,
    validate_consistency_check_batch_plan,
    validate_consistency_check_prompt,
)
from app.agents.prompt_registry import PromptDefinition
from app.models.inference import InferenceRun, InferenceRunStatus, InferenceTaskType
from app.repositories import inference as inference_repository
from app.schemas.agent_consistency_check import ConsistencyCheckResponse
from app.schemas.consistency_check import (
    ConsistencyCheckBatchPlan,
    ConsistencyCheckPlan,
)
from app.schemas.consistency_check_execution import (
    ConsistencyCheckBatchExecutionResult,
    InferenceInputBlockSnapshot,
    MaterializedConsistencyCheckBatch,
)
from app.services.consistency_check import (
    ConsistencyCheckPlanError,
    build_consistency_check_plan,
)
from app.services.inference import (
    claim_inference_run_for_execution,
    complete_inference_run,
    create_inference_input_batch,
    fail_inference_run,
    prepare_inference_run,
    build_inference_input_batch_snapshot_hash,
)
from app.services.llm import (
    LLMClient,
    LLMIncompleteResponseError,
    LLMResponseError,
    LLMTransportError,
)


logger = logging.getLogger(__name__)

CONSISTENCY_CHECK_EXECUTOR_NAME = "agent2_consistency_check_batch_executor"
CONSISTENCY_CHECK_EXECUTOR_VERSION = "1.0.0"
_CONSISTENCY_CHECK_TASK_TYPE = InferenceTaskType.CONSISTENCY_CHECK.value


class ConsistencyCheckExecutionError(Exception):
    """Base class for consistency-check execution failures."""


class ConsistencyCheckPlanMismatchError(ConsistencyCheckExecutionError):
    """Raised when the caller-provided plan does not match a rebuilt authoritative plan."""


class ConsistencyCheckBatchMaterializationError(ConsistencyCheckExecutionError):
    """Raised when the persisted inference input batch diverges from the target batch."""


class ConsistencyCheckRunAlreadyRunningError(ConsistencyCheckExecutionError):
    """Raised when another executor already claimed the same run."""


class _InferenceRunSnapshot:
    __slots__ = (
        "run_id",
        "input_batch_id",
        "status",
        "request_hash",
        "response_json",
        "response_model",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    )

    def __init__(
        self,
        *,
        run_id: uuid.UUID,
        input_batch_id: uuid.UUID,
        status: str,
        request_hash: str | None,
        response_json: dict[str, Any] | None,
        response_model: str | None,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
    ) -> None:
        self.run_id = run_id
        self.input_batch_id = input_batch_id
        self.status = status
        self.request_hash = request_hash
        self.response_json = response_json
        self.response_model = response_model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


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
        raise ConsistencyCheckExecutionError(f"{field_name} must be a UUID")
    return value


def _require_batch_index(value: int, batch_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConsistencyCheckExecutionError("batch_index must be an integer")
    if not 0 <= value < batch_count:
        raise ConsistencyCheckExecutionError("batch_index is out of range")
    return value


def _require_plan(plan: ConsistencyCheckPlan) -> ConsistencyCheckPlan:
    if not isinstance(plan, ConsistencyCheckPlan):
        raise ConsistencyCheckExecutionError("plan must be a ConsistencyCheckPlan")
    return plan


def _get_batch_plan(plan: ConsistencyCheckPlan, batch_index: int) -> ConsistencyCheckBatchPlan:
    resolved_index = _require_batch_index(batch_index, len(plan.batches))
    batch = plan.batches[resolved_index]
    if batch.batch_index != resolved_index:
        raise ConsistencyCheckExecutionError("batch plan index does not match the requested batch_index")
    return batch


def _build_selection_metadata() -> dict[str, Any]:
    return {
        "executor_name": CONSISTENCY_CHECK_EXECUTOR_NAME,
        "executor_version": CONSISTENCY_CHECK_EXECUTOR_VERSION,
        "input_kind": "consistency_check_evidence_blocks",
    }


def _collect_batch_document_block_ids(batch: ConsistencyCheckBatchPlan) -> tuple[uuid.UUID, ...]:
    ordered_block_ids: list[uuid.UUID] = []
    seen_block_ids: set[uuid.UUID] = set()
    for candidate in batch.candidates:
        for member in candidate.members:
            for evidence in member.evidences:
                block_id = evidence.document_block_id
                if block_id in seen_block_ids:
                    continue
                seen_block_ids.add(block_id)
                ordered_block_ids.append(block_id)
    return tuple(ordered_block_ids)


def _build_snapshot_records(
    blocks: Sequence[Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for block in blocks:
        records.append(
            {
                "source_order": block.source_order,
                "block_ref": block.block_ref,
                "source_block_id": str(block.source_block_id_snapshot),
                "extraction_run_id": str(block.extraction_run_id_snapshot),
                "block_type": block.block_type,
                "location_key": block.location_key,
                "anchor_hash": block.anchor_hash,
                "page_no": block.page_no,
                "start_line": block.start_line,
                "end_line": block.end_line,
                "heading_path": list(block.heading_path),
                "content_hash": block.content_hash,
            }
        )
    return records


def _raise_materialization_error() -> None:
    raise ConsistencyCheckBatchMaterializationError("consistency_check_execution_materialization_invalid")


def _materialize_batch_snapshot(
    *,
    batch: Any,
    project_id: uuid.UUID,
    expected_block_ids: Sequence[uuid.UUID],
) -> MaterializedConsistencyCheckBatch:
    if batch.project_id != project_id:
        _raise_materialization_error()
    if batch.task_type != _CONSISTENCY_CHECK_TASK_TYPE:
        _raise_materialization_error()
    if batch.selection_strategy != CONSISTENCY_CHECK_EXECUTOR_NAME:
        _raise_materialization_error()
    if batch.selection_metadata != _build_selection_metadata():
        _raise_materialization_error()
    if "blocks" not in batch.__dict__:
        _raise_materialization_error()

    ordered_blocks = tuple(sorted(batch.__dict__["blocks"], key=lambda block: block.source_order))
    if len(ordered_blocks) != batch.block_count:
        _raise_materialization_error()
    if len(ordered_blocks) != len(expected_block_ids):
        _raise_materialization_error()

    block_ids = tuple(block.source_block_id_snapshot for block in ordered_blocks)
    if block_ids != tuple(expected_block_ids):
        _raise_materialization_error()

    character_count = 0
    snapshots: list[InferenceInputBlockSnapshot] = []
    for expected_block_id, block in zip(expected_block_ids, ordered_blocks, strict=True):
        if block.document_block_id is None:
            _raise_materialization_error()
        if block.document_block_id != block.source_block_id_snapshot:
            _raise_materialization_error()
        if block.document_block_id != expected_block_id:
            _raise_materialization_error()
        content_hash = _sha256_text(block.content_text)
        if content_hash != block.content_hash:
            _raise_materialization_error()
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
        _raise_materialization_error()
    snapshot_hash = build_inference_input_batch_snapshot_hash(_build_snapshot_records(ordered_blocks))
    if snapshot_hash != batch.snapshot_hash:
        _raise_materialization_error()

    return MaterializedConsistencyCheckBatch(
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


def _build_request_metadata(
    *,
    plan: ConsistencyCheckPlan,
    batch: ConsistencyCheckBatchPlan,
    prompt: PromptDefinition,
    message_content_hash: str,
) -> dict[str, Any]:
    return {
        "project_id": str(plan.project_id),
        "consistency_application_id": str(plan.consistency_application_id),
        "source_result_manifest_hash": plan.source_result_manifest_hash,
        "plan_manifest_hash": plan.plan_manifest_hash,
        "batch_index": batch.batch_index,
        "batch_manifest_hash": batch.batch_manifest_hash,
        "message_content_hash": message_content_hash,
        "executor_name": CONSISTENCY_CHECK_EXECUTOR_NAME,
        "executor_version": CONSISTENCY_CHECK_EXECUTOR_VERSION,
        "planner_name": plan.planner_name,
        "planner_version": plan.planner_version,
        "prompt_contract_hash": prompt.contract_hash,
    }


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


def _build_prepare_run_kwargs(
    *,
    project_id: uuid.UUID,
    input_batch_id: uuid.UUID,
    prompt: PromptDefinition,
    provider: str,
    requested_model: str,
    request_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "project_id": project_id,
        "input_batch_id": input_batch_id,
        "task_type": _CONSISTENCY_CHECK_TASK_TYPE,
        "agent_name": prompt.agent_name,
        "agent_version": prompt.agent_version,
        "prompt_name": prompt.prompt_name,
        "prompt_version": prompt.prompt_version,
        "prompt_contract_hash": prompt.contract_hash,
        "provider": provider,
        "requested_model": requested_model,
        "temperature": prompt.temperature,
        "max_output_tokens": prompt.max_output_tokens,
        "request_metadata": request_metadata,
    }


async def _reprepare_completed_run(
    session_factory: Callable[[], AsyncSession],
    *,
    expected_run_id: uuid.UUID,
    prepare_kwargs: dict[str, Any],
) -> _InferenceRunSnapshot:
    async with session_factory() as session:
        try:
            prepared = await prepare_inference_run(session, **prepare_kwargs)
            if not prepared.reused_completed:
                raise ConsistencyCheckExecutionError(
                    "consistency_check_execution_completed_reprepare_invalid"
                )
            if prepared.run.id != expected_run_id:
                raise ConsistencyCheckExecutionError(
                    "consistency_check_execution_completed_run_id_mismatch"
                )
            return _capture_run_snapshot(prepared.run)
        except BaseException:
            await session.rollback()
            raise


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
        logger.warning(
            "Failed to record consistency check inference run failure",
            extra={"inference_run_id": str(run_id)},
        )


def classify_consistency_check_batch_failure(error: BaseException) -> str:
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
    if isinstance(
        error,
        (
            AgentConsistencyCheckContextError,
            ConsistencyCheckPlanError,
            ConsistencyCheckPlanMismatchError,
            ConsistencyCheckBatchMaterializationError,
            ConsistencyCheckExecutionError,
        ),
    ):
        return "consistency_check_context_invalid"
    if isinstance(error, AgentConsistencyCheckResponseError):
        return "consistency_check_response_invalid"
    return "consistency_check_response_invalid"


def _build_execution_result(
    *,
    authoritative_plan: ConsistencyCheckPlan,
    batch: ConsistencyCheckBatchPlan,
    input_batch_id: uuid.UUID | None,
    run_snapshot: _InferenceRunSnapshot | None,
    message_content_hash: str | None,
    skipped_empty: bool,
    reused_completed_run: bool,
    response: ConsistencyCheckResponse,
) -> ConsistencyCheckBatchExecutionResult:
    return ConsistencyCheckBatchExecutionResult(
        project_id=authoritative_plan.project_id,
        consistency_application_id=authoritative_plan.consistency_application_id,
        source_result_manifest_hash=authoritative_plan.source_result_manifest_hash,
        plan_manifest_hash=authoritative_plan.plan_manifest_hash,
        batch_index=batch.batch_index,
        batch_manifest_hash=batch.batch_manifest_hash,
        input_batch_id=input_batch_id,
        inference_run_id=None if run_snapshot is None else run_snapshot.run_id,
        request_hash=None if run_snapshot is None else run_snapshot.request_hash,
        message_content_hash=message_content_hash,
        skipped_empty=skipped_empty,
        reused_completed_run=reused_completed_run,
        response=response,
        response_model=None if run_snapshot is None else run_snapshot.response_model,
        prompt_tokens=None if run_snapshot is None else run_snapshot.prompt_tokens,
        completion_tokens=None if run_snapshot is None else run_snapshot.completion_tokens,
        total_tokens=None if run_snapshot is None else run_snapshot.total_tokens,
    )


async def execute_consistency_check_batch(
    session_factory: Callable[[], AsyncSession],
    *,
    project_id: uuid.UUID,
    plan: ConsistencyCheckPlan,
    batch_index: int,
    prompt: PromptDefinition,
    llm_client: LLMClient,
    provider: str,
    requested_model: str,
) -> ConsistencyCheckBatchExecutionResult:
    _require_uuid_instance(project_id, field_name="project_id")
    provided_plan = _require_plan(plan)
    _require_batch_index(batch_index, len(provided_plan.batches))
    validate_consistency_check_prompt(prompt)

    authoritative_plan = await build_consistency_check_plan(
        session_factory,
        consistency_application_id=provided_plan.consistency_application_id,
        config=provided_plan.config,
    )
    authoritative_batch = _get_batch_plan(authoritative_plan, batch_index)
    validate_consistency_check_batch_plan(plan=authoritative_plan, batch=authoritative_batch)

    if provided_plan != authoritative_plan:
        raise ConsistencyCheckPlanMismatchError("consistency_check_execution_plan_mismatch")
    if project_id != authoritative_plan.project_id:
        raise ConsistencyCheckExecutionError("consistency_check_execution_project_id_mismatch")

    if authoritative_batch.candidate_count == 0:
        return _build_execution_result(
            authoritative_plan=authoritative_plan,
            batch=authoritative_batch,
            input_batch_id=None,
            run_snapshot=None,
            message_content_hash=None,
            skipped_empty=True,
            reused_completed_run=False,
            response=ConsistencyCheckResponse(assessments=[]),
        )

    claimed_run_id: uuid.UUID | None = None
    try:
        expected_block_ids = _collect_batch_document_block_ids(authoritative_batch)
        selection_metadata = _build_selection_metadata()
        async with session_factory() as session:
            try:
                created_batch = await create_inference_input_batch(
                    session,
                    project_id=authoritative_plan.project_id,
                    task_type=_CONSISTENCY_CHECK_TASK_TYPE,
                    block_ids=expected_block_ids,
                    selection_strategy=CONSISTENCY_CHECK_EXECUTOR_NAME,
                    selection_metadata=selection_metadata,
                )
                loaded_batch = await inference_repository.get_batch_by_identity(
                    session,
                    authoritative_plan.project_id,
                    _CONSISTENCY_CHECK_TASK_TYPE,
                    created_batch.snapshot_hash,
                )
                if loaded_batch is None:
                    raise ConsistencyCheckBatchMaterializationError(
                        "materialized input batch could not be reloaded"
                    )
                materialized_batch = _materialize_batch_snapshot(
                    batch=loaded_batch,
                    project_id=authoritative_plan.project_id,
                    expected_block_ids=expected_block_ids,
                )
                await session.commit()
            except BaseException:
                await session.rollback()
                raise

        messages = render_consistency_check_messages(
            prompt=prompt,
            plan=authoritative_plan,
            batch=authoritative_batch,
        )
        message_content_hash = _build_message_content_hash(messages)
        request_metadata = _build_request_metadata(
            plan=authoritative_plan,
            batch=authoritative_batch,
            prompt=prompt,
            message_content_hash=message_content_hash,
        )
        prepare_run_kwargs = _build_prepare_run_kwargs(
            project_id=authoritative_plan.project_id,
            input_batch_id=materialized_batch.id,
            prompt=prompt,
            provider=provider,
            requested_model=requested_model,
            request_metadata=request_metadata,
        )

        async with session_factory() as session:
            try:
                prepared = await prepare_inference_run(session, **prepare_run_kwargs)
                prepared_snapshot = _capture_run_snapshot(prepared.run)
            except BaseException:
                await session.rollback()
                raise

        if prepared.reused_completed:
            response = parse_consistency_check_response_object(
                prepared_snapshot.response_json or {},
                batch=authoritative_batch,
            )
            return _build_execution_result(
                authoritative_plan=authoritative_plan,
                batch=authoritative_batch,
                input_batch_id=materialized_batch.id,
                run_snapshot=prepared_snapshot,
                message_content_hash=message_content_hash,
                skipped_empty=False,
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
            completed_snapshot = await _reprepare_completed_run(
                session_factory,
                expected_run_id=prepared_snapshot.run_id,
                prepare_kwargs=prepare_run_kwargs,
            )
            response = parse_consistency_check_response_object(
                completed_snapshot.response_json or {},
                batch=authoritative_batch,
            )
            return _build_execution_result(
                authoritative_plan=authoritative_plan,
                batch=authoritative_batch,
                input_batch_id=materialized_batch.id,
                run_snapshot=completed_snapshot,
                message_content_hash=message_content_hash,
                skipped_empty=False,
                reused_completed_run=True,
                response=response,
            )
        if claim.status == InferenceRunStatus.RUNNING.value and not claim.claimed:
            raise ConsistencyCheckRunAlreadyRunningError(
                "consistency check inference run is already running"
            )
        if claim.claimed:
            claimed_run_id = prepared_snapshot.run_id

        completion = await llm_client.complete(
            messages,
            temperature=prompt.temperature,
            max_tokens=prompt.max_output_tokens,
            response_format_json=True,
        )
        response = parse_consistency_check_completion(
            completion,
            batch=authoritative_batch,
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
            authoritative_plan=authoritative_plan,
            batch=authoritative_batch,
            input_batch_id=materialized_batch.id,
            run_snapshot=completed_snapshot,
            message_content_hash=message_content_hash,
            skipped_empty=False,
            reused_completed_run=False,
            response=response,
        )
    except asyncio.CancelledError:
        if claimed_run_id is not None:
            await asyncio.shield(
                _safe_record_failed_run(
                    session_factory,
                    run_id=claimed_run_id,
                    failure_code="consistency_check_execution_cancelled",
                )
            )
        raise
    except BaseException as error:
        if claimed_run_id is not None:
            await _safe_record_failed_run(
                session_factory,
                run_id=claimed_run_id,
                failure_code=classify_consistency_check_batch_failure(error),
            )
        raise
