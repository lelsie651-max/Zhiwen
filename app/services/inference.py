"""Inference input-batch and run-lifecycle persistence.

This layer turns a chosen set of document blocks into an immutable, hashed input
snapshot, and records the lifecycle of each LLM call against it. It never calls an
LLM, produces facts/schemas, or touches HTTP. All statistics and hashes are
computed server-side so runs stay reproducible and traceable.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document_content import ExtractionRunOutcome, ExtractionRunStatus
from app.models.inference import (
    InferenceInputBatch,
    InferenceInputBlock,
    InferenceRun,
    InferenceRunStatus,
    InferenceTaskType,
)
from app.models.project import ProjectStatus
from app.repositories import inference as inference_repository
from app.services.llm import LLMCompletion, LLMResponseError, parse_strict_json_object
from app.models.base import utc_now


_VALID_TASK_TYPES = {item.value for item in InferenceTaskType}
_ADMISSIBLE_EXTRACTION_OUTCOMES = {
    ExtractionRunOutcome.SUCCESS.value,
    ExtractionRunOutcome.PARTIAL.value,
}


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class InferenceError(Exception):
    """Base class for inference persistence failures."""


class InferenceProjectNotFoundError(InferenceError):
    """Raised when the target project does not exist."""


class InferenceProjectInactiveError(InferenceError):
    """Raised when the target project is not active."""


class InvalidInferenceInputError(InferenceError):
    """Raised when batch inputs (task type, block ids, strategy) are invalid."""


class InferenceBlockNotFoundError(InferenceError):
    """Raised when a requested block does not exist."""


class InferenceBlockNotReadyError(InferenceError):
    """Raised when a block's extraction run is not completed/admissible."""


class InferenceBatchNotFoundError(InferenceError):
    """Raised when the target input batch does not exist."""


class InferenceBatchMismatchError(InferenceError):
    """Raised when a batch does not match the run's project or task type."""


class InferenceRunNotFoundError(InferenceError):
    """Raised when the target run does not exist."""


class InferenceRunStateError(InferenceError):
    """Raised when a lifecycle transition is not allowed from the current state."""


class InferenceProviderMismatchError(InferenceError):
    """Raised when a completion's provider differs from the run's provider."""


class InvalidInferenceCompletionError(InferenceError):
    """Raised when a completion cannot be admitted (finish reason, attempt, JSON)."""


class InferenceCompletionConflictError(InferenceError):
    """Raised when a completed run is re-completed with a different response."""


class InvalidInferenceFailureError(InferenceError):
    """Raised when failure inputs are unsafe (exception objects, empty code)."""


class InferenceFailureConflictError(InferenceError):
    """Raised when a failed run is re-failed with a different result."""


# --------------------------------------------------------------------------- #
# Hashing helpers
# --------------------------------------------------------------------------- #


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _block_ref(index: int) -> str:
    return f"B{index + 1:04d}"


# --------------------------------------------------------------------------- #
# Input batch
# --------------------------------------------------------------------------- #


async def create_inference_input_batch(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    task_type: str,
    block_ids: Sequence[uuid.UUID],
    selection_strategy: str,
    selection_metadata: dict[str, Any] | None = None,
) -> InferenceInputBatch:
    task_type_value = _require_task_type(task_type)
    strategy = _require_selection_strategy(selection_strategy)
    metadata = _require_selection_metadata(selection_metadata)
    ordered_block_ids = _require_block_ids(block_ids)

    project = await inference_repository.get_project(session, project_id)
    if project is None:
        raise InferenceProjectNotFoundError("Target project not found.")
    if project.status != ProjectStatus.ACTIVE.value:
        raise InferenceProjectInactiveError("Target project is not active.")

    rows = await inference_repository.get_blocks_with_extraction_context(
        session, ordered_block_ids
    )
    row_by_block_id = {row.DocumentBlock.id: row for row in rows}

    ordered_blocks = []
    for block_id in ordered_block_ids:
        row = row_by_block_id.get(block_id)
        if row is None:
            raise InferenceBlockNotFoundError(f"Block {block_id} not found.")
        if row.project_id != project_id:
            raise InvalidInferenceInputError(
                f"Block {block_id} does not belong to the target project."
            )
        if (
            row.run_status != ExtractionRunStatus.COMPLETED.value
            or row.run_outcome not in _ADMISSIBLE_EXTRACTION_OUTCOMES
        ):
            raise InferenceBlockNotReadyError(
                f"Block {block_id} is not from a completed, admissible extraction run."
            )
        ordered_blocks.append(row.DocumentBlock)

    snapshot_records = _build_snapshot_records(ordered_blocks)
    snapshot_hash = _sha256(_canonical_json(snapshot_records))
    character_count = sum(len(block.raw_text) for block in ordered_blocks)

    existing = await inference_repository.get_batch_by_identity(
        session, project_id, task_type_value, snapshot_hash
    )
    if existing is not None:
        return existing

    batch = InferenceInputBatch(
        project_id=project_id,
        task_type=task_type_value,
        selection_strategy=strategy,
        selection_metadata=metadata,
        block_count=len(ordered_blocks),
        character_count=character_count,
        snapshot_hash=snapshot_hash,
    )
    batch.blocks = [
        InferenceInputBlock(
            source_order=index,
            block_ref=_block_ref(index),
            document_block_id=block.id,
            source_block_id_snapshot=block.id,
            extraction_run_id_snapshot=block.extraction_run_id,
            block_type=block.block_type,
            location_key=block.location_key,
            anchor_hash=block.anchor_hash,
            page_no=block.page_no,
            start_line=block.start_line,
            end_line=block.end_line,
            heading_path=list(block.heading_path),
            content_text=block.raw_text,
            content_hash=_sha256(block.raw_text),
        )
        for index, block in enumerate(ordered_blocks)
    ]

    try:
        await inference_repository.create_inference_batch_with_blocks(session, batch)
        await session.commit()
    except IntegrityError:
        await session.rollback()
        existing = await inference_repository.get_batch_by_identity(
            session, project_id, task_type_value, snapshot_hash
        )
        if existing is not None:
            return existing
        raise
    except Exception:
        await session.rollback()
        raise

    return batch


def _build_snapshot_records(blocks: Sequence[Any]) -> list[dict[str, Any]]:
    records = []
    for index, block in enumerate(blocks):
        records.append(
            {
                "source_order": index,
                "block_ref": _block_ref(index),
                "source_block_id": str(block.id),
                "extraction_run_id": str(block.extraction_run_id),
                "block_type": block.block_type,
                "location_key": block.location_key,
                "anchor_hash": block.anchor_hash,
                "page_no": block.page_no,
                "start_line": block.start_line,
                "end_line": block.end_line,
                "heading_path": list(block.heading_path),
                "content_hash": _sha256(block.raw_text),
            }
        )
    return records


# --------------------------------------------------------------------------- #
# Run lifecycle
# --------------------------------------------------------------------------- #


async def create_inference_run(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    input_batch_id: uuid.UUID,
    task_type: str,
    agent_name: str,
    agent_version: str,
    prompt_name: str,
    prompt_version: str,
    prompt_contract_hash: str | None,
    provider: str,
    requested_model: str,
    temperature: float,
    max_output_tokens: int,
    request_metadata: dict[str, Any] | None = None,
) -> InferenceRun:
    task_type_value = _require_task_type(task_type)
    temperature_value = _require_temperature(temperature)
    max_tokens_value = _require_max_output_tokens(max_output_tokens)
    metadata = _require_selection_metadata(request_metadata)
    contract_hash = _optional_hash(prompt_contract_hash, "prompt_contract_hash")

    batch = await inference_repository.get_batch_for_update(session, input_batch_id)
    if batch is None:
        raise InferenceBatchNotFoundError("Target input batch not found.")
    if batch.project_id != project_id:
        raise InferenceBatchMismatchError("Batch does not belong to the given project.")
    if batch.task_type != task_type_value:
        raise InferenceBatchMismatchError("Batch task_type does not match the run task_type.")

    attempt_no = await inference_repository.get_next_run_attempt_no(
        session, input_batch_id, agent_name, prompt_version
    )

    request_hash = _sha256(
        _canonical_json(
            {
                "snapshot_hash": batch.snapshot_hash,
                "task_type": task_type_value,
                "agent_name": agent_name,
                "agent_version": agent_version,
                "prompt_name": prompt_name,
                "prompt_version": prompt_version,
                "prompt_contract_hash": contract_hash,
                "provider": provider,
                "requested_model": requested_model,
                "temperature": temperature_value,
                "max_output_tokens": max_tokens_value,
                "request_metadata": metadata,
            }
        )
    )

    run = InferenceRun(
        project_id=project_id,
        input_batch_id=input_batch_id,
        task_type=task_type_value,
        attempt_no=attempt_no,
        status=InferenceRunStatus.PENDING.value,
        agent_name=agent_name,
        agent_version=agent_version,
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        prompt_contract_hash=contract_hash,
        request_hash=request_hash,
        request_metadata=metadata,
        provider=provider,
        requested_model=requested_model,
        temperature=temperature_value,
        max_output_tokens=max_tokens_value,
        attempt_count=0,
    )

    try:
        await inference_repository.create_inference_run(session, run)
        await session.commit()
    except Exception:
        await session.rollback()
        raise

    return run


async def start_inference_run(
    session: AsyncSession,
    *,
    run_id: uuid.UUID,
) -> InferenceRun:
    run = await inference_repository.get_run_for_update(session, run_id)
    if run is None:
        raise InferenceRunNotFoundError("Target run not found.")

    if run.status == InferenceRunStatus.RUNNING.value:
        return run  # idempotent; do not move started_at
    if run.status != InferenceRunStatus.PENDING.value:
        raise InferenceRunStateError("Only pending runs can be started.")

    run.status = InferenceRunStatus.RUNNING.value
    run.started_at = utc_now()

    await _persist(session)
    return run


async def complete_inference_run(
    session: AsyncSession,
    *,
    run_id: uuid.UUID,
    completion: LLMCompletion,
) -> InferenceRun:
    response_hash = _sha256(completion.content)

    run = await inference_repository.get_run_for_update(session, run_id)
    if run is None:
        raise InferenceRunNotFoundError("Target run not found.")

    if run.status == InferenceRunStatus.COMPLETED.value:
        if run.response_hash == response_hash:
            return run  # idempotent
        raise InferenceCompletionConflictError(
            "Run is already completed with a different response."
        )
    if run.status != InferenceRunStatus.RUNNING.value:
        raise InferenceRunStateError("Only running runs can be completed.")

    if completion.provider != run.provider:
        raise InferenceProviderMismatchError(
            "Completion provider does not match the run provider."
        )
    if completion.finish_reason != "stop":
        raise InvalidInferenceCompletionError("Completion finish_reason must be 'stop'.")
    if completion.attempt_count < 1:
        raise InvalidInferenceCompletionError("Completion attempt_count must be positive.")

    try:
        response_object = parse_strict_json_object(completion.content)
    except LLMResponseError as error:
        raise InvalidInferenceCompletionError(
            "Completion content must be a single strict JSON object."
        ) from error

    run.status = InferenceRunStatus.COMPLETED.value
    run.completed_at = utc_now()
    run.response_model = completion.model
    run.response_id = completion.response_id
    run.system_fingerprint = completion.system_fingerprint
    run.finish_reason = completion.finish_reason
    run.attempt_count = completion.attempt_count
    run.prompt_tokens = completion.usage.prompt_tokens
    run.completion_tokens = completion.usage.completion_tokens
    run.total_tokens = completion.usage.total_tokens
    run.prompt_cache_hit_tokens = completion.usage.prompt_cache_hit_tokens
    run.prompt_cache_miss_tokens = completion.usage.prompt_cache_miss_tokens
    run.reasoning_tokens = completion.usage.reasoning_tokens
    run.response_json = response_object
    run.response_hash = response_hash

    await _persist(session)
    return run


async def fail_inference_run(
    session: AsyncSession,
    *,
    run_id: uuid.UUID,
    failure_code: str,
    failure_message: str | None = None,
) -> InferenceRun:
    safe_code = _require_failure_code(failure_code)
    safe_message = _require_failure_message(failure_message)

    run = await inference_repository.get_run_for_update(session, run_id)
    if run is None:
        raise InferenceRunNotFoundError("Target run not found.")

    if run.status == InferenceRunStatus.FAILED.value:
        if run.failure_code == safe_code and run.failure_message == safe_message:
            return run  # idempotent
        raise InferenceFailureConflictError(
            "Run is already failed with a different result."
        )
    if run.status == InferenceRunStatus.COMPLETED.value:
        raise InferenceRunStateError("Completed runs cannot be failed.")
    if run.status not in (
        InferenceRunStatus.PENDING.value,
        InferenceRunStatus.RUNNING.value,
    ):
        raise InferenceRunStateError("Only pending or running runs can fail.")

    now = utc_now()
    if run.status == InferenceRunStatus.PENDING.value:
        run.started_at = now  # pending never started; stamp both for a valid shape
    run.completed_at = now
    run.status = InferenceRunStatus.FAILED.value
    run.failure_code = safe_code
    run.failure_message = safe_message

    await _persist(session)
    return run


async def _persist(session: AsyncSession) -> None:
    try:
        await session.flush()
        await session.commit()
    except Exception:
        await session.rollback()
        raise


# --------------------------------------------------------------------------- #
# Input validation helpers
# --------------------------------------------------------------------------- #


def _require_task_type(task_type: str) -> str:
    if not isinstance(task_type, str) or task_type not in _VALID_TASK_TYPES:
        raise InvalidInferenceInputError("Unknown inference task_type.")
    return task_type


def _require_selection_strategy(selection_strategy: str) -> str:
    if not isinstance(selection_strategy, str):
        raise InvalidInferenceInputError("selection_strategy must be a string.")
    stripped = selection_strategy.strip()
    if not 1 <= len(stripped) <= 64:
        raise InvalidInferenceInputError("selection_strategy must be 1-64 characters.")
    return stripped


def _require_selection_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise InvalidInferenceInputError("metadata must be an object.")
    return dict(metadata)


def _require_block_ids(block_ids: Sequence[uuid.UUID]) -> list[uuid.UUID]:
    ordered = list(block_ids)
    if not ordered:
        raise InvalidInferenceInputError("At least one block id is required.")
    if len(set(ordered)) != len(ordered):
        raise InvalidInferenceInputError("Duplicate block ids are not allowed.")
    return ordered


def _require_temperature(temperature: float) -> float:
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise InvalidInferenceInputError("temperature must be a number.")
    value = float(temperature)
    if not math.isfinite(value) or not 0.0 <= value <= 2.0:
        raise InvalidInferenceInputError("temperature must be finite within [0, 2].")
    return value


def _require_max_output_tokens(max_output_tokens: int) -> int:
    if isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int):
        raise InvalidInferenceInputError("max_output_tokens must be an integer.")
    if max_output_tokens <= 0:
        raise InvalidInferenceInputError("max_output_tokens must be positive.")
    return max_output_tokens


def _optional_hash(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidInferenceInputError(f"{field_name} must be a string.")
    normalized = value.lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise InvalidInferenceInputError(f"{field_name} must be a SHA-256 hex string.")
    return normalized


def _require_failure_code(failure_code: str) -> str:
    if isinstance(failure_code, BaseException):
        raise InvalidInferenceFailureError("failure_code must not be an exception object.")
    if not isinstance(failure_code, str):
        raise InvalidInferenceFailureError("failure_code must be a string.")
    stripped = failure_code.strip()
    if not 1 <= len(stripped) <= 64:
        raise InvalidInferenceFailureError("failure_code must be 1-64 characters.")
    return stripped


def _require_failure_message(failure_message: str | None) -> str | None:
    if failure_message is None:
        return None
    if isinstance(failure_message, BaseException):
        raise InvalidInferenceFailureError("failure_message must not be an exception object.")
    if not isinstance(failure_message, str):
        raise InvalidInferenceFailureError("failure_message must be a string.")
    stripped = failure_message.strip()
    return stripped or None
