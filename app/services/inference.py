"""Inference input-batch and run-lifecycle persistence.

This layer turns a chosen set of document blocks into an immutable, hashed input
snapshot, and records the lifecycle of each LLM call against it. It never calls an
LLM, produces facts/schemas, or touches HTTP. All statistics and hashes are
computed server-side so runs stay reproducible and traceable.
"""

from __future__ import annotations

import copy
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
from app.utils.validation import normalize_text


_VALID_TASK_TYPES = {item.value for item in InferenceTaskType}
_ADMISSIBLE_EXTRACTION_OUTCOMES = {
    ExtractionRunOutcome.SUCCESS.value,
    ExtractionRunOutcome.PARTIAL.value,
}
_BATCH_IDENTITY_CONSTRAINT = (
    "uq_inference_input_batches_project_id_task_type_snapshot_hash"
)

# Length bounds for the run identity fields, matching the ORM column sizes.
_RUN_IDENTITY_LIMITS = {
    "agent_name": 100,
    "agent_version": 32,
    "prompt_name": 100,
    "prompt_version": 32,
    "provider": 64,
    "requested_model": 128,
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
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _is_batch_identity_conflict(error: IntegrityError) -> bool:
    """True only when the failure is the batch identity unique constraint.

    Uses PostgreSQL's structured diagnostics so an unrelated IntegrityError is
    never mistaken for a concurrent duplicate batch.
    """

    constraint = getattr(
        getattr(getattr(error, "orig", None), "diag", None),
        "constraint_name",
        None,
    )
    return constraint == _BATCH_IDENTITY_CONSTRAINT


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
    # Pure input validation (no DB) — safe to raise before any transaction opens.
    task_type_value = _require_task_type(task_type)
    strategy = _require_selection_strategy(selection_strategy)
    metadata = _strict_json_metadata(selection_metadata, field_name="selection_metadata")
    ordered_block_ids = _require_block_ids(block_ids)

    try:
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
            await session.commit()  # release the read transaction/lock
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

        await inference_repository.create_inference_batch_with_blocks(session, batch)
        await session.commit()
        return batch
    except IntegrityError as error:
        await session.rollback()
        # Only the batch identity constraint is treated as a concurrent duplicate;
        # any other integrity error is re-raised untouched.
        if _is_batch_identity_conflict(error):
            existing = await inference_repository.get_batch_by_identity(
                session, project_id, task_type_value, snapshot_hash
            )
            if existing is not None:
                await session.commit()
                return existing
        raise
    except BaseException:
        await session.rollback()
        raise


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
    # Pure validation + normalization (no DB). Normalize identity BEFORE it is used
    # for the attempt query, the request hash, and the persisted row, so the stored
    # record can always recompute its own request hash.
    task_type_value = _require_task_type(task_type)
    temperature_value = _require_temperature(temperature)
    max_tokens_value = _require_max_output_tokens(max_output_tokens)
    metadata = _strict_json_metadata(request_metadata, field_name="request_metadata")
    contract_hash = _optional_hash(prompt_contract_hash, "prompt_contract_hash")
    agent_name_value = _require_identity_text(agent_name, "agent_name")
    agent_version_value = _require_identity_text(agent_version, "agent_version")
    prompt_name_value = _require_identity_text(prompt_name, "prompt_name")
    prompt_version_value = _require_identity_text(prompt_version, "prompt_version")
    provider_value = _require_identity_text(provider, "provider")
    requested_model_value = _require_identity_text(requested_model, "requested_model")

    try:
        batch = await inference_repository.get_batch_for_update(session, input_batch_id)
        if batch is None:
            raise InferenceBatchNotFoundError("Target input batch not found.")
        if batch.project_id != project_id:
            raise InferenceBatchMismatchError(
                "Batch does not belong to the given project."
            )
        if batch.task_type != task_type_value:
            raise InferenceBatchMismatchError(
                "Batch task_type does not match the run task_type."
            )

        attempt_no = await inference_repository.get_next_run_attempt_no(
            session, input_batch_id, agent_name_value, prompt_version_value
        )

        request_hash = _sha256(
            _canonical_json(
                {
                    "snapshot_hash": batch.snapshot_hash,
                    "task_type": task_type_value,
                    "agent_name": agent_name_value,
                    "agent_version": agent_version_value,
                    "prompt_name": prompt_name_value,
                    "prompt_version": prompt_version_value,
                    "prompt_contract_hash": contract_hash,
                    "provider": provider_value,
                    "requested_model": requested_model_value,
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
            agent_name=agent_name_value,
            agent_version=agent_version_value,
            prompt_name=prompt_name_value,
            prompt_version=prompt_version_value,
            prompt_contract_hash=contract_hash,
            request_hash=request_hash,
            request_metadata=metadata,
            provider=provider_value,
            requested_model=requested_model_value,
            temperature=temperature_value,
            max_output_tokens=max_tokens_value,
            attempt_count=0,
        )

        await inference_repository.create_inference_run(session, run)
        await session.commit()
        return run
    except BaseException:
        await session.rollback()
        raise


async def start_inference_run(
    session: AsyncSession,
    *,
    run_id: uuid.UUID,
) -> InferenceRun:
    try:
        run = await inference_repository.get_run_for_update(session, run_id)
        if run is None:
            raise InferenceRunNotFoundError("Target run not found.")

        if run.status == InferenceRunStatus.RUNNING.value:
            await session.commit()  # idempotent; release lock, do not move started_at
            return run
        if run.status != InferenceRunStatus.PENDING.value:
            raise InferenceRunStateError("Only pending runs can be started.")

        run.status = InferenceRunStatus.RUNNING.value
        run.started_at = utc_now()

        await session.flush()
        await session.commit()
        return run
    except BaseException:
        await session.rollback()
        raise


async def complete_inference_run(
    session: AsyncSession,
    *,
    run_id: uuid.UUID,
    completion: LLMCompletion,
) -> InferenceRun:
    response_hash = _sha256(completion.content)

    try:
        run = await inference_repository.get_run_for_update(session, run_id)
        if run is None:
            raise InferenceRunNotFoundError("Target run not found.")

        if run.status == InferenceRunStatus.COMPLETED.value:
            if run.response_hash == response_hash:
                await session.commit()  # idempotent; release lock
                return run
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
            raise InvalidInferenceCompletionError(
                "Completion attempt_count must be positive."
            )

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

        await session.flush()
        await session.commit()
        return run
    except BaseException:
        await session.rollback()
        raise


async def fail_inference_run(
    session: AsyncSession,
    *,
    run_id: uuid.UUID,
    failure_code: str,
    failure_message: str | None = None,
) -> InferenceRun:
    # Pure validation (no DB) — reject unsafe inputs before opening a transaction.
    safe_code = _require_failure_code(failure_code)
    safe_message = _require_failure_message(failure_message)

    try:
        run = await inference_repository.get_run_for_update(session, run_id)
        if run is None:
            raise InferenceRunNotFoundError("Target run not found.")

        if run.status == InferenceRunStatus.FAILED.value:
            if run.failure_code == safe_code and run.failure_message == safe_message:
                await session.commit()  # idempotent; release lock
                return run
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

        await session.flush()
        await session.commit()
        return run
    except BaseException:
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


def _strict_json_metadata(
    metadata: dict[str, Any] | None, *, field_name: str
) -> dict[str, Any]:
    """Validate and deep-copy caller metadata into a strict JSON object.

    Top level must be an object; every key must be a string; values may only be
    null / bool / int / finite float / str / list / object. NaN/Infinity, UUIDs,
    datetimes, sets, and arbitrary objects are rejected. The returned value is a
    deep copy, so later caller mutation cannot affect the stored batch/run.
    """

    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise InvalidInferenceInputError(f"{field_name} must be an object.")
    try:
        _validate_json_value(metadata, top_level=True)
    except _JsonBoundaryError as error:
        raise InvalidInferenceInputError(
            f"{field_name} must be a strict JSON object: {error.reason}"
        ) from None
    return copy.deepcopy(metadata)


class _JsonBoundaryError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _validate_json_value(value: Any, *, top_level: bool = False) -> None:
    if top_level and not isinstance(value, dict):
        raise _JsonBoundaryError("top level must be an object")
    if value is None or isinstance(value, str):
        return
    if isinstance(value, bool):
        return
    if isinstance(value, int):  # bool already handled above
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _JsonBoundaryError("NaN and Infinity are not allowed")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise _JsonBoundaryError("object keys must be strings")
            _validate_json_value(item)
        return
    raise _JsonBoundaryError(f"unsupported value type: {type(value).__name__}")


def _require_identity_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise InvalidInferenceInputError(f"{field_name} must be a string.")
    normalized = normalize_text(value)
    limit = _RUN_IDENTITY_LIMITS[field_name]
    if not 1 <= len(normalized) <= limit:
        raise InvalidInferenceInputError(
            f"{field_name} must be 1-{limit} characters."
        )
    return normalized


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
    if value == 0.0:
        value = 0.0  # normalize -0.0 to 0.0 so the hash is stable
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
