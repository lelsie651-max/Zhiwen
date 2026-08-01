from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
import hashlib
import json
import logging
import math
import uuid
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.fact_extraction_planner import PLANNER_NAME, PLANNER_VERSION
from app.agents.prompt_registry import PromptDefinition
from app.models.base import utc_now
from app.models.document_content import ExtractionRunOutcome, ExtractionRunStatus
from app.models.fact_extraction_application import FactExtractionBatchApplication
from app.models.fact_extraction_orchestration import (
    FactExtractionOrchestration,
    FactExtractionOrchestrationBatch,
)
from app.models.inference import InferenceRunStatus
from app.repositories import fact_extraction_orchestration as orchestration_repository
from app.repositories import inference as inference_repository
from app.schemas.fact_extraction_execution import FactExtractionBatchExecutionResult
from app.schemas.fact_extraction_orchestration import (
    FactExtractionOrchestrationBatchResult,
    FactExtractionOrchestrationBatchStatus,
    FactExtractionOrchestrationResult,
    FactExtractionOrchestrationStatus,
    StaleInferenceRecoveryResult,
    StaleInferenceRecoveryStatus,
)
from app.schemas.fact_extraction_persistence import FactExtractionBatchPersistenceResult
from app.schemas.fact_extraction_plan import FactExtractionPlan
import app.services.fact_extraction_execution as execution_service
import app.services.fact_value_duplicate_grouping as duplicate_grouping_service
from app.services.fact_extraction_execution import (
    PreparedFactExtractionRunNotice,
    execute_fact_extraction_batch,
)
from app.services.fact_extraction_persistence import (
    ENTITY_RESOLUTION_POLICY_NAME,
    ENTITY_RESOLUTION_POLICY_VERSION,
    FACT_EXTRACTION_PERSISTENCE_NAME,
    FACT_EXTRACTION_PERSISTENCE_VERSION,
    build_fact_extraction_application_result_hash,
    FactExtractionApplicationReplayConflictError,
    FactExtractionPersistenceContextError,
    persist_completed_fact_extraction_batch,
    validate_fact_extraction_application_result_envelope,
)
from app.services.inference import fail_inference_run
from app.services.llm import LLMClient
from app.utils.validation import normalize_text


FACT_EXTRACTION_COORDINATOR_NAME = "agent1_fact_extraction_coordinator"
FACT_EXTRACTION_COORDINATOR_VERSION = "1.0.0"

logger = logging.getLogger(__name__)

_ORCH_STATUS_PLANNED = "planned"
_ORCH_STATUS_RUNNING = "running"
_ORCH_STATUS_COMPLETED = "completed"
_ORCH_STATUS_PARTIAL = "partial"
_ORCH_STATUS_FAILED = "failed"

_BATCH_STATUS_PENDING = "pending"
_BATCH_STATUS_RUNNING = "running"
_BATCH_STATUS_COMPLETED = "completed"
_BATCH_STATUS_FAILED = "failed"

_ORCH_ACTIVE_CONSTRAINT = "uq_feo_active_request"
_SHA256_PATTERN = "0123456789abcdef"

_RETRYABLE_FAILURE_CODES = {
    "llm_rate_limited",
    "llm_request_timeout",
    "llm_network_error",
    "llm_server_error",
    "llm_transport_error",
    "fact_extraction_execution_cancelled",
    "fact_extraction_execution_stale",
}
_NON_RETRYABLE_FAILURE_CODES = {
    "llm_authentication_failed",
    "llm_incomplete_response",
    "llm_response_invalid",
    "agent_context_invalid",
    "agent_response_invalid",
    "agent_evidence_bounds_invalid",
    "persistence_context_invalid",
    "application_replay_conflict",
    "fact_extraction_batch_lease_lost",
}


class FactExtractionOrchestrationError(Exception):
    """Base class for multi-batch fact extraction orchestration failures."""


class FactExtractionOrchestrationStateError(FactExtractionOrchestrationError):
    """Raised when orchestration state is missing, corrupt, or mismatched."""


class PreparedInferenceRunRegistrationError(FactExtractionOrchestrationError):
    """Raised when a prepared run notice cannot be authenticated against the DB."""


class FactExtractionBatchLeaseLostError(FactExtractionOrchestrationError):
    """Raised when a worker loses the lease for the current batch."""


class BatchAttemptReconciliationStatus(StrEnum):
    COMPLETED_APPLICATION = "completed_application"
    COMPLETED_RUN = "completed_run"
    FAILED_RUN = "failed_run"
    ACTIVE_RUN = "active_run"


@dataclass(frozen=True, slots=True)
class PreparedFactExtractionOrchestration:
    orchestration_id: uuid.UUID
    attempt_no: int
    request_hash: str
    plan_hash: str
    reused_completed: bool


@dataclass(frozen=True, slots=True)
class FactExtractionBatchLeaseClaim:
    orchestration_id: uuid.UUID
    batch_id: uuid.UUID
    batch_index: int
    batch_plan_hash: str
    status: str
    claimed: bool
    attempt_count: int
    current_input_batch_id: uuid.UUID | None
    current_inference_run_id: uuid.UUID | None
    application_id: uuid.UUID | None
    lease_token: uuid.UUID | None
    lease_expires_at: datetime | None
    failure_code: str | None


@dataclass(frozen=True, slots=True)
class BatchRetryTransition:
    orchestration_id: uuid.UUID
    batch_id: uuid.UUID
    batch_index: int
    status: str
    attempt_count: int
    current_input_batch_id: uuid.UUID | None
    current_inference_run_id: uuid.UUID | None
    failure_code: str | None


@dataclass(frozen=True, slots=True)
class BatchInterruptionReconciliation:
    reconciliation_status: BatchAttemptReconciliationStatus | None
    batch_status: str
    attempt_count: int
    input_batch_id: uuid.UUID | None
    inference_run_id: uuid.UUID | None
    application_id: uuid.UUID | None
    failure_code: str | None


@dataclass(frozen=True, slots=True)
class AuthenticatedBatchApplicationLedger:
    application: FactExtractionBatchApplication
    result: FactExtractionBatchPersistenceResult


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_uuid(value: uuid.UUID, *, field_name: str) -> uuid.UUID:
    if not isinstance(value, uuid.UUID):
        raise FactExtractionOrchestrationError(f"{field_name} must be a UUID")
    return value


def _require_positive_int(value: int, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FactExtractionOrchestrationError(f"{field_name} must be a positive integer")
    if value <= 0:
        raise FactExtractionOrchestrationError(f"{field_name} must be a positive integer")
    return value


def _require_text(value: str, *, field_name: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise FactExtractionOrchestrationError(f"{field_name} must be a string")
    normalized = normalize_text(value)
    if not 1 <= len(normalized) <= max_length:
        raise FactExtractionOrchestrationError(f"{field_name} must be 1-{max_length} characters")
    return normalized


def _require_hash(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise FactExtractionOrchestrationError(f"{field_name} must be a string")
    normalized = value.lower()
    if len(normalized) != 64 or any(char not in _SHA256_PATTERN for char in normalized):
        raise FactExtractionOrchestrationError(f"{field_name} must be a SHA-256 hex string")
    return normalized


def _require_aware_datetime(value: datetime, *, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise FactExtractionOrchestrationError(f"{field_name} must be timezone-aware")
    return value


def _strict_json_value(value: Any, *, top_level: bool = False) -> None:
    if top_level and type(value) is not dict:
        raise FactExtractionOrchestrationError("plan_json must be a JSON object")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FactExtractionOrchestrationError("NaN and Infinity are not allowed")
        return
    if isinstance(value, list):
        for item in value:
            _strict_json_value(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise FactExtractionOrchestrationError("object keys must be strings")
            _strict_json_value(item)
        return
    raise FactExtractionOrchestrationError(f"unsupported JSON value type: {type(value).__name__}")


def _validate_orchestration_plan(
    *,
    extraction_run_id: uuid.UUID,
    plan: FactExtractionPlan,
    prompt: PromptDefinition,
) -> dict[str, Any]:
    if plan.extraction_run_id != extraction_run_id:
        raise FactExtractionOrchestrationError("plan extraction_run_id does not match the requested extraction_run_id")
    if plan.prompt_contract_hash != prompt.contract_hash:
        raise FactExtractionOrchestrationError("plan prompt contract hash does not match the prompt")
    if plan.planner_name != PLANNER_NAME:
        raise FactExtractionOrchestrationError("unexpected fact extraction planner name")
    if plan.planner_version != PLANNER_VERSION:
        raise FactExtractionOrchestrationError("unexpected fact extraction planner version")
    _require_hash(plan.plan_hash, field_name="plan.plan_hash")
    _require_hash(plan.prompt_contract_hash, field_name="plan.prompt_contract_hash")
    batch_indexes = set()
    for expected_index, batch in enumerate(plan.batches):
        if batch.batch_index != expected_index:
            raise FactExtractionOrchestrationError("batch plan indexes must be continuous and ordered")
        if batch.batch_index in batch_indexes:
            raise FactExtractionOrchestrationError("duplicate batch indexes are not allowed")
        batch_indexes.add(batch.batch_index)
        _require_hash(batch.plan_hash, field_name="batch.plan_hash")
        _require_hash(batch.message_template_hash, field_name="batch.message_template_hash")

    expected_plan_hash = _sha256_json(
        {
            "planner_name": plan.planner_name,
            "planner_version": plan.planner_version,
            "prompt_contract_hash": plan.prompt_contract_hash,
            "config": plan.config.model_dump(mode="json"),
            "extraction_run_id": str(extraction_run_id),
            "batch_hashes": [batch.plan_hash for batch in plan.batches],
            "source_block_count": plan.source_block_count,
            "source_character_count": plan.source_character_count,
        }
    )
    if plan.plan_hash != expected_plan_hash:
        raise FactExtractionOrchestrationError("plan_hash does not match the current plan content")

    plan_json = plan.model_dump(mode="json")
    _strict_json_value(plan_json, top_level=True)
    if plan_json.get("extraction_run_id") != str(extraction_run_id):
        raise FactExtractionOrchestrationError("plan_json extraction_run_id does not match the requested extraction_run_id")
    return plan_json


def build_fact_extraction_orchestration_request_hash(
    *,
    project_id: uuid.UUID,
    extraction_run_id: uuid.UUID,
    plan_hash: str,
    plan_json_hash: str,
    planner_name: str,
    planner_version: str,
    agent_name: str,
    agent_version: str,
    prompt_name: str,
    prompt_version: str,
    prompt_contract_hash: str,
    provider: str,
    requested_model: str,
    executor_name: str,
    executor_version: str,
    persistence_name: str,
    persistence_version: str,
    entity_resolution_policy_name: str,
    entity_resolution_policy_version: str,
    coordinator_name: str,
    coordinator_version: str,
    max_batch_attempts: int,
    batch_lease_seconds: int,
    stale_inference_seconds: int,
) -> str:
    payload = {
        "project_id": str(_require_uuid(project_id, field_name="project_id")),
        "extraction_run_id": str(_require_uuid(extraction_run_id, field_name="extraction_run_id")),
        "plan_hash": _require_hash(plan_hash, field_name="plan_hash"),
        "plan_json_hash": _require_hash(plan_json_hash, field_name="plan_json_hash"),
        "planner_name": _require_text(planner_name, field_name="planner_name", max_length=64),
        "planner_version": _require_text(planner_version, field_name="planner_version", max_length=32),
        "agent_name": _require_text(agent_name, field_name="agent_name", max_length=100),
        "agent_version": _require_text(agent_version, field_name="agent_version", max_length=32),
        "prompt_name": _require_text(prompt_name, field_name="prompt_name", max_length=100),
        "prompt_version": _require_text(prompt_version, field_name="prompt_version", max_length=32),
        "prompt_contract_hash": _require_hash(prompt_contract_hash, field_name="prompt_contract_hash"),
        "provider": _require_text(provider, field_name="provider", max_length=64),
        "requested_model": _require_text(requested_model, field_name="requested_model", max_length=128),
        "executor_name": _require_text(executor_name, field_name="executor_name", max_length=64),
        "executor_version": _require_text(executor_version, field_name="executor_version", max_length=32),
        "persistence_name": _require_text(persistence_name, field_name="persistence_name", max_length=64),
        "persistence_version": _require_text(persistence_version, field_name="persistence_version", max_length=32),
        "entity_resolution_policy_name": _require_text(
            entity_resolution_policy_name,
            field_name="entity_resolution_policy_name",
            max_length=64,
        ),
        "entity_resolution_policy_version": _require_text(
            entity_resolution_policy_version,
            field_name="entity_resolution_policy_version",
            max_length=32,
        ),
        "coordinator_name": _require_text(coordinator_name, field_name="coordinator_name", max_length=64),
        "coordinator_version": _require_text(coordinator_version, field_name="coordinator_version", max_length=32),
        "max_batch_attempts": _require_positive_int(max_batch_attempts, field_name="max_batch_attempts"),
        "batch_lease_seconds": _require_positive_int(batch_lease_seconds, field_name="batch_lease_seconds"),
        "stale_inference_seconds": _require_positive_int(
            stale_inference_seconds,
            field_name="stale_inference_seconds",
        ),
    }
    return _sha256_json(payload)


def _capture_batch_claim(batch: FactExtractionOrchestrationBatch, *, claimed: bool) -> FactExtractionBatchLeaseClaim:
    return FactExtractionBatchLeaseClaim(
        orchestration_id=batch.orchestration_id,
        batch_id=batch.id,
        batch_index=batch.batch_index,
        batch_plan_hash=batch.batch_plan_hash,
        status=batch.status,
        claimed=claimed,
        attempt_count=batch.attempt_count,
        current_input_batch_id=batch.current_input_batch_id,
        current_inference_run_id=batch.current_inference_run_id,
        application_id=batch.application_id,
        lease_token=batch.lease_token,
        lease_expires_at=batch.lease_expires_at,
        failure_code=batch.failure_code,
    )


def _normalize_persistence_result_for_ledger_compare(
    result: FactExtractionBatchPersistenceResult,
) -> FactExtractionBatchPersistenceResult:
    if result.replayed_application:
        return result.model_copy(update={"replayed_application": False})
    return result


def _validate_metadata_value(
    metadata: dict[str, Any],
    *,
    key: str,
    expected: Any,
) -> None:
    value = metadata.get(key)
    if value != expected:
        raise PreparedInferenceRunRegistrationError(f"prepared run metadata {key} mismatch")


def _validate_registered_run_context(
    *,
    orchestration: FactExtractionOrchestration,
    batch: FactExtractionOrchestrationBatch,
    registration_context: inference_repository.PreparedInferenceRunRegistrationContext,
    notice: PreparedFactExtractionRunNotice,
) -> None:
    if orchestration.status != _ORCH_STATUS_RUNNING:
        raise PreparedInferenceRunRegistrationError("prepared-run registration requires a running orchestration")
    if batch.status != _BATCH_STATUS_RUNNING:
        raise PreparedInferenceRunRegistrationError("prepared-run registration requires a running batch")
    if notice.inference_run_id != registration_context.inference_run_id:
        raise PreparedInferenceRunRegistrationError("prepared run notice inference_run_id mismatch")
    if notice.input_batch_id != registration_context.input_batch_id:
        raise PreparedInferenceRunRegistrationError("prepared run notice input_batch_id mismatch")
    if notice.inference_request_hash != registration_context.inference_request_hash:
        raise PreparedInferenceRunRegistrationError("prepared run notice inference_request_hash mismatch")
    if registration_context.project_id != orchestration.project_id:
        raise PreparedInferenceRunRegistrationError("prepared run project mismatch")
    if registration_context.batch_project_id != orchestration.project_id:
        raise PreparedInferenceRunRegistrationError("prepared run input batch project mismatch")
    if registration_context.task_type != "fact_extraction":
        raise PreparedInferenceRunRegistrationError("prepared run task_type mismatch")
    if registration_context.batch_task_type != "fact_extraction":
        raise PreparedInferenceRunRegistrationError("prepared run input batch task_type mismatch")
    if registration_context.status not in {
        InferenceRunStatus.PENDING.value,
        InferenceRunStatus.RUNNING.value,
        InferenceRunStatus.COMPLETED.value,
    }:
        raise PreparedInferenceRunRegistrationError("prepared run status is not eligible for registration")
    if registration_context.agent_name != orchestration.agent_name:
        raise PreparedInferenceRunRegistrationError("prepared run agent_name mismatch")
    if registration_context.agent_version != orchestration.agent_version:
        raise PreparedInferenceRunRegistrationError("prepared run agent_version mismatch")
    if registration_context.prompt_name != orchestration.prompt_name:
        raise PreparedInferenceRunRegistrationError("prepared run prompt_name mismatch")
    if registration_context.prompt_version != orchestration.prompt_version:
        raise PreparedInferenceRunRegistrationError("prepared run prompt_version mismatch")
    if registration_context.prompt_contract_hash != orchestration.prompt_contract_hash:
        raise PreparedInferenceRunRegistrationError("prepared run prompt_contract_hash mismatch")
    if registration_context.provider != orchestration.provider:
        raise PreparedInferenceRunRegistrationError("prepared run provider mismatch")
    if registration_context.requested_model != orchestration.requested_model:
        raise PreparedInferenceRunRegistrationError("prepared run requested_model mismatch")

    metadata = registration_context.request_metadata
    if type(metadata) is not dict:
        raise PreparedInferenceRunRegistrationError("prepared run request_metadata must be a JSON object")
    _validate_metadata_value(
        metadata,
        key="extraction_run_id",
        expected=str(orchestration.extraction_run_id),
    )
    _validate_metadata_value(
        metadata,
        key="plan_hash",
        expected=orchestration.plan_hash,
    )
    _validate_metadata_value(
        metadata,
        key="batch_index",
        expected=batch.batch_index,
    )
    _validate_metadata_value(
        metadata,
        key="batch_plan_hash",
        expected=batch.batch_plan_hash,
    )
    _validate_metadata_value(
        metadata,
        key="executor_name",
        expected=orchestration.executor_name,
    )
    _validate_metadata_value(
        metadata,
        key="executor_version",
        expected=orchestration.executor_version,
    )
    _validate_metadata_value(
        metadata,
        key="planner_name",
        expected=orchestration.planner_name,
    )
    _validate_metadata_value(
        metadata,
        key="planner_version",
        expected=orchestration.planner_version,
    )
    _validate_metadata_value(
        metadata,
        key="prompt_contract_hash",
        expected=orchestration.prompt_contract_hash,
    )
    if notice.plan_hash != orchestration.plan_hash:
        raise PreparedInferenceRunRegistrationError("prepared run notice plan_hash mismatch")
    if notice.batch_plan_hash != batch.batch_plan_hash:
        raise PreparedInferenceRunRegistrationError("prepared run notice batch_plan_hash mismatch")


def _batch_counts_match_result(
    batch: FactExtractionOrchestrationBatch,
    result: FactExtractionBatchPersistenceResult,
) -> bool:
    return (
        batch.proposal_count == result.proposal_count
        and batch.created_count == result.created_count
        and batch.reused_count == result.reused_count
        and batch.withheld_count == result.withheld_count
    )


def _load_authenticated_completed_applications(
    *,
    orchestration: FactExtractionOrchestration,
    batches: Sequence[FactExtractionOrchestrationBatch],
    applications_by_id: dict[uuid.UUID, FactExtractionBatchApplication],
) -> dict[uuid.UUID, AuthenticatedBatchApplicationLedger]:
    authenticated: dict[uuid.UUID, AuthenticatedBatchApplicationLedger] = {}
    for batch in batches:
        if batch.status != _BATCH_STATUS_COMPLETED:
            continue
        if batch.application_id is None:
            raise FactExtractionOrchestrationStateError("completed batch is missing application_id")
        application = applications_by_id.get(batch.application_id)
        if application is None:
            raise FactExtractionOrchestrationStateError("completed batch application was not found")
        if application.status != "completed":
            raise FactExtractionOrchestrationStateError("batch application must be completed")
        if application.id != batch.application_id:
            raise FactExtractionOrchestrationStateError("batch application_id mismatch")
        if application.project_id != orchestration.project_id:
            raise FactExtractionOrchestrationStateError("application project mismatch")
        if application.extraction_run_id != orchestration.extraction_run_id:
            raise FactExtractionOrchestrationStateError("application extraction run mismatch")
        if application.input_batch_id != batch.current_input_batch_id:
            raise FactExtractionOrchestrationStateError("application input batch mismatch")
        if application.inference_run_id != batch.current_inference_run_id:
            raise FactExtractionOrchestrationStateError("application inference run mismatch")
        result = validate_fact_extraction_application_result_envelope(application=application)
        if not _batch_counts_match_result(batch, result):
            raise FactExtractionOrchestrationStateError("batch counts do not match the application ledger")
        authenticated[application.id] = AuthenticatedBatchApplicationLedger(
            application=application,
            result=result,
        )
    return authenticated


def validate_terminal_orchestration_state(
    *,
    orchestration: FactExtractionOrchestration,
    batches: Sequence[FactExtractionOrchestrationBatch],
    authenticated_applications: dict[uuid.UUID, AuthenticatedBatchApplicationLedger],
) -> None:
    if orchestration.status not in {
        _ORCH_STATUS_COMPLETED,
        _ORCH_STATUS_PARTIAL,
        _ORCH_STATUS_FAILED,
    }:
        raise FactExtractionOrchestrationStateError("orchestration is not terminal")
    if len(batches) != orchestration.batch_count:
        raise FactExtractionOrchestrationStateError("orchestration batch_count does not match stored batch rows")
    expected_indexes = list(range(orchestration.batch_count))
    actual_indexes = [batch.batch_index for batch in batches]
    if actual_indexes != expected_indexes:
        raise FactExtractionOrchestrationStateError("orchestration batch indexes are not continuous")

    completed_batches = [batch for batch in batches if batch.status == _BATCH_STATUS_COMPLETED]
    failed_batches = [batch for batch in batches if batch.status == _BATCH_STATUS_FAILED]
    active_batches = [batch for batch in batches if batch.status in {_BATCH_STATUS_PENDING, _BATCH_STATUS_RUNNING}]
    if active_batches:
        raise FactExtractionOrchestrationStateError("terminal orchestration cannot contain pending or running batches")
    if orchestration.completed_batch_count != len(completed_batches):
        raise FactExtractionOrchestrationStateError("orchestration completed_batch_count mismatch")
    if orchestration.failed_batch_count != len(failed_batches):
        raise FactExtractionOrchestrationStateError("orchestration failed_batch_count mismatch")

    proposal_count = 0
    created_count = 0
    reused_count = 0
    withheld_count = 0
    for batch in completed_batches:
        if batch.application_id is None:
            raise FactExtractionOrchestrationStateError("completed batch is missing application_id")
        authenticated = authenticated_applications.get(batch.application_id)
        if authenticated is None:
            raise FactExtractionOrchestrationStateError("authenticated application missing for completed batch")
        proposal_count += authenticated.result.proposal_count
        created_count += authenticated.result.created_count
        reused_count += authenticated.result.reused_count
        withheld_count += authenticated.result.withheld_count

    if orchestration.proposal_count != proposal_count:
        raise FactExtractionOrchestrationStateError("orchestration proposal_count mismatch")
    if orchestration.created_count != created_count:
        raise FactExtractionOrchestrationStateError("orchestration created_count mismatch")
    if orchestration.reused_count != reused_count:
        raise FactExtractionOrchestrationStateError("orchestration reused_count mismatch")
    if orchestration.withheld_count != withheld_count:
        raise FactExtractionOrchestrationStateError("orchestration withheld_count mismatch")

    if orchestration.status == _ORCH_STATUS_COMPLETED:
        if len(completed_batches) != orchestration.batch_count or failed_batches:
            raise FactExtractionOrchestrationStateError("completed orchestration batch states are inconsistent")
    elif orchestration.status == _ORCH_STATUS_PARTIAL:
        if not completed_batches or not failed_batches:
            raise FactExtractionOrchestrationStateError("partial orchestration must have completed and failed batches")
        if len(completed_batches) + len(failed_batches) != orchestration.batch_count:
            raise FactExtractionOrchestrationStateError("partial orchestration batch states are inconsistent")
    elif orchestration.status == _ORCH_STATUS_FAILED:
        if completed_batches or len(failed_batches) != orchestration.batch_count:
            raise FactExtractionOrchestrationStateError("failed orchestration batch states are inconsistent")


def _resolve_reconciliation_application_identity(
    context: orchestration_repository.BatchAttemptReconciliationContext,
) -> tuple[uuid.UUID | None, str | None]:
    if context.batch_application_id is not None and context.run_application_id is not None:
        if context.batch_application_id != context.run_application_id:
            raise FactExtractionOrchestrationStateError(
                "batch_application_binding_conflict"
            )
    return context.application_id, context.application_status


async def _lock_batch_attempt_reconciliation_state(
    session: AsyncSession,
    *,
    orchestration_id: uuid.UUID,
    batch_index: int,
) -> tuple[
    FactExtractionOrchestration,
    FactExtractionOrchestrationBatch,
    Any | None,
    FactExtractionBatchApplication | None,
    FactExtractionBatchApplication | None,
    orchestration_repository.BatchAttemptReconciliationContext,
]:
    orchestration = await orchestration_repository.get_orchestration_for_update(
        session,
        orchestration_id=orchestration_id,
    )
    if orchestration is None:
        raise FactExtractionOrchestrationStateError("orchestration not found")
    batch = await orchestration_repository.get_batch_for_update(
        session,
        orchestration_id=orchestration_id,
        batch_index=batch_index,
    )
    if batch is None:
        raise FactExtractionOrchestrationStateError("orchestration batch not found")

    run = None
    if batch.current_inference_run_id is not None:
        run = await inference_repository.get_run_for_update(session, batch.current_inference_run_id)
        if run is None:
            raise FactExtractionOrchestrationStateError("current_inference_run_missing")
        if run.id != batch.current_inference_run_id:
            raise FactExtractionOrchestrationStateError("batch current inference run mismatch")
        if run.project_id != orchestration.project_id:
            raise FactExtractionOrchestrationStateError("inference run project mismatch")
        if run.task_type != "fact_extraction":
            raise FactExtractionOrchestrationStateError("inference run task_type mismatch")
        if run.input_batch_id != batch.current_input_batch_id:
            raise FactExtractionOrchestrationStateError("inference run input_batch_id mismatch")

    batch_application = None
    if batch.application_id is not None:
        batch_application = await orchestration_repository.get_application_for_update(
            session,
            application_id=batch.application_id,
        )
        if batch_application is None:
            raise FactExtractionOrchestrationStateError("batch application not found")

    run_application = None
    if run is not None:
        run_application = await orchestration_repository.get_application_by_inference_run_for_update(
            session,
            inference_run_id=run.id,
        )

    if batch.current_inference_run_id is None and (batch_application is not None or run_application is not None):
        raise FactExtractionOrchestrationStateError("batch application requires current inference run")
    if batch_application is not None and run is None:
        raise FactExtractionOrchestrationStateError("batch application requires current inference run")
    if batch_application is not None and batch_application.inference_run_id != batch.current_inference_run_id:
        raise FactExtractionOrchestrationStateError("batch application inference run mismatch")
    if batch_application is not None and run_application is not None and batch_application.id != run_application.id:
        raise FactExtractionOrchestrationStateError("batch_application_binding_conflict")

    application = batch_application or run_application
    context = orchestration_repository.BatchAttemptReconciliationContext(
        orchestration_id=orchestration.id,
        orchestration_status=orchestration.status,
        orchestration_project_id=orchestration.project_id,
        orchestration_extraction_run_id=orchestration.extraction_run_id,
        batch_id=batch.id,
        batch_index=batch.batch_index,
        batch_status=batch.status,
        attempt_count=batch.attempt_count,
        lease_token=batch.lease_token,
        input_batch_id=batch.current_input_batch_id,
        inference_run_id=batch.current_inference_run_id,
        inference_run_status=None if run is None else run.status,
        inference_run_project_id=None if run is None else run.project_id,
        inference_run_task_type=None if run is None else run.task_type,
        inference_run_input_batch_id=None if run is None else run.input_batch_id,
        inference_run_failure_code=None if run is None else run.failure_code,
        application_id=None if application is None else application.id,
        application_status=None if application is None else application.status,
        batch_application_id=None if batch_application is None else batch_application.id,
        batch_application_status=None if batch_application is None else batch_application.status,
        run_application_id=None if run_application is None else run_application.id,
        run_application_status=None if run_application is None else run_application.status,
    )
    return orchestration, batch, run, batch_application, run_application, context


def _determine_reconciliation_status(
    context: orchestration_repository.BatchAttemptReconciliationContext,
) -> BatchAttemptReconciliationStatus | None:
    application_id, application_status = _resolve_reconciliation_application_identity(context)
    if context.batch_status == _BATCH_STATUS_COMPLETED:
        return BatchAttemptReconciliationStatus.COMPLETED_APPLICATION
    if application_id is not None and application_status == "completed":
        return BatchAttemptReconciliationStatus.COMPLETED_APPLICATION
    if context.inference_run_status == InferenceRunStatus.COMPLETED.value:
        return BatchAttemptReconciliationStatus.COMPLETED_RUN
    if context.inference_run_status == InferenceRunStatus.FAILED.value:
        return BatchAttemptReconciliationStatus.FAILED_RUN
    if context.inference_run_status in {
        InferenceRunStatus.PENDING.value,
        InferenceRunStatus.RUNNING.value,
    }:
        return BatchAttemptReconciliationStatus.ACTIVE_RUN
    return None


def _build_reconciliation_result(
    *,
    reconciliation_status: BatchAttemptReconciliationStatus | None,
    batch: FactExtractionOrchestrationBatch,
) -> BatchInterruptionReconciliation:
    return BatchInterruptionReconciliation(
        reconciliation_status=reconciliation_status,
        batch_status=batch.status,
        attempt_count=batch.attempt_count,
        input_batch_id=batch.current_input_batch_id,
        inference_run_id=batch.current_inference_run_id,
        application_id=batch.application_id,
        failure_code=batch.failure_code,
    )


def _finalize_batch_from_locked_completed_application(
    *,
    orchestration: FactExtractionOrchestration,
    batch: FactExtractionOrchestrationBatch,
    run: Any,
    application: FactExtractionBatchApplication,
    worker_token: uuid.UUID | None,
) -> tuple[FactExtractionBatchPersistenceResult, FactExtractionOrchestrationBatchResult]:
    ledger_result = validate_fact_extraction_application_result_envelope(application=application)
    batch_result = _apply_completed_application_to_batch(
        orchestration=orchestration,
        batch=batch,
        run=run,
        application=application,
        ledger_result=ledger_result,
        worker_token=worker_token,
    )
    return ledger_result, batch_result


def _validate_existing_batches(
    *,
    plan: FactExtractionPlan,
    batches: Sequence[FactExtractionOrchestrationBatch],
) -> None:
    if len(batches) != len(plan.batches):
        raise FactExtractionOrchestrationStateError("orchestration batch rows do not match the plan batch count")
    for expected, row in zip(plan.batches, batches, strict=True):
        if row.batch_index != expected.batch_index:
            raise FactExtractionOrchestrationStateError("orchestration batch indexes are out of order")
        if row.batch_plan_hash != expected.plan_hash:
            raise FactExtractionOrchestrationStateError("orchestration batch plan hash mismatch")


async def prepare_fact_extraction_orchestration(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    extraction_run_id: uuid.UUID,
    plan: FactExtractionPlan,
    prompt: PromptDefinition,
    provider: str,
    requested_model: str,
    max_batch_attempts: int,
    batch_lease_seconds: int,
    stale_inference_seconds: int,
) -> PreparedFactExtractionOrchestration:
    project_id = _require_uuid(project_id, field_name="project_id")
    extraction_run_id = _require_uuid(extraction_run_id, field_name="extraction_run_id")
    _require_positive_int(max_batch_attempts, field_name="max_batch_attempts")
    _require_positive_int(batch_lease_seconds, field_name="batch_lease_seconds")
    _require_positive_int(stale_inference_seconds, field_name="stale_inference_seconds")
    provider_value = _require_text(provider, field_name="provider", max_length=64)
    requested_model_value = _require_text(requested_model, field_name="requested_model", max_length=128)
    plan_json = _validate_orchestration_plan(
        extraction_run_id=extraction_run_id,
        plan=plan,
        prompt=prompt,
    )
    plan_json_hash = _sha256_json(plan_json)
    request_hash = build_fact_extraction_orchestration_request_hash(
        project_id=project_id,
        extraction_run_id=extraction_run_id,
        plan_hash=plan.plan_hash,
        plan_json_hash=plan_json_hash,
        planner_name=plan.planner_name,
        planner_version=plan.planner_version,
        agent_name=prompt.agent_name,
        agent_version=prompt.agent_version,
        prompt_name=prompt.prompt_name,
        prompt_version=prompt.prompt_version,
        prompt_contract_hash=prompt.contract_hash,
        provider=provider_value,
        requested_model=requested_model_value,
        executor_name=execution_service.FACT_EXTRACTION_EXECUTOR_NAME,
        executor_version=execution_service.FACT_EXTRACTION_EXECUTOR_VERSION,
        persistence_name=FACT_EXTRACTION_PERSISTENCE_NAME,
        persistence_version=FACT_EXTRACTION_PERSISTENCE_VERSION,
        entity_resolution_policy_name=ENTITY_RESOLUTION_POLICY_NAME,
        entity_resolution_policy_version=ENTITY_RESOLUTION_POLICY_VERSION,
        coordinator_name=FACT_EXTRACTION_COORDINATOR_NAME,
        coordinator_version=FACT_EXTRACTION_COORDINATOR_VERSION,
        max_batch_attempts=max_batch_attempts,
        batch_lease_seconds=batch_lease_seconds,
        stale_inference_seconds=stale_inference_seconds,
    )

    try:
        context = await orchestration_repository.get_extraction_run_with_project_for_update(
            session,
            extraction_run_id=extraction_run_id,
        )
        if context is None:
            raise FactExtractionOrchestrationStateError("extraction run not found")
        if context.project_id != project_id:
            raise FactExtractionOrchestrationStateError("extraction run does not belong to the requested project")
        if context.extraction_run.status != ExtractionRunStatus.COMPLETED.value:
            raise FactExtractionOrchestrationStateError("extraction run must already be completed before Agent 1 starts")
        if context.extraction_run.outcome not in {
            ExtractionRunOutcome.SUCCESS.value,
            ExtractionRunOutcome.PARTIAL.value,
        }:
            raise FactExtractionOrchestrationStateError("extraction run outcome must be success or partial")

        existing = await orchestration_repository.list_orchestrations_by_request_for_update(
            session,
            extraction_run_id=extraction_run_id,
            request_hash=request_hash,
        )
        latest = existing[0] if existing else None
        if latest is not None and latest.status == _ORCH_STATUS_COMPLETED:
            batches = await orchestration_repository.list_batches_for_orchestration_for_update(
                session,
                orchestration_id=latest.id,
            )
            _validate_existing_batches(plan=plan, batches=batches)
            await session.commit()
            return PreparedFactExtractionOrchestration(
                orchestration_id=latest.id,
                attempt_no=latest.attempt_no,
                request_hash=request_hash,
                plan_hash=plan.plan_hash,
                reused_completed=True,
            )
        if latest is not None and latest.status in {_ORCH_STATUS_PLANNED, _ORCH_STATUS_RUNNING}:
            batches = await orchestration_repository.list_batches_for_orchestration_for_update(
                session,
                orchestration_id=latest.id,
            )
            _validate_existing_batches(plan=plan, batches=batches)
            await session.commit()
            return PreparedFactExtractionOrchestration(
                orchestration_id=latest.id,
                attempt_no=latest.attempt_no,
                request_hash=request_hash,
                plan_hash=plan.plan_hash,
                reused_completed=False,
            )

        next_attempt = 1 if latest is None else latest.attempt_no + 1
        orchestration = FactExtractionOrchestration(
            project_id=project_id,
            extraction_run_id=extraction_run_id,
            attempt_no=next_attempt,
            request_hash=request_hash,
            plan_hash=plan.plan_hash,
            plan_json_hash=plan_json_hash,
            plan_json=plan_json,
            status=_ORCH_STATUS_PLANNED,
            coordinator_name=FACT_EXTRACTION_COORDINATOR_NAME,
            coordinator_version=FACT_EXTRACTION_COORDINATOR_VERSION,
            planner_name=plan.planner_name,
            planner_version=plan.planner_version,
            agent_name=prompt.agent_name,
            agent_version=prompt.agent_version,
            prompt_name=prompt.prompt_name,
            prompt_version=prompt.prompt_version,
            prompt_contract_hash=prompt.contract_hash,
            provider=provider_value,
            requested_model=requested_model_value,
            executor_name=execution_service.FACT_EXTRACTION_EXECUTOR_NAME,
            executor_version=execution_service.FACT_EXTRACTION_EXECUTOR_VERSION,
            persistence_name=FACT_EXTRACTION_PERSISTENCE_NAME,
            persistence_version=FACT_EXTRACTION_PERSISTENCE_VERSION,
            entity_resolution_policy_name=ENTITY_RESOLUTION_POLICY_NAME,
            entity_resolution_policy_version=ENTITY_RESOLUTION_POLICY_VERSION,
            batch_count=len(plan.batches),
            completed_batch_count=0,
            failed_batch_count=0,
            proposal_count=0,
            created_count=0,
            reused_count=0,
            withheld_count=0,
            failure_code=None,
            started_at=None,
            completed_at=None,
        )
        savepoint = await session.begin_nested()
        try:
            await orchestration_repository.create_orchestration(session, orchestration)
            batch_rows = [
                FactExtractionOrchestrationBatch(
                    orchestration_id=orchestration.id,
                    batch_index=batch.batch_index,
                    batch_plan_hash=batch.plan_hash,
                    status=_BATCH_STATUS_PENDING,
                    attempt_count=0,
                    current_input_batch_id=None,
                    current_inference_run_id=None,
                    application_id=None,
                    lease_token=None,
                    lease_expires_at=None,
                    proposal_count=0,
                    created_count=0,
                    reused_count=0,
                    withheld_count=0,
                    failure_code=None,
                    started_at=None,
                    completed_at=None,
                )
                for batch in plan.batches
            ]
            await orchestration_repository.create_orchestration_batches(session, batch_rows)
            await savepoint.commit()
        except IntegrityError as error:
            await savepoint.rollback()
            constraint_name = getattr(getattr(error.orig, "diag", None), "constraint_name", None)
            if constraint_name != _ORCH_ACTIVE_CONSTRAINT:
                raise
            active = await orchestration_repository.list_orchestrations_by_request_for_update(
                session,
                extraction_run_id=extraction_run_id,
                request_hash=request_hash,
            )
            if not active:
                raise error
            active_orchestration = active[0]
            batches = await orchestration_repository.list_batches_for_orchestration_for_update(
                session,
                orchestration_id=active_orchestration.id,
            )
            _validate_existing_batches(plan=plan, batches=batches)
            await session.commit()
            return PreparedFactExtractionOrchestration(
                orchestration_id=active_orchestration.id,
                attempt_no=active_orchestration.attempt_no,
                request_hash=request_hash,
                plan_hash=plan.plan_hash,
                reused_completed=active_orchestration.status == _ORCH_STATUS_COMPLETED,
            )

        await session.commit()
        return PreparedFactExtractionOrchestration(
            orchestration_id=orchestration.id,
            attempt_no=orchestration.attempt_no,
            request_hash=request_hash,
            plan_hash=plan.plan_hash,
            reused_completed=False,
        )
    except BaseException:
        await session.rollback()
        raise


async def claim_fact_extraction_orchestration_batch(
    session: AsyncSession,
    *,
    orchestration_id: uuid.UUID,
    batch_index: int,
    worker_token: uuid.UUID,
    lease_seconds: int,
    max_batch_attempts: int,
    stale_before: datetime,
) -> FactExtractionBatchLeaseClaim:
    orchestration_id = _require_uuid(orchestration_id, field_name="orchestration_id")
    worker_token = _require_uuid(worker_token, field_name="worker_token")
    _require_positive_int(lease_seconds, field_name="lease_seconds")
    _require_positive_int(max_batch_attempts, field_name="max_batch_attempts")
    stale_before = _require_aware_datetime(stale_before, field_name="stale_before")

    try:
        orchestration = await orchestration_repository.get_orchestration_for_update(
            session,
            orchestration_id=orchestration_id,
        )
        if orchestration is None:
            raise FactExtractionOrchestrationStateError("orchestration not found")
        batch = await orchestration_repository.get_batch_for_update(
            session,
            orchestration_id=orchestration_id,
            batch_index=batch_index,
        )
        if batch is None:
            raise FactExtractionOrchestrationStateError("orchestration batch not found")

        now = utc_now()
        expires_at = now + timedelta(seconds=lease_seconds)
        if batch.status in {_BATCH_STATUS_COMPLETED, _BATCH_STATUS_FAILED}:
            await session.commit()
            return _capture_batch_claim(batch, claimed=False)

        if batch.status == _BATCH_STATUS_PENDING:
            if batch.attempt_count >= max_batch_attempts:
                raise FactExtractionOrchestrationStateError(
                    "pending batch has already exhausted max_batch_attempts"
                )
            batch.status = _BATCH_STATUS_RUNNING
            batch.attempt_count += 1
            batch.lease_token = worker_token
            batch.lease_expires_at = expires_at
            batch.started_at = now
            batch.completed_at = None
            batch.failure_code = None
            if orchestration.status == _ORCH_STATUS_PLANNED:
                orchestration.status = _ORCH_STATUS_RUNNING
                orchestration.started_at = now
                orchestration.completed_at = None
                orchestration.failure_code = None
            await session.flush()
            await session.commit()
            return _capture_batch_claim(batch, claimed=True)

        if batch.lease_token == worker_token:
            batch.lease_expires_at = expires_at
            if orchestration.status == _ORCH_STATUS_PLANNED:
                orchestration.status = _ORCH_STATUS_RUNNING
                orchestration.started_at = now
            await session.flush()
            await session.commit()
            return _capture_batch_claim(batch, claimed=True)

        if batch.lease_expires_at is not None and batch.lease_expires_at > now:
            if orchestration.status == _ORCH_STATUS_PLANNED:
                orchestration.status = _ORCH_STATUS_RUNNING
                orchestration.started_at = now
            await session.commit()
            return _capture_batch_claim(batch, claimed=False)

        current_run = None
        if batch.current_inference_run_id is not None:
            current_run = await inference_repository.get_run_for_update(
                session,
                batch.current_inference_run_id,
            )
        if (
            current_run is not None
            and current_run.status == InferenceRunStatus.RUNNING.value
            and current_run.started_at is not None
            and current_run.started_at > stale_before
        ):
            await session.commit()
            return _capture_batch_claim(batch, claimed=False)

        batch.status = _BATCH_STATUS_RUNNING
        batch.lease_token = worker_token
        batch.lease_expires_at = expires_at
        if batch.started_at is None:
            batch.started_at = now
        batch.completed_at = None
        batch.failure_code = None
        if orchestration.status == _ORCH_STATUS_PLANNED:
            orchestration.status = _ORCH_STATUS_RUNNING
            orchestration.started_at = now
            orchestration.completed_at = None
            orchestration.failure_code = None
        await session.flush()
        await session.commit()
        return _capture_batch_claim(batch, claimed=True)
    except BaseException:
        await session.rollback()
        raise


async def renew_fact_extraction_orchestration_batch_lease(
    session: AsyncSession,
    *,
    orchestration_id: uuid.UUID,
    batch_index: int,
    worker_token: uuid.UUID,
    lease_seconds: int,
) -> FactExtractionBatchLeaseClaim:
    orchestration_id = _require_uuid(orchestration_id, field_name="orchestration_id")
    worker_token = _require_uuid(worker_token, field_name="worker_token")
    _require_positive_int(lease_seconds, field_name="lease_seconds")

    try:
        orchestration = await orchestration_repository.get_orchestration_for_update(
            session,
            orchestration_id=orchestration_id,
        )
        if orchestration is None:
            raise FactExtractionOrchestrationStateError("orchestration not found")
        if orchestration.status != _ORCH_STATUS_RUNNING:
            raise FactExtractionBatchLeaseLostError("orchestration is no longer running")
        batch = await orchestration_repository.get_batch_for_update(
            session,
            orchestration_id=orchestration_id,
            batch_index=batch_index,
        )
        if batch is None:
            raise FactExtractionOrchestrationStateError("orchestration batch not found")
        if batch.status != _BATCH_STATUS_RUNNING or batch.lease_token != worker_token:
            raise FactExtractionBatchLeaseLostError("batch lease is not owned by the current worker")

        batch.lease_expires_at = utc_now() + timedelta(seconds=lease_seconds)
        await session.flush()
        await session.commit()
        return _capture_batch_claim(batch, claimed=True)
    except BaseException:
        await session.rollback()
        raise


async def recover_stale_fact_extraction_inference_run(
    session: AsyncSession,
    *,
    inference_run_id: uuid.UUID,
    stale_before: datetime,
) -> StaleInferenceRecoveryResult:
    inference_run_id = _require_uuid(inference_run_id, field_name="inference_run_id")
    stale_before = _require_aware_datetime(stale_before, field_name="stale_before")

    try:
        run = await inference_repository.get_run_for_update(session, inference_run_id)
        if run is None:
            raise FactExtractionOrchestrationStateError("inference run not found")
        if run.status == InferenceRunStatus.COMPLETED.value:
            await session.commit()
            return StaleInferenceRecoveryResult(
                inference_run_id=inference_run_id,
                status=StaleInferenceRecoveryStatus.COMPLETED,
                recovered_stale_run=False,
                failure_code=None,
            )
        if run.status == InferenceRunStatus.FAILED.value:
            await session.commit()
            return StaleInferenceRecoveryResult(
                inference_run_id=inference_run_id,
                status=StaleInferenceRecoveryStatus.FAILED,
                recovered_stale_run=False,
                failure_code=run.failure_code,
            )
        if run.status == InferenceRunStatus.PENDING.value:
            await session.commit()
            return StaleInferenceRecoveryResult(
                inference_run_id=inference_run_id,
                status=StaleInferenceRecoveryStatus.PENDING,
                recovered_stale_run=False,
                failure_code=None,
            )
        if run.started_at is not None and run.started_at > stale_before:
            await session.commit()
            return StaleInferenceRecoveryResult(
                inference_run_id=inference_run_id,
                status=StaleInferenceRecoveryStatus.RUNNING,
                recovered_stale_run=False,
                failure_code=None,
            )

        run.status = InferenceRunStatus.FAILED.value
        run.completed_at = utc_now()
        run.failure_code = "fact_extraction_execution_stale"
        run.failure_message = None
        run.response_json = None
        run.response_hash = None
        run.response_json_hash = None
        await session.flush()
        await session.commit()
        return StaleInferenceRecoveryResult(
            inference_run_id=inference_run_id,
            status=StaleInferenceRecoveryStatus.FAILED,
            recovered_stale_run=True,
            failure_code="fact_extraction_execution_stale",
        )
    except BaseException:
        await session.rollback()
        raise


async def _record_prepared_run_notice(
    session: AsyncSession,
    *,
    orchestration_id: uuid.UUID,
    worker_token: uuid.UUID,
    notice: PreparedFactExtractionRunNotice,
) -> None:
    try:
        orchestration = await orchestration_repository.get_orchestration_for_update(
            session,
            orchestration_id=orchestration_id,
        )
        if orchestration is None:
            raise PreparedInferenceRunRegistrationError(
                "orchestration not found during prepared-run registration"
            )
        if orchestration.project_id != notice.project_id:
            raise PreparedInferenceRunRegistrationError("prepared run project mismatch")
        if orchestration.extraction_run_id != notice.extraction_run_id:
            raise PreparedInferenceRunRegistrationError("prepared run extraction_run_id mismatch")
        batch = await orchestration_repository.get_batch_for_update(
            session,
            orchestration_id=orchestration_id,
            batch_index=notice.batch_index,
        )
        if batch is None:
            raise PreparedInferenceRunRegistrationError(
                "orchestration batch not found during prepared-run registration"
            )
        if batch.status != _BATCH_STATUS_RUNNING:
            raise PreparedInferenceRunRegistrationError("prepared-run registration requires a running batch")
        if batch.lease_token != worker_token:
            raise PreparedInferenceRunRegistrationError("prepared-run registration lease token mismatch")
        registration_context = await inference_repository.get_prepared_inference_run_registration_context(
            session,
            inference_run_id=notice.inference_run_id,
        )
        if registration_context is None:
            raise PreparedInferenceRunRegistrationError(
                "prepared-run registration inference run was not found"
            )
        _validate_registered_run_context(
            orchestration=orchestration,
            batch=batch,
            registration_context=registration_context,
            notice=notice,
        )
        if batch.current_input_batch_id not in {None, registration_context.input_batch_id}:
            raise PreparedInferenceRunRegistrationError(
                "prepared-run registration input_batch_id mismatch"
            )
        if batch.current_inference_run_id not in {None, registration_context.inference_run_id}:
            raise PreparedInferenceRunRegistrationError(
                "prepared-run registration inference_run_id mismatch"
            )
        batch.current_input_batch_id = registration_context.input_batch_id
        batch.current_inference_run_id = registration_context.inference_run_id
        await session.flush()
        await session.commit()
    except BaseException:
        await session.rollback()
        raise


def _apply_completed_application_to_batch(
    *,
    orchestration: FactExtractionOrchestration,
    batch: FactExtractionOrchestrationBatch,
    run: Any,
    application: FactExtractionBatchApplication,
    ledger_result: FactExtractionBatchPersistenceResult,
    worker_token: uuid.UUID | None,
) -> FactExtractionOrchestrationBatchResult:
    if application.status != "completed":
        raise FactExtractionOrchestrationStateError("batch application must be completed")
    if application.project_id != orchestration.project_id:
        raise FactExtractionOrchestrationStateError("application project mismatch")
    if application.extraction_run_id != orchestration.extraction_run_id:
        raise FactExtractionOrchestrationStateError("application extraction run mismatch")
    if run.id != application.inference_run_id:
        raise FactExtractionOrchestrationStateError("application inference run mismatch")
    if run.status != InferenceRunStatus.COMPLETED.value:
        raise FactExtractionOrchestrationStateError("completed application requires a completed inference run")
    if run.project_id != orchestration.project_id:
        raise FactExtractionOrchestrationStateError("inference run project mismatch")
    if run.task_type != "fact_extraction":
        raise FactExtractionOrchestrationStateError("inference run task_type mismatch")
    if run.input_batch_id != application.input_batch_id:
        raise FactExtractionOrchestrationStateError("application input batch mismatch")
    if batch.current_input_batch_id not in {None, application.input_batch_id}:
        raise FactExtractionOrchestrationStateError("batch current_input_batch_id mismatch")
    if batch.current_inference_run_id not in {None, application.inference_run_id}:
        raise FactExtractionOrchestrationStateError("batch current_inference_run_id mismatch")
    if worker_token is not None and batch.status != _BATCH_STATUS_COMPLETED and batch.lease_token != worker_token:
        raise FactExtractionOrchestrationStateError("batch lease is not owned by the current worker")
    if batch.status == _BATCH_STATUS_COMPLETED:
        if batch.application_id != application.id:
            raise FactExtractionOrchestrationStateError("completed batch cannot be overwritten by a different application")
        if not _batch_counts_match_result(batch, ledger_result):
            raise FactExtractionOrchestrationStateError("batch counts do not match the application ledger")
        return _build_batch_result(batch)

    batch.status = _BATCH_STATUS_COMPLETED
    batch.current_input_batch_id = application.input_batch_id
    batch.current_inference_run_id = application.inference_run_id
    batch.application_id = application.id
    batch.proposal_count = ledger_result.proposal_count
    batch.created_count = ledger_result.created_count
    batch.reused_count = ledger_result.reused_count
    batch.withheld_count = ledger_result.withheld_count
    batch.failure_code = None
    batch.lease_token = None
    batch.lease_expires_at = None
    if batch.completed_at is None:
        batch.completed_at = utc_now()
    return _build_batch_result(batch)


async def finalize_batch_from_completed_application(
    session: AsyncSession,
    *,
    orchestration_id: uuid.UUID,
    batch_index: int,
    worker_token: uuid.UUID | None,
    inference_run_id: uuid.UUID,
    application_id: uuid.UUID,
) -> FactExtractionOrchestrationBatchResult:
    try:
        orchestration, batch, run, _batch_application, _run_application, _context = await _lock_batch_attempt_reconciliation_state(
            session,
            orchestration_id=orchestration_id,
            batch_index=batch_index,
        )
        if run is None:
            raise FactExtractionOrchestrationStateError("inference run not found")
        if run.id != inference_run_id:
            raise FactExtractionOrchestrationStateError("inference run mismatch")
        application = await orchestration_repository.get_application_for_update(
            session,
            application_id=application_id,
        )
        if application is None:
            raise FactExtractionOrchestrationStateError("batch application not found")
        _ledger_result, result = _finalize_batch_from_locked_completed_application(
            orchestration=orchestration,
            batch=batch,
            run=run,
            application=application,
            worker_token=worker_token,
        )
        await session.flush()
        await session.commit()
        return result
    except BaseException:
        await session.rollback()
        raise


async def transition_batch_after_failed_inference_attempt(
    session: AsyncSession,
    *,
    orchestration_id: uuid.UUID,
    batch_index: int,
    worker_token: uuid.UUID,
    inference_run_id: uuid.UUID,
    failure_code: str,
    max_batch_attempts: int,
) -> BatchRetryTransition:
    orchestration_id = _require_uuid(orchestration_id, field_name="orchestration_id")
    worker_token = _require_uuid(worker_token, field_name="worker_token")
    inference_run_id = _require_uuid(inference_run_id, field_name="inference_run_id")
    _require_positive_int(max_batch_attempts, field_name="max_batch_attempts")
    try:
        orchestration = await orchestration_repository.get_orchestration_for_update(
            session,
            orchestration_id=orchestration_id,
        )
        if orchestration is None:
            raise FactExtractionOrchestrationStateError("orchestration not found")
        batch = await orchestration_repository.get_batch_for_update(
            session,
            orchestration_id=orchestration_id,
            batch_index=batch_index,
        )
        if batch is None:
            raise FactExtractionOrchestrationStateError("orchestration batch not found")
        if batch.status == _BATCH_STATUS_COMPLETED:
            await session.commit()
            return BatchRetryTransition(
                orchestration_id=batch.orchestration_id,
                batch_id=batch.id,
                batch_index=batch.batch_index,
                status=batch.status,
                attempt_count=batch.attempt_count,
                current_input_batch_id=batch.current_input_batch_id,
                current_inference_run_id=batch.current_inference_run_id,
                failure_code=batch.failure_code,
            )
        if batch.lease_token != worker_token:
            raise FactExtractionOrchestrationStateError("batch lease is not owned by the current worker")
        if batch.current_inference_run_id != inference_run_id:
            raise FactExtractionOrchestrationStateError("batch current inference run mismatch")
        run = await inference_repository.get_run_for_update(session, inference_run_id)
        if run is None:
            raise FactExtractionOrchestrationStateError("inference run not found")
        if run.status == InferenceRunStatus.COMPLETED.value:
            raise FactExtractionOrchestrationStateError("completed_inference_requires_reconciliation")
        if run.status in {InferenceRunStatus.PENDING.value, InferenceRunStatus.RUNNING.value}:
            raise FactExtractionOrchestrationStateError("active_inference_requires_recovery")
        if run.status != InferenceRunStatus.FAILED.value:
            raise FactExtractionOrchestrationStateError("failed transition requires a failed inference run")
        if run.project_id != orchestration.project_id:
            raise FactExtractionOrchestrationStateError("inference run project mismatch")
        if run.task_type != "fact_extraction":
            raise FactExtractionOrchestrationStateError("inference run task_type mismatch")
        if run.input_batch_id != batch.current_input_batch_id:
            raise FactExtractionOrchestrationStateError("inference run input_batch_id mismatch")
        if run.failure_code is None:
            raise FactExtractionOrchestrationStateError("failed inference run is missing failure_code")
        if failure_code != run.failure_code:
            raise FactExtractionOrchestrationStateError("batch failure_code is incompatible with the failed inference run")

        retryable = _is_retryable_failure_code(failure_code) and batch.attempt_count < max_batch_attempts
        if retryable:
            batch.status = _BATCH_STATUS_PENDING
            batch.current_inference_run_id = None
            batch.application_id = None
            batch.lease_token = None
            batch.lease_expires_at = None
            batch.started_at = None
            batch.completed_at = None
            batch.failure_code = None
            batch.proposal_count = 0
            batch.created_count = 0
            batch.reused_count = 0
            batch.withheld_count = 0
        else:
            batch.status = _BATCH_STATUS_FAILED
            batch.failure_code = failure_code
            batch.completed_at = utc_now()
            batch.lease_token = None
            batch.lease_expires_at = None
        await session.flush()
        await session.commit()
        return BatchRetryTransition(
            orchestration_id=batch.orchestration_id,
            batch_id=batch.id,
            batch_index=batch.batch_index,
            status=batch.status,
            attempt_count=batch.attempt_count,
            current_input_batch_id=batch.current_input_batch_id,
            current_inference_run_id=batch.current_inference_run_id,
            failure_code=batch.failure_code,
        )
    except BaseException:
        await session.rollback()
        raise


def _classify_batch_failure(error: BaseException) -> str:
    if isinstance(error, asyncio.CancelledError):
        return "fact_extraction_execution_cancelled"
    if isinstance(error, FactExtractionBatchLeaseLostError):
        return "fact_extraction_batch_lease_lost"
    if isinstance(error, FactExtractionPersistenceContextError):
        return "persistence_context_invalid"
    if isinstance(error, FactExtractionApplicationReplayConflictError):
        return "application_replay_conflict"
    return execution_service.classify_fact_extraction_batch_failure(error)


def _is_retryable_failure_code(failure_code: str) -> bool:
    return failure_code in _RETRYABLE_FAILURE_CODES


async def run_with_batch_lease_heartbeat(
    session_factory: Callable[[], AsyncSession],
    *,
    orchestration_id: uuid.UUID,
    batch_index: int,
    worker_token: uuid.UUID,
    batch_lease_seconds: int,
    operation: Callable[[], Any],
) -> Any:
    heartbeat_interval = max(1, batch_lease_seconds // 3)
    lease_lost = asyncio.Event()

    async def _heartbeat_loop() -> None:
        while not lease_lost.is_set():
            await asyncio.sleep(heartbeat_interval)
            if lease_lost.is_set():
                return
            try:
                async with session_factory() as session:
                    await renew_fact_extraction_orchestration_batch_lease(
                        session,
                        orchestration_id=orchestration_id,
                        batch_index=batch_index,
                        worker_token=worker_token,
                        lease_seconds=batch_lease_seconds,
                    )
            except BaseException as error:
                lease_lost.set()
                if isinstance(error, asyncio.CancelledError):
                    raise
                raise FactExtractionBatchLeaseLostError("batch lease heartbeat failed") from None

    operation_task = asyncio.create_task(operation())
    heartbeat_task = asyncio.create_task(_heartbeat_loop())
    try:
        done, pending = await asyncio.wait(
            {operation_task, heartbeat_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if heartbeat_task in done:
            operation_task.cancel()
            try:
                await operation_task
            except BaseException:
                pass
            await heartbeat_task
            raise FactExtractionBatchLeaseLostError("batch lease heartbeat failed")
        result = await operation_task
        lease_lost.set()
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        return result
    finally:
        lease_lost.set()
        for task in (operation_task, heartbeat_task):
            if not task.done():
                task.cancel()
        for task in (operation_task, heartbeat_task):
            try:
                await task
            except BaseException:
                pass


async def _best_effort_finalize_cancelled_batch(
    session_factory: Callable[[], AsyncSession],
    *,
    orchestration_id: uuid.UUID,
    batch_index: int,
    worker_token: uuid.UUID,
    inference_run_id: uuid.UUID | None,
    max_batch_attempts: int,
) -> None:
    try:
        async def _cleanup() -> None:
            async with session_factory() as session:
                await reconcile_fact_extraction_batch_after_interruption(
                    session,
                    orchestration_id=orchestration_id,
                    batch_index=batch_index,
                    worker_token=worker_token,
                    failure_code="fact_extraction_execution_cancelled",
                    max_batch_attempts=max_batch_attempts,
                )

        await asyncio.shield(_cleanup())
    except BaseException:
        logger.warning(
            "Failed to finalize cancelled fact extraction batch cleanup for orchestration %s batch %s",
            orchestration_id,
            batch_index,
        )


async def reconcile_fact_extraction_batch_after_interruption(
    session: AsyncSession,
    *,
    orchestration_id: uuid.UUID,
    batch_index: int,
    worker_token: uuid.UUID,
    failure_code: str,
    max_batch_attempts: int,
) -> BatchInterruptionReconciliation:
    try:
        _orchestration, batch, run, batch_application, run_application, context = await _lock_batch_attempt_reconciliation_state(
            session,
            orchestration_id=orchestration_id,
            batch_index=batch_index,
        )
        if context.orchestration_status in {
            _ORCH_STATUS_COMPLETED,
            _ORCH_STATUS_PARTIAL,
            _ORCH_STATUS_FAILED,
        }:
            await session.commit()
            return _build_reconciliation_result(
                reconciliation_status=_determine_reconciliation_status(context),
                batch=batch,
            )

        reconciliation_status = _determine_reconciliation_status(context)
        if context.batch_status == _BATCH_STATUS_COMPLETED or reconciliation_status == BatchAttemptReconciliationStatus.COMPLETED_APPLICATION:
            application_id, _ = _resolve_reconciliation_application_identity(context)
            if context.inference_run_id is None or application_id is None:
                raise FactExtractionOrchestrationStateError("completed application reconciliation requires run and application")
            result = await finalize_batch_from_completed_application(
                session,
                orchestration_id=orchestration_id,
                batch_index=batch_index,
                worker_token=worker_token if context.batch_status != _BATCH_STATUS_COMPLETED else None,
                inference_run_id=context.inference_run_id,
                application_id=application_id,
            )
            return BatchInterruptionReconciliation(
                reconciliation_status=BatchAttemptReconciliationStatus.COMPLETED_APPLICATION,
                batch_status=result.status.value,
                attempt_count=result.attempt_count,
                input_batch_id=result.input_batch_id,
                inference_run_id=result.inference_run_id,
                application_id=result.application_id,
                failure_code=result.failure_code,
            )

        if context.batch_status == _BATCH_STATUS_RUNNING and batch.lease_token not in {None, worker_token}:
            await session.commit()
            return _build_reconciliation_result(reconciliation_status=reconciliation_status, batch=batch)

        if reconciliation_status == BatchAttemptReconciliationStatus.COMPLETED_RUN:
            if failure_code == "persistence_context_invalid":
                batch.status = _BATCH_STATUS_FAILED
                batch.failure_code = failure_code
                if batch.completed_at is None:
                    batch.completed_at = utc_now()
                batch.lease_token = None
                batch.lease_expires_at = None
                await session.flush()
                await session.commit()
                return _build_reconciliation_result(
                    reconciliation_status=BatchAttemptReconciliationStatus.COMPLETED_RUN,
                    batch=batch,
                )
            if batch.status == _BATCH_STATUS_RUNNING and batch.lease_token == worker_token:
                batch.lease_expires_at = utc_now() - timedelta(seconds=1)
                await session.flush()
            await session.commit()
            return _build_reconciliation_result(
                reconciliation_status=BatchAttemptReconciliationStatus.COMPLETED_RUN,
                batch=batch,
            )

        if reconciliation_status == BatchAttemptReconciliationStatus.FAILED_RUN:
            if context.inference_run_id is None:
                raise FactExtractionOrchestrationStateError("failed-run reconciliation requires inference_run_id")
            transition = await transition_batch_after_failed_inference_attempt(
                session,
                orchestration_id=orchestration_id,
                batch_index=batch_index,
                worker_token=worker_token,
                inference_run_id=context.inference_run_id,
                failure_code=failure_code,
                max_batch_attempts=max_batch_attempts,
            )
            return BatchInterruptionReconciliation(
                reconciliation_status=BatchAttemptReconciliationStatus.FAILED_RUN,
                batch_status=transition.status,
                attempt_count=transition.attempt_count,
                input_batch_id=transition.current_input_batch_id,
                inference_run_id=transition.current_inference_run_id,
                application_id=None,
                failure_code=transition.failure_code,
            )

        if reconciliation_status == BatchAttemptReconciliationStatus.ACTIVE_RUN:
            if batch.status == _BATCH_STATUS_RUNNING and batch.lease_token == worker_token:
                if context.inference_run_status == InferenceRunStatus.PENDING.value:
                    batch.lease_expires_at = utc_now() - timedelta(seconds=1)
                    await session.flush()
            await session.commit()
            return _build_reconciliation_result(
                reconciliation_status=BatchAttemptReconciliationStatus.ACTIVE_RUN,
                batch=batch,
            )

        retryable = _is_retryable_failure_code(failure_code) and batch.attempt_count < max_batch_attempts
        if retryable:
            batch.status = _BATCH_STATUS_PENDING
            batch.application_id = None
            batch.lease_token = None
            batch.lease_expires_at = None
            batch.started_at = None
            batch.completed_at = None
            batch.failure_code = None
            batch.proposal_count = 0
            batch.created_count = 0
            batch.reused_count = 0
            batch.withheld_count = 0
        else:
            batch.status = _BATCH_STATUS_FAILED
            batch.failure_code = failure_code
            if batch.completed_at is None:
                batch.completed_at = utc_now()
            batch.lease_token = None
            batch.lease_expires_at = None
        await session.flush()
        await session.commit()
        return _build_reconciliation_result(reconciliation_status=None, batch=batch)
    except BaseException:
        await session.rollback()
        raise


async def _finalize_batch_success(
    session: AsyncSession,
    *,
    orchestration_id: uuid.UUID,
    worker_token: uuid.UUID,
    execution_result: FactExtractionBatchExecutionResult,
    persistence_result: FactExtractionBatchPersistenceResult,
) -> None:
    try:
        orchestration = await orchestration_repository.get_orchestration_for_update(
            session,
            orchestration_id=orchestration_id,
        )
        if orchestration is None:
            raise FactExtractionOrchestrationStateError("orchestration not found")
        batch = await orchestration_repository.get_batch_for_update(
            session,
            orchestration_id=orchestration_id,
            batch_index=execution_result.batch_index,
        )
        if batch is None:
            raise FactExtractionOrchestrationStateError("orchestration batch not found")
        if batch.batch_plan_hash != execution_result.batch_plan_hash:
            raise FactExtractionOrchestrationStateError("execution result batch hash mismatch")
        run = await inference_repository.get_run_for_update(session, execution_result.inference_run_id)
        if run is None:
            raise FactExtractionOrchestrationStateError("inference run not found")
        application = await orchestration_repository.get_application_for_update(
            session,
            application_id=persistence_result.application_id,
        )
        if application is None:
            raise FactExtractionOrchestrationStateError("batch application not found")
        ledger_result = validate_fact_extraction_application_result_envelope(application=application)
        normalized_persistence_result = _normalize_persistence_result_for_ledger_compare(persistence_result)
        if normalized_persistence_result.model_dump(mode="json") != ledger_result.model_dump(mode="json"):
            raise FactExtractionOrchestrationStateError("persistence result does not match the application ledger")
        if ledger_result.project_id != orchestration.project_id:
            raise FactExtractionOrchestrationStateError("ledger project mismatch")
        if ledger_result.extraction_run_id != orchestration.extraction_run_id:
            raise FactExtractionOrchestrationStateError("ledger extraction run mismatch")
        if execution_result.inference_run_id != ledger_result.inference_run_id:
            raise FactExtractionOrchestrationStateError("execution result inference run mismatch")
        if execution_result.input_batch_id != ledger_result.input_batch_id:
            raise FactExtractionOrchestrationStateError("execution result input batch mismatch")
        if execution_result.project_id != ledger_result.project_id:
            raise FactExtractionOrchestrationStateError("execution result project mismatch")
        if execution_result.extraction_run_id != ledger_result.extraction_run_id:
            raise FactExtractionOrchestrationStateError("execution result extraction run mismatch")
        if execution_result.project_id != orchestration.project_id:
            raise FactExtractionOrchestrationStateError("execution result project does not match orchestration")
        if execution_result.extraction_run_id != orchestration.extraction_run_id:
            raise FactExtractionOrchestrationStateError("execution result extraction run does not match orchestration")
        if batch.current_input_batch_id != execution_result.input_batch_id:
            raise FactExtractionOrchestrationStateError("batch current_input_batch_id mismatch")
        if batch.current_inference_run_id != execution_result.inference_run_id:
            raise FactExtractionOrchestrationStateError("batch current_inference_run_id mismatch")
        _ledger_result, _batch_result = _finalize_batch_from_locked_completed_application(
            orchestration=orchestration,
            batch=batch,
            run=run,
            application=application,
            worker_token=worker_token,
        )
        await session.flush()
        await session.commit()
    except BaseException:
        await session.rollback()
        raise


async def _finalize_batch_failure(
    session: AsyncSession,
    *,
    orchestration_id: uuid.UUID,
    batch_index: int,
    worker_token: uuid.UUID,
    failure_code: str,
    max_batch_attempts: int,
) -> None:
    try:
        await reconcile_fact_extraction_batch_after_interruption(
            session,
            orchestration_id=orchestration_id,
            batch_index=batch_index,
            worker_token=worker_token,
            failure_code=failure_code,
            max_batch_attempts=max_batch_attempts,
        )
    except BaseException:
        await session.rollback()
        raise


def _build_batch_result(batch: FactExtractionOrchestrationBatch) -> FactExtractionOrchestrationBatchResult:
    return FactExtractionOrchestrationBatchResult(
        batch_index=batch.batch_index,
        batch_plan_hash=batch.batch_plan_hash,
        status=FactExtractionOrchestrationBatchStatus(batch.status),
        attempt_count=batch.attempt_count,
        input_batch_id=batch.current_input_batch_id,
        inference_run_id=batch.current_inference_run_id,
        application_id=batch.application_id,
        proposal_count=batch.proposal_count,
        created_count=batch.created_count,
        reused_count=batch.reused_count,
        withheld_count=batch.withheld_count,
        failure_code=batch.failure_code,
    )


async def _maybe_ensure_cross_batch_duplicate_grouping(
    session_factory: Callable[[], AsyncSession],
    *,
    extraction_run_id: uuid.UUID,
    orchestration_result: FactExtractionOrchestrationResult,
) -> FactExtractionOrchestrationResult:
    if orchestration_result.status not in {
        FactExtractionOrchestrationStatus.COMPLETED,
        FactExtractionOrchestrationStatus.PARTIAL,
    }:
        return orchestration_result
    try:
        await duplicate_grouping_service.ensure_cross_batch_duplicate_grouping(
            session_factory,
            extraction_run_id=extraction_run_id,
        )
    except BaseException as error:
        logger.warning(
            "Cross-batch duplicate grouping did not complete after orchestration finalization",
            extra={
                "extraction_run_id": str(extraction_run_id),
                "orchestration_id": str(orchestration_result.orchestration_id),
                "orchestration_status": orchestration_result.status.value,
                "error_type": type(error).__name__,
            },
        )
    return orchestration_result


async def _read_completed_orchestration_result(
    session: AsyncSession,
    *,
    orchestration_id: uuid.UUID,
) -> FactExtractionOrchestrationResult:
    try:
        orchestration = await orchestration_repository.get_orchestration_for_update(
            session,
            orchestration_id=orchestration_id,
        )
        if orchestration is None:
            raise FactExtractionOrchestrationStateError("orchestration not found")
        batches = await orchestration_repository.list_batches_for_orchestration_for_update(
            session,
            orchestration_id=orchestration_id,
        )
        if orchestration.status not in {
            _ORCH_STATUS_COMPLETED,
            _ORCH_STATUS_PARTIAL,
            _ORCH_STATUS_FAILED,
        }:
            raise FactExtractionOrchestrationStateError("orchestration is not terminal")
        application_ids = [batch.application_id for batch in batches if batch.application_id is not None]
        applications = await orchestration_repository.list_applications(
            session,
            application_ids=[application_id for application_id in application_ids if application_id is not None],
        )
        applications_by_id = {application.id: application for application in applications}
        authenticated_applications = _load_authenticated_completed_applications(
            orchestration=orchestration,
            batches=batches,
            applications_by_id=applications_by_id,
        )
        validate_terminal_orchestration_state(
            orchestration=orchestration,
            batches=batches,
            authenticated_applications=authenticated_applications,
        )
        await session.commit()
        return FactExtractionOrchestrationResult(
            orchestration_id=orchestration.id,
            attempt_no=orchestration.attempt_no,
            request_hash=orchestration.request_hash,
            plan_hash=orchestration.plan_hash,
            status=FactExtractionOrchestrationStatus(orchestration.status),
            batch_count=orchestration.batch_count,
            completed_batch_count=orchestration.completed_batch_count,
            failed_batch_count=orchestration.failed_batch_count,
            proposal_count=orchestration.proposal_count,
            created_count=orchestration.created_count,
            reused_count=orchestration.reused_count,
            withheld_count=orchestration.withheld_count,
            batches=tuple(_build_batch_result(batch) for batch in batches),
        )
    except BaseException:
        await session.rollback()
        raise


async def _finalize_orchestration(
    session: AsyncSession,
    *,
    orchestration_id: uuid.UUID,
) -> FactExtractionOrchestrationResult:
    try:
        orchestration = await orchestration_repository.get_orchestration_for_update(
            session,
            orchestration_id=orchestration_id,
        )
        if orchestration is None:
            raise FactExtractionOrchestrationStateError("orchestration not found")
        batches = await orchestration_repository.list_batches_for_orchestration_for_update(
            session,
            orchestration_id=orchestration_id,
        )
        completed_batches = [batch for batch in batches if batch.status == _BATCH_STATUS_COMPLETED]
        failed_batches = [batch for batch in batches if batch.status == _BATCH_STATUS_FAILED]
        running_or_pending = [
            batch for batch in batches if batch.status in {_BATCH_STATUS_PENDING, _BATCH_STATUS_RUNNING}
        ]
        applications = await orchestration_repository.list_applications(
            session,
            application_ids=[
                application_id
                for application_id in [batch.application_id for batch in completed_batches]
                if application_id is not None
            ],
        )
        applications_by_id = {application.id: application for application in applications}
        authenticated_applications = _load_authenticated_completed_applications(
            orchestration=orchestration,
            batches=batches,
            applications_by_id=applications_by_id,
        )
        if orchestration.status in {_ORCH_STATUS_COMPLETED, _ORCH_STATUS_PARTIAL, _ORCH_STATUS_FAILED}:
            validate_terminal_orchestration_state(
                orchestration=orchestration,
                batches=batches,
                authenticated_applications=authenticated_applications,
            )
            await session.commit()
            return FactExtractionOrchestrationResult(
                orchestration_id=orchestration.id,
                attempt_no=orchestration.attempt_no,
                request_hash=orchestration.request_hash,
                plan_hash=orchestration.plan_hash,
                status=FactExtractionOrchestrationStatus(orchestration.status),
                batch_count=orchestration.batch_count,
                completed_batch_count=orchestration.completed_batch_count,
                failed_batch_count=orchestration.failed_batch_count,
                proposal_count=orchestration.proposal_count,
                created_count=orchestration.created_count,
                reused_count=orchestration.reused_count,
                withheld_count=orchestration.withheld_count,
                batches=tuple(_build_batch_result(batch) for batch in batches),
            )
        proposal_count = sum(item.result.proposal_count for item in authenticated_applications.values())
        created_count = sum(item.result.created_count for item in authenticated_applications.values())
        reused_count = sum(item.result.reused_count for item in authenticated_applications.values())
        withheld_count = sum(item.result.withheld_count for item in authenticated_applications.values())

        target_completed_batch_count = len(completed_batches)
        target_failed_batch_count = len(failed_batches)
        if len(completed_batches) == len(batches):
            target_status = _ORCH_STATUS_COMPLETED
            target_failure_code = None
        elif completed_batches and len(completed_batches) + len(failed_batches) == len(batches):
            target_status = _ORCH_STATUS_PARTIAL
            target_failure_code = None
        elif not completed_batches and failed_batches and not running_or_pending:
            target_status = _ORCH_STATUS_FAILED
            target_failure_code = failed_batches[0].failure_code
        else:
            target_status = _ORCH_STATUS_RUNNING
            target_failure_code = None

        now = utc_now()
        orchestration.completed_batch_count = target_completed_batch_count
        orchestration.failed_batch_count = target_failed_batch_count
        orchestration.proposal_count = proposal_count
        orchestration.created_count = created_count
        orchestration.reused_count = reused_count
        orchestration.withheld_count = withheld_count
        if orchestration.started_at is None and any(batch.attempt_count > 0 for batch in batches):
            orchestration.started_at = now
        orchestration.status = target_status
        orchestration.failure_code = target_failure_code
        if target_status in {_ORCH_STATUS_COMPLETED, _ORCH_STATUS_PARTIAL, _ORCH_STATUS_FAILED}:
            if orchestration.completed_at is None:
                orchestration.completed_at = now
            validate_terminal_orchestration_state(
                orchestration=orchestration,
                batches=batches,
                authenticated_applications=authenticated_applications,
            )
        else:
            orchestration.completed_at = None

        await session.flush()
        await session.commit()
        return FactExtractionOrchestrationResult(
            orchestration_id=orchestration.id,
            attempt_no=orchestration.attempt_no,
            request_hash=orchestration.request_hash,
            plan_hash=orchestration.plan_hash,
            status=FactExtractionOrchestrationStatus(orchestration.status),
            batch_count=orchestration.batch_count,
            completed_batch_count=orchestration.completed_batch_count,
            failed_batch_count=orchestration.failed_batch_count,
            proposal_count=orchestration.proposal_count,
            created_count=orchestration.created_count,
            reused_count=orchestration.reused_count,
            withheld_count=orchestration.withheld_count,
            batches=tuple(_build_batch_result(batch) for batch in batches),
        )
    except BaseException:
        await session.rollback()
        raise


async def execute_fact_extraction_orchestration(
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
) -> FactExtractionOrchestrationResult:
    project_id = _require_uuid(project_id, field_name="project_id")
    extraction_run_id = _require_uuid(extraction_run_id, field_name="extraction_run_id")
    worker_token = _require_uuid(worker_token, field_name="worker_token")
    _require_positive_int(max_batch_attempts, field_name="max_batch_attempts")
    _require_positive_int(batch_lease_seconds, field_name="batch_lease_seconds")
    _require_positive_int(stale_inference_seconds, field_name="stale_inference_seconds")
    provider_value = _require_text(provider, field_name="provider", max_length=64)
    requested_model_value = _require_text(requested_model, field_name="requested_model", max_length=128)
    _validate_orchestration_plan(
        extraction_run_id=extraction_run_id,
        plan=plan,
        prompt=prompt,
    )

    async with session_factory() as session:
        prepared = await prepare_fact_extraction_orchestration(
            session,
            project_id=project_id,
            extraction_run_id=extraction_run_id,
            plan=plan,
            prompt=prompt,
            provider=provider_value,
            requested_model=requested_model_value,
            max_batch_attempts=max_batch_attempts,
            batch_lease_seconds=batch_lease_seconds,
            stale_inference_seconds=stale_inference_seconds,
        )
    if prepared.reused_completed:
        async with session_factory() as session:
            result = await _read_completed_orchestration_result(
                session,
                orchestration_id=prepared.orchestration_id,
            )
        return await _maybe_ensure_cross_batch_duplicate_grouping(
            session_factory,
            extraction_run_id=extraction_run_id,
            orchestration_result=result,
        )

    async with session_factory() as session:
        try:
            batches = await orchestration_repository.list_batches_for_orchestration(
                session,
                orchestration_id=prepared.orchestration_id,
            )
            await session.commit()
        except BaseException:
            await session.rollback()
            raise

    for batch in batches:
        while True:
            stale_before = utc_now() - timedelta(seconds=stale_inference_seconds)
            async with session_factory() as session:
                claim = await claim_fact_extraction_orchestration_batch(
                    session,
                    orchestration_id=prepared.orchestration_id,
                    batch_index=batch.batch_index,
                    worker_token=worker_token,
                    lease_seconds=batch_lease_seconds,
                    max_batch_attempts=max_batch_attempts,
                    stale_before=stale_before,
                )

            if claim.status in {_BATCH_STATUS_COMPLETED, _BATCH_STATUS_FAILED}:
                break
            if not claim.claimed:
                break

            if claim.current_inference_run_id is not None:
                async with session_factory() as session:
                    recovery = await recover_stale_fact_extraction_inference_run(
                        session,
                        inference_run_id=claim.current_inference_run_id,
                        stale_before=stale_before,
                    )
                if recovery.status == StaleInferenceRecoveryStatus.RUNNING and not recovery.recovered_stale_run:
                    break
                if recovery.status == StaleInferenceRecoveryStatus.FAILED:
                    async with session_factory() as session:
                        transition = await transition_batch_after_failed_inference_attempt(
                            session,
                            orchestration_id=prepared.orchestration_id,
                            batch_index=batch.batch_index,
                            worker_token=worker_token,
                            inference_run_id=claim.current_inference_run_id,
                            failure_code=recovery.failure_code or "fact_extraction_execution_stale",
                            max_batch_attempts=max_batch_attempts,
                        )
                    if transition.status == _BATCH_STATUS_PENDING:
                        continue
                    break

            async def prepared_run_observer(notice: PreparedFactExtractionRunNotice) -> None:
                async with session_factory() as session:
                    await _record_prepared_run_notice(
                        session,
                        orchestration_id=prepared.orchestration_id,
                        worker_token=worker_token,
                        notice=notice,
                    )

            try:
                async def _attempt_operation():
                    execution_result = await execute_fact_extraction_batch(
                        session_factory,
                        project_id=project_id,
                        extraction_run_id=extraction_run_id,
                        plan=plan,
                        batch_index=batch.batch_index,
                        prompt=prompt,
                        llm_client=llm_client,
                        provider=provider_value,
                        requested_model=requested_model_value,
                        prepared_run_observer=prepared_run_observer,
                    )
                    async with session_factory() as session:
                        persistence_result = await persist_completed_fact_extraction_batch(
                            session,
                            project_id=project_id,
                            extraction_run_id=extraction_run_id,
                            inference_run_id=execution_result.inference_run_id,
                        )
                    return execution_result, persistence_result

                execution_result, persistence_result = await run_with_batch_lease_heartbeat(
                    session_factory,
                    orchestration_id=prepared.orchestration_id,
                    batch_index=batch.batch_index,
                    worker_token=worker_token,
                    batch_lease_seconds=batch_lease_seconds,
                    operation=_attempt_operation,
                )
                async with session_factory() as session:
                    await renew_fact_extraction_orchestration_batch_lease(
                        session,
                        orchestration_id=prepared.orchestration_id,
                        batch_index=batch.batch_index,
                        worker_token=worker_token,
                        lease_seconds=batch_lease_seconds,
                    )
                async with session_factory() as session:
                    await _finalize_batch_success(
                        session,
                        orchestration_id=prepared.orchestration_id,
                        worker_token=worker_token,
                        execution_result=execution_result,
                        persistence_result=persistence_result,
                    )
                break
            except asyncio.CancelledError:
                await _best_effort_finalize_cancelled_batch(
                    session_factory,
                    orchestration_id=prepared.orchestration_id,
                    batch_index=batch.batch_index,
                    worker_token=worker_token,
                    inference_run_id=None,
                    max_batch_attempts=max_batch_attempts,
                )
                raise
            except BaseException as error:
                failure_code = _classify_batch_failure(error)
                async with session_factory() as session:
                    reconciliation = await reconcile_fact_extraction_batch_after_interruption(
                        session,
                        orchestration_id=prepared.orchestration_id,
                        batch_index=batch.batch_index,
                        worker_token=worker_token,
                        failure_code=failure_code,
                        max_batch_attempts=max_batch_attempts,
                    )
                if reconciliation.batch_status == _BATCH_STATUS_COMPLETED:
                    break
                if reconciliation.batch_status == _BATCH_STATUS_PENDING:
                    continue
                if reconciliation.batch_status in {_BATCH_STATUS_RUNNING, _BATCH_STATUS_FAILED}:
                    break

    async with session_factory() as session:
        result = await _finalize_orchestration(
            session,
            orchestration_id=prepared.orchestration_id,
        )
    return await _maybe_ensure_cross_batch_duplicate_grouping(
        session_factory,
        extraction_run_id=extraction_run_id,
        orchestration_result=result,
    )
