from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import uuid

from pydantic import BaseModel, ConfigDict


class FactExtractionOrchestrationStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class FactExtractionOrchestrationBatchStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StaleInferenceRecoveryStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class FactExtractionOrchestrationBatchResult(BaseModel):
    batch_index: int
    batch_plan_hash: str
    status: FactExtractionOrchestrationBatchStatus
    attempt_count: int
    input_batch_id: uuid.UUID | None
    inference_run_id: uuid.UUID | None
    application_id: uuid.UUID | None
    proposal_count: int
    created_count: int
    reused_count: int
    withheld_count: int
    failure_code: str | None

    model_config = ConfigDict(frozen=True, extra="forbid")


class FactExtractionOrchestrationResult(BaseModel):
    orchestration_id: uuid.UUID
    attempt_no: int
    request_hash: str
    plan_hash: str
    status: FactExtractionOrchestrationStatus
    batch_count: int
    completed_batch_count: int
    failed_batch_count: int
    proposal_count: int
    created_count: int
    reused_count: int
    withheld_count: int
    batches: tuple[FactExtractionOrchestrationBatchResult, ...]

    model_config = ConfigDict(frozen=True, extra="forbid")


@dataclass(frozen=True, slots=True)
class StaleInferenceRecoveryResult:
    inference_run_id: uuid.UUID
    status: StaleInferenceRecoveryStatus
    recovered_stale_run: bool
    failure_code: str | None
