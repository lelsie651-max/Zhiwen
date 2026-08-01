from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass(frozen=True, slots=True)
class ConsistencyAssessmentCitationSpec:
    assessment_source_consistency_application_id: uuid.UUID
    assessment_source_consistency_candidate_id: uuid.UUID
    source_fact_value_id: uuid.UUID
    evidence_link_id: uuid.UUID
    citation_order: int


@dataclass(frozen=True, slots=True)
class ConsistencyAssessmentSpec:
    source_consistency_application_id: uuid.UUID
    source_consistency_candidate_id: uuid.UUID
    batch_index: int
    verdict: str
    severity: str
    confidence: float
    explanation: str
    impact_json: tuple[str, ...]
    recommended_actions_json: tuple[str, ...]
    assessment_manifest_hash: str
    citations: tuple[ConsistencyAssessmentCitationSpec, ...]


@dataclass(frozen=True, slots=True)
class ConsistencyCheckBatchSpec:
    batch_index: int
    batch_manifest_hash: str
    skipped_empty: bool
    input_batch_id: uuid.UUID | None
    inference_run_id: uuid.UUID | None
    request_hash: str | None
    message_content_hash: str | None


@dataclass(frozen=True, slots=True)
class ConsistencyCheckPersistencePlan:
    project_id: uuid.UUID
    consistency_application_id: uuid.UUID
    orchestration_id: uuid.UUID
    source_result_manifest_hash: str
    plan_manifest_hash: str
    execution_identity_hash: str
    result_manifest_hash: str
    prompt_contract_hash: str
    provider: str
    requested_model: str
    executor_name: str
    executor_version: str
    batch_count: int
    executed_batch_count: int
    skipped_empty_batch_count: int
    inference_run_count: int
    assessment_count: int
    batches: tuple[ConsistencyCheckBatchSpec, ...]
    assessments: tuple[ConsistencyAssessmentSpec, ...]


@dataclass(frozen=True, slots=True)
class ConsistencyCheckApplicationLedgerRecord:
    id: uuid.UUID
    project_id: uuid.UUID
    consistency_application_id: uuid.UUID
    orchestration_id: uuid.UUID
    source_result_manifest_hash: str
    plan_manifest_hash: str
    execution_identity_hash: str
    result_manifest_hash: str
    prompt_contract_hash: str
    provider: str
    requested_model: str
    executor_name: str
    executor_version: str
    batch_count: int
    executed_batch_count: int
    skipped_empty_batch_count: int
    inference_run_count: int
    assessment_count: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ConsistencyCheckBatchLedgerRecord:
    id: uuid.UUID
    consistency_check_application_id: uuid.UUID
    batch_index: int
    batch_manifest_hash: str
    skipped_empty: bool
    input_batch_id: uuid.UUID | None
    inference_run_id: uuid.UUID | None
    request_hash: str | None
    message_content_hash: str | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ConsistencyAssessmentLedgerRecord:
    id: uuid.UUID
    consistency_check_application_id: uuid.UUID
    source_consistency_application_id: uuid.UUID
    source_consistency_candidate_id: uuid.UUID
    batch_index: int
    verdict: str
    severity: str
    confidence: float
    explanation: str
    impact_json: tuple[str, ...]
    recommended_actions_json: tuple[str, ...]
    assessment_manifest_hash: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ConsistencyAssessmentCitationLedgerRecord:
    id: uuid.UUID
    assessment_id: uuid.UUID
    source_consistency_application_id: uuid.UUID
    source_consistency_candidate_id: uuid.UUID
    source_fact_value_id: uuid.UUID
    evidence_link_id: uuid.UUID
    citation_order: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ConsistencyCheckPersistenceResult:
    consistency_check_application_id: uuid.UUID
    created_new: bool
    batch_count: int
    assessment_count: int
