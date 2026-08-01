from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass(frozen=True, slots=True)
class ConsistencyReviewCandidateMemberRecord:
    consistency_application_id: uuid.UUID
    candidate_id: uuid.UUID
    fact_value_id: uuid.UUID
    source_batch_id: uuid.UUID
    semantic_key_hash: str


@dataclass(frozen=True, slots=True)
class ConsistencyReviewDecisionLedgerRecord:
    id: uuid.UUID
    project_id: uuid.UUID
    consistency_check_application_id: uuid.UUID
    assessment_id: uuid.UUID
    source_consistency_application_id: uuid.UUID
    source_consistency_candidate_id: uuid.UUID
    actor_id: uuid.UUID
    decision_no: int
    supersedes_decision_id: uuid.UUID | None
    decision_kind: str
    selected_value_count: int
    comment: str | None
    decision_manifest_hash: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ConsistencyReviewDecisionSelectionLedgerRecord:
    id: uuid.UUID
    decision_id: uuid.UUID
    assessment_id: uuid.UUID
    source_consistency_application_id: uuid.UUID
    source_consistency_candidate_id: uuid.UUID
    fact_value_id: uuid.UUID
    selection_order: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class AppendConsistencyReviewDecisionResult:
    decision_id: uuid.UUID
    decision_no: int
    supersedes_decision_id: uuid.UUID | None
    decision_manifest_hash: str
    selected_fact_value_ids: tuple[uuid.UUID, ...]
    created_new: bool
