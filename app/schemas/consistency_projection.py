from __future__ import annotations

from dataclasses import dataclass
import uuid
from datetime import datetime
from typing import Any, Literal


ConsistencyReviewStatus = Literal[
    "pending_review",
    "not_required",
    "reviewed",
    "deferred",
]


@dataclass(frozen=True, slots=True)
class ConsistencyReviewProjectionDecision:
    decision_id: uuid.UUID
    decision_no: int
    supersedes_decision_id: uuid.UUID | None
    actor_id: uuid.UUID
    decision_kind: str
    selected_fact_value_ids: tuple[uuid.UUID, ...]
    comment: str | None
    decision_manifest_hash: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ConsistencyReviewProjectionEvidence:
    evidence_link_id: uuid.UUID
    evidence_id: uuid.UUID
    document_revision_id: uuid.UUID
    document_block_id: uuid.UUID
    location_key: str
    page_no: int | None
    start_line: int | None
    end_line: int | None
    start_offset: int
    end_offset: int
    excerpt: str
    excerpt_hash: str
    content_hash: str
    cited_by_assessment: bool


@dataclass(frozen=True, slots=True)
class ConsistencyReviewProjectionMember:
    fact_value_id: uuid.UUID
    value_type: str
    value_json: Any | None
    normalized_value_text: str | None
    referenced_entity_id: uuid.UUID | None
    selected_by_current_decision: bool
    current_selection_order: int | None
    evidences: tuple[ConsistencyReviewProjectionEvidence, ...]


@dataclass(frozen=True, slots=True)
class ConsistencyReviewProjectionItem:
    assessment_id: uuid.UUID
    fact_id: uuid.UUID
    candidate_id: uuid.UUID
    batch_index: int
    verdict: str
    severity: str
    confidence: float
    explanation: str
    impact: tuple[str, ...]
    recommended_actions: tuple[str, ...]
    review_status: ConsistencyReviewStatus
    current_decision: ConsistencyReviewProjectionDecision | None
    decision_history: tuple[ConsistencyReviewProjectionDecision, ...]
    selected_fact_value_ids: tuple[uuid.UUID, ...]
    members: tuple[ConsistencyReviewProjectionMember, ...]


@dataclass(frozen=True, slots=True)
class ConsistencyReviewProjection:
    project_id: uuid.UUID
    consistency_check_application_id: uuid.UUID
    source_consistency_application_id: uuid.UUID
    plan_manifest_hash: str
    result_manifest_hash: str
    assessment_count: int
    conflict_count: int
    compatible_count: int
    insufficient_evidence_count: int
    red_count: int
    yellow_count: int
    pending_review_count: int
    reviewed_count: int
    deferred_count: int
    not_required_count: int
    decision_count: int
    items: tuple[ConsistencyReviewProjectionItem, ...]
