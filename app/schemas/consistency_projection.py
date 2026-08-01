from __future__ import annotations

from dataclasses import dataclass
import uuid
from typing import Any, Literal


ConsistencyReviewStatus = Literal["pending_review", "not_required"]


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
    evidences: tuple[ConsistencyReviewProjectionEvidence, ...]


@dataclass(frozen=True, slots=True)
class ConsistencyReviewProjectionItem:
    candidate_id: uuid.UUID
    batch_index: int
    verdict: str
    severity: str
    confidence: float
    explanation: str
    impact: tuple[str, ...]
    recommended_actions: tuple[str, ...]
    review_status: ConsistencyReviewStatus
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
    items: tuple[ConsistencyReviewProjectionItem, ...]
