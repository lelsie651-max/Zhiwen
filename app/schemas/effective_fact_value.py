from __future__ import annotations

from dataclasses import dataclass
import uuid
from typing import Literal

from app.schemas.consistency_projection import ConsistencyReviewProjectionMember


EffectiveFactValueResolutionStatus = Literal[
    "resolved",
    "pending_review",
    "deferred",
    "unreviewed_compatible",
]

EffectiveFactValueResolutionBasis = Literal[
    "human_selection",
    "human_confirmed_compatibility",
    "none",
]


@dataclass(frozen=True, slots=True)
class EffectiveFactValueProjectionItem:
    fact_id: uuid.UUID
    candidate_id: uuid.UUID
    assessment_id: uuid.UUID
    agent_verdict: str
    review_status: str
    resolution_status: EffectiveFactValueResolutionStatus
    resolution_basis: EffectiveFactValueResolutionBasis
    current_decision_id: uuid.UUID | None
    current_decision_kind: str | None
    effective_fact_value_ids: tuple[uuid.UUID, ...]
    candidate_members: tuple[ConsistencyReviewProjectionMember, ...]


@dataclass(frozen=True, slots=True)
class EffectiveFactValueProjection:
    project_id: uuid.UUID
    consistency_check_application_id: uuid.UUID
    source_consistency_application_id: uuid.UUID
    result_manifest_hash: str
    fact_count: int
    resolved_count: int
    pending_count: int
    deferred_count: int
    items: tuple[EffectiveFactValueProjectionItem, ...]
