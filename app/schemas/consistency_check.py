from __future__ import annotations

from dataclasses import dataclass
import uuid
from typing import Any


CONSISTENCY_CHECK_PLANNER_NAME = "deterministic_consistency_check_planner"
CONSISTENCY_CHECK_PLANNER_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class ConsistencyCheckPlannerConfig:
    max_candidates_per_batch: int
    max_evidence_characters_per_batch: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_candidates_per_batch, int)
            or isinstance(self.max_candidates_per_batch, bool)
            or self.max_candidates_per_batch <= 0
        ):
            raise ValueError("max_candidates_per_batch must be a positive integer")
        if (
            not isinstance(self.max_evidence_characters_per_batch, int)
            or isinstance(self.max_evidence_characters_per_batch, bool)
            or self.max_evidence_characters_per_batch <= 0
        ):
            raise ValueError("max_evidence_characters_per_batch must be a positive integer")


@dataclass(frozen=True, slots=True)
class ConsistencyCheckEvidenceBundle:
    evidence_link_id: uuid.UUID
    evidence_id: uuid.UUID
    role: str
    is_primary: bool
    source_order: int
    document_block_id: uuid.UUID
    location_key: str
    page_no: int | None
    start_line: int | None
    end_line: int | None
    start_offset: int
    end_offset: int
    excerpt: str
    evidence_content_hash: str


@dataclass(frozen=True, slots=True)
class ConsistencyCheckMemberBundle:
    fact_value_id: uuid.UUID
    source_batch_id: uuid.UUID
    semantic_key_hash: str
    value_type: str
    value_json: Any | None
    referenced_entity_id: uuid.UUID | None
    evidences: tuple[ConsistencyCheckEvidenceBundle, ...]


@dataclass(frozen=True, slots=True)
class ConsistencyCheckCandidateBundle:
    candidate_id: uuid.UUID
    fact_id: uuid.UUID
    candidate_kind: str
    members: tuple[ConsistencyCheckMemberBundle, ...]


@dataclass(frozen=True, slots=True)
class ConsistencyCheckBatchPlan:
    batch_index: int
    candidate_ids: tuple[uuid.UUID, ...]
    candidate_count: int
    evidence_character_count: int
    batch_manifest_hash: str
    candidates: tuple[ConsistencyCheckCandidateBundle, ...]


@dataclass(frozen=True, slots=True)
class ConsistencyCheckPlan:
    project_id: uuid.UUID
    consistency_application_id: uuid.UUID
    source_result_manifest_hash: str
    planner_name: str
    planner_version: str
    config: ConsistencyCheckPlannerConfig
    batches: tuple[ConsistencyCheckBatchPlan, ...]
    plan_manifest_hash: str
