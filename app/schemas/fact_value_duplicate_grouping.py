from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any
import uuid


CROSS_BATCH_DUPLICATE_ALGORITHM_VERSION = "cross_batch_exact_v2"
CROSS_BATCH_MULTI_VALUE_CANDIDATE_ALGORITHM_VERSION = "cross_batch_multi_value_v1"


class DuplicateGroupingMemberSourceKind(StrEnum):
    AI = "ai"


@dataclass(frozen=True, slots=True)
class DuplicateCandidate:
    fact_value_id: uuid.UUID
    fact_id: uuid.UUID
    orchestration_id: uuid.UUID
    extraction_run_id: uuid.UUID
    source_batch_id: uuid.UUID
    value_type: str
    value_json: Any | None
    referenced_entity_id: uuid.UUID | None
    evidence_link_ids: tuple[uuid.UUID, ...]
    evidence_ids: tuple[uuid.UUID, ...] = ()


@dataclass(frozen=True, slots=True)
class DuplicateFingerprint:
    canonical_bytes: bytes
    sha256_hex: str


@dataclass(frozen=True, slots=True)
class DuplicateGroupMemberPlan:
    fact_value_id: uuid.UUID
    source_batch_id: uuid.UUID


@dataclass(frozen=True, slots=True)
class DuplicateGroupPlan:
    duplicate_key_hash: str
    member_count: int
    distinct_batch_count: int
    members: tuple[DuplicateGroupMemberPlan, ...]


@dataclass(frozen=True, slots=True)
class DuplicateGroupingWritePlan:
    algorithm_version: str
    input_manifest_hash: str
    result_manifest_hash: str
    input_fact_value_count: int
    duplicate_group_count: int
    duplicate_member_count: int
    groups: tuple[DuplicateGroupPlan, ...]


@dataclass(frozen=True, slots=True)
class DuplicateGroupingResult:
    grouping_application_id: uuid.UUID
    algorithm_version: str
    input_fact_value_count: int
    duplicate_group_count: int
    duplicate_member_count: int
    created_new: bool


@dataclass(frozen=True, slots=True)
class DuplicateGroupingApplicationLedger:
    id: uuid.UUID
    orchestration_id: uuid.UUID
    extraction_run_id: uuid.UUID
    algorithm_version: str
    input_manifest_hash: str
    result_manifest_hash: str
    input_fact_value_count: int
    duplicate_group_count: int
    duplicate_member_count: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class DuplicateGroupLedger:
    id: uuid.UUID
    grouping_application_id: uuid.UUID
    duplicate_key_hash: str
    member_count: int
    distinct_batch_count: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class DuplicateGroupMemberLedger:
    id: uuid.UUID
    orchestration_id: uuid.UUID
    grouping_application_id: uuid.UUID
    group_id: uuid.UUID
    fact_value_id: uuid.UUID
    source_batch_id: uuid.UUID
    created_at: datetime


@dataclass(frozen=True, slots=True)
class DuplicateGroupEvidenceProjection:
    group_id: uuid.UUID
    duplicate_key_hash: str
    fact_value_id: uuid.UUID
    source_batch_id: uuid.UUID
    evidence_link_ids: tuple[uuid.UUID, ...]
    evidence_ids: tuple[uuid.UUID, ...]


@dataclass(frozen=True, slots=True)
class FactValueConsistencyCandidateMemberPlan:
    fact_value_id: uuid.UUID
    source_batch_id: uuid.UUID
    semantic_key_hash: str


@dataclass(frozen=True, slots=True)
class FactValueConsistencyCandidatePlan:
    fact_id: uuid.UUID
    candidate_kind: str
    member_count: int
    distinct_semantic_key_count: int
    distinct_batch_count: int
    members: tuple[FactValueConsistencyCandidateMemberPlan, ...]


@dataclass(frozen=True, slots=True)
class FactValueConsistencyCandidateWritePlan:
    algorithm_version: str
    source_duplicate_grouping_algorithm_version: str
    input_manifest_hash: str
    result_manifest_hash: str
    candidate_count: int
    member_count: int
    candidates: tuple[FactValueConsistencyCandidatePlan, ...]


@dataclass(frozen=True, slots=True)
class FactValueConsistencyCandidateResult:
    consistency_application_id: uuid.UUID
    duplicate_grouping_application_id: uuid.UUID
    algorithm_version: str
    candidate_count: int
    member_count: int
    created_new: bool


@dataclass(frozen=True, slots=True)
class FactValueConsistencyCandidateApplicationLedger:
    id: uuid.UUID
    duplicate_grouping_application_id: uuid.UUID
    orchestration_id: uuid.UUID
    extraction_run_id: uuid.UUID
    algorithm_version: str
    input_manifest_hash: str
    result_manifest_hash: str
    candidate_count: int
    member_count: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class FactValueConsistencyCandidateLedger:
    id: uuid.UUID
    consistency_application_id: uuid.UUID
    fact_id: uuid.UUID
    candidate_kind: str
    member_count: int
    distinct_semantic_key_count: int
    distinct_batch_count: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class FactValueConsistencyCandidateMemberLedger:
    id: uuid.UUID
    consistency_application_id: uuid.UUID
    candidate_id: uuid.UUID
    orchestration_id: uuid.UUID
    fact_value_id: uuid.UUID
    source_batch_id: uuid.UUID
    semantic_key_hash: str
    created_at: datetime
