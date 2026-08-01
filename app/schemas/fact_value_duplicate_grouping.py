from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any
import uuid


CROSS_BATCH_DUPLICATE_ALGORITHM_VERSION = "cross_batch_exact_v1"


class DuplicateGroupingMemberSourceKind(StrEnum):
    AI = "ai"


@dataclass(frozen=True, slots=True)
class DuplicateCandidate:
    fact_value_id: uuid.UUID
    fact_id: uuid.UUID
    extraction_run_id: uuid.UUID
    source_batch_id: uuid.UUID
    value_type: str
    value_json: Any | None
    referenced_entity_id: uuid.UUID | None
    evidence_link_ids: tuple[uuid.UUID, ...]


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
