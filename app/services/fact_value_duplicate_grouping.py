from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
import math
import unicodedata
import uuid
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document_content import ExtractionRunOutcome, ExtractionRunStatus
from app.models.fact_value_duplicate_grouping import (
    FactValueConsistencyCandidate,
    FactValueConsistencyCandidateApplication,
    FactValueConsistencyCandidateMember,
    FactValueDuplicateGroup,
    FactValueDuplicateGroupMember,
    FactValueDuplicateGroupingApplication,
    normalize_duplicate_grouping_algorithm_version as normalize_model_algorithm_version,
)
from app.repositories import fact_value_duplicate_grouping as duplicate_grouping_repository
from app.services import fact_extraction_persistence as fact_extraction_persistence_service
from app.utils.validation import normalize_text
from app.schemas.fact_value_duplicate_grouping import (
    CROSS_BATCH_DUPLICATE_ALGORITHM_VERSION,
    CROSS_BATCH_MULTI_VALUE_CANDIDATE_ALGORITHM_VERSION,
    DuplicateCandidate,
    DuplicateFingerprint,
    DuplicateGroupEvidenceProjection,
    DuplicateGroupLedger,
    DuplicateGroupMemberLedger,
    DuplicateGroupMemberPlan,
    DuplicateGroupPlan,
    DuplicateGroupingApplicationLedger,
    DuplicateGroupingResult,
    DuplicateGroupingWritePlan,
    FactValueConsistencyCandidateApplicationLedger,
    FactValueConsistencyCandidateLedger,
    FactValueConsistencyCandidateMemberLedger,
    FactValueConsistencyCandidateMemberPlan,
    FactValueConsistencyCandidatePlan,
    FactValueConsistencyCandidateResult,
    FactValueConsistencyCandidateWritePlan,
)


logger = logging.getLogger(__name__)

_APPLICATION_UNIQUE_CONSTRAINT = "uq_dupgrp_app_orch_alg"
_CONSISTENCY_APPLICATION_UNIQUE_CONSTRAINT = "uq_fvcca_dupgrp_alg"
_MISSING = object()
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ORCHESTRATION_IDENTITY_STRING_LIMITS: tuple[tuple[str, int], ...] = (
    ("planner_name", 64),
    ("planner_version", 32),
    ("agent_name", 100),
    ("agent_version", 32),
    ("provider", 64),
    ("requested_model", 128),
    ("executor_name", 64),
    ("executor_version", 32),
    ("persistence_name", 64),
    ("persistence_version", 32),
    ("entity_resolution_policy_name", 64),
    ("entity_resolution_policy_version", 32),
)


class CrossBatchDuplicateGroupingError(Exception):
    """Base class for duplicate grouping failures."""


class CrossBatchDuplicateGroupingStateError(CrossBatchDuplicateGroupingError):
    """Raised when the extraction run is not ready for duplicate grouping."""


class CrossBatchDuplicateGroupingInvariantError(CrossBatchDuplicateGroupingError):
    """Raised when immutable duplicate-grouping invariants are violated."""


class FactValueConsistencyCandidateError(Exception):
    """Base class for consistency candidate failures."""


class FactValueConsistencyCandidateStateError(FactValueConsistencyCandidateError):
    """Raised when the source duplicate grouping application is not available."""


class FactValueConsistencyCandidateInvariantError(FactValueConsistencyCandidateError):
    """Raised when immutable consistency-candidate invariants are violated."""


@dataclass(frozen=True, slots=True)
class AuthenticatedFactValueConsistencyCandidateApplication:
    project_id: uuid.UUID
    application: FactValueConsistencyCandidateApplicationLedger
    source_duplicate_grouping_application: DuplicateGroupingApplicationLedger
    write_plan: FactValueConsistencyCandidateWritePlan
    candidate_ledgers: tuple[FactValueConsistencyCandidateLedger, ...]
    member_ledgers: tuple[FactValueConsistencyCandidateMemberLedger, ...]


@dataclass(frozen=True, slots=True)
class AuthenticatedDuplicateGroupingSourceSnapshot:
    state: duplicate_grouping_repository.DuplicateGroupingOrchestrationState
    candidate_count: int
    candidates: tuple[DuplicateCandidate, ...]
    application_snapshots: tuple[
        fact_extraction_persistence_service.AuthenticatedCompletedFactExtractionApplicationSnapshot,
        ...,
    ] = ()


@dataclass(frozen=True, slots=True)
class _ConsistencyCandidateSourceItem:
    fact_value_id: uuid.UUID
    fact_id: uuid.UUID
    source_batch_id: uuid.UUID
    semantic_key_hash: str


def _require_uuid(value: uuid.UUID, *, field_name: str) -> uuid.UUID:
    if not isinstance(value, uuid.UUID):
        raise CrossBatchDuplicateGroupingError(f"{field_name} must be a UUID")
    return value


def _normalize_string(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _require_orchestration_identity_string(
    value: object,
    *,
    max_length: int,
) -> str:
    if not isinstance(value, str):
        raise CrossBatchDuplicateGroupingInvariantError(
            "cross_batch_duplicate_grouping_orchestration_identity_invalid"
        )
    normalized = normalize_text(value)
    if not normalized or len(normalized) > max_length:
        raise CrossBatchDuplicateGroupingInvariantError(
            "cross_batch_duplicate_grouping_orchestration_identity_invalid"
        )
    return normalized


def _require_orchestration_identity_hash(value: object) -> str:
    if not isinstance(value, str):
        raise CrossBatchDuplicateGroupingInvariantError(
            "cross_batch_duplicate_grouping_orchestration_identity_invalid"
        )
    normalized = normalize_text(value).lower()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise CrossBatchDuplicateGroupingInvariantError(
            "cross_batch_duplicate_grouping_orchestration_identity_invalid"
        )
    return normalized


def _validate_completed_batch_counts(
    state: duplicate_grouping_repository.DuplicateGroupingOrchestrationState,
) -> None:
    if (
        not isinstance(state.batch_count, int)
        or not isinstance(state.completed_batch_count, int)
        or not isinstance(state.failed_batch_count, int)
        or state.batch_count <= 0
        or state.completed_batch_count < 0
        or state.failed_batch_count < 0
    ):
        raise CrossBatchDuplicateGroupingInvariantError(
            "cross_batch_duplicate_grouping_batch_count_mismatch"
        )
    if state.orchestration_status == "completed":
        if (
            state.completed_batch_count != state.batch_count
            or state.failed_batch_count != 0
        ):
            raise CrossBatchDuplicateGroupingInvariantError(
                "cross_batch_duplicate_grouping_batch_count_mismatch"
            )
        return
    if state.orchestration_status == "partial":
        if (
            state.completed_batch_count <= 0
            or state.failed_batch_count <= 0
            or state.completed_batch_count + state.failed_batch_count != state.batch_count
        ):
            raise CrossBatchDuplicateGroupingInvariantError(
                "cross_batch_duplicate_grouping_batch_count_mismatch"
            )


def _validate_orchestration_identity(
    state: duplicate_grouping_repository.DuplicateGroupingOrchestrationState,
) -> None:
    for field_name, max_length in _ORCHESTRATION_IDENTITY_STRING_LIMITS:
        value = getattr(state, field_name)
        normalized = _require_orchestration_identity_string(
            value,
            max_length=max_length,
        )
        if value != normalized:
            raise CrossBatchDuplicateGroupingInvariantError(
                "cross_batch_duplicate_grouping_orchestration_identity_invalid"
            )
    if state.prompt_contract_hash != _require_orchestration_identity_hash(
        state.prompt_contract_hash
    ):
        raise CrossBatchDuplicateGroupingInvariantError(
            "cross_batch_duplicate_grouping_orchestration_identity_invalid"
        )


def _validate_completed_batch_applications(
    completed_batch_applications: Sequence[
        duplicate_grouping_repository.CompletedOrchestrationBatchApplication
    ],
    *,
    state: duplicate_grouping_repository.DuplicateGroupingOrchestrationState,
) -> None:
    if len(completed_batch_applications) != state.completed_batch_count:
        raise CrossBatchDuplicateGroupingInvariantError(
            "cross_batch_duplicate_grouping_batch_count_mismatch"
        )
    seen_batch_ids: set[uuid.UUID] = set()
    seen_batch_indexes: set[int] = set()
    for completed_batch_application in completed_batch_applications:
        if not isinstance(completed_batch_application.source_batch_id, uuid.UUID):
            raise CrossBatchDuplicateGroupingInvariantError(
                "cross_batch_duplicate_grouping_completed_batch_source_mismatch"
            )
        if completed_batch_application.source_batch_id in seen_batch_ids:
            raise CrossBatchDuplicateGroupingInvariantError(
                "cross_batch_duplicate_grouping_completed_batch_source_mismatch"
            )
        seen_batch_ids.add(completed_batch_application.source_batch_id)
        if (
            not isinstance(completed_batch_application.batch_index, int)
            or completed_batch_application.batch_index < 0
            or completed_batch_application.batch_index >= state.batch_count
        ):
            raise CrossBatchDuplicateGroupingInvariantError(
                "cross_batch_duplicate_grouping_completed_batch_source_mismatch"
            )
        if completed_batch_application.batch_index in seen_batch_indexes:
            raise CrossBatchDuplicateGroupingInvariantError(
                "cross_batch_duplicate_grouping_completed_batch_source_mismatch"
            )
        seen_batch_indexes.add(completed_batch_application.batch_index)
        if (
            not isinstance(completed_batch_application.application_id, uuid.UUID)
            or not isinstance(completed_batch_application.current_input_batch_id, uuid.UUID)
            or not isinstance(completed_batch_application.current_inference_run_id, uuid.UUID)
        ):
            raise CrossBatchDuplicateGroupingInvariantError(
                "cross_batch_duplicate_grouping_completed_batch_source_mismatch"
            )


def normalize_duplicate_grouping_algorithm_version(value: str) -> str:
    try:
        return normalize_model_algorithm_version(value)
    except ValueError:
        raise CrossBatchDuplicateGroupingError(
            "cross_batch_duplicate_grouping_invalid_algorithm_version"
        ) from None


def normalize_consistency_candidate_algorithm_version(value: str) -> str:
    try:
        return normalize_model_algorithm_version(value)
    except ValueError:
        raise FactValueConsistencyCandidateError(
            "fact_value_consistency_candidate_invalid_algorithm_version"
        ) from None


def _canonicalize_value(value: Any) -> Any:
    if value is _MISSING:
        return _MISSING
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Enum):
        return _normalize_string(value.value if isinstance(value.value, str) else str(value.value))
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise CrossBatchDuplicateGroupingError("duplicate grouping values must be finite")
        return format(value, "f")
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise CrossBatchDuplicateGroupingError("duplicate grouping datetimes must be timezone-aware")
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, time):
        if value.tzinfo is not None and value.utcoffset() is None:
            raise CrossBatchDuplicateGroupingError("duplicate grouping times must be fully qualified")
        return value.isoformat()
    if isinstance(value, str):
        return _normalize_string(value)
    if isinstance(value, Mapping):
        normalized_items: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CrossBatchDuplicateGroupingError("duplicate grouping payload keys must be strings")
            normalized_item = _canonicalize_value(item)
            if normalized_item is _MISSING:
                continue
            normalized_key = _normalize_string(key)
            if normalized_key in normalized_items:
                raise CrossBatchDuplicateGroupingInvariantError(
                    "cross_batch_duplicate_grouping_nfc_key_collision"
                )
            normalized_items[normalized_key] = normalized_item
        return normalized_items
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonicalize_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CrossBatchDuplicateGroupingError("duplicate grouping floats must be finite")
        return value
    if value is None or isinstance(value, bool | int):
        return value
    raise CrossBatchDuplicateGroupingError("duplicate grouping payload contains unsupported value types")


def _canonical_json_bytes(payload: Any) -> bytes:
    canonical_payload = _canonicalize_value(payload)
    return json.dumps(
        canonical_payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _hash_canonical_bytes(
    canonical_bytes: bytes,
    *,
    digest_bytes_by_hash: dict[str, bytes],
) -> str:
    digest = hashlib.sha256(canonical_bytes).hexdigest()
    existing_bytes = digest_bytes_by_hash.get(digest)
    if existing_bytes is not None and existing_bytes != canonical_bytes:
        raise CrossBatchDuplicateGroupingInvariantError(
            "cross_batch_duplicate_grouping_hash_collision"
        )
    digest_bytes_by_hash[digest] = canonical_bytes
    return digest


def canonicalize_deterministic_payload(payload: Any) -> bytes:
    return _canonical_json_bytes(payload)


def hash_deterministic_payload(
    payload: Any,
    *,
    digest_bytes_by_hash: dict[str, bytes] | None = None,
) -> str:
    digest_map = digest_bytes_by_hash if digest_bytes_by_hash is not None else {}
    return _hash_canonical_bytes(
        _canonical_json_bytes(payload),
        digest_bytes_by_hash=digest_map,
    )


def build_duplicate_fingerprint(
    candidate: DuplicateCandidate,
    *,
    algorithm_version: str = CROSS_BATCH_DUPLICATE_ALGORITHM_VERSION,
    digest_bytes_by_hash: dict[str, bytes] | None = None,
) -> DuplicateFingerprint:
    algorithm_version = normalize_duplicate_grouping_algorithm_version(algorithm_version)
    digest_map = digest_bytes_by_hash if digest_bytes_by_hash is not None else {}
    canonical_bytes = _canonical_json_bytes(
        {
            "algorithm_version": algorithm_version,
            "orchestration_id": candidate.orchestration_id,
            "extraction_run_id": candidate.extraction_run_id,
            "fact_identity": {
                "fact_id": candidate.fact_id,
            },
            "semantic_value": {
                "referenced_entity_id": candidate.referenced_entity_id,
                "value_json": candidate.value_json,
                "value_type": candidate.value_type,
            },
        }
    )
    return DuplicateFingerprint(
        canonical_bytes=canonical_bytes,
        sha256_hex=_hash_canonical_bytes(canonical_bytes, digest_bytes_by_hash=digest_map),
    )


def _validate_candidate(candidate: DuplicateCandidate) -> None:
    _require_uuid(candidate.fact_value_id, field_name="candidate.fact_value_id")
    _require_uuid(candidate.fact_id, field_name="candidate.fact_id")
    _require_uuid(candidate.orchestration_id, field_name="candidate.orchestration_id")
    _require_uuid(candidate.extraction_run_id, field_name="candidate.extraction_run_id")
    _require_uuid(candidate.source_batch_id, field_name="candidate.source_batch_id")
    if not isinstance(candidate.value_type, str) or not candidate.value_type:
        raise CrossBatchDuplicateGroupingInvariantError("duplicate grouping candidate is missing value_type")
    if not candidate.evidence_link_ids:
        raise CrossBatchDuplicateGroupingInvariantError("duplicate grouping candidate is missing evidence links")
    for evidence_link_id in candidate.evidence_link_ids:
        _require_uuid(evidence_link_id, field_name="candidate.evidence_link_id")


def _normalize_candidate_evidence_link_ids(
    evidence_link_ids: Sequence[uuid.UUID],
) -> tuple[uuid.UUID, ...]:
    normalized_ids = tuple(sorted(set(evidence_link_ids), key=str))
    if len(normalized_ids) != len(evidence_link_ids):
        raise CrossBatchDuplicateGroupingInvariantError(
            "cross_batch_duplicate_grouping_duplicate_evidence_link_id"
        )
    return normalized_ids


def build_duplicate_grouping_write_plan(
    candidates: Sequence[DuplicateCandidate],
    *,
    algorithm_version: str = CROSS_BATCH_DUPLICATE_ALGORITHM_VERSION,
) -> DuplicateGroupingWritePlan:
    algorithm_version = normalize_duplicate_grouping_algorithm_version(algorithm_version)
    digest_bytes_by_hash: dict[str, bytes] = {}
    fingerprint_by_fact_value_id: dict[uuid.UUID, DuplicateFingerprint] = {}
    normalized_evidence_link_ids_by_fact_value_id: dict[uuid.UUID, tuple[uuid.UUID, ...]] = {}
    distinct_orchestration_ids = {candidate.orchestration_id for candidate in candidates}
    distinct_extraction_run_ids = {candidate.extraction_run_id for candidate in candidates}
    if len(distinct_orchestration_ids) > 1:
        raise CrossBatchDuplicateGroupingInvariantError(
            "duplicate grouping candidates must belong to a single orchestration"
        )
    if len(distinct_extraction_run_ids) > 1:
        raise CrossBatchDuplicateGroupingInvariantError(
            "duplicate grouping candidates must belong to a single extraction run"
        )

    for candidate in candidates:
        _validate_candidate(candidate)
        if candidate.fact_value_id in fingerprint_by_fact_value_id:
            raise CrossBatchDuplicateGroupingInvariantError(
                "cross_batch_duplicate_grouping_duplicate_fact_value_id"
            )
        normalized_evidence_link_ids_by_fact_value_id[candidate.fact_value_id] = (
            _normalize_candidate_evidence_link_ids(candidate.evidence_link_ids)
        )
        fingerprint_by_fact_value_id[candidate.fact_value_id] = build_duplicate_fingerprint(
            candidate,
            algorithm_version=algorithm_version,
            digest_bytes_by_hash=digest_bytes_by_hash,
        )

    input_manifest_entries = [
        {
            "duplicate_key_hash": fingerprint_by_fact_value_id[candidate.fact_value_id].sha256_hex,
            "evidence_link_ids": [
                str(evidence_link_id)
                for evidence_link_id in normalized_evidence_link_ids_by_fact_value_id[candidate.fact_value_id]
            ],
            "fact_value_id": str(candidate.fact_value_id),
            "source_batch_id": str(candidate.source_batch_id),
        }
        for candidate in sorted(candidates, key=lambda item: str(item.fact_value_id))
    ]
    input_manifest_hash = _hash_canonical_bytes(
        _canonical_json_bytes(input_manifest_entries),
        digest_bytes_by_hash=digest_bytes_by_hash,
    )

    grouped_candidates: dict[str, list[DuplicateCandidate]] = {}
    for candidate in candidates:
        fingerprint = fingerprint_by_fact_value_id[candidate.fact_value_id]
        grouped_candidates.setdefault(fingerprint.sha256_hex, []).append(candidate)

    group_plans: list[DuplicateGroupPlan] = []
    for duplicate_key_hash, grouped in grouped_candidates.items():
        distinct_batch_ids = {candidate.source_batch_id for candidate in grouped}
        if len(distinct_batch_ids) < 2:
            continue
        ordered_members = tuple(
            DuplicateGroupMemberPlan(
                fact_value_id=candidate.fact_value_id,
                source_batch_id=candidate.source_batch_id,
            )
            for candidate in sorted(
                grouped,
                key=lambda item: (str(item.fact_value_id), str(item.source_batch_id)),
            )
        )
        group_plans.append(
            DuplicateGroupPlan(
                duplicate_key_hash=duplicate_key_hash,
                member_count=len(ordered_members),
                distinct_batch_count=len(distinct_batch_ids),
                members=ordered_members,
            )
        )

    group_plans = sorted(group_plans, key=lambda group: group.duplicate_key_hash)
    result_manifest_entries = [
        {
            "duplicate_key_hash": group.duplicate_key_hash,
            "members": [
                {
                    "fact_value_id": str(member.fact_value_id),
                    "source_batch_id": str(member.source_batch_id),
                }
                for member in group.members
            ],
        }
        for group in group_plans
    ]
    result_manifest_hash = _hash_canonical_bytes(
        _canonical_json_bytes(result_manifest_entries),
        digest_bytes_by_hash=digest_bytes_by_hash,
    )

    return DuplicateGroupingWritePlan(
        algorithm_version=algorithm_version,
        input_manifest_hash=input_manifest_hash,
        result_manifest_hash=result_manifest_hash,
        input_fact_value_count=len(candidates),
        duplicate_group_count=len(group_plans),
        duplicate_member_count=sum(group.member_count for group in group_plans),
        groups=tuple(group_plans),
    )


def build_duplicate_group_evidence_union(
    projections: Sequence[DuplicateGroupEvidenceProjection],
) -> tuple[uuid.UUID, ...]:
    seen: set[uuid.UUID] = set()
    union_ids: list[uuid.UUID] = []
    for projection in projections:
        for evidence_id in projection.evidence_ids:
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            union_ids.append(evidence_id)
    return tuple(union_ids)


def _assert_application_matches_plan(
    application: DuplicateGroupingApplicationLedger,
    plan: DuplicateGroupingWritePlan,
) -> None:
    if application.algorithm_version != plan.algorithm_version:
        raise CrossBatchDuplicateGroupingInvariantError(
            "cross_batch_duplicate_grouping_immutable_ledger_mismatch"
        )
    if application.input_manifest_hash != plan.input_manifest_hash:
        raise CrossBatchDuplicateGroupingInvariantError(
            "cross_batch_duplicate_grouping_immutable_ledger_mismatch"
        )
    if application.result_manifest_hash != plan.result_manifest_hash:
        raise CrossBatchDuplicateGroupingInvariantError(
            "cross_batch_duplicate_grouping_immutable_ledger_mismatch"
        )
    if application.input_fact_value_count != plan.input_fact_value_count:
        raise CrossBatchDuplicateGroupingInvariantError(
            "cross_batch_duplicate_grouping_immutable_ledger_mismatch"
        )
    if application.duplicate_group_count != plan.duplicate_group_count:
        raise CrossBatchDuplicateGroupingInvariantError(
            "cross_batch_duplicate_grouping_immutable_ledger_mismatch"
        )
    if application.duplicate_member_count != plan.duplicate_member_count:
        raise CrossBatchDuplicateGroupingInvariantError(
            "cross_batch_duplicate_grouping_immutable_ledger_mismatch"
        )


def _build_result(
    application: DuplicateGroupingApplicationLedger,
    *,
    created_new: bool,
) -> DuplicateGroupingResult:
    return DuplicateGroupingResult(
        grouping_application_id=application.id,
        algorithm_version=application.algorithm_version,
        input_fact_value_count=application.input_fact_value_count,
        duplicate_group_count=application.duplicate_group_count,
        duplicate_member_count=application.duplicate_member_count,
        created_new=created_new,
    )


def _require_consistency_uuid(value: uuid.UUID, *, field_name: str) -> uuid.UUID:
    if not isinstance(value, uuid.UUID):
        raise FactValueConsistencyCandidateError(f"{field_name} must be a UUID")
    return value


def _normalize_supported_source_duplicate_grouping_algorithm_version(
    algorithm_version: str,
) -> str:
    normalized = normalize_duplicate_grouping_algorithm_version(algorithm_version)
    if normalized != CROSS_BATCH_DUPLICATE_ALGORITHM_VERSION:
        raise FactValueConsistencyCandidateStateError(
            "fact_value_consistency_candidate_source_algorithm_unsupported"
        )
    return normalized


def _assert_source_duplicate_grouping_application_matches_snapshot(
    read_snapshot: DuplicateGroupingApplicationLedger,
    current_snapshot: DuplicateGroupingApplicationLedger,
) -> None:
    if read_snapshot.id != current_snapshot.id:
        raise FactValueConsistencyCandidateInvariantError(
            "fact_value_consistency_candidate_source_snapshot_mismatch"
        )
    if read_snapshot.orchestration_id != current_snapshot.orchestration_id:
        raise FactValueConsistencyCandidateInvariantError(
            "fact_value_consistency_candidate_source_snapshot_mismatch"
        )
    if read_snapshot.extraction_run_id != current_snapshot.extraction_run_id:
        raise FactValueConsistencyCandidateInvariantError(
            "fact_value_consistency_candidate_source_snapshot_mismatch"
        )
    if read_snapshot.algorithm_version != current_snapshot.algorithm_version:
        raise FactValueConsistencyCandidateInvariantError(
            "fact_value_consistency_candidate_source_snapshot_mismatch"
        )
    if read_snapshot.input_manifest_hash != current_snapshot.input_manifest_hash:
        raise FactValueConsistencyCandidateInvariantError(
            "fact_value_consistency_candidate_source_snapshot_mismatch"
        )
    if read_snapshot.result_manifest_hash != current_snapshot.result_manifest_hash:
        raise FactValueConsistencyCandidateInvariantError(
            "fact_value_consistency_candidate_source_snapshot_mismatch"
        )
    if read_snapshot.input_fact_value_count != current_snapshot.input_fact_value_count:
        raise FactValueConsistencyCandidateInvariantError(
            "fact_value_consistency_candidate_source_snapshot_mismatch"
        )
    if read_snapshot.duplicate_group_count != current_snapshot.duplicate_group_count:
        raise FactValueConsistencyCandidateInvariantError(
            "fact_value_consistency_candidate_source_snapshot_mismatch"
        )
    if read_snapshot.duplicate_member_count != current_snapshot.duplicate_member_count:
        raise FactValueConsistencyCandidateInvariantError(
            "fact_value_consistency_candidate_source_snapshot_mismatch"
        )


def _assert_source_duplicate_grouping_application_matches_input(
    application: DuplicateGroupingApplicationLedger,
    *,
    source_input_manifest_hash: str,
    source_input_fact_value_count: int,
    orchestration_id: uuid.UUID,
    extraction_run_id: uuid.UUID,
) -> None:
    if application.orchestration_id != orchestration_id:
        raise FactValueConsistencyCandidateInvariantError(
            "fact_value_consistency_candidate_source_orchestration_mismatch"
        )
    if application.extraction_run_id != extraction_run_id:
        raise FactValueConsistencyCandidateInvariantError(
            "fact_value_consistency_candidate_source_extraction_run_mismatch"
        )
    if application.input_manifest_hash != source_input_manifest_hash:
        raise FactValueConsistencyCandidateInvariantError(
            "fact_value_consistency_candidate_source_input_manifest_mismatch"
        )
    if application.input_fact_value_count != source_input_fact_value_count:
        raise FactValueConsistencyCandidateInvariantError(
            "fact_value_consistency_candidate_source_input_count_mismatch"
        )


def build_fact_value_consistency_candidate_write_plan(
    candidates: Sequence[DuplicateCandidate],
    *,
    source_duplicate_grouping_application: DuplicateGroupingApplicationLedger,
    algorithm_version: str = CROSS_BATCH_MULTI_VALUE_CANDIDATE_ALGORITHM_VERSION,
) -> FactValueConsistencyCandidateWritePlan:
    algorithm_version = normalize_consistency_candidate_algorithm_version(algorithm_version)
    source_algorithm_version = _normalize_supported_source_duplicate_grouping_algorithm_version(
        source_duplicate_grouping_application.algorithm_version
    )
    try:
        source_duplicate_plan = build_duplicate_grouping_write_plan(
            candidates,
            algorithm_version=source_algorithm_version,
        )
    except CrossBatchDuplicateGroupingError as error:
        raise FactValueConsistencyCandidateInvariantError(str(error)) from None

    orchestration_ids = {candidate.orchestration_id for candidate in candidates}
    extraction_run_ids = {candidate.extraction_run_id for candidate in candidates}
    if len(orchestration_ids) > 1 or (
        orchestration_ids and source_duplicate_grouping_application.orchestration_id not in orchestration_ids
    ):
        raise FactValueConsistencyCandidateInvariantError(
            "fact_value_consistency_candidate_source_orchestration_mismatch"
        )
    if len(extraction_run_ids) > 1 or (
        extraction_run_ids and source_duplicate_grouping_application.extraction_run_id not in extraction_run_ids
    ):
        raise FactValueConsistencyCandidateInvariantError(
            "fact_value_consistency_candidate_source_extraction_run_mismatch"
        )
    _assert_source_duplicate_grouping_application_matches_input(
        source_duplicate_grouping_application,
        source_input_manifest_hash=source_duplicate_plan.input_manifest_hash,
        source_input_fact_value_count=source_duplicate_plan.input_fact_value_count,
        orchestration_id=source_duplicate_grouping_application.orchestration_id,
        extraction_run_id=source_duplicate_grouping_application.extraction_run_id,
    )

    digest_bytes_by_hash: dict[str, bytes] = {}
    grouped_by_fact_id: dict[uuid.UUID, list[_ConsistencyCandidateSourceItem]] = {}
    seen_fact_value_ids: set[uuid.UUID] = set()
    for candidate in candidates:
        if candidate.fact_value_id in seen_fact_value_ids:
            raise FactValueConsistencyCandidateInvariantError(
                "fact_value_consistency_candidate_duplicate_fact_value_id"
            )
        seen_fact_value_ids.add(candidate.fact_value_id)
        try:
            semantic_key_hash = build_duplicate_fingerprint(
                candidate,
                algorithm_version=source_algorithm_version,
                digest_bytes_by_hash=digest_bytes_by_hash,
            ).sha256_hex
        except CrossBatchDuplicateGroupingError as error:
            raise FactValueConsistencyCandidateInvariantError(str(error)) from None
        grouped_by_fact_id.setdefault(candidate.fact_id, []).append(
            _ConsistencyCandidateSourceItem(
                fact_value_id=candidate.fact_value_id,
                fact_id=candidate.fact_id,
                source_batch_id=candidate.source_batch_id,
                semantic_key_hash=semantic_key_hash,
            )
        )

    candidate_plans: list[FactValueConsistencyCandidatePlan] = []
    for fact_id, fact_items in grouped_by_fact_id.items():
        distinct_semantic_key_count = len({item.semantic_key_hash for item in fact_items})
        if distinct_semantic_key_count < 2:
            continue
        if not any(
            left.semantic_key_hash != right.semantic_key_hash and left.source_batch_id != right.source_batch_id
            for left in fact_items
            for right in fact_items
        ):
            continue
        ordered_members = tuple(
            FactValueConsistencyCandidateMemberPlan(
                fact_value_id=item.fact_value_id,
                source_batch_id=item.source_batch_id,
                semantic_key_hash=item.semantic_key_hash,
            )
            for item in sorted(
                fact_items,
                key=lambda current: (
                    str(current.fact_value_id),
                    str(current.source_batch_id),
                    current.semantic_key_hash,
                ),
            )
        )
        candidate_plans.append(
            FactValueConsistencyCandidatePlan(
                fact_id=fact_id,
                candidate_kind="multi_value",
                member_count=len(ordered_members),
                distinct_semantic_key_count=distinct_semantic_key_count,
                distinct_batch_count=len({item.source_batch_id for item in fact_items}),
                members=ordered_members,
            )
        )

    candidate_plans = sorted(candidate_plans, key=lambda item: (str(item.fact_id), item.candidate_kind))
    result_manifest_hash = _hash_canonical_bytes(
        _canonical_json_bytes(
            [
                {
                    "candidate_kind": candidate_plan.candidate_kind,
                    "fact_id": str(candidate_plan.fact_id),
                    "members": [
                        {
                            "fact_value_id": str(member.fact_value_id),
                            "semantic_key_hash": member.semantic_key_hash,
                            "source_batch_id": str(member.source_batch_id),
                        }
                        for member in candidate_plan.members
                    ],
                }
                for candidate_plan in candidate_plans
            ]
        ),
        digest_bytes_by_hash=digest_bytes_by_hash,
    )
    return FactValueConsistencyCandidateWritePlan(
        algorithm_version=algorithm_version,
        source_duplicate_grouping_algorithm_version=source_algorithm_version,
        input_manifest_hash=source_duplicate_grouping_application.input_manifest_hash,
        result_manifest_hash=result_manifest_hash,
        candidate_count=len(candidate_plans),
        member_count=sum(candidate.member_count for candidate in candidate_plans),
        candidates=tuple(candidate_plans),
    )


def _assert_consistency_application_matches_plan(
    application: FactValueConsistencyCandidateApplicationLedger,
    *,
    duplicate_grouping_application_id: uuid.UUID,
    orchestration_id: uuid.UUID,
    extraction_run_id: uuid.UUID,
    plan: FactValueConsistencyCandidateWritePlan,
) -> None:
    if application.duplicate_grouping_application_id != duplicate_grouping_application_id:
        raise FactValueConsistencyCandidateInvariantError(
            "fact_value_consistency_candidate_immutable_ledger_mismatch"
        )
    if application.orchestration_id != orchestration_id:
        raise FactValueConsistencyCandidateInvariantError(
            "fact_value_consistency_candidate_immutable_ledger_mismatch"
        )
    if application.extraction_run_id != extraction_run_id:
        raise FactValueConsistencyCandidateInvariantError(
            "fact_value_consistency_candidate_immutable_ledger_mismatch"
        )
    if application.algorithm_version != plan.algorithm_version:
        raise FactValueConsistencyCandidateInvariantError(
            "fact_value_consistency_candidate_immutable_ledger_mismatch"
        )
    if application.input_manifest_hash != plan.input_manifest_hash:
        raise FactValueConsistencyCandidateInvariantError(
            "fact_value_consistency_candidate_immutable_ledger_mismatch"
        )
    if application.result_manifest_hash != plan.result_manifest_hash:
        raise FactValueConsistencyCandidateInvariantError(
            "fact_value_consistency_candidate_immutable_ledger_mismatch"
        )
    if application.candidate_count != plan.candidate_count:
        raise FactValueConsistencyCandidateInvariantError(
            "fact_value_consistency_candidate_immutable_ledger_mismatch"
        )
    if application.member_count != plan.member_count:
        raise FactValueConsistencyCandidateInvariantError(
            "fact_value_consistency_candidate_immutable_ledger_mismatch"
        )


def _build_consistency_candidate_result(
    application: FactValueConsistencyCandidateApplicationLedger,
    *,
    created_new: bool,
) -> FactValueConsistencyCandidateResult:
    return FactValueConsistencyCandidateResult(
        consistency_application_id=application.id,
        duplicate_grouping_application_id=application.duplicate_grouping_application_id,
        algorithm_version=application.algorithm_version,
        candidate_count=application.candidate_count,
        member_count=application.member_count,
        created_new=created_new,
    )


def _compute_consistency_candidate_result_manifest_hash(
    candidate_ledgers: Sequence[FactValueConsistencyCandidateLedger],
    member_ledgers: Sequence[FactValueConsistencyCandidateMemberLedger],
    *,
    application: FactValueConsistencyCandidateApplicationLedger,
    plan: FactValueConsistencyCandidateWritePlan,
) -> str:
    if len(candidate_ledgers) != application.candidate_count:
        raise FactValueConsistencyCandidateInvariantError(
            "fact_value_consistency_candidate_subledger_mismatch"
        )
    if len(member_ledgers) != application.member_count:
        raise FactValueConsistencyCandidateInvariantError(
            "fact_value_consistency_candidate_subledger_mismatch"
        )
    if len(candidate_ledgers) != plan.candidate_count:
        raise FactValueConsistencyCandidateInvariantError(
            "fact_value_consistency_candidate_subledger_mismatch"
        )
    if len(member_ledgers) != plan.member_count:
        raise FactValueConsistencyCandidateInvariantError(
            "fact_value_consistency_candidate_subledger_mismatch"
        )

    members_by_candidate_id: dict[uuid.UUID, list[FactValueConsistencyCandidateMemberLedger]] = {}
    seen_fact_value_ids: set[uuid.UUID] = set()
    for member in member_ledgers:
        if member.consistency_application_id != application.id:
            raise FactValueConsistencyCandidateInvariantError(
                "fact_value_consistency_candidate_subledger_mismatch"
            )
        if member.orchestration_id != application.orchestration_id:
            raise FactValueConsistencyCandidateInvariantError(
                "fact_value_consistency_candidate_subledger_mismatch"
            )
        if member.fact_value_id in seen_fact_value_ids:
            raise FactValueConsistencyCandidateInvariantError(
                "fact_value_consistency_candidate_subledger_mismatch"
            )
        seen_fact_value_ids.add(member.fact_value_id)
        members_by_candidate_id.setdefault(member.candidate_id, []).append(member)

    manifest_entries: list[dict[str, object]] = []
    seen_candidate_business_keys: set[tuple[uuid.UUID, str]] = set()
    candidate_ids = {candidate.id for candidate in candidate_ledgers}
    if set(members_by_candidate_id) - candidate_ids:
        raise FactValueConsistencyCandidateInvariantError(
            "fact_value_consistency_candidate_subledger_mismatch"
        )

    for candidate in sorted(
        candidate_ledgers,
        key=lambda item: (str(item.fact_id), item.candidate_kind, str(item.id)),
    ):
        if candidate.consistency_application_id != application.id:
            raise FactValueConsistencyCandidateInvariantError(
                "fact_value_consistency_candidate_subledger_mismatch"
            )
        candidate_business_key = (candidate.fact_id, candidate.candidate_kind)
        if candidate_business_key in seen_candidate_business_keys:
            raise FactValueConsistencyCandidateInvariantError(
                "fact_value_consistency_candidate_subledger_mismatch"
            )
        seen_candidate_business_keys.add(candidate_business_key)

        candidate_members = members_by_candidate_id.get(candidate.id, [])
        actual_member_count = len(candidate_members)
        actual_distinct_semantic_key_count = len(
            {member.semantic_key_hash for member in candidate_members}
        )
        actual_distinct_batch_count = len({member.source_batch_id for member in candidate_members})
        if candidate.member_count != actual_member_count:
            raise FactValueConsistencyCandidateInvariantError(
                "fact_value_consistency_candidate_subledger_mismatch"
            )
        if candidate.distinct_semantic_key_count != actual_distinct_semantic_key_count:
            raise FactValueConsistencyCandidateInvariantError(
                "fact_value_consistency_candidate_subledger_mismatch"
            )
        if candidate.distinct_batch_count != actual_distinct_batch_count:
            raise FactValueConsistencyCandidateInvariantError(
                "fact_value_consistency_candidate_subledger_mismatch"
            )

        ordered_members = sorted(
            candidate_members,
            key=lambda item: (
                str(item.fact_value_id),
                str(item.source_batch_id),
                item.semantic_key_hash,
                str(item.id),
            ),
        )
        manifest_entries.append(
            {
                "candidate_kind": candidate.candidate_kind,
                "fact_id": str(candidate.fact_id),
                "members": [
                    {
                        "fact_value_id": str(member.fact_value_id),
                        "semantic_key_hash": member.semantic_key_hash,
                        "source_batch_id": str(member.source_batch_id),
                    }
                    for member in ordered_members
                ],
            }
        )

    try:
        result_manifest_hash = _hash_canonical_bytes(
            _canonical_json_bytes(manifest_entries),
            digest_bytes_by_hash={},
        )
    except CrossBatchDuplicateGroupingError as error:
        raise FactValueConsistencyCandidateInvariantError(str(error)) from None

    if result_manifest_hash != application.result_manifest_hash:
        raise FactValueConsistencyCandidateInvariantError(
            "fact_value_consistency_candidate_subledger_mismatch"
        )
    if result_manifest_hash != plan.result_manifest_hash:
        raise FactValueConsistencyCandidateInvariantError(
            "fact_value_consistency_candidate_subledger_mismatch"
        )
    return result_manifest_hash


async def _assert_consistency_candidate_subledgers_match_plan(
    session: AsyncSession,
    *,
    application: FactValueConsistencyCandidateApplicationLedger,
    plan: FactValueConsistencyCandidateWritePlan,
) -> None:
    candidate_ledgers, member_ledgers = await _load_consistency_candidate_subledgers(
        session,
        consistency_application_id=application.id,
    )
    _compute_consistency_candidate_result_manifest_hash(
        candidate_ledgers,
        member_ledgers,
        application=application,
        plan=plan,
    )


async def _load_consistency_candidate_subledgers(
    session: AsyncSession,
    *,
    consistency_application_id: uuid.UUID,
) -> tuple[
    tuple[FactValueConsistencyCandidateLedger, ...],
    tuple[FactValueConsistencyCandidateMemberLedger, ...],
]:
    candidate_ledgers = await duplicate_grouping_repository.list_consistency_candidate_ledgers(
        session,
        consistency_application_id=consistency_application_id,
    )
    member_ledgers = await duplicate_grouping_repository.list_consistency_candidate_member_ledgers(
        session,
        consistency_application_id=consistency_application_id,
    )
    return candidate_ledgers, member_ledgers


def _get_integrity_constraint_name(error: IntegrityError) -> str | None:
    diag = getattr(error.orig, "diag", None)
    constraint_name = getattr(diag, "constraint_name", None)
    if constraint_name is None or not isinstance(constraint_name, str):
        return None
    return constraint_name


def _validate_run_state(
    state: duplicate_grouping_repository.DuplicateGroupingOrchestrationState | None,
    *,
    orchestration_id: uuid.UUID,
) -> duplicate_grouping_repository.DuplicateGroupingOrchestrationState:
    if state is None:
        raise CrossBatchDuplicateGroupingStateError("cross_batch_duplicate_grouping_orchestration_not_found")
    if state.orchestration_id != orchestration_id:
        raise CrossBatchDuplicateGroupingInvariantError(
            "cross_batch_duplicate_grouping_orchestration_identity_mismatch"
        )
    if state.extraction_run_status != ExtractionRunStatus.COMPLETED.value:
        raise CrossBatchDuplicateGroupingStateError("cross_batch_duplicate_grouping_run_not_ready")
    if state.extraction_run_outcome not in {
        ExtractionRunOutcome.SUCCESS.value,
        ExtractionRunOutcome.PARTIAL.value,
    }:
        raise CrossBatchDuplicateGroupingStateError("cross_batch_duplicate_grouping_run_not_ready")
    if state.orchestration_status not in {
        "completed",
        "partial",
    }:
        raise CrossBatchDuplicateGroupingStateError("cross_batch_duplicate_grouping_orchestration_not_ready")
    _validate_orchestration_identity(state)
    _validate_completed_batch_counts(state)
    return state


async def _read_existing_application_result(
    session: AsyncSession,
    *,
    orchestration_id: uuid.UUID,
    algorithm_version: str,
    plan: DuplicateGroupingWritePlan,
) -> DuplicateGroupingResult | None:
    existing_application = await duplicate_grouping_repository.get_grouping_application_ledger(
        session,
        orchestration_id=orchestration_id,
        algorithm_version=algorithm_version,
    )
    if existing_application is None:
        return None
    _assert_application_matches_plan(existing_application, plan)
    return _build_result(existing_application, created_new=False)


async def list_duplicate_group_ledgers(
    session: AsyncSession,
    *,
    grouping_application_id: uuid.UUID,
) -> tuple[DuplicateGroupLedger, ...]:
    return await duplicate_grouping_repository.list_group_ledgers(
        session,
        grouping_application_id=grouping_application_id,
    )


async def list_duplicate_group_member_ledgers(
    session: AsyncSession,
    *,
    grouping_application_id: uuid.UUID,
) -> tuple[DuplicateGroupMemberLedger, ...]:
    return await duplicate_grouping_repository.list_member_ledgers(
        session,
        grouping_application_id=grouping_application_id,
    )


async def list_duplicate_group_evidence_projections(
    session: AsyncSession,
    *,
    group_id: uuid.UUID,
) -> tuple[DuplicateGroupEvidenceProjection, ...]:
    return await duplicate_grouping_repository.list_duplicate_group_evidence_projections(
        session,
        group_id=group_id,
    )


async def ensure_cross_batch_duplicate_grouping(
    session_factory: Callable[[], AsyncSession],
    *,
    orchestration_id: uuid.UUID,
    algorithm_version: str = CROSS_BATCH_DUPLICATE_ALGORITHM_VERSION,
) -> DuplicateGroupingResult:
    orchestration_id = _require_uuid(orchestration_id, field_name="orchestration_id")
    algorithm_version = normalize_duplicate_grouping_algorithm_version(algorithm_version)
    source_snapshot = await authenticate_duplicate_grouping_source_snapshot(
        session_factory,
        orchestration_id=orchestration_id,
    )
    state = source_snapshot.state
    candidates = source_snapshot.candidates

    write_plan = build_duplicate_grouping_write_plan(
        candidates,
        algorithm_version=algorithm_version,
    )

    async with session_factory() as write_session:
        try:
            state = await duplicate_grouping_repository.get_duplicate_grouping_orchestration_state(
                write_session,
                orchestration_id=orchestration_id,
            )
            state = _validate_run_state(state, orchestration_id=orchestration_id)
            if await duplicate_grouping_repository.has_invalid_completed_batch_bindings(
                write_session,
                orchestration_id=orchestration_id,
            ):
                raise CrossBatchDuplicateGroupingInvariantError(
                    "cross_batch_duplicate_grouping_completed_batch_binding_mismatch"
                )

            existing_application = await duplicate_grouping_repository.get_grouping_application_for_update(
                write_session,
                orchestration_id=orchestration_id,
                algorithm_version=algorithm_version,
            )
            if existing_application is not None:
                existing_ledger = DuplicateGroupingApplicationLedger(
                    id=existing_application.id,
                    orchestration_id=existing_application.orchestration_id,
                    extraction_run_id=existing_application.extraction_run_id,
                    algorithm_version=existing_application.algorithm_version,
                    input_manifest_hash=existing_application.input_manifest_hash,
                    result_manifest_hash=existing_application.result_manifest_hash,
                    input_fact_value_count=existing_application.input_fact_value_count,
                    duplicate_group_count=existing_application.duplicate_group_count,
                    duplicate_member_count=existing_application.duplicate_member_count,
                    created_at=existing_application.created_at,
                )
                _assert_application_matches_plan(existing_ledger, write_plan)
                await write_session.commit()
                logger.info(
                    "Cross-batch duplicate grouping hit existing ledger",
                    extra={
                        "extraction_run_id": str(existing_ledger.extraction_run_id),
                        "orchestration_id": str(orchestration_id),
                        "grouping_application_id": str(existing_ledger.id),
                        "algorithm_version": algorithm_version,
                        "candidate_count": write_plan.input_fact_value_count,
                        "duplicate_group_count": write_plan.duplicate_group_count,
                        "duplicate_member_count": write_plan.duplicate_member_count,
                    },
                )
                return _build_result(existing_ledger, created_new=False)

            application = FactValueDuplicateGroupingApplication(
                id=uuid.uuid4(),
                orchestration_id=orchestration_id,
                extraction_run_id=state.extraction_run_id,
                algorithm_version=algorithm_version,
                input_manifest_hash=write_plan.input_manifest_hash,
                result_manifest_hash=write_plan.result_manifest_hash,
                input_fact_value_count=write_plan.input_fact_value_count,
                duplicate_group_count=write_plan.duplicate_group_count,
                duplicate_member_count=write_plan.duplicate_member_count,
            )
            await duplicate_grouping_repository.create_grouping_application(write_session, application)

            groups: list[FactValueDuplicateGroup] = []
            group_id_by_hash: dict[str, uuid.UUID] = {}
            for group_plan in write_plan.groups:
                group_id = uuid.uuid4()
                group_id_by_hash[group_plan.duplicate_key_hash] = group_id
                groups.append(
                    FactValueDuplicateGroup(
                        id=group_id,
                        grouping_application_id=application.id,
                        duplicate_key_hash=group_plan.duplicate_key_hash,
                        member_count=group_plan.member_count,
                        distinct_batch_count=group_plan.distinct_batch_count,
                    )
                )
            if groups:
                await duplicate_grouping_repository.create_duplicate_groups(write_session, groups)

            members: list[FactValueDuplicateGroupMember] = []
            for group_plan in write_plan.groups:
                group_id = group_id_by_hash[group_plan.duplicate_key_hash]
                for member_plan in group_plan.members:
                    members.append(
                        FactValueDuplicateGroupMember(
                            id=uuid.uuid4(),
                            orchestration_id=orchestration_id,
                            grouping_application_id=application.id,
                            group_id=group_id,
                            fact_value_id=member_plan.fact_value_id,
                            source_batch_id=member_plan.source_batch_id,
                        )
                    )
            if members:
                await duplicate_grouping_repository.create_duplicate_group_members(write_session, members)

            if len(groups) != write_plan.duplicate_group_count:
                raise CrossBatchDuplicateGroupingInvariantError(
                    "cross_batch_duplicate_grouping_write_count_mismatch"
                )
            if len(members) != write_plan.duplicate_member_count:
                raise CrossBatchDuplicateGroupingInvariantError(
                    "cross_batch_duplicate_grouping_write_count_mismatch"
                )

            await write_session.commit()
            application_ledger = DuplicateGroupingApplicationLedger(
                id=application.id,
                orchestration_id=application.orchestration_id,
                extraction_run_id=application.extraction_run_id,
                algorithm_version=application.algorithm_version,
                input_manifest_hash=application.input_manifest_hash,
                result_manifest_hash=application.result_manifest_hash,
                input_fact_value_count=application.input_fact_value_count,
                duplicate_group_count=application.duplicate_group_count,
                duplicate_member_count=application.duplicate_member_count,
                created_at=application.created_at,
            )
            logger.info(
                "Cross-batch duplicate grouping created new ledger",
                extra={
                    "extraction_run_id": str(application.extraction_run_id),
                    "orchestration_id": str(orchestration_id),
                    "grouping_application_id": str(application.id),
                    "algorithm_version": algorithm_version,
                    "candidate_count": write_plan.input_fact_value_count,
                    "duplicate_group_count": write_plan.duplicate_group_count,
                    "duplicate_member_count": write_plan.duplicate_member_count,
                },
            )
            return _build_result(application_ledger, created_new=True)
        except IntegrityError as error:
            constraint_name = _get_integrity_constraint_name(error)
            await write_session.rollback()
            if constraint_name != _APPLICATION_UNIQUE_CONSTRAINT:
                raise
            logger.info(
                "Cross-batch duplicate grouping hit concurrent application create",
                extra={
                    "extraction_run_id": str(state.extraction_run_id),
                    "orchestration_id": str(orchestration_id),
                    "algorithm_version": algorithm_version,
                    "constraint_name": constraint_name,
                    "candidate_count": write_plan.input_fact_value_count,
                    "duplicate_group_count": write_plan.duplicate_group_count,
                    "duplicate_member_count": write_plan.duplicate_member_count,
                },
            )
        except BaseException:
            await write_session.rollback()
            raise

    async with session_factory() as read_session:
        try:
            existing_result = await _read_existing_application_result(
                read_session,
                orchestration_id=orchestration_id,
                algorithm_version=algorithm_version,
                plan=write_plan,
            )
        except BaseException:
            await read_session.rollback()
            raise
        else:
            await read_session.rollback()
    if existing_result is None:
        raise CrossBatchDuplicateGroupingInvariantError(
            "cross_batch_duplicate_grouping_concurrent_ledger_missing"
        )
    return existing_result


async def authenticate_duplicate_grouping_source_snapshot(
    session_factory: Callable[[], AsyncSession],
    *,
    orchestration_id: uuid.UUID,
) -> AuthenticatedDuplicateGroupingSourceSnapshot:
    orchestration_id = _require_uuid(orchestration_id, field_name="orchestration_id")

    async with session_factory() as read_session:
        try:
            state = await duplicate_grouping_repository.get_duplicate_grouping_orchestration_state(
                read_session,
                orchestration_id=orchestration_id,
            )
            state = _validate_run_state(state, orchestration_id=orchestration_id)
            if await duplicate_grouping_repository.has_invalid_completed_batch_bindings(
                read_session,
                orchestration_id=orchestration_id,
            ):
                raise CrossBatchDuplicateGroupingInvariantError(
                    "cross_batch_duplicate_grouping_completed_batch_binding_mismatch"
                )
            application_snapshots = []
            authoritative_fact_value_ids: set[uuid.UUID] = set()
            authoritative_items_by_fact_value_id: dict[
                uuid.UUID,
                fact_extraction_persistence_service.AuthenticatedPersistedFactProposalItem,
            ] = {}
            authoritative_source_batch_by_fact_value_id: dict[uuid.UUID, uuid.UUID] = {}
            completed_batch_applications = (
                await duplicate_grouping_repository.list_completed_orchestration_batch_applications(
                    read_session,
                    orchestration_id=orchestration_id,
                    batch_count=state.batch_count,
                )
            )
            _validate_completed_batch_applications(
                completed_batch_applications,
                state=state,
            )
            for completed_batch_application in completed_batch_applications:
                application_snapshot = (
                    await fact_extraction_persistence_service.authenticate_completed_fact_extraction_application(
                        read_session,
                        application_id=completed_batch_application.application_id,
                    )
                )
                if application_snapshot.project_id != state.project_id:
                    raise CrossBatchDuplicateGroupingInvariantError(
                        "cross_batch_duplicate_grouping_application_project_mismatch"
                    )
                if application_snapshot.extraction_run_id != state.extraction_run_id:
                    raise CrossBatchDuplicateGroupingInvariantError(
                        "cross_batch_duplicate_grouping_application_extraction_run_mismatch"
                    )
                if (
                    application_snapshot.input_batch_id
                    != completed_batch_application.current_input_batch_id
                ):
                    raise CrossBatchDuplicateGroupingInvariantError(
                        "cross_batch_duplicate_grouping_application_input_batch_mismatch"
                    )
                if (
                    application_snapshot.inference_run_id
                    != completed_batch_application.current_inference_run_id
                ):
                    raise CrossBatchDuplicateGroupingInvariantError(
                        "cross_batch_duplicate_grouping_application_inference_run_mismatch"
                    )
                if (
                    application_snapshot.persistence_name != state.persistence_name
                    or application_snapshot.persistence_version != state.persistence_version
                    or application_snapshot.entity_resolution_policy_name
                    != state.entity_resolution_policy_name
                    or application_snapshot.entity_resolution_policy_version
                    != state.entity_resolution_policy_version
                ):
                    raise CrossBatchDuplicateGroupingInvariantError(
                        "cross_batch_duplicate_grouping_application_identity_mismatch"
                    )
                application_snapshots.append(application_snapshot)
                for item in application_snapshot.items:
                    if item.fact_value_id in authoritative_fact_value_ids:
                        raise CrossBatchDuplicateGroupingInvariantError(
                            "cross_batch_duplicate_grouping_candidate_source_mismatch"
                        )
                    authoritative_fact_value_ids.add(item.fact_value_id)
                    authoritative_items_by_fact_value_id[item.fact_value_id] = item
                    authoritative_source_batch_by_fact_value_id[item.fact_value_id] = (
                        completed_batch_application.source_batch_id
                    )
            candidate_count = await duplicate_grouping_repository.count_duplicate_candidate_fact_values(
                read_session,
                orchestration_id=orchestration_id,
            )
            candidates = await duplicate_grouping_repository.list_duplicate_candidates(
                read_session,
                orchestration_id=orchestration_id,
            )
            if candidate_count != len(candidates):
                raise CrossBatchDuplicateGroupingInvariantError(
                    "cross_batch_duplicate_grouping_candidate_source_mismatch"
                )
            if len(application_snapshots) != state.completed_batch_count:
                raise CrossBatchDuplicateGroupingInvariantError(
                    "cross_batch_duplicate_grouping_batch_count_mismatch"
                )
            if len(authoritative_fact_value_ids) != candidate_count:
                raise CrossBatchDuplicateGroupingInvariantError(
                    "cross_batch_duplicate_grouping_candidate_source_mismatch"
                )
            candidate_by_fact_value_id = {
                candidate.fact_value_id: candidate for candidate in candidates
            }
            if len(candidate_by_fact_value_id) != len(candidates):
                raise CrossBatchDuplicateGroupingInvariantError(
                    "cross_batch_duplicate_grouping_candidate_source_mismatch"
                )
            if set(candidate_by_fact_value_id) != authoritative_fact_value_ids:
                raise CrossBatchDuplicateGroupingInvariantError(
                    "cross_batch_duplicate_grouping_candidate_source_mismatch"
                )
            for fact_value_id, candidate in candidate_by_fact_value_id.items():
                authoritative_item = authoritative_items_by_fact_value_id[fact_value_id]
                if candidate.fact_id != authoritative_item.fact_id:
                    raise CrossBatchDuplicateGroupingInvariantError(
                        "cross_batch_duplicate_grouping_candidate_source_mismatch"
                    )
                if (
                    candidate.source_batch_id
                    != authoritative_source_batch_by_fact_value_id[fact_value_id]
                ):
                    raise CrossBatchDuplicateGroupingInvariantError(
                        "cross_batch_duplicate_grouping_candidate_source_mismatch"
                    )
                if candidate.evidence_ids != authoritative_item.evidence_ids:
                    raise CrossBatchDuplicateGroupingInvariantError(
                        "cross_batch_duplicate_grouping_candidate_source_mismatch"
                    )
        except duplicate_grouping_repository.DuplicateGroupingRepositoryInvariantError as error:
            await read_session.rollback()
            raise CrossBatchDuplicateGroupingInvariantError(str(error)) from None
        except fact_extraction_persistence_service.FactExtractionApplicationReplayConflictError:
            await read_session.rollback()
            raise CrossBatchDuplicateGroupingInvariantError(
                "cross_batch_duplicate_grouping_application_result_mismatch"
            ) from None
        except fact_extraction_persistence_service.FactExtractionPersistenceContextError:
            await read_session.rollback()
            raise CrossBatchDuplicateGroupingInvariantError(
                "cross_batch_duplicate_grouping_application_result_mismatch"
            ) from None
        except BaseException:
            await read_session.rollback()
            raise
        else:
            await read_session.rollback()

    return AuthenticatedDuplicateGroupingSourceSnapshot(
        state=state,
        candidate_count=candidate_count,
        candidates=candidates,
        application_snapshots=tuple(application_snapshots),
    )


async def _read_existing_consistency_candidate_result(
    session: AsyncSession,
    *,
    source_duplicate_grouping_application: DuplicateGroupingApplicationLedger,
    algorithm_version: str,
    plan: FactValueConsistencyCandidateWritePlan,
) -> FactValueConsistencyCandidateResult | None:
    existing_application = await duplicate_grouping_repository.get_consistency_candidate_application_ledger(
        session,
        duplicate_grouping_application_id=source_duplicate_grouping_application.id,
        algorithm_version=algorithm_version,
    )
    if existing_application is None:
        return None
    _assert_consistency_application_matches_plan(
        existing_application,
        duplicate_grouping_application_id=source_duplicate_grouping_application.id,
        orchestration_id=source_duplicate_grouping_application.orchestration_id,
        extraction_run_id=source_duplicate_grouping_application.extraction_run_id,
        plan=plan,
    )
    await _assert_consistency_candidate_subledgers_match_plan(
        session,
        application=existing_application,
        plan=plan,
    )
    return _build_consistency_candidate_result(existing_application, created_new=False)


async def list_fact_value_consistency_candidate_ledgers(
    session: AsyncSession,
    *,
    consistency_application_id: uuid.UUID,
) -> tuple[FactValueConsistencyCandidateLedger, ...]:
    return await duplicate_grouping_repository.list_consistency_candidate_ledgers(
        session,
        consistency_application_id=consistency_application_id,
    )


async def list_fact_value_consistency_candidate_member_ledgers(
    session: AsyncSession,
    *,
    consistency_application_id: uuid.UUID,
) -> tuple[FactValueConsistencyCandidateMemberLedger, ...]:
    return await duplicate_grouping_repository.list_consistency_candidate_member_ledgers(
        session,
        consistency_application_id=consistency_application_id,
    )


async def authenticate_fact_value_consistency_candidate_application(
    session_factory: Callable[[], AsyncSession],
    *,
    consistency_application_id: uuid.UUID,
) -> AuthenticatedFactValueConsistencyCandidateApplication:
    consistency_application_id = _require_consistency_uuid(
        consistency_application_id,
        field_name="consistency_application_id",
    )

    async with session_factory() as read_session:
        try:
            application = (
                await duplicate_grouping_repository.get_consistency_candidate_application_ledger_by_id(
                    read_session,
                    consistency_application_id=consistency_application_id,
                )
            )
            if application is None:
                raise FactValueConsistencyCandidateStateError(
                    "fact_value_consistency_candidate_application_not_found"
                )
            source_application = await duplicate_grouping_repository.get_grouping_application_ledger_by_id(
                read_session,
                grouping_application_id=application.duplicate_grouping_application_id,
            )
            if source_application is None:
                raise FactValueConsistencyCandidateStateError(
                    "fact_value_consistency_candidate_source_duplicate_grouping_not_found"
                )
            _normalize_supported_source_duplicate_grouping_algorithm_version(
                source_application.algorithm_version
            )
            state = await duplicate_grouping_repository.get_duplicate_grouping_orchestration_state(
                read_session,
                orchestration_id=application.orchestration_id,
            )
            _validate_run_state(state, orchestration_id=application.orchestration_id)
            if await duplicate_grouping_repository.has_invalid_completed_batch_bindings(
                read_session,
                orchestration_id=application.orchestration_id,
            ):
                raise FactValueConsistencyCandidateInvariantError(
                    "fact_value_consistency_candidate_completed_batch_binding_mismatch"
                )
            candidate_count = await duplicate_grouping_repository.count_duplicate_candidate_fact_values(
                read_session,
                orchestration_id=application.orchestration_id,
            )
            candidates = await duplicate_grouping_repository.list_duplicate_candidates(
                read_session,
                orchestration_id=application.orchestration_id,
            )
            if candidate_count != len(candidates):
                raise FactValueConsistencyCandidateInvariantError(
                    "fact_value_consistency_candidate_source_candidate_mismatch"
                )
            write_plan = build_fact_value_consistency_candidate_write_plan(
                candidates,
                source_duplicate_grouping_application=source_application,
                algorithm_version=application.algorithm_version,
            )
            _assert_consistency_application_matches_plan(
                application,
                duplicate_grouping_application_id=application.duplicate_grouping_application_id,
                orchestration_id=application.orchestration_id,
                extraction_run_id=application.extraction_run_id,
                plan=write_plan,
            )
            candidate_ledgers, member_ledgers = await _load_consistency_candidate_subledgers(
                read_session,
                consistency_application_id=application.id,
            )
            _compute_consistency_candidate_result_manifest_hash(
                candidate_ledgers,
                member_ledgers,
                application=application,
                plan=write_plan,
            )
        except duplicate_grouping_repository.DuplicateGroupingRepositoryInvariantError as error:
            await read_session.rollback()
            raise FactValueConsistencyCandidateInvariantError(str(error)) from None
        except CrossBatchDuplicateGroupingError as error:
            await read_session.rollback()
            raise FactValueConsistencyCandidateStateError(str(error)) from None
        except BaseException:
            await read_session.rollback()
            raise
        else:
            await read_session.rollback()

    return AuthenticatedFactValueConsistencyCandidateApplication(
        project_id=state.project_id,
        application=application,
        source_duplicate_grouping_application=source_application,
        write_plan=write_plan,
        candidate_ledgers=candidate_ledgers,
        member_ledgers=member_ledgers,
    )


async def ensure_cross_batch_multi_value_consistency_candidates(
    session_factory: Callable[[], AsyncSession],
    *,
    duplicate_grouping_application_id: uuid.UUID,
    algorithm_version: str = CROSS_BATCH_MULTI_VALUE_CANDIDATE_ALGORITHM_VERSION,
) -> FactValueConsistencyCandidateResult:
    duplicate_grouping_application_id = _require_consistency_uuid(
        duplicate_grouping_application_id,
        field_name="duplicate_grouping_application_id",
    )
    algorithm_version = normalize_consistency_candidate_algorithm_version(algorithm_version)

    async with session_factory() as read_session:
        try:
            source_application = await duplicate_grouping_repository.get_grouping_application_ledger_by_id(
                read_session,
                grouping_application_id=duplicate_grouping_application_id,
            )
            if source_application is None:
                raise FactValueConsistencyCandidateStateError(
                    "fact_value_consistency_candidate_source_duplicate_grouping_not_found"
                )
            _normalize_supported_source_duplicate_grouping_algorithm_version(
                source_application.algorithm_version
            )
            state = await duplicate_grouping_repository.get_duplicate_grouping_orchestration_state(
                read_session,
                orchestration_id=source_application.orchestration_id,
            )
            _validate_run_state(state, orchestration_id=source_application.orchestration_id)
            if await duplicate_grouping_repository.has_invalid_completed_batch_bindings(
                read_session,
                orchestration_id=source_application.orchestration_id,
            ):
                raise FactValueConsistencyCandidateInvariantError(
                    "fact_value_consistency_candidate_completed_batch_binding_mismatch"
                )
            candidate_count = await duplicate_grouping_repository.count_duplicate_candidate_fact_values(
                read_session,
                orchestration_id=source_application.orchestration_id,
            )
            candidates = await duplicate_grouping_repository.list_duplicate_candidates(
                read_session,
                orchestration_id=source_application.orchestration_id,
            )
            if candidate_count != len(candidates):
                raise FactValueConsistencyCandidateInvariantError(
                    "fact_value_consistency_candidate_source_candidate_mismatch"
                )
        except duplicate_grouping_repository.DuplicateGroupingRepositoryInvariantError as error:
            await read_session.rollback()
            raise FactValueConsistencyCandidateInvariantError(str(error)) from None
        except CrossBatchDuplicateGroupingError as error:
            await read_session.rollback()
            raise FactValueConsistencyCandidateStateError(str(error)) from None
        except BaseException:
            await read_session.rollback()
            raise
        else:
            await read_session.rollback()

    write_plan = build_fact_value_consistency_candidate_write_plan(
        candidates,
        source_duplicate_grouping_application=source_application,
        algorithm_version=algorithm_version,
    )

    async with session_factory() as write_session:
        try:
            current_source_application = await duplicate_grouping_repository.get_grouping_application_ledger_by_id(
                write_session,
                grouping_application_id=duplicate_grouping_application_id,
            )
            if current_source_application is None:
                raise FactValueConsistencyCandidateStateError(
                    "fact_value_consistency_candidate_source_duplicate_grouping_not_found"
                )
            _assert_source_duplicate_grouping_application_matches_snapshot(
                source_application,
                current_source_application,
            )
            state = await duplicate_grouping_repository.get_duplicate_grouping_orchestration_state(
                write_session,
                orchestration_id=source_application.orchestration_id,
            )
            _validate_run_state(state, orchestration_id=source_application.orchestration_id)
            if await duplicate_grouping_repository.has_invalid_completed_batch_bindings(
                write_session,
                orchestration_id=source_application.orchestration_id,
            ):
                raise FactValueConsistencyCandidateInvariantError(
                    "fact_value_consistency_candidate_completed_batch_binding_mismatch"
                )

            existing_application = await duplicate_grouping_repository.get_consistency_candidate_application_for_update(
                write_session,
                duplicate_grouping_application_id=duplicate_grouping_application_id,
                algorithm_version=algorithm_version,
            )
            if existing_application is not None:
                existing_ledger = FactValueConsistencyCandidateApplicationLedger(
                    id=existing_application.id,
                    duplicate_grouping_application_id=existing_application.duplicate_grouping_application_id,
                    orchestration_id=existing_application.orchestration_id,
                    extraction_run_id=existing_application.extraction_run_id,
                    algorithm_version=existing_application.algorithm_version,
                    input_manifest_hash=existing_application.input_manifest_hash,
                    result_manifest_hash=existing_application.result_manifest_hash,
                    candidate_count=existing_application.candidate_count,
                    member_count=existing_application.member_count,
                    created_at=existing_application.created_at,
                )
                _assert_consistency_application_matches_plan(
                    existing_ledger,
                    duplicate_grouping_application_id=duplicate_grouping_application_id,
                    orchestration_id=source_application.orchestration_id,
                    extraction_run_id=source_application.extraction_run_id,
                    plan=write_plan,
                )
                await _assert_consistency_candidate_subledgers_match_plan(
                    write_session,
                    application=existing_ledger,
                    plan=write_plan,
                )
                await write_session.commit()
                return _build_consistency_candidate_result(existing_ledger, created_new=False)

            application = FactValueConsistencyCandidateApplication(
                id=uuid.uuid4(),
                duplicate_grouping_application_id=duplicate_grouping_application_id,
                orchestration_id=source_application.orchestration_id,
                extraction_run_id=source_application.extraction_run_id,
                algorithm_version=algorithm_version,
                input_manifest_hash=write_plan.input_manifest_hash,
                result_manifest_hash=write_plan.result_manifest_hash,
                candidate_count=write_plan.candidate_count,
                member_count=write_plan.member_count,
            )
            await duplicate_grouping_repository.create_consistency_candidate_application(
                write_session,
                application,
            )

            candidate_rows: list[FactValueConsistencyCandidate] = []
            candidate_id_by_fact_id: dict[uuid.UUID, uuid.UUID] = {}
            for candidate_plan in write_plan.candidates:
                candidate_id = uuid.uuid4()
                candidate_id_by_fact_id[candidate_plan.fact_id] = candidate_id
                candidate_rows.append(
                    FactValueConsistencyCandidate(
                        id=candidate_id,
                        consistency_application_id=application.id,
                        fact_id=candidate_plan.fact_id,
                        candidate_kind=candidate_plan.candidate_kind,
                        member_count=candidate_plan.member_count,
                        distinct_semantic_key_count=candidate_plan.distinct_semantic_key_count,
                        distinct_batch_count=candidate_plan.distinct_batch_count,
                    )
                )
            if candidate_rows:
                await duplicate_grouping_repository.create_consistency_candidates(
                    write_session,
                    candidate_rows,
                )

            member_rows: list[FactValueConsistencyCandidateMember] = []
            for candidate_plan in write_plan.candidates:
                candidate_id = candidate_id_by_fact_id[candidate_plan.fact_id]
                for member_plan in candidate_plan.members:
                    member_rows.append(
                        FactValueConsistencyCandidateMember(
                            id=uuid.uuid4(),
                            consistency_application_id=application.id,
                            candidate_id=candidate_id,
                            orchestration_id=application.orchestration_id,
                            fact_value_id=member_plan.fact_value_id,
                            source_batch_id=member_plan.source_batch_id,
                            semantic_key_hash=member_plan.semantic_key_hash,
                        )
                    )
            if member_rows:
                await duplicate_grouping_repository.create_consistency_candidate_members(
                    write_session,
                    member_rows,
                )

            if len(candidate_rows) != write_plan.candidate_count:
                raise FactValueConsistencyCandidateInvariantError(
                    "fact_value_consistency_candidate_write_count_mismatch"
                )
            if len(member_rows) != write_plan.member_count:
                raise FactValueConsistencyCandidateInvariantError(
                    "fact_value_consistency_candidate_write_count_mismatch"
                )

            await write_session.commit()
            application_ledger = FactValueConsistencyCandidateApplicationLedger(
                id=application.id,
                duplicate_grouping_application_id=application.duplicate_grouping_application_id,
                orchestration_id=application.orchestration_id,
                extraction_run_id=application.extraction_run_id,
                algorithm_version=application.algorithm_version,
                input_manifest_hash=application.input_manifest_hash,
                result_manifest_hash=application.result_manifest_hash,
                candidate_count=application.candidate_count,
                member_count=application.member_count,
                created_at=application.created_at,
            )
            return _build_consistency_candidate_result(application_ledger, created_new=True)
        except IntegrityError as error:
            constraint_name = _get_integrity_constraint_name(error)
            await write_session.rollback()
            if constraint_name != _CONSISTENCY_APPLICATION_UNIQUE_CONSTRAINT:
                raise
        except BaseException:
            await write_session.rollback()
            raise

    async with session_factory() as read_session:
        try:
            existing_result = await _read_existing_consistency_candidate_result(
                read_session,
                source_duplicate_grouping_application=source_application,
                algorithm_version=algorithm_version,
                plan=write_plan,
            )
        except BaseException:
            await read_session.rollback()
            raise
        else:
            await read_session.rollback()
    if existing_result is None:
        raise FactValueConsistencyCandidateInvariantError(
            "fact_value_consistency_candidate_concurrent_ledger_missing"
        )
    return existing_result
