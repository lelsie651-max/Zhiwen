from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
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
    FactValueDuplicateGroup,
    FactValueDuplicateGroupMember,
    FactValueDuplicateGroupingApplication,
)
from app.repositories import fact_value_duplicate_grouping as duplicate_grouping_repository
from app.schemas.fact_value_duplicate_grouping import (
    CROSS_BATCH_DUPLICATE_ALGORITHM_VERSION,
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
)


logger = logging.getLogger(__name__)

_APPLICATION_UNIQUE_CONSTRAINT = "uq_dupgrp_app_orch_alg"
_MISSING = object()


class CrossBatchDuplicateGroupingError(Exception):
    """Base class for duplicate grouping failures."""


class CrossBatchDuplicateGroupingStateError(CrossBatchDuplicateGroupingError):
    """Raised when the extraction run is not ready for duplicate grouping."""


class CrossBatchDuplicateGroupingInvariantError(CrossBatchDuplicateGroupingError):
    """Raised when immutable duplicate-grouping invariants are violated."""


def _require_uuid(value: uuid.UUID, *, field_name: str) -> uuid.UUID:
    if not isinstance(value, uuid.UUID):
        raise CrossBatchDuplicateGroupingError(f"{field_name} must be a UUID")
    return value


def _normalize_string(value: str) -> str:
    return unicodedata.normalize("NFC", value)


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
            normalized_items[_normalize_string(key)] = normalized_item
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


def build_duplicate_fingerprint(
    candidate: DuplicateCandidate,
    *,
    algorithm_version: str = CROSS_BATCH_DUPLICATE_ALGORITHM_VERSION,
    digest_bytes_by_hash: dict[str, bytes] | None = None,
) -> DuplicateFingerprint:
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


def build_duplicate_grouping_write_plan(
    candidates: Sequence[DuplicateCandidate],
    *,
    algorithm_version: str = CROSS_BATCH_DUPLICATE_ALGORITHM_VERSION,
) -> DuplicateGroupingWritePlan:
    digest_bytes_by_hash: dict[str, bytes] = {}
    fingerprint_by_fact_value_id: dict[uuid.UUID, DuplicateFingerprint] = {}
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
        fingerprint_by_fact_value_id[candidate.fact_value_id] = build_duplicate_fingerprint(
            candidate,
            algorithm_version=algorithm_version,
            digest_bytes_by_hash=digest_bytes_by_hash,
        )

    input_manifest_entries = [
        {
            "duplicate_key_hash": fingerprint_by_fact_value_id[candidate.fact_value_id].sha256_hex,
            "evidence_link_ids": [str(evidence_link_id) for evidence_link_id in candidate.evidence_link_ids],
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
    if not isinstance(algorithm_version, str) or not algorithm_version:
        raise CrossBatchDuplicateGroupingError("algorithm_version must be a non-empty string")

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
        except BaseException:
            await read_session.rollback()
            raise
        else:
            await read_session.rollback()

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
