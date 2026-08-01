from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
import hashlib
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories import consistency_check as consistency_check_repository
from app.schemas.consistency_check import (
    CONSISTENCY_CHECK_PLANNER_NAME,
    CONSISTENCY_CHECK_PLANNER_VERSION,
    ConsistencyCheckBatchPlan,
    ConsistencyCheckCandidateBundle,
    ConsistencyCheckEvidenceBundle,
    ConsistencyCheckMemberBundle,
    ConsistencyCheckPlan,
    ConsistencyCheckPlannerConfig,
)
from app.services import fact_value_duplicate_grouping as duplicate_grouping_service


class ConsistencyCheckPlanError(Exception):
    """Base class for consistency check planning failures."""


class ConsistencyCheckPlanStateError(ConsistencyCheckPlanError):
    """Raised when the consistency application or planner config is invalid."""


class ConsistencyCheckPlanInvariantError(ConsistencyCheckPlanError):
    """Raised when authenticated ledgers and evidence rows diverge."""


def _require_uuid(value: uuid.UUID, *, field_name: str) -> uuid.UUID:
    if not isinstance(value, uuid.UUID):
        raise ConsistencyCheckPlanError(f"{field_name} must be a UUID")
    return value


def _normalize_config(config: ConsistencyCheckPlannerConfig) -> ConsistencyCheckPlannerConfig:
    if not isinstance(config, ConsistencyCheckPlannerConfig):
        raise ConsistencyCheckPlanStateError("consistency_check_plan_invalid_config")
    return config


def _build_candidate_manifest_payload(
    candidate: ConsistencyCheckCandidateBundle,
) -> dict[str, object]:
    return {
        "candidate_id": str(candidate.candidate_id),
        "candidate_kind": candidate.candidate_kind,
        "fact_id": str(candidate.fact_id),
        "members": [
            {
                "fact_value_id": str(member.fact_value_id),
                "semantic_key_hash": member.semantic_key_hash,
                "source_batch_id": str(member.source_batch_id),
                "value_type": member.value_type,
                "value_json": member.value_json,
                "referenced_entity_id": (
                    None if member.referenced_entity_id is None else str(member.referenced_entity_id)
                ),
                "evidences": [
                    {
                        "evidence_content_hash": evidence.evidence_content_hash,
                        "evidence_id": str(evidence.evidence_id),
                        "evidence_link_id": str(evidence.evidence_link_id),
                        "role": evidence.role,
                        "is_primary": evidence.is_primary,
                        "source_order": evidence.source_order,
                        "document_block_id": str(evidence.document_block_id),
                        "location_key": evidence.location_key,
                        "page_no": evidence.page_no,
                        "start_line": evidence.start_line,
                        "end_line": evidence.end_line,
                        "start_offset": evidence.start_offset,
                        "end_offset": evidence.end_offset,
                    }
                    for evidence in member.evidences
                ],
            }
            for member in candidate.members
        ],
    }


def _candidate_evidence_character_count(candidate: ConsistencyCheckCandidateBundle) -> int:
    return sum(len(evidence.excerpt) for member in candidate.members for evidence in member.evidences)


def _hash_excerpt_text(excerpt: str) -> str:
    return hashlib.sha256(excerpt.encode("utf-8")).hexdigest()


def _build_candidate_bundles(
    authenticated: duplicate_grouping_service.AuthenticatedFactValueConsistencyCandidateApplication,
    rows: Sequence[consistency_check_repository.ConsistencyCheckCandidateRow],
) -> tuple[ConsistencyCheckCandidateBundle, ...]:
    application = authenticated.application
    candidate_ledger_by_id = {candidate.id: candidate for candidate in authenticated.candidate_ledgers}
    member_ledger_by_id = {member.id: member for member in authenticated.member_ledgers}
    expected_candidate_ids = set(candidate_ledger_by_id)
    expected_member_ids = set(member_ledger_by_id)

    candidate_order: list[uuid.UUID] = []
    seen_candidate_ids: set[uuid.UUID] = set()
    seen_member_ids: set[uuid.UUID] = set()
    seen_link_owner_by_id: dict[uuid.UUID, uuid.UUID] = {}
    candidate_builders: dict[uuid.UUID, dict[str, object]] = {}

    def get_candidate_builder(
        row: consistency_check_repository.ConsistencyCheckCandidateRow,
    ) -> dict[str, object]:
        candidate_builder = candidate_builders.get(row.candidate_id)
        if candidate_builder is not None:
            return candidate_builder
        candidate_builder = {
            "candidate_id": row.candidate_id,
            "fact_id": row.candidate_fact_id,
            "candidate_kind": row.candidate_kind,
            "member_order": [],
            "members": {},
        }
        candidate_builders[row.candidate_id] = candidate_builder
        candidate_order.append(row.candidate_id)
        return candidate_builder

    def get_member_builder(
        candidate_builder: dict[str, object],
        row: consistency_check_repository.ConsistencyCheckCandidateRow,
    ) -> dict[str, object]:
        members_by_id = candidate_builder["members"]
        member_builder = members_by_id.get(row.member_id)
        if member_builder is not None:
            return member_builder
        member_builder = {
            "member_id": row.member_id,
            "fact_value_id": row.member_fact_value_id,
            "source_batch_id": row.member_source_batch_id,
            "semantic_key_hash": row.member_semantic_key_hash,
            "value_type": row.fact_value_value_type,
            "value_json": row.fact_value_value_json,
            "referenced_entity_id": row.fact_value_referenced_entity_id,
            "evidences": [],
            "seen_link_ids": set(),
        }
        members_by_id[row.member_id] = member_builder
        candidate_builder["member_order"].append(row.member_id)
        return member_builder

    for row in rows:
        candidate_ledger = candidate_ledger_by_id.get(row.candidate_id)
        member_ledger = member_ledger_by_id.get(row.member_id)
        if candidate_ledger is None or member_ledger is None:
            raise ConsistencyCheckPlanInvariantError("consistency_check_plan_bundle_row_mismatch")
        if row.consistency_application_id != application.id:
            raise ConsistencyCheckPlanInvariantError("consistency_check_plan_bundle_row_mismatch")
        if row.member_consistency_application_id != application.id:
            raise ConsistencyCheckPlanInvariantError("consistency_check_plan_bundle_row_mismatch")
        if row.member_candidate_id != row.candidate_id:
            raise ConsistencyCheckPlanInvariantError("consistency_check_plan_bundle_row_mismatch")
        if candidate_ledger.consistency_application_id != application.id:
            raise ConsistencyCheckPlanInvariantError("consistency_check_plan_bundle_row_mismatch")
        if member_ledger.consistency_application_id != application.id:
            raise ConsistencyCheckPlanInvariantError("consistency_check_plan_bundle_row_mismatch")
        if member_ledger.candidate_id != row.candidate_id:
            raise ConsistencyCheckPlanInvariantError("consistency_check_plan_bundle_row_mismatch")
        if row.candidate_fact_id != candidate_ledger.fact_id:
            raise ConsistencyCheckPlanInvariantError("consistency_check_plan_bundle_row_mismatch")
        if row.candidate_kind != candidate_ledger.candidate_kind:
            raise ConsistencyCheckPlanInvariantError("consistency_check_plan_bundle_row_mismatch")
        if row.member_fact_value_id != member_ledger.fact_value_id:
            raise ConsistencyCheckPlanInvariantError("consistency_check_plan_bundle_row_mismatch")
        if row.member_source_batch_id != member_ledger.source_batch_id:
            raise ConsistencyCheckPlanInvariantError("consistency_check_plan_bundle_row_mismatch")
        if row.member_semantic_key_hash != member_ledger.semantic_key_hash:
            raise ConsistencyCheckPlanInvariantError("consistency_check_plan_bundle_row_mismatch")
        if row.member_orchestration_id != application.orchestration_id:
            raise ConsistencyCheckPlanInvariantError("consistency_check_plan_member_batch_mismatch")
        if row.batch_orchestration_id != application.orchestration_id:
            raise ConsistencyCheckPlanInvariantError("consistency_check_plan_member_batch_mismatch")
        if row.fact_value_fact_id != row.candidate_fact_id:
            raise ConsistencyCheckPlanInvariantError("consistency_check_plan_member_fact_mismatch")
        if row.evidence_link_id is None or row.evidence_id is None:
            raise ConsistencyCheckPlanInvariantError("consistency_check_plan_member_evidence_missing")
        if row.evidence_link_fact_value_id != row.member_fact_value_id:
            raise ConsistencyCheckPlanInvariantError("consistency_check_plan_evidence_link_binding_mismatch")
        if row.block_id is None or row.excerpt is None or row.excerpt_hash is None:
            raise ConsistencyCheckPlanInvariantError("consistency_check_plan_member_evidence_missing")
        if row.location_key is None:
            raise ConsistencyCheckPlanInvariantError("consistency_check_plan_member_evidence_missing")
        if row.evidence_role is None:
            raise ConsistencyCheckPlanInvariantError("consistency_check_plan_member_evidence_missing")
        if row.evidence_is_primary is None:
            raise ConsistencyCheckPlanInvariantError("consistency_check_plan_member_evidence_missing")
        if row.evidence_source_order is None:
            raise ConsistencyCheckPlanInvariantError("consistency_check_plan_member_evidence_missing")
        if row.start_offset is None:
            raise ConsistencyCheckPlanInvariantError("consistency_check_plan_member_evidence_missing")
        if row.end_offset is None:
            raise ConsistencyCheckPlanInvariantError("consistency_check_plan_member_evidence_missing")
        if _hash_excerpt_text(row.excerpt) != row.excerpt_hash:
            raise ConsistencyCheckPlanInvariantError("consistency_check_plan_evidence_hash_mismatch")

        seen_candidate_ids.add(row.candidate_id)
        seen_member_ids.add(row.member_id)
        link_owner = seen_link_owner_by_id.get(row.evidence_link_id)
        if link_owner is not None and link_owner != row.member_id:
            raise ConsistencyCheckPlanInvariantError("consistency_check_plan_cross_member_link_reuse")
        seen_link_owner_by_id.setdefault(row.evidence_link_id, row.member_id)

        candidate_builder = get_candidate_builder(row)
        member_builder = get_member_builder(candidate_builder, row)
        if row.evidence_link_id in member_builder["seen_link_ids"]:
            continue
        member_builder["seen_link_ids"].add(row.evidence_link_id)
        member_builder["evidences"].append(
            ConsistencyCheckEvidenceBundle(
                evidence_link_id=row.evidence_link_id,
                evidence_id=row.evidence_id,
                role=row.evidence_role,
                is_primary=row.evidence_is_primary,
                source_order=row.evidence_source_order,
                document_block_id=row.block_id,
                location_key=row.location_key,
                page_no=row.page_no,
                start_line=row.start_line,
                end_line=row.end_line,
                start_offset=row.start_offset,
                end_offset=row.end_offset,
                excerpt=row.excerpt,
                evidence_content_hash=row.excerpt_hash,
            )
        )

    if seen_candidate_ids != expected_candidate_ids:
        raise ConsistencyCheckPlanInvariantError("consistency_check_plan_bundle_row_mismatch")
    if seen_member_ids != expected_member_ids:
        raise ConsistencyCheckPlanInvariantError("consistency_check_plan_bundle_row_mismatch")

    candidate_bundles: list[ConsistencyCheckCandidateBundle] = []
    for candidate_id in candidate_order:
        candidate_builder = candidate_builders[candidate_id]
        members: list[ConsistencyCheckMemberBundle] = []
        for member_id in candidate_builder["member_order"]:
            member_builder = candidate_builder["members"][member_id]
            evidences = tuple(member_builder["evidences"])
            if not evidences:
                raise ConsistencyCheckPlanInvariantError("consistency_check_plan_member_evidence_missing")
            members.append(
                ConsistencyCheckMemberBundle(
                    fact_value_id=member_builder["fact_value_id"],
                    source_batch_id=member_builder["source_batch_id"],
                    semantic_key_hash=member_builder["semantic_key_hash"],
                    value_type=member_builder["value_type"],
                    value_json=member_builder["value_json"],
                    referenced_entity_id=member_builder["referenced_entity_id"],
                    evidences=evidences,
                )
            )
        ordered_members = tuple(
            sorted(
                members,
                key=lambda item: (
                    item.semantic_key_hash,
                    str(item.source_batch_id),
                    str(item.fact_value_id),
                ),
            )
        )
        candidate_bundles.append(
            ConsistencyCheckCandidateBundle(
                candidate_id=candidate_builder["candidate_id"],
                fact_id=candidate_builder["fact_id"],
                candidate_kind=candidate_builder["candidate_kind"],
                members=ordered_members,
            )
        )

    return tuple(
        sorted(
            candidate_bundles,
            key=lambda item: (str(item.fact_id), str(item.candidate_id)),
        )
    )


def _build_consistency_check_batches(
    *,
    consistency_application_id: uuid.UUID,
    source_result_manifest_hash: str,
    config: ConsistencyCheckPlannerConfig,
    candidate_bundles: Sequence[ConsistencyCheckCandidateBundle],
) -> tuple[ConsistencyCheckBatchPlan, ...]:
    digest_bytes_by_hash: dict[str, bytes] = {}
    candidate_manifest_hash_by_id: dict[uuid.UUID, str] = {}
    evidence_character_count_by_id: dict[uuid.UUID, int] = {}

    for candidate in candidate_bundles:
        candidate_manifest_hash_by_id[candidate.candidate_id] = duplicate_grouping_service.hash_deterministic_payload(
            _build_candidate_manifest_payload(candidate),
            digest_bytes_by_hash=digest_bytes_by_hash,
        )
        evidence_character_count_by_id[candidate.candidate_id] = _candidate_evidence_character_count(candidate)

    batches: list[ConsistencyCheckBatchPlan] = []
    current_candidates: list[ConsistencyCheckCandidateBundle] = []
    current_character_count = 0

    def flush_batch() -> None:
        nonlocal current_candidates, current_character_count
        if not current_candidates:
            return
        batch_index = len(batches)
        batch_manifest_hash = duplicate_grouping_service.hash_deterministic_payload(
            {
                "consistency_application_id": str(consistency_application_id),
                "source_result_manifest_hash": source_result_manifest_hash,
                "batch_index": batch_index,
                "candidate_count": len(current_candidates),
                "candidate_ids": [str(candidate.candidate_id) for candidate in current_candidates],
                "candidate_manifest_hashes": [
                    candidate_manifest_hash_by_id[candidate.candidate_id] for candidate in current_candidates
                ],
                "evidence_character_count": current_character_count,
            },
            digest_bytes_by_hash=digest_bytes_by_hash,
        )
        batches.append(
            ConsistencyCheckBatchPlan(
                batch_index=batch_index,
                candidate_ids=tuple(candidate.candidate_id for candidate in current_candidates),
                candidate_count=len(current_candidates),
                evidence_character_count=current_character_count,
                batch_manifest_hash=batch_manifest_hash,
                candidates=tuple(current_candidates),
            )
        )
        current_candidates = []
        current_character_count = 0

    for candidate in candidate_bundles:
        candidate_character_count = evidence_character_count_by_id[candidate.candidate_id]
        if candidate_character_count > config.max_evidence_characters_per_batch:
            raise ConsistencyCheckPlanStateError("consistency_check_plan_candidate_too_large")
        would_exceed_candidate_count = len(current_candidates) >= config.max_candidates_per_batch
        would_exceed_character_count = (
            bool(current_candidates)
            and current_character_count + candidate_character_count > config.max_evidence_characters_per_batch
        )
        if would_exceed_candidate_count or would_exceed_character_count:
            flush_batch()
        current_candidates.append(candidate)
        current_character_count += candidate_character_count
    flush_batch()

    authenticated_candidate_count = len(candidate_bundles)
    if not batches and authenticated_candidate_count:
        raise ConsistencyCheckPlanInvariantError("consistency_check_plan_batch_construction_mismatch")
    if not batches and authenticated_candidate_count == 0:
        batch_manifest_hash = duplicate_grouping_service.hash_deterministic_payload(
            {
                "consistency_application_id": str(consistency_application_id),
                "source_result_manifest_hash": source_result_manifest_hash,
                "batch_index": 0,
                "candidate_count": 0,
                "candidate_ids": [],
                "candidate_manifest_hashes": [],
                "evidence_character_count": 0,
            },
            digest_bytes_by_hash=digest_bytes_by_hash,
        )
        batches.append(
            ConsistencyCheckBatchPlan(
                batch_index=0,
                candidate_ids=(),
                candidate_count=0,
                evidence_character_count=0,
                batch_manifest_hash=batch_manifest_hash,
                candidates=(),
            )
        )

    return tuple(batches)


async def build_consistency_check_plan(
    session_factory: Callable[[], AsyncSession],
    *,
    consistency_application_id: uuid.UUID,
    config: ConsistencyCheckPlannerConfig,
) -> ConsistencyCheckPlan:
    consistency_application_id = _require_uuid(
        consistency_application_id,
        field_name="consistency_application_id",
    )
    config = _normalize_config(config)
    authenticated = await duplicate_grouping_service.authenticate_fact_value_consistency_candidate_application(
        session_factory,
        consistency_application_id=consistency_application_id,
    )

    async with session_factory() as read_session:
        try:
            rows = await consistency_check_repository.list_consistency_check_candidate_rows(
                read_session,
                consistency_application_id=consistency_application_id,
            )
        except BaseException:
            await read_session.rollback()
            raise
        else:
            await read_session.rollback()

    candidate_bundles = _build_candidate_bundles(authenticated, rows)
    batches = _build_consistency_check_batches(
        consistency_application_id=consistency_application_id,
        source_result_manifest_hash=authenticated.application.result_manifest_hash,
        config=config,
        candidate_bundles=candidate_bundles,
    )
    plan_manifest_hash = duplicate_grouping_service.hash_deterministic_payload(
        {
            "project_id": str(authenticated.project_id),
            "consistency_application_id": str(consistency_application_id),
            "source_result_manifest_hash": authenticated.application.result_manifest_hash,
            "planner_name": CONSISTENCY_CHECK_PLANNER_NAME,
            "planner_version": CONSISTENCY_CHECK_PLANNER_VERSION,
            "config": asdict(config),
            "batches": [
                {
                    "batch_index": batch.batch_index,
                    "candidate_ids": [str(candidate_id) for candidate_id in batch.candidate_ids],
                    "candidate_count": batch.candidate_count,
                    "evidence_character_count": batch.evidence_character_count,
                    "batch_manifest_hash": batch.batch_manifest_hash,
                }
                for batch in batches
            ],
        },
    )
    return ConsistencyCheckPlan(
        project_id=authenticated.project_id,
        consistency_application_id=consistency_application_id,
        source_result_manifest_hash=authenticated.application.result_manifest_hash,
        planner_name=CONSISTENCY_CHECK_PLANNER_NAME,
        planner_version=CONSISTENCY_CHECK_PLANNER_VERSION,
        config=config,
        batches=batches,
        plan_manifest_hash=plan_manifest_hash,
    )
