from __future__ import annotations

import copy
from dataclasses import dataclass
import uuid

from sqlalchemy import case, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_content import ExtractionRun as DocumentExtractionRun
from app.models.document_revision import DocumentRevision
from app.models.fact import Fact, FactEvidenceLink, FactValue, FactValueSourceKind
from app.models.fact_extraction_application import FactExtractionBatchApplication
from app.models.fact_extraction_orchestration import (
    FactExtractionOrchestration,
    FactExtractionOrchestrationBatch,
)
from app.models.fact_value_duplicate_grouping import (
    FactValueConsistencyCandidate,
    FactValueConsistencyCandidateApplication,
    FactValueConsistencyCandidateMember,
    FactValueDuplicateGroup,
    FactValueDuplicateGroupMember,
    FactValueDuplicateGroupingApplication,
)
from app.schemas.fact_value_duplicate_grouping import (
    DuplicateCandidate,
    DuplicateGroupEvidenceProjection,
    DuplicateGroupLedger,
    DuplicateGroupMemberLedger,
    DuplicateGroupingApplicationLedger,
    FactValueConsistencyCandidateApplicationLedger,
    FactValueConsistencyCandidateLedger,
    FactValueConsistencyCandidateMemberLedger,
)


class DuplicateGroupingRepositoryInvariantError(Exception):
    """Raised when duplicate grouping repository rows violate runtime invariants."""


@dataclass(frozen=True, slots=True)
class DuplicateGroupingOrchestrationState:
    orchestration_id: uuid.UUID
    extraction_run_id: uuid.UUID
    project_id: uuid.UUID
    extraction_run_status: str
    extraction_run_outcome: str | None
    orchestration_status: str


async def get_duplicate_grouping_orchestration_state(
    session: AsyncSession,
    *,
    orchestration_id: uuid.UUID,
) -> DuplicateGroupingOrchestrationState | None:
    result = await session.execute(
        select(
            FactExtractionOrchestration.id.label("orchestration_id"),
            FactExtractionOrchestration.extraction_run_id.label("extraction_run_id"),
            FactExtractionOrchestration.project_id.label("project_id"),
            FactExtractionOrchestration.status.label("orchestration_status"),
            DocumentExtractionRun.status.label("extraction_run_status"),
            DocumentExtractionRun.outcome.label("extraction_run_outcome"),
        )
        .select_from(FactExtractionOrchestration)
        .join(DocumentExtractionRun, FactExtractionOrchestration.extraction_run_id == DocumentExtractionRun.id)
        .join(DocumentRevision, DocumentExtractionRun.revision_id == DocumentRevision.id)
        .join(Document, DocumentRevision.document_id == Document.id)
        .where(FactExtractionOrchestration.id == orchestration_id)
    )
    row = result.one_or_none()
    if row is None:
        return None
    return DuplicateGroupingOrchestrationState(
        orchestration_id=row.orchestration_id,
        extraction_run_id=row.extraction_run_id,
        project_id=row.project_id,
        extraction_run_status=row.extraction_run_status,
        extraction_run_outcome=row.extraction_run_outcome,
        orchestration_status=row.orchestration_status,
    )


async def has_invalid_completed_batch_bindings(
    session: AsyncSession,
    *,
    orchestration_id: uuid.UUID,
) -> bool:
    result = await session.execute(
        select(FactExtractionOrchestrationBatch.id)
        .select_from(FactExtractionOrchestrationBatch)
        .join(FactExtractionOrchestration, FactExtractionOrchestrationBatch.orchestration_id == FactExtractionOrchestration.id)
        .outerjoin(
            FactExtractionBatchApplication,
            FactExtractionOrchestrationBatch.application_id == FactExtractionBatchApplication.id,
        )
        .where(
            FactExtractionOrchestrationBatch.orchestration_id == orchestration_id,
            FactExtractionOrchestrationBatch.status == "completed",
            case(
                (
                    FactExtractionBatchApplication.id.is_(None),
                    True,
                ),
                (
                    FactExtractionBatchApplication.status != "completed",
                    True,
                ),
                (
                    FactExtractionBatchApplication.extraction_run_id != FactExtractionOrchestration.extraction_run_id,
                    True,
                ),
                (
                    FactExtractionOrchestrationBatch.current_inference_run_id
                    != FactExtractionBatchApplication.inference_run_id,
                    True,
                ),
                else_=False,
            ),
        )
        .limit(1)
    )
    return result.first() is not None


async def list_duplicate_candidates(
    session: AsyncSession,
    *,
    orchestration_id: uuid.UUID,
) -> tuple[DuplicateCandidate, ...]:
    result = await session.execute(
        select(
            FactValue.id.label("fact_value_id"),
            FactValue.fact_id.label("fact_id"),
            FactExtractionOrchestration.id.label("orchestration_id"),
            FactValue.extraction_run_id.label("extraction_run_id"),
            FactExtractionOrchestrationBatch.id.label("source_batch_id"),
            FactValue.value_type.label("value_type"),
            FactValue.value_json.label("value_json"),
            FactValue.referenced_entity_id.label("referenced_entity_id"),
            FactEvidenceLink.id.label("evidence_link_id"),
        )
        .select_from(FactExtractionOrchestrationBatch)
        .join(FactExtractionOrchestration, FactExtractionOrchestrationBatch.orchestration_id == FactExtractionOrchestration.id)
        .join(
            FactExtractionBatchApplication,
            FactExtractionOrchestrationBatch.application_id == FactExtractionBatchApplication.id,
        )
        .join(FactValue, FactValue.inference_run_id == FactExtractionBatchApplication.inference_run_id)
        .join(Fact, FactValue.fact_id == Fact.id)
        .outerjoin(FactEvidenceLink, FactEvidenceLink.fact_value_id == FactValue.id)
        .where(
            FactExtractionOrchestrationBatch.orchestration_id == orchestration_id,
            FactExtractionOrchestrationBatch.status == "completed",
            FactExtractionBatchApplication.status == "completed",
            FactExtractionBatchApplication.extraction_run_id == FactExtractionOrchestration.extraction_run_id,
            FactExtractionBatchApplication.inference_run_id == FactExtractionOrchestrationBatch.current_inference_run_id,
            FactValue.extraction_run_id == FactExtractionOrchestration.extraction_run_id,
            FactValue.source_kind == FactValueSourceKind.AI.value,
        )
        .order_by(
            FactValue.id.asc(),
            FactEvidenceLink.source_order.asc().nulls_last(),
            FactEvidenceLink.id.asc().nulls_last(),
        )
    )
    rows = list(result.all())
    if not rows:
        return ()

    candidates: list[DuplicateCandidate] = []
    current_fact_value_id: uuid.UUID | None = None
    current_evidence_link_ids: set[uuid.UUID] = set()
    current_candidate_fields: dict[str, object] | None = None

    def flush_current() -> None:
        nonlocal current_fact_value_id, current_evidence_link_ids, current_candidate_fields
        if current_candidate_fields is None:
            return
        candidates.append(
            DuplicateCandidate(
                fact_value_id=current_candidate_fields["fact_value_id"],
                fact_id=current_candidate_fields["fact_id"],
                orchestration_id=current_candidate_fields["orchestration_id"],
                extraction_run_id=current_candidate_fields["extraction_run_id"],
                source_batch_id=current_candidate_fields["source_batch_id"],
                value_type=current_candidate_fields["value_type"],
                value_json=copy.deepcopy(current_candidate_fields["value_json"]),
                referenced_entity_id=current_candidate_fields["referenced_entity_id"],
                evidence_link_ids=tuple(sorted(current_evidence_link_ids, key=str)),
            )
        )
        current_fact_value_id = None
        current_evidence_link_ids = set()
        current_candidate_fields = None

    for row in rows:
        if current_fact_value_id != row.fact_value_id:
            flush_current()
            current_fact_value_id = row.fact_value_id
            current_candidate_fields = {
                "fact_value_id": row.fact_value_id,
                "fact_id": row.fact_id,
                "orchestration_id": row.orchestration_id,
                "extraction_run_id": row.extraction_run_id,
                "source_batch_id": row.source_batch_id,
                "value_type": row.value_type,
                "value_json": row.value_json,
                "referenced_entity_id": row.referenced_entity_id,
            }
        elif current_candidate_fields is not None:
            stable_row_fields = {
                "fact_id": row.fact_id,
                "orchestration_id": row.orchestration_id,
                "extraction_run_id": row.extraction_run_id,
                "source_batch_id": row.source_batch_id,
                "value_type": row.value_type,
                "value_json": row.value_json,
                "referenced_entity_id": row.referenced_entity_id,
            }
            if any(
                current_candidate_fields[field_name] != stable_row_fields[field_name]
                for field_name in stable_row_fields
            ):
                raise DuplicateGroupingRepositoryInvariantError(
                    "cross_batch_duplicate_grouping_candidate_row_mismatch"
                )
        if row.evidence_link_id is not None:
            current_evidence_link_ids.add(row.evidence_link_id)
    flush_current()
    return tuple(candidates)


async def count_duplicate_candidate_fact_values(
    session: AsyncSession,
    *,
    orchestration_id: uuid.UUID,
) -> int:
    result = await session.execute(
        select(FactValue.id)
        .select_from(FactExtractionOrchestrationBatch)
        .join(FactExtractionOrchestration, FactExtractionOrchestrationBatch.orchestration_id == FactExtractionOrchestration.id)
        .join(
            FactExtractionBatchApplication,
            FactExtractionOrchestrationBatch.application_id == FactExtractionBatchApplication.id,
        )
        .join(FactValue, FactValue.inference_run_id == FactExtractionBatchApplication.inference_run_id)
        .where(
            FactExtractionOrchestrationBatch.orchestration_id == orchestration_id,
            FactExtractionOrchestrationBatch.status == "completed",
            FactExtractionBatchApplication.status == "completed",
            FactExtractionBatchApplication.inference_run_id == FactExtractionOrchestrationBatch.current_inference_run_id,
            FactExtractionBatchApplication.extraction_run_id == FactExtractionOrchestration.extraction_run_id,
            FactValue.extraction_run_id == FactExtractionOrchestration.extraction_run_id,
            FactValue.source_kind == FactValueSourceKind.AI.value,
        )
        .order_by(FactValue.id.asc())
    )
    return len(result.scalars().all())


async def get_grouping_application_for_update(
    session: AsyncSession,
    *,
    orchestration_id: uuid.UUID,
    algorithm_version: str,
) -> FactValueDuplicateGroupingApplication | None:
    result = await session.execute(
        select(FactValueDuplicateGroupingApplication)
        .where(
            FactValueDuplicateGroupingApplication.orchestration_id == orchestration_id,
            FactValueDuplicateGroupingApplication.algorithm_version == algorithm_version,
        )
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def get_grouping_application_ledger(
    session: AsyncSession,
    *,
    orchestration_id: uuid.UUID,
    algorithm_version: str,
) -> DuplicateGroupingApplicationLedger | None:
    result = await session.execute(
        select(FactValueDuplicateGroupingApplication)
        .where(
            FactValueDuplicateGroupingApplication.orchestration_id == orchestration_id,
            FactValueDuplicateGroupingApplication.algorithm_version == algorithm_version,
        )
    )
    application = result.scalar_one_or_none()
    if application is None:
        return None
    return DuplicateGroupingApplicationLedger(
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


async def get_grouping_application_ledger_by_id(
    session: AsyncSession,
    *,
    grouping_application_id: uuid.UUID,
) -> DuplicateGroupingApplicationLedger | None:
    result = await session.execute(
        select(FactValueDuplicateGroupingApplication).where(
            FactValueDuplicateGroupingApplication.id == grouping_application_id,
        )
    )
    application = result.scalar_one_or_none()
    if application is None:
        return None
    return DuplicateGroupingApplicationLedger(
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


async def list_group_ledgers(
    session: AsyncSession,
    *,
    grouping_application_id: uuid.UUID,
) -> tuple[DuplicateGroupLedger, ...]:
    result = await session.execute(
        select(FactValueDuplicateGroup)
        .where(FactValueDuplicateGroup.grouping_application_id == grouping_application_id)
        .order_by(
            FactValueDuplicateGroup.duplicate_key_hash.asc(),
            FactValueDuplicateGroup.id.asc(),
        )
    )
    groups = list(result.scalars().all())
    return tuple(
        DuplicateGroupLedger(
            id=group.id,
            grouping_application_id=group.grouping_application_id,
            duplicate_key_hash=group.duplicate_key_hash,
            member_count=group.member_count,
            distinct_batch_count=group.distinct_batch_count,
            created_at=group.created_at,
        )
        for group in groups
    )


async def list_member_ledgers(
    session: AsyncSession,
    *,
    grouping_application_id: uuid.UUID,
) -> tuple[DuplicateGroupMemberLedger, ...]:
    result = await session.execute(
        select(FactValueDuplicateGroupMember)
        .where(FactValueDuplicateGroupMember.grouping_application_id == grouping_application_id)
        .order_by(
            FactValueDuplicateGroupMember.group_id.asc(),
            FactValueDuplicateGroupMember.fact_value_id.asc(),
        )
    )
    members = list(result.scalars().all())
    return tuple(
        DuplicateGroupMemberLedger(
            id=member.id,
            orchestration_id=member.orchestration_id,
            grouping_application_id=member.grouping_application_id,
            group_id=member.group_id,
            fact_value_id=member.fact_value_id,
            source_batch_id=member.source_batch_id,
            created_at=member.created_at,
        )
        for member in members
    )


async def list_duplicate_group_evidence_projections(
    session: AsyncSession,
    *,
    group_id: uuid.UUID,
) -> tuple[DuplicateGroupEvidenceProjection, ...]:
    result = await session.execute(
        select(
            FactValueDuplicateGroup.id.label("group_id"),
            FactValueDuplicateGroup.duplicate_key_hash.label("duplicate_key_hash"),
            FactValueDuplicateGroupMember.fact_value_id.label("fact_value_id"),
            FactValueDuplicateGroupMember.source_batch_id.label("source_batch_id"),
            FactEvidenceLink.id.label("evidence_link_id"),
            FactEvidenceLink.evidence_id.label("evidence_id"),
        )
        .select_from(FactValueDuplicateGroupMember)
        .join(FactValueDuplicateGroup, FactValueDuplicateGroupMember.group_id == FactValueDuplicateGroup.id)
        .outerjoin(FactEvidenceLink, FactEvidenceLink.fact_value_id == FactValueDuplicateGroupMember.fact_value_id)
        .where(FactValueDuplicateGroupMember.group_id == group_id)
        .order_by(
            FactValueDuplicateGroupMember.fact_value_id.asc(),
            FactEvidenceLink.source_order.asc().nulls_last(),
            FactEvidenceLink.id.asc().nulls_last(),
        )
    )
    rows = list(result.all())
    if not rows:
        return ()

    projections: list[DuplicateGroupEvidenceProjection] = []
    current_fact_value_id: uuid.UUID | None = None
    current_link_ids: list[uuid.UUID] = []
    current_link_ids_seen: set[uuid.UUID] = set()
    current_evidence_ids: list[uuid.UUID] = []
    current_evidence_ids_seen: set[uuid.UUID] = set()
    current_fields: dict[str, object] | None = None

    def flush_current() -> None:
        nonlocal current_fact_value_id
        nonlocal current_link_ids
        nonlocal current_link_ids_seen
        nonlocal current_evidence_ids
        nonlocal current_evidence_ids_seen
        nonlocal current_fields
        if current_fields is None:
            return
        projections.append(
            DuplicateGroupEvidenceProjection(
                group_id=current_fields["group_id"],
                duplicate_key_hash=current_fields["duplicate_key_hash"],
                fact_value_id=current_fields["fact_value_id"],
                source_batch_id=current_fields["source_batch_id"],
                evidence_link_ids=tuple(current_link_ids),
                evidence_ids=tuple(current_evidence_ids),
            )
        )
        current_fact_value_id = None
        current_link_ids = []
        current_link_ids_seen = set()
        current_evidence_ids = []
        current_evidence_ids_seen = set()
        current_fields = None

    for row in rows:
        if current_fact_value_id != row.fact_value_id:
            flush_current()
            current_fact_value_id = row.fact_value_id
            current_fields = {
                "group_id": row.group_id,
                "duplicate_key_hash": row.duplicate_key_hash,
                "fact_value_id": row.fact_value_id,
                "source_batch_id": row.source_batch_id,
            }
        if row.evidence_link_id is not None and row.evidence_link_id not in current_link_ids_seen:
            current_link_ids_seen.add(row.evidence_link_id)
            current_link_ids.append(row.evidence_link_id)
        if row.evidence_id is not None and row.evidence_id not in current_evidence_ids_seen:
            current_evidence_ids_seen.add(row.evidence_id)
            current_evidence_ids.append(row.evidence_id)
    flush_current()
    return tuple(projections)


async def create_grouping_application(
    session: AsyncSession,
    application: FactValueDuplicateGroupingApplication,
) -> FactValueDuplicateGroupingApplication:
    session.add(application)
    await session.flush()
    return application


async def create_duplicate_groups(
    session: AsyncSession,
    groups: list[FactValueDuplicateGroup],
) -> list[FactValueDuplicateGroup]:
    session.add_all(groups)
    await session.flush()
    return groups


async def create_duplicate_group_members(
    session: AsyncSession,
    members: list[FactValueDuplicateGroupMember],
) -> list[FactValueDuplicateGroupMember]:
    session.add_all(members)
    await session.flush()
    return members


async def get_consistency_candidate_application_for_update(
    session: AsyncSession,
    *,
    duplicate_grouping_application_id: uuid.UUID,
    algorithm_version: str,
) -> FactValueConsistencyCandidateApplication | None:
    result = await session.execute(
        select(FactValueConsistencyCandidateApplication)
        .where(
            FactValueConsistencyCandidateApplication.duplicate_grouping_application_id
            == duplicate_grouping_application_id,
            FactValueConsistencyCandidateApplication.algorithm_version == algorithm_version,
        )
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def get_consistency_candidate_application_ledger(
    session: AsyncSession,
    *,
    duplicate_grouping_application_id: uuid.UUID,
    algorithm_version: str,
) -> FactValueConsistencyCandidateApplicationLedger | None:
    result = await session.execute(
        select(FactValueConsistencyCandidateApplication).where(
            FactValueConsistencyCandidateApplication.duplicate_grouping_application_id
            == duplicate_grouping_application_id,
            FactValueConsistencyCandidateApplication.algorithm_version == algorithm_version,
        )
    )
    application = result.scalar_one_or_none()
    if application is None:
        return None
    return FactValueConsistencyCandidateApplicationLedger(
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


async def list_consistency_candidate_ledgers(
    session: AsyncSession,
    *,
    consistency_application_id: uuid.UUID,
) -> tuple[FactValueConsistencyCandidateLedger, ...]:
    result = await session.execute(
        select(FactValueConsistencyCandidate)
        .where(FactValueConsistencyCandidate.consistency_application_id == consistency_application_id)
        .order_by(
            FactValueConsistencyCandidate.fact_id.asc(),
            FactValueConsistencyCandidate.candidate_kind.asc(),
            FactValueConsistencyCandidate.id.asc(),
        )
    )
    candidates = list(result.scalars().all())
    return tuple(
        FactValueConsistencyCandidateLedger(
            id=item.id,
            consistency_application_id=item.consistency_application_id,
            fact_id=item.fact_id,
            candidate_kind=item.candidate_kind,
            member_count=item.member_count,
            distinct_semantic_key_count=item.distinct_semantic_key_count,
            distinct_batch_count=item.distinct_batch_count,
            created_at=item.created_at,
        )
        for item in candidates
    )


async def list_consistency_candidate_member_ledgers(
    session: AsyncSession,
    *,
    consistency_application_id: uuid.UUID,
) -> tuple[FactValueConsistencyCandidateMemberLedger, ...]:
    result = await session.execute(
        select(FactValueConsistencyCandidateMember)
        .where(FactValueConsistencyCandidateMember.consistency_application_id == consistency_application_id)
        .order_by(
            FactValueConsistencyCandidateMember.candidate_id.asc(),
            FactValueConsistencyCandidateMember.fact_value_id.asc(),
        )
    )
    members = list(result.scalars().all())
    return tuple(
        FactValueConsistencyCandidateMemberLedger(
            id=item.id,
            consistency_application_id=item.consistency_application_id,
            candidate_id=item.candidate_id,
            orchestration_id=item.orchestration_id,
            fact_value_id=item.fact_value_id,
            source_batch_id=item.source_batch_id,
            semantic_key_hash=item.semantic_key_hash,
            created_at=item.created_at,
        )
        for item in members
    )


async def create_consistency_candidate_application(
    session: AsyncSession,
    application: FactValueConsistencyCandidateApplication,
) -> FactValueConsistencyCandidateApplication:
    session.add(application)
    await session.flush()
    return application


async def create_consistency_candidates(
    session: AsyncSession,
    candidates: list[FactValueConsistencyCandidate],
) -> list[FactValueConsistencyCandidate]:
    session.add_all(candidates)
    await session.flush()
    return candidates


async def create_consistency_candidate_members(
    session: AsyncSession,
    members: list[FactValueConsistencyCandidateMember],
) -> list[FactValueConsistencyCandidateMember]:
    session.add_all(members)
    await session.flush()
    return members
