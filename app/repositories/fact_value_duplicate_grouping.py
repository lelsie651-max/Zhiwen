from __future__ import annotations

import copy
from dataclasses import dataclass
import uuid

from sqlalchemy import select
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
)


@dataclass(frozen=True, slots=True)
class DuplicateGroupingRunState:
    extraction_run_id: uuid.UUID
    project_id: uuid.UUID
    extraction_run_status: str
    extraction_run_outcome: str | None
    latest_terminal_orchestration_status: str | None
    active_orchestration_count: int


async def get_duplicate_grouping_run_state(
    session: AsyncSession,
    *,
    extraction_run_id: uuid.UUID,
) -> DuplicateGroupingRunState | None:
    run_result = await session.execute(
        select(
            DocumentExtractionRun.id.label("extraction_run_id"),
            Document.project_id.label("project_id"),
            DocumentExtractionRun.status.label("extraction_run_status"),
            DocumentExtractionRun.outcome.label("extraction_run_outcome"),
        )
        .join(DocumentRevision, DocumentExtractionRun.revision_id == DocumentRevision.id)
        .join(Document, DocumentRevision.document_id == Document.id)
        .where(DocumentExtractionRun.id == extraction_run_id)
    )
    run_row = run_result.one_or_none()
    if run_row is None:
        return None

    orchestration_result = await session.execute(
        select(FactExtractionOrchestration.status)
        .where(FactExtractionOrchestration.extraction_run_id == extraction_run_id)
        .order_by(
            FactExtractionOrchestration.attempt_no.desc(),
            FactExtractionOrchestration.created_at.desc(),
        )
    )
    orchestration_statuses = list(orchestration_result.scalars().all())
    active_orchestration_count = sum(status in {"planned", "running"} for status in orchestration_statuses)
    latest_terminal_orchestration_status = next(
        (status for status in orchestration_statuses if status in {"completed", "partial", "failed"}),
        None,
    )

    return DuplicateGroupingRunState(
        extraction_run_id=run_row.extraction_run_id,
        project_id=run_row.project_id,
        extraction_run_status=run_row.extraction_run_status,
        extraction_run_outcome=run_row.extraction_run_outcome,
        latest_terminal_orchestration_status=latest_terminal_orchestration_status,
        active_orchestration_count=active_orchestration_count,
    )


async def list_duplicate_candidates(
    session: AsyncSession,
    *,
    extraction_run_id: uuid.UUID,
) -> tuple[DuplicateCandidate, ...]:
    result = await session.execute(
        select(
            FactValue.id.label("fact_value_id"),
            FactValue.fact_id.label("fact_id"),
            FactValue.extraction_run_id.label("extraction_run_id"),
            FactExtractionOrchestrationBatch.id.label("source_batch_id"),
            FactValue.value_type.label("value_type"),
            FactValue.value_json.label("value_json"),
            FactValue.referenced_entity_id.label("referenced_entity_id"),
            FactEvidenceLink.id.label("evidence_link_id"),
        )
        .select_from(FactValue)
        .join(Fact, FactValue.fact_id == Fact.id)
        .join(
            FactExtractionBatchApplication,
            FactExtractionBatchApplication.inference_run_id == FactValue.inference_run_id,
        )
        .join(
            FactExtractionOrchestrationBatch,
            FactExtractionOrchestrationBatch.application_id == FactExtractionBatchApplication.id,
        )
        .outerjoin(FactEvidenceLink, FactEvidenceLink.fact_value_id == FactValue.id)
        .where(
            FactValue.extraction_run_id == extraction_run_id,
            FactValue.source_kind == FactValueSourceKind.AI.value,
            FactExtractionBatchApplication.extraction_run_id == extraction_run_id,
            FactExtractionBatchApplication.status == "completed",
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
    current_evidence_link_ids: list[uuid.UUID] = []
    current_candidate_fields: dict[str, object] | None = None

    def flush_current() -> None:
        nonlocal current_fact_value_id, current_evidence_link_ids, current_candidate_fields
        if current_candidate_fields is None:
            return
        candidates.append(
            DuplicateCandidate(
                fact_value_id=current_candidate_fields["fact_value_id"],
                fact_id=current_candidate_fields["fact_id"],
                extraction_run_id=current_candidate_fields["extraction_run_id"],
                source_batch_id=current_candidate_fields["source_batch_id"],
                value_type=current_candidate_fields["value_type"],
                value_json=copy.deepcopy(current_candidate_fields["value_json"]),
                referenced_entity_id=current_candidate_fields["referenced_entity_id"],
                evidence_link_ids=tuple(current_evidence_link_ids),
            )
        )
        current_fact_value_id = None
        current_evidence_link_ids = []
        current_candidate_fields = None

    for row in rows:
        if current_fact_value_id != row.fact_value_id:
            flush_current()
            current_fact_value_id = row.fact_value_id
            current_candidate_fields = {
                "fact_value_id": row.fact_value_id,
                "fact_id": row.fact_id,
                "extraction_run_id": row.extraction_run_id,
                "source_batch_id": row.source_batch_id,
                "value_type": row.value_type,
                "value_json": row.value_json,
                "referenced_entity_id": row.referenced_entity_id,
            }
        if row.evidence_link_id is not None:
            current_evidence_link_ids.append(row.evidence_link_id)
    flush_current()
    return tuple(candidates)


async def count_ai_fact_values(
    session: AsyncSession,
    *,
    extraction_run_id: uuid.UUID,
) -> int:
    result = await session.execute(
        select(FactValue.id)
        .where(
            FactValue.extraction_run_id == extraction_run_id,
            FactValue.source_kind == FactValueSourceKind.AI.value,
        )
        .order_by(FactValue.id.asc())
    )
    return len(result.scalars().all())


async def get_grouping_application_for_update(
    session: AsyncSession,
    *,
    extraction_run_id: uuid.UUID,
    algorithm_version: str,
) -> FactValueDuplicateGroupingApplication | None:
    result = await session.execute(
        select(FactValueDuplicateGroupingApplication)
        .where(
            FactValueDuplicateGroupingApplication.extraction_run_id == extraction_run_id,
            FactValueDuplicateGroupingApplication.algorithm_version == algorithm_version,
        )
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def get_grouping_application_ledger(
    session: AsyncSession,
    *,
    extraction_run_id: uuid.UUID,
    algorithm_version: str,
) -> DuplicateGroupingApplicationLedger | None:
    result = await session.execute(
        select(FactValueDuplicateGroupingApplication)
        .where(
            FactValueDuplicateGroupingApplication.extraction_run_id == extraction_run_id,
            FactValueDuplicateGroupingApplication.algorithm_version == algorithm_version,
        )
    )
    application = result.scalar_one_or_none()
    if application is None:
        return None
    return DuplicateGroupingApplicationLedger(
        id=application.id,
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
    current_evidence_ids: list[uuid.UUID] = []
    current_fields: dict[str, object] | None = None

    def flush_current() -> None:
        nonlocal current_fact_value_id, current_link_ids, current_evidence_ids, current_fields
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
        current_evidence_ids = []
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
        if row.evidence_link_id is not None:
            current_link_ids.append(row.evidence_link_id)
        if row.evidence_id is not None:
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
