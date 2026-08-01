from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.consistency_check import ConsistencyAssessmentLedger, ConsistencyCheckApplication
from app.models.consistency_review import (
    ConsistencyReviewDecision,
    ConsistencyReviewDecisionSelection,
)
from app.models.fact_value_duplicate_grouping import FactValueConsistencyCandidateMember
from app.models.project_member import ProjectMember
from app.models.user import User
from app.schemas.consistency_review import (
    ConsistencyReviewCandidateMemberRecord,
    ConsistencyReviewDecisionLedgerRecord,
    ConsistencyReviewDecisionSelectionLedgerRecord,
)


async def get_active_user_by_id(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
) -> User | None:
    result = await session.execute(
        select(User).where(
            User.id == user_id,
            User.status == "active",
        )
    )
    return result.scalar_one_or_none()


async def get_active_user_by_id_for_update(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
) -> User | None:
    result = await session.execute(
        select(User)
        .where(
            User.id == user_id,
            User.status == "active",
        )
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def get_project_member_for_project(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    user_id: uuid.UUID,
) -> ProjectMember | None:
    result = await session.execute(
        select(ProjectMember).where(
            ProjectMember.project_id == project_id,
            ProjectMember.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()


async def get_project_member_for_project_for_update(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    user_id: uuid.UUID,
) -> ProjectMember | None:
    result = await session.execute(
        select(ProjectMember)
        .where(
            ProjectMember.project_id == project_id,
            ProjectMember.user_id == user_id,
        )
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def get_consistency_check_application_by_id(
    session: AsyncSession,
    *,
    consistency_check_application_id: uuid.UUID,
) -> ConsistencyCheckApplication | None:
    result = await session.execute(
        select(ConsistencyCheckApplication).where(
            ConsistencyCheckApplication.id == consistency_check_application_id
        )
    )
    return result.scalar_one_or_none()


async def get_consistency_assessment_for_update(
    session: AsyncSession,
    *,
    consistency_check_application_id: uuid.UUID,
    assessment_id: uuid.UUID,
) -> ConsistencyAssessmentLedger | None:
    result = await session.execute(
        select(ConsistencyAssessmentLedger)
        .where(
            ConsistencyAssessmentLedger.id == assessment_id,
            ConsistencyAssessmentLedger.consistency_check_application_id
            == consistency_check_application_id,
        )
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def list_candidate_member_records(
    session: AsyncSession,
    *,
    source_consistency_application_id: uuid.UUID,
    source_consistency_candidate_id: uuid.UUID,
) -> tuple[ConsistencyReviewCandidateMemberRecord, ...]:
    result = await session.execute(
        select(FactValueConsistencyCandidateMember)
        .where(
            FactValueConsistencyCandidateMember.consistency_application_id
            == source_consistency_application_id,
            FactValueConsistencyCandidateMember.candidate_id
            == source_consistency_candidate_id,
        )
        .order_by(
            FactValueConsistencyCandidateMember.semantic_key_hash.asc(),
            FactValueConsistencyCandidateMember.source_batch_id.asc(),
            FactValueConsistencyCandidateMember.fact_value_id.asc(),
        )
    )
    members = list(result.scalars().all())
    return tuple(
        ConsistencyReviewCandidateMemberRecord(
            consistency_application_id=member.consistency_application_id,
            candidate_id=member.candidate_id,
            fact_value_id=member.fact_value_id,
            source_batch_id=member.source_batch_id,
            semantic_key_hash=member.semantic_key_hash,
        )
        for member in members
    )


async def list_decision_ledgers(
    session: AsyncSession,
    *,
    assessment_id: uuid.UUID,
) -> tuple[ConsistencyReviewDecisionLedgerRecord, ...]:
    result = await session.execute(
        select(ConsistencyReviewDecision)
        .where(ConsistencyReviewDecision.assessment_id == assessment_id)
        .order_by(
            ConsistencyReviewDecision.decision_no.asc(),
            ConsistencyReviewDecision.id.asc(),
        )
    )
    decisions = list(result.scalars().all())
    return tuple(
        ConsistencyReviewDecisionLedgerRecord(
            id=decision.id,
            project_id=decision.project_id,
            consistency_check_application_id=decision.consistency_check_application_id,
            assessment_id=decision.assessment_id,
            source_consistency_application_id=decision.source_consistency_application_id,
            source_consistency_candidate_id=decision.source_consistency_candidate_id,
            actor_id=decision.actor_id,
            decision_no=decision.decision_no,
            supersedes_decision_id=decision.supersedes_decision_id,
            decision_kind=decision.decision_kind,
            selected_value_count=decision.selected_value_count,
            comment=decision.comment,
            decision_manifest_hash=decision.decision_manifest_hash,
            created_at=decision.created_at,
        )
        for decision in decisions
    )


async def list_selection_ledgers(
    session: AsyncSession,
    *,
    assessment_id: uuid.UUID,
) -> tuple[ConsistencyReviewDecisionSelectionLedgerRecord, ...]:
    result = await session.execute(
        select(ConsistencyReviewDecisionSelection)
        .join(
            ConsistencyReviewDecision,
            ConsistencyReviewDecision.id == ConsistencyReviewDecisionSelection.decision_id,
        )
        .where(ConsistencyReviewDecisionSelection.assessment_id == assessment_id)
        .order_by(
            ConsistencyReviewDecision.decision_no.asc(),
            ConsistencyReviewDecisionSelection.selection_order.asc(),
            ConsistencyReviewDecisionSelection.id.asc(),
        )
    )
    selections = list(result.scalars().all())
    return tuple(
        ConsistencyReviewDecisionSelectionLedgerRecord(
            id=selection.id,
            decision_id=selection.decision_id,
            assessment_id=selection.assessment_id,
            source_consistency_application_id=selection.source_consistency_application_id,
            source_consistency_candidate_id=selection.source_consistency_candidate_id,
            fact_value_id=selection.fact_value_id,
            selection_order=selection.selection_order,
            created_at=selection.created_at,
        )
        for selection in selections
    )


async def create_decision(
    session: AsyncSession,
    decision: ConsistencyReviewDecision,
) -> ConsistencyReviewDecision:
    session.add(decision)
    await session.flush()
    return decision


async def create_selections(
    session: AsyncSession,
    selections: list[ConsistencyReviewDecisionSelection],
) -> list[ConsistencyReviewDecisionSelection]:
    session.add_all(selections)
    await session.flush()
    return selections
