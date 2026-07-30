from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import joinedload
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_content import DocumentBlock, ExtractionRun, SourceEvidence
from app.models.document_revision import DocumentRevision
from app.models.fact import Fact, FactEvidenceLink, FactValue
from app.models.project import Project
from app.models.project_member import ProjectMember


async def get_extraction_run_with_project(
    session: AsyncSession,
    extraction_run_id: uuid.UUID,
) -> ExtractionRun | None:
    result = await session.execute(
        select(ExtractionRun)
        .where(ExtractionRun.id == extraction_run_id)
        .options(
            joinedload(ExtractionRun.revision)
            .joinedload(DocumentRevision.document)
            .joinedload(Document.project)
        )
    )
    return result.scalar_one_or_none()


async def list_source_evidences_with_context(
    session: AsyncSession,
    evidence_ids: list[uuid.UUID],
) -> list[SourceEvidence]:
    result = await session.execute(
        select(SourceEvidence)
        .where(SourceEvidence.id.in_(evidence_ids))
        .options(
            joinedload(SourceEvidence.block)
            .joinedload(DocumentBlock.extraction_run)
            .joinedload(ExtractionRun.revision)
            .joinedload(DocumentRevision.document)
            .joinedload(Document.project)
        )
    )
    return list(result.scalars().unique().all())


async def get_fact_by_identity(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    identity_hash: str,
) -> Fact | None:
    result = await session.execute(
        select(Fact).where(
            Fact.project_id == project_id,
            Fact.identity_hash == identity_hash,
        )
    )
    return result.scalar_one_or_none()


async def get_fact_by_identity_for_update(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    identity_hash: str,
) -> Fact | None:
    result = await session.execute(
        select(Fact)
        .where(
            Fact.project_id == project_id,
            Fact.identity_hash == identity_hash,
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


async def get_fact_by_id_for_update(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    fact_id: uuid.UUID,
) -> Fact | None:
    result = await session.execute(
        select(Fact)
        .where(
            Fact.id == fact_id,
            Fact.project_id == project_id,
        )
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def get_fact_value_for_update(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    fact_value_id: uuid.UUID,
) -> FactValue | None:
    result = await session.execute(
        select(FactValue)
        .join(Fact, FactValue.fact_id == Fact.id)
        .where(
            FactValue.id == fact_value_id,
            Fact.project_id == project_id,
        )
        .options(joinedload(FactValue.fact))
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def create_fact(
    session: AsyncSession,
    fact: Fact,
) -> Fact:
    session.add(fact)
    await session.flush()
    return fact


async def get_next_fact_version_no(
    session: AsyncSession,
    fact_id: uuid.UUID,
) -> int:
    result = await session.execute(
        select(func.max(FactValue.version_no)).where(FactValue.fact_id == fact_id)
    )
    current_max = result.scalar_one_or_none()
    return (current_max or 0) + 1


async def create_fact_value(
    session: AsyncSession,
    fact_value: FactValue,
) -> FactValue:
    session.add(fact_value)
    await session.flush()
    return fact_value


async def update_fact(
    session: AsyncSession,
    fact: Fact,
) -> Fact:
    session.add(fact)
    await session.flush()
    return fact


async def update_fact_value(
    session: AsyncSession,
    fact_value: FactValue,
) -> FactValue:
    session.add(fact_value)
    await session.flush()
    return fact_value


async def create_fact_evidence_links(
    session: AsyncSession,
    links: list[FactEvidenceLink],
) -> list[FactEvidenceLink]:
    session.add_all(links)
    await session.flush()
    return links
