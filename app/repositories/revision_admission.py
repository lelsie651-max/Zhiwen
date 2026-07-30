from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_revision import DocumentRevision
from app.models.ingestion_validation import IngestionValidationReport
from app.models.project_member import ProjectMember
from app.models.user import User


async def get_revision_for_admission_update(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    revision_id: uuid.UUID,
) -> DocumentRevision | None:
    result = await session.execute(
        select(DocumentRevision)
        .join(Document, Document.id == DocumentRevision.document_id)
        .where(
            Document.project_id == project_id,
            DocumentRevision.id == revision_id,
        )
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def get_latest_validation_report_for_update(
    session: AsyncSession,
    *,
    revision_id: uuid.UUID,
) -> IngestionValidationReport | None:
    result = await session.execute(
        select(IngestionValidationReport)
        .where(IngestionValidationReport.revision_id == revision_id)
        .order_by(IngestionValidationReport.attempt_no.desc())
        .limit(1)
        .with_for_update()
    )
    return result.scalar_one_or_none()


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
