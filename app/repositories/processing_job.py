from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_content import ExtractionRun
from app.models.document_revision import DocumentRevision
from app.models.processing_job import ACTIVE_PROCESSING_JOB_STATUSES, ProcessingJob
from app.models.project_member import ProjectMember
from app.models.user import User


async def get_revision_for_processing_job_update(
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


async def get_active_processing_job_for_update(
    session: AsyncSession,
    *,
    revision_id: uuid.UUID,
    job_type: str,
) -> ProcessingJob | None:
    result = await session.execute(
        select(ProcessingJob)
        .where(
            ProcessingJob.revision_id == revision_id,
            ProcessingJob.job_type == job_type,
            ProcessingJob.status.in_(ACTIVE_PROCESSING_JOB_STATUSES),
        )
        .order_by(ProcessingJob.attempt_no.desc())
        .limit(1)
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def get_next_processing_job_attempt_no(
    session: AsyncSession,
    *,
    revision_id: uuid.UUID,
    job_type: str,
) -> int:
    result = await session.execute(
        select(func.max(ProcessingJob.attempt_no)).where(
            ProcessingJob.revision_id == revision_id,
            ProcessingJob.job_type == job_type,
        )
    )
    current_max = result.scalar_one_or_none()
    return (current_max or 0) + 1


async def create_processing_job(
    session: AsyncSession,
    *,
    job: ProcessingJob,
) -> ProcessingJob:
    session.add(job)
    await session.flush()
    return job


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


async def get_latest_extraction_run_for_revision(
    session: AsyncSession,
    *,
    revision_id: uuid.UUID,
) -> ExtractionRun | None:
    result = await session.execute(
        select(ExtractionRun)
        .where(ExtractionRun.revision_id == revision_id)
        .order_by(ExtractionRun.attempt_no.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_latest_processing_job_for_revision(
    session: AsyncSession,
    *,
    revision_id: uuid.UUID,
    job_type: str,
) -> ProcessingJob | None:
    result = await session.execute(
        select(ProcessingJob)
        .where(
            ProcessingJob.revision_id == revision_id,
            ProcessingJob.job_type == job_type,
        )
        .order_by(ProcessingJob.attempt_no.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_latest_failed_processing_job(
    session: AsyncSession,
    *,
    revision_id: uuid.UUID,
    job_type: str,
) -> ProcessingJob | None:
    result = await session.execute(
        select(ProcessingJob)
        .where(
            ProcessingJob.revision_id == revision_id,
            ProcessingJob.job_type == job_type,
            ProcessingJob.status == "failed",
        )
        .order_by(ProcessingJob.attempt_no.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_processing_job_for_update(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    job_id: uuid.UUID,
) -> ProcessingJob | None:
    result = await session.execute(
        select(ProcessingJob)
        .where(
            ProcessingJob.id == job_id,
            ProcessingJob.project_id == project_id,
        )
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def get_processing_job_by_id(
    session: AsyncSession,
    *,
    job_id: uuid.UUID,
) -> ProcessingJob | None:
    result = await session.execute(
        select(ProcessingJob).where(ProcessingJob.id == job_id)
    )
    return result.scalar_one_or_none()


async def get_processing_job_by_id_for_update(
    session: AsyncSession,
    *,
    job_id: uuid.UUID,
) -> ProcessingJob | None:
    result = await session.execute(
        select(ProcessingJob)
        .where(ProcessingJob.id == job_id)
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def get_extraction_run_by_id(
    session: AsyncSession,
    *,
    extraction_run_id: uuid.UUID,
) -> ExtractionRun | None:
    result = await session.execute(
        select(ExtractionRun).where(ExtractionRun.id == extraction_run_id)
    )
    return result.scalar_one_or_none()


async def get_revision_by_id(
    session: AsyncSession,
    *,
    revision_id: uuid.UUID,
) -> DocumentRevision | None:
    result = await session.execute(
        select(DocumentRevision).where(DocumentRevision.id == revision_id)
    )
    return result.scalar_one_or_none()
