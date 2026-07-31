from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import uuid

from sqlalchemy import exists, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_content import ExtractionRun
from app.models.document_revision import DocumentRevision
from app.models.processing_job import ACTIVE_PROCESSING_JOB_STATUSES, ProcessingJob
from app.models.project_member import ProjectMember
from app.models.user import User


@dataclass(frozen=True, slots=True)
class ProcessingExtractionResultContext:
    job_id: uuid.UUID
    project_id: uuid.UUID
    job_status: str
    job_type: str
    job_revision_id: uuid.UUID
    result_run_id: uuid.UUID | None
    run_revision_id: uuid.UUID | None
    run_status: str | None
    run_outcome: str | None
    run_completed_at: datetime | None
    revision_status: str
    lease_token: uuid.UUID | None
    lease_expires_at: datetime | None


@dataclass(frozen=True, slots=True)
class ProcessingJobIdentitySnapshot:
    job_id: uuid.UUID
    project_id: uuid.UUID
    revision_id: uuid.UUID
    job_type: str


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


async def get_processing_job_identity_snapshot(
    session: AsyncSession,
    *,
    job_id: uuid.UUID,
) -> ProcessingJobIdentitySnapshot | None:
    result = await session.execute(
        select(
            ProcessingJob.id,
            ProcessingJob.project_id,
            ProcessingJob.revision_id,
            ProcessingJob.job_type,
        ).where(ProcessingJob.id == job_id)
    )
    row = result.one_or_none()
    if row is None:
        return None
    return ProcessingJobIdentitySnapshot(
        job_id=row[0],
        project_id=row[1],
        revision_id=row[2],
        job_type=row[3],
    )


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


async def get_extraction_run_by_id_for_update(
    session: AsyncSession,
    *,
    extraction_run_id: uuid.UUID,
) -> ExtractionRun | None:
    result = await session.execute(
        select(ExtractionRun)
        .where(ExtractionRun.id == extraction_run_id)
        .with_for_update()
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


async def get_processing_extraction_result_context(
    session: AsyncSession,
    *,
    job_id: uuid.UUID,
    for_update: bool = False,
) -> ProcessingExtractionResultContext | None:
    statement = _build_processing_extraction_result_context_statement(
        job_id=job_id,
        for_update=for_update,
    )
    result = await session.execute(statement)
    row = result.one_or_none()
    if row is None:
        return None

    return ProcessingExtractionResultContext(
        job_id=row[0],
        project_id=row[1],
        job_status=row[2],
        job_type=row[3],
        job_revision_id=row[4],
        result_run_id=row[5],
        run_revision_id=row[6],
        run_status=row[7],
        run_outcome=row[8],
        run_completed_at=row[9],
        revision_status=row[10],
        lease_token=row[11],
        lease_expires_at=row[12],
    )


def _build_processing_extraction_result_context_statement(
    *,
    job_id: uuid.UUID,
    for_update: bool,
):
    statement = (
        select(
            ProcessingJob.id,
            ProcessingJob.project_id,
            ProcessingJob.status,
            ProcessingJob.job_type,
            ProcessingJob.revision_id,
            ProcessingJob.result_extraction_run_id,
            ExtractionRun.revision_id,
            ExtractionRun.status,
            ExtractionRun.outcome,
            ExtractionRun.completed_at,
            DocumentRevision.status,
            ProcessingJob.lease_token,
            ProcessingJob.lease_expires_at,
        )
        .join(DocumentRevision, DocumentRevision.id == ProcessingJob.revision_id)
        .outerjoin(ExtractionRun, ExtractionRun.id == ProcessingJob.result_extraction_run_id)
        .where(ProcessingJob.id == job_id)
    )
    if for_update:
        statement = statement.with_for_update(of=ProcessingJob)
    return statement


async def revision_has_unlinked_terminal_extraction_run(
    session: AsyncSession,
    *,
    revision_id: uuid.UUID,
) -> bool:
    linked_job_exists = exists(
        select(1)
        .select_from(ProcessingJob)
        .where(ProcessingJob.result_extraction_run_id == ExtractionRun.id)
    )
    result = await session.execute(
        select(
            exists().where(
                ExtractionRun.revision_id == revision_id,
                ExtractionRun.status.in_(("completed", "failed")),
                ~linked_job_exists,
            )
        )
    )
    return bool(result.scalar_one())
