from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.engine import Row
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.document import Document
from app.models.document_content import DocumentBlock, ExtractionRun
from app.models.document_revision import DocumentRevision
from app.models.inference import InferenceInputBatch, InferenceInputBlock, InferenceRun
from app.models.inference import InferenceRunStatus
from app.models.project import Project


@dataclass(frozen=True, slots=True)
class CompletedFactExtractionRunContext:
    run_id: uuid.UUID
    project_id: uuid.UUID
    task_type: str
    status: str
    batch_id: uuid.UUID
    extraction_run_id_snapshots: frozenset[uuid.UUID]
    source_block_id_snapshots: frozenset[uuid.UUID]


async def get_project(session: AsyncSession, project_id: uuid.UUID) -> Project | None:
    result = await session.execute(select(Project).where(Project.id == project_id))
    return result.scalar_one_or_none()


async def get_blocks_with_extraction_context(
    session: AsyncSession,
    block_ids: Sequence[uuid.UUID],
) -> list[Row]:
    """Return each requested block joined to its extraction run and project.

    The join is explicit so no async lazy-loaded relationship is ever touched.
    """

    result = await session.execute(
        select(
            DocumentBlock,
            ExtractionRun.status.label("run_status"),
            ExtractionRun.outcome.label("run_outcome"),
            Document.project_id.label("project_id"),
        )
        .join(ExtractionRun, DocumentBlock.extraction_run_id == ExtractionRun.id)
        .join(DocumentRevision, ExtractionRun.revision_id == DocumentRevision.id)
        .join(Document, DocumentRevision.document_id == Document.id)
        .where(DocumentBlock.id.in_(list(block_ids)))
    )
    return list(result.all())


async def get_batch_by_identity(
    session: AsyncSession,
    project_id: uuid.UUID,
    task_type: str,
    snapshot_hash: str,
) -> InferenceInputBatch | None:
    result = await session.execute(
        select(InferenceInputBatch)
        .where(
            InferenceInputBatch.project_id == project_id,
            InferenceInputBatch.task_type == task_type,
            InferenceInputBatch.snapshot_hash == snapshot_hash,
        )
        # Eagerly load blocks (ordered by source_order via the relationship) so an
        # idempotently returned batch never triggers an async lazy load later.
        .options(selectinload(InferenceInputBatch.blocks))
    )
    return result.scalar_one_or_none()


async def create_inference_batch_with_blocks(
    session: AsyncSession,
    batch: InferenceInputBatch,
) -> InferenceInputBatch:
    # Blocks live in batch.blocks; the cascade inserts them with the batch.
    session.add(batch)
    await session.flush()
    return batch


async def get_batch_for_update(
    session: AsyncSession,
    batch_id: uuid.UUID,
) -> InferenceInputBatch | None:
    result = await session.execute(
        select(InferenceInputBatch)
        .where(InferenceInputBatch.id == batch_id)
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def get_next_run_attempt_no(
    session: AsyncSession,
    input_batch_id: uuid.UUID,
    agent_name: str,
    prompt_version: str,
) -> int:
    result = await session.execute(
        select(func.max(InferenceRun.attempt_no)).where(
            InferenceRun.input_batch_id == input_batch_id,
            InferenceRun.agent_name == agent_name,
            InferenceRun.prompt_version == prompt_version,
        )
    )
    current_max = result.scalar_one_or_none()
    return (current_max or 0) + 1


async def create_inference_run(
    session: AsyncSession,
    run: InferenceRun,
) -> InferenceRun:
    session.add(run)
    await session.flush()
    return run


async def get_run_for_update(
    session: AsyncSession,
    run_id: uuid.UUID,
) -> InferenceRun | None:
    result = await session.execute(
        select(InferenceRun).where(InferenceRun.id == run_id).with_for_update()
    )
    return result.scalar_one_or_none()


async def get_runs_by_request_for_update(
    session: AsyncSession,
    *,
    input_batch_id: uuid.UUID,
    request_hash: str,
    agent_name: str,
    prompt_version: str,
) -> list[InferenceRun]:
    result = await session.execute(
        select(InferenceRun)
        .where(
            InferenceRun.input_batch_id == input_batch_id,
            InferenceRun.request_hash == request_hash,
            InferenceRun.agent_name == agent_name,
            InferenceRun.prompt_version == prompt_version,
        )
        .order_by(InferenceRun.attempt_no.desc(), InferenceRun.created_at.desc())
        .with_for_update()
    )
    return list(result.scalars().all())


async def get_active_run_by_request_for_update(
    session: AsyncSession,
    *,
    input_batch_id: uuid.UUID,
    request_hash: str,
    agent_name: str,
    prompt_version: str,
) -> InferenceRun | None:
    result = await session.execute(
        select(InferenceRun)
        .where(
            InferenceRun.input_batch_id == input_batch_id,
            InferenceRun.request_hash == request_hash,
            InferenceRun.agent_name == agent_name,
            InferenceRun.prompt_version == prompt_version,
            InferenceRun.status.in_(
                (
                    InferenceRunStatus.PENDING.value,
                    InferenceRunStatus.RUNNING.value,
                )
            ),
        )
        .order_by(InferenceRun.attempt_no.desc(), InferenceRun.created_at.desc())
        .with_for_update()
    )
    return result.scalars().first()


async def get_completed_fact_extraction_run_context(
    session: AsyncSession,
    *,
    inference_run_id: uuid.UUID,
) -> CompletedFactExtractionRunContext | None:
    result = await session.execute(
        select(
            InferenceRun.id.label("run_id"),
            InferenceRun.project_id.label("project_id"),
            InferenceRun.task_type.label("task_type"),
            InferenceRun.status.label("status"),
            InferenceRun.input_batch_id.label("batch_id"),
            InferenceInputBlock.extraction_run_id_snapshot.label("extraction_run_id_snapshot"),
            InferenceInputBlock.source_block_id_snapshot.label("source_block_id_snapshot"),
        )
        .join(InferenceInputBatch, InferenceRun.input_batch_id == InferenceInputBatch.id)
        .outerjoin(InferenceInputBlock, InferenceInputBlock.batch_id == InferenceInputBatch.id)
        .where(InferenceRun.id == inference_run_id)
    )
    rows = list(result.all())
    if not rows:
        return None

    first = rows[0]
    extraction_run_id_snapshots = frozenset(
        row.extraction_run_id_snapshot
        for row in rows
        if row.extraction_run_id_snapshot is not None
    )
    source_block_id_snapshots = frozenset(
        row.source_block_id_snapshot
        for row in rows
        if row.source_block_id_snapshot is not None
    )
    return CompletedFactExtractionRunContext(
        run_id=first.run_id,
        project_id=first.project_id,
        task_type=first.task_type,
        status=first.status,
        batch_id=first.batch_id,
        extraction_run_id_snapshots=extraction_run_id_snapshots,
        source_block_id_snapshots=source_block_id_snapshots,
    )
