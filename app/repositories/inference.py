from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.engine import Row
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_content import DocumentBlock, ExtractionRun
from app.models.document_revision import DocumentRevision
from app.models.inference import InferenceInputBatch, InferenceRun
from app.models.project import Project


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
        select(InferenceInputBatch).where(
            InferenceInputBatch.project_id == project_id,
            InferenceInputBatch.task_type == task_type,
            InferenceInputBatch.snapshot_hash == snapshot_hash,
        )
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
