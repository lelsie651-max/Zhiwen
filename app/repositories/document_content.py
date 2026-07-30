from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document_content import DocumentBlock, ExtractionRun, SourceEvidence
from app.models.document_revision import DocumentRevision


async def get_revision_for_extraction_update(
    session: AsyncSession,
    revision_id: uuid.UUID,
) -> DocumentRevision | None:
    result = await session.execute(
        select(DocumentRevision)
        .where(DocumentRevision.id == revision_id)
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def get_next_extraction_attempt_no(
    session: AsyncSession,
    revision_id: uuid.UUID,
) -> int:
    result = await session.execute(
        select(func.max(ExtractionRun.attempt_no)).where(ExtractionRun.revision_id == revision_id)
    )
    current_max = result.scalar_one_or_none()
    return (current_max or 0) + 1


async def create_extraction_run(
    session: AsyncSession,
    extraction_run: ExtractionRun,
) -> ExtractionRun:
    session.add(extraction_run)
    await session.flush()
    return extraction_run


async def create_document_blocks(
    session: AsyncSession,
    blocks: list[DocumentBlock],
) -> list[DocumentBlock]:
    session.add_all(blocks)
    await session.flush()
    return blocks


async def get_document_block_by_id(
    session: AsyncSession,
    block_id: uuid.UUID,
) -> DocumentBlock | None:
    result = await session.execute(
        select(DocumentBlock).where(DocumentBlock.id == block_id)
    )
    return result.scalar_one_or_none()


async def get_source_evidence_by_offsets(
    session: AsyncSession,
    block_id: uuid.UUID,
    start_offset: int,
    end_offset: int,
) -> SourceEvidence | None:
    result = await session.execute(
        select(SourceEvidence).where(
            SourceEvidence.block_id == block_id,
            SourceEvidence.start_offset == start_offset,
            SourceEvidence.end_offset == end_offset,
        )
    )
    return result.scalar_one_or_none()


async def create_source_evidence(
    session: AsyncSession,
    evidence: SourceEvidence,
) -> SourceEvidence:
    session.add(evidence)
    await session.flush()
    return evidence
