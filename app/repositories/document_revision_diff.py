from __future__ import annotations

from dataclasses import dataclass
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_content import DocumentBlock, ExtractionRun
from app.models.document_revision import DocumentRevision


@dataclass(frozen=True, slots=True)
class DocumentRevisionDiffDocumentRecord:
    id: uuid.UUID
    project_id: uuid.UUID


@dataclass(frozen=True, slots=True)
class DocumentRevisionDiffRevisionRecord:
    id: uuid.UUID
    document_id: uuid.UUID
    revision_no: int
    supersedes_revision_id: uuid.UUID | None


@dataclass(frozen=True, slots=True)
class DocumentRevisionDiffRunRecord:
    id: uuid.UUID
    revision_id: uuid.UUID
    status: str
    outcome: str
    extractor_name: str
    extractor_version: str
    detected_format: str
    character_count: int
    block_count: int


@dataclass(frozen=True, slots=True)
class DocumentRevisionDiffBlockRecord:
    id: uuid.UUID
    extraction_run_id: uuid.UUID
    source_order: int
    block_type: str
    raw_text: str
    normalized_text: str
    location_key: str
    anchor_hash: str
    page_no: int | None
    start_line: int | None
    end_line: int | None
    table_index: int | None
    row_index: int | None


async def get_document_for_project(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    document_id: uuid.UUID,
) -> DocumentRevisionDiffDocumentRecord | None:
    result = await session.execute(
        select(Document).where(
            Document.project_id == project_id,
            Document.id == document_id,
        )
    )
    document = result.scalar_one_or_none()
    if document is None:
        return None
    return DocumentRevisionDiffDocumentRecord(
        id=document.id,
        project_id=document.project_id,
    )


async def get_document_revision_by_id(
    session: AsyncSession,
    *,
    revision_id: uuid.UUID,
) -> DocumentRevisionDiffRevisionRecord | None:
    result = await session.execute(
        select(DocumentRevision).where(DocumentRevision.id == revision_id)
    )
    revision = result.scalar_one_or_none()
    if revision is None:
        return None
    return DocumentRevisionDiffRevisionRecord(
        id=revision.id,
        document_id=revision.document_id,
        revision_no=revision.revision_no,
        supersedes_revision_id=revision.supersedes_revision_id,
    )


async def get_extraction_run_by_id(
    session: AsyncSession,
    *,
    extraction_run_id: uuid.UUID,
) -> DocumentRevisionDiffRunRecord | None:
    result = await session.execute(
        select(ExtractionRun).where(ExtractionRun.id == extraction_run_id)
    )
    run = result.scalar_one_or_none()
    if run is None:
        return None
    return DocumentRevisionDiffRunRecord(
        id=run.id,
        revision_id=run.revision_id,
        status=run.status,
        outcome=run.outcome,
        extractor_name=run.extractor_name,
        extractor_version=run.extractor_version,
        detected_format=run.detected_format,
        character_count=run.character_count,
        block_count=run.block_count,
    )


async def list_document_blocks_for_extraction_run(
    session: AsyncSession,
    *,
    extraction_run_id: uuid.UUID,
) -> tuple[DocumentRevisionDiffBlockRecord, ...]:
    result = await session.execute(
        select(DocumentBlock)
        .where(DocumentBlock.extraction_run_id == extraction_run_id)
        .order_by(DocumentBlock.source_order.asc(), DocumentBlock.id.asc())
    )
    blocks = result.scalars().all()
    return tuple(
        DocumentRevisionDiffBlockRecord(
            id=block.id,
            extraction_run_id=block.extraction_run_id,
            source_order=block.source_order,
            block_type=block.block_type,
            raw_text=block.raw_text,
            normalized_text=block.normalized_text,
            location_key=block.location_key,
            anchor_hash=block.anchor_hash,
            page_no=block.page_no,
            start_line=block.start_line,
            end_line=block.end_line,
            table_index=block.table_index,
            row_index=block.row_index,
        )
        for block in blocks
    )
