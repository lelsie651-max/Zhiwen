from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload

from app.models.document import Document
from app.models.document_revision import DocumentRevision
from app.models.ingestion_validation import IngestionValidationReport


async def list_documents_for_project(
    session: AsyncSession,
    project_id: uuid.UUID,
) -> list[Document]:
    result = await session.execute(
        select(Document)
        .where(Document.project_id == project_id)
        .options(selectinload(Document.revisions))
        .order_by(Document.created_at.desc())
    )
    return list(result.scalars().unique().all())


async def get_document_for_project(
    session: AsyncSession,
    project_id: uuid.UUID,
    document_id: uuid.UUID,
) -> Document | None:
    result = await session.execute(
        select(Document)
        .where(Document.project_id == project_id, Document.id == document_id)
        .options(selectinload(Document.revisions))
    )
    return result.scalar_one_or_none()


async def get_document_for_update(
    session: AsyncSession,
    project_id: uuid.UUID,
    document_id: uuid.UUID,
) -> Document | None:
    result = await session.execute(
        select(Document)
        .where(Document.project_id == project_id, Document.id == document_id)
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def get_latest_revision_for_document(
    session: AsyncSession,
    document_id: uuid.UUID,
) -> DocumentRevision | None:
    result = await session.execute(
        select(DocumentRevision)
        .where(DocumentRevision.document_id == document_id)
        .order_by(DocumentRevision.revision_no.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_revision_detail_for_project(
    session: AsyncSession,
    project_id: uuid.UUID,
    document_id: uuid.UUID,
    revision_id: uuid.UUID,
) -> DocumentRevision | None:
    result = await session.execute(
        select(DocumentRevision)
        .join(Document, Document.id == DocumentRevision.document_id)
        .where(
            Document.project_id == project_id,
            Document.id == document_id,
            DocumentRevision.id == revision_id,
        )
        .options(
            joinedload(DocumentRevision.document),
            joinedload(DocumentRevision.uploaded_by),
            selectinload(DocumentRevision.validation_reports).selectinload(
                IngestionValidationReport.risk_flags
            ),
        )
    )
    return result.scalar_one_or_none()


async def create_document(session: AsyncSession, document: Document) -> Document:
    session.add(document)
    await session.flush()
    return document


async def create_document_revision(
    session: AsyncSession,
    revision: DocumentRevision,
) -> DocumentRevision:
    session.add(revision)
    await session.flush()
    return revision
