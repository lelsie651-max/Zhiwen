from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_revision import DocumentRevision
from app.models.ingestion_validation import IngestionValidationReport


async def get_revision_for_extraction_update(
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
