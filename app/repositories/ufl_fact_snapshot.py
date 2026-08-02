from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.document_revision_fact_diff import (
    DocumentRevisionFactDiffSourceRow,
    list_document_revision_fact_diff_source_rows,
)


async def list_orchestration_ufl_fact_source_rows(
    session: AsyncSession,
    *,
    orchestration_id: uuid.UUID,
) -> tuple[DocumentRevisionFactDiffSourceRow, ...]:
    return await list_document_revision_fact_diff_source_rows(
        session,
        orchestration_id=orchestration_id,
    )
