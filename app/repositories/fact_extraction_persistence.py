from __future__ import annotations

import copy
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_content import DocumentBlock, ExtractionRun
from app.models.document_revision import DocumentRevision
from app.models.inference import InferenceInputBatch, InferenceInputBlock, InferenceRun
from app.schemas.fact_extraction_persistence import (
    CompletedFactExtractionPersistenceContext,
    FactExtractionPersistenceBlock,
)


async def get_completed_fact_extraction_persistence_context(
    session: AsyncSession,
    *,
    inference_run_id: uuid.UUID,
) -> CompletedFactExtractionPersistenceContext | None:
    result = await session.execute(
        select(
            InferenceRun.id.label("inference_run_id"),
            InferenceRun.project_id.label("project_id"),
            InferenceRun.task_type.label("task_type"),
            InferenceRun.status.label("status"),
            InferenceRun.input_batch_id.label("input_batch_id"),
            InferenceRun.response_json.label("response_json"),
            InferenceRun.response_hash.label("response_hash"),
            InferenceInputBlock.id.label("input_block_id"),
            InferenceInputBlock.block_ref.label("block_ref"),
            InferenceInputBlock.source_order.label("source_order"),
            InferenceInputBlock.document_block_id.label("document_block_id"),
            InferenceInputBlock.source_block_id_snapshot.label("source_block_id_snapshot"),
            InferenceInputBlock.extraction_run_id_snapshot.label("extraction_run_id_snapshot"),
            InferenceInputBlock.content_text.label("content_text"),
            InferenceInputBlock.content_hash.label("content_hash"),
            DocumentBlock.extraction_run_id.label("document_block_extraction_run_id"),
            Document.project_id.label("document_block_project_id"),
            DocumentBlock.raw_text.label("document_block_raw_text"),
        )
        .join(InferenceInputBatch, InferenceRun.input_batch_id == InferenceInputBatch.id)
        .join(InferenceInputBlock, InferenceInputBlock.batch_id == InferenceInputBatch.id)
        .join(DocumentBlock, InferenceInputBlock.document_block_id == DocumentBlock.id)
        .join(ExtractionRun, DocumentBlock.extraction_run_id == ExtractionRun.id)
        .join(DocumentRevision, ExtractionRun.revision_id == DocumentRevision.id)
        .join(Document, DocumentRevision.document_id == Document.id)
        .where(InferenceRun.id == inference_run_id)
        .order_by(InferenceInputBlock.source_order.asc(), InferenceInputBlock.id.asc())
    )
    rows = list(result.all())
    if not rows:
        return None

    first = rows[0]
    blocks = tuple(
        FactExtractionPersistenceBlock(
            input_block_id=row.input_block_id,
            block_ref=row.block_ref,
            source_order=row.source_order,
            document_block_id=row.document_block_id,
            source_block_id_snapshot=row.source_block_id_snapshot,
            extraction_run_id_snapshot=row.extraction_run_id_snapshot,
            content_text=row.content_text,
            content_hash=row.content_hash,
            document_block_extraction_run_id=row.document_block_extraction_run_id,
            document_block_project_id=row.document_block_project_id,
            document_block_raw_text=row.document_block_raw_text,
        )
        for row in rows
    )
    return CompletedFactExtractionPersistenceContext(
        inference_run_id=first.inference_run_id,
        project_id=first.project_id,
        task_type=first.task_type,
        status=first.status,
        input_batch_id=first.input_batch_id,
        response_json=copy.deepcopy(first.response_json) if first.response_json is not None else {},
        response_hash=first.response_hash,
        blocks=blocks,
    )
