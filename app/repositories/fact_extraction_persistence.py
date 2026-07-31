from __future__ import annotations

import copy
from dataclasses import dataclass
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_content import DocumentBlock, ExtractionRun
from app.models.document_revision import DocumentRevision
from app.models.fact_extraction_application import FactExtractionBatchApplication
from app.models.inference import InferenceInputBatch, InferenceInputBlock, InferenceRun
from app.schemas.fact_extraction_persistence import (
    CompletedFactExtractionPersistenceContext,
    FactExtractionPersistenceBlock,
)


@dataclass(frozen=True, slots=True)
class InferenceRunBatchHeader:
    inference_run_id: uuid.UUID
    project_id: uuid.UUID
    task_type: str
    status: str
    input_batch_id: uuid.UUID
    batch_project_id: uuid.UUID
    batch_task_type: str
    batch_block_count: int
    batch_character_count: int
    batch_snapshot_hash: str
    response_json: dict | None
    response_hash: str | None
    response_json_hash: str | None


async def get_inference_run_batch_header(
    session: AsyncSession,
    *,
    inference_run_id: uuid.UUID,
) -> InferenceRunBatchHeader | None:
    result = await session.execute(
        select(
            InferenceRun.id.label("inference_run_id"),
            InferenceRun.project_id.label("project_id"),
            InferenceRun.task_type.label("task_type"),
            InferenceRun.status.label("status"),
            InferenceRun.input_batch_id.label("input_batch_id"),
            InferenceInputBatch.project_id.label("batch_project_id"),
            InferenceInputBatch.task_type.label("batch_task_type"),
            InferenceInputBatch.block_count.label("batch_block_count"),
            InferenceInputBatch.character_count.label("batch_character_count"),
            InferenceInputBatch.snapshot_hash.label("batch_snapshot_hash"),
            InferenceRun.response_json.label("response_json"),
            InferenceRun.response_hash.label("response_hash"),
            InferenceRun.response_json_hash.label("response_json_hash"),
        )
        .join(InferenceInputBatch, InferenceRun.input_batch_id == InferenceInputBatch.id)
        .where(InferenceRun.id == inference_run_id)
    )
    row = result.one_or_none()
    if row is None:
        return None

    return InferenceRunBatchHeader(
        inference_run_id=row.inference_run_id,
        project_id=row.project_id,
        task_type=row.task_type,
        status=row.status,
        input_batch_id=row.input_batch_id,
        batch_project_id=row.batch_project_id,
        batch_task_type=row.batch_task_type,
        batch_block_count=row.batch_block_count,
        batch_character_count=row.batch_character_count,
        batch_snapshot_hash=row.batch_snapshot_hash,
        response_json=copy.deepcopy(row.response_json) if row.response_json is not None else None,
        response_hash=row.response_hash,
        response_json_hash=row.response_json_hash,
    )


async def list_input_blocks_with_live_context(
    session: AsyncSession,
    *,
    input_batch_id: uuid.UUID,
) -> tuple[FactExtractionPersistenceBlock, ...]:
    result = await session.execute(
        select(
            InferenceInputBlock.id.label("input_block_id"),
            InferenceInputBlock.block_ref.label("block_ref"),
            InferenceInputBlock.source_order.label("source_order"),
            InferenceInputBlock.block_type.label("block_type"),
            InferenceInputBlock.location_key.label("location_key"),
            InferenceInputBlock.anchor_hash.label("anchor_hash"),
            InferenceInputBlock.page_no.label("page_no"),
            InferenceInputBlock.start_line.label("start_line"),
            InferenceInputBlock.end_line.label("end_line"),
            InferenceInputBlock.heading_path.label("heading_path"),
            InferenceInputBlock.document_block_id.label("document_block_id"),
            InferenceInputBlock.source_block_id_snapshot.label("source_block_id_snapshot"),
            InferenceInputBlock.extraction_run_id_snapshot.label("extraction_run_id_snapshot"),
            InferenceInputBlock.content_text.label("content_text"),
            InferenceInputBlock.content_hash.label("content_hash"),
            DocumentBlock.extraction_run_id.label("document_block_extraction_run_id"),
            Document.project_id.label("document_block_project_id"),
            DocumentBlock.raw_text.label("document_block_raw_text"),
        )
        .select_from(InferenceInputBlock)
        .outerjoin(DocumentBlock, InferenceInputBlock.document_block_id == DocumentBlock.id)
        .outerjoin(ExtractionRun, DocumentBlock.extraction_run_id == ExtractionRun.id)
        .outerjoin(DocumentRevision, ExtractionRun.revision_id == DocumentRevision.id)
        .outerjoin(Document, DocumentRevision.document_id == Document.id)
        .where(InferenceInputBlock.batch_id == input_batch_id)
        .order_by(InferenceInputBlock.source_order.asc(), InferenceInputBlock.id.asc())
    )
    rows = list(result.all())
    return tuple(
        FactExtractionPersistenceBlock(
            input_block_id=row.input_block_id,
            block_ref=row.block_ref,
            source_order=row.source_order,
            block_type=row.block_type,
            location_key=row.location_key,
            anchor_hash=row.anchor_hash,
            page_no=row.page_no,
            start_line=row.start_line,
            end_line=row.end_line,
            heading_path=tuple(row.heading_path),
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


async def get_completed_fact_extraction_persistence_context(
    session: AsyncSession,
    *,
    inference_run_id: uuid.UUID,
) -> CompletedFactExtractionPersistenceContext | None:
    header = await get_inference_run_batch_header(session, inference_run_id=inference_run_id)
    if header is None:
        return None
    blocks = await list_input_blocks_with_live_context(
        session,
        input_batch_id=header.input_batch_id,
    )
    return CompletedFactExtractionPersistenceContext(
        inference_run_id=header.inference_run_id,
        project_id=header.project_id,
        task_type=header.task_type,
        status=header.status,
        input_batch_id=header.input_batch_id,
        batch_project_id=header.batch_project_id,
        batch_task_type=header.batch_task_type,
        batch_block_count=header.batch_block_count,
        batch_character_count=header.batch_character_count,
        batch_snapshot_hash=header.batch_snapshot_hash,
        response_json=copy.deepcopy(header.response_json) if header.response_json is not None else {},
        response_hash=header.response_hash or "",
        response_json_hash=header.response_json_hash or "",
        blocks=blocks,
    )


async def get_batch_application_for_update(
    session: AsyncSession,
    *,
    inference_run_id: uuid.UUID,
) -> FactExtractionBatchApplication | None:
    result = await session.execute(
        select(FactExtractionBatchApplication)
        .where(FactExtractionBatchApplication.inference_run_id == inference_run_id)
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def create_batch_application(
    session: AsyncSession,
    application: FactExtractionBatchApplication,
) -> FactExtractionBatchApplication:
    session.add(application)
    await session.flush()
    return application
