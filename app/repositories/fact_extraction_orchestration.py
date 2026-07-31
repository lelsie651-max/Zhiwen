from __future__ import annotations

from dataclasses import dataclass
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.models.document import Document
from app.models.document_content import ExtractionRun
from app.models.document_revision import DocumentRevision
from app.models.fact_extraction_application import FactExtractionBatchApplication
from app.models.fact_extraction_orchestration import (
    FactExtractionOrchestration,
    FactExtractionOrchestrationBatch,
)


@dataclass(frozen=True, slots=True)
class ExtractionRunProjectContext:
    extraction_run: ExtractionRun
    project_id: uuid.UUID
    revision_status: str


@dataclass(frozen=True, slots=True)
class BatchAttemptReconciliationContext:
    orchestration_id: uuid.UUID
    orchestration_status: str
    orchestration_project_id: uuid.UUID
    orchestration_extraction_run_id: uuid.UUID
    batch_id: uuid.UUID
    batch_index: int
    batch_status: str
    attempt_count: int
    lease_token: uuid.UUID | None
    input_batch_id: uuid.UUID | None
    inference_run_id: uuid.UUID | None
    inference_run_status: str | None
    inference_run_project_id: uuid.UUID | None
    inference_run_task_type: str | None
    inference_run_input_batch_id: uuid.UUID | None
    inference_run_failure_code: str | None
    application_id: uuid.UUID | None
    application_status: str | None
    batch_application_id: uuid.UUID | None
    batch_application_status: str | None
    run_application_id: uuid.UUID | None
    run_application_status: str | None


async def get_extraction_run_with_project_for_update(
    session: AsyncSession,
    *,
    extraction_run_id: uuid.UUID,
) -> ExtractionRunProjectContext | None:
    result = await session.execute(
        select(
            ExtractionRun,
            Document.project_id.label("project_id"),
            DocumentRevision.status.label("revision_status"),
        )
        .join(DocumentRevision, ExtractionRun.revision_id == DocumentRevision.id)
        .join(Document, DocumentRevision.document_id == Document.id)
        .where(ExtractionRun.id == extraction_run_id)
        .with_for_update(of=ExtractionRun)
    )
    row = result.one_or_none()
    if row is None:
        return None
    return ExtractionRunProjectContext(
        extraction_run=row.ExtractionRun,
        project_id=row.project_id,
        revision_status=row.revision_status,
    )


async def get_orchestration_for_update(
    session: AsyncSession,
    *,
    orchestration_id: uuid.UUID,
) -> FactExtractionOrchestration | None:
    result = await session.execute(
        select(FactExtractionOrchestration)
        .where(FactExtractionOrchestration.id == orchestration_id)
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def list_orchestrations_by_request_for_update(
    session: AsyncSession,
    *,
    extraction_run_id: uuid.UUID,
    request_hash: str,
) -> list[FactExtractionOrchestration]:
    result = await session.execute(
        select(FactExtractionOrchestration)
        .where(
            FactExtractionOrchestration.extraction_run_id == extraction_run_id,
            FactExtractionOrchestration.request_hash == request_hash,
        )
        .order_by(
            FactExtractionOrchestration.attempt_no.desc(),
            FactExtractionOrchestration.created_at.desc(),
        )
        .with_for_update()
    )
    return list(result.scalars().all())


async def list_batches_for_orchestration_for_update(
    session: AsyncSession,
    *,
    orchestration_id: uuid.UUID,
) -> list[FactExtractionOrchestrationBatch]:
    result = await session.execute(
        select(FactExtractionOrchestrationBatch)
        .where(FactExtractionOrchestrationBatch.orchestration_id == orchestration_id)
        .order_by(FactExtractionOrchestrationBatch.batch_index.asc())
        .with_for_update()
    )
    return list(result.scalars().all())


async def list_batches_for_orchestration(
    session: AsyncSession,
    *,
    orchestration_id: uuid.UUID,
) -> list[FactExtractionOrchestrationBatch]:
    result = await session.execute(
        select(FactExtractionOrchestrationBatch)
        .where(FactExtractionOrchestrationBatch.orchestration_id == orchestration_id)
        .order_by(FactExtractionOrchestrationBatch.batch_index.asc())
    )
    return list(result.scalars().all())


async def get_batch_for_update(
    session: AsyncSession,
    *,
    orchestration_id: uuid.UUID,
    batch_index: int,
) -> FactExtractionOrchestrationBatch | None:
    result = await session.execute(
        select(FactExtractionOrchestrationBatch)
        .where(
            FactExtractionOrchestrationBatch.orchestration_id == orchestration_id,
            FactExtractionOrchestrationBatch.batch_index == batch_index,
        )
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def create_orchestration(
    session: AsyncSession,
    orchestration: FactExtractionOrchestration,
) -> FactExtractionOrchestration:
    session.add(orchestration)
    await session.flush()
    return orchestration


async def create_orchestration_batches(
    session: AsyncSession,
    batches: list[FactExtractionOrchestrationBatch],
) -> list[FactExtractionOrchestrationBatch]:
    session.add_all(batches)
    await session.flush()
    return batches


async def get_application_for_update(
    session: AsyncSession,
    *,
    application_id: uuid.UUID,
) -> FactExtractionBatchApplication | None:
    result = await session.execute(
        select(FactExtractionBatchApplication)
        .where(FactExtractionBatchApplication.id == application_id)
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def list_applications(
    session: AsyncSession,
    *,
    application_ids: list[uuid.UUID],
) -> list[FactExtractionBatchApplication]:
    if not application_ids:
        return []
    result = await session.execute(
        select(FactExtractionBatchApplication)
        .where(FactExtractionBatchApplication.id.in_(application_ids))
    )
    return list(result.scalars().all())


async def get_batch_attempt_reconciliation_context_for_update(
    session: AsyncSession,
    *,
    orchestration_id: uuid.UUID,
    batch_index: int,
) -> BatchAttemptReconciliationContext | None:
    batch_application = aliased(FactExtractionBatchApplication)
    run_application = aliased(FactExtractionBatchApplication)
    result = await session.execute(
        select(
            FactExtractionOrchestration.id.label("orchestration_id"),
            FactExtractionOrchestration.status.label("orchestration_status"),
            FactExtractionOrchestration.project_id.label("orchestration_project_id"),
            FactExtractionOrchestration.extraction_run_id.label("orchestration_extraction_run_id"),
            FactExtractionOrchestrationBatch.id.label("batch_id"),
            FactExtractionOrchestrationBatch.batch_index.label("batch_index"),
            FactExtractionOrchestrationBatch.status.label("batch_status"),
            FactExtractionOrchestrationBatch.attempt_count.label("attempt_count"),
            FactExtractionOrchestrationBatch.lease_token.label("lease_token"),
            FactExtractionOrchestrationBatch.current_input_batch_id.label("input_batch_id"),
            FactExtractionOrchestrationBatch.current_inference_run_id.label("inference_run_id"),
            ExtractionRun.status.label("inference_run_status"),
            ExtractionRun.project_id.label("inference_run_project_id"),
            ExtractionRun.task_type.label("inference_run_task_type"),
            ExtractionRun.input_batch_id.label("inference_run_input_batch_id"),
            ExtractionRun.failure_code.label("inference_run_failure_code"),
            batch_application.id.label("batch_application_id"),
            batch_application.status.label("batch_application_status"),
            run_application.id.label("run_application_id"),
            run_application.status.label("run_application_status"),
        )
        .select_from(FactExtractionOrchestration)
        .join(
            FactExtractionOrchestrationBatch,
            FactExtractionOrchestrationBatch.orchestration_id == FactExtractionOrchestration.id,
        )
        .outerjoin(
            ExtractionRun,
            ExtractionRun.id == FactExtractionOrchestrationBatch.current_inference_run_id,
        )
        .outerjoin(
            batch_application,
            batch_application.id == FactExtractionOrchestrationBatch.application_id,
        )
        .outerjoin(
            run_application,
            run_application.inference_run_id == ExtractionRun.id,
        )
        .where(
            FactExtractionOrchestration.id == orchestration_id,
            FactExtractionOrchestrationBatch.batch_index == batch_index,
        )
        .with_for_update(
            of=(
                FactExtractionOrchestration,
                FactExtractionOrchestrationBatch,
                ExtractionRun,
                batch_application,
                run_application,
            )
        )
    )
    row = result.one_or_none()
    if row is None:
        return None
    if row.batch_application_id is not None and row.run_application_id is not None:
        if row.batch_application_id != row.run_application_id:
            raise RuntimeError("batch application binding conflicts with inference-run application")
    application_id = row.batch_application_id or row.run_application_id
    application_status = row.batch_application_status or row.run_application_status
    return BatchAttemptReconciliationContext(
        orchestration_id=row.orchestration_id,
        orchestration_status=row.orchestration_status,
        orchestration_project_id=row.orchestration_project_id,
        orchestration_extraction_run_id=row.orchestration_extraction_run_id,
        batch_id=row.batch_id,
        batch_index=row.batch_index,
        batch_status=row.batch_status,
        attempt_count=row.attempt_count,
        lease_token=row.lease_token,
        input_batch_id=row.input_batch_id,
        inference_run_id=row.inference_run_id,
        inference_run_status=row.inference_run_status,
        inference_run_project_id=row.inference_run_project_id,
        inference_run_task_type=row.inference_run_task_type,
        inference_run_input_batch_id=row.inference_run_input_batch_id,
        inference_run_failure_code=row.inference_run_failure_code,
        application_id=application_id,
        application_status=application_status,
        batch_application_id=row.batch_application_id,
        batch_application_status=row.batch_application_status,
        run_application_id=row.run_application_id,
        run_application_status=row.run_application_status,
    )
