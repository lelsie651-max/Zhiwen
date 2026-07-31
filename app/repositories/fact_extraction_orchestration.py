from __future__ import annotations

from dataclasses import dataclass
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_content import ExtractionRun as DocumentExtractionRun
from app.models.document_revision import DocumentRevision
from app.models.fact_extraction_application import FactExtractionBatchApplication
from app.models.fact_extraction_orchestration import (
    FactExtractionOrchestration,
    FactExtractionOrchestrationBatch,
)
from app.models.inference import InferenceRun


@dataclass(frozen=True, slots=True)
class ExtractionRunProjectContext:
    extraction_run: DocumentExtractionRun
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
            DocumentExtractionRun,
            Document.project_id.label("project_id"),
            DocumentRevision.status.label("revision_status"),
        )
        .join(DocumentRevision, DocumentExtractionRun.revision_id == DocumentRevision.id)
        .join(Document, DocumentRevision.document_id == Document.id)
        .where(DocumentExtractionRun.id == extraction_run_id)
        .with_for_update(of=DocumentExtractionRun)
    )
    row = result.one_or_none()
    if row is None:
        return None
    extraction_run = row[0]
    return ExtractionRunProjectContext(
        extraction_run=extraction_run,
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


async def get_application_by_inference_run_for_update(
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
