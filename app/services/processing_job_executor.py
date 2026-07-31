from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document_revision import DocumentRevisionStatus
from app.models.processing_job import ProcessingJobStatus
from app.repositories import processing_job as processing_job_repository
from app.schemas.processing_job_executor import (
    ProcessingJobExecutionResult,
    ProcessingJobExecutionStatus,
)
from app.services.processing_job import (
    ProcessingJobLeaseError,
    ProcessingJobNotFoundError,
    ProcessingJobStateError,
    claim_processing_job,
    complete_processing_job_in_transaction,
    fail_processing_job,
    renew_processing_job_lease,
)
from app.services.revision_extraction import run_revision_extraction
from app.storage import FileStorage

logger = logging.getLogger(__name__)

WORKER_FAILURE_CODE = "worker_execution_failed"
WORKER_FAILURE_MESSAGE = "Revision extraction worker failed before producing a persisted result."
LEASE_LOST_DURING_COMPLETION = "lease_lost_during_completion"
LEASE_LOST_DURING_FAILURE_HANDLING = "lease_lost_during_failure_handling"


@dataclass(frozen=True)
class _ProcessingJobSnapshot:
    job_id: uuid.UUID
    project_id: uuid.UUID
    revision_id: uuid.UUID
    job_status: ProcessingJobStatus
    extraction_run_id: uuid.UUID | None
    revision_status: DocumentRevisionStatus
    failure_code: str | None


async def execute_revision_extraction_job(
    session_factory: Callable[[], AsyncSession],
    *,
    job_id: uuid.UUID,
    storage: FileStorage,
    lease_seconds: int = 900,
    heartbeat_seconds: float | None = None,
) -> ProcessingJobExecutionResult:
    if lease_seconds < 30:
        raise ValueError("lease_seconds must be greater than or equal to 30.")
    heartbeat_interval = _resolve_heartbeat_seconds(
        lease_seconds=lease_seconds,
        heartbeat_seconds=heartbeat_seconds,
    )

    initial_snapshot = await _load_job_snapshot(session_factory, job_id=job_id)
    try:
        claimed_job = await _claim_job(
            session_factory,
            job_id=job_id,
            project_id=initial_snapshot.project_id,
            lease_seconds=lease_seconds,
        )
    except ProcessingJobStateError:
        current_snapshot = await _load_job_snapshot(session_factory, job_id=job_id)
        return _build_skipped_result(current_snapshot)

    copied_job_id = claimed_job.id
    copied_project_id = claimed_job.project_id
    copied_revision_id = claimed_job.revision_id
    copied_lease_token = claimed_job.lease_token
    if copied_lease_token is None:
        raise ProcessingJobLeaseError("Claimed processing job did not return a lease token.")

    heartbeat_stop = asyncio.Event()
    heartbeat_task = asyncio.create_task(
        _heartbeat_loop(
            session_factory,
            job_id=copied_job_id,
            revision_id=copied_revision_id,
            lease_token=copied_lease_token,
            lease_seconds=lease_seconds,
            heartbeat_seconds=heartbeat_interval,
            stop_event=heartbeat_stop,
        )
    )

    try:
        extraction_result = await _run_revision_extraction(
            session_factory,
            project_id=copied_project_id,
            revision_id=copied_revision_id,
            job_id=copied_job_id,
            lease_token=copied_lease_token,
            storage=storage,
        )
    except asyncio.CancelledError:
        await _cancel_heartbeat_task(heartbeat_task)
        raise
    except (ProcessingJobLeaseError, ProcessingJobStateError):
        await _stop_heartbeat_task(heartbeat_task, heartbeat_stop)
        logger.warning(
            "Processing job completion lost its lease or state precondition: "
            "job_id=%s revision_id=%s failure_code=%s",
            copied_job_id,
            copied_revision_id,
            LEASE_LOST_DURING_COMPLETION,
        )
        current_snapshot = await _load_job_snapshot(session_factory, job_id=copied_job_id)
        return ProcessingJobExecutionResult(
            job_id=copied_job_id,
            revision_id=copied_revision_id,
            execution_status=ProcessingJobExecutionStatus.FAILED,
            job_status=current_snapshot.job_status,
            extraction_run_id=current_snapshot.extraction_run_id,
            revision_status=current_snapshot.revision_status,
            failure_code=current_snapshot.failure_code or LEASE_LOST_DURING_COMPLETION,
            skip_reason=None,
        )
    except Exception:
        await _stop_heartbeat_task(heartbeat_task, heartbeat_stop)
        logger.warning(
            "Processing job execution failed before a persisted result was produced: "
            "job_id=%s revision_id=%s failure_code=%s",
            copied_job_id,
            copied_revision_id,
            WORKER_FAILURE_CODE,
        )
        return await _handle_execution_failure(
            session_factory,
            job_id=copied_job_id,
            project_id=copied_project_id,
            revision_id=copied_revision_id,
            lease_token=copied_lease_token,
        )

    await _stop_heartbeat_task(heartbeat_task, heartbeat_stop)

    return ProcessingJobExecutionResult(
        job_id=copied_job_id,
        revision_id=copied_revision_id,
        execution_status=ProcessingJobExecutionStatus.COMPLETED,
        job_status=ProcessingJobStatus.COMPLETED,
        extraction_run_id=extraction_result.extraction_run_id,
        revision_status=DocumentRevisionStatus(extraction_result.revision_status),
        failure_code=None,
        skip_reason=None,
    )


async def _claim_job(
    session_factory: Callable[[], AsyncSession],
    *,
    job_id: uuid.UUID,
    project_id: uuid.UUID,
    lease_seconds: int,
):
    async with session_factory() as session:
        return await claim_processing_job(
            session,
            project_id=project_id,
            job_id=job_id,
            lease_seconds=lease_seconds,
        )


async def _run_revision_extraction(
    session_factory: Callable[[], AsyncSession],
    *,
    project_id: uuid.UUID,
    revision_id: uuid.UUID,
    job_id: uuid.UUID,
    lease_token: uuid.UUID,
    storage: FileStorage,
):
    async with session_factory() as session:
        return await run_revision_extraction(
            session,
            project_id=project_id,
            revision_id=revision_id,
            storage=storage,
            finalizer=lambda finalize_session, _revision, extraction_run: complete_processing_job_in_transaction(
                finalize_session,
                project_id=project_id,
                job_id=job_id,
                lease_token=lease_token,
                extraction_run_id=extraction_run.id,
            ),
        )


async def _heartbeat_loop(
    session_factory: Callable[[], AsyncSession],
    *,
    job_id: uuid.UUID,
    revision_id: uuid.UUID,
    lease_token: uuid.UUID,
    lease_seconds: int,
    heartbeat_seconds: float,
    stop_event: asyncio.Event,
) -> None:
    while True:
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=heartbeat_seconds)
            return
        except TimeoutError:
            pass

        try:
            async with session_factory() as session:
                await renew_processing_job_lease(
                    session,
                    job_id=job_id,
                    lease_token=lease_token,
                    lease_seconds=lease_seconds,
                )
        except Exception:
            logger.warning(
                "Processing job heartbeat stopped after lease renewal failure: "
                "job_id=%s revision_id=%s failure_code=lease_renewal_failed",
                job_id,
                revision_id,
            )
            return


async def _handle_execution_failure(
    session_factory: Callable[[], AsyncSession],
    *,
    job_id: uuid.UUID,
    project_id: uuid.UUID,
    revision_id: uuid.UUID,
    lease_token: uuid.UUID,
) -> ProcessingJobExecutionResult:
    try:
        async with session_factory() as session:
            failed_job = await fail_processing_job(
                session,
                project_id=project_id,
                job_id=job_id,
                lease_token=lease_token,
                failure_code=WORKER_FAILURE_CODE,
                failure_message=WORKER_FAILURE_MESSAGE,
            )
        current_snapshot = await _load_job_snapshot(session_factory, job_id=job_id)
        return ProcessingJobExecutionResult(
            job_id=job_id,
            revision_id=revision_id,
            execution_status=ProcessingJobExecutionStatus.FAILED,
            job_status=ProcessingJobStatus(failed_job.status),
            extraction_run_id=None,
            revision_status=current_snapshot.revision_status,
            failure_code=failed_job.failure_code,
            skip_reason=None,
        )
    except (ProcessingJobLeaseError, ProcessingJobStateError):
        logger.warning(
            "Processing job failure handling lost its lease or state precondition: "
            "job_id=%s revision_id=%s failure_code=%s",
            job_id,
            revision_id,
            LEASE_LOST_DURING_FAILURE_HANDLING,
        )
        current_snapshot = await _load_job_snapshot(session_factory, job_id=job_id)
        return ProcessingJobExecutionResult(
            job_id=job_id,
            revision_id=revision_id,
            execution_status=ProcessingJobExecutionStatus.FAILED,
            job_status=current_snapshot.job_status,
            extraction_run_id=current_snapshot.extraction_run_id,
            revision_status=current_snapshot.revision_status,
            failure_code=current_snapshot.failure_code or LEASE_LOST_DURING_FAILURE_HANDLING,
            skip_reason=None,
        )


async def _load_job_snapshot(
    session_factory: Callable[[], AsyncSession],
    *,
    job_id: uuid.UUID,
) -> _ProcessingJobSnapshot:
    async with session_factory() as session:
        job = await processing_job_repository.get_processing_job_by_id(
            session,
            job_id=job_id,
        )
        if job is None:
            raise ProcessingJobNotFoundError("Processing job was not found.")
        revision = await processing_job_repository.get_revision_by_id(
            session,
            revision_id=job.revision_id,
        )
        if revision is None:
            raise ProcessingJobNotFoundError("Revision referenced by the processing job was not found.")
        return _ProcessingJobSnapshot(
            job_id=job.id,
            project_id=job.project_id,
            revision_id=job.revision_id,
            job_status=ProcessingJobStatus(job.status),
            extraction_run_id=job.result_extraction_run_id,
            revision_status=DocumentRevisionStatus(revision.status),
            failure_code=job.failure_code,
        )


def _resolve_heartbeat_seconds(
    *,
    lease_seconds: int,
    heartbeat_seconds: float | None,
) -> float:
    if heartbeat_seconds is None:
        return float(min(60, max(5, lease_seconds // 3)))
    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be greater than 0.")
    if heartbeat_seconds >= lease_seconds:
        raise ValueError("heartbeat_seconds must be less than lease_seconds.")
    return float(heartbeat_seconds)


def _build_skipped_result(snapshot: _ProcessingJobSnapshot) -> ProcessingJobExecutionResult:
    return ProcessingJobExecutionResult(
        job_id=snapshot.job_id,
        revision_id=snapshot.revision_id,
        execution_status=ProcessingJobExecutionStatus.SKIPPED,
        job_status=snapshot.job_status,
        extraction_run_id=snapshot.extraction_run_id,
        revision_status=snapshot.revision_status,
        failure_code=snapshot.failure_code,
        skip_reason=_map_skip_reason(snapshot.job_status),
    )


def _map_skip_reason(job_status: ProcessingJobStatus) -> str:
    if job_status == ProcessingJobStatus.RUNNING:
        return "already_running"
    if job_status == ProcessingJobStatus.COMPLETED:
        return "already_completed"
    if job_status == ProcessingJobStatus.FAILED:
        return "already_failed"
    if job_status == ProcessingJobStatus.CANCELLED:
        return "already_cancelled"
    return "not_queueable"


async def _stop_heartbeat_task(task: asyncio.Task[None], stop_event: asyncio.Event) -> None:
    stop_event.set()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


async def _cancel_heartbeat_task(task: asyncio.Task[None]) -> None:
    task.cancel()
    with suppress(asyncio.CancelledError, Exception):
        await task
