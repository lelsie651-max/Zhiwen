from __future__ import annotations

from datetime import datetime, timedelta, timezone
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document_content import ExtractionRun, ExtractionRunOutcome
from app.models.document_revision import DocumentRevision, DocumentRevisionStatus
from app.models.processing_job import (
    ProcessingJob,
    ProcessingJobStatus,
    ProcessingJobType,
    ProcessingTriggerKind,
)
from app.models.project_member import ProjectMemberRole
from app.repositories import processing_job as processing_job_repository

LEASE_EXPIRED_FAILURE_CODE = "lease_expired"
LEASE_EXPIRED_FAILURE_MESSAGE = "Worker lease expired before extraction completed."
ORPHANED_EXTRACTION_RESULT_FAILURE_CODE = "orphaned_extraction_result"
ORPHANED_EXTRACTION_RESULT_FAILURE_MESSAGE = (
    "A persisted extraction result could not be linked deterministically to the stale processing job."
)
PROCESSING_RESULT_STATE_MISMATCH_CODE = "processing_result_state_mismatch"


class ProcessingJobError(Exception):
    """Raised when processing job orchestration cannot proceed."""


class ProcessingJobNotFoundError(ProcessingJobError):
    """Raised when the target revision or job cannot be found in the project."""


class ProcessingJobPermissionError(ProcessingJobError):
    """Raised when the actor cannot create or recover processing jobs."""


class ProcessingJobStateError(ProcessingJobError):
    """Raised when revision or job state prevents the requested transition."""


class ProcessingJobLeaseError(ProcessingJobError):
    """Raised when a lease token is invalid or expired."""


async def enqueue_revision_extraction_job(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    revision_id: uuid.UUID,
    trigger_kind: ProcessingTriggerKind | str,
    actor_id: uuid.UUID | None = None,
) -> ProcessingJob:
    normalized_trigger_kind = ProcessingTriggerKind(trigger_kind)
    revision = await _lock_revision_for_project(
        session,
        project_id=project_id,
        revision_id=revision_id,
    )
    if revision.status != DocumentRevisionStatus.ACCEPTED.value:
        raise ProcessingJobStateError("Only accepted revisions can be queued for extraction.")

    active_job = await processing_job_repository.get_active_processing_job_for_update(
        session,
        revision_id=revision.id,
        job_type=ProcessingJobType.REVISION_EXTRACTION.value,
    )
    if active_job is not None:
        return active_job

    requested_by_id = await _resolve_enqueue_requester(
        session,
        project_id=project_id,
        trigger_kind=normalized_trigger_kind,
        actor_id=actor_id,
    )
    job = await _create_queued_job(
        session,
        project_id=project_id,
        revision_id=revision.id,
        trigger_kind=normalized_trigger_kind,
        requested_by_id=requested_by_id,
    )
    try:
        await session.commit()
    except BaseException:
        await session.rollback()
        raise
    return job


async def retry_failed_revision_extraction(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    revision_id: uuid.UUID,
    actor_id: uuid.UUID,
) -> ProcessingJob:
    actor = await _require_active_owner_or_editor(
        session,
        project_id=project_id,
        actor_id=actor_id,
    )
    revision = await _lock_revision_for_project(
        session,
        project_id=project_id,
        revision_id=revision_id,
    )
    if revision.status != DocumentRevisionStatus.FAILED.value:
        raise ProcessingJobStateError("Only failed revisions can be retried.")

    active_job = await processing_job_repository.get_active_processing_job_for_update(
        session,
        revision_id=revision.id,
        job_type=ProcessingJobType.REVISION_EXTRACTION.value,
    )
    if active_job is not None:
        raise ProcessingJobStateError("Active extraction jobs block retry.")

    latest_run = await processing_job_repository.get_latest_extraction_run_for_revision(
        session,
        revision_id=revision.id,
    )
    latest_failed_job = await processing_job_repository.get_latest_failed_processing_job(
        session,
        revision_id=revision.id,
        job_type=ProcessingJobType.REVISION_EXTRACTION.value,
    )
    has_retryable_run = latest_run is not None and latest_run.outcome in {
        ExtractionRunOutcome.FAILED.value,
        ExtractionRunOutcome.NEEDS_OCR.value,
    }
    has_retryable_job = (
        latest_failed_job is not None and latest_failed_job.result_extraction_run_id is None
    )
    if not (has_retryable_run or has_retryable_job):
        raise ProcessingJobStateError("Retry requires a failed extraction run or failed processing job.")

    revision.status = DocumentRevisionStatus.ACCEPTED.value
    job = await _create_queued_job(
        session,
        project_id=project_id,
        revision_id=revision.id,
        trigger_kind=ProcessingTriggerKind.RETRY,
        requested_by_id=actor.id,
    )
    try:
        await session.flush()
        await session.commit()
    except BaseException:
        await session.rollback()
        raise
    return job


async def recover_stale_revision_extraction(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    revision_id: uuid.UUID,
    actor_id: uuid.UUID,
    stale_before: datetime,
) -> ProcessingJob:
    actor = await _require_active_owner_or_editor(
        session,
        project_id=project_id,
        actor_id=actor_id,
    )
    revision = await _lock_revision_for_project(
        session,
        project_id=project_id,
        revision_id=revision_id,
    )
    now = datetime.now(timezone.utc)
    active_job = await processing_job_repository.get_active_processing_job_for_update(
        session,
        revision_id=revision.id,
        job_type=ProcessingJobType.REVISION_EXTRACTION.value,
    )
    try:
        if active_job is not None:
            if active_job.status != ProcessingJobStatus.RUNNING.value:
                raise ProcessingJobStateError("Queued extraction jobs cannot be recovered as stale.")
            if active_job.lease_expires_at is None or active_job.lease_expires_at > now:
                raise ProcessingJobStateError("Running job lease has not expired.")

            result_context = await processing_job_repository.get_processing_extraction_result_context(
                session,
                job_id=active_job.id,
            )
            if result_context is None:
                raise ProcessingJobNotFoundError("Processing job was not found during recovery.")

            if result_context.result_run_id is not None:
                _validate_result_context_consistency(result_context)
                active_job.status = ProcessingJobStatus.COMPLETED.value
                active_job.completed_at = now
                active_job.lease_token = None
                active_job.lease_expires_at = None
                active_job.failure_code = None
                active_job.failure_message = None
                await session.flush()
                await session.commit()
                return active_job

            has_terminal_run = await processing_job_repository.revision_has_terminal_extraction_run(
                session,
                revision_id=revision.id,
            )
            if _revision_has_orphaned_result(revision_status=revision.status, has_terminal_run=has_terminal_run):
                _mark_job_failed(
                    active_job,
                    now=now,
                    failure_code=ORPHANED_EXTRACTION_RESULT_FAILURE_CODE,
                    failure_message=ORPHANED_EXTRACTION_RESULT_FAILURE_MESSAGE,
                )
                await session.flush()
                await session.commit()
                raise ProcessingJobStateError(ORPHANED_EXTRACTION_RESULT_FAILURE_CODE)

            _mark_job_failed(
                active_job,
                now=now,
                failure_code=LEASE_EXPIRED_FAILURE_CODE,
                failure_message=LEASE_EXPIRED_FAILURE_MESSAGE,
            )
            if revision.status in {
                DocumentRevisionStatus.PARSING.value,
                DocumentRevisionStatus.EXTRACTING.value,
            }:
                revision.status = DocumentRevisionStatus.ACCEPTED.value

            job = await _create_queued_job(
                session,
                project_id=project_id,
                revision_id=revision.id,
                trigger_kind=ProcessingTriggerKind.RECOVERY,
                requested_by_id=actor.id,
            )
            await session.flush()
            await session.commit()
            return job

        if revision.status not in {
            DocumentRevisionStatus.ACCEPTED.value,
            DocumentRevisionStatus.PARSING.value,
            DocumentRevisionStatus.EXTRACTING.value,
        }:
            raise ProcessingJobStateError("Recovery only supports accepted, parsing, or extracting revisions.")
        if revision.updated_at > stale_before:
            raise ProcessingJobStateError("Revision is not stale enough to recover.")

        has_terminal_run = await processing_job_repository.revision_has_terminal_extraction_run(
            session,
            revision_id=revision.id,
        )
        if _revision_has_orphaned_result(revision_status=revision.status, has_terminal_run=has_terminal_run):
            raise ProcessingJobStateError(ORPHANED_EXTRACTION_RESULT_FAILURE_CODE)

        if revision.status in {
            DocumentRevisionStatus.PARSING.value,
            DocumentRevisionStatus.EXTRACTING.value,
        }:
            revision.status = DocumentRevisionStatus.ACCEPTED.value

        job = await _create_queued_job(
            session,
            project_id=project_id,
            revision_id=revision.id,
            trigger_kind=ProcessingTriggerKind.RECOVERY,
            requested_by_id=actor.id,
        )
        await session.flush()
        await session.commit()
        return job
    except BaseException:
        await session.rollback()
        raise


async def claim_processing_job(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    job_id: uuid.UUID,
    lease_seconds: int,
) -> ProcessingJob:
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be greater than 0.")

    job = await _lock_processing_job_for_project(
        session,
        project_id=project_id,
        job_id=job_id,
    )
    if job.status != ProcessingJobStatus.QUEUED.value:
        raise ProcessingJobStateError("Only queued jobs can be claimed.")

    now = datetime.now(timezone.utc)
    job.status = ProcessingJobStatus.RUNNING.value
    job.lease_token = uuid.uuid4()
    job.started_at = now
    job.lease_expires_at = now + timedelta(seconds=lease_seconds)
    try:
        await session.flush()
        await session.commit()
    except BaseException:
        await session.rollback()
        raise
    return job


async def complete_processing_job(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    job_id: uuid.UUID,
    lease_token: uuid.UUID,
    extraction_run_id: uuid.UUID,
) -> ProcessingJob:
    try:
        job = await complete_processing_job_in_transaction(
            session,
            project_id=project_id,
            job_id=job_id,
            lease_token=lease_token,
            extraction_run_id=extraction_run_id,
        )
        await session.flush()
        await session.commit()
    except BaseException:
        await session.rollback()
        raise
    return job


async def fail_processing_job(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    job_id: uuid.UUID,
    lease_token: uuid.UUID,
    failure_code: str,
    failure_message: str,
) -> ProcessingJob:
    try:
        job = await fail_processing_job_in_transaction(
            session,
            project_id=project_id,
            job_id=job_id,
            lease_token=lease_token,
            failure_code=failure_code,
            failure_message=failure_message,
        )
        await session.flush()
        await session.commit()
    except BaseException:
        await session.rollback()
        raise
    return job


async def renew_processing_job_lease(
    session: AsyncSession,
    *,
    job_id: uuid.UUID,
    lease_token: uuid.UUID,
    lease_seconds: int,
) -> ProcessingJob:
    if lease_seconds < 30:
        raise ValueError("lease_seconds must be greater than or equal to 30.")

    job = await _lock_processing_job_by_id(session, job_id=job_id)
    now = datetime.now(timezone.utc)
    _validate_running_job_lease(job, lease_token=lease_token, now=now)
    job.lease_expires_at = now + timedelta(seconds=lease_seconds)
    try:
        await session.flush()
        await session.commit()
    except BaseException:
        await session.rollback()
        raise
    return job


async def complete_processing_job_in_transaction(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    job_id: uuid.UUID,
    lease_token: uuid.UUID,
    extraction_run_id: uuid.UUID,
) -> ProcessingJob:
    job = await _lock_processing_job_for_project(
        session,
        project_id=project_id,
        job_id=job_id,
    )
    now = datetime.now(timezone.utc)
    extraction_run = await processing_job_repository.get_extraction_run_by_id_for_update(
        session,
        extraction_run_id=extraction_run_id,
    )
    if extraction_run is None:
        raise ProcessingJobStateError("Extraction run not found.")
    if job.job_type != ProcessingJobType.REVISION_EXTRACTION.value:
        raise ProcessingJobStateError("Only revision extraction jobs can store extraction run results.")
    if extraction_run.revision_id != job.revision_id:
        raise ProcessingJobStateError("Extraction run must belong to the same revision as the job.")
    if job.result_extraction_run_id is not None and job.result_extraction_run_id != extraction_run.id:
        raise ProcessingJobStateError("Processing job is already bound to another extraction run.")

    if job.status == ProcessingJobStatus.COMPLETED.value:
        if job.result_extraction_run_id == extraction_run.id:
            return job
        raise ProcessingJobStateError("Completed processing jobs cannot be rebound to another extraction run.")

    _validate_running_job_lease(job, lease_token=lease_token, now=now)
    job.status = ProcessingJobStatus.COMPLETED.value
    job.completed_at = now
    job.result_extraction_run_id = extraction_run.id
    job.lease_token = None
    job.lease_expires_at = None
    job.failure_code = None
    job.failure_message = None
    return job


async def fail_processing_job_in_transaction(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    job_id: uuid.UUID,
    lease_token: uuid.UUID,
    failure_code: str,
    failure_message: str,
) -> ProcessingJob:
    job = await _lock_processing_job_for_project(
        session,
        project_id=project_id,
        job_id=job_id,
    )
    now = datetime.now(timezone.utc)
    _validate_running_job_lease(job, lease_token=lease_token, now=now)
    _mark_job_failed(
        job,
        now=now,
        failure_code=failure_code,
        failure_message=failure_message,
    )
    return job


async def _lock_revision_for_project(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    revision_id: uuid.UUID,
) -> DocumentRevision:
    revision = await processing_job_repository.get_revision_for_processing_job_update(
        session,
        project_id=project_id,
        revision_id=revision_id,
    )
    if revision is None:
        raise ProcessingJobNotFoundError("Revision must belong to the target project.")
    return revision


async def _lock_processing_job_for_project(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    job_id: uuid.UUID,
) -> ProcessingJob:
    job = await processing_job_repository.get_processing_job_for_update(
        session,
        project_id=project_id,
        job_id=job_id,
    )
    if job is None:
        raise ProcessingJobNotFoundError("Processing job must belong to the target project.")
    return job


async def _lock_processing_job_by_id(
    session: AsyncSession,
    *,
    job_id: uuid.UUID,
) -> ProcessingJob:
    job = await processing_job_repository.get_processing_job_by_id_for_update(
        session,
        job_id=job_id,
    )
    if job is None:
        raise ProcessingJobNotFoundError("Processing job was not found.")
    return job


async def _require_active_owner_or_editor(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    actor_id: uuid.UUID,
) -> object:
    actor = await processing_job_repository.get_active_user_by_id(
        session,
        user_id=actor_id,
    )
    if actor is None:
        raise ProcessingJobPermissionError("Only active users can manage extraction jobs.")
    membership = await processing_job_repository.get_project_member_for_project(
        session,
        project_id=project_id,
        user_id=actor.id,
    )
    if membership is None or membership.role not in {
        ProjectMemberRole.OWNER.value,
        ProjectMemberRole.EDITOR.value,
    }:
        raise ProcessingJobPermissionError("Only owners and editors can manage extraction jobs.")
    return actor


async def _resolve_enqueue_requester(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    trigger_kind: ProcessingTriggerKind,
    actor_id: uuid.UUID | None,
) -> uuid.UUID | None:
    if trigger_kind == ProcessingTriggerKind.AUTOMATIC:
        return None
    if actor_id is None:
        raise ProcessingJobPermissionError("Manual, retry, and recovery jobs require an actor.")
    actor = await _require_active_owner_or_editor(
        session,
        project_id=project_id,
        actor_id=actor_id,
    )
    return actor.id


async def _create_queued_job(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    revision_id: uuid.UUID,
    trigger_kind: ProcessingTriggerKind,
    requested_by_id: uuid.UUID | None,
) -> ProcessingJob:
    attempt_no = await processing_job_repository.get_next_processing_job_attempt_no(
        session,
        revision_id=revision_id,
        job_type=ProcessingJobType.REVISION_EXTRACTION.value,
    )
    job = ProcessingJob(
        project_id=project_id,
        revision_id=revision_id,
        job_type=ProcessingJobType.REVISION_EXTRACTION.value,
        status=ProcessingJobStatus.QUEUED.value,
        trigger_kind=trigger_kind.value,
        attempt_no=attempt_no,
        requested_by_id=requested_by_id,
        lease_token=None,
        lease_expires_at=None,
        started_at=None,
        completed_at=None,
        result_extraction_run_id=None,
        failure_code=None,
        failure_message=None,
    )
    return await processing_job_repository.create_processing_job(session, job=job)


def _reject_recovery_for_terminal_revision_status(revision: DocumentRevision) -> None:
    if revision.status == DocumentRevisionStatus.FAILED.value:
        raise ProcessingJobStateError("Failed revisions must use the retry service.")
    if revision.status in {
        DocumentRevisionStatus.AWAITING_REVIEW.value,
        DocumentRevisionStatus.COMPLETED.value,
        DocumentRevisionStatus.REJECTED.value,
        DocumentRevisionStatus.QUARANTINED.value,
        DocumentRevisionStatus.NEEDS_CONFIRMATION.value,
    }:
        raise ProcessingJobStateError("Recovery is not allowed for terminal revision states.")


def _validate_running_job_lease(
    job: ProcessingJob,
    *,
    lease_token: uuid.UUID,
    now: datetime,
) -> None:
    if job.status != ProcessingJobStatus.RUNNING.value:
        raise ProcessingJobStateError("Only running jobs can be completed or failed.")
    if job.lease_token is None or job.lease_expires_at is None:
        raise ProcessingJobLeaseError("Running job is missing lease metadata.")
    if job.lease_token != lease_token:
        raise ProcessingJobLeaseError("Lease token does not match the running job.")
    if job.lease_expires_at <= now:
        raise ProcessingJobLeaseError("Lease has already expired.")


def _mark_job_failed(
    job: ProcessingJob,
    *,
    now: datetime,
    failure_code: str,
    failure_message: str,
) -> None:
    job.status = ProcessingJobStatus.FAILED.value
    job.completed_at = now
    job.result_extraction_run_id = None
    job.lease_token = None
    job.lease_expires_at = None
    job.failure_code = failure_code
    job.failure_message = failure_message


def _validate_result_context_consistency(
    context: processing_job_repository.ProcessingExtractionResultContext,
) -> None:
    if context.job_type != ProcessingJobType.REVISION_EXTRACTION.value:
        raise ProcessingJobStateError(PROCESSING_RESULT_STATE_MISMATCH_CODE)
    if context.result_run_id is None or context.run_revision_id is None:
        raise ProcessingJobStateError(PROCESSING_RESULT_STATE_MISMATCH_CODE)
    if context.run_revision_id != context.job_revision_id:
        raise ProcessingJobStateError(PROCESSING_RESULT_STATE_MISMATCH_CODE)
    if context.run_status not in {"completed", "failed"}:
        raise ProcessingJobStateError(PROCESSING_RESULT_STATE_MISMATCH_CODE)

    expected_revision_status = _map_extraction_outcome_to_revision_status(context.run_outcome)
    if context.revision_status != expected_revision_status:
        raise ProcessingJobStateError(PROCESSING_RESULT_STATE_MISMATCH_CODE)


def _map_extraction_outcome_to_revision_status(run_outcome: str | None) -> str:
    if run_outcome in {
        ExtractionRunOutcome.SUCCESS.value,
        ExtractionRunOutcome.PARTIAL.value,
    }:
        return DocumentRevisionStatus.AWAITING_REVIEW.value
    if run_outcome in {
        ExtractionRunOutcome.NEEDS_OCR.value,
        ExtractionRunOutcome.FAILED.value,
    }:
        return DocumentRevisionStatus.FAILED.value
    raise ProcessingJobStateError(PROCESSING_RESULT_STATE_MISMATCH_CODE)


def _revision_has_orphaned_result(*, revision_status: str, has_terminal_run: bool) -> bool:
    if has_terminal_run:
        return True
    return revision_status in {
        DocumentRevisionStatus.AWAITING_REVIEW.value,
        DocumentRevisionStatus.COMPLETED.value,
        DocumentRevisionStatus.FAILED.value,
    }
