import asyncio
import inspect
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.models.document_revision import DocumentRevision, DocumentRevisionStatus, SourceAuthority, UploadIntent
from app.models.processing_job import ProcessingJob, ProcessingJobStatus, ProcessingJobType, ProcessingTriggerKind
from app.schemas.processing_job_executor import ProcessingJobExecutionStatus
from app.schemas.revision_extraction import RevisionExtractionResult
from app.services import processing_job as processing_job_service
from app.services import processing_job_executor as processing_job_executor_service
from app.services.processing_job_executor import (
    LEASE_LOST_DURING_COMPLETION,
    WORKER_FAILURE_CODE,
    WORKER_FAILURE_MESSAGE,
    execute_revision_extraction_job,
)


def run_async(awaitable):
    return asyncio.run(awaitable)


def build_revision(
    *,
    revision_id: uuid.UUID | None = None,
    status: str = DocumentRevisionStatus.ACCEPTED.value,
) -> DocumentRevision:
    now = datetime.now(timezone.utc)
    return DocumentRevision(
        id=revision_id or uuid.uuid4(),
        document_id=uuid.uuid4(),
        revision_no=1,
        upload_intent=UploadIntent.NEW_DOCUMENT.value,
        supersedes_revision_id=None,
        original_filename="sample.pdf",
        storage_key="files/aa/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.pdf",
        mime_type="application/pdf",
        file_size_bytes=100,
        sha256="a" * 64,
        detected_language=None,
        language_confidence=None,
        source_authority=SourceAuthority.PENDING.value,
        status=status,
        uploaded_by_id=uuid.uuid4(),
        uploaded_at=now,
        updated_at=now,
    )


def build_processing_job(
    *,
    project_id: uuid.UUID,
    revision_id: uuid.UUID,
    status: str = ProcessingJobStatus.QUEUED.value,
    trigger_kind: str = ProcessingTriggerKind.AUTOMATIC.value,
    lease_token: uuid.UUID | None = None,
    lease_expires_at: datetime | None = None,
    result_extraction_run_id: uuid.UUID | None = None,
    failure_code: str | None = None,
) -> ProcessingJob:
    now = datetime.now(timezone.utc)
    started_at = now - timedelta(minutes=1) if status == ProcessingJobStatus.RUNNING.value else None
    completed_at = now if status in {
        ProcessingJobStatus.COMPLETED.value,
        ProcessingJobStatus.FAILED.value,
        ProcessingJobStatus.CANCELLED.value,
    } else None
    return ProcessingJob(
        id=uuid.uuid4(),
        project_id=project_id,
        revision_id=revision_id,
        job_type=ProcessingJobType.REVISION_EXTRACTION.value,
        status=status,
        trigger_kind=trigger_kind,
        attempt_no=1,
        requested_by_id=None,
        lease_token=lease_token,
        lease_expires_at=lease_expires_at,
        started_at=started_at,
        completed_at=completed_at,
        result_extraction_run_id=result_extraction_run_id,
        failure_code=failure_code,
        failure_message=None,
        created_at=now,
        updated_at=now,
    )


def build_execution_result(
    *,
    revision_id: uuid.UUID,
    extraction_run_id: uuid.UUID | None = None,
    outcome: str = "success",
    revision_status: str = DocumentRevisionStatus.AWAITING_REVIEW.value,
) -> RevisionExtractionResult:
    return RevisionExtractionResult(
        revision_id=revision_id,
        extraction_run_id=extraction_run_id or uuid.uuid4(),
        outcome=outcome,
        revision_status=revision_status,
        block_count=1,
        character_count=5,
        page_count=1,
        warnings=[],
        requires_ocr=outcome == "needs_ocr",
    )


class FakeManagedSession:
    def __init__(self, name: str) -> None:
        self.name = name
        self.closed = False
        self.flush_count = 0
        self.commit_count = 0
        self.rollback_count = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.closed = True
        return False

    async def flush(self) -> None:
        self.flush_count += 1

    async def commit(self) -> None:
        self.commit_count += 1

    async def rollback(self) -> None:
        self.rollback_count += 1


class FakeSessionFactory:
    def __init__(self) -> None:
        self.sessions: list[FakeManagedSession] = []

    def __call__(self) -> FakeManagedSession:
        session = FakeManagedSession(f"session-{len(self.sessions) + 1}")
        self.sessions.append(session)
        return session


class DetachedClaimedJob:
    def __init__(self, *, job_id: uuid.UUID, project_id: uuid.UUID, revision_id: uuid.UUID, lease_token: uuid.UUID):
        self.id = job_id
        self.project_id = project_id
        self.revision_id = revision_id
        self.lease_token = lease_token

    @property
    def revision(self):
        raise AssertionError("Executor must not access ORM relationships after the claim session closes.")


def patch_job_snapshot_repository(monkeypatch, *, job: ProcessingJob, revision: DocumentRevision) -> None:
    async def fake_get_processing_job_by_id(_session, *, job_id):
        if job_id == job.id:
            return job
        return None

    async def fake_get_revision_by_id(_session, *, revision_id):
        if revision_id == revision.id:
            return revision
        return None

    monkeypatch.setattr(
        processing_job_executor_service.processing_job_repository,
        "get_processing_job_by_id",
        fake_get_processing_job_by_id,
    )
    monkeypatch.setattr(
        processing_job_executor_service.processing_job_repository,
        "get_revision_by_id",
        fake_get_revision_by_id,
    )


@pytest.mark.parametrize(
    ("outcome", "revision_status"),
    [
        ("success", DocumentRevisionStatus.AWAITING_REVIEW.value),
        ("partial", DocumentRevisionStatus.AWAITING_REVIEW.value),
        ("needs_ocr", DocumentRevisionStatus.FAILED.value),
        ("failed", DocumentRevisionStatus.FAILED.value),
    ],
)
def test_queued_job_is_claimed_and_completed_for_all_run_outcomes(
    monkeypatch,
    outcome: str,
    revision_status: str,
) -> None:
    session_factory = FakeSessionFactory()
    project_id = uuid.uuid4()
    revision = build_revision(status=DocumentRevisionStatus.ACCEPTED.value)
    job = build_processing_job(project_id=project_id, revision_id=revision.id)
    patch_job_snapshot_repository(monkeypatch, job=job, revision=revision)
    claim_calls: list[uuid.UUID] = []
    run_calls: list[uuid.UUID] = []
    complete_calls: list[tuple[uuid.UUID, uuid.UUID]] = []

    async def fake_claim_processing_job(_session, *, project_id, job_id, lease_seconds):
        claim_calls.append(job_id)
        job.status = ProcessingJobStatus.RUNNING.value
        job.lease_token = uuid.uuid4()
        job.lease_expires_at = datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)
        return DetachedClaimedJob(
            job_id=job.id,
            project_id=project_id,
            revision_id=job.revision_id,
            lease_token=job.lease_token,
        )

    async def fake_run_revision_extraction(_session, *, project_id, revision_id, storage, finalizer=None):
        run_calls.append(revision_id)
        revision.status = revision_status
        result = build_execution_result(
            revision_id=revision_id,
            outcome=outcome,
            revision_status=revision_status,
        )
        if finalizer is not None:
            complete_calls.append((job.id, result.extraction_run_id))
            await finalizer(_session, revision, type("Run", (), {"id": result.extraction_run_id})())
        return result

    monkeypatch.setattr(processing_job_executor_service, "claim_processing_job", fake_claim_processing_job)
    monkeypatch.setattr(processing_job_executor_service, "run_revision_extraction", fake_run_revision_extraction)
    monkeypatch.setattr(
        processing_job_executor_service,
        "complete_processing_job_in_transaction",
        lambda _session, **kwargs: asyncio.sleep(0, result=job),
    )

    result = run_async(
        execute_revision_extraction_job(
            session_factory,
            job_id=job.id,
            storage=object(),
            heartbeat_seconds=0.01,
        )
    )

    assert result.execution_status == ProcessingJobExecutionStatus.COMPLETED
    assert result.job_status == ProcessingJobStatus.COMPLETED
    assert result.revision_status == revision_status
    assert result.failure_code is None
    assert claim_calls == [job.id]
    assert run_calls == [revision.id]
    assert len(complete_calls) == 1


def test_executor_finalizes_job_in_same_session_as_revision_persistence(monkeypatch) -> None:
    session_factory = FakeSessionFactory()
    project_id = uuid.uuid4()
    revision = build_revision(status=DocumentRevisionStatus.ACCEPTED.value)
    job = build_processing_job(project_id=project_id, revision_id=revision.id)
    patch_job_snapshot_repository(monkeypatch, job=job, revision=revision)
    recorded: dict[str, str] = {}

    async def fake_claim_processing_job(_session, *, project_id, job_id, lease_seconds):
        job.status = ProcessingJobStatus.RUNNING.value
        job.lease_token = uuid.uuid4()
        job.lease_expires_at = datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)
        return DetachedClaimedJob(
            job_id=job.id,
            project_id=project_id,
            revision_id=job.revision_id,
            lease_token=job.lease_token,
        )

    async def fake_run_revision_extraction(_session, *, project_id, revision_id, storage, finalizer=None):
        recorded["run_session"] = _session.name
        result = build_execution_result(revision_id=revision_id)
        if finalizer is not None:
            await finalizer(_session, revision, type("Run", (), {"id": result.extraction_run_id})())
        return result

    async def fake_complete_processing_job_in_transaction(_session, **kwargs):
        recorded["finalize_session"] = _session.name
        job.status = ProcessingJobStatus.COMPLETED.value
        job.result_extraction_run_id = kwargs["extraction_run_id"]
        return job

    monkeypatch.setattr(processing_job_executor_service, "claim_processing_job", fake_claim_processing_job)
    monkeypatch.setattr(processing_job_executor_service, "run_revision_extraction", fake_run_revision_extraction)
    monkeypatch.setattr(
        processing_job_executor_service,
        "complete_processing_job_in_transaction",
        fake_complete_processing_job_in_transaction,
    )

    result = run_async(
        execute_revision_extraction_job(
            session_factory,
            job_id=job.id,
            storage=object(),
            heartbeat_seconds=0.01,
        )
    )

    assert result.execution_status == ProcessingJobExecutionStatus.COMPLETED
    assert recorded["run_session"] == recorded["finalize_session"]


def test_ordinary_execution_exception_marks_job_failed_without_leaking_raw_error(monkeypatch, caplog) -> None:
    session_factory = FakeSessionFactory()
    project_id = uuid.uuid4()
    revision = build_revision(status=DocumentRevisionStatus.ACCEPTED.value)
    job = build_processing_job(project_id=project_id, revision_id=revision.id)
    patch_job_snapshot_repository(monkeypatch, job=job, revision=revision)
    stored_failure_messages: list[str] = []

    async def fake_claim_processing_job(_session, *, project_id, job_id, lease_seconds):
        job.status = ProcessingJobStatus.RUNNING.value
        job.lease_token = uuid.uuid4()
        job.lease_expires_at = datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)
        return DetachedClaimedJob(
            job_id=job.id,
            project_id=project_id,
            revision_id=job.revision_id,
            lease_token=job.lease_token,
        )

    async def fake_run_revision_extraction(_session, *, project_id, revision_id, storage, finalizer=None):
        raise RuntimeError("storage_key=secret lease_token=topsecret 正文 body")

    async def fake_fail_processing_job(_session, *, project_id, job_id, lease_token, failure_code, failure_message):
        assert failure_code == WORKER_FAILURE_CODE
        assert failure_message == WORKER_FAILURE_MESSAGE
        stored_failure_messages.append(failure_message)
        job.status = ProcessingJobStatus.FAILED.value
        job.failure_code = failure_code
        job.failure_message = failure_message
        job.lease_token = None
        job.lease_expires_at = None
        return job

    monkeypatch.setattr(processing_job_executor_service, "claim_processing_job", fake_claim_processing_job)
    monkeypatch.setattr(processing_job_executor_service, "run_revision_extraction", fake_run_revision_extraction)
    monkeypatch.setattr(processing_job_executor_service, "fail_processing_job", fake_fail_processing_job)

    with caplog.at_level("WARNING"):
        result = run_async(
            execute_revision_extraction_job(
                session_factory,
                job_id=job.id,
                storage=object(),
                heartbeat_seconds=0.01,
            )
        )

    assert result.execution_status == ProcessingJobExecutionStatus.FAILED
    assert result.job_status == ProcessingJobStatus.FAILED
    assert result.failure_code == WORKER_FAILURE_CODE
    assert stored_failure_messages == [WORKER_FAILURE_MESSAGE]
    assert "topsecret" not in caplog.text
    assert "storage_key" not in caplog.text
    assert "正文 body" not in caplog.text


@pytest.mark.parametrize(
    ("job_status", "skip_reason", "failure_code"),
    [
        (ProcessingJobStatus.RUNNING.value, "already_running", None),
        (ProcessingJobStatus.COMPLETED.value, "already_completed", None),
        (ProcessingJobStatus.FAILED.value, "already_failed", "worker_failed"),
        (ProcessingJobStatus.CANCELLED.value, "already_cancelled", None),
    ],
)
def test_running_or_terminal_jobs_are_safely_skipped(
    monkeypatch,
    job_status: str,
    skip_reason: str,
    failure_code: str | None,
) -> None:
    session_factory = FakeSessionFactory()
    project_id = uuid.uuid4()
    revision = build_revision(status=DocumentRevisionStatus.FAILED.value)
    job = build_processing_job(
        project_id=project_id,
        revision_id=revision.id,
        status=job_status,
        failure_code=failure_code,
    )
    patch_job_snapshot_repository(monkeypatch, job=job, revision=revision)

    async def fake_claim_processing_job(_session, *, project_id, job_id, lease_seconds):
        raise processing_job_service.ProcessingJobStateError("not queued")

    monkeypatch.setattr(processing_job_executor_service, "claim_processing_job", fake_claim_processing_job)

    result = run_async(
        execute_revision_extraction_job(
            session_factory,
            job_id=job.id,
            storage=object(),
        )
    )

    assert result.execution_status == ProcessingJobExecutionStatus.SKIPPED
    assert result.job_status == job_status
    assert result.skip_reason == skip_reason
    assert result.failure_code == failure_code


def test_concurrent_execution_of_same_job_runs_extraction_only_once(monkeypatch) -> None:
    async def scenario():
        session_factory = FakeSessionFactory()
        project_id = uuid.uuid4()
        revision = build_revision(status=DocumentRevisionStatus.ACCEPTED.value)
        job = build_processing_job(project_id=project_id, revision_id=revision.id)
        patch_job_snapshot_repository(monkeypatch, job=job, revision=revision)
        run_count = 0
        claim_guard = asyncio.Lock()

        async def fake_claim_processing_job(_session, *, project_id, job_id, lease_seconds):
            nonlocal run_count
            async with claim_guard:
                if job.status != ProcessingJobStatus.QUEUED.value:
                    raise processing_job_service.ProcessingJobStateError("already claimed")
                job.status = ProcessingJobStatus.RUNNING.value
                job.lease_token = uuid.uuid4()
                job.lease_expires_at = datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)
                return DetachedClaimedJob(
                    job_id=job.id,
                    project_id=project_id,
                    revision_id=job.revision_id,
                    lease_token=job.lease_token,
                )

        async def fake_run_revision_extraction(_session, *, project_id, revision_id, storage, finalizer=None):
            nonlocal run_count
            run_count += 1
            await asyncio.sleep(0.02)
            revision.status = DocumentRevisionStatus.AWAITING_REVIEW.value
            result = build_execution_result(
                revision_id=revision_id,
                outcome="success",
                revision_status=DocumentRevisionStatus.AWAITING_REVIEW.value,
            )
            if finalizer is not None:
                await finalizer(_session, revision, type("Run", (), {"id": result.extraction_run_id})())
            return result

        monkeypatch.setattr(processing_job_executor_service, "claim_processing_job", fake_claim_processing_job)
        monkeypatch.setattr(processing_job_executor_service, "run_revision_extraction", fake_run_revision_extraction)
        monkeypatch.setattr(
            processing_job_executor_service,
            "complete_processing_job_in_transaction",
            lambda _session, **kwargs: asyncio.sleep(0, result=job),
        )

        first, second = await asyncio.gather(
            execute_revision_extraction_job(session_factory, job_id=job.id, storage=object(), heartbeat_seconds=0.01),
            execute_revision_extraction_job(session_factory, job_id=job.id, storage=object(), heartbeat_seconds=0.01),
        )
        return first, second, run_count

    first, second, run_count = run_async(scenario())

    assert run_count == 1
    assert {first.execution_status, second.execution_status} == {
        ProcessingJobExecutionStatus.COMPLETED,
        ProcessingJobExecutionStatus.SKIPPED,
    }


def test_heartbeat_uses_independent_sessions_and_renews_periodically(monkeypatch) -> None:
    async def scenario():
        session_factory = FakeSessionFactory()
        project_id = uuid.uuid4()
        revision = build_revision(status=DocumentRevisionStatus.ACCEPTED.value)
        job = build_processing_job(project_id=project_id, revision_id=revision.id)
        patch_job_snapshot_repository(monkeypatch, job=job, revision=revision)
        renew_session_names: list[str] = []
        run_session_names: list[str] = []

        async def fake_claim_processing_job(_session, *, project_id, job_id, lease_seconds):
            job.status = ProcessingJobStatus.RUNNING.value
            job.lease_token = uuid.uuid4()
            job.lease_expires_at = datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)
            return DetachedClaimedJob(
                job_id=job.id,
                project_id=project_id,
                revision_id=job.revision_id,
                lease_token=job.lease_token,
            )

        async def fake_renew_processing_job_lease(_session, *, job_id, lease_token, lease_seconds):
            renew_session_names.append(_session.name)
            return job

        async def fake_run_revision_extraction(_session, *, project_id, revision_id, storage, finalizer=None):
            run_session_names.append(_session.name)
            await asyncio.sleep(0.035)
            revision.status = DocumentRevisionStatus.AWAITING_REVIEW.value
            result = build_execution_result(
                revision_id=revision_id,
                outcome="success",
                revision_status=DocumentRevisionStatus.AWAITING_REVIEW.value,
            )
            if finalizer is not None:
                await finalizer(_session, revision, type("Run", (), {"id": result.extraction_run_id})())
            return result

        monkeypatch.setattr(processing_job_executor_service, "claim_processing_job", fake_claim_processing_job)
        monkeypatch.setattr(processing_job_executor_service, "renew_processing_job_lease", fake_renew_processing_job_lease)
        monkeypatch.setattr(processing_job_executor_service, "run_revision_extraction", fake_run_revision_extraction)
        monkeypatch.setattr(
            processing_job_executor_service,
            "complete_processing_job_in_transaction",
            lambda _session, **kwargs: asyncio.sleep(0, result=job),
        )

        result = await execute_revision_extraction_job(
            session_factory,
            job_id=job.id,
            storage=object(),
            lease_seconds=90,
            heartbeat_seconds=0.01,
        )
        return result, renew_session_names, run_session_names, session_factory.sessions

    result, renew_session_names, run_session_names, sessions = run_async(scenario())

    assert result.execution_status == ProcessingJobExecutionStatus.COMPLETED
    assert len(renew_session_names) >= 2
    assert len(set(renew_session_names)) == len(renew_session_names)
    assert run_session_names
    assert all(name not in run_session_names for name in renew_session_names)
    assert all(session.closed is True for session in sessions)


def test_renew_processing_job_lease_validates_token_and_expiry(monkeypatch) -> None:
    original_expiry = datetime.now(timezone.utc) + timedelta(minutes=5)
    lease_token = uuid.uuid4()
    job = build_processing_job(
        project_id=uuid.uuid4(),
        revision_id=uuid.uuid4(),
        status=ProcessingJobStatus.RUNNING.value,
        lease_token=lease_token,
        lease_expires_at=original_expiry,
    )
    session = FakeManagedSession("renew")

    async def fake_get_processing_job_by_id_for_update(_session, *, job_id):
        assert job_id == job.id
        return job

    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_processing_job_by_id_for_update",
        fake_get_processing_job_by_id_for_update,
    )

    renewal_started_at = datetime.now(timezone.utc)
    renewed = run_async(
        processing_job_service.renew_processing_job_lease(
            session,
            job_id=job.id,
            lease_token=lease_token,
            lease_seconds=60,
        )
    )

    assert renewed.lease_expires_at > renewal_started_at
    assert renewed.started_at == job.started_at
    assert session.flush_count == 1
    assert session.commit_count == 1

    with pytest.raises(processing_job_service.ProcessingJobLeaseError):
        run_async(
            processing_job_service.renew_processing_job_lease(
                session,
                job_id=job.id,
                lease_token=uuid.uuid4(),
                lease_seconds=60,
            )
        )

    job.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    with pytest.raises(processing_job_service.ProcessingJobLeaseError):
        run_async(
            processing_job_service.renew_processing_job_lease(
                session,
                job_id=job.id,
                lease_token=lease_token,
                lease_seconds=60,
            )
        )


def test_completion_lease_loss_does_not_overwrite_job(monkeypatch) -> None:
    session_factory = FakeSessionFactory()
    project_id = uuid.uuid4()
    revision = build_revision(status=DocumentRevisionStatus.ACCEPTED.value)
    job = build_processing_job(project_id=project_id, revision_id=revision.id)
    patch_job_snapshot_repository(monkeypatch, job=job, revision=revision)

    async def fake_claim_processing_job(_session, *, project_id, job_id, lease_seconds):
        job.status = ProcessingJobStatus.RUNNING.value
        job.lease_token = uuid.uuid4()
        job.lease_expires_at = datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)
        return DetachedClaimedJob(
            job_id=job.id,
            project_id=project_id,
            revision_id=job.revision_id,
            lease_token=job.lease_token,
        )

    async def fake_run_revision_extraction(_session, *, project_id, revision_id, storage, finalizer=None):
        revision.status = DocumentRevisionStatus.AWAITING_REVIEW.value
        result = build_execution_result(
            revision_id=revision_id,
            outcome="success",
            revision_status=DocumentRevisionStatus.AWAITING_REVIEW.value,
        )
        if finalizer is not None:
            await finalizer(_session, revision, type("Run", (), {"id": result.extraction_run_id})())
        return result

    monkeypatch.setattr(processing_job_executor_service, "claim_processing_job", fake_claim_processing_job)
    monkeypatch.setattr(processing_job_executor_service, "run_revision_extraction", fake_run_revision_extraction)
    monkeypatch.setattr(
        processing_job_executor_service,
        "complete_processing_job_in_transaction",
        lambda _session, **kwargs: asyncio.sleep(
            0,
            result=(_ for _ in ()).throw(processing_job_service.ProcessingJobLeaseError("lease lost")),
        ),
    )

    result = run_async(
        execute_revision_extraction_job(
            session_factory,
            job_id=job.id,
            storage=object(),
            heartbeat_seconds=0.01,
        )
    )

    assert result.execution_status == ProcessingJobExecutionStatus.FAILED
    assert result.failure_code == LEASE_LOST_DURING_COMPLETION
    assert job.status == ProcessingJobStatus.RUNNING.value


def test_cancelled_error_propagates_and_does_not_mark_job_finished(monkeypatch) -> None:
    async def scenario():
        session_factory = FakeSessionFactory()
        project_id = uuid.uuid4()
        revision = build_revision(status=DocumentRevisionStatus.ACCEPTED.value)
        job = build_processing_job(project_id=project_id, revision_id=revision.id)
        patch_job_snapshot_repository(monkeypatch, job=job, revision=revision)
        fail_called = False

        async def fake_claim_processing_job(_session, *, project_id, job_id, lease_seconds):
            job.status = ProcessingJobStatus.RUNNING.value
            job.lease_token = uuid.uuid4()
            job.lease_expires_at = datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)
            return DetachedClaimedJob(
                job_id=job.id,
                project_id=project_id,
                revision_id=job.revision_id,
                lease_token=job.lease_token,
            )

        async def fake_run_revision_extraction(_session, *, project_id, revision_id, storage, finalizer=None):
            raise asyncio.CancelledError

        async def fake_fail_processing_job(_session, *, project_id, job_id, lease_token, failure_code, failure_message):
            nonlocal fail_called
            fail_called = True
            return job

        monkeypatch.setattr(processing_job_executor_service, "claim_processing_job", fake_claim_processing_job)
        monkeypatch.setattr(processing_job_executor_service, "run_revision_extraction", fake_run_revision_extraction)
        monkeypatch.setattr(processing_job_executor_service, "fail_processing_job", fake_fail_processing_job)

        with pytest.raises(asyncio.CancelledError):
            await execute_revision_extraction_job(
                session_factory,
                job_id=job.id,
                storage=object(),
                heartbeat_seconds=0.01,
            )
        return fail_called, job

    fail_called, job = run_async(scenario())

    assert fail_called is False
    assert job.status == ProcessingJobStatus.RUNNING.value


def test_executor_source_keeps_10b_transaction_boundary_and_safe_calls() -> None:
    source = inspect.getsource(processing_job_executor_service)
    assert "run_revision_extraction(" in source
    assert "persist_extraction_result(" not in source
    assert "complete_processing_job(" not in source
    assert "extract_document(" not in source
