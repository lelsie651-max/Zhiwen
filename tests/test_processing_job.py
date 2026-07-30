import asyncio
import inspect
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import CheckConstraint, UniqueConstraint

from app.models import Base
from app.models.document_content import ExtractionRun
from app.models.document_revision import DocumentRevision, DocumentRevisionStatus, SourceAuthority, UploadIntent
from app.models.processing_job import ProcessingJob, ProcessingJobStatus, ProcessingJobType, ProcessingTriggerKind
from app.models.project import Project
from app.models.project_member import ProjectMember
from app.models.user import User
from app.schemas.processing_job import ProcessingJobRead
from app.services import processing_job as processing_job_service


def run_async(awaitable):
    return asyncio.run(awaitable)


def async_return(value):
    async def _inner(*args, **kwargs):
        return value

    return _inner


async def async_return_job(*args, **kwargs):
    return kwargs["job"]


class FakeSession:
    def __init__(self, *, fail_on_commit: Exception | None = None, fail_on_flush: Exception | None = None) -> None:
        self.flush_count = 0
        self.commit_count = 0
        self.rollback_count = 0
        self.fail_on_commit = fail_on_commit
        self.fail_on_flush = fail_on_flush

    async def flush(self) -> None:
        self.flush_count += 1
        if self.fail_on_flush is not None:
            raise self.fail_on_flush

    async def commit(self) -> None:
        self.commit_count += 1
        if self.fail_on_commit is not None:
            raise self.fail_on_commit

    async def rollback(self) -> None:
        self.rollback_count += 1


def build_project(*, project_id: uuid.UUID | None = None) -> Project:
    actual_project_id = project_id or uuid.uuid4()
    return Project(
        id=actual_project_id,
        name="Processing Project",
        slug=f"processing-{str(actual_project_id)[:8]}",
        created_by_id=uuid.uuid4(),
        status="active",
    )


def build_user(*, user_id: uuid.UUID | None = None, status: str = "active") -> User:
    actual_user_id = user_id or uuid.uuid4()
    return User(
        id=actual_user_id,
        handle=f"user-{str(actual_user_id)[:8]}",
        display_name="Processing Actor",
        status=status,
    )


def build_project_member(*, project_id: uuid.UUID, user_id: uuid.UUID, role: str) -> ProjectMember:
    return ProjectMember(
        id=uuid.uuid4(),
        project_id=project_id,
        user_id=user_id,
        role=role,
    )


def build_revision(
    *,
    revision_id: uuid.UUID | None = None,
    document_id: uuid.UUID | None = None,
    status: str = DocumentRevisionStatus.ACCEPTED.value,
    updated_at: datetime | None = None,
) -> DocumentRevision:
    now = updated_at or datetime.now(timezone.utc)
    return DocumentRevision(
        id=revision_id or uuid.uuid4(),
        document_id=document_id or uuid.uuid4(),
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


def build_extraction_run(
    *,
    revision_id: uuid.UUID,
    run_id: uuid.UUID | None = None,
    attempt_no: int = 1,
    outcome: str = "failed",
) -> ExtractionRun:
    now = datetime.now(timezone.utc)
    return ExtractionRun(
        id=run_id or uuid.uuid4(),
        revision_id=revision_id,
        attempt_no=attempt_no,
        status="failed" if outcome == "failed" else "completed",
        outcome=outcome,
        extractor_name="x",
        extractor_version="1.0.0",
        detected_format="pdf",
        detected_encoding=None,
        page_count=None,
        character_count=0,
        block_count=0,
        warnings=[],
        content_metadata={},
        failure_code="deterministic_extraction_failed" if outcome == "failed" else "ocr_required",
        failure_message=None,
        started_at=now,
        completed_at=now,
        created_at=now,
    )


def build_processing_job(
    *,
    project_id: uuid.UUID,
    revision_id: uuid.UUID,
    requested_by_id: uuid.UUID | None = None,
    result_extraction_run_id: uuid.UUID | None = None,
    attempt_no: int = 1,
    status: str = ProcessingJobStatus.QUEUED.value,
    trigger_kind: str = ProcessingTriggerKind.AUTOMATIC.value,
    lease_token: uuid.UUID | None = None,
    lease_expires_at: datetime | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    failure_code: str | None = None,
    failure_message: str | None = None,
) -> ProcessingJob:
    now = datetime.now(timezone.utc)
    return ProcessingJob(
        id=uuid.uuid4(),
        project_id=project_id,
        revision_id=revision_id,
        job_type=ProcessingJobType.REVISION_EXTRACTION.value,
        status=status,
        trigger_kind=trigger_kind,
        attempt_no=attempt_no,
        requested_by_id=requested_by_id,
        lease_token=lease_token,
        lease_expires_at=lease_expires_at,
        started_at=started_at,
        completed_at=completed_at,
        result_extraction_run_id=result_extraction_run_id,
        failure_code=failure_code,
        failure_message=failure_message,
        created_at=now,
        updated_at=now,
    )


def test_processing_job_partial_unique_active_index_exists() -> None:
    table = Base.metadata.tables["processing_jobs"]
    active_index = next(index for index in table.indexes if index.name == "uq_processing_jobs_active_revision_id_job_type")
    assert active_index.unique is True
    assert [column.name for column in active_index.columns] == ["revision_id", "job_type"]
    assert str(active_index.dialect_options["postgresql"]["where"]) == "status IN ('queued','running')"


def test_processing_job_constraints_and_safe_read_schema_exist() -> None:
    table = Base.metadata.tables["processing_jobs"]
    unique_constraints = {
        tuple(constraint.columns.keys())
        for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    check_sql = {
        str(constraint.sqltext)
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }
    assert ("revision_id", "job_type", "attempt_no") in unique_constraints
    assert any("trigger_kind NOT IN ('manual', 'retry') OR requested_by_id IS NOT NULL" in sql for sql in check_sql)
    assert "lease_token" not in ProcessingJobRead.model_fields
    assert "storage_key" not in ProcessingJobRead.model_fields


def patch_actor_and_member(monkeypatch, *, actor: User | None, membership: ProjectMember | None) -> None:
    async def fake_get_active_user_by_id(_session, *, user_id):
        if actor is not None and user_id == actor.id:
            return actor
        return None

    async def fake_get_project_member_for_project(_session, *, project_id, user_id):
        if (
            membership is not None
            and project_id == membership.project_id
            and user_id == membership.user_id
        ):
            return membership
        return None

    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_active_user_by_id",
        fake_get_active_user_by_id,
    )
    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_project_member_for_project",
        fake_get_project_member_for_project,
    )


def test_manual_permission_and_automatic_enqueue(monkeypatch) -> None:
    project = build_project()
    actor = build_user()
    membership = build_project_member(project_id=project.id, user_id=actor.id, role="owner")

    for trigger_kind, actor_id, expected_requester in [
        (ProcessingTriggerKind.AUTOMATIC, None, None),
        (ProcessingTriggerKind.MANUAL, actor.id, actor.id),
    ]:
        session = FakeSession()
        revision = build_revision(status=DocumentRevisionStatus.ACCEPTED.value)
        created_jobs: list[ProcessingJob] = []

        async def fake_get_revision_for_processing_job_update(_session, *, project_id, revision_id):
            assert project_id == project.id
            assert revision_id == revision.id
            return revision

        async def fake_get_active_processing_job_for_update(_session, *, revision_id, job_type):
            assert revision_id == revision.id
            assert job_type == ProcessingJobType.REVISION_EXTRACTION.value
            return None

        async def fake_get_next_processing_job_attempt_no(_session, *, revision_id, job_type):
            assert revision_id == revision.id
            assert job_type == ProcessingJobType.REVISION_EXTRACTION.value
            return 1

        async def fake_create_processing_job(_session, *, job):
            created_jobs.append(job)
            return job

        monkeypatch.setattr(
            processing_job_service.processing_job_repository,
            "get_revision_for_processing_job_update",
            fake_get_revision_for_processing_job_update,
        )
        monkeypatch.setattr(
            processing_job_service.processing_job_repository,
            "get_active_processing_job_for_update",
            fake_get_active_processing_job_for_update,
        )
        monkeypatch.setattr(
            processing_job_service.processing_job_repository,
            "get_next_processing_job_attempt_no",
            fake_get_next_processing_job_attempt_no,
        )
        monkeypatch.setattr(
            processing_job_service.processing_job_repository,
            "create_processing_job",
            fake_create_processing_job,
        )
        patch_actor_and_member(monkeypatch, actor=actor, membership=membership)

        job = run_async(
            processing_job_service.enqueue_revision_extraction_job(
                session,
                project_id=project.id,
                revision_id=revision.id,
                trigger_kind=trigger_kind,
                actor_id=actor_id,
            )
        )

        assert job.status == ProcessingJobStatus.QUEUED
        assert job.trigger_kind == trigger_kind
        assert job.requested_by_id == expected_requester
        assert session.commit_count == 1
        assert created_jobs[0] is job

    session = FakeSession()
    revision = build_revision(status=DocumentRevisionStatus.ACCEPTED.value)
    viewer = build_user()
    patch_actor_and_member(
        monkeypatch,
        actor=viewer,
        membership=build_project_member(project_id=project.id, user_id=viewer.id, role="viewer"),
    )
    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_revision_for_processing_job_update",
        async_return(revision),
    )
    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_active_processing_job_for_update",
        async_return(None),
    )
    with pytest.raises(processing_job_service.ProcessingJobPermissionError):
        run_async(
            processing_job_service.enqueue_revision_extraction_job(
                session,
                project_id=project.id,
                revision_id=revision.id,
                trigger_kind=ProcessingTriggerKind.MANUAL,
                actor_id=viewer.id,
            )
        )


def test_duplicate_enqueue_returns_existing_active_job(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    revision = build_revision(status=DocumentRevisionStatus.ACCEPTED.value)
    existing_job = build_processing_job(
        project_id=project.id,
        revision_id=revision.id,
        status=ProcessingJobStatus.QUEUED.value,
    )

    async def fake_get_revision_for_processing_job_update(_session, *, project_id, revision_id):
        assert project_id == project.id
        assert revision_id == revision.id
        return revision

    async def fake_get_active_processing_job_for_update(_session, *, revision_id, job_type):
        assert revision_id == revision.id
        assert job_type == ProcessingJobType.REVISION_EXTRACTION.value
        return existing_job

    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_revision_for_processing_job_update",
        fake_get_revision_for_processing_job_update,
    )
    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_active_processing_job_for_update",
        fake_get_active_processing_job_for_update,
    )

    result = run_async(
        processing_job_service.enqueue_revision_extraction_job(
            session,
            project_id=project.id,
            revision_id=revision.id,
            trigger_kind=ProcessingTriggerKind.AUTOMATIC,
        )
    )

    assert result is existing_job
    assert session.commit_count == 0


def test_attempt_no_increments_inside_lock_order(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    revision = build_revision(status=DocumentRevisionStatus.ACCEPTED.value)
    call_order: list[str] = []
    actor = build_user()
    membership = build_project_member(project_id=project.id, user_id=actor.id, role="editor")

    async def fake_get_revision_for_processing_job_update(_session, *, project_id, revision_id):
        call_order.append("lock_revision")
        assert project_id == project.id
        assert revision_id == revision.id
        return revision

    async def fake_get_active_processing_job_for_update(_session, *, revision_id, job_type):
        call_order.append("lock_active_job")
        assert revision_id == revision.id
        assert job_type == ProcessingJobType.REVISION_EXTRACTION.value
        return None

    async def fake_get_next_processing_job_attempt_no(_session, *, revision_id, job_type):
        call_order.append("next_attempt")
        assert revision_id == revision.id
        assert job_type == ProcessingJobType.REVISION_EXTRACTION.value
        return 3

    async def fake_create_processing_job(_session, *, job):
        call_order.append("create_job")
        return job

    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_revision_for_processing_job_update",
        fake_get_revision_for_processing_job_update,
    )
    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_active_processing_job_for_update",
        fake_get_active_processing_job_for_update,
    )
    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_next_processing_job_attempt_no",
        fake_get_next_processing_job_attempt_no,
    )
    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "create_processing_job",
        fake_create_processing_job,
    )
    patch_actor_and_member(monkeypatch, actor=actor, membership=membership)

    result = run_async(
        processing_job_service.enqueue_revision_extraction_job(
            session,
            project_id=project.id,
            revision_id=revision.id,
            trigger_kind=ProcessingTriggerKind.MANUAL,
            actor_id=actor.id,
        )
    )

    assert result.attempt_no == 3
    assert call_order[:2] == ["lock_revision", "lock_active_job"]
    assert session.commit_count == 1


def test_failed_revision_retries_without_overwriting_history(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    membership = build_project_member(project_id=project.id, user_id=actor.id, role="owner")
    revision = build_revision(status=DocumentRevisionStatus.FAILED.value)
    latest_run = build_extraction_run(
        revision_id=revision.id,
        attempt_no=2,
        outcome="needs_ocr",
    )
    old_failed_job = build_processing_job(
        project_id=project.id,
        revision_id=revision.id,
        status=ProcessingJobStatus.FAILED.value,
        trigger_kind=ProcessingTriggerKind.RETRY.value,
        attempt_no=2,
        requested_by_id=actor.id,
        started_at=datetime.now(timezone.utc) - timedelta(minutes=2),
        completed_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        failure_code="deterministic_extraction_failed",
        failure_message="safe failure",
    )

    async def fake_get_revision_for_processing_job_update(_session, *, project_id, revision_id):
        assert project_id == project.id
        assert revision_id == revision.id
        return revision

    async def fake_get_active_processing_job_for_update(_session, *, revision_id, job_type):
        assert revision_id == revision.id
        return None

    async def fake_get_latest_extraction_run_for_revision(_session, *, revision_id):
        assert revision_id == revision.id
        return latest_run

    async def fake_get_latest_failed_processing_job(_session, *, revision_id, job_type):
        assert revision_id == revision.id
        return old_failed_job

    async def fake_get_next_processing_job_attempt_no(_session, *, revision_id, job_type):
        assert revision_id == revision.id
        return 3

    async def fake_create_processing_job(_session, *, job):
        return job

    patch_actor_and_member(monkeypatch, actor=actor, membership=membership)
    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_revision_for_processing_job_update",
        fake_get_revision_for_processing_job_update,
    )
    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_active_processing_job_for_update",
        fake_get_active_processing_job_for_update,
    )
    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_latest_extraction_run_for_revision",
        fake_get_latest_extraction_run_for_revision,
    )
    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_latest_failed_processing_job",
        fake_get_latest_failed_processing_job,
    )
    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_next_processing_job_attempt_no",
        fake_get_next_processing_job_attempt_no,
    )
    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "create_processing_job",
        fake_create_processing_job,
    )

    result = run_async(
        processing_job_service.retry_failed_revision_extraction(
            session,
            project_id=project.id,
            revision_id=revision.id,
            actor_id=actor.id,
        )
    )

    assert revision.status == DocumentRevisionStatus.ACCEPTED.value
    assert result.trigger_kind == ProcessingTriggerKind.RETRY
    assert result.attempt_no == 3
    assert old_failed_job.status == ProcessingJobStatus.FAILED.value
    assert latest_run.outcome == "needs_ocr"
    assert session.commit_count == 1


def test_expired_running_job_can_be_recovered(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    membership = build_project_member(project_id=project.id, user_id=actor.id, role="owner")
    revision = build_revision(status=DocumentRevisionStatus.EXTRACTING.value)
    expired_job = build_processing_job(
        project_id=project.id,
        revision_id=revision.id,
        status=ProcessingJobStatus.RUNNING.value,
        trigger_kind=ProcessingTriggerKind.AUTOMATIC.value,
        attempt_no=1,
        lease_token=uuid.uuid4(),
        started_at=datetime.now(timezone.utc) - timedelta(minutes=10),
        lease_expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )

    async def fake_get_revision_for_processing_job_update(_session, *, project_id, revision_id):
        return revision

    async def fake_get_active_processing_job_for_update(_session, *, revision_id, job_type):
        return expired_job

    async def fake_get_next_processing_job_attempt_no(_session, *, revision_id, job_type):
        return 2

    async def fake_create_processing_job(_session, *, job):
        return job

    patch_actor_and_member(monkeypatch, actor=actor, membership=membership)
    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_revision_for_processing_job_update",
        fake_get_revision_for_processing_job_update,
    )
    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_active_processing_job_for_update",
        fake_get_active_processing_job_for_update,
    )
    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_next_processing_job_attempt_no",
        fake_get_next_processing_job_attempt_no,
    )
    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "create_processing_job",
        fake_create_processing_job,
    )

    result = run_async(
        processing_job_service.recover_stale_revision_extraction(
            session,
            project_id=project.id,
            revision_id=revision.id,
            actor_id=actor.id,
            stale_before=datetime.now(timezone.utc) - timedelta(minutes=5),
        )
    )

    assert expired_job.status == ProcessingJobStatus.FAILED.value
    assert expired_job.failure_code == "lease_expired"
    assert expired_job.lease_token is None
    assert expired_job.lease_expires_at is None
    assert expired_job.completed_at is not None
    assert revision.status == DocumentRevisionStatus.ACCEPTED.value
    assert result.trigger_kind == ProcessingTriggerKind.RECOVERY


def test_stale_parsing_or_extracting_without_job_can_be_recovered(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    membership = build_project_member(project_id=project.id, user_id=actor.id, role="editor")
    stale_time = datetime.now(timezone.utc) - timedelta(hours=1)
    revision = build_revision(
        status=DocumentRevisionStatus.PARSING.value,
        updated_at=stale_time,
    )

    async def fake_get_revision_for_processing_job_update(_session, *, project_id, revision_id):
        return revision

    async def fake_get_active_processing_job_for_update(_session, *, revision_id, job_type):
        return None

    async def fake_get_next_processing_job_attempt_no(_session, *, revision_id, job_type):
        return 1

    async def fake_create_processing_job(_session, *, job):
        return job

    patch_actor_and_member(monkeypatch, actor=actor, membership=membership)
    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_revision_for_processing_job_update",
        fake_get_revision_for_processing_job_update,
    )
    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_active_processing_job_for_update",
        fake_get_active_processing_job_for_update,
    )
    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_next_processing_job_attempt_no",
        fake_get_next_processing_job_attempt_no,
    )
    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "create_processing_job",
        fake_create_processing_job,
    )

    result = run_async(
        processing_job_service.recover_stale_revision_extraction(
            session,
            project_id=project.id,
            revision_id=revision.id,
            actor_id=actor.id,
            stale_before=datetime.now(timezone.utc) - timedelta(minutes=30),
        )
    )

    assert revision.status == DocumentRevisionStatus.ACCEPTED.value
    assert result.status == ProcessingJobStatus.QUEUED
    assert result.trigger_kind == ProcessingTriggerKind.RECOVERY


def test_unexpired_lease_recovery_is_rejected(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    membership = build_project_member(project_id=project.id, user_id=actor.id, role="owner")
    revision = build_revision(status=DocumentRevisionStatus.EXTRACTING.value)
    running_job = build_processing_job(
        project_id=project.id,
        revision_id=revision.id,
        status=ProcessingJobStatus.RUNNING.value,
        lease_token=uuid.uuid4(),
        started_at=datetime.now(timezone.utc) - timedelta(minutes=2),
        lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )

    patch_actor_and_member(monkeypatch, actor=actor, membership=membership)
    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_revision_for_processing_job_update",
        async_return(revision),
    )
    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_active_processing_job_for_update",
        async_return(running_job),
    )

    with pytest.raises(processing_job_service.ProcessingJobStateError):
        run_async(
            processing_job_service.recover_stale_revision_extraction(
                session,
                project_id=project.id,
                revision_id=revision.id,
                actor_id=actor.id,
                stale_before=datetime.now(timezone.utc) - timedelta(minutes=10),
            )
        )

    assert session.commit_count == 0


@pytest.mark.parametrize(
    "status",
    [
        DocumentRevisionStatus.AWAITING_REVIEW.value,
        DocumentRevisionStatus.COMPLETED.value,
        DocumentRevisionStatus.REJECTED.value,
        DocumentRevisionStatus.QUARANTINED.value,
        DocumentRevisionStatus.NEEDS_CONFIRMATION.value,
        DocumentRevisionStatus.FAILED.value,
    ],
)
def test_terminal_revision_states_reject_recovery(monkeypatch, status: str) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    membership = build_project_member(project_id=project.id, user_id=actor.id, role="owner")
    revision = build_revision(status=status)

    patch_actor_and_member(monkeypatch, actor=actor, membership=membership)
    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_revision_for_processing_job_update",
        async_return(revision),
    )

    with pytest.raises(processing_job_service.ProcessingJobStateError):
        run_async(
            processing_job_service.recover_stale_revision_extraction(
                session,
                project_id=project.id,
                revision_id=revision.id,
                actor_id=actor.id,
                stale_before=datetime.now(timezone.utc) - timedelta(minutes=10),
            )
        )


def test_claim_complete_and_fail_state_machine(monkeypatch) -> None:
    project = build_project()
    revision = build_revision(status=DocumentRevisionStatus.ACCEPTED.value)
    session = FakeSession()
    queued_job = build_processing_job(
        project_id=project.id,
        revision_id=revision.id,
        status=ProcessingJobStatus.QUEUED.value,
    )

    async def fake_get_processing_job_for_update(_session, *, project_id, job_id):
        assert project_id == project.id
        assert job_id == queued_job.id
        return queued_job

    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_processing_job_for_update",
        fake_get_processing_job_for_update,
    )

    claimed = run_async(
        processing_job_service.claim_processing_job(
            session,
            project_id=project.id,
            job_id=queued_job.id,
            lease_seconds=30,
        )
    )

    assert claimed.status == ProcessingJobStatus.RUNNING
    assert claimed.lease_token is not None
    assert claimed.started_at is not None
    assert claimed.lease_expires_at is not None

    running_job = claimed
    extraction_run = build_extraction_run(revision_id=revision.id, outcome="failed")
    session = FakeSession()

    async def fake_get_processing_job_for_update_complete(_session, *, project_id, job_id):
        return running_job

    async def fake_get_extraction_run_by_id(_session, *, extraction_run_id):
        assert extraction_run_id == extraction_run.id
        return extraction_run

    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_processing_job_for_update",
        fake_get_processing_job_for_update_complete,
    )
    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_extraction_run_by_id",
        fake_get_extraction_run_by_id,
    )

    completed = run_async(
        processing_job_service.complete_processing_job(
            session,
            project_id=project.id,
            job_id=running_job.id,
            lease_token=running_job.lease_token,
            extraction_run_id=extraction_run.id,
        )
    )

    assert completed.status == ProcessingJobStatus.COMPLETED
    assert completed.result_extraction_run_id == extraction_run.id
    assert completed.lease_token is None
    assert completed.lease_expires_at is None
    assert completed.completed_at is not None

    failed_job = build_processing_job(
        project_id=project.id,
        revision_id=revision.id,
        status=ProcessingJobStatus.RUNNING.value,
        lease_token=uuid.uuid4(),
        started_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
    )
    session = FakeSession()

    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_processing_job_for_update",
        async_return(failed_job),
    )

    failed = run_async(
        processing_job_service.fail_processing_job(
            session,
            project_id=project.id,
            job_id=failed_job.id,
            lease_token=failed_job.lease_token,
            failure_code="worker_failed",
            failure_message="safe worker failure",
        )
    )

    assert failed.status == ProcessingJobStatus.FAILED
    assert failed.failure_code == "worker_failed"
    assert failed.result_extraction_run_id is None
    assert failed.lease_token is None
    assert failed.completed_at is not None


def test_wrong_lease_token_is_rejected(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    revision = build_revision()
    running_job = build_processing_job(
        project_id=project.id,
        revision_id=revision.id,
        status=ProcessingJobStatus.RUNNING.value,
        lease_token=uuid.uuid4(),
        started_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
    )

    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_processing_job_for_update",
        async_return(running_job),
    )

    with pytest.raises(processing_job_service.ProcessingJobLeaseError):
        run_async(
            processing_job_service.fail_processing_job(
                session,
                project_id=project.id,
                job_id=running_job.id,
                lease_token=uuid.uuid4(),
                failure_code="worker_failed",
                failure_message="safe worker failure",
            )
        )


def test_cross_revision_result_run_is_rejected(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    job_revision = build_revision()
    other_revision = build_revision()
    running_job = build_processing_job(
        project_id=project.id,
        revision_id=job_revision.id,
        status=ProcessingJobStatus.RUNNING.value,
        lease_token=uuid.uuid4(),
        started_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
    )
    other_run = build_extraction_run(revision_id=other_revision.id)

    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_processing_job_for_update",
        async_return(running_job),
    )
    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_extraction_run_by_id",
        async_return(other_run),
    )

    with pytest.raises(processing_job_service.ProcessingJobStateError):
        run_async(
            processing_job_service.complete_processing_job(
                session,
                project_id=project.id,
                job_id=running_job.id,
                lease_token=running_job.lease_token,
                extraction_run_id=other_run.id,
            )
        )


def test_transaction_failures_roll_back(monkeypatch) -> None:
    project = build_project()
    actor = build_user()
    membership = build_project_member(project_id=project.id, user_id=actor.id, role="owner")
    revision = build_revision(status=DocumentRevisionStatus.ACCEPTED.value)
    session = FakeSession(fail_on_commit=RuntimeError("commit failed"))

    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_revision_for_processing_job_update",
        async_return(revision),
    )
    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_active_processing_job_for_update",
        async_return(None),
    )
    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "get_next_processing_job_attempt_no",
        async_return(1),
    )
    monkeypatch.setattr(
        processing_job_service.processing_job_repository,
        "create_processing_job",
        async_return_job,
    )
    patch_actor_and_member(monkeypatch, actor=actor, membership=membership)

    with pytest.raises(RuntimeError):
        run_async(
            processing_job_service.enqueue_revision_extraction_job(
                session,
                project_id=project.id,
                revision_id=revision.id,
                trigger_kind=ProcessingTriggerKind.MANUAL,
                actor_id=actor.id,
            )
        )

    assert session.rollback_count == 1


def test_processing_job_service_does_not_call_run_revision_extraction() -> None:
    source = inspect.getsource(processing_job_service)
    assert "run_revision_extraction" not in source
