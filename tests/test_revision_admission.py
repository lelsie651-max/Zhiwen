import asyncio
import inspect
import uuid
from datetime import datetime, timezone

import pytest

from app.models.document_revision import DocumentRevision, DocumentRevisionStatus, SourceAuthority, UploadIntent
from app.models.ingestion_validation import (
    IngestionValidationReport,
    ValidationRecommendedAuthority,
    ValidationReportOutcome,
    ValidationReportStatus,
    ValidationUserDecision,
)
from app.models.project import Project
from app.models.project_member import ProjectMember
from app.models.user import User
from app.schemas.revision_admission import RevisionAdmissionDecisionInput
from app.services import revision_admission as revision_admission_service


def run_async(awaitable):
    return asyncio.run(awaitable)


class FakeSession:
    def __init__(self) -> None:
        self.flush_called = False
        self.commit_called = False
        self.rollback_called = False

    async def flush(self) -> None:
        self.flush_called = True

    async def commit(self) -> None:
        self.commit_called = True

    async def rollback(self) -> None:
        self.rollback_called = True


def build_project(*, project_id: uuid.UUID | None = None, status: str = "active") -> Project:
    actual_project_id = project_id or uuid.uuid4()
    return Project(
        id=actual_project_id,
        name="Admission Project",
        slug=f"admission-{str(actual_project_id)[:8]}",
        created_by_id=uuid.uuid4(),
        status=status,
    )


def build_user(*, user_id: uuid.UUID | None = None, status: str = "active") -> User:
    actual_user_id = user_id or uuid.uuid4()
    return User(
        id=actual_user_id,
        handle=f"user-{str(actual_user_id)[:8]}",
        display_name="Admission Actor",
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
    document_id: uuid.UUID | None = None,
    uploaded_by_id: uuid.UUID | None = None,
    revision_id: uuid.UUID | None = None,
    status: str = DocumentRevisionStatus.PREFLIGHT_CHECKING.value,
    source_authority: str = SourceAuthority.PENDING.value,
) -> DocumentRevision:
    now = datetime.now(timezone.utc)
    return DocumentRevision(
        id=revision_id or uuid.uuid4(),
        document_id=document_id or uuid.uuid4(),
        revision_no=1,
        upload_intent=UploadIntent.NEW_DOCUMENT.value,
        supersedes_revision_id=None,
        original_filename="notes.pdf",
        storage_key="files/aa/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.pdf",
        mime_type="application/pdf",
        file_size_bytes=1234,
        sha256="a" * 64,
        detected_language=None,
        language_confidence=None,
        source_authority=source_authority,
        status=status,
        uploaded_by_id=uploaded_by_id or uuid.uuid4(),
        uploaded_at=now,
        updated_at=now,
    )


def build_report(
    *,
    revision_id: uuid.UUID,
    attempt_no: int = 1,
    outcome: str = ValidationReportOutcome.PASS.value,
    status: str = ValidationReportStatus.COMPLETED.value,
    recommended_authority: str | None = None,
    user_decision: str | None = None,
    decided_by_id: uuid.UUID | None = None,
    decided_at: datetime | None = None,
) -> IngestionValidationReport:
    return IngestionValidationReport(
        id=uuid.uuid4(),
        revision_id=revision_id,
        attempt_no=attempt_no,
        status=status,
        outcome=outcome,
        content_kind=None,
        content_confidence=None,
        recommended_authority=recommended_authority,
        detected_language=None,
        language_confidence=None,
        character_count=0,
        printable_ratio=None,
        garbled_ratio=None,
        repetition_ratio=None,
        file_checks={},
        text_checks={},
        classification_details={},
        summary=None,
        failure_code=None,
        failure_message=None,
        user_decision=user_decision,
        decided_by_id=decided_by_id,
        decided_at=decided_at,
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )


def patch_revision_and_report(monkeypatch, *, revision: DocumentRevision | None, report: IngestionValidationReport | None):
    call_order: list[str] = []

    async def fake_get_revision_for_admission_update(_session, *, project_id, revision_id):
        call_order.append("lock_revision")
        if revision is not None and revision_id == revision.id:
            return revision
        return None

    async def fake_get_latest_validation_report_for_update(_session, *, revision_id):
        call_order.append("lock_report")
        if report is not None and revision_id == report.revision_id:
            return report
        return None

    monkeypatch.setattr(
        revision_admission_service.revision_admission_repository,
        "get_revision_for_admission_update",
        fake_get_revision_for_admission_update,
    )
    monkeypatch.setattr(
        revision_admission_service.revision_admission_repository,
        "get_latest_validation_report_for_update",
        fake_get_latest_validation_report_for_update,
    )
    return call_order


def patch_actor_and_member(monkeypatch, *, actor: User | None, membership: ProjectMember | None):
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
        revision_admission_service.revision_admission_repository,
        "get_active_user_by_id",
        fake_get_active_user_by_id,
    )
    monkeypatch.setattr(
        revision_admission_service.revision_admission_repository,
        "get_project_member_for_project",
        fake_get_project_member_for_project,
    )


def test_pass_auto_enters_accepted(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    revision = build_revision(status=DocumentRevisionStatus.PREFLIGHT_CHECKING.value)
    report = build_report(revision_id=revision.id, outcome=ValidationReportOutcome.PASS.value)
    patch_revision_and_report(monkeypatch, revision=revision, report=report)

    result = run_async(
        revision_admission_service.resolve_revision_admission(
            session,
            project_id=project.id,
            revision_id=revision.id,
        )
    )

    assert result.revision_status == DocumentRevisionStatus.ACCEPTED
    assert result.source_authority == SourceAuthority.PENDING
    assert result.user_decision is None
    assert result.decision_required is False
    assert session.flush_called is True
    assert session.commit_called is True


def test_warning_enters_needs_confirmation(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    revision = build_revision()
    report = build_report(revision_id=revision.id, outcome=ValidationReportOutcome.WARNING.value)
    patch_revision_and_report(monkeypatch, revision=revision, report=report)

    result = run_async(
        revision_admission_service.resolve_revision_admission(
            session,
            project_id=project.id,
            revision_id=revision.id,
        )
    )

    assert result.revision_status == DocumentRevisionStatus.NEEDS_CONFIRMATION
    assert result.decision_required is True
    assert revision.status == DocumentRevisionStatus.NEEDS_CONFIRMATION.value


@pytest.mark.parametrize(
    ("outcome", "expected_status"),
    [
        (ValidationReportOutcome.REJECT.value, DocumentRevisionStatus.REJECTED),
        (ValidationReportOutcome.QUARANTINE.value, DocumentRevisionStatus.QUARANTINED),
    ],
)
def test_reject_and_quarantine_map_correctly(monkeypatch, outcome: str, expected_status) -> None:
    session = FakeSession()
    project = build_project()
    revision = build_revision()
    report = build_report(revision_id=revision.id, outcome=outcome)
    patch_revision_and_report(monkeypatch, revision=revision, report=report)

    result = run_async(
        revision_admission_service.resolve_revision_admission(
            session,
            project_id=project.id,
            revision_id=revision.id,
        )
    )

    assert result.revision_status == expected_status


def test_pass_can_apply_recommended_authority(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    revision = build_revision(source_authority=SourceAuthority.PENDING.value)
    report = build_report(
        revision_id=revision.id,
        outcome=ValidationReportOutcome.PASS.value,
        recommended_authority=ValidationRecommendedAuthority.FORMAL.value,
    )
    patch_revision_and_report(monkeypatch, revision=revision, report=report)

    result = run_async(
        revision_admission_service.resolve_revision_admission(
            session,
            project_id=project.id,
            revision_id=revision.id,
        )
    )

    assert result.source_authority == SourceAuthority.FORMAL
    assert revision.source_authority == SourceAuthority.FORMAL.value


def test_warning_does_not_auto_accept(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    revision = build_revision()
    report = build_report(revision_id=revision.id, outcome=ValidationReportOutcome.WARNING.value)
    patch_revision_and_report(monkeypatch, revision=revision, report=report)

    result = run_async(
        revision_admission_service.resolve_revision_admission(
            session,
            project_id=project.id,
            revision_id=revision.id,
        )
    )

    assert result.revision_status != DocumentRevisionStatus.ACCEPTED
    assert revision.status == DocumentRevisionStatus.NEEDS_CONFIRMATION.value


@pytest.mark.parametrize(
    ("decision", "expected_authority"),
    [
        (ValidationUserDecision.CONTINUE_FORMAL, SourceAuthority.FORMAL),
        (ValidationUserDecision.CONTINUE_REFERENCE, SourceAuthority.REFERENCE),
    ],
)
def test_formal_and_reference_decisions_set_authority(monkeypatch, decision, expected_authority) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    membership = build_project_member(project_id=project.id, user_id=actor.id, role="owner")
    revision = build_revision(status=DocumentRevisionStatus.NEEDS_CONFIRMATION.value)
    report = build_report(revision_id=revision.id, outcome=ValidationReportOutcome.WARNING.value)
    patch_revision_and_report(monkeypatch, revision=revision, report=report)
    patch_actor_and_member(monkeypatch, actor=actor, membership=membership)

    result = run_async(
        revision_admission_service.decide_revision_admission(
            session,
            project_id=project.id,
            actor_id=actor.id,
            revision_id=revision.id,
            payload=RevisionAdmissionDecisionInput(decision=decision),
        )
    )

    assert result.revision_status == DocumentRevisionStatus.ACCEPTED
    assert result.source_authority == expected_authority
    assert report.decided_by_id == actor.id
    assert report.decided_at is not None
    assert report.decided_at.tzinfo == timezone.utc


def test_cancel_decision_enters_rejected(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    membership = build_project_member(project_id=project.id, user_id=actor.id, role="editor")
    revision = build_revision(status=DocumentRevisionStatus.NEEDS_CONFIRMATION.value)
    report = build_report(revision_id=revision.id, outcome=ValidationReportOutcome.WARNING.value)
    patch_revision_and_report(monkeypatch, revision=revision, report=report)
    patch_actor_and_member(monkeypatch, actor=actor, membership=membership)

    result = run_async(
        revision_admission_service.decide_revision_admission(
            session,
            project_id=project.id,
            actor_id=actor.id,
            revision_id=revision.id,
            payload=RevisionAdmissionDecisionInput(decision=ValidationUserDecision.CANCEL),
        )
    )

    assert result.revision_status == DocumentRevisionStatus.REJECTED
    assert result.source_authority == SourceAuthority.PENDING
    assert report.user_decision == ValidationUserDecision.CANCEL.value


def test_owner_and_editor_allowed_but_viewer_rejected(monkeypatch) -> None:
    project = build_project()

    for role in ("owner", "editor"):
        session = FakeSession()
        actor = build_user()
        revision = build_revision(status=DocumentRevisionStatus.NEEDS_CONFIRMATION.value)
        report = build_report(revision_id=revision.id, outcome=ValidationReportOutcome.WARNING.value)
        patch_revision_and_report(monkeypatch, revision=revision, report=report)
        patch_actor_and_member(
            monkeypatch,
            actor=actor,
            membership=build_project_member(project_id=project.id, user_id=actor.id, role=role),
        )
        result = run_async(
            revision_admission_service.decide_revision_admission(
                session,
                project_id=project.id,
                actor_id=actor.id,
                revision_id=revision.id,
                payload=RevisionAdmissionDecisionInput(decision=ValidationUserDecision.CONTINUE_FORMAL),
            )
        )
        assert result.revision_status == DocumentRevisionStatus.ACCEPTED

    session = FakeSession()
    actor = build_user()
    revision = build_revision(status=DocumentRevisionStatus.NEEDS_CONFIRMATION.value)
    report = build_report(revision_id=revision.id, outcome=ValidationReportOutcome.WARNING.value)
    patch_revision_and_report(monkeypatch, revision=revision, report=report)
    patch_actor_and_member(
        monkeypatch,
        actor=actor,
        membership=build_project_member(project_id=project.id, user_id=actor.id, role="viewer"),
    )
    with pytest.raises(revision_admission_service.RevisionAdmissionPermissionError):
        run_async(
            revision_admission_service.decide_revision_admission(
                session,
                project_id=project.id,
                actor_id=actor.id,
                revision_id=revision.id,
                payload=RevisionAdmissionDecisionInput(decision=ValidationUserDecision.CONTINUE_FORMAL),
            )
        )


def test_same_decision_is_idempotent(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    decided_at = datetime.now(timezone.utc)
    revision = build_revision(
        status=DocumentRevisionStatus.ACCEPTED.value,
        source_authority=SourceAuthority.FORMAL.value,
    )
    report = build_report(
        revision_id=revision.id,
        outcome=ValidationReportOutcome.WARNING.value,
        user_decision=ValidationUserDecision.CONTINUE_FORMAL.value,
        decided_by_id=actor.id,
        decided_at=decided_at,
    )
    patch_revision_and_report(monkeypatch, revision=revision, report=report)
    patch_actor_and_member(
        monkeypatch,
        actor=actor,
        membership=build_project_member(project_id=project.id, user_id=actor.id, role="owner"),
    )

    result = run_async(
        revision_admission_service.decide_revision_admission(
            session,
            project_id=project.id,
            actor_id=actor.id,
            revision_id=revision.id,
            payload=RevisionAdmissionDecisionInput(decision=ValidationUserDecision.CONTINUE_FORMAL),
        )
    )

    assert result.user_decision == ValidationUserDecision.CONTINUE_FORMAL
    assert report.decided_at == decided_at
    assert session.flush_called is False
    assert session.commit_called is False


def test_different_decision_cannot_overwrite(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    revision = build_revision(
        status=DocumentRevisionStatus.ACCEPTED.value,
        source_authority=SourceAuthority.FORMAL.value,
    )
    report = build_report(
        revision_id=revision.id,
        outcome=ValidationReportOutcome.WARNING.value,
        user_decision=ValidationUserDecision.CONTINUE_FORMAL.value,
        decided_by_id=actor.id,
        decided_at=datetime.now(timezone.utc),
    )
    patch_revision_and_report(monkeypatch, revision=revision, report=report)
    patch_actor_and_member(
        monkeypatch,
        actor=actor,
        membership=build_project_member(project_id=project.id, user_id=actor.id, role="owner"),
    )

    with pytest.raises(revision_admission_service.RevisionAdmissionStateError):
        run_async(
            revision_admission_service.decide_revision_admission(
                session,
                project_id=project.id,
                actor_id=actor.id,
                revision_id=revision.id,
                payload=RevisionAdmissionDecisionInput(decision=ValidationUserDecision.CANCEL),
            )
        )


def test_later_processing_status_cannot_be_overwritten(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    revision = build_revision(status=DocumentRevisionStatus.PARSING.value)
    report = build_report(revision_id=revision.id, outcome=ValidationReportOutcome.PASS.value)
    patch_revision_and_report(monkeypatch, revision=revision, report=report)

    with pytest.raises(revision_admission_service.RevisionAdmissionStateCorruptionError):
        run_async(
            revision_admission_service.resolve_revision_admission(
                session,
                project_id=project.id,
                revision_id=revision.id,
            )
        )


def test_latest_attempt_report_is_used(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    revision = build_revision()
    latest_report = build_report(revision_id=revision.id, attempt_no=2, outcome=ValidationReportOutcome.WARNING.value)
    patch_revision_and_report(monkeypatch, revision=revision, report=latest_report)

    result = run_async(
        revision_admission_service.resolve_revision_admission(
            session,
            project_id=project.id,
            revision_id=revision.id,
        )
    )

    assert result.report_id == latest_report.id
    assert result.revision_status == DocumentRevisionStatus.NEEDS_CONFIRMATION


@pytest.mark.parametrize(
    ("report", "error_type"),
    [
        (None, revision_admission_service.RevisionAdmissionStateCorruptionError),
        (lambda revision_id: build_report(revision_id=revision_id, status=ValidationReportStatus.RUNNING.value), revision_admission_service.RevisionAdmissionStateError),
    ],
)
def test_missing_or_incomplete_report_is_rejected(monkeypatch, report, error_type) -> None:
    session = FakeSession()
    project = build_project()
    revision = build_revision()
    actual_report = report(revision.id) if callable(report) else report
    patch_revision_and_report(monkeypatch, revision=revision, report=actual_report)

    with pytest.raises(error_type):
        run_async(
            revision_admission_service.resolve_revision_admission(
                session,
                project_id=project.id,
                revision_id=revision.id,
            )
        )


def test_state_corruption_is_rejected(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    revision = build_revision(
        status=DocumentRevisionStatus.ACCEPTED.value,
        source_authority=SourceAuthority.REFERENCE.value,
    )
    report = build_report(
        revision_id=revision.id,
        outcome=ValidationReportOutcome.WARNING.value,
        user_decision=ValidationUserDecision.CONTINUE_FORMAL.value,
        decided_by_id=uuid.uuid4(),
        decided_at=datetime.now(timezone.utc),
    )
    patch_revision_and_report(monkeypatch, revision=revision, report=report)

    with pytest.raises(revision_admission_service.RevisionAdmissionStateCorruptionError):
        run_async(
            revision_admission_service.resolve_revision_admission(
                session,
                project_id=project.id,
                revision_id=revision.id,
            )
        )


def test_failure_rolls_back_transaction(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    revision = build_revision()
    report = build_report(revision_id=revision.id, outcome=ValidationReportOutcome.PASS.value)
    patch_revision_and_report(monkeypatch, revision=revision, report=report)

    async def failing_flush():
        session.flush_called = True
        raise RuntimeError("flush failed")

    session.flush = failing_flush

    with pytest.raises(RuntimeError):
        run_async(
            revision_admission_service.resolve_revision_admission(
                session,
                project_id=project.id,
                revision_id=revision.id,
            )
        )

    assert session.rollback_called is True
    assert session.commit_called is False


def test_decision_lock_order_and_actor_injection(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    membership = build_project_member(project_id=project.id, user_id=actor.id, role="owner")
    revision = build_revision(status=DocumentRevisionStatus.NEEDS_CONFIRMATION.value)
    report = build_report(revision_id=revision.id, outcome=ValidationReportOutcome.WARNING.value)
    call_order = patch_revision_and_report(monkeypatch, revision=revision, report=report)

    async def fake_get_active_user_by_id(_session, *, user_id):
        call_order.append("load_actor")
        if user_id == actor.id:
            return actor
        return None

    async def fake_get_project_member_for_project(_session, *, project_id, user_id):
        call_order.append("load_member")
        if project_id == membership.project_id and user_id == membership.user_id:
            return membership
        return None

    monkeypatch.setattr(
        revision_admission_service.revision_admission_repository,
        "get_active_user_by_id",
        fake_get_active_user_by_id,
    )
    monkeypatch.setattr(
        revision_admission_service.revision_admission_repository,
        "get_project_member_for_project",
        fake_get_project_member_for_project,
    )

    result = run_async(
        revision_admission_service.decide_revision_admission(
            session,
            project_id=project.id,
            actor_id=actor.id,
            revision_id=revision.id,
            payload=RevisionAdmissionDecisionInput(decision=ValidationUserDecision.CONTINUE_REFERENCE),
        )
    )

    assert call_order == ["lock_revision", "lock_report", "load_actor", "load_member"]
    assert result.user_decision == ValidationUserDecision.CONTINUE_REFERENCE
    assert report.decided_by_id == actor.id
    assert report.decided_at is not None


def test_service_does_not_call_extraction_ai_or_fact_services() -> None:
    source = inspect.getsource(revision_admission_service)
    assert "extract_document" not in source
    assert "persist_extraction_result" not in source
    assert "propose_ai_fact_value" not in source
    assert "create_human_fact_value" not in source
