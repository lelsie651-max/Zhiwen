import asyncio
import inspect
import io
import re
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from starlette.testclient import TestClient

from app.core.database import get_db_session
from app.main import app
from app.models.document_content import ExtractionRun, ExtractionRunOutcome, ExtractionRunStatus
from app.models.document import Document, DocumentStatus
from app.models.document_revision import (
    DocumentRevision,
    DocumentRevisionStatus,
    SourceAuthority,
    UploadIntent,
)
from app.models.ingestion_validation import (
    IngestionRiskFlag,
    IngestionRiskSeverity,
    IngestionValidationReport,
    ValidationRecommendedAuthority,
    ValidationReportOutcome,
    ValidationReportStatus,
    ValidationUserDecision,
)
from app.models.processing_job import ProcessingJob, ProcessingJobStatus, ProcessingJobType, ProcessingTriggerKind
from app.models.project import Project, ProjectStatus, ProjectVisibility
from app.models.project_member import ProjectMember, ProjectMemberRole
from app.models.user import User, UserStatus
from app.repositories import document as document_repository
from app.routers import projects as projects_router
from app.schemas.document_upload import DocumentUploadSubmit
from app.schemas.file_ingestion import (
    DetectedFileFormat,
    ReceivedUpload,
    TechnicalPreflightOutcome,
    TechnicalPreflightResult,
    TechnicalRiskFlag,
    TempFileReference,
)
from app.services import document_upload as document_upload_service
from app.services import identity as identity_service
from app.services import processing_job as processing_job_service
from app.services import revision_admission as revision_admission_service


def extract_csrf_token(html: str) -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', html)
    assert match is not None
    return match.group(1)


def run_async(awaitable):
    return asyncio.run(awaitable)


def build_user(
    *,
    user_id: uuid.UUID | None = None,
    handle: str = "writer01",
    display_name: str = "织文用户",
) -> User:
    now = datetime.now(timezone.utc)
    return User(
        id=user_id or uuid.uuid4(),
        handle=handle,
        display_name=display_name,
        email=None,
        avatar_url=None,
        locale="zh-CN",
        timezone="Asia/Shanghai",
        status=UserStatus.ACTIVE.value,
        created_at=now,
        updated_at=now,
    )


def build_project(
    *,
    owner_id: uuid.UUID,
    slug: str = "project-a",
    name: str = "织文项目",
) -> Project:
    now = datetime.now(timezone.utc)
    return Project(
        id=uuid.uuid4(),
        name=name,
        slug=slug,
        description="项目说明",
        visibility=ProjectVisibility.PRIVATE.value,
        status=ProjectStatus.ACTIVE.value,
        created_by_id=owner_id,
        created_at=now,
        updated_at=now,
    )


def build_document(
    *,
    project_id: uuid.UUID,
    created_by_id: uuid.UUID,
    title: str = "会议纪要",
) -> Document:
    now = datetime.now(timezone.utc)
    return Document(
        id=uuid.uuid4(),
        project_id=project_id,
        title=title,
        description="文档描述",
        status=DocumentStatus.ACTIVE.value,
        created_by_id=created_by_id,
        current_revision_id=None,
        logical_order_kind="chapter",
        logical_order_value="1",
        logical_order_index=Decimal("1.0"),
        effective_from=None,
        effective_to=None,
        created_at=now,
        updated_at=now,
    )


def build_revision(
    *,
    document_id: uuid.UUID,
    uploaded_by_id: uuid.UUID,
    revision_no: int = 1,
    supersedes_revision_id: uuid.UUID | None = None,
    status: str = DocumentRevisionStatus.PREFLIGHT_CHECKING.value,
    storage_key: str = "files/aa/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.pdf",
) -> DocumentRevision:
    now = datetime.now(timezone.utc)
    return DocumentRevision(
        id=uuid.uuid4(),
        document_id=document_id,
        revision_no=revision_no,
        upload_intent=UploadIntent.NEW_DOCUMENT.value if revision_no == 1 else UploadIntent.REVISION.value,
        supersedes_revision_id=supersedes_revision_id,
        original_filename="notes.pdf",
        storage_key=storage_key,
        mime_type="application/pdf",
        file_size_bytes=1234,
        sha256="a" * 64,
        detected_language=None,
        language_confidence=None,
        source_authority=SourceAuthority.PENDING.value,
        status=status,
        uploaded_by_id=uploaded_by_id,
        uploaded_at=now,
        updated_at=now,
    )


def build_membership(project: Project, user: User, role: ProjectMemberRole) -> ProjectMember:
    membership = ProjectMember(
        id=uuid.uuid4(),
        project_id=project.id,
        user_id=user.id,
        role=role.value,
        joined_at=datetime.now(timezone.utc),
    )
    membership.project = project
    membership.user = user
    return membership


def build_received_upload(*, extension: str = ".pdf", declared_mime: str | None = "application/pdf") -> ReceivedUpload:
    return ReceivedUpload(
        temp_file=TempFileReference(temp_key="tmp/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.upload"),
        safe_original_filename=f"upload{extension}",
        filename_sanitized=False,
        file_size=128,
        sha256="a" * 64,
        declared_mime=declared_mime,
        extension=extension,
    )


def build_preflight_result(
    *,
    outcome: TechnicalPreflightOutcome,
    detected_format: DetectedFileFormat = DetectedFileFormat.PDF,
    detected_mime: str = "application/pdf",
    risk_flags: list[TechnicalRiskFlag] | None = None,
) -> TechnicalPreflightResult:
    return TechnicalPreflightResult(
        outcome=outcome,
        detected_format=detected_format,
        detected_mime=detected_mime,
        detected_encoding=None,
        file_checks={"header_valid": True},
        risk_flags=risk_flags or [],
    )


class FakeSession:
    def __init__(self) -> None:
        self.commit_called = False
        self.rollback_called = False

    async def commit(self) -> None:
        self.commit_called = True

    async def rollback(self) -> None:
        self.rollback_called = True


class FakeStorage:
    def __init__(
        self,
        *,
        storage_key: str = "files/aa/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.pdf",
        promote_error: Exception | None = None,
        delete_file_error: Exception | None = None,
        delete_temp_error: Exception | None = None,
    ) -> None:
        self.storage_key = storage_key
        self.promote_error = promote_error
        self.delete_file_error = delete_file_error
        self.delete_temp_error = delete_temp_error
        self.promoted = False
        self.deleted_keys: list[str] = []
        self.deleted_temp_keys: list[str] = []

    async def promote_temp_file(self, _temp_file: TempFileReference, *, suffix: str = "") -> str:
        if self.promote_error is not None:
            raise self.promote_error
        self.promoted = True
        return self.storage_key if suffix else self.storage_key

    async def delete_file(self, storage_key: str) -> None:
        if self.delete_file_error is not None:
            raise self.delete_file_error
        self.deleted_keys.append(storage_key)

    async def delete_temp_file(self, temp_file: TempFileReference) -> None:
        if self.delete_temp_error is not None:
            raise self.delete_temp_error
        self.deleted_temp_keys.append(temp_file.temp_key)


class FakeBackgroundTasks:
    def __init__(self, *, add_error: Exception | None = None) -> None:
        self.add_error = add_error
        self.calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    def add_task(self, func, *args, **kwargs) -> None:
        if self.add_error is not None:
            raise self.add_error
        self.calls.append((func, args, kwargs))


class AsyncUploadFileStub:
    def __init__(self, *, filename: str = "notes.pdf", content_type: str = "application/pdf") -> None:
        self.filename = filename
        self.content_type = content_type
        self.closed = False

    async def read(self, size: int = -1) -> bytes:
        return b""

    async def close(self) -> None:
        self.closed = True


def build_processing_job(*, project_id: uuid.UUID, revision_id: uuid.UUID) -> ProcessingJob:
    now = datetime.now(timezone.utc)
    return ProcessingJob(
        id=uuid.uuid4(),
        project_id=project_id,
        revision_id=revision_id,
        job_type=ProcessingJobType.REVISION_EXTRACTION.value,
        status=ProcessingJobStatus.QUEUED.value,
        trigger_kind=ProcessingTriggerKind.AUTOMATIC.value,
        attempt_no=1,
        requested_by_id=None,
        lease_token=None,
        lease_expires_at=None,
        started_at=None,
        completed_at=None,
        result_extraction_run_id=None,
        failure_code=None,
        failure_message=None,
        created_at=now,
        updated_at=now,
    )


def build_extraction_run(*, revision_id: uuid.UUID) -> ExtractionRun:
    now = datetime.now(timezone.utc)
    return ExtractionRun(
        id=uuid.uuid4(),
        revision_id=revision_id,
        attempt_no=1,
        status=ExtractionRunStatus.COMPLETED.value,
        outcome=ExtractionRunOutcome.SUCCESS.value,
        extractor_name="zhiwen-deterministic-extractor",
        extractor_version="1.0.0",
        detected_format=DetectedFileFormat.PDF.value,
        detected_encoding=None,
        page_count=1,
        character_count=128,
        block_count=4,
        warnings=[],
        content_metadata={},
        failure_code=None,
        failure_message=None,
        started_at=now - timedelta(seconds=5),
        completed_at=now,
        created_at=now,
    )


def build_revision_detail_context(
    *,
    membership_role: ProjectMemberRole = ProjectMemberRole.OWNER,
    revision_status: str = DocumentRevisionStatus.ACCEPTED.value,
    latest_processing_job: ProcessingJob | None = None,
    latest_extraction_run: ExtractionRun | None = None,
    report_outcome: str = ValidationReportOutcome.PASS.value,
    user_decision: str | None = None,
) -> tuple[User, Project, Document, DocumentRevision, document_upload_service.RevisionDetailContext]:
    user = build_user()
    project = build_project(owner_id=user.id)
    membership = build_membership(project, user, membership_role)
    document = build_document(project_id=project.id, created_by_id=user.id)
    revision = build_revision(
        document_id=document.id,
        uploaded_by_id=user.id,
        status=revision_status,
    )
    revision.document = document
    revision.uploaded_by = user
    report = IngestionValidationReport(
        id=uuid.uuid4(),
        revision_id=revision.id,
        attempt_no=1,
        status=ValidationReportStatus.COMPLETED.value,
        outcome=report_outcome,
        user_decision=user_decision,
        file_checks={"detected_mime": "application/pdf"},
        text_checks={},
        classification_details={},
        character_count=0,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    context = document_upload_service.RevisionDetailContext(
        project=project,
        membership=membership,
        revision=revision,
        report=report,
        risk_flags=[],
        latest_processing_job=latest_processing_job,
        latest_extraction_run=latest_extraction_run,
    )
    return user, project, document, revision, context


@pytest.fixture
def fake_db_session() -> object:
    return object()


@pytest.fixture(autouse=True)
def override_db_dependency(fake_db_session: object):
    async def _override() -> AsyncIterator[object]:
        yield fake_db_session

    app.dependency_overrides[get_db_session] = _override
    yield fake_db_session
    app.dependency_overrides.clear()


def login_client(client: TestClient, monkeypatch, user: User) -> None:
    async def fake_register_user(_session: object, _payload) -> User:
        return user

    async def fake_get_active_user_by_id(_session: object, user_id: uuid.UUID) -> User | None:
        return user if user_id == user.id else None

    monkeypatch.setattr(identity_service, "register_user", fake_register_user)
    monkeypatch.setattr(identity_service, "get_active_user_by_id", fake_get_active_user_by_id)

    setup_page = client.get("/setup")
    csrf_token = extract_csrf_token(setup_page.text)
    response = client.post(
        "/setup",
        data={
            "handle": user.handle,
            "display_name": user.display_name,
            "email": "",
            "locale": "zh-CN",
            "timezone": "Asia/Shanghai",
            "csrf_token": csrf_token,
        },
        follow_redirects=False,
    )

    assert response.status_code == 303


@pytest.mark.parametrize(
    ("outcome", "risk_flags", "expected_report_outcome", "expected_revision_status"),
    [
        (
            TechnicalPreflightOutcome.PASS,
            [],
            ValidationReportOutcome.PASS.value,
            DocumentRevisionStatus.PREFLIGHT_CHECKING.value,
        ),
        (
            TechnicalPreflightOutcome.PASS,
            [
                TechnicalRiskFlag(
                    category="file",
                    kind="declared_mime_mismatch",
                    severity="warning",
                    detector="deterministic",
                    message="warning",
                )
            ],
            ValidationReportOutcome.WARNING.value,
            DocumentRevisionStatus.PREFLIGHT_CHECKING.value,
        ),
        (
            TechnicalPreflightOutcome.REJECT,
            [
                TechnicalRiskFlag(
                    category="file",
                    kind="encrypted_pdf",
                    severity="blocker",
                    detector="deterministic",
                    message="reject",
                )
            ],
            ValidationReportOutcome.REJECT.value,
            DocumentRevisionStatus.REJECTED.value,
        ),
        (
            TechnicalPreflightOutcome.QUARANTINE,
            [
                TechnicalRiskFlag(
                    category="security",
                    kind="suspicious_docx_archive",
                    severity="blocker",
                    detector="deterministic",
                    message="quarantine",
                )
            ],
            ValidationReportOutcome.QUARANTINE.value,
            DocumentRevisionStatus.QUARANTINED.value,
        ),
    ],
)
def test_new_document_upload_creates_document_revision_report_and_risk_flags(
    monkeypatch,
    outcome: TechnicalPreflightOutcome,
    risk_flags: list[TechnicalRiskFlag],
    expected_report_outcome: str,
    expected_revision_status: str,
) -> None:
    session = FakeSession()
    user = build_user()
    project = build_project(owner_id=user.id)
    membership = build_membership(project, user, ProjectMemberRole.OWNER)
    storage = FakeStorage()
    created: dict[str, object] = {}

    async def fake_get_project_member_for_user_by_slug(_session, _user_id, _slug):
        return membership

    async def fake_list_documents_for_project(_session, _project_id):
        return []

    async def fake_receive_upload_stream(*_args, **_kwargs):
        return build_received_upload()

    async def fake_run_technical_preflight(*_args, **_kwargs):
        return build_preflight_result(outcome=outcome, risk_flags=risk_flags)

    async def fake_create_document(_session, document: Document) -> Document:
        document.id = uuid.uuid4()
        created["document"] = document
        return document

    async def fake_create_document_revision(_session, revision: DocumentRevision) -> DocumentRevision:
        revision.id = uuid.uuid4()
        created["revision"] = revision
        return revision

    async def fake_create_validation_report(_session, report: IngestionValidationReport) -> IngestionValidationReport:
        report.id = uuid.uuid4()
        created["report"] = report
        return report

    async def fake_create_risk_flags(_session, items: list[IngestionRiskFlag]) -> list[IngestionRiskFlag]:
        created["risk_flags"] = items
        for item in items:
            item.id = uuid.uuid4()
        return items

    monkeypatch.setattr(
        document_upload_service.project_repository,
        "get_project_member_for_user_by_slug",
        fake_get_project_member_for_user_by_slug,
    )
    monkeypatch.setattr(
        document_upload_service.document_repository,
        "list_documents_for_project",
        fake_list_documents_for_project,
    )
    monkeypatch.setattr(document_upload_service, "receive_upload_stream", fake_receive_upload_stream)
    monkeypatch.setattr(document_upload_service, "run_technical_preflight", fake_run_technical_preflight)
    monkeypatch.setattr(document_upload_service.document_repository, "create_document", fake_create_document)
    monkeypatch.setattr(
        document_upload_service.document_repository,
        "create_document_revision",
        fake_create_document_revision,
    )
    monkeypatch.setattr(
        document_upload_service.validation_repository,
        "create_validation_report",
        fake_create_validation_report,
    )
    monkeypatch.setattr(
        document_upload_service.validation_repository,
        "create_risk_flags",
        fake_create_risk_flags,
    )

    result = run_async(
        document_upload_service.upload_document_for_project(
            session,
            user_id=user.id,
            project_slug=project.slug,
            payload=DocumentUploadSubmit(upload_intent="new_document", title="会议纪要"),
            chunks=iter([b"data"]),
            original_filename="upload.pdf",
            declared_mime="application/pdf",
            storage=storage,
        )
    )

    assert session.commit_called is True
    assert created["document"] is result.document
    assert created["revision"].revision_no == 1
    assert created["revision"].upload_intent == UploadIntent.NEW_DOCUMENT.value
    assert created["revision"].supersedes_revision_id is None
    assert created["revision"].status == expected_revision_status
    assert created["document"].current_revision_id is None
    assert created["report"].attempt_no == 1
    assert created["report"].status == ValidationReportStatus.COMPLETED.value
    assert created["report"].outcome == expected_report_outcome
    assert result.report.file_checks["detected_mime"] == "application/pdf"
    assert len(result.risk_flags) == len(risk_flags)


def test_revision_upload_uses_row_lock_and_auto_numbering(monkeypatch) -> None:
    session = FakeSession()
    user = build_user()
    project = build_project(owner_id=user.id)
    membership = build_membership(project, user, ProjectMemberRole.EDITOR)
    document = build_document(project_id=project.id, created_by_id=user.id)
    previous_revision = build_revision(
        document_id=document.id,
        uploaded_by_id=user.id,
        revision_no=3,
    )
    storage = FakeStorage()
    captured = {"locked": False}

    async def fake_get_project_member_for_user_by_slug(_session, _user_id, _slug):
        return membership

    async def fake_list_documents_for_project(_session, _project_id):
        return [document]

    async def fake_receive_upload_stream(*_args, **_kwargs):
        return build_received_upload()

    async def fake_run_technical_preflight(*_args, **_kwargs):
        return build_preflight_result(outcome=TechnicalPreflightOutcome.PASS)

    async def fake_get_document_for_update(_session, _project_id, _document_id):
        captured["locked"] = True
        return document

    async def fake_get_latest_revision_for_document(_session, _document_id):
        return previous_revision

    async def fake_create_document_revision(_session, revision: DocumentRevision):
        revision.id = uuid.uuid4()
        captured["revision"] = revision
        return revision

    async def fake_create_validation_report(_session, report: IngestionValidationReport):
        report.id = uuid.uuid4()
        return report

    monkeypatch.setattr(
        document_upload_service.project_repository,
        "get_project_member_for_user_by_slug",
        fake_get_project_member_for_user_by_slug,
    )
    monkeypatch.setattr(
        document_upload_service.document_repository,
        "list_documents_for_project",
        fake_list_documents_for_project,
    )
    monkeypatch.setattr(document_upload_service, "receive_upload_stream", fake_receive_upload_stream)
    monkeypatch.setattr(document_upload_service, "run_technical_preflight", fake_run_technical_preflight)
    monkeypatch.setattr(
        document_upload_service.document_repository,
        "get_document_for_update",
        fake_get_document_for_update,
    )
    monkeypatch.setattr(
        document_upload_service.document_repository,
        "get_latest_revision_for_document",
        fake_get_latest_revision_for_document,
    )
    monkeypatch.setattr(
        document_upload_service.document_repository,
        "create_document_revision",
        fake_create_document_revision,
    )
    monkeypatch.setattr(
        document_upload_service.validation_repository,
        "create_validation_report",
        fake_create_validation_report,
    )

    result = run_async(
        document_upload_service.upload_document_for_project(
            session,
            user_id=user.id,
            project_slug=project.slug,
            payload=DocumentUploadSubmit(upload_intent="revision", document_id=str(document.id)),
            chunks=iter([b"data"]),
            original_filename="upload.pdf",
            declared_mime="application/pdf",
            storage=storage,
        )
    )

    assert captured["locked"] is True
    assert captured["revision"].revision_no == 4
    assert captured["revision"].supersedes_revision_id == previous_revision.id
    assert result.document.id == document.id
    assert session.commit_called is True


def test_database_failure_deletes_promoted_file(monkeypatch) -> None:
    session = FakeSession()
    user = build_user()
    project = build_project(owner_id=user.id)
    membership = build_membership(project, user, ProjectMemberRole.OWNER)
    storage = FakeStorage(storage_key="files/ab/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.pdf")

    async def fake_get_project_member_for_user_by_slug(_session, _user_id, _slug):
        return membership

    async def fake_list_documents_for_project(_session, _project_id):
        return []

    async def fake_receive_upload_stream(*_args, **_kwargs):
        return build_received_upload()

    async def fake_run_technical_preflight(*_args, **_kwargs):
        return build_preflight_result(outcome=TechnicalPreflightOutcome.PASS)

    async def fake_create_document(_session, document: Document):
        document.id = uuid.uuid4()
        return document

    async def fake_create_document_revision(_session, _revision: DocumentRevision):
        raise RuntimeError("db failed")

    monkeypatch.setattr(
        document_upload_service.project_repository,
        "get_project_member_for_user_by_slug",
        fake_get_project_member_for_user_by_slug,
    )
    monkeypatch.setattr(
        document_upload_service.document_repository,
        "list_documents_for_project",
        fake_list_documents_for_project,
    )
    monkeypatch.setattr(document_upload_service, "receive_upload_stream", fake_receive_upload_stream)
    monkeypatch.setattr(document_upload_service, "run_technical_preflight", fake_run_technical_preflight)
    monkeypatch.setattr(document_upload_service.document_repository, "create_document", fake_create_document)
    monkeypatch.setattr(
        document_upload_service.document_repository,
        "create_document_revision",
        fake_create_document_revision,
    )

    with pytest.raises(RuntimeError, match="db failed"):
        run_async(
            document_upload_service.upload_document_for_project(
                session,
                user_id=user.id,
                project_slug=project.slug,
                payload=DocumentUploadSubmit(upload_intent="new_document", title="失败文档"),
                chunks=iter([b"data"]),
                original_filename="upload.pdf",
                declared_mime="application/pdf",
                storage=storage,
            )
        )

    assert session.rollback_called is True
    assert storage.deleted_keys == [storage.storage_key]


def test_promotion_failure_does_not_write_database(monkeypatch) -> None:
    session = FakeSession()
    user = build_user()
    project = build_project(owner_id=user.id)
    membership = build_membership(project, user, ProjectMemberRole.OWNER)
    storage = FakeStorage(promote_error=RuntimeError("promote failed"))
    called = {"create_document": False}

    async def fake_get_project_member_for_user_by_slug(_session, _user_id, _slug):
        return membership

    async def fake_list_documents_for_project(_session, _project_id):
        return []

    async def fake_receive_upload_stream(*_args, **_kwargs):
        return build_received_upload()

    async def fake_run_technical_preflight(*_args, **_kwargs):
        return build_preflight_result(outcome=TechnicalPreflightOutcome.PASS)

    async def fake_create_document(_session, _document: Document):
        called["create_document"] = True
        return _document

    monkeypatch.setattr(
        document_upload_service.project_repository,
        "get_project_member_for_user_by_slug",
        fake_get_project_member_for_user_by_slug,
    )
    monkeypatch.setattr(
        document_upload_service.document_repository,
        "list_documents_for_project",
        fake_list_documents_for_project,
    )
    monkeypatch.setattr(document_upload_service, "receive_upload_stream", fake_receive_upload_stream)
    monkeypatch.setattr(document_upload_service, "run_technical_preflight", fake_run_technical_preflight)
    monkeypatch.setattr(document_upload_service.document_repository, "create_document", fake_create_document)

    with pytest.raises(RuntimeError, match="promote failed"):
        run_async(
            document_upload_service.upload_document_for_project(
                session,
                user_id=user.id,
                project_slug=project.slug,
                payload=DocumentUploadSubmit(upload_intent="new_document", title="文档"),
                chunks=iter([b"data"]),
                original_filename="upload.pdf",
                declared_mime="application/pdf",
                storage=storage,
            )
        )

    assert called["create_document"] is False
    assert session.commit_called is False
    assert session.rollback_called is False


def test_db_failure_and_delete_failure_still_raises_original_exception(monkeypatch) -> None:
    session = FakeSession()
    user = build_user()
    project = build_project(owner_id=user.id)
    membership = build_membership(project, user, ProjectMemberRole.OWNER)
    storage = FakeStorage(
        delete_file_error=RuntimeError("cleanup failed"),
    )
    logged_messages: list[str] = []

    async def fake_get_project_member_for_user_by_slug(_session, _user_id, _slug):
        return membership

    async def fake_list_documents_for_project(_session, _project_id):
        return []

    async def fake_receive_upload_stream(*_args, **_kwargs):
        return build_received_upload()

    async def fake_run_technical_preflight(*_args, **_kwargs):
        return build_preflight_result(outcome=TechnicalPreflightOutcome.PASS)

    async def fake_create_document(_session, document: Document):
        document.id = uuid.uuid4()
        return document

    async def fake_create_document_revision(_session, _revision: DocumentRevision):
        raise RuntimeError("db failed")

    def fake_warning(message: str, *args, **kwargs) -> None:
        logged_messages.append(message)

    monkeypatch.setattr(
        document_upload_service.project_repository,
        "get_project_member_for_user_by_slug",
        fake_get_project_member_for_user_by_slug,
    )
    monkeypatch.setattr(
        document_upload_service.document_repository,
        "list_documents_for_project",
        fake_list_documents_for_project,
    )
    monkeypatch.setattr(document_upload_service, "receive_upload_stream", fake_receive_upload_stream)
    monkeypatch.setattr(document_upload_service, "run_technical_preflight", fake_run_technical_preflight)
    monkeypatch.setattr(document_upload_service.document_repository, "create_document", fake_create_document)
    monkeypatch.setattr(
        document_upload_service.document_repository,
        "create_document_revision",
        fake_create_document_revision,
    )
    monkeypatch.setattr(document_upload_service.logger, "warning", fake_warning)

    with pytest.raises(RuntimeError, match="db failed"):
        run_async(
            document_upload_service.upload_document_for_project(
                session,
                user_id=user.id,
                project_slug=project.slug,
                payload=DocumentUploadSubmit(upload_intent="new_document", title="失败文档"),
                chunks=iter([b"data"]),
                original_filename="upload.pdf",
                declared_mime="application/pdf",
                storage=storage,
            )
        )

    assert session.rollback_called is True
    assert logged_messages == ["Failed to delete promoted upload file during compensation cleanup."]


def test_promotion_failure_attempts_temp_cleanup(monkeypatch) -> None:
    session = FakeSession()
    user = build_user()
    project = build_project(owner_id=user.id)
    membership = build_membership(project, user, ProjectMemberRole.OWNER)
    storage = FakeStorage(promote_error=RuntimeError("promote failed"))

    async def fake_get_project_member_for_user_by_slug(_session, _user_id, _slug):
        return membership

    async def fake_list_documents_for_project(_session, _project_id):
        return []

    async def fake_receive_upload_stream(*_args, **_kwargs):
        return build_received_upload()

    async def fake_run_technical_preflight(*_args, **_kwargs):
        return build_preflight_result(outcome=TechnicalPreflightOutcome.PASS)

    monkeypatch.setattr(
        document_upload_service.project_repository,
        "get_project_member_for_user_by_slug",
        fake_get_project_member_for_user_by_slug,
    )
    monkeypatch.setattr(
        document_upload_service.document_repository,
        "list_documents_for_project",
        fake_list_documents_for_project,
    )
    monkeypatch.setattr(document_upload_service, "receive_upload_stream", fake_receive_upload_stream)
    monkeypatch.setattr(document_upload_service, "run_technical_preflight", fake_run_technical_preflight)

    with pytest.raises(RuntimeError, match="promote failed"):
        run_async(
            document_upload_service.upload_document_for_project(
                session,
                user_id=user.id,
                project_slug=project.slug,
                payload=DocumentUploadSubmit(upload_intent="new_document", title="文档"),
                chunks=iter([b"data"]),
                original_filename="upload.pdf",
                declared_mime="application/pdf",
                storage=storage,
            )
        )

    assert storage.deleted_temp_keys == ["tmp/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.upload"]


def test_temp_cleanup_failure_does_not_override_promotion_exception(monkeypatch) -> None:
    session = FakeSession()
    user = build_user()
    project = build_project(owner_id=user.id)
    membership = build_membership(project, user, ProjectMemberRole.OWNER)
    storage = FakeStorage(
        promote_error=RuntimeError("promote failed"),
        delete_temp_error=RuntimeError("temp cleanup failed"),
    )
    logged_messages: list[str] = []

    async def fake_get_project_member_for_user_by_slug(_session, _user_id, _slug):
        return membership

    async def fake_list_documents_for_project(_session, _project_id):
        return []

    async def fake_receive_upload_stream(*_args, **_kwargs):
        return build_received_upload()

    async def fake_run_technical_preflight(*_args, **_kwargs):
        return build_preflight_result(outcome=TechnicalPreflightOutcome.PASS)

    def fake_warning(message: str, *args, **kwargs) -> None:
        logged_messages.append(message)

    monkeypatch.setattr(
        document_upload_service.project_repository,
        "get_project_member_for_user_by_slug",
        fake_get_project_member_for_user_by_slug,
    )
    monkeypatch.setattr(
        document_upload_service.document_repository,
        "list_documents_for_project",
        fake_list_documents_for_project,
    )
    monkeypatch.setattr(document_upload_service, "receive_upload_stream", fake_receive_upload_stream)
    monkeypatch.setattr(document_upload_service, "run_technical_preflight", fake_run_technical_preflight)
    monkeypatch.setattr(document_upload_service.logger, "warning", fake_warning)

    with pytest.raises(RuntimeError, match="promote failed"):
        run_async(
            document_upload_service.upload_document_for_project(
                session,
                user_id=user.id,
                project_slug=project.slug,
                payload=DocumentUploadSubmit(upload_intent="new_document", title="文档"),
                chunks=iter([b"data"]),
                original_filename="upload.pdf",
                declared_mime="application/pdf",
                storage=storage,
            )
        )

    assert logged_messages == ["Failed to delete temporary upload file after promotion failure."]


def test_document_id_outside_project_is_rejected(monkeypatch) -> None:
    session = FakeSession()
    user = build_user()
    project = build_project(owner_id=user.id)
    membership = build_membership(project, user, ProjectMemberRole.OWNER)
    storage = FakeStorage()

    async def fake_get_project_member_for_user_by_slug(_session, _user_id, _slug):
        return membership

    async def fake_list_documents_for_project(_session, _project_id):
        return []

    async def fake_receive_upload_stream(*_args, **_kwargs):
        return build_received_upload()

    async def fake_run_technical_preflight(*_args, **_kwargs):
        return build_preflight_result(outcome=TechnicalPreflightOutcome.PASS)

    async def fake_get_document_for_update(_session, _project_id, _document_id):
        return None

    monkeypatch.setattr(
        document_upload_service.project_repository,
        "get_project_member_for_user_by_slug",
        fake_get_project_member_for_user_by_slug,
    )
    monkeypatch.setattr(
        document_upload_service.document_repository,
        "list_documents_for_project",
        fake_list_documents_for_project,
    )
    monkeypatch.setattr(document_upload_service, "receive_upload_stream", fake_receive_upload_stream)
    monkeypatch.setattr(document_upload_service, "run_technical_preflight", fake_run_technical_preflight)
    monkeypatch.setattr(
        document_upload_service.document_repository,
        "get_document_for_update",
        fake_get_document_for_update,
    )

    with pytest.raises(document_upload_service.UploadFormError) as exc_info:
        run_async(
            document_upload_service.upload_document_for_project(
                session,
                user_id=user.id,
                project_slug=project.slug,
                payload=DocumentUploadSubmit(upload_intent="revision", document_id=str(uuid.uuid4())),
                chunks=iter([b"data"]),
                original_filename="upload.pdf",
                declared_mime="application/pdf",
                storage=storage,
            )
        )

    assert exc_info.value.field == "document_id"


def test_report_preflight_timestamps_come_from_preflight_window(monkeypatch) -> None:
    session = FakeSession()
    user = build_user()
    project = build_project(owner_id=user.id)
    membership = build_membership(project, user, ProjectMemberRole.OWNER)
    storage = FakeStorage()
    created: dict[str, object] = {}

    class FakeDateTime:
        values = iter(
            [
                datetime(2026, 7, 30, 10, 0, 0, tzinfo=timezone.utc),
                datetime(2026, 7, 30, 10, 0, 3, tzinfo=timezone.utc),
            ]
        )

        @classmethod
        def now(cls, tz=None):
            value = next(cls.values)
            assert tz == timezone.utc
            return value

    async def fake_get_project_member_for_user_by_slug(_session, _user_id, _slug):
        return membership

    async def fake_list_documents_for_project(_session, _project_id):
        return []

    async def fake_receive_upload_stream(*_args, **_kwargs):
        return build_received_upload()

    async def fake_run_technical_preflight(*_args, **_kwargs):
        return build_preflight_result(outcome=TechnicalPreflightOutcome.PASS)

    async def fake_create_document(_session, document: Document):
        document.id = uuid.uuid4()
        return document

    async def fake_create_document_revision(_session, revision: DocumentRevision):
        revision.id = uuid.uuid4()
        return revision

    async def fake_create_validation_report(_session, report: IngestionValidationReport):
        created["report"] = report
        report.id = uuid.uuid4()
        return report

    monkeypatch.setattr(
        document_upload_service.project_repository,
        "get_project_member_for_user_by_slug",
        fake_get_project_member_for_user_by_slug,
    )
    monkeypatch.setattr(
        document_upload_service.document_repository,
        "list_documents_for_project",
        fake_list_documents_for_project,
    )
    monkeypatch.setattr(document_upload_service, "receive_upload_stream", fake_receive_upload_stream)
    monkeypatch.setattr(document_upload_service, "run_technical_preflight", fake_run_technical_preflight)
    monkeypatch.setattr(document_upload_service.document_repository, "create_document", fake_create_document)
    monkeypatch.setattr(
        document_upload_service.document_repository,
        "create_document_revision",
        fake_create_document_revision,
    )
    monkeypatch.setattr(
        document_upload_service.validation_repository,
        "create_validation_report",
        fake_create_validation_report,
    )
    monkeypatch.setattr(document_upload_service, "datetime", FakeDateTime)

    run_async(
        document_upload_service.upload_document_for_project(
            session,
            user_id=user.id,
            project_slug=project.slug,
            payload=DocumentUploadSubmit(upload_intent="new_document", title="会议纪要"),
            chunks=iter([b"data"]),
            original_filename="upload.pdf",
            declared_mime="application/pdf",
            storage=storage,
        )
    )

    report = created["report"]
    assert report.started_at == datetime(2026, 7, 30, 10, 0, 0, tzinfo=timezone.utc)
    assert report.completed_at == datetime(2026, 7, 30, 10, 0, 3, tzinfo=timezone.utc)
    assert report.started_at <= report.completed_at


@pytest.mark.parametrize("role", [ProjectMemberRole.OWNER, ProjectMemberRole.EDITOR])
def test_owner_and_editor_can_open_upload_page_and_submit(monkeypatch, role: ProjectMemberRole) -> None:
    user = build_user()
    project = build_project(owner_id=user.id)
    membership = build_membership(project, user, role)
    document = build_document(project_id=project.id, created_by_id=user.id)

    async def fake_get_project_upload_context(_session, _user_id, _slug):
        return document_upload_service.ProjectUploadContext(
            project=project,
            membership=membership,
            documents=[document],
        )

    async def fake_upload_document_for_project(
        _session,
        *,
        user_id,
        project_slug,
        payload,
        chunks,
        original_filename,
        declared_mime,
        storage=None,
    ):
        chunk_sizes = []
        async for chunk in chunks:
            chunk_sizes.append(len(chunk))
        assert user_id == user.id
        assert project_slug == project.slug
        assert payload.upload_intent == UploadIntent.NEW_DOCUMENT
        assert original_filename == "notes.pdf"
        assert declared_mime == "application/pdf"
        assert all(size > 0 for size in chunk_sizes)
        revision = build_revision(document_id=document.id, uploaded_by_id=user.id)
        report = IngestionValidationReport(
            id=uuid.uuid4(),
            revision_id=revision.id,
            attempt_no=1,
            status=ValidationReportStatus.COMPLETED.value,
            outcome=ValidationReportOutcome.PASS.value,
            file_checks={"detected_mime": "application/pdf"},
            text_checks={},
            classification_details={},
            character_count=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        return document_upload_service.UploadTransactionResult(
            project=project,
            document=document,
            revision=revision,
            report=report,
            risk_flags=[],
        )

    async def fake_handle_post_upload_processing(**kwargs):
        return None

    monkeypatch.setattr(document_upload_service, "get_project_upload_context", fake_get_project_upload_context)
    monkeypatch.setattr(document_upload_service, "upload_document_for_project", fake_upload_document_for_project)
    monkeypatch.setattr(projects_router, "handle_post_upload_processing", fake_handle_post_upload_processing)

    with TestClient(app) as client:
        login_client(client, monkeypatch, user)
        upload_page = client.get(f"/projects/{project.slug}/documents/upload")
        csrf_token = extract_csrf_token(upload_page.text)
        response = client.post(
            f"/projects/{project.slug}/documents/upload",
            data={"upload_intent": "new_document", "title": "新文档", "csrf_token": csrf_token},
            files={"file": ("notes.pdf", io.BytesIO(b"%PDF-demo"), "application/pdf")},
            follow_redirects=False,
        )

    assert upload_page.status_code == 200
    assert response.status_code == 303
    assert f"/projects/{project.slug}/documents/{document.id}/revisions/" in response.headers["location"]


def test_viewer_is_forbidden_for_upload_page(monkeypatch) -> None:
    user = build_user()
    project = build_project(owner_id=user.id)
    membership = build_membership(project, user, ProjectMemberRole.VIEWER)

    async def fake_get_project_upload_context(_session, _user_id, _slug):
        return document_upload_service.ProjectUploadContext(
            project=project,
            membership=membership,
            documents=[],
        )

    monkeypatch.setattr(document_upload_service, "get_project_upload_context", fake_get_project_upload_context)

    with TestClient(app) as client:
        login_client(client, monkeypatch, user)
        response = client.get(f"/projects/{project.slug}/documents/upload")

    assert response.status_code == 403


def test_upload_post_rejects_invalid_csrf(monkeypatch) -> None:
    user = build_user()
    project = build_project(owner_id=user.id)
    membership = build_membership(project, user, ProjectMemberRole.OWNER)

    async def fake_get_project_upload_context(_session, _user_id, _slug):
        return document_upload_service.ProjectUploadContext(
            project=project,
            membership=membership,
            documents=[],
        )

    monkeypatch.setattr(document_upload_service, "get_project_upload_context", fake_get_project_upload_context)

    with TestClient(app) as client:
        login_client(client, monkeypatch, user)
        client.get(f"/projects/{project.slug}/documents/upload")
        response = client.post(
            f"/projects/{project.slug}/documents/upload",
            data={"upload_intent": "new_document", "title": "新文档", "csrf_token": "bad-token"},
            files={"file": ("notes.pdf", io.BytesIO(b"%PDF-demo"), "application/pdf")},
        )

    assert response.status_code == 403


def test_non_project_document_id_is_shown_as_form_error(monkeypatch) -> None:
    user = build_user()
    project = build_project(owner_id=user.id)
    membership = build_membership(project, user, ProjectMemberRole.OWNER)
    document = build_document(project_id=project.id, created_by_id=user.id)

    async def fake_get_project_upload_context(_session, _user_id, _slug):
        return document_upload_service.ProjectUploadContext(
            project=project,
            membership=membership,
            documents=[document],
        )

    async def fake_upload_document_for_project(*_args, **_kwargs):
        raise document_upload_service.UploadFormError("document_id", "目标文档不存在或不属于当前项目。")

    monkeypatch.setattr(document_upload_service, "get_project_upload_context", fake_get_project_upload_context)
    monkeypatch.setattr(document_upload_service, "upload_document_for_project", fake_upload_document_for_project)

    with TestClient(app) as client:
        login_client(client, monkeypatch, user)
        upload_page = client.get(f"/projects/{project.slug}/documents/upload")
        csrf_token = extract_csrf_token(upload_page.text)
        response = client.post(
            f"/projects/{project.slug}/documents/upload",
            data={
                "upload_intent": "revision",
                "document_id": str(uuid.uuid4()),
                "csrf_token": csrf_token,
            },
            files={"file": ("notes.pdf", io.BytesIO(b"%PDF-demo"), "application/pdf")},
        )

    assert response.status_code == 400
    assert "目标文档不存在或不属于当前项目。" in response.text


def test_project_detail_shows_upload_button_and_documents(monkeypatch) -> None:
    user = build_user()
    project = build_project(owner_id=user.id)
    membership = build_membership(project, user, ProjectMemberRole.OWNER)
    document = build_document(project_id=project.id, created_by_id=user.id, title="会议纪要")
    revision = build_revision(document_id=document.id, uploaded_by_id=user.id)

    async def fake_get_project_workspace_context(_session, _user_id, _slug):
        context = document_upload_service.ProjectUploadContext(
            project=project,
            membership=membership,
            documents=[document],
        )
        summaries = [
            document_upload_service.ProjectDocumentSummary(
                document=document,
                latest_revision=revision,
            )
        ]
        return context, summaries

    monkeypatch.setattr(document_upload_service, "get_project_workspace_context", fake_get_project_workspace_context)

    with TestClient(app) as client:
        login_client(client, monkeypatch, user)
        response = client.get(f"/projects/{project.slug}")

    assert response.status_code == 200
    assert "上传文档" in response.text
    assert "会议纪要" in response.text


def test_revision_detail_page_does_not_leak_storage_key(monkeypatch) -> None:
    user = build_user()
    project = build_project(owner_id=user.id)
    membership = build_membership(project, user, ProjectMemberRole.OWNER)
    document = build_document(project_id=project.id, created_by_id=user.id)
    revision = build_revision(
        document_id=document.id,
        uploaded_by_id=user.id,
        storage_key="files/secret/internal.bin",
    )
    revision.document = document
    revision.uploaded_by = user
    report = IngestionValidationReport(
        id=uuid.uuid4(),
        revision_id=revision.id,
        attempt_no=1,
        status=ValidationReportStatus.COMPLETED.value,
        outcome=ValidationReportOutcome.WARNING.value,
        file_checks={"detected_mime": "application/pdf"},
        text_checks={},
        classification_details={},
        character_count=0,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    risk_flag = IngestionRiskFlag(
        id=uuid.uuid4(),
        report_id=report.id,
        category="file",
        kind="declared_mime_mismatch",
        severity="warning",
        detector="deterministic",
        message="MIME 不一致",
        confidence=None,
        location=None,
        details={},
        evidence_excerpt=None,
        created_at=datetime.now(timezone.utc),
    )

    async def fake_get_revision_detail_for_user(_session, _user_id, _slug, _document_id, _revision_id):
        return document_upload_service.RevisionDetailContext(
            project=project,
            membership=membership,
            revision=revision,
            report=report,
            risk_flags=[risk_flag],
            latest_processing_job=None,
            latest_extraction_run=None,
        )

    monkeypatch.setattr(document_upload_service, "get_revision_detail_for_user", fake_get_revision_detail_for_user)

    with TestClient(app) as client:
        login_client(client, monkeypatch, user)
        response = client.get(f"/projects/{project.slug}/documents/{document.id}/revisions/{revision.id}")

    assert response.status_code == 200
    assert "files/secret/internal.bin" not in response.text
    assert "storage_key" not in response.text


def test_loader_uses_model_attribute_and_no_string_relationship_loader() -> None:
    repository_path = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "repositories"
        / "document.py"
    )
    content = repository_path.read_text(encoding="utf-8")

    assert 'selectinload("' not in content
    assert "selectinload(DocumentRevision.validation_reports).selectinload(" in content
    assert "IngestionValidationReport.risk_flags" in content


def test_revision_detail_query_executes_with_mock_session() -> None:
    revision = build_revision(document_id=uuid.uuid4(), uploaded_by_id=uuid.uuid4())
    captured = {"statement": None, "executed": False}

    class FakeResult:
        def scalar_one_or_none(self):
            return revision

    class FakeQuerySession:
        async def execute(self, statement):
            captured["executed"] = True
            captured["statement"] = statement
            return FakeResult()

    result = run_async(
        document_repository.get_revision_detail_for_project(
            FakeQuerySession(),
            uuid.uuid4(),
            uuid.uuid4(),
            uuid.uuid4(),
        )
    )

    assert captured["executed"] is True
    assert result is revision
    assert captured["statement"] is not None


def test_upload_router_always_closes_upload_file(monkeypatch) -> None:
    closed_states: list[bool] = []
    user = build_user()
    session = object()

    class TrackingUploadFile:
        def __init__(self, filename: str, content_type: str, chunks: list[bytes]) -> None:
            self.filename = filename
            self.content_type = content_type
            self._chunks = list(chunks)
            self.closed = False

        async def read(self, size: int = -1) -> bytes:
            if self._chunks:
                return self._chunks.pop(0)
            return b""

        async def close(self) -> None:
            self.closed = True
            closed_states.append(self.closed)

    async def fake_render_upload_page(*_args, **_kwargs):
        return HTMLResponse("rendered", status_code=400)

    def fake_verify_csrf_token(_request, _csrf_token):
        return None

    class FakeSettings:
        upload_chunk_bytes = 4

    async def fake_upload_success(*_args, **_kwargs):
        return document_upload_service.UploadTransactionResult(
            project=build_project(owner_id=user.id),
            document=build_document(project_id=uuid.uuid4(), created_by_id=user.id),
            revision=build_revision(document_id=uuid.uuid4(), uploaded_by_id=user.id),
            report=IngestionValidationReport(
                id=uuid.uuid4(),
                revision_id=uuid.uuid4(),
                attempt_no=1,
                status=ValidationReportStatus.COMPLETED.value,
                outcome=ValidationReportOutcome.PASS.value,
                file_checks={},
                text_checks={},
                classification_details={},
                character_count=0,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            ),
            risk_flags=[],
        )

    async def fake_handle_post_upload_processing(**kwargs):
        return None

    async def fake_upload_form_error(*_args, **_kwargs):
        raise document_upload_service.UploadFormError("file", "bad file")

    monkeypatch.setattr(projects_router, "render_upload_page", fake_render_upload_page)
    monkeypatch.setattr(projects_router, "verify_csrf_token", fake_verify_csrf_token)
    monkeypatch.setattr(projects_router, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(document_upload_service, "upload_document_for_project", fake_upload_success)
    monkeypatch.setattr(projects_router, "handle_post_upload_processing", fake_handle_post_upload_processing)

    success_file = TrackingUploadFile("notes.pdf", "application/pdf", [b"aa", b"bb"])
    success_response = run_async(
        projects_router.document_upload_submit(
            request=SimpleNamespace(session={}),
            background_tasks=BackgroundTasks(),
            slug="project-a",
            upload_intent="new_document",
            title="新文档",
            description="",
            logical_order_kind="",
            logical_order_value="",
            logical_order_index="",
            effective_from="",
            effective_to="",
            document_id="",
            file=success_file,
            csrf_token="token",
            current_user=user,
            session=session,
        )
    )

    monkeypatch.setattr(document_upload_service, "upload_document_for_project", fake_upload_form_error)
    error_file = TrackingUploadFile("notes.pdf", "application/pdf", [b"aa"])
    error_response = run_async(
        projects_router.document_upload_submit(
            request=SimpleNamespace(session={}),
            background_tasks=BackgroundTasks(),
            slug="project-a",
            upload_intent="new_document",
            title="新文档",
            description="",
            logical_order_kind="",
            logical_order_value="",
            logical_order_index="",
            effective_from="",
            effective_to="",
            document_id="",
            file=error_file,
            csrf_token="token",
            current_user=user,
            session=session,
        )
    )

    empty_file = TrackingUploadFile("", "application/pdf", [])
    empty_response = run_async(
        projects_router.document_upload_submit(
            request=SimpleNamespace(session={}),
            background_tasks=BackgroundTasks(),
            slug="project-a",
            upload_intent="new_document",
            title="新文档",
            description="",
            logical_order_kind="",
            logical_order_value="",
            logical_order_index="",
            effective_from="",
            effective_to="",
            document_id="",
            file=empty_file,
            csrf_token="token",
            current_user=user,
            session=session,
        )
    )

    assert success_response.status_code == 303
    assert error_response.status_code == 400
    assert empty_response.status_code == 400
    assert success_file.closed is True
    assert error_file.closed is True
    assert empty_file.closed is True
    assert len(closed_states) == 3


def test_pass_upload_auto_accepts_enqueues_and_registers_background_task(monkeypatch) -> None:
    user = build_user()
    project = build_project(owner_id=user.id)
    document = build_document(project_id=project.id, created_by_id=user.id)
    revision = build_revision(
        document_id=document.id,
        uploaded_by_id=user.id,
        status=DocumentRevisionStatus.PREFLIGHT_CHECKING.value,
    )
    result = document_upload_service.UploadTransactionResult(
        project=project,
        document=document,
        revision=revision,
        report=IngestionValidationReport(
            id=uuid.uuid4(),
            revision_id=revision.id,
            attempt_no=1,
            status=ValidationReportStatus.COMPLETED.value,
            outcome=ValidationReportOutcome.PASS.value,
            recommended_authority=ValidationRecommendedAuthority.FORMAL.value,
            file_checks={"detected_mime": "application/pdf"},
            text_checks={},
            classification_details={},
            character_count=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ),
        risk_flags=[],
    )
    queued_job = build_processing_job(project_id=project.id, revision_id=revision.id)
    background_tasks = FakeBackgroundTasks()
    storage = FakeStorage()
    session_factory_marker = object()
    recorded: dict[str, object] = {}

    async def fake_upload_document_for_project(*args, **kwargs):
        assert kwargs["storage"] is storage
        return result

    async def fake_resolve_revision_admission(_session, *, project_id, revision_id):
        recorded["resolved"] = (project_id, revision_id)
        return revision_admission_service.RevisionAdmissionResult(
            revision_id=revision_id,
            report_id=result.report.id,
            revision_status=DocumentRevisionStatus.ACCEPTED.value,
            source_authority=SourceAuthority.FORMAL.value,
            user_decision=None,
            decision_required=False,
        )

    async def fake_enqueue_revision_extraction_job(_session, *, project_id, revision_id, trigger_kind, actor_id=None):
        recorded["enqueued"] = (project_id, revision_id, trigger_kind, actor_id)
        return queued_job

    monkeypatch.setattr(document_upload_service, "upload_document_for_project", fake_upload_document_for_project)
    monkeypatch.setattr(revision_admission_service, "resolve_revision_admission", fake_resolve_revision_admission)
    monkeypatch.setattr(
        processing_job_service,
        "enqueue_revision_extraction_job",
        fake_enqueue_revision_extraction_job,
    )
    monkeypatch.setattr(projects_router.file_ingestion_service, "get_local_file_storage", lambda: storage)
    monkeypatch.setattr(projects_router, "get_async_session_factory", lambda: session_factory_marker)
    monkeypatch.setattr(projects_router, "verify_csrf_token", lambda request, csrf_token: None)
    monkeypatch.setattr(projects_router, "get_settings", lambda: SimpleNamespace(upload_chunk_bytes=4))

    request_session = object()
    response = run_async(
        projects_router.document_upload_submit(
            request=SimpleNamespace(session={}),
            background_tasks=background_tasks,
            slug=project.slug,
            upload_intent="new_document",
            title="新文档",
            description="",
            logical_order_kind="",
            logical_order_value="",
            logical_order_index="",
            effective_from="",
            effective_to="",
            document_id="",
            file=AsyncUploadFileStub(),
            csrf_token="token",
            current_user=user,
            session=request_session,
        )
    )

    assert response.status_code == 303
    assert recorded["resolved"] == (project.id, revision.id)
    assert recorded["enqueued"] == (
        project.id,
        revision.id,
        ProcessingTriggerKind.AUTOMATIC,
        None,
    )
    assert len(background_tasks.calls) == 1
    func, args, kwargs = background_tasks.calls[0]
    assert func is projects_router.dispatch_revision_extraction_job
    assert kwargs["job_id"] == queued_job.id
    assert kwargs["session_factory"] is session_factory_marker
    assert kwargs["storage"] is storage
    assert kwargs["session_factory"] is not request_session


def test_warning_goes_to_needs_confirmation_without_queue(monkeypatch) -> None:
    background_tasks = FakeBackgroundTasks()
    result = document_upload_service.UploadTransactionResult(
        project=build_project(owner_id=uuid.uuid4()),
        document=build_document(project_id=uuid.uuid4(), created_by_id=uuid.uuid4()),
        revision=build_revision(document_id=uuid.uuid4(), uploaded_by_id=uuid.uuid4()),
        report=IngestionValidationReport(
            id=uuid.uuid4(),
            revision_id=uuid.uuid4(),
            attempt_no=1,
            status=ValidationReportStatus.COMPLETED.value,
            outcome=ValidationReportOutcome.WARNING.value,
            file_checks={},
            text_checks={},
            classification_details={},
            character_count=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ),
        risk_flags=[],
    )

    async def fake_resolve_revision_admission(_session, *, project_id, revision_id):
        return revision_admission_service.RevisionAdmissionResult(
            revision_id=revision_id,
            report_id=result.report.id,
            revision_status=DocumentRevisionStatus.NEEDS_CONFIRMATION.value,
            source_authority=SourceAuthority.PENDING.value,
            user_decision=None,
            decision_required=True,
        )

    async def fake_enqueue_revision_extraction_job(*args, **kwargs):
        raise AssertionError("warning revisions must not enqueue background jobs")

    monkeypatch.setattr(revision_admission_service, "resolve_revision_admission", fake_resolve_revision_admission)
    monkeypatch.setattr(
        processing_job_service,
        "enqueue_revision_extraction_job",
        fake_enqueue_revision_extraction_job,
    )

    notice = run_async(
        projects_router.handle_post_upload_processing(
            session=object(),
            background_tasks=background_tasks,  # type: ignore[arg-type]
            schedule_job_ids=set(),
            project_id=result.project.id,
            revision_id=result.revision.id,
            current_revision_status=DocumentRevisionStatus.PREFLIGHT_CHECKING.value,
            storage=object(),
        )
    )

    assert notice is None
    assert background_tasks.calls == []


@pytest.mark.parametrize(
    "status",
    [
        DocumentRevisionStatus.REJECTED.value,
        DocumentRevisionStatus.QUARANTINED.value,
    ],
)
def test_reject_and_quarantine_do_not_queue(monkeypatch, status: str) -> None:
    background_tasks = FakeBackgroundTasks()

    async def fake_resolve_revision_admission(*args, **kwargs):
        raise AssertionError("terminal upload results must not re-enter resolve_revision_admission")

    monkeypatch.setattr(revision_admission_service, "resolve_revision_admission", fake_resolve_revision_admission)

    notice = run_async(
        projects_router.handle_post_upload_processing(
            session=object(),
            background_tasks=background_tasks,  # type: ignore[arg-type]
            schedule_job_ids=set(),
            project_id=uuid.uuid4(),
            revision_id=uuid.uuid4(),
            current_revision_status=status,
            storage=object(),
        )
    )

    assert notice is None
    assert background_tasks.calls == []


def test_schedule_processing_job_once_deduplicates_same_job_id() -> None:
    background_tasks = FakeBackgroundTasks()
    job_id = uuid.uuid4()
    schedule_job_ids: set[uuid.UUID] = set()

    first = projects_router.schedule_processing_job_once(
        background_tasks=background_tasks,  # type: ignore[arg-type]
        schedule_job_ids=schedule_job_ids,
        job_id=job_id,
        session_factory=object(),
        storage=object(),
    )
    second = projects_router.schedule_processing_job_once(
        background_tasks=background_tasks,  # type: ignore[arg-type]
        schedule_job_ids=schedule_job_ids,
        job_id=job_id,
        session_factory=object(),
        storage=object(),
    )

    assert first is True
    assert second is False
    assert len(background_tasks.calls) == 1


def test_enqueue_failure_still_redirects_and_does_not_delete_uploaded_file(monkeypatch) -> None:
    user = build_user()
    project = build_project(owner_id=user.id)
    document = build_document(project_id=project.id, created_by_id=user.id)
    revision = build_revision(
        document_id=document.id,
        uploaded_by_id=user.id,
        status=DocumentRevisionStatus.PREFLIGHT_CHECKING.value,
    )
    result = document_upload_service.UploadTransactionResult(
        project=project,
        document=document,
        revision=revision,
        report=IngestionValidationReport(
            id=uuid.uuid4(),
            revision_id=revision.id,
            attempt_no=1,
            status=ValidationReportStatus.COMPLETED.value,
            outcome=ValidationReportOutcome.PASS.value,
            file_checks={},
            text_checks={},
            classification_details={},
            character_count=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ),
        risk_flags=[],
    )
    storage = FakeStorage()

    async def fake_upload_document_for_project(*args, **kwargs):
        return result

    async def fake_resolve_revision_admission(_session, *, project_id, revision_id):
        return revision_admission_service.RevisionAdmissionResult(
            revision_id=revision_id,
            report_id=result.report.id,
            revision_status=DocumentRevisionStatus.ACCEPTED.value,
            source_authority=SourceAuthority.PENDING.value,
            user_decision=None,
            decision_required=False,
        )

    async def fake_enqueue_revision_extraction_job(*args, **kwargs):
        raise processing_job_service.ProcessingJobStateError("storage_key=hidden")

    class TrackingUploadFile:
        filename = "notes.pdf"
        content_type = "application/pdf"

        async def read(self, size: int = -1) -> bytes:
            return b""

        async def close(self) -> None:
            return None

    monkeypatch.setattr(document_upload_service, "upload_document_for_project", fake_upload_document_for_project)
    monkeypatch.setattr(revision_admission_service, "resolve_revision_admission", fake_resolve_revision_admission)
    monkeypatch.setattr(
        processing_job_service,
        "enqueue_revision_extraction_job",
        fake_enqueue_revision_extraction_job,
    )
    monkeypatch.setattr(projects_router.file_ingestion_service, "get_local_file_storage", lambda: storage)
    monkeypatch.setattr(projects_router, "verify_csrf_token", lambda request, csrf_token: None)
    monkeypatch.setattr(projects_router, "get_settings", lambda: SimpleNamespace(upload_chunk_bytes=4))

    response = run_async(
        projects_router.document_upload_submit(
            request=SimpleNamespace(session={}),
            background_tasks=FakeBackgroundTasks(),
            slug=project.slug,
            upload_intent="new_document",
            title="新文档",
            description="",
            logical_order_kind="",
            logical_order_value="",
            logical_order_index="",
            effective_from="",
            effective_to="",
            document_id="",
            file=TrackingUploadFile(),
            csrf_token="token",
            current_user=user,
            session=object(),
        )
    )

    assert response.status_code == 303
    assert response.headers["location"].endswith("?processing_notice=enqueue_failed")
    assert storage.deleted_keys == []
    assert storage.deleted_temp_keys == []


def test_existing_job_is_reused_and_registered_once(monkeypatch) -> None:
    background_tasks = FakeBackgroundTasks()
    job = build_processing_job(project_id=uuid.uuid4(), revision_id=uuid.uuid4())
    enqueue_calls = 0

    async def fake_resolve_revision_admission(_session, *, project_id, revision_id):
        return revision_admission_service.RevisionAdmissionResult(
            revision_id=revision_id,
            report_id=uuid.uuid4(),
            revision_status=DocumentRevisionStatus.ACCEPTED.value,
            source_authority=SourceAuthority.PENDING.value,
            user_decision=None,
            decision_required=False,
        )

    async def fake_enqueue_revision_extraction_job(_session, *, project_id, revision_id, trigger_kind, actor_id=None):
        nonlocal enqueue_calls
        enqueue_calls += 1
        return job

    monkeypatch.setattr(revision_admission_service, "resolve_revision_admission", fake_resolve_revision_admission)
    monkeypatch.setattr(
        processing_job_service,
        "enqueue_revision_extraction_job",
        fake_enqueue_revision_extraction_job,
    )

    notice = run_async(
        projects_router.handle_post_upload_processing(
            session=object(),
            background_tasks=background_tasks,  # type: ignore[arg-type]
            schedule_job_ids=set(),
            project_id=job.project_id,
            revision_id=job.revision_id,
            current_revision_status=DocumentRevisionStatus.PREFLIGHT_CHECKING.value,
            storage=object(),
        )
    )

    assert notice is None
    assert enqueue_calls == 1
    assert len(background_tasks.calls) == 1


def test_job_created_but_background_add_task_failure_keeps_queued_job(monkeypatch) -> None:
    background_tasks = FakeBackgroundTasks(add_error=RuntimeError("lease_token=secret"))
    queued_job = build_processing_job(project_id=uuid.uuid4(), revision_id=uuid.uuid4())

    async def fake_resolve_revision_admission(_session, *, project_id, revision_id):
        return revision_admission_service.RevisionAdmissionResult(
            revision_id=revision_id,
            report_id=uuid.uuid4(),
            revision_status=DocumentRevisionStatus.ACCEPTED.value,
            source_authority=SourceAuthority.PENDING.value,
            user_decision=None,
            decision_required=False,
        )

    async def fake_enqueue_revision_extraction_job(_session, *, project_id, revision_id, trigger_kind, actor_id=None):
        return queued_job

    monkeypatch.setattr(revision_admission_service, "resolve_revision_admission", fake_resolve_revision_admission)
    monkeypatch.setattr(
        processing_job_service,
        "enqueue_revision_extraction_job",
        fake_enqueue_revision_extraction_job,
    )

    notice = run_async(
        projects_router.handle_post_upload_processing(
            session=object(),
            background_tasks=background_tasks,  # type: ignore[arg-type]
            schedule_job_ids=set(),
            project_id=queued_job.project_id,
            revision_id=queued_job.revision_id,
            current_revision_status=DocumentRevisionStatus.PREFLIGHT_CHECKING.value,
            storage=object(),
        )
    )

    assert notice == "enqueue_failed"
    assert queued_job.status == ProcessingJobStatus.QUEUED.value


@pytest.mark.parametrize(
    ("decision", "expected_authority"),
    [
        (ValidationUserDecision.CONTINUE_FORMAL.value, SourceAuthority.FORMAL.value),
        (ValidationUserDecision.CONTINUE_REFERENCE.value, SourceAuthority.REFERENCE.value),
    ],
)
def test_admission_continue_routes_enqueue_manual_job_and_schedule(
    monkeypatch,
    decision: str,
    expected_authority: str,
) -> None:
    user, project, document, revision, context = build_revision_detail_context(
        revision_status=DocumentRevisionStatus.NEEDS_CONFIRMATION.value,
        report_outcome=ValidationReportOutcome.WARNING.value,
    )
    background_tasks = FakeBackgroundTasks()
    queued_job = build_processing_job(project_id=project.id, revision_id=revision.id)
    storage = FakeStorage()
    recorded: dict[str, object] = {}

    async def fake_context_loader(**kwargs):
        return context

    async def fake_decide_revision_admission(_session, *, project_id, actor_id, revision_id, payload):
        recorded["decision"] = (project_id, actor_id, revision_id, payload.decision)
        return revision_admission_service.RevisionAdmissionResult(
            revision_id=revision_id,
            report_id=context.report.id,
            revision_status=DocumentRevisionStatus.ACCEPTED.value,
            source_authority=expected_authority,
            user_decision=payload.decision,
            decision_required=False,
        )

    async def fake_enqueue_revision_extraction_job(_session, *, project_id, revision_id, trigger_kind, actor_id=None):
        recorded["enqueue"] = (project_id, revision_id, trigger_kind, actor_id)
        return queued_job

    monkeypatch.setattr(projects_router, "_get_revision_detail_context_or_404", fake_context_loader)
    monkeypatch.setattr(projects_router, "verify_csrf_token", lambda request, csrf_token: None)
    monkeypatch.setattr(projects_router.file_ingestion_service, "get_local_file_storage", lambda: storage)
    monkeypatch.setattr(
        revision_admission_service,
        "decide_revision_admission",
        fake_decide_revision_admission,
    )
    monkeypatch.setattr(
        processing_job_service,
        "enqueue_revision_extraction_job",
        fake_enqueue_revision_extraction_job,
    )

    response = run_async(
        projects_router.revision_admission_submit(
            request=SimpleNamespace(session={}),
            background_tasks=background_tasks,
            slug=project.slug,
            document_id=document.id,
            revision_id=revision.id,
            decision=decision,
            csrf_token="token",
            current_user=user,
            session=object(),
        )
    )

    assert response.status_code == 303
    assert recorded["decision"] == (
        project.id,
        user.id,
        revision.id,
        decision,
    )
    assert recorded["enqueue"] == (
        project.id,
        revision.id,
        ProcessingTriggerKind.MANUAL,
        user.id,
    )
    assert len(background_tasks.calls) == 1
    _func, _args, kwargs = background_tasks.calls[0]
    assert kwargs["job_id"] == queued_job.id
    assert kwargs["storage"] is storage


def test_admission_cancel_does_not_enqueue_job(monkeypatch) -> None:
    user, project, document, revision, context = build_revision_detail_context(
        revision_status=DocumentRevisionStatus.NEEDS_CONFIRMATION.value,
        report_outcome=ValidationReportOutcome.WARNING.value,
    )

    async def fake_context_loader(**kwargs):
        return context

    async def fake_decide_revision_admission(_session, *, project_id, actor_id, revision_id, payload):
        return revision_admission_service.RevisionAdmissionResult(
            revision_id=revision_id,
            report_id=context.report.id,
            revision_status=DocumentRevisionStatus.REJECTED.value,
            source_authority=SourceAuthority.PENDING.value,
            user_decision=ValidationUserDecision.CANCEL,
            decision_required=False,
        )

    async def fake_enqueue_revision_extraction_job(*args, **kwargs):
        raise AssertionError("cancel must not enqueue processing jobs")

    monkeypatch.setattr(projects_router, "_get_revision_detail_context_or_404", fake_context_loader)
    monkeypatch.setattr(projects_router, "verify_csrf_token", lambda request, csrf_token: None)
    monkeypatch.setattr(
        revision_admission_service,
        "decide_revision_admission",
        fake_decide_revision_admission,
    )
    monkeypatch.setattr(
        processing_job_service,
        "enqueue_revision_extraction_job",
        fake_enqueue_revision_extraction_job,
    )

    response = run_async(
        projects_router.revision_admission_submit(
            request=SimpleNamespace(session={}),
            background_tasks=FakeBackgroundTasks(),
            slug=project.slug,
            document_id=document.id,
            revision_id=revision.id,
            decision=ValidationUserDecision.CANCEL.value,
            csrf_token="token",
            current_user=user,
            session=object(),
        )
    )

    assert response.status_code == 303
    assert "processing_notice" not in response.headers["location"]


def test_admission_decision_persists_when_enqueue_fails(monkeypatch) -> None:
    user, project, document, revision, context = build_revision_detail_context(
        revision_status=DocumentRevisionStatus.NEEDS_CONFIRMATION.value,
        report_outcome=ValidationReportOutcome.WARNING.value,
    )
    recorded: dict[str, object] = {}

    async def fake_context_loader(**kwargs):
        return context

    async def fake_decide_revision_admission(_session, *, project_id, actor_id, revision_id, payload):
        recorded["decision_saved"] = payload.decision
        return revision_admission_service.RevisionAdmissionResult(
            revision_id=revision_id,
            report_id=context.report.id,
            revision_status=DocumentRevisionStatus.ACCEPTED.value,
            source_authority=SourceAuthority.FORMAL.value,
            user_decision=payload.decision,
            decision_required=False,
        )

    async def fake_enqueue_revision_extraction_job(*args, **kwargs):
        raise processing_job_service.ProcessingJobStateError("lease_token=secret storage_key=hidden")

    monkeypatch.setattr(projects_router, "_get_revision_detail_context_or_404", fake_context_loader)
    monkeypatch.setattr(projects_router, "verify_csrf_token", lambda request, csrf_token: None)
    monkeypatch.setattr(
        revision_admission_service,
        "decide_revision_admission",
        fake_decide_revision_admission,
    )
    monkeypatch.setattr(
        processing_job_service,
        "enqueue_revision_extraction_job",
        fake_enqueue_revision_extraction_job,
    )

    response = run_async(
        projects_router.revision_admission_submit(
            request=SimpleNamespace(session={}),
            background_tasks=FakeBackgroundTasks(),
            slug=project.slug,
            document_id=document.id,
            revision_id=revision.id,
            decision=ValidationUserDecision.CONTINUE_FORMAL.value,
            csrf_token="token",
            current_user=user,
            session=object(),
        )
    )

    assert recorded["decision_saved"] == ValidationUserDecision.CONTINUE_FORMAL.value
    assert response.status_code == 303
    assert response.headers["location"].endswith("?processing_notice=enqueue_failed")


def test_processing_start_reuses_queued_job_and_uses_session_factory(monkeypatch) -> None:
    user, project, document, revision, context = build_revision_detail_context(
        revision_status=DocumentRevisionStatus.ACCEPTED.value,
    )
    queued_job = build_processing_job(project_id=project.id, revision_id=revision.id)
    background_tasks = FakeBackgroundTasks()
    session_factory_marker = object()
    request_session = object()

    async def fake_context_loader(**kwargs):
        return context

    async def fake_enqueue_revision_extraction_job(_session, *, project_id, revision_id, trigger_kind, actor_id=None):
        assert _session is request_session
        return queued_job

    monkeypatch.setattr(projects_router, "_get_revision_detail_context_or_404", fake_context_loader)
    monkeypatch.setattr(projects_router, "verify_csrf_token", lambda request, csrf_token: None)
    monkeypatch.setattr(projects_router, "get_async_session_factory", lambda: session_factory_marker)
    monkeypatch.setattr(
        processing_job_service,
        "enqueue_revision_extraction_job",
        fake_enqueue_revision_extraction_job,
    )

    response = run_async(
        projects_router.revision_processing_start(
            request=SimpleNamespace(session={}),
            background_tasks=background_tasks,
            slug=project.slug,
            document_id=document.id,
            revision_id=revision.id,
            csrf_token="token",
            current_user=user,
            session=request_session,
        )
    )

    assert response.status_code == 303
    assert response.headers["location"].endswith("?processing_notice=queued")
    assert len(background_tasks.calls) == 1
    _func, _args, kwargs = background_tasks.calls[0]
    assert kwargs["session_factory"] is session_factory_marker
    assert kwargs["session_factory"] is not request_session


def test_processing_start_does_not_reschedule_running_job(monkeypatch) -> None:
    user, project, document, revision, context = build_revision_detail_context(
        revision_status=DocumentRevisionStatus.ACCEPTED.value,
    )
    running_job = build_processing_job(project_id=project.id, revision_id=revision.id)
    running_job.status = ProcessingJobStatus.RUNNING.value
    running_job.started_at = datetime.now(timezone.utc)
    running_job.lease_token = uuid.uuid4()
    running_job.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

    async def fake_context_loader(**kwargs):
        return context

    async def fake_enqueue_revision_extraction_job(_session, *, project_id, revision_id, trigger_kind, actor_id=None):
        return running_job

    monkeypatch.setattr(projects_router, "_get_revision_detail_context_or_404", fake_context_loader)
    monkeypatch.setattr(projects_router, "verify_csrf_token", lambda request, csrf_token: None)
    monkeypatch.setattr(
        processing_job_service,
        "enqueue_revision_extraction_job",
        fake_enqueue_revision_extraction_job,
    )

    background_tasks = FakeBackgroundTasks()
    response = run_async(
        projects_router.revision_processing_start(
            request=SimpleNamespace(session={}),
            background_tasks=background_tasks,
            slug=project.slug,
            document_id=document.id,
            revision_id=revision.id,
            csrf_token="token",
            current_user=user,
            session=object(),
        )
    )

    assert response.status_code == 303
    assert background_tasks.calls == []


def test_processing_retry_queues_job_and_schedule_failure_keeps_queued(monkeypatch) -> None:
    user, project, document, revision, context = build_revision_detail_context(
        revision_status=DocumentRevisionStatus.FAILED.value,
    )
    queued_job = build_processing_job(project_id=project.id, revision_id=revision.id)

    async def fake_context_loader(**kwargs):
        return context

    async def fake_retry_failed_revision_extraction(_session, *, project_id, revision_id, actor_id):
        return queued_job

    monkeypatch.setattr(projects_router, "_get_revision_detail_context_or_404", fake_context_loader)
    monkeypatch.setattr(projects_router, "verify_csrf_token", lambda request, csrf_token: None)
    monkeypatch.setattr(
        processing_job_service,
        "retry_failed_revision_extraction",
        fake_retry_failed_revision_extraction,
    )

    response = run_async(
        projects_router.revision_processing_retry(
            request=SimpleNamespace(session={}),
            background_tasks=FakeBackgroundTasks(add_error=RuntimeError("lease_token=secret")),
            slug=project.slug,
            document_id=document.id,
            revision_id=revision.id,
            csrf_token="token",
            current_user=user,
            session=object(),
        )
    )

    assert response.status_code == 303
    assert response.headers["location"].endswith("?processing_notice=enqueue_failed")
    assert queued_job.status == ProcessingJobStatus.QUEUED.value


def test_processing_recover_uses_server_stale_minutes_only(monkeypatch) -> None:
    user, project, document, revision, context = build_revision_detail_context(
        revision_status=DocumentRevisionStatus.EXTRACTING.value,
    )
    queued_job = build_processing_job(project_id=project.id, revision_id=revision.id)
    recorded: dict[str, object] = {}

    async def fake_context_loader(**kwargs):
        return context

    async def fake_recover_stale_revision_extraction(
        _session,
        *,
        project_id,
        revision_id,
        actor_id,
        stale_before,
    ):
        recorded["stale_before"] = stale_before
        return queued_job

    monkeypatch.setattr(projects_router, "_get_revision_detail_context_or_404", fake_context_loader)
    monkeypatch.setattr(projects_router, "verify_csrf_token", lambda request, csrf_token: None)
    monkeypatch.setattr(projects_router, "get_settings", lambda: SimpleNamespace(processing_stale_minutes=45))
    monkeypatch.setattr(
        processing_job_service,
        "recover_stale_revision_extraction",
        fake_recover_stale_revision_extraction,
    )

    before = datetime.now(timezone.utc)
    response = run_async(
        projects_router.revision_processing_recover(
            request=SimpleNamespace(session={}),
            background_tasks=FakeBackgroundTasks(),
            slug=project.slug,
            document_id=document.id,
            revision_id=revision.id,
            csrf_token="token",
            current_user=user,
            session=object(),
        )
    )
    after = datetime.now(timezone.utc)

    assert response.status_code == 303
    assert response.headers["location"].endswith("?processing_notice=recovery_queued")
    expected_latest = after - timedelta(minutes=45)
    expected_earliest = before - timedelta(minutes=45)
    assert expected_earliest <= recorded["stale_before"] <= expected_latest


def test_processing_recover_unexpired_job_returns_409(monkeypatch) -> None:
    user, project, document, revision, context = build_revision_detail_context(
        revision_status=DocumentRevisionStatus.EXTRACTING.value,
    )

    async def fake_context_loader(**kwargs):
        return context

    async def fake_recover_stale_revision_extraction(*args, **kwargs):
        raise processing_job_service.ProcessingJobStateError("Running job lease has not expired.")

    monkeypatch.setattr(projects_router, "_get_revision_detail_context_or_404", fake_context_loader)
    monkeypatch.setattr(projects_router, "verify_csrf_token", lambda request, csrf_token: None)
    monkeypatch.setattr(
        processing_job_service,
        "recover_stale_revision_extraction",
        fake_recover_stale_revision_extraction,
    )

    with pytest.raises(HTTPException) as exc_info:
        run_async(
            projects_router.revision_processing_recover(
                request=SimpleNamespace(session={}),
                background_tasks=FakeBackgroundTasks(),
                slug=project.slug,
                document_id=document.id,
                revision_id=revision.id,
                csrf_token="token",
                current_user=user,
                session=object(),
            )
        )

    assert exc_info.value.status_code == 409


def test_processing_routes_return_404_on_path_mismatch(monkeypatch) -> None:
    user = build_user()

    async def fake_context_loader(**kwargs):
        raise HTTPException(status_code=404, detail="Revision not found.")

    monkeypatch.setattr(projects_router, "_get_revision_detail_context_or_404", fake_context_loader)
    monkeypatch.setattr(projects_router, "verify_csrf_token", lambda request, csrf_token: None)

    with pytest.raises(HTTPException) as exc_info:
        run_async(
            projects_router.revision_processing_start(
                request=SimpleNamespace(session={}),
                background_tasks=FakeBackgroundTasks(),
                slug="project-a",
                document_id=uuid.uuid4(),
                revision_id=uuid.uuid4(),
                csrf_token="token",
                current_user=user,
                session=object(),
            )
        )

    assert exc_info.value.status_code == 404


def test_revision_detail_page_hides_viewer_actions_and_uses_notice_whitelist(monkeypatch) -> None:
    latest_job = build_processing_job(project_id=uuid.uuid4(), revision_id=uuid.uuid4())
    latest_job.failure_message = "lease_token=secret storage_key=hidden"
    latest_run = build_extraction_run(revision_id=latest_job.revision_id)
    latest_run.failure_message = "raw exception body"
    user, project, document, revision, context = build_revision_detail_context(
        membership_role=ProjectMemberRole.VIEWER,
        revision_status=DocumentRevisionStatus.ACCEPTED.value,
        latest_processing_job=latest_job,
        latest_extraction_run=latest_run,
    )
    latest_job.project_id = project.id
    latest_job.revision_id = revision.id
    latest_run.revision_id = revision.id

    async def fake_get_revision_detail_for_user(_session, _user_id, _slug, _document_id, _revision_id):
        return context

    monkeypatch.setattr(document_upload_service, "get_revision_detail_for_user", fake_get_revision_detail_for_user)

    with TestClient(app) as client:
        login_client(client, monkeypatch, user)
        queued_response = client.get(
            f"/projects/{project.slug}/documents/{document.id}/revisions/{revision.id}?processing_notice=queued"
        )
        unknown_response = client.get(
            f"/projects/{project.slug}/documents/{document.id}/revisions/{revision.id}?processing_notice=lease_token"
        )

    assert queued_response.status_code == 200
    assert "处理任务已进入队列。" in queued_response.text
    assert "开始/继续处理" not in queued_response.text
    assert "作为正式资料继续" not in queued_response.text
    assert "重试提取" not in queued_response.text
    assert "恢复卡住任务" not in queued_response.text
    assert "lease_token=secret" not in queued_response.text
    assert "storage_key=hidden" not in queued_response.text
    assert "raw exception body" not in queued_response.text
    assert "lease_token" not in unknown_response.text


@pytest.mark.parametrize(
    ("revision_status", "latest_job_status", "latest_job_expired", "expected_button"),
    [
        (
            DocumentRevisionStatus.NEEDS_CONFIRMATION.value,
            None,
            False,
            "作为正式资料继续",
        ),
        (
            DocumentRevisionStatus.ACCEPTED.value,
            None,
            False,
            "开始/继续处理",
        ),
        (
            DocumentRevisionStatus.FAILED.value,
            None,
            False,
            "重试提取",
        ),
        (
            DocumentRevisionStatus.EXTRACTING.value,
            ProcessingJobStatus.RUNNING.value,
            True,
            "恢复卡住任务",
        ),
    ],
)
def test_revision_detail_page_shows_expected_processing_buttons(
    monkeypatch,
    revision_status: str,
    latest_job_status: str | None,
    latest_job_expired: bool,
    expected_button: str,
) -> None:
    latest_job = None
    if latest_job_status is not None:
        latest_job = build_processing_job(project_id=uuid.uuid4(), revision_id=uuid.uuid4())
        latest_job.status = latest_job_status
        latest_job.started_at = datetime.now(timezone.utc) - timedelta(minutes=40)
        latest_job.lease_token = uuid.uuid4()
        latest_job.lease_expires_at = (
            datetime.now(timezone.utc) - timedelta(minutes=5)
            if latest_job_expired
            else datetime.now(timezone.utc) + timedelta(minutes=5)
        )
    user, project, document, revision, context = build_revision_detail_context(
        revision_status=revision_status,
        latest_processing_job=latest_job,
        report_outcome=ValidationReportOutcome.WARNING.value
        if revision_status == DocumentRevisionStatus.NEEDS_CONFIRMATION.value
        else ValidationReportOutcome.PASS.value,
    )
    if revision_status == DocumentRevisionStatus.EXTRACTING.value:
        revision.updated_at = datetime.now(timezone.utc) - timedelta(minutes=40)
    if latest_job is not None:
        latest_job.project_id = project.id
        latest_job.revision_id = revision.id

    async def fake_get_revision_detail_for_user(_session, _user_id, _slug, _document_id, _revision_id):
        return context

    monkeypatch.setattr(document_upload_service, "get_revision_detail_for_user", fake_get_revision_detail_for_user)
    monkeypatch.setattr(projects_router, "get_settings", lambda: SimpleNamespace(processing_stale_minutes=30))

    with TestClient(app) as client:
        login_client(client, monkeypatch, user)
        response = client.get(f"/projects/{project.slug}/documents/{document.id}/revisions/{revision.id}")

    assert response.status_code == 200
    assert expected_button in response.text


def test_viewer_post_processing_returns_403(monkeypatch) -> None:
    user, project, document, revision, context = build_revision_detail_context(
        membership_role=ProjectMemberRole.VIEWER,
        revision_status=DocumentRevisionStatus.ACCEPTED.value,
    )

    async def fake_get_revision_detail_for_user(_session, _user_id, _slug, _document_id, _revision_id):
        return context

    monkeypatch.setattr(document_upload_service, "get_revision_detail_for_user", fake_get_revision_detail_for_user)
    monkeypatch.setattr(projects_router, "verify_csrf_token", lambda request, csrf_token: None)

    with pytest.raises(HTTPException) as exc_info:
        run_async(
            projects_router.revision_processing_start(
                request=SimpleNamespace(session={}),
                background_tasks=FakeBackgroundTasks(),
                slug=project.slug,
                document_id=document.id,
                revision_id=revision.id,
                csrf_token="token",
                current_user=user,
                session=object(),
            )
        )

    assert exc_info.value.status_code == 403


def test_processing_start_state_conflict_returns_409(monkeypatch) -> None:
    user, project, document, revision, context = build_revision_detail_context(
        revision_status=DocumentRevisionStatus.ACCEPTED.value,
    )

    async def fake_context_loader(**kwargs):
        return context

    async def fake_enqueue_revision_extraction_job(*args, **kwargs):
        raise processing_job_service.ProcessingJobStateError("Only accepted revisions can be queued for extraction.")

    monkeypatch.setattr(projects_router, "_get_revision_detail_context_or_404", fake_context_loader)
    monkeypatch.setattr(projects_router, "verify_csrf_token", lambda request, csrf_token: None)
    monkeypatch.setattr(
        processing_job_service,
        "enqueue_revision_extraction_job",
        fake_enqueue_revision_extraction_job,
    )

    with pytest.raises(HTTPException) as exc_info:
        run_async(
            projects_router.revision_processing_start(
                request=SimpleNamespace(session={}),
                background_tasks=FakeBackgroundTasks(),
                slug=project.slug,
                document_id=document.id,
                revision_id=revision.id,
                csrf_token="token",
                current_user=user,
                session=object(),
            )
        )

    assert exc_info.value.status_code == 409
    assert "accepted revisions" in exc_info.value.detail


def test_invalid_csrf_is_rejected_for_processing_start(monkeypatch) -> None:
    user, project, document, revision, context = build_revision_detail_context(
        revision_status=DocumentRevisionStatus.ACCEPTED.value,
    )
    running_job = build_processing_job(project_id=project.id, revision_id=revision.id)
    running_job.status = ProcessingJobStatus.RUNNING.value
    running_job.started_at = datetime.now(timezone.utc)
    running_job.lease_token = uuid.uuid4()
    running_job.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

    async def fake_get_revision_detail_for_user(_session, _user_id, _slug, _document_id, _revision_id):
        return context

    async def fake_enqueue_revision_extraction_job(_session, *, project_id, revision_id, trigger_kind, actor_id=None):
        return running_job

    monkeypatch.setattr(document_upload_service, "get_revision_detail_for_user", fake_get_revision_detail_for_user)
    monkeypatch.setattr(
        processing_job_service,
        "enqueue_revision_extraction_job",
        fake_enqueue_revision_extraction_job,
    )

    with TestClient(app) as client:
        login_client(client, monkeypatch, user)
        page = client.get(f"/projects/{project.slug}/documents/{document.id}/revisions/{revision.id}")
        csrf_token = extract_csrf_token(page.text)
        valid = client.post(
            f"/projects/{project.slug}/documents/{document.id}/revisions/{revision.id}/processing/start",
            data={"csrf_token": csrf_token},
        )
        bad_csrf = client.post(
            f"/projects/{project.slug}/documents/{document.id}/revisions/{revision.id}/processing/start",
            data={"csrf_token": "bad-token"},
        )

    assert valid.status_code != 403
    assert bad_csrf.status_code == 403


def test_router_source_does_not_call_executor_or_extractor_directly() -> None:
    source = inspect.getsource(projects_router)

    assert "execute_revision_extraction_job(" not in source
    assert "run_revision_extraction(" not in source
