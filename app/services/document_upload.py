from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterable
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_content import ExtractionRun
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
    ValidationReportOutcome,
    ValidationReportStatus,
)
from app.models.processing_job import ProcessingJob, ProcessingJobType
from app.models.project import Project
from app.models.project_member import ProjectMember, ProjectMemberRole
from app.repositories import document as document_repository
from app.repositories import ingestion_validation as validation_repository
from app.repositories import processing_job as processing_job_repository
from app.repositories import project as project_repository
from app.schemas.document import DocumentCreate
from app.schemas.document_revision import DocumentRevisionCreate
from app.schemas.document_upload import DocumentUploadSubmit
from app.schemas.file_ingestion import ReceivedUpload, TechnicalPreflightOutcome, TechnicalPreflightResult
from app.services.file_ingestion import (
    get_local_file_storage,
    receive_upload_stream,
    run_technical_preflight,
)
from app.storage import FileStorage

logger = logging.getLogger(__name__)


class UploadAccessError(Exception):
    """Raised when a user cannot access an upload workflow resource."""


class UploadFormError(Exception):
    def __init__(self, field: str, message: str):
        super().__init__(message)
        self.field = field
        self.message = message


class UploadProjectNotFoundError(Exception):
    """Raised when the target project is unavailable for the current user."""


class UploadRevisionNotFoundError(Exception):
    """Raised when the target revision is unavailable for the current user."""


@dataclass(frozen=True)
class ProjectUploadContext:
    project: Project
    membership: ProjectMember
    documents: list[Document]

    @property
    def can_upload(self) -> bool:
        return self.membership.role in {ProjectMemberRole.OWNER, ProjectMemberRole.EDITOR}


@dataclass(frozen=True)
class ProjectDocumentSummary:
    document: Document
    latest_revision: DocumentRevision | None


@dataclass(frozen=True)
class RevisionDetailContext:
    project: Project
    membership: ProjectMember
    revision: DocumentRevision
    report: IngestionValidationReport | None
    risk_flags: list[IngestionRiskFlag]
    latest_processing_job: ProcessingJob | None
    latest_extraction_run: ExtractionRun | None

    @property
    def can_manage_processing(self) -> bool:
        return self.membership.role in {ProjectMemberRole.OWNER, ProjectMemberRole.EDITOR}


@dataclass(frozen=True)
class UploadTransactionResult:
    project: Project
    document: Document
    revision: DocumentRevision
    report: IngestionValidationReport
    risk_flags: list[IngestionRiskFlag]


async def get_project_upload_context(
    session: AsyncSession,
    user_id: uuid.UUID,
    slug: str,
) -> ProjectUploadContext:
    membership = await project_repository.get_project_member_for_user_by_slug(session, user_id, slug)
    if membership is None:
        raise UploadProjectNotFoundError()

    documents = await document_repository.list_documents_for_project(session, membership.project.id)
    return ProjectUploadContext(
        project=membership.project,
        membership=membership,
        documents=documents,
    )


async def get_project_workspace_context(
    session: AsyncSession,
    user_id: uuid.UUID,
    slug: str,
) -> tuple[ProjectUploadContext, list[ProjectDocumentSummary]]:
    context = await get_project_upload_context(session, user_id, slug)
    summaries = [
        ProjectDocumentSummary(
            document=document,
            latest_revision=max(document.revisions, key=lambda revision: revision.revision_no, default=None),
        )
        for document in context.documents
    ]
    return context, summaries


async def get_revision_detail_for_user(
    session: AsyncSession,
    user_id: uuid.UUID,
    slug: str,
    document_id: uuid.UUID,
    revision_id: uuid.UUID,
) -> RevisionDetailContext:
    context = await get_project_upload_context(session, user_id, slug)
    revision = await document_repository.get_revision_detail_for_project(
        session,
        context.project.id,
        document_id,
        revision_id,
    )
    if revision is None:
        raise UploadRevisionNotFoundError()

    latest_report = max(
        revision.validation_reports,
        key=lambda report: report.attempt_no,
        default=None,
    )
    risk_flags = list(latest_report.risk_flags) if latest_report is not None else []
    latest_processing_job = await processing_job_repository.get_latest_processing_job_for_revision(
        session,
        revision_id=revision.id,
        job_type=ProcessingJobType.REVISION_EXTRACTION.value,
    )
    latest_extraction_run = await processing_job_repository.get_latest_extraction_run_for_revision(
        session,
        revision_id=revision.id,
    )
    return RevisionDetailContext(
        project=context.project,
        membership=context.membership,
        revision=revision,
        report=latest_report,
        risk_flags=risk_flags,
        latest_processing_job=latest_processing_job,
        latest_extraction_run=latest_extraction_run,
    )


async def upload_document_for_project(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    project_slug: str,
    payload: DocumentUploadSubmit,
    chunks: AsyncIterable[bytes],
    original_filename: str,
    declared_mime: str | None,
    storage: FileStorage | None = None,
) -> UploadTransactionResult:
    context = await get_project_upload_context(session, user_id, project_slug)
    if not context.can_upload:
        raise UploadAccessError("Only owners and editors can upload documents.")

    storage_backend = storage or get_local_file_storage()
    received_upload = await receive_upload_stream(
        chunks,
        original_filename=original_filename,
        declared_mime=declared_mime,
        storage=storage_backend,
    )
    preflight_started_at = datetime.now(timezone.utc)
    preflight_result = await run_technical_preflight(received_upload, storage=storage_backend)
    preflight_completed_at = datetime.now(timezone.utc)
    if preflight_completed_at < preflight_started_at:
        raise ValueError("preflight_completed_at must not be earlier than preflight_started_at")

    try:
        storage_key = await storage_backend.promote_temp_file(
            received_upload.temp_file,
            suffix=received_upload.extension,
        )
    except Exception as exc:
        await _cleanup_temp_file_after_promotion_failure(storage_backend, received_upload, exc)
        raise

    try:
        if payload.upload_intent == UploadIntent.NEW_DOCUMENT:
            document = await _create_new_document(
                session,
                project_id=context.project.id,
                user_id=user_id,
                payload=payload,
            )
            latest_revision = None
        else:
            document = await _get_revision_target_document(
                session,
                project_id=context.project.id,
                payload=payload,
            )
            latest_revision = await document_repository.get_latest_revision_for_document(session, document.id)
            if latest_revision is None:
                raise UploadFormError("document_id", "目标文档还没有可衔接的上一版。")

        revision = await _create_revision_and_report(
            session,
            document=document,
            latest_revision=latest_revision,
            payload=payload,
            user_id=user_id,
            received_upload=received_upload,
            preflight_result=preflight_result,
            storage_key=storage_key,
            preflight_started_at=preflight_started_at,
            preflight_completed_at=preflight_completed_at,
        )
        report = revision.validation_reports[-1]
        risk_flags = list(report.risk_flags)
        await session.commit()
    except Exception as exc:
        await session.rollback()
        await _cleanup_promoted_file_after_failure(storage_backend, storage_key, exc)
        raise

    return UploadTransactionResult(
        project=context.project,
        document=document,
        revision=revision,
        report=report,
        risk_flags=risk_flags,
    )


async def _create_new_document(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    user_id: uuid.UUID,
    payload: DocumentUploadSubmit,
) -> Document:
    document_payload = DocumentCreate(
        title=payload.title,
        description=payload.description,
        logical_order_kind=payload.logical_order_kind,
        logical_order_value=payload.logical_order_value,
        logical_order_index=payload.logical_order_index,
        effective_from=payload.effective_from,
        effective_to=payload.effective_to,
    )
    document = Document(
        **document_payload.model_dump(mode="python"),
        project_id=project_id,
        created_by_id=user_id,
    )
    return await document_repository.create_document(session, document)


async def _get_revision_target_document(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    payload: DocumentUploadSubmit,
) -> Document:
    assert payload.document_id is not None
    document = await document_repository.get_document_for_update(session, project_id, payload.document_id)
    if document is None:
        raise UploadFormError("document_id", "目标文档不存在或不属于当前项目。")
    return document


async def _create_revision_and_report(
    session: AsyncSession,
    *,
    document: Document,
    latest_revision: DocumentRevision | None,
    payload: DocumentUploadSubmit,
    user_id: uuid.UUID,
    received_upload: ReceivedUpload,
    preflight_result: TechnicalPreflightResult,
    storage_key: str,
    preflight_started_at: datetime,
    preflight_completed_at: datetime,
) -> DocumentRevision:
    report_status, report_outcome, revision_status = _map_preflight_status(preflight_result)
    revision_no = 1 if latest_revision is None else latest_revision.revision_no + 1
    supersedes_revision_id = None if latest_revision is None else latest_revision.id
    revision_payload = DocumentRevisionCreate(
        document_id=document.id,
        revision_no=revision_no,
        upload_intent=payload.upload_intent,
        supersedes_revision_id=supersedes_revision_id,
        original_filename=received_upload.safe_original_filename,
        storage_key=storage_key,
        mime_type=preflight_result.detected_mime or "application/octet-stream",
        file_size_bytes=received_upload.file_size,
        sha256=received_upload.sha256,
        detected_language=None,
        language_confidence=None,
        source_authority=SourceAuthority.PENDING,
        status=revision_status,
    )
    revision = DocumentRevision(
        **revision_payload.model_dump(mode="python"),
        uploaded_by_id=user_id,
    )
    await document_repository.create_document_revision(session, revision)

    report = IngestionValidationReport(
        revision_id=revision.id,
        attempt_no=1,
        status=report_status,
        outcome=report_outcome,
        content_kind=None,
        content_confidence=None,
        recommended_authority=None,
        detected_language=None,
        language_confidence=None,
        character_count=0,
        printable_ratio=None,
        garbled_ratio=None,
        repetition_ratio=None,
        file_checks=_build_report_file_checks(preflight_result),
        text_checks={},
        classification_details={},
        summary=None,
        failure_code=None,
        failure_message=None,
        user_decision=None,
        decided_by_id=None,
        decided_at=None,
        started_at=preflight_started_at,
        completed_at=preflight_completed_at,
    )
    await validation_repository.create_validation_report(session, report)

    risk_flags = [
        IngestionRiskFlag(
            report_id=report.id,
            category=flag.category.value,
            kind=flag.kind,
            severity=flag.severity.value,
            detector=flag.detector.value,
            message=flag.message,
            confidence=flag.confidence,
            location=flag.location,
            details=flag.details,
            evidence_excerpt=flag.evidence_excerpt,
        )
        for flag in preflight_result.risk_flags
    ]
    if risk_flags:
        await validation_repository.create_risk_flags(session, risk_flags)

    revision.validation_reports.append(report)
    report.risk_flags.extend(risk_flags)
    return revision


def _build_report_file_checks(preflight_result: TechnicalPreflightResult) -> dict[str, object]:
    return {
        "detected_format": (
            preflight_result.detected_format.value if preflight_result.detected_format is not None else None
        ),
        "detected_mime": preflight_result.detected_mime,
        "detected_encoding": preflight_result.detected_encoding,
        **preflight_result.file_checks,
    }


def _map_preflight_status(
    preflight_result: TechnicalPreflightResult,
) -> tuple[str, str, str]:
    if preflight_result.outcome == TechnicalPreflightOutcome.REJECT:
        return (
            ValidationReportStatus.COMPLETED.value,
            ValidationReportOutcome.REJECT.value,
            DocumentRevisionStatus.REJECTED.value,
        )
    if preflight_result.outcome == TechnicalPreflightOutcome.QUARANTINE:
        return (
            ValidationReportStatus.COMPLETED.value,
            ValidationReportOutcome.QUARANTINE.value,
            DocumentRevisionStatus.QUARANTINED.value,
        )

    has_warning_risks = any(
        risk_flag.severity == IngestionRiskSeverity.WARNING for risk_flag in preflight_result.risk_flags
    )
    return (
        ValidationReportStatus.COMPLETED.value,
        ValidationReportOutcome.WARNING.value if has_warning_risks else ValidationReportOutcome.PASS.value,
        DocumentRevisionStatus.PREFLIGHT_CHECKING.value,
    )


async def _cleanup_promoted_file_after_failure(
    storage: FileStorage,
    storage_key: str,
    original_error: Exception,
) -> None:
    try:
        await storage.delete_file(storage_key)
    except Exception as cleanup_exc:
        logger.warning(
            "Failed to delete promoted upload file during compensation cleanup.",
            exc_info=cleanup_exc,
        )


async def _cleanup_temp_file_after_promotion_failure(
    storage: FileStorage,
    received_upload: ReceivedUpload,
    original_error: Exception,
) -> None:
    try:
        await storage.delete_temp_file(received_upload.temp_file)
    except Exception as cleanup_exc:
        logger.warning(
            "Failed to delete temporary upload file after promotion failure.",
            exc_info=cleanup_exc,
        )
