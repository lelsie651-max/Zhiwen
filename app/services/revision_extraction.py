from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document_content import ExtractionRun
from app.models.document_revision import DocumentRevision, DocumentRevisionStatus, SourceAuthority
from app.models.ingestion_validation import (
    IngestionValidationReport,
    ValidationRecommendedAuthority,
    ValidationReportOutcome,
    ValidationReportStatus,
    ValidationUserDecision,
)
from app.repositories import revision_extraction as revision_extraction_repository
from app.schemas.document_extraction import ExtractedDocument, ExtractionOutcome
from app.schemas.file_ingestion import DetectedFileFormat
from app.schemas.revision_extraction import RevisionExtractionResult
from app.services.document_content import persist_extraction_result
from app.services.document_extraction import extract_document
from app.storage import FileStorage

logger = logging.getLogger(__name__)

EXTRACTOR_NAME = "zhiwen-deterministic-extractor"
EXTRACTOR_VERSION = "1.0.0"
SUPPORTED_EXTRACTION_FORMATS = {
    DetectedFileFormat.PDF,
    DetectedFileFormat.DOCX,
    DetectedFileFormat.TXT,
    DetectedFileFormat.MD,
}

RevisionExtractionFinalizer = Callable[
    [AsyncSession, DocumentRevision, ExtractionRun],
    Awaitable[None],
]


class RevisionExtractionError(Exception):
    """Raised when revision extraction orchestration cannot proceed."""


class RevisionExtractionNotFoundError(RevisionExtractionError):
    """Raised when the target revision does not belong to the project."""


class RevisionExtractionAdmissionStateError(RevisionExtractionError):
    """Raised when admission state is inconsistent with extraction entry rules."""


class RevisionExtractionTransitionError(RevisionExtractionError):
    """Raised when revision extraction status transitions are invalid."""


class RevisionExtractionConfigError(RevisionExtractionError):
    """Raised when extraction configuration is missing or invalid."""


async def run_revision_extraction(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    revision_id: uuid.UUID,
    storage: FileStorage,
    finalizer: RevisionExtractionFinalizer | None = None,
) -> RevisionExtractionResult:
    file_format, detected_encoding, storage_key = await _enter_parsing_phase(
        session,
        project_id=project_id,
        revision_id=revision_id,
    )
    await _enter_extracting_phase(
        session,
        project_id=project_id,
        revision_id=revision_id,
    )

    started_at = datetime.now(timezone.utc)
    try:
        extracted_document = await extract_document(
            storage,
            storage_key,
            file_format,
            detected_encoding,
        )
    except Exception:
        logger.warning("Revision extraction failed without exposing storage details.")
        extracted_document = _build_safe_failed_document(
            detected_format=file_format,
            detected_encoding=detected_encoding,
            warning="提取失败，当前文件无法安全读取。",
        )
    completed_at = datetime.now(timezone.utc)

    return await _finalize_extraction_phase(
        session,
        project_id=project_id,
        revision_id=revision_id,
        extracted_document=extracted_document,
        started_at=started_at,
        completed_at=completed_at,
        finalizer=finalizer,
    )


async def _enter_parsing_phase(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    revision_id: uuid.UUID,
) -> tuple[DetectedFileFormat, str | None, str]:
    revision = await _lock_revision_for_project(
        session,
        project_id=project_id,
        revision_id=revision_id,
    )
    if revision.status != DocumentRevisionStatus.ACCEPTED.value:
        raise RevisionExtractionTransitionError("Only accepted revisions can enter extraction.")

    report = await _lock_latest_report(session, revision_id=revision.id)
    _validate_admission_consistency_for_extraction(revision=revision, report=report)
    file_format, detected_encoding = _read_extraction_config(report)

    revision.status = DocumentRevisionStatus.PARSING.value
    try:
        await session.flush()
        await session.commit()
    except Exception:
        await session.rollback()
        raise

    return file_format, detected_encoding, revision.storage_key


async def _enter_extracting_phase(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    revision_id: uuid.UUID,
) -> None:
    revision = await _lock_revision_for_project(
        session,
        project_id=project_id,
        revision_id=revision_id,
    )
    if revision.status != DocumentRevisionStatus.PARSING.value:
        raise RevisionExtractionTransitionError("Revision must be parsing before entering extracting.")

    revision.status = DocumentRevisionStatus.EXTRACTING.value
    try:
        await session.flush()
        await session.commit()
    except Exception:
        await session.rollback()
        raise


async def _finalize_extraction_phase(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    revision_id: uuid.UUID,
    extracted_document: ExtractedDocument,
    started_at: datetime,
    completed_at: datetime,
    finalizer: RevisionExtractionFinalizer | None = None,
) -> RevisionExtractionResult:
    revision = await _lock_revision_for_project(
        session,
        project_id=project_id,
        revision_id=revision_id,
    )
    if revision.status != DocumentRevisionStatus.EXTRACTING.value:
        raise RevisionExtractionTransitionError("Revision must remain extracting before final persistence.")

    previous_status = revision.status
    failure_code = _resolve_failure_code(extracted_document.outcome)
    try:
        extraction_run = await persist_extraction_result(
            session,
            revision_id=revision.id,
            extracted_document=extracted_document,
            extractor_name=EXTRACTOR_NAME,
            extractor_version=EXTRACTOR_VERSION,
            failure_code=failure_code,
            failure_message=None,
            started_at=started_at,
            completed_at=completed_at,
            commit=False,
        )
        revision.status = _map_extraction_outcome_to_revision_status(extracted_document.outcome)
        if finalizer is not None:
            await finalizer(session, revision, extraction_run)
        await session.flush()
        await session.commit()
    except BaseException:
        revision.status = previous_status
        await session.rollback()
        raise

    return RevisionExtractionResult(
        revision_id=revision.id,
        extraction_run_id=extraction_run.id,
        outcome=extracted_document.outcome,
        revision_status=revision.status,
        block_count=extracted_document.block_count,
        character_count=extracted_document.character_count,
        page_count=extracted_document.page_count,
        warnings=list(extracted_document.warnings),
        requires_ocr=extracted_document.outcome == ExtractionOutcome.NEEDS_OCR,
    )


async def _lock_revision_for_project(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    revision_id: uuid.UUID,
) -> DocumentRevision:
    revision = await revision_extraction_repository.get_revision_for_extraction_update(
        session,
        project_id=project_id,
        revision_id=revision_id,
    )
    if revision is None:
        raise RevisionExtractionNotFoundError("Revision must belong to the target project.")
    return revision


async def _lock_latest_report(
    session: AsyncSession,
    *,
    revision_id: uuid.UUID,
) -> IngestionValidationReport:
    report = await revision_extraction_repository.get_latest_validation_report_for_update(
        session,
        revision_id=revision_id,
    )
    if report is None:
        raise RevisionExtractionAdmissionStateError("Revision has no validation report.")
    if report.revision_id != revision_id:
        raise RevisionExtractionAdmissionStateError("Latest report does not belong to the revision.")
    if report.status != ValidationReportStatus.COMPLETED.value:
        raise RevisionExtractionAdmissionStateError("Latest validation report must be completed.")
    if report.outcome == ValidationReportOutcome.PENDING.value:
        raise RevisionExtractionAdmissionStateError("Pending validation reports cannot enter extraction.")
    return report


def _validate_admission_consistency_for_extraction(
    *,
    revision: DocumentRevision,
    report: IngestionValidationReport,
) -> None:
    if report.outcome == ValidationReportOutcome.PASS.value:
        if report.user_decision is not None:
            raise RevisionExtractionAdmissionStateError("Pass reports must not carry manual decisions.")
        if report.recommended_authority in {
            ValidationRecommendedAuthority.FORMAL.value,
            ValidationRecommendedAuthority.REFERENCE.value,
        }:
            if revision.source_authority != report.recommended_authority:
                raise RevisionExtractionAdmissionStateError(
                    "Accepted revision authority must match the pass report recommendation."
                )
        elif revision.source_authority != SourceAuthority.PENDING.value:
            raise RevisionExtractionAdmissionStateError(
                "Pass reports without recommendation must keep pending source authority."
            )
        return

    if report.outcome == ValidationReportOutcome.WARNING.value:
        if report.user_decision == ValidationUserDecision.CONTINUE_FORMAL.value:
            if revision.source_authority != SourceAuthority.FORMAL.value:
                raise RevisionExtractionAdmissionStateError(
                    "Formal warning decisions require source_authority=formal."
                )
            return
        if report.user_decision == ValidationUserDecision.CONTINUE_REFERENCE.value:
            if revision.source_authority != SourceAuthority.REFERENCE.value:
                raise RevisionExtractionAdmissionStateError(
                    "Reference warning decisions require source_authority=reference."
                )
            return
        if report.user_decision == ValidationUserDecision.CANCEL.value:
            raise RevisionExtractionAdmissionStateError("Canceled warning reports cannot enter extraction.")
        raise RevisionExtractionAdmissionStateError(
            "Warning reports must have a continue_formal or continue_reference decision before extraction."
        )

    if report.outcome in {ValidationReportOutcome.REJECT.value, ValidationReportOutcome.QUARANTINE.value}:
        raise RevisionExtractionAdmissionStateError("Rejected or quarantined revisions cannot enter extraction.")

    raise RevisionExtractionAdmissionStateError("Unsupported admission report outcome for extraction.")


def _read_extraction_config(
    report: IngestionValidationReport,
) -> tuple[DetectedFileFormat, str | None]:
    if not isinstance(report.file_checks, dict):
        raise RevisionExtractionConfigError("Report file_checks must be a JSON object.")

    detected_format_raw = report.file_checks.get("detected_format")
    if not isinstance(detected_format_raw, str) or not detected_format_raw.strip():
        raise RevisionExtractionConfigError("detected_format is missing from file_checks.")
    try:
        detected_format = DetectedFileFormat(detected_format_raw.strip().lower())
    except ValueError as exc:
        raise RevisionExtractionConfigError("detected_format is not supported for extraction.") from exc
    if detected_format not in SUPPORTED_EXTRACTION_FORMATS:
        raise RevisionExtractionConfigError("detected_format is not supported for extraction.")

    detected_encoding_raw = report.file_checks.get("detected_encoding")
    if detected_encoding_raw is None:
        if detected_format in {DetectedFileFormat.TXT, DetectedFileFormat.MD}:
            raise RevisionExtractionConfigError("Text formats require detected_encoding for extraction.")
        return detected_format, None
    if not isinstance(detected_encoding_raw, str) or not detected_encoding_raw.strip():
        raise RevisionExtractionConfigError("detected_encoding must be a non-empty string when present.")
    return detected_format, detected_encoding_raw.strip()


def _build_safe_failed_document(
    *,
    detected_format: DetectedFileFormat,
    detected_encoding: str | None,
    warning: str,
) -> ExtractedDocument:
    return ExtractedDocument(
        outcome=ExtractionOutcome.FAILED,
        detected_format=detected_format,
        detected_encoding=detected_encoding,
        page_count=None,
        character_count=0,
        block_count=0,
        blocks=[],
        warnings=[warning],
        metadata={},
    )


def _resolve_failure_code(outcome: ExtractionOutcome) -> str | None:
    if outcome == ExtractionOutcome.NEEDS_OCR:
        return "ocr_required"
    if outcome == ExtractionOutcome.FAILED:
        return "deterministic_extraction_failed"
    return None


def _map_extraction_outcome_to_revision_status(outcome: ExtractionOutcome) -> str:
    if outcome in {ExtractionOutcome.SUCCESS, ExtractionOutcome.PARTIAL}:
        return DocumentRevisionStatus.AWAITING_REVIEW.value
    if outcome in {ExtractionOutcome.NEEDS_OCR, ExtractionOutcome.FAILED}:
        return DocumentRevisionStatus.FAILED.value
    raise RevisionExtractionTransitionError("Unsupported extraction outcome.")
