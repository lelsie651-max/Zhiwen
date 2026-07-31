import asyncio
import hashlib
import io
import inspect
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.models.document_content import ExtractionRun
from app.models.document_revision import DocumentRevision, DocumentRevisionStatus, SourceAuthority, UploadIntent
from app.models.ingestion_validation import (
    IngestionValidationReport,
    ValidationRecommendedAuthority,
    ValidationReportOutcome,
    ValidationReportStatus,
    ValidationUserDecision,
)
from app.schemas.document_extraction import ExtractedBlock, ExtractedBlockType, ExtractedDocument, ExtractionOutcome
from app.schemas.file_ingestion import DetectedFileFormat
from app.services import document_content as document_content_service
from app.services import document_extraction, revision_extraction as revision_extraction_service
from app.services.document_extraction import extract_document
from app.storage import LocalFileStorage


def run_async(awaitable):
    return asyncio.run(awaitable)


class FakeSession:
    def __init__(self) -> None:
        self.flush_count = 0
        self.commit_count = 0
        self.rollback_count = 0
        self.events: list[str] = []

    async def flush(self) -> None:
        self.flush_count += 1
        self.events.append("flush")

    async def commit(self) -> None:
        self.commit_count += 1
        self.events.append("commit")

    async def rollback(self) -> None:
        self.rollback_count += 1
        self.events.append("rollback")


def build_revision(
    *,
    revision_id: uuid.UUID | None = None,
    status: str = DocumentRevisionStatus.ACCEPTED.value,
    source_authority: str = SourceAuthority.PENDING.value,
    storage_key: str = "files/aa/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.pdf",
) -> DocumentRevision:
    now = datetime.now(timezone.utc)
    return DocumentRevision(
        id=revision_id or uuid.uuid4(),
        document_id=uuid.uuid4(),
        revision_no=1,
        upload_intent=UploadIntent.NEW_DOCUMENT.value,
        supersedes_revision_id=None,
        original_filename="sample.pdf",
        storage_key=storage_key,
        mime_type="application/pdf",
        file_size_bytes=10,
        sha256="a" * 64,
        detected_language=None,
        language_confidence=None,
        source_authority=source_authority,
        status=status,
        uploaded_by_id=uuid.uuid4(),
        uploaded_at=now,
        updated_at=now,
    )


def build_report(
    *,
    revision_id: uuid.UUID,
    outcome: str = ValidationReportOutcome.PASS.value,
    status: str = ValidationReportStatus.COMPLETED.value,
    recommended_authority: str | None = None,
    user_decision: str | None = None,
    file_checks: dict[str, object] | None = None,
) -> IngestionValidationReport:
    now = datetime.now(timezone.utc)
    return IngestionValidationReport(
        id=uuid.uuid4(),
        revision_id=revision_id,
        attempt_no=1,
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
        file_checks=file_checks
        or {
            "detected_format": DetectedFileFormat.PDF.value,
            "detected_encoding": None,
        },
        text_checks={},
        classification_details={},
        summary=None,
        failure_code=None,
        failure_message=None,
        user_decision=user_decision,
        decided_by_id=uuid.uuid4() if user_decision is not None else None,
        decided_at=now if user_decision is not None else None,
        started_at=now,
        completed_at=now,
    )


def build_extracted_document(
    *,
    outcome: ExtractionOutcome,
    detected_format: DetectedFileFormat = DetectedFileFormat.PDF,
    block_count: int = 1,
    character_count: int = 5,
    page_count: int | None = 1,
    warnings: list[str] | None = None,
) -> ExtractedDocument:
    location_key = "pdf:p1:b0"
    raw_text = "alpha"
    blocks = [
        ExtractedBlock(
            source_order=0,
            block_type=ExtractedBlockType.PAGE_TEXT,
            raw_text=raw_text,
            normalized_text=raw_text,
            location_key=location_key,
            anchor_hash=hashlib.sha256(
                f"{detected_format.value}|{location_key}|{raw_text}".encode("utf-8")
            ).hexdigest(),
            page_no=1,
            block_index=0,
            heading_path=[],
            metadata={},
        )
    ]
    if block_count == 0:
        blocks = []
        character_count = 0
        page_count = None
    return ExtractedDocument(
        outcome=outcome,
        detected_format=detected_format,
        detected_encoding="utf-8" if detected_format in {DetectedFileFormat.TXT, DetectedFileFormat.MD} else None,
        page_count=page_count,
        character_count=character_count,
        block_count=block_count,
        blocks=blocks,
        warnings=warnings or [],
        metadata={},
    )


def patch_revision_and_report(monkeypatch, *, revision: DocumentRevision, report: IngestionValidationReport):
    state = {"revision_calls": 0}

    async def fake_get_revision_for_extraction_update(_session, *, project_id, revision_id):
        assert revision_id == revision.id
        state["revision_calls"] += 1
        return revision

    async def fake_get_latest_validation_report_for_update(_session, *, revision_id):
        assert revision_id == revision.id
        return report

    monkeypatch.setattr(
        revision_extraction_service.revision_extraction_repository,
        "get_revision_for_extraction_update",
        fake_get_revision_for_extraction_update,
    )
    monkeypatch.setattr(
        revision_extraction_service.revision_extraction_repository,
        "get_latest_validation_report_for_update",
        fake_get_latest_validation_report_for_update,
    )
    return state


def test_accepted_revision_flows_to_awaiting_review(monkeypatch) -> None:
    session = FakeSession()
    project_id = uuid.uuid4()
    revision = build_revision(status=DocumentRevisionStatus.ACCEPTED.value)
    report = build_report(
        revision_id=revision.id,
        outcome=ValidationReportOutcome.PASS.value,
    )
    patch_revision_and_report(monkeypatch, revision=revision, report=report)

    extracted = build_extracted_document(outcome=ExtractionOutcome.SUCCESS, block_count=2, character_count=10, page_count=2)
    captured: dict[str, object] = {}

    async def fake_extract_document(_storage, storage_key, detected_format, encoding_hint):
        captured["extract_storage_key"] = storage_key
        captured["extract_format"] = detected_format
        captured["extract_encoding"] = encoding_hint
        captured["status_during_extract"] = revision.status
        return extracted

    async def fake_persist_extraction_result(_session, **kwargs):
        captured["persist_kwargs"] = kwargs
        return ExtractionRun(
            id=uuid.uuid4(),
            revision_id=revision.id,
            attempt_no=1,
            status="completed",
            outcome=kwargs["extracted_document"].outcome.value,
            extractor_name=kwargs["extractor_name"],
            extractor_version=kwargs["extractor_version"],
            detected_format=kwargs["extracted_document"].detected_format.value,
            detected_encoding=kwargs["extracted_document"].detected_encoding,
            page_count=kwargs["extracted_document"].page_count,
            character_count=kwargs["extracted_document"].character_count,
            block_count=kwargs["extracted_document"].block_count,
            warnings=list(kwargs["extracted_document"].warnings),
            content_metadata={},
            failure_code=kwargs["failure_code"],
            failure_message=kwargs["failure_message"],
            started_at=kwargs["started_at"],
            completed_at=kwargs["completed_at"],
        )

    monkeypatch.setattr(revision_extraction_service, "extract_document", fake_extract_document)
    monkeypatch.setattr(
        revision_extraction_service,
        "persist_extraction_result_in_transaction",
        fake_persist_extraction_result,
    )

    result = run_async(
        revision_extraction_service.run_revision_extraction(
            session,
            project_id=project_id,
            revision_id=revision.id,
            storage=object(),
        )
    )

    assert result.revision_status == DocumentRevisionStatus.AWAITING_REVIEW
    assert result.outcome == ExtractionOutcome.SUCCESS
    assert result.block_count == 2
    assert result.character_count == 10
    assert captured["status_during_extract"] == DocumentRevisionStatus.EXTRACTING.value
    assert captured["extract_storage_key"] == revision.storage_key
    assert captured["extract_format"] == DetectedFileFormat.PDF
    assert captured["extract_encoding"] is None
    assert "commit" not in captured["persist_kwargs"]
    assert captured["persist_kwargs"]["extractor_name"] == revision_extraction_service.EXTRACTOR_NAME
    assert captured["persist_kwargs"]["extractor_version"] == revision_extraction_service.EXTRACTOR_VERSION
    assert captured["persist_kwargs"]["failure_code"] is None
    assert revision.status == DocumentRevisionStatus.AWAITING_REVIEW.value
    assert session.commit_count == 3


def test_partial_flows_to_awaiting_review(monkeypatch) -> None:
    session = FakeSession()
    revision = build_revision(status=DocumentRevisionStatus.ACCEPTED.value)
    report = build_report(revision_id=revision.id, outcome=ValidationReportOutcome.PASS.value)
    patch_revision_and_report(monkeypatch, revision=revision, report=report)

    async def fake_extract_document(_storage, _storage_key, _detected_format, _encoding_hint):
        return build_extracted_document(outcome=ExtractionOutcome.PARTIAL, warnings=["page 2 empty"])

    async def fake_persist_extraction_result(_session, **kwargs):
        return ExtractionRun(
            id=uuid.uuid4(),
            revision_id=revision.id,
            attempt_no=1,
            status="completed",
            outcome=kwargs["extracted_document"].outcome.value,
            extractor_name="x",
            extractor_version="y",
            detected_format=kwargs["extracted_document"].detected_format.value,
            detected_encoding=kwargs["extracted_document"].detected_encoding,
            page_count=kwargs["extracted_document"].page_count,
            character_count=kwargs["extracted_document"].character_count,
            block_count=kwargs["extracted_document"].block_count,
            warnings=[],
            content_metadata={},
        )

    monkeypatch.setattr(revision_extraction_service, "extract_document", fake_extract_document)
    monkeypatch.setattr(
        revision_extraction_service,
        "persist_extraction_result_in_transaction",
        fake_persist_extraction_result,
    )

    result = run_async(
        revision_extraction_service.run_revision_extraction(
            session,
            project_id=uuid.uuid4(),
            revision_id=revision.id,
            storage=object(),
        )
    )

    assert result.revision_status == DocumentRevisionStatus.AWAITING_REVIEW
    assert result.outcome == ExtractionOutcome.PARTIAL


@pytest.mark.parametrize(
    ("outcome", "expected_failure_code"),
    [
        (ExtractionOutcome.NEEDS_OCR, "ocr_required"),
        (ExtractionOutcome.FAILED, "deterministic_extraction_failed"),
    ],
)
def test_needs_ocr_and_failed_enter_failed_and_keep_run(monkeypatch, outcome, expected_failure_code) -> None:
    session = FakeSession()
    revision = build_revision(status=DocumentRevisionStatus.ACCEPTED.value)
    report = build_report(
        revision_id=revision.id,
        outcome=ValidationReportOutcome.PASS.value,
        file_checks={"detected_format": DetectedFileFormat.PDF.value, "detected_encoding": None},
    )
    patch_revision_and_report(monkeypatch, revision=revision, report=report)
    captured: dict[str, object] = {}

    async def fake_extract_document(_storage, _storage_key, _detected_format, _encoding_hint):
        return build_extracted_document(outcome=outcome, block_count=0, page_count=None)

    async def fake_persist_extraction_result(_session, **kwargs):
        captured["failure_code"] = kwargs["failure_code"]
        return ExtractionRun(
            id=uuid.uuid4(),
            revision_id=revision.id,
            attempt_no=1,
            status="failed" if outcome == ExtractionOutcome.FAILED else "completed",
            outcome=kwargs["extracted_document"].outcome.value,
            extractor_name="x",
            extractor_version="y",
            detected_format=kwargs["extracted_document"].detected_format.value,
            detected_encoding=kwargs["extracted_document"].detected_encoding,
            page_count=kwargs["extracted_document"].page_count,
            character_count=kwargs["extracted_document"].character_count,
            block_count=kwargs["extracted_document"].block_count,
            warnings=[],
            content_metadata={},
            failure_code=kwargs["failure_code"],
        )

    monkeypatch.setattr(revision_extraction_service, "extract_document", fake_extract_document)
    monkeypatch.setattr(
        revision_extraction_service,
        "persist_extraction_result_in_transaction",
        fake_persist_extraction_result,
    )

    result = run_async(
        revision_extraction_service.run_revision_extraction(
            session,
            project_id=uuid.uuid4(),
            revision_id=revision.id,
            storage=object(),
        )
    )

    assert result.revision_status == DocumentRevisionStatus.FAILED
    assert result.requires_ocr is (outcome == ExtractionOutcome.NEEDS_OCR)
    assert captured["failure_code"] == expected_failure_code


def test_parsing_and_extracting_are_committed_separately(monkeypatch) -> None:
    session = FakeSession()
    revision = build_revision(status=DocumentRevisionStatus.ACCEPTED.value)
    report = build_report(revision_id=revision.id)
    patch_revision_and_report(monkeypatch, revision=revision, report=report)
    events: list[str] = []

    async def fake_extract_document(_storage, _storage_key, _detected_format, _encoding_hint):
        events.append("extract")
        return build_extracted_document(outcome=ExtractionOutcome.SUCCESS)

    async def fake_persist_extraction_result(_session, **kwargs):
        events.append("persist")
        return ExtractionRun(
            id=uuid.uuid4(),
            revision_id=revision.id,
            attempt_no=1,
            status="completed",
            outcome="success",
            extractor_name="x",
            extractor_version="y",
            detected_format="pdf",
            detected_encoding=None,
            page_count=1,
            character_count=5,
            block_count=1,
            warnings=[],
            content_metadata={},
        )

    original_flush = session.flush
    original_commit = session.commit

    async def tracked_flush():
        events.append(f"flush:{revision.status}")
        await original_flush()

    async def tracked_commit():
        events.append(f"commit:{revision.status}")
        await original_commit()

    session.flush = tracked_flush
    session.commit = tracked_commit
    monkeypatch.setattr(revision_extraction_service, "extract_document", fake_extract_document)
    monkeypatch.setattr(
        revision_extraction_service,
        "persist_extraction_result_in_transaction",
        fake_persist_extraction_result,
    )

    run_async(
        revision_extraction_service.run_revision_extraction(
            session,
            project_id=uuid.uuid4(),
            revision_id=revision.id,
            storage=object(),
        )
    )

    assert events[:4] == [
        f"flush:{DocumentRevisionStatus.PARSING.value}",
        f"commit:{DocumentRevisionStatus.PARSING.value}",
        f"flush:{DocumentRevisionStatus.EXTRACTING.value}",
        f"commit:{DocumentRevisionStatus.EXTRACTING.value}",
    ]
    assert "extract" in events


def test_extraction_happens_outside_database_transaction(monkeypatch) -> None:
    session = FakeSession()
    revision = build_revision(status=DocumentRevisionStatus.ACCEPTED.value)
    report = build_report(revision_id=revision.id)
    patch_revision_and_report(monkeypatch, revision=revision, report=report)
    events: list[str] = []

    original_commit = session.commit

    async def tracked_commit():
        events.append(f"commit:{revision.status}")
        await original_commit()

    session.commit = tracked_commit

    async def fake_extract_document(_storage, _storage_key, _detected_format, _encoding_hint):
        events.append(f"extract:{revision.status}")
        return build_extracted_document(outcome=ExtractionOutcome.SUCCESS)

    async def fake_persist_extraction_result(_session, **kwargs):
        events.append(f"persist:{revision.status}")
        return ExtractionRun(
            id=uuid.uuid4(),
            revision_id=revision.id,
            attempt_no=1,
            status="completed",
            outcome="success",
            extractor_name="x",
            extractor_version="y",
            detected_format="pdf",
            detected_encoding=None,
            page_count=1,
            character_count=5,
            block_count=1,
            warnings=[],
            content_metadata={},
        )

    monkeypatch.setattr(revision_extraction_service, "extract_document", fake_extract_document)
    monkeypatch.setattr(
        revision_extraction_service,
        "persist_extraction_result_in_transaction",
        fake_persist_extraction_result,
    )

    run_async(
        revision_extraction_service.run_revision_extraction(
            session,
            project_id=uuid.uuid4(),
            revision_id=revision.id,
            storage=object(),
        )
    )

    assert events[0] == f"commit:{DocumentRevisionStatus.PARSING.value}"
    assert events[1] == f"commit:{DocumentRevisionStatus.EXTRACTING.value}"
    assert events[2] == f"extract:{DocumentRevisionStatus.EXTRACTING.value}"


def test_final_persistence_failure_rolls_back_and_leaves_extracting(monkeypatch) -> None:
    session = FakeSession()
    revision = build_revision(status=DocumentRevisionStatus.ACCEPTED.value)
    report = build_report(revision_id=revision.id)
    patch_revision_and_report(monkeypatch, revision=revision, report=report)

    async def fake_extract_document(_storage, _storage_key, _detected_format, _encoding_hint):
        return build_extracted_document(outcome=ExtractionOutcome.SUCCESS)

    async def failing_persist(_session, **_kwargs):
        raise RuntimeError("persist failed")

    monkeypatch.setattr(revision_extraction_service, "extract_document", fake_extract_document)
    monkeypatch.setattr(
        revision_extraction_service,
        "persist_extraction_result_in_transaction",
        failing_persist,
    )

    with pytest.raises(RuntimeError):
        run_async(
            revision_extraction_service.run_revision_extraction(
                session,
                project_id=uuid.uuid4(),
                revision_id=revision.id,
                storage=object(),
            )
        )

    assert session.rollback_count == 1
    assert revision.status == DocumentRevisionStatus.EXTRACTING.value


def test_pass_and_warning_admission_consistency(monkeypatch) -> None:
    session = FakeSession()
    accepted_warning_revision = build_revision(
        status=DocumentRevisionStatus.ACCEPTED.value,
        source_authority=SourceAuthority.FORMAL.value,
    )
    accepted_warning_report = build_report(
        revision_id=accepted_warning_revision.id,
        outcome=ValidationReportOutcome.WARNING.value,
        user_decision=ValidationUserDecision.CONTINUE_FORMAL.value,
    )
    patch_revision_and_report(monkeypatch, revision=accepted_warning_revision, report=accepted_warning_report)

    async def fake_extract_document(_storage, _storage_key, _detected_format, _encoding_hint):
        return build_extracted_document(outcome=ExtractionOutcome.SUCCESS)

    async def fake_persist_extraction_result(_session, **kwargs):
        return ExtractionRun(
            id=uuid.uuid4(),
            revision_id=accepted_warning_revision.id,
            attempt_no=1,
            status="completed",
            outcome=kwargs["extracted_document"].outcome.value,
            extractor_name="x",
            extractor_version="y",
            detected_format="pdf",
            detected_encoding=None,
            page_count=1,
            character_count=5,
            block_count=1,
            warnings=[],
            content_metadata={},
        )

    monkeypatch.setattr(revision_extraction_service, "extract_document", fake_extract_document)
    monkeypatch.setattr(
        revision_extraction_service,
        "persist_extraction_result_in_transaction",
        fake_persist_extraction_result,
    )
    result = run_async(
        revision_extraction_service.run_revision_extraction(
            session,
            project_id=uuid.uuid4(),
            revision_id=accepted_warning_revision.id,
            storage=object(),
        )
    )
    assert result.revision_status == DocumentRevisionStatus.AWAITING_REVIEW


@pytest.mark.parametrize(
    ("outcome", "user_decision"),
    [
        (ValidationReportOutcome.REJECT.value, None),
        (ValidationReportOutcome.QUARANTINE.value, None),
        (ValidationReportOutcome.WARNING.value, ValidationUserDecision.CANCEL.value),
    ],
)
def test_reject_quarantine_and_cancel_cannot_extract(monkeypatch, outcome: str, user_decision: str | None) -> None:
    session = FakeSession()
    revision = build_revision(status=DocumentRevisionStatus.ACCEPTED.value)
    report = build_report(revision_id=revision.id, outcome=outcome, user_decision=user_decision)
    patch_revision_and_report(monkeypatch, revision=revision, report=report)

    with pytest.raises(revision_extraction_service.RevisionExtractionAdmissionStateError):
        run_async(
            revision_extraction_service.run_revision_extraction(
                session,
                project_id=uuid.uuid4(),
                revision_id=revision.id,
                storage=object(),
            )
        )

    assert revision.status == DocumentRevisionStatus.ACCEPTED.value
    assert session.commit_count == 0


@pytest.mark.parametrize(
    "file_checks",
    [
        {"detected_format": "exe", "detected_encoding": None},
        {"detected_format": DetectedFileFormat.MD.value, "detected_encoding": None},
        {"detected_format": DetectedFileFormat.TXT.value, "detected_encoding": 123},
    ],
)
def test_invalid_format_or_encoding_keeps_accepted(monkeypatch, file_checks: dict[str, object]) -> None:
    session = FakeSession()
    revision = build_revision(status=DocumentRevisionStatus.ACCEPTED.value)
    report = build_report(revision_id=revision.id, file_checks=file_checks)
    patch_revision_and_report(monkeypatch, revision=revision, report=report)

    with pytest.raises(revision_extraction_service.RevisionExtractionConfigError):
        run_async(
            revision_extraction_service.run_revision_extraction(
                session,
                project_id=uuid.uuid4(),
                revision_id=revision.id,
                storage=object(),
            )
        )

    assert revision.status == DocumentRevisionStatus.ACCEPTED.value
    assert session.commit_count == 0


@pytest.mark.parametrize(
    "status",
    [
        DocumentRevisionStatus.PARSING.value,
        DocumentRevisionStatus.EXTRACTING.value,
        DocumentRevisionStatus.AWAITING_REVIEW.value,
        DocumentRevisionStatus.COMPLETED.value,
        DocumentRevisionStatus.FAILED.value,
        DocumentRevisionStatus.REJECTED.value,
        DocumentRevisionStatus.QUARANTINED.value,
        DocumentRevisionStatus.NEEDS_CONFIRMATION.value,
    ],
)
def test_non_accepted_statuses_do_not_repeat_extraction(monkeypatch, status: str) -> None:
    session = FakeSession()
    revision = build_revision(status=status)
    report = build_report(revision_id=revision.id)
    patch_revision_and_report(monkeypatch, revision=revision, report=report)

    with pytest.raises(revision_extraction_service.RevisionExtractionTransitionError):
        run_async(
            revision_extraction_service.run_revision_extraction(
                session,
                project_id=uuid.uuid4(),
                revision_id=revision.id,
                storage=object(),
            )
        )


def test_real_extraction_times_are_passed_to_run(monkeypatch) -> None:
    session = FakeSession()
    revision = build_revision(status=DocumentRevisionStatus.ACCEPTED.value)
    report = build_report(revision_id=revision.id)
    patch_revision_and_report(monkeypatch, revision=revision, report=report)
    captured: dict[str, object] = {}

    async def fake_extract_document(_storage, _storage_key, _detected_format, _encoding_hint):
        await asyncio.sleep(0)
        return build_extracted_document(outcome=ExtractionOutcome.SUCCESS)

    async def fake_persist_extraction_result(_session, **kwargs):
        captured["started_at"] = kwargs["started_at"]
        captured["completed_at"] = kwargs["completed_at"]
        return ExtractionRun(
            id=uuid.uuid4(),
            revision_id=revision.id,
            attempt_no=1,
            status="completed",
            outcome="success",
            extractor_name="x",
            extractor_version="y",
            detected_format="pdf",
            detected_encoding=None,
            page_count=1,
            character_count=5,
            block_count=1,
            warnings=[],
            content_metadata={},
        )

    monkeypatch.setattr(revision_extraction_service, "extract_document", fake_extract_document)
    monkeypatch.setattr(
        revision_extraction_service,
        "persist_extraction_result_in_transaction",
        fake_persist_extraction_result,
    )

    run_async(
        revision_extraction_service.run_revision_extraction(
            session,
            project_id=uuid.uuid4(),
            revision_id=revision.id,
            storage=object(),
        )
    )

    assert captured["started_at"] <= captured["completed_at"]
    assert captured["started_at"].tzinfo == timezone.utc
    assert captured["completed_at"].tzinfo == timezone.utc


class StubStorage:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    async def read_file_chunks(self, _storage_key: str, *, chunk_size: int):
        for start in range(0, len(self.payload), chunk_size):
            yield self.payload[start : start + chunk_size]


def build_pdf_bytes() -> bytes:
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    page = writer.add_blank_page(width=300, height=300)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
    )
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 260 Td (hello) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_four_formats_use_asyncio_to_thread(monkeypatch) -> None:
    recorded: list[str] = []

    async def fake_to_thread(func, *args, **kwargs):
        recorded.append(func.__name__)
        return func(*args, **kwargs)

    monkeypatch.setattr(document_extraction.asyncio, "to_thread", fake_to_thread)

    monkeypatch.setattr(
        document_extraction,
        "_extract_pdf",
        lambda _spool_file: build_extracted_document(outcome=ExtractionOutcome.SUCCESS, detected_format=DetectedFileFormat.PDF),
    )
    monkeypatch.setattr(
        document_extraction,
        "_extract_docx",
        lambda _spool_file: build_extracted_document(outcome=ExtractionOutcome.SUCCESS, detected_format=DetectedFileFormat.DOCX),
    )
    monkeypatch.setattr(
        document_extraction,
        "_extract_markdown",
        lambda _spool_file, _encoding_hint: build_extracted_document(outcome=ExtractionOutcome.SUCCESS, detected_format=DetectedFileFormat.MD),
    )
    monkeypatch.setattr(
        document_extraction,
        "_extract_txt",
        lambda _spool_file, _encoding_hint: build_extracted_document(outcome=ExtractionOutcome.SUCCESS, detected_format=DetectedFileFormat.TXT),
    )

    run_async(extract_document(StubStorage(build_pdf_bytes()), "key", DetectedFileFormat.PDF))
    run_async(extract_document(StubStorage(b"docx"), "key", DetectedFileFormat.DOCX))
    run_async(extract_document(StubStorage(b"# md"), "key", DetectedFileFormat.MD, "utf-8"))
    run_async(extract_document(StubStorage(b"txt"), "key", DetectedFileFormat.TXT, "utf-8"))

    assert recorded == ["<lambda>", "<lambda>", "<lambda>", "<lambda>"]


def test_persist_extraction_result_default_commit_behavior_unchanged(monkeypatch) -> None:
    session = FakeSession()
    revision = build_revision(status=DocumentRevisionStatus.ACCEPTED.value)
    captured: dict[str, object] = {}

    async def fake_get_revision_for_extraction_update(_session, _revision_id):
        return revision

    async def fake_get_next_extraction_attempt_no(_session, _revision_id):
        return 1

    async def fake_create_extraction_run(_session, extraction_run):
        captured["run"] = extraction_run
        return extraction_run

    async def fake_create_document_blocks(_session, blocks):
        captured["blocks"] = blocks
        return blocks

    monkeypatch.setattr(
        document_content_service.document_content_repository,
        "get_revision_for_extraction_update",
        fake_get_revision_for_extraction_update,
    )
    monkeypatch.setattr(
        document_content_service.document_content_repository,
        "get_next_extraction_attempt_no",
        fake_get_next_extraction_attempt_no,
    )
    monkeypatch.setattr(
        document_content_service.document_content_repository,
        "create_extraction_run",
        fake_create_extraction_run,
    )
    monkeypatch.setattr(
        document_content_service.document_content_repository,
        "create_document_blocks",
        fake_create_document_blocks,
    )

    run_async(
        document_content_service.persist_extraction_result(
            session,
            revision_id=revision.id,
            extracted_document=build_extracted_document(outcome=ExtractionOutcome.SUCCESS),
            extractor_name="x",
            extractor_version="1.0.0",
        )
    )

    assert session.commit_count == 1
    assert session.rollback_count == 0
    assert captured["run"].attempt_no == 1


def test_persist_extraction_result_commit_false_does_not_commit_or_rollback(monkeypatch) -> None:
    session = FakeSession()
    revision = build_revision(status=DocumentRevisionStatus.ACCEPTED.value)

    async def fake_get_revision_for_extraction_update(_session, _revision_id):
        return revision

    async def fake_get_next_extraction_attempt_no(_session, _revision_id):
        return 2

    async def fake_create_extraction_run(_session, extraction_run):
        return extraction_run

    async def fake_create_document_blocks(_session, blocks):
        return blocks

    monkeypatch.setattr(
        document_content_service.document_content_repository,
        "get_revision_for_extraction_update",
        fake_get_revision_for_extraction_update,
    )
    monkeypatch.setattr(
        document_content_service.document_content_repository,
        "get_next_extraction_attempt_no",
        fake_get_next_extraction_attempt_no,
    )
    monkeypatch.setattr(
        document_content_service.document_content_repository,
        "create_extraction_run",
        fake_create_extraction_run,
    )
    monkeypatch.setattr(
        document_content_service.document_content_repository,
        "create_document_blocks",
        fake_create_document_blocks,
    )

    result = run_async(
        document_content_service.persist_extraction_result(
            session,
            revision_id=revision.id,
            extracted_document=build_extracted_document(outcome=ExtractionOutcome.SUCCESS),
            extractor_name="x",
            extractor_version="1.0.0",
            commit=False,
        )
    )

    assert result.attempt_no == 2
    assert session.commit_count == 0
    assert session.rollback_count == 0


def test_service_does_not_update_document_current_revision_or_import_fact(monkeypatch) -> None:
    source = inspect.getsource(revision_extraction_service)
    assert "current_revision_id" not in source
    assert "Fact" not in source


def test_service_log_does_not_leak_storage_key(monkeypatch, caplog) -> None:
    session = FakeSession()
    storage_key = "files/secret/internal.bin"
    revision = build_revision(status=DocumentRevisionStatus.ACCEPTED.value, storage_key=storage_key)
    report = build_report(revision_id=revision.id)
    patch_revision_and_report(monkeypatch, revision=revision, report=report)

    async def broken_extract_document(_storage, _storage_key, _detected_format, _encoding_hint):
        raise RuntimeError(f"boom {storage_key}")

    async def fake_persist_extraction_result(_session, **kwargs):
        return ExtractionRun(
            id=uuid.uuid4(),
            revision_id=revision.id,
            attempt_no=1,
            status="failed",
            outcome=kwargs["extracted_document"].outcome.value,
            extractor_name="x",
            extractor_version="y",
            detected_format="pdf",
            detected_encoding=None,
            page_count=None,
            character_count=0,
            block_count=0,
            warnings=kwargs["extracted_document"].warnings,
            content_metadata={},
            failure_code=kwargs["failure_code"],
        )

    monkeypatch.setattr(revision_extraction_service, "extract_document", broken_extract_document)
    monkeypatch.setattr(
        revision_extraction_service,
        "persist_extraction_result_in_transaction",
        fake_persist_extraction_result,
    )

    with caplog.at_level("WARNING"):
        result = run_async(
            revision_extraction_service.run_revision_extraction(
                session,
                project_id=uuid.uuid4(),
                revision_id=revision.id,
                storage=object(),
            )
        )

    assert result.outcome == ExtractionOutcome.FAILED
    assert storage_key not in caplog.text


def test_enter_parsing_phase_rolls_back_on_report_error(monkeypatch) -> None:
    session = FakeSession()
    revision = build_revision(status=DocumentRevisionStatus.ACCEPTED.value)

    async def fake_get_revision_for_extraction_update(_session, *, project_id, revision_id):
        return revision

    async def fake_get_latest_validation_report_for_update(_session, *, revision_id):
        return None

    monkeypatch.setattr(
        revision_extraction_service.revision_extraction_repository,
        "get_revision_for_extraction_update",
        fake_get_revision_for_extraction_update,
    )
    monkeypatch.setattr(
        revision_extraction_service.revision_extraction_repository,
        "get_latest_validation_report_for_update",
        fake_get_latest_validation_report_for_update,
    )

    with pytest.raises(revision_extraction_service.RevisionExtractionAdmissionStateError):
        run_async(
            revision_extraction_service._enter_parsing_phase(
                session,
                project_id=uuid.uuid4(),
                revision_id=revision.id,
            )
        )

    assert session.rollback_count == 1
    assert session.commit_count == 0


def test_enter_extracting_phase_cancelled_error_rolls_back(monkeypatch) -> None:
    session = FakeSession()
    revision = build_revision(status=DocumentRevisionStatus.PARSING.value)

    async def fake_lock_revision_for_project(_session, *, project_id, revision_id):
        raise asyncio.CancelledError

    monkeypatch.setattr(
        revision_extraction_service,
        "_lock_revision_for_project",
        fake_lock_revision_for_project,
    )

    with pytest.raises(asyncio.CancelledError):
        run_async(
            revision_extraction_service._enter_extracting_phase(
                session,
                project_id=uuid.uuid4(),
                revision_id=revision.id,
            )
        )

    assert session.rollback_count == 1
    assert session.commit_count == 0


def test_finalize_phase_cancelled_error_rolls_back(monkeypatch) -> None:
    session = FakeSession()
    revision = build_revision(status=DocumentRevisionStatus.EXTRACTING.value)

    async def fake_lock_revision_for_project(_session, *, project_id, revision_id):
        return revision

    async def fake_persist_extraction_result_in_transaction(_session, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(
        revision_extraction_service,
        "_lock_revision_for_project",
        fake_lock_revision_for_project,
    )
    monkeypatch.setattr(
        revision_extraction_service,
        "persist_extraction_result_in_transaction",
        fake_persist_extraction_result_in_transaction,
    )

    with pytest.raises(asyncio.CancelledError):
        run_async(
            revision_extraction_service._finalize_extraction_phase(
                session,
                project_id=uuid.uuid4(),
                revision_id=revision.id,
                extracted_document=build_extracted_document(outcome=ExtractionOutcome.SUCCESS),
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            )
        )

    assert session.rollback_count == 1
    assert revision.status == DocumentRevisionStatus.EXTRACTING.value


def test_finalize_phase_lock_revision_not_found_rolls_back_without_unbound_local(monkeypatch) -> None:
    session = FakeSession()

    async def fake_lock_revision_for_project(_session, *, project_id, revision_id):
        raise revision_extraction_service.RevisionExtractionNotFoundError("missing revision")

    monkeypatch.setattr(
        revision_extraction_service,
        "_lock_revision_for_project",
        fake_lock_revision_for_project,
    )

    with pytest.raises(revision_extraction_service.RevisionExtractionNotFoundError) as exc_info:
        run_async(
            revision_extraction_service._finalize_extraction_phase(
                session,
                project_id=uuid.uuid4(),
                revision_id=uuid.uuid4(),
                extracted_document=build_extracted_document(outcome=ExtractionOutcome.SUCCESS),
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            )
        )

    assert "missing revision" in str(exc_info.value)
    assert session.rollback_count == 1


def test_finalize_phase_lock_revision_cancelled_rolls_back_without_unbound_local(monkeypatch) -> None:
    session = FakeSession()

    async def fake_lock_revision_for_project(_session, *, project_id, revision_id):
        raise asyncio.CancelledError

    monkeypatch.setattr(
        revision_extraction_service,
        "_lock_revision_for_project",
        fake_lock_revision_for_project,
    )

    with pytest.raises(asyncio.CancelledError):
        run_async(
            revision_extraction_service._finalize_extraction_phase(
                session,
                project_id=uuid.uuid4(),
                revision_id=uuid.uuid4(),
                extracted_document=build_extracted_document(outcome=ExtractionOutcome.SUCCESS),
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            )
        )

    assert session.rollback_count == 1


def test_finalize_phase_transition_error_rolls_back_and_preserves_revision_status(monkeypatch) -> None:
    session = FakeSession()
    revision = build_revision(status=DocumentRevisionStatus.PARSING.value)

    async def fake_lock_revision_for_project(_session, *, project_id, revision_id):
        return revision

    monkeypatch.setattr(
        revision_extraction_service,
        "_lock_revision_for_project",
        fake_lock_revision_for_project,
    )

    with pytest.raises(revision_extraction_service.RevisionExtractionTransitionError):
        run_async(
            revision_extraction_service._finalize_extraction_phase(
                session,
                project_id=uuid.uuid4(),
                revision_id=revision.id,
                extracted_document=build_extracted_document(outcome=ExtractionOutcome.SUCCESS),
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            )
        )

    assert session.rollback_count == 1
    assert revision.status == DocumentRevisionStatus.PARSING.value


def test_finalize_phase_finalizer_failure_rolls_back_and_restores_extracting(monkeypatch) -> None:
    session = FakeSession()
    revision = build_revision(status=DocumentRevisionStatus.EXTRACTING.value)
    extraction_run = ExtractionRun(
        id=uuid.uuid4(),
        revision_id=revision.id,
        attempt_no=1,
        status="completed",
        outcome="success",
        extractor_name="x",
        extractor_version="y",
        detected_format="pdf",
        detected_encoding=None,
        page_count=1,
        character_count=5,
        block_count=1,
        warnings=[],
        content_metadata={},
        completed_at=datetime.now(timezone.utc),
    )

    async def fake_lock_revision_for_project(_session, *, project_id, revision_id):
        return revision

    async def fake_persist_extraction_result_in_transaction(_session, **kwargs):
        return extraction_run

    async def fake_finalizer(_session, _revision, _extraction_run):
        raise RuntimeError("finalizer failed")

    monkeypatch.setattr(
        revision_extraction_service,
        "_lock_revision_for_project",
        fake_lock_revision_for_project,
    )
    monkeypatch.setattr(
        revision_extraction_service,
        "persist_extraction_result_in_transaction",
        fake_persist_extraction_result_in_transaction,
    )

    with pytest.raises(RuntimeError):
        run_async(
            revision_extraction_service._finalize_extraction_phase(
                session,
                project_id=uuid.uuid4(),
                revision_id=revision.id,
                extracted_document=build_extracted_document(outcome=ExtractionOutcome.SUCCESS),
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                finalizer=fake_finalizer,
            )
        )

    assert session.rollback_count == 1
    assert revision.status == DocumentRevisionStatus.EXTRACTING.value


def test_finalize_phase_commit_failure_rolls_back_and_restores_extracting(monkeypatch) -> None:
    session = FakeSession()
    revision = build_revision(status=DocumentRevisionStatus.EXTRACTING.value)
    extraction_run = ExtractionRun(
        id=uuid.uuid4(),
        revision_id=revision.id,
        attempt_no=1,
        status="completed",
        outcome="success",
        extractor_name="x",
        extractor_version="y",
        detected_format="pdf",
        detected_encoding=None,
        page_count=1,
        character_count=5,
        block_count=1,
        warnings=[],
        content_metadata={},
        completed_at=datetime.now(timezone.utc),
    )

    async def fake_lock_revision_for_project(_session, *, project_id, revision_id):
        return revision

    async def fake_persist_extraction_result_in_transaction(_session, **kwargs):
        return extraction_run

    async def failing_commit():
        session.commit_count += 1
        session.events.append("commit")
        raise RuntimeError("commit failed")

    monkeypatch.setattr(
        revision_extraction_service,
        "_lock_revision_for_project",
        fake_lock_revision_for_project,
    )
    monkeypatch.setattr(
        revision_extraction_service,
        "persist_extraction_result_in_transaction",
        fake_persist_extraction_result_in_transaction,
    )
    session.commit = failing_commit

    with pytest.raises(RuntimeError):
        run_async(
            revision_extraction_service._finalize_extraction_phase(
                session,
                project_id=uuid.uuid4(),
                revision_id=revision.id,
                extracted_document=build_extracted_document(outcome=ExtractionOutcome.SUCCESS),
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            )
        )

    assert session.rollback_count == 1
    assert revision.status == DocumentRevisionStatus.EXTRACTING.value


def test_finalize_phase_source_uses_in_transaction_persistence() -> None:
    source = inspect.getsource(revision_extraction_service)
    assert "persist_extraction_result_in_transaction(" in source
    assert "persist_extraction_result(" not in source
    assert "commit=False" not in source
