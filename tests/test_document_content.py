import asyncio
import hashlib
import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError
from sqlalchemy import CheckConstraint, UniqueConstraint
from sqlalchemy.exc import IntegrityError

from app.models import Base
from app.models.document_content import DocumentBlock, ExtractionRun, SourceEvidence
from app.models.document_revision import DocumentRevision
from app.schemas.document_content import (
    ExtractionRunInternalRead,
    ExtractionRunRead,
    SourceEvidenceCreate,
)
from app.schemas.document_extraction import (
    ExtractedBlock,
    ExtractedBlockType,
    ExtractedDocument,
    ExtractionOutcome,
)
from app.schemas.file_ingestion import DetectedFileFormat
from app.services import document_content as document_content_service


def run_async(awaitable):
    return asyncio.run(awaitable)


def build_anchor_hash(*, detected_format: str, location_key: str, raw_text: str) -> str:
    return hashlib.sha256(
        f"{detected_format}|{location_key}|{raw_text}".encode("utf-8")
    ).hexdigest()


def build_extracted_block(
    *,
    source_order: int,
    raw_text: str,
    normalized_text: str | None = None,
    location_key: str | None = None,
    anchor_hash: str | None = None,
) -> ExtractedBlock:
    actual_normalized_text = normalized_text or raw_text
    actual_location_key = location_key or f"loc-{source_order}"
    actual_anchor_hash = anchor_hash or build_anchor_hash(
        detected_format=DetectedFileFormat.MD.value,
        location_key=actual_location_key,
        raw_text=raw_text,
    )
    return ExtractedBlock(
        source_order=source_order,
        block_type=ExtractedBlockType.PARAGRAPH,
        raw_text=raw_text,
        normalized_text=actual_normalized_text,
        location_key=actual_location_key,
        anchor_hash=actual_anchor_hash,
        block_index=source_order,
        heading_path=[],
        metadata={},
    )


def build_extracted_document(*blocks: ExtractedBlock, outcome: ExtractionOutcome = ExtractionOutcome.SUCCESS) -> ExtractedDocument:
    block_list = list(blocks)
    return ExtractedDocument(
        outcome=outcome,
        detected_format=DetectedFileFormat.MD,
        detected_encoding="utf-8",
        page_count=None,
        character_count=sum(len(block.normalized_text) for block in block_list),
        block_count=len(block_list),
        blocks=block_list,
        warnings=[],
        metadata={},
    )


class FakeSession:
    def __init__(self) -> None:
        self.commit_called = False
        self.rollback_called = False

    async def commit(self) -> None:
        self.commit_called = True

    async def rollback(self) -> None:
        self.rollback_called = True


def test_document_block_preserves_raw_text_and_normalized_text_whitespace() -> None:
    raw_text = "  alpha \n"
    normalized_text = "\tbeta  \n"
    block = DocumentBlock(
        id=uuid.uuid4(),
        extraction_run_id=uuid.uuid4(),
        source_order=0,
        block_type="paragraph",
        raw_text=raw_text,
        normalized_text=normalized_text,
        location_key=" loc-0 ",
        anchor_hash="A" * 64,
        block_index=0,
        heading_path=[],
        block_metadata={},
    )

    assert block.raw_text == raw_text
    assert block.normalized_text == normalized_text
    assert block.location_key == "loc-0"
    assert block.anchor_hash == "a" * 64


def test_source_evidence_preserves_excerpt_whitespace() -> None:
    excerpt = "  alpha \n"
    evidence = SourceEvidence(
        id=uuid.uuid4(),
        block_id=uuid.uuid4(),
        start_offset=0,
        end_offset=len(excerpt),
        excerpt=excerpt,
        excerpt_hash="A" * 64,
    )

    assert evidence.excerpt == excerpt
    assert evidence.excerpt_hash == "a" * 64


def test_document_content_tables_are_registered() -> None:
    assert {"extraction_runs", "document_blocks", "source_evidences"} <= set(Base.metadata.tables)


def test_extraction_run_history_unique_constraint_exists() -> None:
    table = Base.metadata.tables["extraction_runs"]
    assert any(
        isinstance(constraint, UniqueConstraint)
        and tuple(constraint.columns.keys()) == ("revision_id", "attempt_no")
        for constraint in table.constraints
    )


def test_document_block_unique_constraints_exist() -> None:
    table = Base.metadata.tables["document_blocks"]
    unique_columns = {
        tuple(constraint.columns.keys())
        for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    assert ("extraction_run_id", "source_order") in unique_columns
    assert ("extraction_run_id", "location_key") in unique_columns
    assert ("extraction_run_id", "anchor_hash") in unique_columns


def test_source_evidence_offset_constraints_exist() -> None:
    table = Base.metadata.tables["source_evidences"]
    check_sql = {
        str(constraint.sqltext)
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }
    assert any("start_offset >= 0" in sql for sql in check_sql)
    assert any("end_offset > start_offset" in sql for sql in check_sql)


def test_source_evidence_create_rejects_excerpt_and_hash() -> None:
    with pytest.raises(ValidationError):
        SourceEvidenceCreate(
            block_id=uuid.uuid4(),
            start_offset=0,
            end_offset=1,
            excerpt="x",
            excerpt_hash="a" * 64,
        )


def test_extraction_run_safe_read_hides_failure_message() -> None:
    assert "failure_message" not in ExtractionRunRead.model_fields


def test_extraction_run_internal_read_includes_failure_message() -> None:
    assert "failure_message" in ExtractionRunInternalRead.model_fields


def test_persist_extraction_result_auto_generates_attempt_no(monkeypatch) -> None:
    session = FakeSession()
    revision = DocumentRevision(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        revision_no=1,
        upload_intent="new_document",
        supersedes_revision_id=None,
        original_filename="notes.md",
        storage_key="files/aa/a.bin",
        mime_type="text/markdown",
        file_size_bytes=10,
        sha256="a" * 64,
        source_authority="pending",
        status="uploaded",
        uploaded_by_id=uuid.uuid4(),
        uploaded_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    captured: dict[str, object] = {}

    async def fake_get_revision_for_extraction_update(_session, _revision_id):
        return revision

    async def fake_get_next_extraction_attempt_no(_session, _revision_id):
        return 3

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

    result = run_async(
        document_content_service.persist_extraction_result(
            session,
            revision_id=revision.id,
            extracted_document=build_extracted_document(
                build_extracted_block(source_order=0, raw_text="alpha")
            ),
            extractor_name="deterministic-extractor",
            extractor_version="1.0.0",
        )
    )

    assert result.attempt_no == 3
    assert captured["run"].attempt_no == 3
    assert len(captured["blocks"]) == 1
    assert session.commit_called is True


def test_persist_extraction_result_rejects_invalid_anchor_hash_without_writing(monkeypatch) -> None:
    session = FakeSession()
    called = {"get_revision": False, "create_run": False}

    async def fake_get_revision_for_extraction_update(_session, _revision_id):
        called["get_revision"] = True
        return None

    async def fake_create_extraction_run(_session, _run):
        called["create_run"] = True
        return _run

    monkeypatch.setattr(
        document_content_service.document_content_repository,
        "get_revision_for_extraction_update",
        fake_get_revision_for_extraction_update,
    )
    monkeypatch.setattr(
        document_content_service.document_content_repository,
        "create_extraction_run",
        fake_create_extraction_run,
    )

    with pytest.raises(document_content_service.InvalidExtractionResultError):
        run_async(
            document_content_service.persist_extraction_result(
                session,
                revision_id=uuid.uuid4(),
                extracted_document=build_extracted_document(
                    build_extracted_block(
                        source_order=0,
                        raw_text="alpha",
                        anchor_hash="b" * 64,
                    )
                ),
                extractor_name="deterministic-extractor",
                extractor_version="1.0.0",
            )
        )

    assert called["get_revision"] is False
    assert called["create_run"] is False
    assert session.commit_called is False
    assert session.rollback_called is True


def test_persist_extraction_result_accepts_correct_anchor_hash_and_preserves_raw_text(monkeypatch) -> None:
    session = FakeSession()
    revision = DocumentRevision(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        revision_no=1,
        upload_intent="new_document",
        supersedes_revision_id=None,
        original_filename="notes.md",
        storage_key="files/aa/a.bin",
        mime_type="text/markdown",
        file_size_bytes=10,
        sha256="a" * 64,
        source_authority="pending",
        status="uploaded",
        uploaded_by_id=uuid.uuid4(),
        uploaded_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    raw_text = "  alpha \n"
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

    extracted_block = build_extracted_block(
        source_order=0,
        raw_text=raw_text,
        anchor_hash=build_anchor_hash(
            detected_format=DetectedFileFormat.MD.value,
            location_key="loc-0",
            raw_text=raw_text,
        ),
    )
    result = run_async(
        document_content_service.persist_extraction_result(
            session,
            revision_id=revision.id,
            extracted_document=build_extracted_document(extracted_block),
            extractor_name="deterministic-extractor",
            extractor_version="1.0.0",
        )
    )

    stored_block = captured["blocks"][0]
    assert result.block_count == 1
    assert stored_block.raw_text == raw_text
    assert stored_block.raw_text == extracted_block.raw_text
    assert stored_block.anchor_hash == extracted_block.anchor_hash
    assert session.commit_called is True


def test_persist_extraction_result_rejects_non_continuous_block_order() -> None:
    session = FakeSession()
    extracted_document = build_extracted_document(
        build_extracted_block(source_order=1, raw_text="alpha"),
    )

    with pytest.raises(document_content_service.InvalidExtractionResultError):
        run_async(
            document_content_service.persist_extraction_result(
                session,
                revision_id=uuid.uuid4(),
                extracted_document=extracted_document,
                extractor_name="deterministic-extractor",
                extractor_version="1.0.0",
            )
        )

    assert session.commit_called is False


def test_persist_extraction_result_rejects_count_mismatch_without_writing(monkeypatch) -> None:
    session = FakeSession()
    extracted_document = build_extracted_document(
        build_extracted_block(source_order=0, raw_text="alpha"),
    )
    extracted_document = extracted_document.model_copy(update={"block_count": 2})
    called = {"create_run": False}

    async def fake_create_extraction_run(_session, _run):
        called["create_run"] = True
        return _run

    monkeypatch.setattr(
        document_content_service.document_content_repository,
        "create_extraction_run",
        fake_create_extraction_run,
    )

    with pytest.raises(document_content_service.InvalidExtractionResultError):
        run_async(
            document_content_service.persist_extraction_result(
                session,
                revision_id=uuid.uuid4(),
                extracted_document=extracted_document,
                extractor_name="deterministic-extractor",
                extractor_version="1.0.0",
            )
        )

    assert called["create_run"] is False


def test_persist_extraction_result_allows_needs_ocr_without_blocks(monkeypatch) -> None:
    session = FakeSession()
    revision = DocumentRevision(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        revision_no=1,
        upload_intent="new_document",
        supersedes_revision_id=None,
        original_filename="scan.pdf",
        storage_key="files/aa/a.bin",
        mime_type="application/pdf",
        file_size_bytes=10,
        sha256="a" * 64,
        source_authority="pending",
        status="uploaded",
        uploaded_by_id=uuid.uuid4(),
        uploaded_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    async def fake_get_revision_for_extraction_update(_session, _revision_id):
        return revision

    async def fake_get_next_extraction_attempt_no(_session, _revision_id):
        return 1

    async def fake_create_extraction_run(_session, extraction_run):
        return extraction_run

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

    result = run_async(
        document_content_service.persist_extraction_result(
            session,
            revision_id=revision.id,
            extracted_document=ExtractedDocument(
                outcome=ExtractionOutcome.NEEDS_OCR,
                detected_format=DetectedFileFormat.PDF,
                detected_encoding=None,
                page_count=1,
                character_count=0,
                block_count=0,
                blocks=[],
                warnings=["needs ocr"],
                metadata={},
            ),
            extractor_name="deterministic-extractor",
            extractor_version="1.0.0",
        )
    )

    assert result.outcome == "needs_ocr"
    assert session.commit_called is True


def test_create_source_evidence_excerpt_matches_raw_text_slice(monkeypatch) -> None:
    session = FakeSession()
    block = DocumentBlock(
        id=uuid.uuid4(),
        extraction_run_id=uuid.uuid4(),
        source_order=0,
        block_type="paragraph",
        raw_text=" a b \n",
        normalized_text=" a b \n",
        location_key="loc-0",
        anchor_hash="a" * 64,
        block_index=0,
        heading_path=[],
        block_metadata={},
    )
    captured: dict[str, object] = {}

    async def fake_get_document_block_by_id(_session, _block_id):
        return block

    async def fake_get_source_evidence_by_offsets(_session, _block_id, _start_offset, _end_offset):
        return None

    async def fake_create_source_evidence(_session, evidence):
        captured["evidence"] = evidence
        return evidence

    monkeypatch.setattr(
        document_content_service.document_content_repository,
        "get_document_block_by_id",
        fake_get_document_block_by_id,
    )
    monkeypatch.setattr(
        document_content_service.document_content_repository,
        "get_source_evidence_by_offsets",
        fake_get_source_evidence_by_offsets,
    )
    monkeypatch.setattr(
        document_content_service.document_content_repository,
        "create_source_evidence",
        fake_create_source_evidence,
    )

    result = run_async(
        document_content_service.create_source_evidence(
            session,
            SourceEvidenceCreate(block_id=block.id, start_offset=0, end_offset=5),
        )
    )

    assert result.excerpt == " a b "
    assert result.excerpt == block.raw_text[0:5]
    assert result.excerpt_hash == hashlib.sha256(" a b ".encode("utf-8")).hexdigest()
    assert captured["evidence"].excerpt == " a b "
    assert session.commit_called is True


def test_create_source_evidence_rejects_out_of_bounds_excerpt(monkeypatch) -> None:
    session = FakeSession()
    block = DocumentBlock(
        id=uuid.uuid4(),
        extraction_run_id=uuid.uuid4(),
        source_order=0,
        block_type="paragraph",
        raw_text="abc",
        normalized_text="abc",
        location_key="loc-0",
        anchor_hash="a" * 64,
        block_index=0,
        heading_path=[],
        block_metadata={},
    )

    async def fake_get_document_block_by_id(_session, _block_id):
        return block

    monkeypatch.setattr(
        document_content_service.document_content_repository,
        "get_document_block_by_id",
        fake_get_document_block_by_id,
    )

    with pytest.raises(document_content_service.EvidenceOffsetError):
        run_async(
            document_content_service.create_source_evidence(
                session,
                SourceEvidenceCreate(block_id=block.id, start_offset=0, end_offset=10),
            )
        )


def test_create_source_evidence_rejects_empty_excerpt_after_service_validation(monkeypatch) -> None:
    session = FakeSession()
    block = DocumentBlock(
        id=uuid.uuid4(),
        extraction_run_id=uuid.uuid4(),
        source_order=0,
        block_type="paragraph",
        raw_text="abc",
        normalized_text="abc",
        location_key="loc-0",
        anchor_hash="a" * 64,
        block_index=0,
        heading_path=[],
        block_metadata={},
    )

    async def fake_get_document_block_by_id(_session, _block_id):
        return block

    monkeypatch.setattr(
        document_content_service.document_content_repository,
        "get_document_block_by_id",
        fake_get_document_block_by_id,
    )

    with pytest.raises(document_content_service.EvidenceOffsetError):
        run_async(
            document_content_service.create_source_evidence(
                session,
                SourceEvidenceCreate.model_construct(
                    block_id=block.id,
                    start_offset=1,
                    end_offset=1,
                ),
            )
        )


def test_create_source_evidence_returns_existing_record(monkeypatch) -> None:
    session = FakeSession()
    block = DocumentBlock(
        id=uuid.uuid4(),
        extraction_run_id=uuid.uuid4(),
        source_order=0,
        block_type="paragraph",
        raw_text="abcdef",
        normalized_text="abcdef",
        location_key="loc-0",
        anchor_hash="a" * 64,
        block_index=0,
        heading_path=[],
        block_metadata={},
    )
    existing = SourceEvidence(
        id=uuid.uuid4(),
        block_id=block.id,
        start_offset=1,
        end_offset=4,
        excerpt="bcd",
        excerpt_hash=hashlib.sha256("bcd".encode("utf-8")).hexdigest(),
    )
    called = {"create": False}

    async def fake_get_document_block_by_id(_session, _block_id):
        return block

    async def fake_get_source_evidence_by_offsets(_session, _block_id, _start_offset, _end_offset):
        return existing

    async def fake_create_source_evidence(_session, _evidence):
        called["create"] = True
        return _evidence

    monkeypatch.setattr(
        document_content_service.document_content_repository,
        "get_document_block_by_id",
        fake_get_document_block_by_id,
    )
    monkeypatch.setattr(
        document_content_service.document_content_repository,
        "get_source_evidence_by_offsets",
        fake_get_source_evidence_by_offsets,
    )
    monkeypatch.setattr(
        document_content_service.document_content_repository,
        "create_source_evidence",
        fake_create_source_evidence,
    )

    result = run_async(
        document_content_service.create_source_evidence(
            session,
            SourceEvidenceCreate(block_id=block.id, start_offset=1, end_offset=4),
        )
    )

    assert result is existing
    assert called["create"] is False
    assert session.commit_called is False


def test_create_source_evidence_returns_existing_record_after_integrity_error(monkeypatch) -> None:
    session = FakeSession()
    block = DocumentBlock(
        id=uuid.uuid4(),
        extraction_run_id=uuid.uuid4(),
        source_order=0,
        block_type="paragraph",
        raw_text="  abc  ",
        normalized_text="  abc  ",
        location_key="loc-0",
        anchor_hash="a" * 64,
        block_index=0,
        heading_path=[],
        block_metadata={},
    )
    existing = SourceEvidence(
        id=uuid.uuid4(),
        block_id=block.id,
        start_offset=1,
        end_offset=6,
        excerpt=" abc ",
        excerpt_hash=hashlib.sha256(" abc ".encode("utf-8")).hexdigest(),
    )
    lookup_count = {"count": 0}

    async def fake_get_document_block_by_id(_session, _block_id):
        return block

    async def fake_get_source_evidence_by_offsets(_session, _block_id, _start_offset, _end_offset):
        lookup_count["count"] += 1
        if lookup_count["count"] == 1:
            return None
        return existing

    async def fake_create_source_evidence(_session, _evidence):
        raise IntegrityError("insert", {}, Exception("duplicate key"))

    monkeypatch.setattr(
        document_content_service.document_content_repository,
        "get_document_block_by_id",
        fake_get_document_block_by_id,
    )
    monkeypatch.setattr(
        document_content_service.document_content_repository,
        "get_source_evidence_by_offsets",
        fake_get_source_evidence_by_offsets,
    )
    monkeypatch.setattr(
        document_content_service.document_content_repository,
        "create_source_evidence",
        fake_create_source_evidence,
    )

    result = run_async(
        document_content_service.create_source_evidence(
            session,
            SourceEvidenceCreate(block_id=block.id, start_offset=1, end_offset=6),
        )
    )

    assert result is existing
    assert session.rollback_called is True
    assert session.commit_called is False
    assert lookup_count["count"] == 2


def test_create_source_evidence_reraises_non_duplicate_integrity_error(monkeypatch) -> None:
    session = FakeSession()
    block = DocumentBlock(
        id=uuid.uuid4(),
        extraction_run_id=uuid.uuid4(),
        source_order=0,
        block_type="paragraph",
        raw_text="abcdef",
        normalized_text="abcdef",
        location_key="loc-0",
        anchor_hash="a" * 64,
        block_index=0,
        heading_path=[],
        block_metadata={},
    )

    async def fake_get_document_block_by_id(_session, _block_id):
        return block

    async def fake_get_source_evidence_by_offsets(_session, _block_id, _start_offset, _end_offset):
        return None

    original_error = IntegrityError("insert", {}, Exception("connection lost"))

    async def fake_create_source_evidence(_session, _evidence):
        raise original_error

    monkeypatch.setattr(
        document_content_service.document_content_repository,
        "get_document_block_by_id",
        fake_get_document_block_by_id,
    )
    monkeypatch.setattr(
        document_content_service.document_content_repository,
        "get_source_evidence_by_offsets",
        fake_get_source_evidence_by_offsets,
    )
    monkeypatch.setattr(
        document_content_service.document_content_repository,
        "create_source_evidence",
        fake_create_source_evidence,
    )

    with pytest.raises(IntegrityError) as exc_info:
        run_async(
            document_content_service.create_source_evidence(
                session,
                SourceEvidenceCreate(block_id=block.id, start_offset=1, end_offset=4),
            )
        )

    assert exc_info.value is original_error
    assert session.rollback_called is True
    assert session.commit_called is False
