from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document_content import (
    DocumentBlock,
    ExtractionRun,
    ExtractionRunOutcome,
    ExtractionRunStatus,
    SourceEvidence,
)
from app.repositories import document_content as document_content_repository
from app.schemas.document_content import SourceEvidenceCreate
from app.schemas.document_extraction import ExtractedDocument, ExtractionOutcome


class ExtractionPersistenceError(Exception):
    """Raised when extraction persistence cannot be completed."""


class ExtractionRevisionNotFoundError(ExtractionPersistenceError):
    """Raised when the target revision does not exist."""


class InvalidExtractionResultError(ExtractionPersistenceError):
    """Raised when extracted document data is internally inconsistent."""


class DocumentBlockNotFoundError(ExtractionPersistenceError):
    """Raised when the target block does not exist."""


class EvidenceOffsetError(ExtractionPersistenceError):
    """Raised when evidence offsets are invalid."""


async def persist_extraction_result(
    session: AsyncSession,
    *,
    revision_id: uuid.UUID,
    extracted_document: ExtractedDocument,
    extractor_name: str,
    extractor_version: str,
    failure_code: str | None = None,
    failure_message: str | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    commit: bool = True,
) -> ExtractionRun:
    try:
        extraction_run = await persist_extraction_result_in_transaction(
            session,
            revision_id=revision_id,
            extracted_document=extracted_document,
            extractor_name=extractor_name,
            extractor_version=extractor_version,
            failure_code=failure_code,
            failure_message=failure_message,
            started_at=started_at,
            completed_at=completed_at,
        )
        if commit:
            await session.commit()
        return extraction_run
    except BaseException:
        if commit:
            await session.rollback()
        raise


async def persist_extraction_result_in_transaction(
    session: AsyncSession,
    *,
    revision_id: uuid.UUID,
    extracted_document: ExtractedDocument,
    extractor_name: str,
    extractor_version: str,
    failure_code: str | None = None,
    failure_message: str | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
) -> ExtractionRun:
    _validate_extracted_document(extracted_document)

    revision = await document_content_repository.get_revision_for_extraction_update(session, revision_id)
    if revision is None:
        raise ExtractionRevisionNotFoundError("Target revision not found.")

    attempt_no = await document_content_repository.get_next_extraction_attempt_no(session, revision_id)
    run_status = (
        ExtractionRunStatus.FAILED.value
        if extracted_document.outcome == ExtractionOutcome.FAILED
        else ExtractionRunStatus.COMPLETED.value
    )
    extraction_run = ExtractionRun(
        revision_id=revision_id,
        attempt_no=attempt_no,
        status=run_status,
        outcome=extracted_document.outcome.value,
        extractor_name=extractor_name,
        extractor_version=extractor_version,
        detected_format=extracted_document.detected_format.value,
        detected_encoding=extracted_document.detected_encoding,
        page_count=extracted_document.page_count,
        character_count=extracted_document.character_count,
        block_count=extracted_document.block_count,
        warnings=list(extracted_document.warnings),
        content_metadata=dict(extracted_document.metadata),
        failure_code=failure_code,
        failure_message=failure_message,
        started_at=started_at,
        completed_at=completed_at,
    )

    blocks = [
        DocumentBlock(
            extraction_run_id=extraction_run.id,
            source_order=block.source_order,
            block_type=block.block_type.value,
            raw_text=block.raw_text,
            normalized_text=block.normalized_text,
            location_key=block.location_key,
            anchor_hash=block.anchor_hash,
            page_no=block.page_no,
            block_index=block.block_index,
            heading_level=block.heading_level,
            heading_path=list(block.heading_path),
            start_line=block.start_line,
            end_line=block.end_line,
            table_index=block.table_index,
            row_index=block.row_index,
            block_metadata=dict(block.metadata),
        )
        for block in extracted_document.blocks
    ]

    await document_content_repository.create_extraction_run(session, extraction_run)
    if blocks:
        await document_content_repository.create_document_blocks(session, blocks)
    extraction_run.blocks.extend(blocks)

    return extraction_run


async def create_source_evidence(
    session: AsyncSession,
    payload: SourceEvidenceCreate,
) -> SourceEvidence:
    block = await document_content_repository.get_document_block_by_id(session, payload.block_id)
    if block is None:
        raise DocumentBlockNotFoundError("Target block not found.")

    _validate_evidence_offsets(block.raw_text, payload.start_offset, payload.end_offset)
    excerpt = block.raw_text[payload.start_offset : payload.end_offset]
    if excerpt == "":
        raise EvidenceOffsetError("Evidence excerpt must not be empty.")

    existing = await document_content_repository.get_source_evidence_by_offsets(
        session,
        payload.block_id,
        payload.start_offset,
        payload.end_offset,
    )
    if existing is not None:
        return existing

    evidence = SourceEvidence(
        block_id=payload.block_id,
        start_offset=payload.start_offset,
        end_offset=payload.end_offset,
        excerpt=excerpt,
        excerpt_hash=_hash_text(excerpt),
    )

    try:
        await document_content_repository.create_source_evidence(session, evidence)
        await session.commit()
    except IntegrityError:
        await session.rollback()
        existing = await document_content_repository.get_source_evidence_by_offsets(
            session,
            payload.block_id,
            payload.start_offset,
            payload.end_offset,
        )
        if existing is not None:
            return existing
        raise
    except Exception:
        await session.rollback()
        raise

    return evidence


def _validate_extracted_document(extracted_document: ExtractedDocument) -> None:
    blocks = extracted_document.blocks
    expected_source_order = list(range(len(blocks)))
    actual_source_order = [block.source_order for block in blocks]
    if actual_source_order != expected_source_order:
        raise InvalidExtractionResultError("Block source_order must be continuous from 0.")

    if len({block.location_key for block in blocks}) != len(blocks):
        raise InvalidExtractionResultError("Block location_key values must be unique.")

    if len({block.anchor_hash for block in blocks}) != len(blocks):
        raise InvalidExtractionResultError("Block anchor_hash values must be unique.")

    detected_format = extracted_document.detected_format.value
    for block in blocks:
        expected_anchor_hash = _build_anchor_hash(
            detected_format=detected_format,
            location_key=block.location_key,
            raw_text=block.raw_text,
        )
        if block.anchor_hash != expected_anchor_hash:
            raise InvalidExtractionResultError("Block anchor_hash does not match the expected deterministic hash.")

    if extracted_document.block_count != len(blocks):
        raise InvalidExtractionResultError("Extracted block_count does not match blocks length.")

    computed_character_count = sum(len(block.normalized_text) for block in blocks)
    if extracted_document.character_count != computed_character_count:
        raise InvalidExtractionResultError("Extracted character_count does not match normalized block text.")

    if (
        extracted_document.outcome not in {ExtractionOutcome.FAILED, ExtractionOutcome.NEEDS_OCR}
        and len(blocks) == 0
    ):
        raise InvalidExtractionResultError("Successful extraction results must contain at least one block.")


def _validate_evidence_offsets(raw_text: str, start_offset: int, end_offset: int) -> None:
    if start_offset < 0:
        raise EvidenceOffsetError("start_offset must be greater than or equal to 0.")
    if end_offset <= start_offset:
        raise EvidenceOffsetError("end_offset must be greater than start_offset.")
    if end_offset > len(raw_text):
        raise EvidenceOffsetError("Evidence offsets exceed block raw_text length.")


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _build_anchor_hash(*, detected_format: str, location_key: str, raw_text: str) -> str:
    return hashlib.sha256(
        f"{detected_format}|{location_key}|{raw_text}".encode("utf-8")
    ).hexdigest()
