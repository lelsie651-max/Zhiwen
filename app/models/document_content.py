from __future__ import annotations

from datetime import datetime
import uuid
from enum import StrEnum
import re
from typing import TYPE_CHECKING, Any

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.models.base import Base, UUIDPrimaryKeyMixin, utc_now
from app.utils.validation import normalize_text

if TYPE_CHECKING:
    from app.models.document_revision import DocumentRevision
    from app.models.fact import FactEvidenceLink, FactValue
    from app.models.processing_job import ProcessingJob


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ExtractionRunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ExtractionRunOutcome(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    NEEDS_OCR = "needs_ocr"
    FAILED = "failed"


class DocumentBlockType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE_ROW = "table_row"
    CODE = "code"
    PAGE_TEXT = "page_text"


class ExtractionRun(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "extraction_runs"
    __table_args__ = (
        UniqueConstraint(
            "revision_id",
            "attempt_no",
            name="uq_extraction_runs_revision_id_attempt_no",
        ),
        CheckConstraint("attempt_no > 0", name="extraction_runs_attempt_no_positive"),
        CheckConstraint(
            "status IN ('running', 'completed', 'failed')",
            name="extraction_runs_status_valid",
        ),
        CheckConstraint(
            "outcome IN ('success', 'partial', 'needs_ocr', 'failed')",
            name="extraction_runs_outcome_valid",
        ),
        CheckConstraint("character_count >= 0", name="extraction_runs_character_count_non_negative"),
        CheckConstraint("block_count >= 0", name="extraction_runs_block_count_non_negative"),
        CheckConstraint(
            "completed_at IS NULL OR started_at IS NULL OR completed_at >= started_at",
            name="extraction_runs_completed_after_started",
        ),
    )

    revision_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document_revisions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    attempt_no: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    outcome: Mapped[str] = mapped_column(String(16), nullable=False)
    extractor_name: Mapped[str] = mapped_column(String(100), nullable=False)
    extractor_version: Mapped[str] = mapped_column(String(32), nullable=False)
    detected_format: Mapped[str] = mapped_column(String(16), nullable=False)
    detected_encoding: Mapped[str | None] = mapped_column(String(64), nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    character_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    block_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    warnings: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )
    content_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    failure_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    failure_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    revision: Mapped["DocumentRevision"] = relationship(
        back_populates="extraction_runs",
        foreign_keys=[revision_id],
    )
    blocks: Mapped[list["DocumentBlock"]] = relationship(
        back_populates="extraction_run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    fact_values: Mapped[list["FactValue"]] = relationship(
        back_populates="extraction_run",
    )
    processing_jobs: Mapped[list["ProcessingJob"]] = relationship(
        back_populates="result_extraction_run",
        foreign_keys="ProcessingJob.result_extraction_run_id",
    )

    @validates(
        "extractor_name",
        "extractor_version",
        "detected_format",
        "detected_encoding",
        "failure_code",
        "failure_message",
    )
    def normalize_optional_text_field(self, _key: str, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized


class DocumentBlock(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "document_blocks"
    __table_args__ = (
        UniqueConstraint(
            "extraction_run_id",
            "source_order",
            name="uq_document_blocks_extraction_run_id_source_order",
        ),
        UniqueConstraint(
            "extraction_run_id",
            "location_key",
            name="uq_document_blocks_extraction_run_id_location_key",
        ),
        UniqueConstraint(
            "extraction_run_id",
            "anchor_hash",
            name="uq_document_blocks_extraction_run_id_anchor_hash",
        ),
        CheckConstraint(
            "block_type IN ('heading', 'paragraph', 'list_item', 'table_row', 'code', 'page_text')",
            name="document_blocks_block_type_valid",
        ),
        CheckConstraint("source_order >= 0", name="document_blocks_source_order_non_negative"),
        CheckConstraint("block_index >= 0", name="document_blocks_block_index_non_negative"),
        CheckConstraint("char_length(raw_text) > 0", name="document_blocks_raw_text_non_empty"),
        CheckConstraint(
            "char_length(normalized_text) > 0",
            name="document_blocks_normalized_text_non_empty",
        ),
        CheckConstraint("char_length(location_key) > 0", name="document_blocks_location_key_non_empty"),
        CheckConstraint(
            "anchor_hash ~ '^[0-9a-f]{64}$'",
            name="document_blocks_anchor_hash_format",
        ),
        CheckConstraint("page_no IS NULL OR page_no > 0", name="document_blocks_page_no_positive"),
        CheckConstraint("heading_level IS NULL OR heading_level > 0", name="document_blocks_heading_level_positive"),
        CheckConstraint("start_line IS NULL OR start_line > 0", name="document_blocks_start_line_positive"),
        CheckConstraint("end_line IS NULL OR end_line > 0", name="document_blocks_end_line_positive"),
        CheckConstraint("table_index IS NULL OR table_index > 0", name="document_blocks_table_index_positive"),
        CheckConstraint("row_index IS NULL OR row_index > 0", name="document_blocks_row_index_positive"),
    )

    extraction_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("extraction_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_order: Mapped[int] = mapped_column(Integer, nullable=False)
    block_type: Mapped[str] = mapped_column(String(16), nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_text: Mapped[str] = mapped_column(Text, nullable=False)
    location_key: Mapped[str] = mapped_column(String(255), nullable=False)
    anchor_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    page_no: Mapped[int | None] = mapped_column(Integer, nullable=True)
    block_index: Mapped[int] = mapped_column(Integer, nullable=False)
    heading_level: Mapped[int | None] = mapped_column(Integer, nullable=True)
    heading_path: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )
    start_line: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_line: Mapped[int | None] = mapped_column(Integer, nullable=True)
    table_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    row_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    block_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    extraction_run: Mapped["ExtractionRun"] = relationship(
        back_populates="blocks",
        foreign_keys=[extraction_run_id],
    )
    evidences: Mapped[list["SourceEvidence"]] = relationship(
        back_populates="block",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    @validates("raw_text", "normalized_text")
    def validate_block_text_field(self, _key: str, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("value must be a string")
        if not value.strip():
            raise ValueError("value must not be empty")
        return value

    @validates("location_key")
    def validate_location_key(self, _key: str, value: str) -> str:
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @validates("anchor_hash")
    def validate_anchor_hash(self, _key: str, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("value must be a string")
        normalized = value.lower()
        if not SHA256_PATTERN.fullmatch(normalized):
            raise ValueError("anchor_hash must be a 64-character lowercase hexadecimal string")
        return normalized


class SourceEvidence(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "source_evidences"
    __table_args__ = (
        UniqueConstraint(
            "block_id",
            "start_offset",
            "end_offset",
            name="uq_source_evidences_block_id_start_offset_end_offset",
        ),
        CheckConstraint("start_offset >= 0", name="source_evidences_start_offset_non_negative"),
        CheckConstraint("end_offset > start_offset", name="source_evidences_end_offset_after_start"),
        CheckConstraint("char_length(excerpt) > 0", name="source_evidences_excerpt_non_empty"),
        CheckConstraint(
            "excerpt_hash ~ '^[0-9a-f]{64}$'",
            name="source_evidences_excerpt_hash_format",
        ),
    )

    block_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document_blocks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    start_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    end_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    excerpt: Mapped[str] = mapped_column(Text, nullable=False)
    excerpt_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    block: Mapped["DocumentBlock"] = relationship(
        back_populates="evidences",
        foreign_keys=[block_id],
    )
    fact_evidence_links: Mapped[list["FactEvidenceLink"]] = relationship(
        back_populates="evidence",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    @validates("excerpt")
    def validate_excerpt(self, _key: str, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("value must be a string")
        if not value.strip():
            raise ValueError("value must not be empty")
        return value

    @validates("excerpt_hash")
    def validate_excerpt_hash(self, _key: str, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("value must be a string")
        normalized = value.lower()
        if not SHA256_PATTERN.fullmatch(normalized):
            raise ValueError("excerpt_hash must be a 64-character lowercase hexadecimal string")
        return normalized
