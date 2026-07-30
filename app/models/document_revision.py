from __future__ import annotations

from datetime import datetime
import uuid
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, CheckConstraint, DateTime, ForeignKey, String, Text, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.models.base import Base, UUIDPrimaryKeyMixin, utc_now
from app.utils.validation import normalize_text

if TYPE_CHECKING:
    from app.models.document_content import ExtractionRun
    from app.models.document import Document
    from app.models.ingestion_validation import IngestionValidationReport
    from app.models.processing_job import ProcessingJob
    from app.models.user import User


class UploadIntent(StrEnum):
    NEW_DOCUMENT = "new_document"
    REVISION = "revision"


class SourceAuthority(StrEnum):
    PENDING = "pending"
    FORMAL = "formal"
    REFERENCE = "reference"


class DocumentRevisionStatus(StrEnum):
    UPLOADED = "uploaded"
    PREFLIGHT_CHECKING = "preflight_checking"
    REJECTED = "rejected"
    NEEDS_CONFIRMATION = "needs_confirmation"
    QUARANTINED = "quarantined"
    ACCEPTED = "accepted"
    PARSING = "parsing"
    EXTRACTING = "extracting"
    AWAITING_REVIEW = "awaiting_review"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentRevision(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "document_revisions"
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "revision_no",
            name="uq_document_revisions_document_id_revision_no",
        ),
        UniqueConstraint("storage_key", name="uq_document_revisions_storage_key"),
        UniqueConstraint(
            "supersedes_revision_id",
            name="uq_document_revisions_supersedes_revision_id",
        ),
        CheckConstraint("revision_no > 0", name="document_revisions_revision_no_positive"),
        CheckConstraint(
            "((revision_no = 1 AND upload_intent = 'new_document') OR (revision_no > 1 AND upload_intent = 'revision'))",
            name="document_revisions_revision_intent_valid",
        ),
        CheckConstraint(
            "((revision_no = 1 AND supersedes_revision_id IS NULL) OR (revision_no > 1 AND supersedes_revision_id IS NOT NULL))",
            name="document_revisions_revision_supersedes_required",
        ),
        CheckConstraint(
            "char_length(original_filename) BETWEEN 1 AND 255",
            name="document_revisions_original_filename_length",
        ),
        CheckConstraint(
            "char_length(storage_key) BETWEEN 1 AND 255",
            name="document_revisions_storage_key_length",
        ),
        CheckConstraint(
            "storage_key <> original_filename",
            name="document_revisions_storage_key_not_filename",
        ),
        CheckConstraint(
            "sha256 ~ '^[0-9a-f]{64}$'",
            name="document_revisions_sha256_format",
        ),
        CheckConstraint("file_size_bytes > 0", name="document_revisions_file_size_positive"),
        CheckConstraint(
            "language_confidence IS NULL OR (language_confidence >= 0 AND language_confidence <= 1)",
            name="document_revisions_language_confidence_range",
        ),
        CheckConstraint(
            "upload_intent IN ('new_document', 'revision')",
            name="document_revisions_upload_intent_valid",
        ),
        CheckConstraint(
            "source_authority IN ('pending', 'formal', 'reference')",
            name="document_revisions_source_authority_valid",
        ),
        CheckConstraint(
            "status IN ('uploaded', 'preflight_checking', 'rejected', 'needs_confirmation', 'quarantined', 'accepted', 'parsing', 'extracting', 'awaiting_review', 'completed', 'failed')",
            name="document_revisions_status_valid",
        ),
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    revision_no: Mapped[int] = mapped_column(nullable=False)
    upload_intent: Mapped[str] = mapped_column(String(16), nullable=False)
    supersedes_revision_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("document_revisions.id", ondelete="RESTRICT"),
        nullable=True,
    )
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(255), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(255), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    detected_language: Mapped[str | None] = mapped_column(String(32), nullable=True)
    language_confidence: Mapped[float | None] = mapped_column(nullable=True)
    source_authority: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=SourceAuthority.PENDING.value,
        server_default=SourceAuthority.PENDING.value,
    )
    status: Mapped[str] = mapped_column(
        String(24),
        nullable=False,
        default=DocumentRevisionStatus.UPLOADED.value,
        server_default=DocumentRevisionStatus.UPLOADED.value,
    )
    uploaded_by_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    document: Mapped["Document"] = relationship(
        back_populates="revisions",
        foreign_keys=[document_id],
    )
    current_for_document: Mapped["Document | None"] = relationship(
        back_populates="current_revision",
        foreign_keys="Document.current_revision_id",
        uselist=False,
    )
    supersedes_revision: Mapped["DocumentRevision | None"] = relationship(
        back_populates="superseded_by_revision",
        remote_side="DocumentRevision.id",
        foreign_keys=[supersedes_revision_id],
        uselist=False,
    )
    superseded_by_revision: Mapped["DocumentRevision | None"] = relationship(
        back_populates="supersedes_revision",
        uselist=False,
    )
    uploaded_by: Mapped["User"] = relationship(
        back_populates="uploaded_revisions",
        foreign_keys=[uploaded_by_id],
    )
    validation_reports: Mapped[list["IngestionValidationReport"]] = relationship(
        back_populates="revision",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    extraction_runs: Mapped[list["ExtractionRun"]] = relationship(
        back_populates="revision",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    processing_jobs: Mapped[list["ProcessingJob"]] = relationship(
        back_populates="revision",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    @validates("original_filename", "storage_key", "mime_type", "detected_language")
    def normalize_optional_text_field(self, _key: str, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized
