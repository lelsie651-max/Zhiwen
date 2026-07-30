from __future__ import annotations

from datetime import datetime
import uuid
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.utils.validation import normalize_text

if TYPE_CHECKING:
    from app.models.document_content import ExtractionRun
    from app.models.document_revision import DocumentRevision
    from app.models.project import Project
    from app.models.user import User


class ProcessingJobType(StrEnum):
    REVISION_EXTRACTION = "revision_extraction"


class ProcessingJobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingTriggerKind(StrEnum):
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    RETRY = "retry"
    RECOVERY = "recovery"


ACTIVE_PROCESSING_JOB_STATUSES = (
    ProcessingJobStatus.QUEUED.value,
    ProcessingJobStatus.RUNNING.value,
)
ACTIVE_PROCESSING_JOB_INDEX_PREDICATE = "status IN ('queued','running')"


class ProcessingJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "processing_jobs"
    __table_args__ = (
        UniqueConstraint(
            "revision_id",
            "job_type",
            "attempt_no",
            name="uq_processing_jobs_revision_id_job_type_attempt_no",
        ),
        CheckConstraint("attempt_no > 0", name="processing_jobs_attempt_no_positive"),
        CheckConstraint(
            "job_type IN ('revision_extraction')",
            name="processing_jobs_job_type_valid",
        ),
        CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'failed', 'cancelled')",
            name="processing_jobs_status_valid",
        ),
        CheckConstraint(
            "trigger_kind IN ('automatic', 'manual', 'retry', 'recovery')",
            name="processing_jobs_trigger_kind_valid",
        ),
        CheckConstraint(
            "(trigger_kind NOT IN ('manual', 'retry') OR requested_by_id IS NOT NULL)",
            name="processing_jobs_manual_retry_requires_requester",
        ),
        CheckConstraint(
            "lease_expires_at IS NULL OR started_at IS NULL OR lease_expires_at >= started_at",
            name="processing_jobs_lease_after_started",
        ),
        CheckConstraint(
            "completed_at IS NULL OR started_at IS NULL OR completed_at >= started_at",
            name="processing_jobs_completed_after_started",
        ),
        CheckConstraint(
            "("
            "status <> 'queued' OR "
            "("
            "lease_token IS NULL AND lease_expires_at IS NULL AND started_at IS NULL AND "
            "completed_at IS NULL AND result_extraction_run_id IS NULL AND "
            "failure_code IS NULL AND failure_message IS NULL"
            ")"
            ")",
            name="processing_jobs_queued_state_shape",
        ),
        CheckConstraint(
            "("
            "status <> 'running' OR "
            "("
            "lease_token IS NOT NULL AND lease_expires_at IS NOT NULL AND started_at IS NOT NULL AND "
            "completed_at IS NULL AND result_extraction_run_id IS NULL AND "
            "failure_code IS NULL AND failure_message IS NULL"
            ")"
            ")",
            name="processing_jobs_running_state_shape",
        ),
        CheckConstraint(
            "("
            "status <> 'completed' OR "
            "("
            "lease_token IS NULL AND lease_expires_at IS NULL AND started_at IS NOT NULL AND "
            "completed_at IS NOT NULL AND result_extraction_run_id IS NOT NULL AND "
            "failure_code IS NULL AND failure_message IS NULL"
            ")"
            ")",
            name="processing_jobs_completed_state_shape",
        ),
        CheckConstraint(
            "("
            "status <> 'failed' OR "
            "("
            "lease_token IS NULL AND lease_expires_at IS NULL AND started_at IS NOT NULL AND "
            "completed_at IS NOT NULL AND result_extraction_run_id IS NULL AND "
            "failure_code IS NOT NULL"
            ")"
            ")",
            name="processing_jobs_failed_state_shape",
        ),
        CheckConstraint(
            "("
            "status <> 'cancelled' OR "
            "("
            "lease_token IS NULL AND lease_expires_at IS NULL AND completed_at IS NOT NULL AND "
            "result_extraction_run_id IS NULL AND failure_code IS NULL AND failure_message IS NULL"
            ")"
            ")",
            name="processing_jobs_cancelled_state_shape",
        ),
        Index(
            "uq_processing_jobs_active_revision_id_job_type",
            "revision_id",
            "job_type",
            unique=True,
            postgresql_where=text(ACTIVE_PROCESSING_JOB_INDEX_PREDICATE),
            sqlite_where=text(ACTIVE_PROCESSING_JOB_INDEX_PREDICATE),
        ),
    )

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    revision_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document_revisions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    job_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    trigger_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    attempt_no: Mapped[int] = mapped_column(Integer, nullable=False)
    requested_by_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    lease_token: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    result_extraction_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("extraction_runs.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    failure_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    failure_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    project: Mapped["Project"] = relationship(
        back_populates="processing_jobs",
        foreign_keys=[project_id],
    )
    revision: Mapped["DocumentRevision"] = relationship(
        back_populates="processing_jobs",
        foreign_keys=[revision_id],
    )
    requested_by: Mapped["User | None"] = relationship(
        back_populates="requested_processing_jobs",
        foreign_keys=[requested_by_id],
    )
    result_extraction_run: Mapped["ExtractionRun | None"] = relationship(
        back_populates="processing_jobs",
        foreign_keys=[result_extraction_run_id],
    )

    @validates("failure_code", "failure_message")
    def normalize_optional_text_field(self, _key: str, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized
