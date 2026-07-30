from __future__ import annotations

from datetime import datetime
import uuid
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, utc_now
from app.utils.validation import normalize_text

if TYPE_CHECKING:
    from app.models.document_revision import DocumentRevision
    from app.models.user import User


class ValidationReportStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ValidationReportOutcome(StrEnum):
    PENDING = "pending"
    PASS = "pass"
    WARNING = "warning"
    REJECT = "reject"
    QUARANTINE = "quarantine"


class ValidationRecommendedAuthority(StrEnum):
    FORMAL = "formal"
    REFERENCE = "reference"


class ValidationUserDecision(StrEnum):
    CONTINUE_FORMAL = "continue_formal"
    CONTINUE_REFERENCE = "continue_reference"
    CANCEL = "cancel"


class IngestionRiskCategory(StrEnum):
    FILE = "file"
    TEXT_QUALITY = "text_quality"
    CONTENT = "content"
    PRIVACY = "privacy"
    CREDENTIAL = "credential"
    PROMPT_INJECTION = "prompt_injection"
    SECURITY = "security"
    OTHER = "other"


class IngestionRiskSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    BLOCKER = "blocker"


class IngestionRiskDetector(StrEnum):
    DETERMINISTIC = "deterministic"
    AI = "ai"
    MANUAL = "manual"


class IngestionValidationReport(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "ingestion_validation_reports"
    __table_args__ = (
        UniqueConstraint(
            "revision_id",
            "attempt_no",
            name="uq_ingestion_validation_reports_revision_id_attempt_no",
        ),
        CheckConstraint("attempt_no > 0", name="ingestion_validation_reports_attempt_no_positive"),
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name="ingestion_validation_reports_status_valid",
        ),
        CheckConstraint(
            "outcome IN ('pending', 'pass', 'warning', 'reject', 'quarantine')",
            name="ingestion_validation_reports_outcome_valid",
        ),
        CheckConstraint(
            "recommended_authority IS NULL OR recommended_authority IN ('formal', 'reference')",
            name="ingestion_validation_reports_recommended_authority_valid",
        ),
        CheckConstraint(
            "user_decision IS NULL OR user_decision IN ('continue_formal', 'continue_reference', 'cancel')",
            name="ingestion_validation_reports_user_decision_valid",
        ),
        CheckConstraint(
            "content_confidence IS NULL OR (content_confidence >= 0 AND content_confidence <= 1)",
            name="ingestion_validation_reports_content_confidence_range",
        ),
        CheckConstraint(
            "language_confidence IS NULL OR (language_confidence >= 0 AND language_confidence <= 1)",
            name="ingestion_validation_reports_language_confidence_range",
        ),
        CheckConstraint(
            "printable_ratio IS NULL OR (printable_ratio >= 0 AND printable_ratio <= 1)",
            name="ingestion_validation_reports_printable_ratio_range",
        ),
        CheckConstraint(
            "garbled_ratio IS NULL OR (garbled_ratio >= 0 AND garbled_ratio <= 1)",
            name="ingestion_validation_reports_garbled_ratio_range",
        ),
        CheckConstraint(
            "repetition_ratio IS NULL OR (repetition_ratio >= 0 AND repetition_ratio <= 1)",
            name="ingestion_validation_reports_repetition_ratio_range",
        ),
        CheckConstraint(
            "character_count >= 0",
            name="ingestion_validation_reports_character_count_non_negative",
        ),
        CheckConstraint(
            "((decided_by_id IS NULL AND decided_at IS NULL) OR (decided_by_id IS NOT NULL AND decided_at IS NOT NULL))",
            name="ingestion_validation_reports_decision_actor_time_pair",
        ),
        CheckConstraint(
            "((user_decision IS NULL AND decided_by_id IS NULL AND decided_at IS NULL) OR (user_decision IS NOT NULL AND decided_by_id IS NOT NULL AND decided_at IS NOT NULL))",
            name="ingestion_validation_reports_user_decision_triplet",
        ),
    )

    revision_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document_revisions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    attempt_no: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=ValidationReportStatus.PENDING.value,
        server_default=ValidationReportStatus.PENDING.value,
    )
    outcome: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=ValidationReportOutcome.PENDING.value,
        server_default=ValidationReportOutcome.PENDING.value,
    )
    content_kind: Mapped[str | None] = mapped_column(String(64), nullable=True)
    content_confidence: Mapped[float | None] = mapped_column(nullable=True)
    recommended_authority: Mapped[str | None] = mapped_column(String(16), nullable=True)
    detected_language: Mapped[str | None] = mapped_column(String(32), nullable=True)
    language_confidence: Mapped[float | None] = mapped_column(nullable=True)
    character_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    printable_ratio: Mapped[float | None] = mapped_column(nullable=True)
    garbled_ratio: Mapped[float | None] = mapped_column(nullable=True)
    repetition_ratio: Mapped[float | None] = mapped_column(nullable=True)
    file_checks: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    text_checks: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    classification_details: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    failure_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    failure_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_decision: Mapped[str | None] = mapped_column(String(24), nullable=True)
    decided_by_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    revision: Mapped["DocumentRevision"] = relationship(
        back_populates="validation_reports",
        foreign_keys=[revision_id],
    )
    risk_flags: Mapped[list["IngestionRiskFlag"]] = relationship(
        back_populates="report",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    decided_by: Mapped["User | None"] = relationship(
        back_populates="decided_validation_reports",
        foreign_keys=[decided_by_id],
    )

    @validates("content_kind", "detected_language", "summary", "failure_code", "failure_message")
    def normalize_optional_text_field(self, _key: str, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_text(value)
        return normalized or None


class IngestionRiskFlag(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "ingestion_risk_flags"
    __table_args__ = (
        CheckConstraint(
            "category IN ('file', 'text_quality', 'content', 'privacy', 'credential', 'prompt_injection', 'security', 'other')",
            name="ingestion_risk_flags_category_valid",
        ),
        CheckConstraint(
            "severity IN ('info', 'warning', 'blocker')",
            name="ingestion_risk_flags_severity_valid",
        ),
        CheckConstraint(
            "detector IN ('deterministic', 'ai', 'manual')",
            name="ingestion_risk_flags_detector_valid",
        ),
        CheckConstraint(
            "char_length(kind) BETWEEN 1 AND 80",
            name="ingestion_risk_flags_kind_length",
        ),
        CheckConstraint(
            "char_length(message) BETWEEN 1 AND 500",
            name="ingestion_risk_flags_message_length",
        ),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="ingestion_risk_flags_confidence_range",
        ),
        CheckConstraint(
            "evidence_excerpt IS NULL OR char_length(evidence_excerpt) <= 500",
            name="ingestion_risk_flags_evidence_excerpt_length",
        ),
    )

    report_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("ingestion_validation_reports.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    kind: Mapped[str] = mapped_column(String(80), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    detector: Mapped[str] = mapped_column(String(16), nullable=False)
    message: Mapped[str] = mapped_column(String(500), nullable=False)
    confidence: Mapped[float | None] = mapped_column(nullable=True)
    location: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    details: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    evidence_excerpt: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    report: Mapped["IngestionValidationReport"] = relationship(
        back_populates="risk_flags",
        foreign_keys=[report_id],
    )

    @validates("kind", "message", "evidence_excerpt")
    def normalize_text_field(self, _key: str, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized
