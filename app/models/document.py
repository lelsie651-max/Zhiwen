from __future__ import annotations

from datetime import datetime
from decimal import Decimal
import uuid
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Numeric, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.utils.validation import normalize_text

if TYPE_CHECKING:
    from app.models.document_revision import DocumentRevision
    from app.models.project import Project
    from app.models.user import User


class DocumentStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class Document(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint(
            "current_revision_id",
            name="uq_documents_current_revision_id",
        ),
        CheckConstraint(
            "char_length(title) BETWEEN 1 AND 200",
            name="documents_title_length",
        ),
        CheckConstraint(
            "status IN ('active', 'archived')",
            name="documents_status_valid",
        ),
        CheckConstraint(
            "effective_from IS NULL OR effective_to IS NULL OR effective_to >= effective_from",
            name="documents_effective_range_valid",
        ),
    )

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=DocumentStatus.ACTIVE.value,
        server_default=DocumentStatus.ACTIVE.value,
    )
    created_by_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    current_revision_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("document_revisions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    logical_order_kind: Mapped[str | None] = mapped_column(String(32), nullable=True)
    logical_order_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    logical_order_index: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 6),
        nullable=True,
    )
    effective_from: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    effective_to: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    project: Mapped["Project"] = relationship(back_populates="documents")
    created_by: Mapped["User"] = relationship(
        back_populates="created_documents",
        foreign_keys=[created_by_id],
    )
    revisions: Mapped[list["DocumentRevision"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        foreign_keys="DocumentRevision.document_id",
    )
    current_revision: Mapped["DocumentRevision | None"] = relationship(
        back_populates="current_for_document",
        foreign_keys=[current_revision_id],
        post_update=True,
        uselist=False,
    )

    @validates("title")
    def normalize_title_field(self, _key: str, value: str) -> str:
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("title must not be empty")
        return normalized

    @validates("logical_order_kind", "logical_order_value")
    def normalize_optional_text_field(self, _key: str, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_text(value)
        return normalized or None
