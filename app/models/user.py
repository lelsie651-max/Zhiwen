from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.utils.validation import normalize_optional_email, normalize_text, validate_handle

if TYPE_CHECKING:
    from app.models.document import Document
    from app.models.document_revision import DocumentRevision
    from app.models.dynamic_schema import DynamicSchema, DynamicSchemaVersion
    from app.models.fact import Fact, FactValue
    from app.models.ingestion_validation import IngestionValidationReport
    from app.models.processing_job import ProcessingJob
    from app.models.project import Project
    from app.models.project_member import ProjectMember


class UserStatus(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("handle", name="uq_users_handle"),
        UniqueConstraint("email", name="uq_users_email"),
        CheckConstraint("char_length(handle) BETWEEN 3 AND 32", name="users_handle_length"),
        CheckConstraint(
            "handle ~ '^[a-z0-9][a-z0-9_-]{2,31}$'",
            name="users_handle_format",
        ),
        CheckConstraint(
            "char_length(display_name) BETWEEN 1 AND 80",
            name="users_display_name_length",
        ),
        CheckConstraint("email IS NULL OR email = lower(email)", name="users_email_lower"),
        CheckConstraint(
            "status IN ('active', 'disabled')",
            name="users_status_valid",
        ),
    )

    handle: Mapped[str] = mapped_column(String(32), nullable=False)
    display_name: Mapped[str] = mapped_column(String(80), nullable=False)
    email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    avatar_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    locale: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="zh-CN",
        server_default="zh-CN",
    )
    timezone: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="Asia/Shanghai",
        server_default="Asia/Shanghai",
    )
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=UserStatus.ACTIVE.value,
        server_default=UserStatus.ACTIVE.value,
    )

    created_projects: Mapped[list["Project"]] = relationship(
        back_populates="created_by",
        foreign_keys="Project.created_by_id",
    )
    project_members: Mapped[list["ProjectMember"]] = relationship(
        back_populates="user",
    )
    created_documents: Mapped[list["Document"]] = relationship(
        back_populates="created_by",
        foreign_keys="Document.created_by_id",
    )
    uploaded_revisions: Mapped[list["DocumentRevision"]] = relationship(
        back_populates="uploaded_by",
        foreign_keys="DocumentRevision.uploaded_by_id",
    )
    decided_validation_reports: Mapped[list["IngestionValidationReport"]] = relationship(
        back_populates="decided_by",
        foreign_keys="IngestionValidationReport.decided_by_id",
    )
    created_facts: Mapped[list["Fact"]] = relationship(
        back_populates="created_by",
        foreign_keys="Fact.created_by_id",
    )
    created_fact_values: Mapped[list["FactValue"]] = relationship(
        back_populates="created_by",
        foreign_keys="FactValue.created_by_id",
    )
    decided_fact_values: Mapped[list["FactValue"]] = relationship(
        back_populates="decided_by",
        foreign_keys="FactValue.decided_by_id",
    )
    created_dynamic_schemas: Mapped[list["DynamicSchema"]] = relationship(
        back_populates="created_by",
        foreign_keys="DynamicSchema.created_by_id",
    )
    created_dynamic_schema_versions: Mapped[list["DynamicSchemaVersion"]] = relationship(
        back_populates="created_by",
        foreign_keys="DynamicSchemaVersion.created_by_id",
    )
    activated_dynamic_schema_versions: Mapped[list["DynamicSchemaVersion"]] = relationship(
        back_populates="activated_by",
        foreign_keys="DynamicSchemaVersion.activated_by_id",
    )
    requested_processing_jobs: Mapped[list["ProcessingJob"]] = relationship(
        back_populates="requested_by",
        foreign_keys="ProcessingJob.requested_by_id",
    )

    @validates("handle")
    def validate_handle_field(self, _key: str, value: str) -> str:
        return validate_handle(value)

    @validates("email")
    def normalize_email_field(self, _key: str, value: str | None) -> str | None:
        return normalize_optional_email(value)

    @validates("display_name", "locale", "timezone")
    def normalize_text_field(self, _key: str, value: str) -> str:
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized
