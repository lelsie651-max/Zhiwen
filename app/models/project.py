from __future__ import annotations

import uuid
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.utils.validation import normalize_text, validate_slug

if TYPE_CHECKING:
    from app.models.document import Document
    from app.models.dynamic_schema import DynamicSchema
    from app.models.entity import Entity
    from app.models.fact import Fact
    from app.models.inference import InferenceInputBatch, InferenceRun
    from app.models.processing_job import ProcessingJob
    from app.models.project_member import ProjectMember
    from app.models.user import User


class ProjectVisibility(StrEnum):
    PRIVATE = "private"
    UNLISTED = "unlisted"
    PUBLIC = "public"


class ProjectStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class Project(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "projects"
    __table_args__ = (
        UniqueConstraint("slug", name="uq_projects_slug"),
        CheckConstraint("char_length(name) BETWEEN 1 AND 100", name="projects_name_length"),
        CheckConstraint("char_length(slug) BETWEEN 3 AND 64", name="projects_slug_length"),
        CheckConstraint(
            "slug ~ '^[a-z0-9][a-z0-9-]{2,63}$'",
            name="projects_slug_format",
        ),
        CheckConstraint(
            "visibility IN ('private', 'unlisted', 'public')",
            name="projects_visibility_valid",
        ),
        CheckConstraint(
            "status IN ('active', 'archived')",
            name="projects_status_valid",
        ),
    )

    name: Mapped[str] = mapped_column(String(100), nullable=False)
    slug: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    visibility: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=ProjectVisibility.PRIVATE.value,
        server_default=ProjectVisibility.PRIVATE.value,
    )
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=ProjectStatus.ACTIVE.value,
        server_default=ProjectStatus.ACTIVE.value,
    )
    created_by_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    created_by: Mapped["User"] = relationship(
        back_populates="created_projects",
        foreign_keys=[created_by_id],
    )
    members: Mapped[list["ProjectMember"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    documents: Mapped[list["Document"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    facts: Mapped[list["Fact"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    entities: Mapped[list["Entity"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    dynamic_schemas: Mapped[list["DynamicSchema"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    processing_jobs: Mapped[list["ProcessingJob"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    inference_input_batches: Mapped[list["InferenceInputBatch"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    inference_runs: Mapped[list["InferenceRun"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    @validates("name")
    def normalize_name_field(self, _key: str, value: str) -> str:
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("name must not be empty")
        return normalized

    @validates("slug")
    def validate_slug_field(self, _key: str, value: str) -> str:
        return validate_slug(value)
