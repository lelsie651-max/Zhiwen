from __future__ import annotations

from datetime import datetime
import re
import uuid
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKeyConstraint,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.models.base import Base, UUIDPrimaryKeyMixin, utc_now
from app.utils.validation import normalize_text

if TYPE_CHECKING:
    from app.models.project import Project
    from app.models.user import User


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ProjectVersionCreationKind(StrEnum):
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    PRE_PUBLISH = "pre_publish"
    ROLLBACK = "rollback"


class ProjectVersion(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "project_versions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name="fk_projver_project_id_projects",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["created_by_id"],
            ["users.id"],
            name="fk_projver_created_by_id_users",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["schema_id", "project_id"],
            ["dynamic_schemas.id", "dynamic_schemas.project_id"],
            name="fk_projver_schema_proj_dynschema",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["schema_version_id", "schema_id"],
            ["dynamic_schema_versions.id", "dynamic_schema_versions.schema_id"],
            name="fk_projver_sver_schema_dynsver",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["orchestration_id", "project_id", "extraction_run_id"],
            [
                "fact_extraction_orchestrations.id",
                "fact_extraction_orchestrations.project_id",
                "fact_extraction_orchestrations.extraction_run_id",
            ],
            name="fk_projver_orch_proj_run_feo",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            [
                "consistency_check_application_id",
                "project_id",
                "orchestration_id",
                "source_consistency_application_id",
            ],
            [
                "consistency_check_applications.id",
                "consistency_check_applications.project_id",
                "consistency_check_applications.orchestration_id",
                "consistency_check_applications.consistency_application_id",
            ],
            name="fk_projver_ccapp_proj_orch_src",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["copied_from_version_id", "project_id"],
            ["project_versions.id", "project_versions.project_id"],
            name="fk_projver_prev_ver_projver",
            ondelete="RESTRICT",
        ),
        UniqueConstraint("project_id", "version_no", name="uq_projver_project_verno"),
        UniqueConstraint("version_manifest_hash", name="uq_projver_manifest_hash"),
        UniqueConstraint("id", "project_id", name="uq_projver_id_project"),
        CheckConstraint("version_no > 0", name="projver_ver_no_pos"),
        CheckConstraint(
            "creation_kind IN ('manual', 'automatic', 'pre_publish', 'rollback')",
            name="projver_creation_kind_valid",
        ),
        CheckConstraint(
            "reason IS NULL OR char_length(reason) BETWEEN 1 AND 2000",
            name="projver_reason_len",
        ),
        CheckConstraint(
            "char_length(knowledge_view_algorithm_name) BETWEEN 1 AND 64",
            name="projver_alg_name_len",
        ),
        CheckConstraint(
            "char_length(knowledge_view_algorithm_version) BETWEEN 1 AND 32",
            name="projver_alg_ver_len",
        ),
        CheckConstraint(
            "char_length(snapshot_format_version) BETWEEN 1 AND 32",
            name="projver_snap_fmt_len",
        ),
        CheckConstraint(
            "jsonb_typeof(snapshot_json) = 'object'",
            name="projver_snapshot_obj",
        ),
        CheckConstraint(
            "schema_definition_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="projver_schema_manifest_fmt",
        ),
        CheckConstraint(
            "ufl_source_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="projver_ufl_manifest_fmt",
        ),
        CheckConstraint(
            "consistency_result_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="projver_cons_manifest_fmt",
        ),
        CheckConstraint(
            "raw_projection_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="projver_raw_manifest_fmt",
        ),
        CheckConstraint(
            "reviewed_projection_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="projver_review_manifest_fmt",
        ),
        CheckConstraint(
            "knowledge_view_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="projver_know_manifest_fmt",
        ),
        CheckConstraint(
            "snapshot_json_hash ~ '^[0-9a-f]{64}$'",
            name="projver_snapshot_hash_fmt",
        ),
        CheckConstraint(
            "version_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="projver_ver_manifest_fmt",
        ),
        CheckConstraint("record_count >= 0", name="projver_record_count_nn"),
        CheckConstraint("section_count >= 0", name="projver_section_count_nn"),
        CheckConstraint("field_count >= 0", name="projver_field_count_nn"),
        CheckConstraint(
            "missing_field_count >= 0",
            name="projver_missing_count_nn",
        ),
        CheckConstraint(
            "review_required_field_count >= 0",
            name="projver_review_req_count_nn",
        ),
        CheckConstraint(
            "resolved_field_count >= 0",
            name="projver_resolved_count_nn",
        ),
        CheckConstraint(
            "observation_only_field_count >= 0",
            name="projver_observe_count_nn",
        ),
        CheckConstraint(
            "mixed_field_count >= 0",
            name="projver_mixed_count_nn",
        ),
        CheckConstraint(
            "missing_field_count + review_required_field_count + resolved_field_count + "
            "observation_only_field_count + mixed_field_count = field_count",
            name="projver_field_count_sum",
        ),
        CheckConstraint(
            "("
            "creation_kind = 'rollback' AND copied_from_version_id IS NOT NULL"
            ") OR ("
            "creation_kind <> 'rollback' AND copied_from_version_id IS NULL"
            ")",
            name="projver_rollback_shape",
        ),
    )

    project_id: Mapped[uuid.UUID] = mapped_column(nullable=False, index=True)
    version_no: Mapped[int] = mapped_column(Integer, nullable=False)
    created_by_id: Mapped[uuid.UUID] = mapped_column(nullable=False, index=True)
    creation_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    copied_from_version_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
        index=True,
    )
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    schema_id: Mapped[uuid.UUID] = mapped_column(nullable=False, index=True)
    schema_version_id: Mapped[uuid.UUID] = mapped_column(nullable=False, index=True)
    orchestration_id: Mapped[uuid.UUID] = mapped_column(nullable=False, index=True)
    extraction_run_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    consistency_check_application_id: Mapped[uuid.UUID] = mapped_column(
        nullable=False,
        index=True,
    )
    source_consistency_application_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    schema_definition_manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    ufl_source_manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    consistency_result_manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    raw_projection_manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    reviewed_projection_manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    knowledge_view_manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    knowledge_view_algorithm_name: Mapped[str] = mapped_column(String(64), nullable=False)
    knowledge_view_algorithm_version: Mapped[str] = mapped_column(String(32), nullable=False)
    snapshot_format_version: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="1.0.0",
        server_default="1.0.0",
    )
    snapshot_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    snapshot_json_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    version_manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    record_count: Mapped[int] = mapped_column(Integer, nullable=False)
    section_count: Mapped[int] = mapped_column(Integer, nullable=False)
    field_count: Mapped[int] = mapped_column(Integer, nullable=False)
    missing_field_count: Mapped[int] = mapped_column(Integer, nullable=False)
    review_required_field_count: Mapped[int] = mapped_column(Integer, nullable=False)
    resolved_field_count: Mapped[int] = mapped_column(Integer, nullable=False)
    observation_only_field_count: Mapped[int] = mapped_column(Integer, nullable=False)
    mixed_field_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    project: Mapped["Project"] = relationship(
        back_populates="project_versions",
        foreign_keys=[project_id],
        overlaps="current_version,current_for_project",
    )
    current_for_project: Mapped["Project | None"] = relationship(
        back_populates="current_version",
        foreign_keys="Project.current_version_id",
        uselist=False,
        overlaps="project,project_versions",
    )
    created_by: Mapped["User"] = relationship(
        back_populates="created_project_versions",
        foreign_keys=[created_by_id],
    )
    @validates(
        "schema_definition_manifest_hash",
        "ufl_source_manifest_hash",
        "consistency_result_manifest_hash",
        "raw_projection_manifest_hash",
        "reviewed_projection_manifest_hash",
        "knowledge_view_manifest_hash",
        "snapshot_json_hash",
        "version_manifest_hash",
    )
    def validate_hash(self, key: str, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        normalized = value.lower()
        if not SHA256_PATTERN.fullmatch(normalized):
            raise ValueError(f"{key} must be a 64-character lowercase hexadecimal string")
        return normalized

    @validates("version_no")
    def validate_version_no(self, _key: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("version_no must be a positive integer")
        return value

    @validates(
        "record_count",
        "section_count",
        "field_count",
        "missing_field_count",
        "review_required_field_count",
        "resolved_field_count",
        "observation_only_field_count",
        "mixed_field_count",
    )
    def validate_count(self, key: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be a non-negative integer")
        return value

    @validates("creation_kind")
    def validate_creation_kind(self, _key: str, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("creation_kind must be a string")
        normalized = normalize_text(value)
        if normalized not in {kind.value for kind in ProjectVersionCreationKind}:
            raise ValueError("creation_kind must be one of: manual, automatic, pre_publish, rollback")
        return normalized

    @validates("reason")
    def validate_reason(self, _key: str, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("reason must not be empty")
        if len(normalized) > 2000:
            raise ValueError("reason must be at most 2000 characters")
        return normalized

    @validates(
        "knowledge_view_algorithm_name",
        "knowledge_view_algorithm_version",
        "snapshot_format_version",
    )
    def validate_short_text(self, key: str, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError(f"{key} must not be empty")
        max_length = 64 if key == "knowledge_view_algorithm_name" else 32
        if len(normalized) > max_length:
            raise ValueError(f"{key} must be at most {max_length} characters")
        return normalized

    @validates("snapshot_json")
    def validate_snapshot_json(self, _key: str, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("snapshot_json must be a JSON object")
        return value
