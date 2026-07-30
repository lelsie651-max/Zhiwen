from __future__ import annotations

from datetime import datetime
import re
import uuid
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, CheckConstraint, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, utc_now
from app.utils.validation import normalize_text

if TYPE_CHECKING:
    from app.models.project import Project
    from app.models.user import User


SCHEMA_KEY_PATTERN = re.compile(r"^[a-z0-9._-]+$")


class DynamicSchemaStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class DynamicSchemaVersionStatus(StrEnum):
    DRAFT = "draft"
    PROPOSED = "proposed"
    ACTIVE = "active"
    RETIRED = "retired"


class DynamicSchemaVersionSourceKind(StrEnum):
    AI = "ai"
    HUMAN = "human"
    IMPORT = "import"
    SYSTEM = "system"


class DynamicSchemaFieldValueType(StrEnum):
    ANY = "any"
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    ENTITY_REF = "entity_ref"
    LIST = "list"
    OBJECT = "object"
    NULL = "null"


class DynamicSchemaFieldCardinality(StrEnum):
    ONE = "one"
    MANY = "many"


class DynamicSchema(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "dynamic_schemas"
    __table_args__ = (
        UniqueConstraint("project_id", "schema_key", name="uq_dynamic_schemas_project_id_schema_key"),
        UniqueConstraint("current_version_id", name="uq_dynamic_schemas_current_version_id"),
        CheckConstraint("char_length(schema_key) BETWEEN 1 AND 120", name="dynamic_schemas_schema_key_length"),
        CheckConstraint("schema_key ~ '^[a-z0-9._-]+$'", name="dynamic_schemas_schema_key_format"),
        CheckConstraint("char_length(name) BETWEEN 1 AND 200", name="dynamic_schemas_name_length"),
        CheckConstraint("char_length(subject_kind) BETWEEN 1 AND 64", name="dynamic_schemas_subject_kind_length"),
        CheckConstraint(
            "status IN ('active', 'archived')",
            name="dynamic_schemas_status_valid",
        ),
    )

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    schema_key: Mapped[str] = mapped_column(String(120), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    subject_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=DynamicSchemaStatus.ACTIVE.value,
        server_default=DynamicSchemaStatus.ACTIVE.value,
    )
    current_version_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("dynamic_schema_versions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    created_by_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )

    project: Mapped["Project"] = relationship(
        back_populates="dynamic_schemas",
        foreign_keys=[project_id],
    )
    current_version: Mapped["DynamicSchemaVersion | None"] = relationship(
        back_populates="current_for_schema",
        foreign_keys=[current_version_id],
        post_update=True,
        uselist=False,
    )
    versions: Mapped[list["DynamicSchemaVersion"]] = relationship(
        back_populates="schema",
        cascade="all, delete-orphan",
        passive_deletes=True,
        foreign_keys="DynamicSchemaVersion.schema_id",
    )
    created_by: Mapped["User | None"] = relationship(
        back_populates="created_dynamic_schemas",
        foreign_keys=[created_by_id],
    )

    @validates("schema_key")
    def validate_schema_key(self, _key: str, value: str) -> str:
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("schema_key must not be empty")
        if not SCHEMA_KEY_PATTERN.fullmatch(normalized):
            raise ValueError("schema_key must contain only lowercase letters, numbers, dots, underscores, or hyphens")
        return normalized

    @validates("name", "subject_kind", "description")
    def normalize_text_field(self, _key: str, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized


class DynamicSchemaVersion(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "dynamic_schema_versions"
    __table_args__ = (
        UniqueConstraint("schema_id", "version_no", name="uq_dynamic_schema_versions_schema_id_version_no"),
        CheckConstraint("version_no > 0", name="dynamic_schema_versions_version_no_positive"),
        CheckConstraint(
            "status IN ('draft', 'proposed', 'active', 'retired')",
            name="dynamic_schema_versions_status_valid",
        ),
        CheckConstraint(
            "source_kind IN ('ai', 'human', 'import', 'system')",
            name="dynamic_schema_versions_source_kind_valid",
        ),
        CheckConstraint(
            "((activated_by_id IS NULL AND activated_at IS NULL) OR (activated_by_id IS NOT NULL AND activated_at IS NOT NULL))",
            name="dynamic_schema_versions_activation_actor_time_pair",
        ),
        CheckConstraint(
            "(status NOT IN ('draft', 'proposed') OR (activated_by_id IS NULL AND activated_at IS NULL))",
            name="dynamic_schema_versions_inactive_status_has_no_activation",
        ),
        CheckConstraint(
            "(status NOT IN ('active', 'retired') OR (activated_by_id IS NOT NULL AND activated_at IS NOT NULL))",
            name="dynamic_schema_versions_active_status_requires_activation",
        ),
        CheckConstraint(
            "(source_kind <> 'human' OR created_by_id IS NOT NULL)",
            name="dynamic_schema_versions_human_requires_created_by",
        ),
    )

    schema_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("dynamic_schemas.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version_no: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=DynamicSchemaVersionStatus.DRAFT.value,
        server_default=DynamicSchemaVersionStatus.DRAFT.value,
    )
    source_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    layout_config: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    created_by_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    activated_by_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    schema: Mapped["DynamicSchema"] = relationship(
        back_populates="versions",
        foreign_keys=[schema_id],
    )
    current_for_schema: Mapped["DynamicSchema | None"] = relationship(
        back_populates="current_version",
        foreign_keys="DynamicSchema.current_version_id",
        uselist=False,
    )
    fields: Mapped[list["DynamicSchemaField"]] = relationship(
        back_populates="schema_version",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    created_by: Mapped["User | None"] = relationship(
        back_populates="created_dynamic_schema_versions",
        foreign_keys=[created_by_id],
    )
    activated_by: Mapped["User | None"] = relationship(
        back_populates="activated_dynamic_schema_versions",
        foreign_keys=[activated_by_id],
    )

    @validates("status", "source_kind", "summary")
    def normalize_optional_text_field(self, _key: str, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized


class DynamicSchemaField(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "dynamic_schema_fields"
    __table_args__ = (
        UniqueConstraint(
            "schema_version_id",
            "field_key",
            name="uq_dynamic_schema_fields_schema_version_id_field_key",
        ),
        CheckConstraint("char_length(field_key) BETWEEN 1 AND 120", name="dynamic_schema_fields_field_key_length"),
        CheckConstraint("field_key ~ '^[a-z0-9._-]+$'", name="dynamic_schema_fields_field_key_format"),
        CheckConstraint("char_length(label) BETWEEN 1 AND 200", name="dynamic_schema_fields_label_length"),
        CheckConstraint("char_length(predicate_key) BETWEEN 1 AND 255", name="dynamic_schema_fields_predicate_key_length"),
        CheckConstraint(
            "scope_key IS NULL OR char_length(scope_key) BETWEEN 1 AND 255",
            name="dynamic_schema_fields_scope_key_length",
        ),
        CheckConstraint(
            "expected_value_type IN ('any', 'string', 'number', 'boolean', 'date', 'datetime', 'entity_ref', 'list', 'object', 'null')",
            name="dynamic_schema_fields_expected_value_type_valid",
        ),
        CheckConstraint(
            "cardinality IN ('one', 'many')",
            name="dynamic_schema_fields_cardinality_valid",
        ),
        CheckConstraint(
            "group_key IS NULL OR char_length(group_key) BETWEEN 1 AND 120",
            name="dynamic_schema_fields_group_key_length",
        ),
        CheckConstraint(
            "display_order >= 0",
            name="dynamic_schema_fields_display_order_non_negative",
        ),
    )

    schema_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("dynamic_schema_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    field_key: Mapped[str] = mapped_column(String(120), nullable=False)
    label: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    predicate_key: Mapped[str] = mapped_column(String(255), nullable=False)
    scope_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    expected_value_type: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=DynamicSchemaFieldValueType.ANY.value,
        server_default=DynamicSchemaFieldValueType.ANY.value,
    )
    cardinality: Mapped[str] = mapped_column(
        String(8),
        nullable=False,
        default=DynamicSchemaFieldCardinality.ONE.value,
        server_default=DynamicSchemaFieldCardinality.ONE.value,
    )
    is_required: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    is_title: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    is_summary: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    is_hidden: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    group_key: Mapped[str | None] = mapped_column(String(120), nullable=True)
    display_order: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    display_config: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    validation_rules: Mapped[dict[str, Any]] = mapped_column(
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

    schema_version: Mapped["DynamicSchemaVersion"] = relationship(
        back_populates="fields",
        foreign_keys=[schema_version_id],
    )

    @validates("field_key", "group_key")
    def validate_key_fields(self, key: str, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError(f"{key} must not be empty")
        if not SCHEMA_KEY_PATTERN.fullmatch(normalized):
            raise ValueError(f"{key} must contain only lowercase letters, numbers, dots, underscores, or hyphens")
        return normalized

    @validates("label", "description", "predicate_key", "scope_key", "expected_value_type", "cardinality")
    def normalize_optional_text_field(self, _key: str, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized
