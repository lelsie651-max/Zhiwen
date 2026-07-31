from __future__ import annotations

import re
import unicodedata
import uuid
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, CheckConstraint, ForeignKey, Index, String, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.utils.validation import normalize_text

if TYPE_CHECKING:
    from app.models.fact import Fact, FactValue
    from app.models.project import Project
    from app.models.user import User


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class EntityStatus(StrEnum):
    ACTIVE = "active"
    MERGED = "merged"
    ARCHIVED = "archived"


class EntityAliasKind(StrEnum):
    CANONICAL = "canonical"
    ALTERNATE = "alternate"
    ABBREVIATION = "abbreviation"
    TRANSLITERATION = "transliteration"


class EntityAliasStatus(StrEnum):
    ACTIVE = "active"
    RETIRED = "retired"


def normalize_entity_alias(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("alias value must be a string")

    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalize_text(normalized)
    normalized = normalized.casefold()
    normalized = " ".join(normalized.split())

    if not normalized:
        raise ValueError("normalized alias must not be empty")
    if len(normalized) > 255:
        raise ValueError("normalized alias must be at most 255 characters")
    return normalized


class Entity(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "entities"
    __table_args__ = (
        UniqueConstraint("project_id", "entity_type", "canonical_key", name="uq_ent_proj_type_key"),
        UniqueConstraint("project_id", "identity_hash", name="uq_ent_proj_hash"),
        Index("ix_entities_project_type", "project_id", "entity_type"),
        CheckConstraint("char_length(entity_type) BETWEEN 1 AND 64", name="ent_type_len"),
        CheckConstraint("char_length(canonical_key) BETWEEN 1 AND 255", name="ent_key_len"),
        CheckConstraint("char_length(display_name) BETWEEN 1 AND 255", name="ent_name_len"),
        CheckConstraint("identity_hash ~ '^[0-9a-f]{64}$'", name="ent_hash_fmt"),
        CheckConstraint(
            "status IN ('active', 'merged', 'archived')",
            name="ent_status_ok",
        ),
        CheckConstraint(
            "((status = 'merged' AND merged_into_entity_id IS NOT NULL) OR "
            "(status <> 'merged' AND merged_into_entity_id IS NULL))",
            name="ent_merge_pair",
        ),
        CheckConstraint(
            "(merged_into_entity_id IS NULL OR merged_into_entity_id <> id)",
            name="ent_not_self_merge",
        ),
    )

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    canonical_key: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    identity_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=EntityStatus.ACTIVE.value,
        server_default=EntityStatus.ACTIVE.value,
    )
    merged_into_entity_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("entities.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    created_by_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )

    project: Mapped["Project"] = relationship(
        back_populates="entities",
        foreign_keys=[project_id],
    )
    aliases: Mapped[list["EntityAlias"]] = relationship(
        back_populates="entity",
        cascade="all, delete-orphan",
        passive_deletes=True,
        foreign_keys="EntityAlias.entity_id",
    )
    subject_facts: Mapped[list["Fact"]] = relationship(
        back_populates="subject_entity",
        foreign_keys="Fact.subject_entity_id",
    )
    referenced_fact_values: Mapped[list["FactValue"]] = relationship(
        back_populates="referenced_entity",
        foreign_keys="FactValue.referenced_entity_id",
    )
    merged_into: Mapped["Entity | None"] = relationship(
        back_populates="merged_from",
        remote_side="Entity.id",
        foreign_keys=[merged_into_entity_id],
    )
    merged_from: Mapped[list["Entity"]] = relationship(
        back_populates="merged_into",
        foreign_keys="Entity.merged_into_entity_id",
    )
    created_by: Mapped["User | None"] = relationship(
        back_populates="created_entities",
        foreign_keys=[created_by_id],
    )

    @validates("entity_type", "display_name")
    def _normalize_text_fields(self, _key: str, value: str) -> str:
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @validates("canonical_key")
    def _normalize_canonical_key(self, _key: str, value: str) -> str:
        return normalize_entity_alias(value)

    @validates("identity_hash")
    def _validate_identity_hash(self, _key: str, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("identity_hash must be a string")
        normalized = value.lower()
        if not SHA256_PATTERN.fullmatch(normalized):
            raise ValueError("identity_hash must be a 64-character lowercase hexadecimal string")
        return normalized


class EntityAlias(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "entity_aliases"
    __table_args__ = (
        UniqueConstraint("entity_id", "normalized_alias", "language_code", name="uq_ea_ent_norm_lang"),
        Index(
            "uq_ea_active_primary",
            "entity_id",
            unique=True,
            postgresql_where=text("status = 'active' AND is_primary = true"),
        ),
        CheckConstraint("char_length(alias_text) BETWEEN 1 AND 255", name="ea_text_len"),
        CheckConstraint("char_length(normalized_alias) BETWEEN 1 AND 255", name="ea_norm_len"),
        CheckConstraint("char_length(language_code) BETWEEN 1 AND 32", name="ea_lang_len"),
        CheckConstraint(
            "alias_kind IN ('canonical', 'alternate', 'abbreviation', 'transliteration')",
            name="ea_kind_ok",
        ),
        CheckConstraint(
            "status IN ('active', 'retired')",
            name="ea_status_ok",
        ),
        CheckConstraint(
            "((NOT is_primary) OR (alias_kind = 'canonical' AND status = 'active'))",
            name="ea_primary_rule",
        ),
        CheckConstraint(
            "(status <> 'retired' OR NOT is_primary)",
            name="ea_retired_no_primary",
        ),
    )

    entity_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    alias_text: Mapped[str] = mapped_column(String(255), nullable=False)
    normalized_alias: Mapped[str] = mapped_column(String(255), nullable=False)
    language_code: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="und",
        server_default="und",
    )
    alias_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=EntityAliasStatus.ACTIVE.value,
        server_default=EntityAliasStatus.ACTIVE.value,
    )
    is_primary: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    created_by_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )

    entity: Mapped["Entity"] = relationship(
        back_populates="aliases",
        foreign_keys=[entity_id],
    )
    created_by: Mapped["User | None"] = relationship(
        back_populates="created_entity_aliases",
        foreign_keys=[created_by_id],
    )

    @validates("alias_text", "language_code", "alias_kind", "status")
    def _normalize_text_values(self, _key: str, value: str) -> str:
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @validates("normalized_alias")
    def _validate_normalized_alias(self, _key: str, value: str) -> str:
        return normalize_entity_alias(value)
