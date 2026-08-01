from __future__ import annotations

from datetime import datetime
import re
import uuid
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, CheckConstraint, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, utc_now
from app.utils.validation import normalize_text

if TYPE_CHECKING:
    from app.models.document_content import ExtractionRun, SourceEvidence
    from app.models.entity import Entity
    from app.models.inference import InferenceRun
    from app.models.project import Project
    from app.models.user import User


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class FactStatus(StrEnum):
    ACTIVE = "active"
    RETIRED = "retired"


class FactValueType(StrEnum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    ENTITY_REF = "entity_ref"
    LIST = "list"
    OBJECT = "object"
    NULL = "null"


class FactValueStatus(StrEnum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class FactValueSourceKind(StrEnum):
    AI = "ai"
    HUMAN = "human"
    IMPORT = "import"
    SYSTEM = "system"


class FactEvidenceRole(StrEnum):
    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    CONTEXT = "context"


class Fact(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "facts"
    __table_args__ = (
        UniqueConstraint("project_id", "identity_hash", name="uq_facts_project_id_identity_hash"),
        UniqueConstraint("current_value_id", name="uq_facts_current_value_id"),
        CheckConstraint("char_length(subject_kind) BETWEEN 1 AND 64", name="facts_subject_kind_length"),
        CheckConstraint("char_length(subject_key) BETWEEN 1 AND 255", name="facts_subject_key_length"),
        CheckConstraint("char_length(predicate_key) BETWEEN 1 AND 255", name="facts_predicate_key_length"),
        CheckConstraint(
            "scope_key IS NULL OR char_length(scope_key) BETWEEN 1 AND 255",
            name="facts_scope_key_length",
        ),
        CheckConstraint(
            "identity_hash ~ '^[0-9a-f]{64}$'",
            name="facts_identity_hash_format",
        ),
        CheckConstraint(
            "status IN ('active', 'retired')",
            name="facts_status_valid",
        ),
    )

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    subject_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    subject_key: Mapped[str] = mapped_column(String(255), nullable=False)
    subject_entity_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("entities.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    predicate_key: Mapped[str] = mapped_column(String(255), nullable=False)
    scope_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    identity_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=FactStatus.ACTIVE.value,
        server_default=FactStatus.ACTIVE.value,
    )
    current_value_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("fact_values.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    created_by_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )

    project: Mapped["Project"] = relationship(
        back_populates="facts",
        foreign_keys=[project_id],
    )
    subject_entity: Mapped["Entity | None"] = relationship(
        back_populates="subject_facts",
        foreign_keys=[subject_entity_id],
    )
    current_value: Mapped["FactValue | None"] = relationship(
        back_populates="current_for_fact",
        foreign_keys=[current_value_id],
        post_update=True,
        uselist=False,
    )
    values: Mapped[list["FactValue"]] = relationship(
        back_populates="fact",
        cascade="all, delete-orphan",
        passive_deletes=True,
        foreign_keys="FactValue.fact_id",
    )
    created_by: Mapped["User | None"] = relationship(
        back_populates="created_facts",
        foreign_keys=[created_by_id],
    )

    @validates("subject_kind", "subject_key", "predicate_key", "scope_key")
    def normalize_text_field(self, _key: str, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @validates("identity_hash")
    def validate_identity_hash(self, _key: str, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("identity_hash must be a string")
        normalized = value.lower()
        if not SHA256_PATTERN.fullmatch(normalized):
            raise ValueError("identity_hash must be a 64-character lowercase hexadecimal string")
        return normalized


class FactValue(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "fact_values"
    __table_args__ = (
        UniqueConstraint("fact_id", "version_no", name="uq_fact_values_fact_id_version_no"),
        UniqueConstraint("fact_id", "inference_run_id", "value_hash", name="uq_fv_fact_ir_value_hash"),
        CheckConstraint("version_no > 0", name="fact_values_version_no_positive"),
        CheckConstraint(
            "value_type IN ('string', 'number', 'boolean', 'date', 'datetime', 'entity_ref', 'list', 'object', 'null')",
            name="fact_values_value_type_valid",
        ),
        CheckConstraint(
            "status IN ('proposed', 'accepted', 'rejected', 'superseded')",
            name="fact_values_status_valid",
        ),
        CheckConstraint(
            "source_kind IN ('ai', 'human', 'import', 'system')",
            name="fact_values_source_kind_valid",
        ),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="fact_values_confidence_range",
        ),
        CheckConstraint(
            "value_hash ~ '^[0-9a-f]{64}$'",
            name="fact_values_value_hash_format",
        ),
        CheckConstraint(
            "((value_type = 'null' AND value_json IS NULL) OR (value_type <> 'null' AND value_json IS NOT NULL))",
            name="fact_values_value_json_null_rule",
        ),
        CheckConstraint(
            "((decided_by_id IS NULL AND decided_at IS NULL) OR (decided_by_id IS NOT NULL AND decided_at IS NOT NULL))",
            name="fact_values_decision_actor_time_pair",
        ),
        CheckConstraint(
            "(status <> 'proposed' OR (decided_by_id IS NULL AND decided_at IS NULL))",
            name="fact_values_proposed_has_no_decision",
        ),
        CheckConstraint(
            "(status NOT IN ('accepted', 'rejected') OR (decided_by_id IS NOT NULL AND decided_at IS NOT NULL))",
            name="fact_values_decided_status_requires_actor_and_time",
        ),
        CheckConstraint(
            "(source_kind <> 'human' OR created_by_id IS NOT NULL)",
            name="fact_values_human_requires_created_by",
        ),
        CheckConstraint(
            "(source_kind <> 'ai' OR (extraction_run_id IS NOT NULL AND inference_run_id IS NOT NULL))",
            name="fact_values_ai_requires_extraction_and_inference_run",
        ),
        CheckConstraint(
            "(source_kind = 'ai' OR inference_run_id IS NULL)",
            name="fact_values_non_ai_forbids_inference_run",
        ),
        CheckConstraint(
            "((value_type = 'entity_ref' AND referenced_entity_id IS NOT NULL) OR "
            "(value_type <> 'entity_ref' AND referenced_entity_id IS NULL))",
            name="ck_fv_entity_ref_pair",
        ),
    )

    fact_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("facts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version_no: Mapped[int] = mapped_column(Integer, nullable=False)
    value_type: Mapped[str] = mapped_column(String(16), nullable=False)
    value_json: Mapped[Any | None] = mapped_column(JSONB(none_as_null=True), nullable=True)
    normalized_value_text: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    value_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    language_code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=FactValueStatus.PROPOSED.value,
        server_default=FactValueStatus.PROPOSED.value,
    )
    source_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    extraction_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("extraction_runs.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    inference_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("inference_runs.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    referenced_entity_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("entities.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_by_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    decided_by_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    fact: Mapped["Fact"] = relationship(
        back_populates="values",
        foreign_keys=[fact_id],
    )
    current_for_fact: Mapped["Fact | None"] = relationship(
        back_populates="current_value",
        foreign_keys="Fact.current_value_id",
        uselist=False,
    )
    extraction_run: Mapped["ExtractionRun | None"] = relationship(
        back_populates="fact_values",
        foreign_keys=[extraction_run_id],
    )
    inference_run: Mapped["InferenceRun | None"] = relationship(
        back_populates="fact_values",
        foreign_keys=[inference_run_id],
    )
    referenced_entity: Mapped["Entity | None"] = relationship(
        back_populates="referenced_fact_values",
        foreign_keys=[referenced_entity_id],
    )
    created_by: Mapped["User | None"] = relationship(
        back_populates="created_fact_values",
        foreign_keys=[created_by_id],
    )
    decided_by: Mapped["User | None"] = relationship(
        back_populates="decided_fact_values",
        foreign_keys=[decided_by_id],
    )
    evidence_links: Mapped[list["FactEvidenceLink"]] = relationship(
        back_populates="fact_value",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    @validates("value_type", "status", "source_kind", "language_code", "normalized_value_text")
    def normalize_optional_text_field(self, _key: str, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @validates("value_hash")
    def validate_value_hash(self, _key: str, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("value_hash must be a string")
        normalized = value.lower()
        if not SHA256_PATTERN.fullmatch(normalized):
            raise ValueError("value_hash must be a 64-character lowercase hexadecimal string")
        return normalized


class FactEvidenceLink(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "fact_evidence_links"
    __table_args__ = (
        UniqueConstraint(
            "id",
            "fact_value_id",
            name="uq_fel_id_fact_value",
        ),
        UniqueConstraint(
            "fact_value_id",
            "evidence_id",
            "role",
            name="uq_fact_evidence_links_fact_value_id_evidence_id_role",
        ),
        CheckConstraint(
            "role IN ('supporting', 'contradicting', 'context')",
            name="fact_evidence_links_role_valid",
        ),
        CheckConstraint(
            "source_order >= 0",
            name="fact_evidence_links_source_order_non_negative",
        ),
    )

    fact_value_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("fact_values.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    evidence_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("source_evidences.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    is_primary: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    source_order: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    fact_value: Mapped["FactValue"] = relationship(
        back_populates="evidence_links",
        foreign_keys=[fact_value_id],
    )
    evidence: Mapped["SourceEvidence"] = relationship(
        back_populates="fact_evidence_links",
        foreign_keys=[evidence_id],
    )

    @validates("role")
    def normalize_role_field(self, _key: str, value: str) -> str:
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("role must not be empty")
        return normalized
