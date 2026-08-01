from __future__ import annotations

from datetime import datetime
import re
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.models.base import Base, UUIDPrimaryKeyMixin, utc_now
from app.utils.validation import normalize_text

if TYPE_CHECKING:
    from app.models.document_content import ExtractionRun
    from app.models.fact import FactValue
    from app.models.fact_extraction_orchestration import (
        FactExtractionOrchestration,
        FactExtractionOrchestrationBatch,
    )


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class FactValueDuplicateGroupingApplication(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "fact_value_duplicate_grouping_applications"
    __table_args__ = (
        UniqueConstraint(
            "orchestration_id",
            "algorithm_version",
            name="uq_dupgrp_app_orch_alg",
        ),
        UniqueConstraint(
            "id",
            "orchestration_id",
            name="uq_dupgrp_app_id_orch",
        ),
        CheckConstraint(
            "input_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="dupgrp_app_input_manifest_hash_format",
        ),
        CheckConstraint(
            "result_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="dupgrp_app_result_manifest_hash_format",
        ),
        CheckConstraint("input_fact_value_count >= 0", name="dupgrp_app_input_fact_value_count_non_negative"),
        CheckConstraint("duplicate_group_count >= 0", name="dupgrp_app_duplicate_group_count_non_negative"),
        CheckConstraint("duplicate_member_count >= 0", name="dupgrp_app_duplicate_member_count_non_negative"),
    )

    extraction_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("extraction_runs.id", ondelete="RESTRICT"),
        nullable=False,
    )
    orchestration_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("fact_extraction_orchestrations.id", ondelete="RESTRICT"),
        nullable=False,
    )
    algorithm_version: Mapped[str] = mapped_column(String(64), nullable=False)
    input_manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    result_manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    input_fact_value_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    duplicate_group_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    duplicate_member_count: Mapped[int] = mapped_column(
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

    extraction_run: Mapped["ExtractionRun"] = relationship(foreign_keys=[extraction_run_id])
    orchestration: Mapped["FactExtractionOrchestration"] = relationship(foreign_keys=[orchestration_id])
    groups: Mapped[list["FactValueDuplicateGroup"]] = relationship(
        back_populates="grouping_application",
        foreign_keys="FactValueDuplicateGroup.grouping_application_id",
        order_by="FactValueDuplicateGroup.duplicate_key_hash",
    )
    members: Mapped[list["FactValueDuplicateGroupMember"]] = relationship(
        back_populates="grouping_application",
        foreign_keys="FactValueDuplicateGroupMember.grouping_application_id",
        order_by="FactValueDuplicateGroupMember.fact_value_id",
    )

    @validates("algorithm_version")
    def validate_algorithm_version(self, _key: str, value: str) -> str:
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("algorithm_version must not be empty")
        return normalized

    @validates("input_manifest_hash", "result_manifest_hash")
    def validate_hash(self, _key: str, value: str) -> str:
        normalized = value.lower()
        if not SHA256_PATTERN.fullmatch(normalized):
            raise ValueError("hash must be a 64-character lowercase hexadecimal string")
        return normalized


class FactValueDuplicateGroup(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "fact_value_duplicate_groups"
    __table_args__ = (
        UniqueConstraint(
            "grouping_application_id",
            "duplicate_key_hash",
            name="uq_dupgrp_group_app_key",
        ),
        UniqueConstraint(
            "id",
            "grouping_application_id",
            name="uq_dupgrp_group_id_app",
        ),
        CheckConstraint("duplicate_key_hash ~ '^[0-9a-f]{64}$'", name="dupgrp_group_key_hash_format"),
        CheckConstraint("member_count >= 2", name="dupgrp_group_member_count_min_two"),
        CheckConstraint("distinct_batch_count >= 2", name="dupgrp_group_distinct_batch_count_min_two"),
    )

    grouping_application_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("fact_value_duplicate_grouping_applications.id", ondelete="RESTRICT"),
        nullable=False,
    )
    duplicate_key_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    member_count: Mapped[int] = mapped_column(Integer, nullable=False)
    distinct_batch_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    grouping_application: Mapped["FactValueDuplicateGroupingApplication"] = relationship(
        back_populates="groups",
        foreign_keys=[grouping_application_id],
    )
    members: Mapped[list["FactValueDuplicateGroupMember"]] = relationship(
        back_populates="group",
        foreign_keys="FactValueDuplicateGroupMember.group_id",
        order_by="FactValueDuplicateGroupMember.fact_value_id",
    )

    @validates("duplicate_key_hash")
    def validate_hash(self, _key: str, value: str) -> str:
        normalized = value.lower()
        if not SHA256_PATTERN.fullmatch(normalized):
            raise ValueError("duplicate_key_hash must be a 64-character lowercase hexadecimal string")
        return normalized


class FactValueDuplicateGroupMember(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "fact_value_duplicate_group_members"
    __table_args__ = (
        ForeignKeyConstraint(
            ["group_id", "grouping_application_id"],
            ["fact_value_duplicate_groups.id", "fact_value_duplicate_groups.grouping_application_id"],
            ondelete="RESTRICT",
            name="fk_dupgrp_member_group_id_grouping_application_id_dupgrp_group",
        ),
        UniqueConstraint(
            "grouping_application_id",
            "fact_value_id",
            name="uq_dupgrp_member_app_fv",
        ),
        UniqueConstraint(
            "group_id",
            "fact_value_id",
            name="uq_dupgrp_member_group_fv",
        ),
        CheckConstraint("group_id IS NOT NULL", name="dupgrp_member_group_id_required"),
        ForeignKeyConstraint(
            ["grouping_application_id", "orchestration_id"],
            [
                "fact_value_duplicate_grouping_applications.id",
                "fact_value_duplicate_grouping_applications.orchestration_id",
            ],
            ondelete="RESTRICT",
            name="fk_dupgrp_member_app_orch_dupgrp_app",
        ),
        ForeignKeyConstraint(
            ["source_batch_id", "orchestration_id"],
            ["fact_extraction_orch_batches.id", "fact_extraction_orch_batches.orchestration_id"],
            ondelete="RESTRICT",
            name="fk_dupgrp_member_batch_orch_feob",
        ),
    )

    orchestration_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("fact_extraction_orchestrations.id", ondelete="RESTRICT"),
        nullable=False,
    )
    grouping_application_id: Mapped[uuid.UUID] = mapped_column(
        nullable=False,
    )
    group_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    fact_value_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("fact_values.id", ondelete="RESTRICT"),
        nullable=False,
    )
    source_batch_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    grouping_application: Mapped["FactValueDuplicateGroupingApplication"] = relationship(
        back_populates="members",
        foreign_keys=[grouping_application_id],
    )
    orchestration: Mapped["FactExtractionOrchestration"] = relationship(foreign_keys=[orchestration_id])
    group: Mapped["FactValueDuplicateGroup"] = relationship(
        back_populates="members",
        foreign_keys=[group_id],
    )
    fact_value: Mapped["FactValue"] = relationship(foreign_keys=[fact_value_id])
    source_batch: Mapped["FactExtractionOrchestrationBatch"] = relationship(foreign_keys=[source_batch_id])


Index(
    "ix_dupgrp_app_extraction_run_id",
    FactValueDuplicateGroupingApplication.extraction_run_id,
)
Index(
    "ix_dupgrp_app_orchestration_id",
    FactValueDuplicateGroupingApplication.orchestration_id,
)
Index("ix_dupgrp_group_grouping_application_id", FactValueDuplicateGroup.grouping_application_id)
Index("ix_dupgrp_member_orchestration_id", FactValueDuplicateGroupMember.orchestration_id)
Index("ix_dupgrp_member_grouping_application_id", FactValueDuplicateGroupMember.grouping_application_id)
Index("ix_dupgrp_member_group_id", FactValueDuplicateGroupMember.group_id)
Index("ix_dupgrp_member_fact_value_id", FactValueDuplicateGroupMember.fact_value_id)
Index("ix_dupgrp_member_source_batch_id", FactValueDuplicateGroupMember.source_batch_id)
