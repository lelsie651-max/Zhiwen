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
    from app.models.fact import Fact, FactValue
    from app.models.fact_extraction_orchestration import (
        FactExtractionOrchestration,
        FactExtractionOrchestrationBatch,
    )


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def normalize_duplicate_grouping_algorithm_version(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("algorithm_version must be a string")
    normalized = normalize_text(value)
    if not normalized:
        raise ValueError("algorithm_version must not be empty")
    if len(normalized) > 64:
        raise ValueError("algorithm_version must be at most 64 characters")
    return normalized


class FactValueDuplicateGroupingApplication(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "fact_value_duplicate_grouping_applications"
    __table_args__ = (
        ForeignKeyConstraint(
            ["orchestration_id", "extraction_run_id"],
            [
                "fact_extraction_orchestrations.id",
                "fact_extraction_orchestrations.extraction_run_id",
            ],
            ondelete="RESTRICT",
            name="fk_dupgrp_app_orch_run_feo",
        ),
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
    orchestration_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
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

    extraction_run: Mapped["ExtractionRun"] = relationship(
        foreign_keys=[extraction_run_id],
        overlaps="orchestration",
    )
    orchestration: Mapped["FactExtractionOrchestration"] = relationship(
        foreign_keys=[orchestration_id, extraction_run_id],
        overlaps="extraction_run",
    )
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
    consistency_candidate_applications: Mapped[list["FactValueConsistencyCandidateApplication"]] = relationship(
        back_populates="duplicate_grouping_application",
        foreign_keys="FactValueConsistencyCandidateApplication.duplicate_grouping_application_id",
        order_by="FactValueConsistencyCandidateApplication.algorithm_version",
    )

    @validates("algorithm_version")
    def validate_algorithm_version(self, _key: str, value: str) -> str:
        return normalize_duplicate_grouping_algorithm_version(value)

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


class FactValueConsistencyCandidateApplication(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "fact_value_consistency_candidate_applications"
    __table_args__ = (
        ForeignKeyConstraint(
            ["duplicate_grouping_application_id", "orchestration_id"],
            [
                "fact_value_duplicate_grouping_applications.id",
                "fact_value_duplicate_grouping_applications.orchestration_id",
            ],
            ondelete="RESTRICT",
            name="fk_fvcca_dupgrp_app_orch",
        ),
        ForeignKeyConstraint(
            ["orchestration_id", "extraction_run_id"],
            [
                "fact_extraction_orchestrations.id",
                "fact_extraction_orchestrations.extraction_run_id",
            ],
            ondelete="RESTRICT",
            name="fk_fvcca_orch_run_feo",
        ),
        UniqueConstraint(
            "duplicate_grouping_application_id",
            "algorithm_version",
            name="uq_fvcca_dupgrp_alg",
        ),
        UniqueConstraint(
            "id",
            "orchestration_id",
            name="uq_fvcca_id_orch",
        ),
        CheckConstraint(
            "input_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="fvcca_input_manifest_hash_format",
        ),
        CheckConstraint(
            "result_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="fvcca_result_manifest_hash_format",
        ),
        CheckConstraint("candidate_count >= 0", name="fvcca_candidate_count_non_negative"),
        CheckConstraint("member_count >= 0", name="fvcca_member_count_non_negative"),
    )

    duplicate_grouping_application_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    orchestration_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    extraction_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("extraction_runs.id", ondelete="RESTRICT"),
        nullable=False,
    )
    algorithm_version: Mapped[str] = mapped_column(String(64), nullable=False)
    input_manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    result_manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    candidate_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    member_count: Mapped[int] = mapped_column(
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

    duplicate_grouping_application: Mapped["FactValueDuplicateGroupingApplication"] = relationship(
        back_populates="consistency_candidate_applications",
        foreign_keys=[duplicate_grouping_application_id, orchestration_id],
        overlaps="orchestration,extraction_run",
    )
    orchestration: Mapped["FactExtractionOrchestration"] = relationship(
        foreign_keys=[orchestration_id, extraction_run_id],
        overlaps="duplicate_grouping_application,extraction_run",
    )
    extraction_run: Mapped["ExtractionRun"] = relationship(
        foreign_keys=[extraction_run_id],
        overlaps="duplicate_grouping_application,orchestration",
    )
    candidates: Mapped[list["FactValueConsistencyCandidate"]] = relationship(
        back_populates="consistency_application",
        foreign_keys="FactValueConsistencyCandidate.consistency_application_id",
        order_by="FactValueConsistencyCandidate.fact_id",
    )
    members: Mapped[list["FactValueConsistencyCandidateMember"]] = relationship(
        back_populates="consistency_application",
        foreign_keys="FactValueConsistencyCandidateMember.consistency_application_id",
        order_by="FactValueConsistencyCandidateMember.fact_value_id",
    )

    @validates("algorithm_version")
    def validate_algorithm_version(self, _key: str, value: str) -> str:
        return normalize_duplicate_grouping_algorithm_version(value)

    @validates("input_manifest_hash", "result_manifest_hash")
    def validate_hash(self, _key: str, value: str) -> str:
        normalized = value.lower()
        if not SHA256_PATTERN.fullmatch(normalized):
            raise ValueError("hash must be a 64-character lowercase hexadecimal string")
        return normalized


class FactValueConsistencyCandidate(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "fact_value_consistency_candidates"
    __table_args__ = (
        UniqueConstraint(
            "consistency_application_id",
            "fact_id",
            "candidate_kind",
            name="uq_fvcc_app_fact_kind",
        ),
        UniqueConstraint(
            "id",
            "consistency_application_id",
            name="uq_fvcc_id_app",
        ),
        CheckConstraint(
            "candidate_kind IN ('multi_value')",
            name="fvcc_candidate_kind_valid",
        ),
        CheckConstraint("member_count >= 2", name="fvcc_member_count_min_two"),
        CheckConstraint("distinct_semantic_key_count >= 2", name="fvcc_semantic_key_count_min_two"),
        CheckConstraint("distinct_batch_count >= 2", name="fvcc_batch_count_min_two"),
    )

    consistency_application_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("fact_value_consistency_candidate_applications.id", ondelete="RESTRICT"),
        nullable=False,
    )
    fact_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("facts.id", ondelete="RESTRICT"),
        nullable=False,
    )
    candidate_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    member_count: Mapped[int] = mapped_column(Integer, nullable=False)
    distinct_semantic_key_count: Mapped[int] = mapped_column(Integer, nullable=False)
    distinct_batch_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    consistency_application: Mapped["FactValueConsistencyCandidateApplication"] = relationship(
        back_populates="candidates",
        foreign_keys=[consistency_application_id],
    )
    fact: Mapped["Fact"] = relationship(foreign_keys=[fact_id])
    members: Mapped[list["FactValueConsistencyCandidateMember"]] = relationship(
        back_populates="candidate",
        foreign_keys="FactValueConsistencyCandidateMember.candidate_id",
        order_by="FactValueConsistencyCandidateMember.fact_value_id",
    )

    @validates("candidate_kind")
    def validate_candidate_kind(self, _key: str, value: str) -> str:
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("candidate_kind must not be empty")
        return normalized


class FactValueConsistencyCandidateMember(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "fact_value_consistency_candidate_members"
    __table_args__ = (
        ForeignKeyConstraint(
            ["candidate_id", "consistency_application_id"],
            [
                "fact_value_consistency_candidates.id",
                "fact_value_consistency_candidates.consistency_application_id",
            ],
            ondelete="RESTRICT",
            name="fk_fvccm_cand_app_fvcc",
        ),
        ForeignKeyConstraint(
            ["consistency_application_id", "orchestration_id"],
            [
                "fact_value_consistency_candidate_applications.id",
                "fact_value_consistency_candidate_applications.orchestration_id",
            ],
            ondelete="RESTRICT",
            name="fk_fvccm_app_orch_fvcca",
        ),
        ForeignKeyConstraint(
            ["source_batch_id", "orchestration_id"],
            ["fact_extraction_orch_batches.id", "fact_extraction_orch_batches.orchestration_id"],
            ondelete="RESTRICT",
            name="fk_fvccm_batch_orch_feob",
        ),
        UniqueConstraint(
            "consistency_application_id",
            "fact_value_id",
            name="uq_fvccm_app_fv",
        ),
        UniqueConstraint(
            "candidate_id",
            "fact_value_id",
            name="uq_fvccm_cand_fv",
        ),
        CheckConstraint("candidate_id IS NOT NULL", name="fvccm_candidate_id_required"),
        CheckConstraint("semantic_key_hash ~ '^[0-9a-f]{64}$'", name="fvccm_semantic_key_hash_format"),
    )

    consistency_application_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    candidate_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    orchestration_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    fact_value_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("fact_values.id", ondelete="RESTRICT"),
        nullable=False,
    )
    source_batch_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    semantic_key_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    consistency_application: Mapped["FactValueConsistencyCandidateApplication"] = relationship(
        back_populates="members",
        foreign_keys=[consistency_application_id],
    )
    candidate: Mapped["FactValueConsistencyCandidate"] = relationship(
        back_populates="members",
        foreign_keys=[candidate_id],
    )
    fact_value: Mapped["FactValue"] = relationship(foreign_keys=[fact_value_id])
    source_batch: Mapped["FactExtractionOrchestrationBatch"] = relationship(foreign_keys=[source_batch_id])

    @validates("semantic_key_hash")
    def validate_semantic_key_hash(self, _key: str, value: str) -> str:
        normalized = value.lower()
        if not SHA256_PATTERN.fullmatch(normalized):
            raise ValueError("semantic_key_hash must be a 64-character lowercase hexadecimal string")
        return normalized


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
Index(
    "ix_fvcca_dupgrp_application_id",
    FactValueConsistencyCandidateApplication.duplicate_grouping_application_id,
)
Index(
    "ix_fvcca_extraction_run_id",
    FactValueConsistencyCandidateApplication.extraction_run_id,
)
Index(
    "ix_fvcca_orchestration_id",
    FactValueConsistencyCandidateApplication.orchestration_id,
)
Index("ix_fvcc_consistency_application_id", FactValueConsistencyCandidate.consistency_application_id)
Index("ix_fvcc_fact_id", FactValueConsistencyCandidate.fact_id)
Index("ix_fvccm_consistency_application_id", FactValueConsistencyCandidateMember.consistency_application_id)
Index("ix_fvccm_candidate_id", FactValueConsistencyCandidateMember.candidate_id)
Index("ix_fvccm_fact_value_id", FactValueConsistencyCandidateMember.fact_value_id)
Index("ix_fvccm_source_batch_id", FactValueConsistencyCandidateMember.source_batch_id)
Index("ix_fvccm_orchestration_id", FactValueConsistencyCandidateMember.orchestration_id)
