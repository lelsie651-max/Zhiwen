from __future__ import annotations

from datetime import datetime
from enum import StrEnum
import re
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.models.base import Base, UUIDPrimaryKeyMixin, utc_now
from app.utils.validation import normalize_text

if TYPE_CHECKING:
    from app.models.consistency_check import (
        ConsistencyAssessmentLedger,
        ConsistencyCheckApplication,
    )
    from app.models.fact_value_duplicate_grouping import FactValueConsistencyCandidateMember
    from app.models.project import Project
    from app.models.user import User


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ConsistencyReviewDecisionKind(StrEnum):
    SELECT_ONE = "select_one"
    KEEP_MULTIPLE = "keep_multiple"
    CONFIRM_COMPATIBLE = "confirm_compatible"
    DEFER = "defer"


_DECISION_KIND_SQL = (
    "('select_one', 'keep_multiple', 'confirm_compatible', 'defer')"
)
_SELECTED_VALUE_COUNT_SHAPE_SQL = (
    "("
    "decision_kind = 'select_one' AND selected_value_count = 1"
    ") OR ("
    "decision_kind = 'keep_multiple' AND selected_value_count BETWEEN 2 AND 200"
    ") OR ("
    "decision_kind IN ('confirm_compatible', 'defer') AND selected_value_count = 0"
    ")"
)
_REVISION_CHAIN_SHAPE_SQL = (
    "("
    "decision_no = 1 AND supersedes_decision_id IS NULL"
    ") OR ("
    "decision_no > 1 AND supersedes_decision_id IS NOT NULL"
    ")"
)


def _normalize_hash(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.lower()
    if not SHA256_PATTERN.fullmatch(normalized):
        raise ValueError(f"{field_name} must be a 64-character lowercase hexadecimal string")
    return normalized


def _normalize_decision_kind(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("decision_kind must be a string")
    normalized = normalize_text(value)
    if not normalized:
        raise ValueError("decision_kind must not be empty")
    allowed_values = {kind.value for kind in ConsistencyReviewDecisionKind}
    if normalized not in allowed_values:
        raise ValueError(
            "decision_kind must be one of: "
            + ", ".join(sorted(allowed_values))
        )
    return normalized


class ConsistencyReviewDecision(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "consistency_review_decisions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name="fk_ccrevd_project_id_projects",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["consistency_check_application_id", "project_id"],
            [
                "consistency_check_applications.id",
                "consistency_check_applications.project_id",
            ],
            name="fk_ccrevd_app_project_ccapp",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            [
                "assessment_id",
                "consistency_check_application_id",
                "source_consistency_application_id",
                "source_consistency_candidate_id",
            ],
            [
                "consistency_assessments.id",
                "consistency_assessments.consistency_check_application_id",
                "consistency_assessments.source_consistency_application_id",
                "consistency_assessments.source_consistency_candidate_id",
            ],
            name="fk_ccrevd_asmt_app_src_ccasmt",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["actor_id"],
            ["users.id"],
            name="fk_ccrevd_actor_id_users",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["supersedes_decision_id", "assessment_id"],
            [
                "consistency_review_decisions.id",
                "consistency_review_decisions.assessment_id",
            ],
            name="fk_ccrevd_prev_asmt_self",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "assessment_id",
            "decision_no",
            name="uq_ccrevd_asmt_dec_no",
        ),
        UniqueConstraint(
            "supersedes_decision_id",
            name="uq_ccrevd_supersedes_id",
        ),
        UniqueConstraint(
            "decision_manifest_hash",
            name="uq_ccrevd_manifest_hash",
        ),
        UniqueConstraint(
            "id",
            "assessment_id",
            name="uq_ccrevd_id_asmt",
        ),
        UniqueConstraint(
            "id",
            "assessment_id",
            "source_consistency_application_id",
            "source_consistency_candidate_id",
            name="uq_ccrevd_id_asmt_src",
        ),
        CheckConstraint("decision_no > 0", name="ccrevd_dec_no_pos"),
        CheckConstraint(
            f"decision_kind IN {_DECISION_KIND_SQL}",
            name="ccrevd_kind_valid",
        ),
        CheckConstraint(
            _SELECTED_VALUE_COUNT_SHAPE_SQL,
            name="ccrevd_sel_count_shape",
        ),
        CheckConstraint(
            _REVISION_CHAIN_SHAPE_SQL,
            name="ccrevd_revision_shape",
        ),
        CheckConstraint(
            "comment IS NULL OR char_length(comment) BETWEEN 1 AND 2000",
            name="ccrevd_comment_len",
        ),
        CheckConstraint(
            "decision_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="ccrevd_manifest_hash_fmt",
        ),
    )

    project_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    consistency_check_application_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    assessment_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    source_consistency_application_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    source_consistency_candidate_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    actor_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    decision_no: Mapped[int] = mapped_column(Integer, nullable=False)
    supersedes_decision_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    decision_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    selected_value_count: Mapped[int] = mapped_column(Integer, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    decision_manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    project: Mapped["Project"] = relationship(foreign_keys=[project_id])
    application: Mapped["ConsistencyCheckApplication"] = relationship(
        foreign_keys=[consistency_check_application_id, project_id],
        overlaps="assessment,project,review_decisions",
    )
    assessment: Mapped["ConsistencyAssessmentLedger"] = relationship(
        back_populates="review_decisions",
        foreign_keys=[
            assessment_id,
            consistency_check_application_id,
            source_consistency_application_id,
            source_consistency_candidate_id,
        ],
        overlaps="application",
    )
    actor: Mapped["User"] = relationship(
        back_populates="consistency_review_decisions",
        foreign_keys=[actor_id],
    )
    selections: Mapped[list["ConsistencyReviewDecisionSelection"]] = relationship(
        back_populates="decision",
        foreign_keys="ConsistencyReviewDecisionSelection.decision_id",
        order_by="ConsistencyReviewDecisionSelection.selection_order",
    )

    @validates("decision_no")
    def validate_decision_no(self, _key: str, value: int) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("decision_no must be a positive integer")
        if value <= 0:
            raise ValueError("decision_no must be a positive integer")
        return value

    @validates("decision_kind")
    def validate_decision_kind(self, _key: str, value: str) -> str:
        return _normalize_decision_kind(value)

    @validates("selected_value_count")
    def validate_selected_value_count(self, _key: str, value: int) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("selected_value_count must be an integer")
        if value < 0 or value > 200:
            raise ValueError("selected_value_count must be between 0 and 200")
        return value

    @validates("comment")
    def validate_comment(self, _key: str, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("comment must be a string")
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("comment must not be empty")
        if len(normalized) > 2000:
            raise ValueError("comment must be at most 2000 characters")
        return normalized

    @validates("decision_manifest_hash")
    def validate_decision_manifest_hash(self, _key: str, value: str) -> str:
        return _normalize_hash(value, field_name="decision_manifest_hash")


class ConsistencyReviewDecisionSelection(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "consistency_review_decision_selections"
    __table_args__ = (
        ForeignKeyConstraint(
            [
                "decision_id",
                "assessment_id",
                "source_consistency_application_id",
                "source_consistency_candidate_id",
            ],
            [
                "consistency_review_decisions.id",
                "consistency_review_decisions.assessment_id",
                "consistency_review_decisions.source_consistency_application_id",
                "consistency_review_decisions.source_consistency_candidate_id",
            ],
            name="fk_ccrevs_decision_src_ccrevd",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            [
                "source_consistency_application_id",
                "source_consistency_candidate_id",
                "fact_value_id",
            ],
            [
                "fact_value_consistency_candidate_members.consistency_application_id",
                "fact_value_consistency_candidate_members.candidate_id",
                "fact_value_consistency_candidate_members.fact_value_id",
            ],
            name="fk_ccrevs_srccand_fv_fvccm",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "decision_id",
            "fact_value_id",
            name="uq_ccrevs_decision_fv",
        ),
        UniqueConstraint(
            "decision_id",
            "selection_order",
            name="uq_ccrevs_decision_order",
        ),
        CheckConstraint(
            "selection_order BETWEEN 0 AND 199",
            name="ccrevs_sel_order_rng",
        ),
    )

    decision_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    assessment_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    source_consistency_application_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    source_consistency_candidate_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    fact_value_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    selection_order: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    decision: Mapped["ConsistencyReviewDecision"] = relationship(
        back_populates="selections",
        foreign_keys=[
            decision_id,
            assessment_id,
            source_consistency_application_id,
            source_consistency_candidate_id,
        ],
    )
    source_candidate_member: Mapped["FactValueConsistencyCandidateMember"] = relationship(
        foreign_keys=[
            source_consistency_application_id,
            source_consistency_candidate_id,
            fact_value_id,
        ],
        overlaps="decision",
    )
    @validates("selection_order")
    def validate_selection_order(self, _key: str, value: int) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("selection_order must be an integer between 0 and 199")
        if value < 0 or value > 199:
            raise ValueError("selection_order must be an integer between 0 and 199")
        return value


Index("ix_ccrevd_project_id", ConsistencyReviewDecision.project_id)
Index(
    "ix_ccrevd_consistency_check_application_id",
    ConsistencyReviewDecision.consistency_check_application_id,
)
Index("ix_ccrevd_assessment_id", ConsistencyReviewDecision.assessment_id)
Index("ix_ccrevd_actor_id", ConsistencyReviewDecision.actor_id)
Index("ix_ccrevd_supersedes_decision_id", ConsistencyReviewDecision.supersedes_decision_id)
Index("ix_ccrevs_decision_id", ConsistencyReviewDecisionSelection.decision_id)
Index("ix_ccrevs_fact_value_id", ConsistencyReviewDecisionSelection.fact_value_id)
