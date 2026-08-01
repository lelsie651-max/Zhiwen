"""Add immutable human consistency review decision ledgers.

Revision ID: 202608010500
Revises: 202608010400
Create Date: 2026-08-02 00:30:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision: str = "202608010500"
down_revision: str | None = "202608010400"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_ccapp_id_project",
        "consistency_check_applications",
        ["id", "project_id"],
    )
    op.create_unique_constraint(
        "uq_ccasmt_id_app_srccand",
        "consistency_assessments",
        [
            "id",
            "consistency_check_application_id",
            "source_consistency_application_id",
            "source_consistency_candidate_id",
        ],
    )

    op.create_table(
        "consistency_review_decisions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("consistency_check_application_id", sa.Uuid(), nullable=False),
        sa.Column("assessment_id", sa.Uuid(), nullable=False),
        sa.Column("source_consistency_application_id", sa.Uuid(), nullable=False),
        sa.Column("source_consistency_candidate_id", sa.Uuid(), nullable=False),
        sa.Column("actor_id", sa.Uuid(), nullable=False),
        sa.Column("decision_no", sa.Integer(), nullable=False),
        sa.Column("supersedes_decision_id", sa.Uuid(), nullable=True),
        sa.Column("decision_kind", sa.String(length=32), nullable=False),
        sa.Column("selected_value_count", sa.Integer(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("decision_manifest_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name="fk_ccrevd_project_id_projects",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["consistency_check_application_id", "project_id"],
            [
                "consistency_check_applications.id",
                "consistency_check_applications.project_id",
            ],
            name="fk_ccrevd_app_project_ccapp",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
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
        sa.ForeignKeyConstraint(
            ["actor_id"],
            ["users.id"],
            name="fk_ccrevd_actor_id_users",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["supersedes_decision_id", "assessment_id"],
            [
                "consistency_review_decisions.id",
                "consistency_review_decisions.assessment_id",
            ],
            name="fk_ccrevd_prev_asmt_self",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_consistency_review_decisions")),
        sa.UniqueConstraint(
            "assessment_id",
            "decision_no",
            name="uq_ccrevd_asmt_dec_no",
        ),
        sa.UniqueConstraint(
            "supersedes_decision_id",
            name="uq_ccrevd_supersedes_id",
        ),
        sa.UniqueConstraint(
            "decision_manifest_hash",
            name="uq_ccrevd_manifest_hash",
        ),
        sa.UniqueConstraint(
            "id",
            "assessment_id",
            name="uq_ccrevd_id_asmt",
        ),
        sa.UniqueConstraint(
            "id",
            "assessment_id",
            "source_consistency_application_id",
            "source_consistency_candidate_id",
            name="uq_ccrevd_id_asmt_src",
        ),
        sa.CheckConstraint("decision_no > 0", name="ccrevd_dec_no_pos"),
        sa.CheckConstraint(
            "decision_kind IN ('select_one', 'keep_multiple', 'confirm_compatible', 'defer')",
            name="ccrevd_kind_valid",
        ),
        sa.CheckConstraint(
            "("
            "decision_kind = 'select_one' AND selected_value_count = 1"
            ") OR ("
            "decision_kind = 'keep_multiple' AND selected_value_count BETWEEN 2 AND 200"
            ") OR ("
            "decision_kind IN ('confirm_compatible', 'defer') AND selected_value_count = 0"
            ")",
            name="ccrevd_sel_count_shape",
        ),
        sa.CheckConstraint(
            "("
            "decision_no = 1 AND supersedes_decision_id IS NULL"
            ") OR ("
            "decision_no > 1 AND supersedes_decision_id IS NOT NULL"
            ")",
            name="ccrevd_revision_shape",
        ),
        sa.CheckConstraint(
            "comment IS NULL OR char_length(comment) BETWEEN 1 AND 2000",
            name="ccrevd_comment_len",
        ),
        sa.CheckConstraint(
            "decision_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="ccrevd_manifest_hash_fmt",
        ),
    )
    op.create_index(
        "ix_ccrevd_project_id",
        "consistency_review_decisions",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        "ix_ccrevd_consistency_check_application_id",
        "consistency_review_decisions",
        ["consistency_check_application_id"],
        unique=False,
    )
    op.create_index(
        "ix_ccrevd_assessment_id",
        "consistency_review_decisions",
        ["assessment_id"],
        unique=False,
    )
    op.create_index(
        "ix_ccrevd_actor_id",
        "consistency_review_decisions",
        ["actor_id"],
        unique=False,
    )
    op.create_index(
        "ix_ccrevd_supersedes_decision_id",
        "consistency_review_decisions",
        ["supersedes_decision_id"],
        unique=False,
    )

    op.create_table(
        "consistency_review_decision_selections",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("decision_id", sa.Uuid(), nullable=False),
        sa.Column("assessment_id", sa.Uuid(), nullable=False),
        sa.Column("source_consistency_application_id", sa.Uuid(), nullable=False),
        sa.Column("source_consistency_candidate_id", sa.Uuid(), nullable=False),
        sa.Column("fact_value_id", sa.Uuid(), nullable=False),
        sa.Column("selection_order", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(
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
        sa.ForeignKeyConstraint(
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
        sa.PrimaryKeyConstraint("id", name=op.f("pk_consistency_review_decision_selections")),
        sa.UniqueConstraint(
            "decision_id",
            "fact_value_id",
            name="uq_ccrevs_decision_fv",
        ),
        sa.UniqueConstraint(
            "decision_id",
            "selection_order",
            name="uq_ccrevs_decision_order",
        ),
        sa.CheckConstraint(
            "selection_order BETWEEN 0 AND 199",
            name="ccrevs_sel_order_rng",
        ),
    )
    op.create_index(
        "ix_ccrevs_decision_id",
        "consistency_review_decision_selections",
        ["decision_id"],
        unique=False,
    )
    op.create_index(
        "ix_ccrevs_fact_value_id",
        "consistency_review_decision_selections",
        ["fact_value_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_ccrevs_fact_value_id",
        table_name="consistency_review_decision_selections",
    )
    op.drop_index(
        "ix_ccrevs_decision_id",
        table_name="consistency_review_decision_selections",
    )
    op.drop_table("consistency_review_decision_selections")

    op.drop_index(
        "ix_ccrevd_supersedes_decision_id",
        table_name="consistency_review_decisions",
    )
    op.drop_index(
        "ix_ccrevd_actor_id",
        table_name="consistency_review_decisions",
    )
    op.drop_index(
        "ix_ccrevd_assessment_id",
        table_name="consistency_review_decisions",
    )
    op.drop_index(
        "ix_ccrevd_consistency_check_application_id",
        table_name="consistency_review_decisions",
    )
    op.drop_index(
        "ix_ccrevd_project_id",
        table_name="consistency_review_decisions",
    )
    op.drop_table("consistency_review_decisions")

    op.drop_constraint(
        "uq_ccasmt_id_app_srccand",
        "consistency_assessments",
        type_="unique",
    )
    op.drop_constraint(
        "uq_ccapp_id_project",
        "consistency_check_applications",
        type_="unique",
    )
