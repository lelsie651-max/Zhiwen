"""Add fact value consistency candidate ledgers.

Revision ID: 202608010300
Revises: 202608010200
Create Date: 2026-08-01 03:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision: str = "202608010300"
down_revision: str | None = "202608010200"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "fact_value_consistency_candidate_applications",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("duplicate_grouping_application_id", sa.Uuid(), nullable=False),
        sa.Column("orchestration_id", sa.Uuid(), nullable=False),
        sa.Column("extraction_run_id", sa.Uuid(), nullable=False),
        sa.Column("algorithm_version", sa.String(length=64), nullable=False),
        sa.Column("input_manifest_hash", sa.String(length=64), nullable=False),
        sa.Column("result_manifest_hash", sa.String(length=64), nullable=False),
        sa.Column("candidate_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("member_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(
            ["duplicate_grouping_application_id", "orchestration_id"],
            [
                "fact_value_duplicate_grouping_applications.id",
                "fact_value_duplicate_grouping_applications.orchestration_id",
            ],
            name="fk_fvcca_dupgrp_app_orch",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["extraction_run_id"],
            ["extraction_runs.id"],
            name=op.f("fk_fact_value_consistency_candidate_applications_extraction_run_id_extraction_runs"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["orchestration_id", "extraction_run_id"],
            ["fact_extraction_orchestrations.id", "fact_extraction_orchestrations.extraction_run_id"],
            name="fk_fvcca_orch_run_feo",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_fact_value_consistency_candidate_applications")),
        sa.UniqueConstraint("duplicate_grouping_application_id", "algorithm_version", name="uq_fvcca_dupgrp_alg"),
        sa.UniqueConstraint("id", "orchestration_id", name="uq_fvcca_id_orch"),
        sa.CheckConstraint("input_manifest_hash ~ '^[0-9a-f]{64}$'", name="fvcca_input_manifest_hash_format"),
        sa.CheckConstraint("result_manifest_hash ~ '^[0-9a-f]{64}$'", name="fvcca_result_manifest_hash_format"),
        sa.CheckConstraint("candidate_count >= 0", name="fvcca_candidate_count_non_negative"),
        sa.CheckConstraint("member_count >= 0", name="fvcca_member_count_non_negative"),
    )
    op.create_index(
        "ix_fvcca_dupgrp_application_id",
        "fact_value_consistency_candidate_applications",
        ["duplicate_grouping_application_id"],
        unique=False,
    )
    op.create_index(
        "ix_fvcca_extraction_run_id",
        "fact_value_consistency_candidate_applications",
        ["extraction_run_id"],
        unique=False,
    )
    op.create_index(
        "ix_fvcca_orchestration_id",
        "fact_value_consistency_candidate_applications",
        ["orchestration_id"],
        unique=False,
    )

    op.create_table(
        "fact_value_consistency_candidates",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("consistency_application_id", sa.Uuid(), nullable=False),
        sa.Column("fact_id", sa.Uuid(), nullable=False),
        sa.Column("candidate_kind", sa.String(length=32), nullable=False),
        sa.Column("member_count", sa.Integer(), nullable=False),
        sa.Column("distinct_semantic_key_count", sa.Integer(), nullable=False),
        sa.Column("distinct_batch_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(
            ["consistency_application_id"],
            ["fact_value_consistency_candidate_applications.id"],
            name=op.f("fk_fact_value_consistency_candidates_consistency_application_id_fact_value_consistency_candidate_applications"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["fact_id"],
            ["facts.id"],
            name=op.f("fk_fact_value_consistency_candidates_fact_id_facts"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_fact_value_consistency_candidates")),
        sa.UniqueConstraint("consistency_application_id", "fact_id", "candidate_kind", name="uq_fvcc_app_fact_kind"),
        sa.UniqueConstraint("id", "consistency_application_id", name="uq_fvcc_id_app"),
        sa.CheckConstraint("candidate_kind IN ('multi_value')", name="fvcc_candidate_kind_valid"),
        sa.CheckConstraint("member_count >= 2", name="fvcc_member_count_min_two"),
        sa.CheckConstraint("distinct_semantic_key_count >= 2", name="fvcc_semantic_key_count_min_two"),
        sa.CheckConstraint("distinct_batch_count >= 2", name="fvcc_batch_count_min_two"),
    )
    op.create_index(
        "ix_fvcc_consistency_application_id",
        "fact_value_consistency_candidates",
        ["consistency_application_id"],
        unique=False,
    )
    op.create_index(
        "ix_fvcc_fact_id",
        "fact_value_consistency_candidates",
        ["fact_id"],
        unique=False,
    )

    op.create_table(
        "fact_value_consistency_candidate_members",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("consistency_application_id", sa.Uuid(), nullable=False),
        sa.Column("candidate_id", sa.Uuid(), nullable=False),
        sa.Column("orchestration_id", sa.Uuid(), nullable=False),
        sa.Column("fact_value_id", sa.Uuid(), nullable=False),
        sa.Column("source_batch_id", sa.Uuid(), nullable=False),
        sa.Column("semantic_key_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(
            ["candidate_id", "consistency_application_id"],
            ["fact_value_consistency_candidates.id", "fact_value_consistency_candidates.consistency_application_id"],
            name="fk_fvccm_cand_app_fvcc",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["consistency_application_id", "orchestration_id"],
            [
                "fact_value_consistency_candidate_applications.id",
                "fact_value_consistency_candidate_applications.orchestration_id",
            ],
            name="fk_fvccm_app_orch_fvcca",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["fact_value_id"],
            ["fact_values.id"],
            name=op.f("fk_fact_value_consistency_candidate_members_fact_value_id_fact_values"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["source_batch_id", "orchestration_id"],
            ["fact_extraction_orch_batches.id", "fact_extraction_orch_batches.orchestration_id"],
            name="fk_fvccm_batch_orch_feob",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_fact_value_consistency_candidate_members")),
        sa.UniqueConstraint("consistency_application_id", "fact_value_id", name="uq_fvccm_app_fv"),
        sa.UniqueConstraint("candidate_id", "fact_value_id", name="uq_fvccm_cand_fv"),
        sa.CheckConstraint("candidate_id IS NOT NULL", name="fvccm_candidate_id_required"),
        sa.CheckConstraint("semantic_key_hash ~ '^[0-9a-f]{64}$'", name="fvccm_semantic_key_hash_format"),
    )
    op.create_index(
        "ix_fvccm_consistency_application_id",
        "fact_value_consistency_candidate_members",
        ["consistency_application_id"],
        unique=False,
    )
    op.create_index(
        "ix_fvccm_candidate_id",
        "fact_value_consistency_candidate_members",
        ["candidate_id"],
        unique=False,
    )
    op.create_index(
        "ix_fvccm_fact_value_id",
        "fact_value_consistency_candidate_members",
        ["fact_value_id"],
        unique=False,
    )
    op.create_index(
        "ix_fvccm_source_batch_id",
        "fact_value_consistency_candidate_members",
        ["source_batch_id"],
        unique=False,
    )
    op.create_index(
        "ix_fvccm_orchestration_id",
        "fact_value_consistency_candidate_members",
        ["orchestration_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_fvccm_orchestration_id", table_name="fact_value_consistency_candidate_members")
    op.drop_index("ix_fvccm_source_batch_id", table_name="fact_value_consistency_candidate_members")
    op.drop_index("ix_fvccm_fact_value_id", table_name="fact_value_consistency_candidate_members")
    op.drop_index("ix_fvccm_candidate_id", table_name="fact_value_consistency_candidate_members")
    op.drop_index("ix_fvccm_consistency_application_id", table_name="fact_value_consistency_candidate_members")
    op.drop_table("fact_value_consistency_candidate_members")

    op.drop_index("ix_fvcc_fact_id", table_name="fact_value_consistency_candidates")
    op.drop_index("ix_fvcc_consistency_application_id", table_name="fact_value_consistency_candidates")
    op.drop_table("fact_value_consistency_candidates")

    op.drop_index("ix_fvcca_orchestration_id", table_name="fact_value_consistency_candidate_applications")
    op.drop_index("ix_fvcca_extraction_run_id", table_name="fact_value_consistency_candidate_applications")
    op.drop_index("ix_fvcca_dupgrp_application_id", table_name="fact_value_consistency_candidate_applications")
    op.drop_table("fact_value_consistency_candidate_applications")
