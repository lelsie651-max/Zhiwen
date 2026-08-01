"""Add cross-batch duplicate grouping ledger tables.

Revision ID: 202608010100
Revises: 202607312230
Create Date: 2026-08-01 01:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision: str = "202608010100"
down_revision: str | None = "202607312230"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "fact_value_duplicate_grouping_applications",
        sa.Column("extraction_run_id", sa.Uuid(), nullable=False),
        sa.Column("algorithm_version", sa.String(length=64), nullable=False),
        sa.Column("input_manifest_hash", sa.String(length=64), nullable=False),
        sa.Column("result_manifest_hash", sa.String(length=64), nullable=False),
        sa.Column("input_fact_value_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("duplicate_group_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("duplicate_member_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(
            ["extraction_run_id"],
            ["extraction_runs.id"],
            name=op.f("fk_fact_value_duplicate_grouping_applications_extraction_run_id_extraction_runs"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_fact_value_duplicate_grouping_applications")),
        sa.UniqueConstraint("extraction_run_id", "algorithm_version", name="uq_dupgrp_app_run_alg"),
        sa.CheckConstraint(
            "input_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="dupgrp_app_input_manifest_hash_format",
        ),
        sa.CheckConstraint(
            "result_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="dupgrp_app_result_manifest_hash_format",
        ),
        sa.CheckConstraint(
            "input_fact_value_count >= 0",
            name="dupgrp_app_input_fact_value_count_non_negative",
        ),
        sa.CheckConstraint(
            "duplicate_group_count >= 0",
            name="dupgrp_app_duplicate_group_count_non_negative",
        ),
        sa.CheckConstraint(
            "duplicate_member_count >= 0",
            name="dupgrp_app_duplicate_member_count_non_negative",
        ),
    )
    op.create_index(
        "ix_dupgrp_app_extraction_run_id",
        "fact_value_duplicate_grouping_applications",
        ["extraction_run_id"],
        unique=False,
    )

    op.create_table(
        "fact_value_duplicate_groups",
        sa.Column("grouping_application_id", sa.Uuid(), nullable=False),
        sa.Column("duplicate_key_hash", sa.String(length=64), nullable=False),
        sa.Column("member_count", sa.Integer(), nullable=False),
        sa.Column("distinct_batch_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(
            ["grouping_application_id"],
            ["fact_value_duplicate_grouping_applications.id"],
            name=op.f("fk_fact_value_duplicate_groups_grouping_application_id_fact_value_duplicate_grouping_applications"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_fact_value_duplicate_groups")),
        sa.UniqueConstraint("grouping_application_id", "duplicate_key_hash", name="uq_dupgrp_group_app_key"),
        sa.UniqueConstraint("id", "grouping_application_id", name="uq_dupgrp_group_id_app"),
        sa.CheckConstraint(
            "duplicate_key_hash ~ '^[0-9a-f]{64}$'",
            name="dupgrp_group_key_hash_format",
        ),
        sa.CheckConstraint("member_count >= 2", name="dupgrp_group_member_count_min_two"),
        sa.CheckConstraint("distinct_batch_count >= 2", name="dupgrp_group_distinct_batch_count_min_two"),
    )
    op.create_index(
        "ix_dupgrp_group_grouping_application_id",
        "fact_value_duplicate_groups",
        ["grouping_application_id"],
        unique=False,
    )

    op.create_table(
        "fact_value_duplicate_group_members",
        sa.Column("grouping_application_id", sa.Uuid(), nullable=False),
        sa.Column("group_id", sa.Uuid(), nullable=False),
        sa.Column("fact_value_id", sa.Uuid(), nullable=False),
        sa.Column("source_batch_id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(
            ["fact_value_id"],
            ["fact_values.id"],
            name=op.f("fk_fact_value_duplicate_group_members_fact_value_id_fact_values"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["group_id", "grouping_application_id"],
            ["fact_value_duplicate_groups.id", "fact_value_duplicate_groups.grouping_application_id"],
            name="fk_dupgrp_member_group_id_grouping_application_id_dupgrp_group",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["grouping_application_id"],
            ["fact_value_duplicate_grouping_applications.id"],
            name=op.f("fk_fact_value_duplicate_group_members_grouping_application_id_fact_value_duplicate_grouping_applications"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["source_batch_id"],
            ["fact_extraction_orch_batches.id"],
            name=op.f("fk_fact_value_duplicate_group_members_source_batch_id_fact_extraction_orch_batches"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_fact_value_duplicate_group_members")),
        sa.UniqueConstraint("group_id", "fact_value_id", name="uq_dupgrp_member_group_fv"),
        sa.UniqueConstraint("grouping_application_id", "fact_value_id", name="uq_dupgrp_member_app_fv"),
        sa.CheckConstraint("group_id IS NOT NULL", name="dupgrp_member_group_id_required"),
    )
    op.create_index(
        "ix_dupgrp_member_grouping_application_id",
        "fact_value_duplicate_group_members",
        ["grouping_application_id"],
        unique=False,
    )
    op.create_index(
        "ix_dupgrp_member_group_id",
        "fact_value_duplicate_group_members",
        ["group_id"],
        unique=False,
    )
    op.create_index(
        "ix_dupgrp_member_fact_value_id",
        "fact_value_duplicate_group_members",
        ["fact_value_id"],
        unique=False,
    )
    op.create_index(
        "ix_dupgrp_member_source_batch_id",
        "fact_value_duplicate_group_members",
        ["source_batch_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_dupgrp_member_source_batch_id", table_name="fact_value_duplicate_group_members")
    op.drop_index("ix_dupgrp_member_fact_value_id", table_name="fact_value_duplicate_group_members")
    op.drop_index("ix_dupgrp_member_group_id", table_name="fact_value_duplicate_group_members")
    op.drop_index("ix_dupgrp_member_grouping_application_id", table_name="fact_value_duplicate_group_members")
    op.drop_table("fact_value_duplicate_group_members")

    op.drop_index("ix_dupgrp_group_grouping_application_id", table_name="fact_value_duplicate_groups")
    op.drop_table("fact_value_duplicate_groups")

    op.drop_index(
        "ix_dupgrp_app_extraction_run_id",
        table_name="fact_value_duplicate_grouping_applications",
    )
    op.drop_table("fact_value_duplicate_grouping_applications")
