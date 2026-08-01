"""Scope duplicate grouping ledgers to orchestration attempts.

Revision ID: 202608010200
Revises: 202608010100
Create Date: 2026-08-01 02:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision: str = "202608010200"
down_revision: str | None = "202608010100"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_feo_id_extraction_run",
        "fact_extraction_orchestrations",
        ["id", "extraction_run_id"],
    )

    op.create_unique_constraint(
        "uq_feob_id_orchestration",
        "fact_extraction_orch_batches",
        ["id", "orchestration_id"],
    )

    op.add_column(
        "fact_value_duplicate_grouping_applications",
        sa.Column("orchestration_id", sa.Uuid(), nullable=True),
    )
    op.add_column(
        "fact_value_duplicate_group_members",
        sa.Column("orchestration_id", sa.Uuid(), nullable=True),
    )

    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM fact_value_duplicate_grouping_applications AS a
                JOIN fact_value_duplicate_group_members AS m
                  ON m.grouping_application_id = a.id
                JOIN fact_extraction_orch_batches AS b
                  ON b.id = m.source_batch_id
                JOIN fact_extraction_orchestrations AS o
                  ON o.id = b.orchestration_id
                WHERE o.extraction_run_id <> a.extraction_run_id
            ) THEN
                RAISE EXCEPTION
                    'Cannot backfill duplicate grouping orchestration scope: derived orchestration extraction_run_id does not match application extraction_run_id.';
            END IF;
        END
        $$;
        """
    )

    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM fact_value_duplicate_group_members AS m
                JOIN fact_extraction_orch_batches AS b
                  ON b.id = m.source_batch_id
                GROUP BY m.grouping_application_id
                HAVING count(DISTINCT b.orchestration_id) <> 1
            ) THEN
                RAISE EXCEPTION
                    'Cannot backfill duplicate grouping orchestration scope: member-backed grouping applications map to multiple orchestration attempts.';
            END IF;
        END
        $$;
        """
    )

    op.execute(
        """
        WITH member_app_orchestration AS (
            SELECT DISTINCT
                m.grouping_application_id AS application_id,
                b.orchestration_id AS orchestration_id
            FROM fact_value_duplicate_group_members AS m
            JOIN fact_extraction_orch_batches AS b
              ON b.id = m.source_batch_id
        )
        UPDATE fact_value_duplicate_grouping_applications AS a
        SET orchestration_id = mao.orchestration_id
        FROM member_app_orchestration AS mao
        WHERE a.id = mao.application_id
          AND a.orchestration_id IS NULL;
        """
    )

    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM fact_value_duplicate_grouping_applications AS a
                LEFT JOIN fact_value_duplicate_group_members AS m
                  ON m.grouping_application_id = a.id
                LEFT JOIN fact_extraction_orchestrations AS o
                  ON o.extraction_run_id = a.extraction_run_id
                 AND o.status IN ('completed', 'partial')
                WHERE m.id IS NULL
                  AND a.orchestration_id IS NULL
                GROUP BY a.id
                HAVING count(o.id) <> 1
            ) THEN
                RAISE EXCEPTION
                    'Cannot backfill duplicate grouping orchestration scope: zero-member grouping applications do not map to exactly one completed or partial orchestration.';
            END IF;
        END
        $$;
        """
    )

    op.execute(
        """
        WITH zero_member_app_orchestration AS (
            SELECT DISTINCT
                a.id AS application_id,
                o.id AS orchestration_id
            FROM fact_value_duplicate_grouping_applications AS a
            LEFT JOIN fact_value_duplicate_group_members AS m
              ON m.grouping_application_id = a.id
            JOIN fact_extraction_orchestrations AS o
              ON o.extraction_run_id = a.extraction_run_id
             AND o.status IN ('completed', 'partial')
            WHERE m.id IS NULL
        )
        UPDATE fact_value_duplicate_grouping_applications AS a
        SET orchestration_id = zmao.orchestration_id
        FROM zero_member_app_orchestration AS zmao
        WHERE a.id = zmao.application_id
          AND a.orchestration_id IS NULL;
        """
    )

    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM fact_value_duplicate_grouping_applications AS a
                JOIN fact_extraction_orchestrations AS o
                  ON o.id = a.orchestration_id
                WHERE o.extraction_run_id <> a.extraction_run_id
            ) THEN
                RAISE EXCEPTION
                    'Cannot backfill duplicate grouping orchestration scope: derived orchestration extraction_run_id does not match application extraction_run_id.';
            END IF;
        END
        $$;
        """
    )

    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM fact_value_duplicate_grouping_applications
                WHERE orchestration_id IS NULL
            ) THEN
                RAISE EXCEPTION
                    'Cannot backfill duplicate grouping orchestration scope: grouping applications remain without orchestration_id.';
            END IF;
        END
        $$;
        """
    )

    op.execute(
        """
        UPDATE fact_value_duplicate_group_members AS m
        SET orchestration_id = a.orchestration_id
        FROM fact_value_duplicate_grouping_applications AS a
        WHERE a.id = m.grouping_application_id
          AND m.orchestration_id IS NULL;
        """
    )

    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM fact_value_duplicate_group_members
                WHERE orchestration_id IS NULL
            ) THEN
                RAISE EXCEPTION
                    'Cannot backfill duplicate grouping orchestration scope: group members remain without orchestration_id.';
            END IF;
        END
        $$;
        """
    )

    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM fact_value_duplicate_group_members AS m
                JOIN fact_value_duplicate_grouping_applications AS a
                  ON a.id = m.grouping_application_id
                JOIN fact_extraction_orch_batches AS b
                  ON b.id = m.source_batch_id
                WHERE a.orchestration_id <> m.orchestration_id
                   OR b.orchestration_id <> m.orchestration_id
            ) THEN
                RAISE EXCEPTION
                    'Cannot backfill duplicate grouping orchestration scope: group members do not align with a single orchestration attempt.';
            END IF;
        END
        $$;
        """
    )

    op.alter_column(
        "fact_value_duplicate_grouping_applications",
        "orchestration_id",
        nullable=False,
    )
    op.alter_column(
        "fact_value_duplicate_group_members",
        "orchestration_id",
        nullable=False,
    )

    op.create_index(
        "ix_dupgrp_app_orchestration_id",
        "fact_value_duplicate_grouping_applications",
        ["orchestration_id"],
        unique=False,
    )
    op.create_foreign_key(
        "fk_dupgrp_app_orch_run_feo",
        "fact_value_duplicate_grouping_applications",
        "fact_extraction_orchestrations",
        ["orchestration_id", "extraction_run_id"],
        ["id", "extraction_run_id"],
        ondelete="RESTRICT",
    )
    op.create_unique_constraint(
        "uq_dupgrp_app_id_orch",
        "fact_value_duplicate_grouping_applications",
        ["id", "orchestration_id"],
    )
    op.drop_constraint(
        "uq_dupgrp_app_run_alg",
        "fact_value_duplicate_grouping_applications",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_dupgrp_app_orch_alg",
        "fact_value_duplicate_grouping_applications",
        ["orchestration_id", "algorithm_version"],
    )

    op.drop_constraint(
        op.f("fk_fact_value_duplicate_group_members_grouping_application_id_fact_value_duplicate_grouping_applications"),
        "fact_value_duplicate_group_members",
        type_="foreignkey",
    )
    op.drop_constraint(
        op.f("fk_fact_value_duplicate_group_members_source_batch_id_fact_extraction_orch_batches"),
        "fact_value_duplicate_group_members",
        type_="foreignkey",
    )
    op.create_foreign_key(
        op.f("fk_fact_value_duplicate_group_members_orchestration_id_fact_extraction_orchestrations"),
        "fact_value_duplicate_group_members",
        "fact_extraction_orchestrations",
        ["orchestration_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_foreign_key(
        "fk_dupgrp_member_app_orch_dupgrp_app",
        "fact_value_duplicate_group_members",
        "fact_value_duplicate_grouping_applications",
        ["grouping_application_id", "orchestration_id"],
        ["id", "orchestration_id"],
        ondelete="RESTRICT",
    )
    op.create_foreign_key(
        "fk_dupgrp_member_batch_orch_feob",
        "fact_value_duplicate_group_members",
        "fact_extraction_orch_batches",
        ["source_batch_id", "orchestration_id"],
        ["id", "orchestration_id"],
        ondelete="RESTRICT",
    )
    op.create_index(
        "ix_dupgrp_member_orchestration_id",
        "fact_value_duplicate_group_members",
        ["orchestration_id"],
        unique=False,
    )


def downgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM fact_value_duplicate_grouping_applications
                GROUP BY extraction_run_id, algorithm_version
                HAVING count(*) > 1
            ) THEN
                RAISE EXCEPTION
                    'Cannot downgrade 202608010200: multiple duplicate grouping applications share the same extraction_run_id and algorithm_version.';
            END IF;
        END
        $$;
        """
    )

    op.drop_index("ix_dupgrp_member_orchestration_id", table_name="fact_value_duplicate_group_members")
    op.drop_constraint(
        "fk_dupgrp_member_batch_orch_feob",
        "fact_value_duplicate_group_members",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_dupgrp_member_app_orch_dupgrp_app",
        "fact_value_duplicate_group_members",
        type_="foreignkey",
    )
    op.drop_constraint(
        op.f("fk_fact_value_duplicate_group_members_orchestration_id_fact_extraction_orchestrations"),
        "fact_value_duplicate_group_members",
        type_="foreignkey",
    )
    op.create_foreign_key(
        op.f("fk_fact_value_duplicate_group_members_grouping_application_id_fact_value_duplicate_grouping_applications"),
        "fact_value_duplicate_group_members",
        "fact_value_duplicate_grouping_applications",
        ["grouping_application_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_foreign_key(
        op.f("fk_fact_value_duplicate_group_members_source_batch_id_fact_extraction_orch_batches"),
        "fact_value_duplicate_group_members",
        "fact_extraction_orch_batches",
        ["source_batch_id"],
        ["id"],
        ondelete="RESTRICT",
    )

    op.drop_constraint(
        "uq_dupgrp_app_orch_alg",
        "fact_value_duplicate_grouping_applications",
        type_="unique",
    )
    op.drop_constraint(
        "uq_dupgrp_app_id_orch",
        "fact_value_duplicate_grouping_applications",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_dupgrp_app_run_alg",
        "fact_value_duplicate_grouping_applications",
        ["extraction_run_id", "algorithm_version"],
    )
    op.drop_index("ix_dupgrp_app_orchestration_id", table_name="fact_value_duplicate_grouping_applications")
    op.drop_constraint("fk_dupgrp_app_orch_run_feo", "fact_value_duplicate_grouping_applications", type_="foreignkey")

    op.drop_column("fact_value_duplicate_group_members", "orchestration_id")
    op.drop_column("fact_value_duplicate_grouping_applications", "orchestration_id")
    op.drop_constraint(
        "uq_feob_id_orchestration",
        "fact_extraction_orch_batches",
        type_="unique",
    )
    op.drop_constraint(
        "uq_feo_id_extraction_run",
        "fact_extraction_orchestrations",
        type_="unique",
    )
