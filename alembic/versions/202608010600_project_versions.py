"""Add immutable project version knowledge snapshot ledger.

Revision ID: 202608010600
Revises: 202608010500
Create Date: 2026-08-03 00:30:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "202608010600"
down_revision: str | None = "202608010500"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_dynschema_id_project",
        "dynamic_schemas",
        ["id", "project_id"],
    )
    op.create_unique_constraint(
        "uq_dynsver_id_schema",
        "dynamic_schema_versions",
        ["id", "schema_id"],
    )
    op.create_unique_constraint(
        "uq_feo_id_proj_run",
        "fact_extraction_orchestrations",
        ["id", "project_id", "extraction_run_id"],
    )
    op.create_unique_constraint(
        "uq_ccapp_id_proj_orch_src",
        "consistency_check_applications",
        ["id", "project_id", "orchestration_id", "consistency_application_id"],
    )

    op.create_table(
        "project_versions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("version_no", sa.Integer(), nullable=False),
        sa.Column("created_by_id", sa.Uuid(), nullable=False),
        sa.Column("creation_kind", sa.String(length=16), nullable=False),
        sa.Column("copied_from_version_id", sa.Uuid(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("schema_id", sa.Uuid(), nullable=False),
        sa.Column("schema_version_id", sa.Uuid(), nullable=False),
        sa.Column("orchestration_id", sa.Uuid(), nullable=False),
        sa.Column("extraction_run_id", sa.Uuid(), nullable=False),
        sa.Column("consistency_check_application_id", sa.Uuid(), nullable=False),
        sa.Column("source_consistency_application_id", sa.Uuid(), nullable=False),
        sa.Column("schema_definition_manifest_hash", sa.String(length=64), nullable=False),
        sa.Column("ufl_source_manifest_hash", sa.String(length=64), nullable=False),
        sa.Column("consistency_result_manifest_hash", sa.String(length=64), nullable=False),
        sa.Column("raw_projection_manifest_hash", sa.String(length=64), nullable=False),
        sa.Column("reviewed_projection_manifest_hash", sa.String(length=64), nullable=False),
        sa.Column("knowledge_view_manifest_hash", sa.String(length=64), nullable=False),
        sa.Column("knowledge_view_algorithm_name", sa.String(length=64), nullable=False),
        sa.Column("knowledge_view_algorithm_version", sa.String(length=32), nullable=False),
        sa.Column(
            "snapshot_format_version",
            sa.String(length=32),
            nullable=False,
            server_default="1.0.0",
        ),
        sa.Column("snapshot_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("snapshot_json_hash", sa.String(length=64), nullable=False),
        sa.Column("version_manifest_hash", sa.String(length=64), nullable=False),
        sa.Column("record_count", sa.Integer(), nullable=False),
        sa.Column("section_count", sa.Integer(), nullable=False),
        sa.Column("field_count", sa.Integer(), nullable=False),
        sa.Column("missing_field_count", sa.Integer(), nullable=False),
        sa.Column("review_required_field_count", sa.Integer(), nullable=False),
        sa.Column("resolved_field_count", sa.Integer(), nullable=False),
        sa.Column("observation_only_field_count", sa.Integer(), nullable=False),
        sa.Column("mixed_field_count", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name="fk_projver_project_id_projects",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["created_by_id"],
            ["users.id"],
            name="fk_projver_created_by_id_users",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["schema_id", "project_id"],
            ["dynamic_schemas.id", "dynamic_schemas.project_id"],
            name="fk_projver_schema_proj_dynschema",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["schema_version_id", "schema_id"],
            ["dynamic_schema_versions.id", "dynamic_schema_versions.schema_id"],
            name="fk_projver_sver_schema_dynsver",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["orchestration_id", "project_id", "extraction_run_id"],
            [
                "fact_extraction_orchestrations.id",
                "fact_extraction_orchestrations.project_id",
                "fact_extraction_orchestrations.extraction_run_id",
            ],
            name="fk_projver_orch_proj_run_feo",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            [
                "consistency_check_application_id",
                "project_id",
                "orchestration_id",
                "source_consistency_application_id",
            ],
            [
                "consistency_check_applications.id",
                "consistency_check_applications.project_id",
                "consistency_check_applications.orchestration_id",
                "consistency_check_applications.consistency_application_id",
            ],
            name="fk_projver_ccapp_proj_orch_src",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["copied_from_version_id", "project_id"],
            ["project_versions.id", "project_versions.project_id"],
            name="fk_projver_prev_ver_projver",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_project_versions")),
        sa.UniqueConstraint("project_id", "version_no", name="uq_projver_project_verno"),
        sa.UniqueConstraint("version_manifest_hash", name="uq_projver_manifest_hash"),
        sa.UniqueConstraint("id", "project_id", name="uq_projver_id_project"),
        sa.CheckConstraint("version_no > 0", name="projver_ver_no_pos"),
        sa.CheckConstraint(
            "creation_kind IN ('manual', 'automatic', 'pre_publish', 'rollback')",
            name="projver_creation_kind_valid",
        ),
        sa.CheckConstraint(
            "reason IS NULL OR char_length(reason) BETWEEN 1 AND 2000",
            name="projver_reason_len",
        ),
        sa.CheckConstraint(
            "char_length(knowledge_view_algorithm_name) BETWEEN 1 AND 64",
            name="projver_alg_name_len",
        ),
        sa.CheckConstraint(
            "char_length(knowledge_view_algorithm_version) BETWEEN 1 AND 32",
            name="projver_alg_ver_len",
        ),
        sa.CheckConstraint(
            "char_length(snapshot_format_version) BETWEEN 1 AND 32",
            name="projver_snap_fmt_len",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(snapshot_json) = 'object'",
            name="projver_snapshot_obj",
        ),
        sa.CheckConstraint(
            "schema_definition_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="projver_schema_manifest_fmt",
        ),
        sa.CheckConstraint(
            "ufl_source_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="projver_ufl_manifest_fmt",
        ),
        sa.CheckConstraint(
            "consistency_result_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="projver_cons_manifest_fmt",
        ),
        sa.CheckConstraint(
            "raw_projection_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="projver_raw_manifest_fmt",
        ),
        sa.CheckConstraint(
            "reviewed_projection_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="projver_review_manifest_fmt",
        ),
        sa.CheckConstraint(
            "knowledge_view_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="projver_know_manifest_fmt",
        ),
        sa.CheckConstraint(
            "snapshot_json_hash ~ '^[0-9a-f]{64}$'",
            name="projver_snapshot_hash_fmt",
        ),
        sa.CheckConstraint(
            "version_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="projver_ver_manifest_fmt",
        ),
        sa.CheckConstraint("record_count >= 0", name="projver_record_count_nn"),
        sa.CheckConstraint("section_count >= 0", name="projver_section_count_nn"),
        sa.CheckConstraint("field_count >= 0", name="projver_field_count_nn"),
        sa.CheckConstraint(
            "missing_field_count >= 0",
            name="projver_missing_count_nn",
        ),
        sa.CheckConstraint(
            "review_required_field_count >= 0",
            name="projver_review_req_count_nn",
        ),
        sa.CheckConstraint(
            "resolved_field_count >= 0",
            name="projver_resolved_count_nn",
        ),
        sa.CheckConstraint(
            "observation_only_field_count >= 0",
            name="projver_observe_count_nn",
        ),
        sa.CheckConstraint(
            "mixed_field_count >= 0",
            name="projver_mixed_count_nn",
        ),
        sa.CheckConstraint(
            "missing_field_count + review_required_field_count + resolved_field_count + "
            "observation_only_field_count + mixed_field_count = field_count",
            name="projver_field_count_sum",
        ),
        sa.CheckConstraint(
            "("
            "creation_kind = 'rollback' AND copied_from_version_id IS NOT NULL"
            ") OR ("
            "creation_kind <> 'rollback' AND copied_from_version_id IS NULL"
            ")",
            name="projver_rollback_shape",
        ),
    )
    op.create_index(
        "ix_project_versions_project_id",
        "project_versions",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        "ix_project_versions_created_by_id",
        "project_versions",
        ["created_by_id"],
        unique=False,
    )
    op.create_index(
        "ix_project_versions_copied_from_version_id",
        "project_versions",
        ["copied_from_version_id"],
        unique=False,
    )
    op.create_index(
        "ix_project_versions_schema_id",
        "project_versions",
        ["schema_id"],
        unique=False,
    )
    op.create_index(
        "ix_project_versions_schema_version_id",
        "project_versions",
        ["schema_version_id"],
        unique=False,
    )
    op.create_index(
        "ix_project_versions_orchestration_id",
        "project_versions",
        ["orchestration_id"],
        unique=False,
    )
    op.create_index(
        "ix_project_versions_consistency_check_application_id",
        "project_versions",
        ["consistency_check_application_id"],
        unique=False,
    )

    op.add_column(
        "projects",
        sa.Column("current_version_id", sa.Uuid(), nullable=True),
    )
    op.create_index(
        "ix_projects_current_version_id",
        "projects",
        ["current_version_id"],
        unique=False,
    )
    op.create_foreign_key(
        "fk_projects_cur_ver_projver",
        "projects",
        "project_versions",
        ["current_version_id", "id"],
        ["id", "project_id"],
        ondelete="RESTRICT",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_projects_cur_ver_projver",
        "projects",
        type_="foreignkey",
    )
    op.drop_index("ix_projects_current_version_id", table_name="projects")
    op.drop_column("projects", "current_version_id")

    op.drop_index(
        "ix_project_versions_consistency_check_application_id",
        table_name="project_versions",
    )
    op.drop_index("ix_project_versions_orchestration_id", table_name="project_versions")
    op.drop_index("ix_project_versions_schema_version_id", table_name="project_versions")
    op.drop_index("ix_project_versions_schema_id", table_name="project_versions")
    op.drop_index(
        "ix_project_versions_copied_from_version_id",
        table_name="project_versions",
    )
    op.drop_index("ix_project_versions_created_by_id", table_name="project_versions")
    op.drop_index("ix_project_versions_project_id", table_name="project_versions")
    op.drop_table("project_versions")

    op.drop_constraint(
        "uq_ccapp_id_proj_orch_src",
        "consistency_check_applications",
        type_="unique",
    )
    op.drop_constraint(
        "uq_feo_id_proj_run",
        "fact_extraction_orchestrations",
        type_="unique",
    )
    op.drop_constraint(
        "uq_dynsver_id_schema",
        "dynamic_schema_versions",
        type_="unique",
    )
    op.drop_constraint(
        "uq_dynschema_id_project",
        "dynamic_schemas",
        type_="unique",
    )
