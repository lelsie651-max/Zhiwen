"""Add Agent 2 consistency check ledger tables.

Revision ID: 202608010400
Revises: 202608010300
Create Date: 2026-08-01 04:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "202608010400"
down_revision: str | None = "202608010300"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "consistency_check_applications",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("consistency_application_id", sa.Uuid(), nullable=False),
        sa.Column("source_result_manifest_hash", sa.String(length=64), nullable=False),
        sa.Column("plan_manifest_hash", sa.String(length=64), nullable=False),
        sa.Column("execution_identity_hash", sa.String(length=64), nullable=False),
        sa.Column("result_manifest_hash", sa.String(length=64), nullable=False),
        sa.Column("prompt_contract_hash", sa.String(length=64), nullable=False),
        sa.Column("provider", sa.String(length=128), nullable=False),
        sa.Column("requested_model", sa.String(length=128), nullable=False),
        sa.Column("executor_name", sa.String(length=64), nullable=False),
        sa.Column("executor_version", sa.String(length=32), nullable=False),
        sa.Column("batch_count", sa.Integer(), nullable=False),
        sa.Column("executed_batch_count", sa.Integer(), nullable=False),
        sa.Column("skipped_empty_batch_count", sa.Integer(), nullable=False),
        sa.Column("inference_run_count", sa.Integer(), nullable=False),
        sa.Column("assessment_count", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name="fk_ccapp_project_id_projects",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["consistency_application_id"],
            ["fact_value_consistency_candidate_applications.id"],
            name="fk_ccapp_srcapp_fvcca",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_consistency_check_applications")),
        sa.UniqueConstraint("execution_identity_hash", name="uq_ccapp_exec_identity_hash"),
        sa.UniqueConstraint("id", "consistency_application_id", name="uq_ccapp_id_srcapp"),
        sa.CheckConstraint(
            "source_result_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="src_result_hash_fmt",
        ),
        sa.CheckConstraint(
            "plan_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="plan_manifest_hash_fmt",
        ),
        sa.CheckConstraint(
            "execution_identity_hash ~ '^[0-9a-f]{64}$'",
            name="exec_identity_hash_fmt",
        ),
        sa.CheckConstraint(
            "result_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="result_manifest_hash_fmt",
        ),
        sa.CheckConstraint(
            "prompt_contract_hash ~ '^[0-9a-f]{64}$'",
            name="prompt_contract_hash_fmt",
        ),
        sa.CheckConstraint("char_length(provider) BETWEEN 1 AND 128", name="provider_len"),
        sa.CheckConstraint(
            "char_length(requested_model) BETWEEN 1 AND 128",
            name="requested_model_len",
        ),
        sa.CheckConstraint(
            "char_length(executor_name) BETWEEN 1 AND 64",
            name="executor_name_len",
        ),
        sa.CheckConstraint(
            "char_length(executor_version) BETWEEN 1 AND 32",
            name="executor_version_len",
        ),
        sa.CheckConstraint("batch_count > 0", name="batch_count_pos"),
        sa.CheckConstraint("executed_batch_count >= 0", name="executed_count_nn"),
        sa.CheckConstraint(
            "skipped_empty_batch_count >= 0",
            name="skipped_count_nn",
        ),
        sa.CheckConstraint("inference_run_count >= 0", name="run_count_nn"),
        sa.CheckConstraint("assessment_count >= 0", name="assessment_count_nn"),
        sa.CheckConstraint(
            "executed_batch_count <= batch_count",
            name="executed_count_lte_batch",
        ),
        sa.CheckConstraint(
            "skipped_empty_batch_count <= executed_batch_count",
            name="skipped_count_lte_executed",
        ),
        sa.CheckConstraint(
            "inference_run_count <= executed_batch_count",
            name="run_count_lte_executed",
        ),
    )
    op.create_index("ix_ccapp_project_id", "consistency_check_applications", ["project_id"], unique=False)
    op.create_index(
        "ix_ccapp_consistency_application_id",
        "consistency_check_applications",
        ["consistency_application_id"],
        unique=False,
    )

    op.create_table(
        "consistency_check_batches",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("consistency_check_application_id", sa.Uuid(), nullable=False),
        sa.Column("batch_index", sa.Integer(), nullable=False),
        sa.Column("batch_manifest_hash", sa.String(length=64), nullable=False),
        sa.Column("skipped_empty", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("input_batch_id", sa.Uuid(), nullable=True),
        sa.Column("inference_run_id", sa.Uuid(), nullable=True),
        sa.Column("request_hash", sa.String(length=64), nullable=True),
        sa.Column("message_content_hash", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(
            ["consistency_check_application_id"],
            ["consistency_check_applications.id"],
            name="fk_ccbatch_app_id_ccapp",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["input_batch_id"],
            ["inference_input_batches.id"],
            name="fk_ccbatch_input_batch_id_iib",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["inference_run_id"],
            ["inference_runs.id"],
            name="fk_ccbatch_inference_run_id_ir",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_consistency_check_batches")),
        sa.UniqueConstraint(
            "consistency_check_application_id",
            "batch_index",
            name="uq_ccbatch_app_batch_index",
        ),
        sa.UniqueConstraint(
            "consistency_check_application_id",
            "inference_run_id",
            name="uq_ccbatch_app_inference_run_id",
        ),
        sa.CheckConstraint("batch_index >= 0", name="batch_index_nn"),
        sa.CheckConstraint(
            "batch_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="batch_manifest_hash_fmt",
        ),
        sa.CheckConstraint(
            "request_hash IS NULL OR request_hash ~ '^[0-9a-f]{64}$'",
            name="request_hash_fmt",
        ),
        sa.CheckConstraint(
            "message_content_hash IS NULL OR message_content_hash ~ '^[0-9a-f]{64}$'",
            name="message_content_hash_fmt",
        ),
        sa.CheckConstraint(
            "("
            "skipped_empty = true "
            "AND input_batch_id IS NULL "
            "AND inference_run_id IS NULL "
            "AND request_hash IS NULL "
            "AND message_content_hash IS NULL"
            ") OR ("
            "skipped_empty = false "
            "AND input_batch_id IS NOT NULL "
            "AND inference_run_id IS NOT NULL "
            "AND request_hash IS NOT NULL "
            "AND message_content_hash IS NOT NULL"
            ")",
            name="batch_shape_valid",
        ),
    )
    op.create_index(
        "ix_ccbatch_consistency_check_application_id",
        "consistency_check_batches",
        ["consistency_check_application_id"],
        unique=False,
    )
    op.create_index("ix_ccbatch_input_batch_id", "consistency_check_batches", ["input_batch_id"], unique=False)
    op.create_index(
        "ix_ccbatch_inference_run_id",
        "consistency_check_batches",
        ["inference_run_id"],
        unique=False,
    )

    op.create_table(
        "consistency_assessments",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("consistency_check_application_id", sa.Uuid(), nullable=False),
        sa.Column("source_consistency_application_id", sa.Uuid(), nullable=False),
        sa.Column("source_consistency_candidate_id", sa.Uuid(), nullable=False),
        sa.Column("batch_index", sa.Integer(), nullable=False),
        sa.Column("verdict", sa.String(length=32), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("explanation", sa.Text(), nullable=False),
        sa.Column(
            "impact_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "recommended_actions_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("assessment_manifest_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(
            ["consistency_check_application_id", "source_consistency_application_id"],
            [
                "consistency_check_applications.id",
                "consistency_check_applications.consistency_application_id",
            ],
            name="fk_ccasmt_app_srcapp_ccapp",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["source_consistency_candidate_id", "source_consistency_application_id"],
            [
                "fact_value_consistency_candidates.id",
                "fact_value_consistency_candidates.consistency_application_id",
            ],
            name="fk_ccasmt_candidate_srcapp_fvcc",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["consistency_check_application_id", "batch_index"],
            [
                "consistency_check_batches.consistency_check_application_id",
                "consistency_check_batches.batch_index",
            ],
            name="fk_ccasmt_app_batch_index_ccbatch",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_consistency_assessments")),
        sa.UniqueConstraint(
            "consistency_check_application_id",
            "source_consistency_candidate_id",
            name="uq_ccasmt_app_candidate_id",
        ),
        sa.CheckConstraint("batch_index >= 0", name="batch_index_nn"),
        sa.CheckConstraint(
            "verdict IN ('conflict', 'compatible', 'insufficient_evidence')",
            name="verdict_valid",
        ),
        sa.CheckConstraint(
            "severity IN ('red', 'yellow', 'none')",
            name="severity_valid",
        ),
        sa.CheckConstraint(
            "("
            "verdict = 'conflict' AND severity IN ('red', 'yellow')"
            ") OR ("
            "verdict IN ('compatible', 'insufficient_evidence') AND severity = 'none'"
            ")",
            name="verdict_severity_pair_valid",
        ),
        sa.CheckConstraint(
            "confidence >= 0.0 "
            "AND confidence <= 1.0 "
            "AND CAST(confidence AS TEXT) NOT IN ('NaN', 'Infinity', '-Infinity')",
            name="confidence_valid",
        ),
        sa.CheckConstraint("char_length(explanation) > 0", name="explanation_non_empty"),
        sa.CheckConstraint("jsonb_typeof(impact_json) = 'array'", name="impact_json_is_array"),
        sa.CheckConstraint(
            "jsonb_typeof(recommended_actions_json) = 'array'",
            name="recommended_actions_json_is_array",
        ),
        sa.CheckConstraint(
            "assessment_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="assessment_manifest_hash_fmt",
        ),
    )
    op.create_index(
        "ix_ccasmt_consistency_check_application_id",
        "consistency_assessments",
        ["consistency_check_application_id"],
        unique=False,
    )
    op.create_index(
        "ix_ccasmt_source_consistency_application_id",
        "consistency_assessments",
        ["source_consistency_application_id"],
        unique=False,
    )
    op.create_index(
        "ix_ccasmt_source_consistency_candidate_id",
        "consistency_assessments",
        ["source_consistency_candidate_id"],
        unique=False,
    )
    op.create_index("ix_ccasmt_batch_index", "consistency_assessments", ["batch_index"], unique=False)

    op.create_table(
        "consistency_assessment_citations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("assessment_id", sa.Uuid(), nullable=False),
        sa.Column("evidence_link_id", sa.Uuid(), nullable=False),
        sa.Column("citation_order", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(
            ["assessment_id"],
            ["consistency_assessments.id"],
            name="fk_cccite_assessment_id_ccasmt",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["evidence_link_id"],
            ["fact_evidence_links.id"],
            name="fk_cccite_evidence_link_id_fel",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_consistency_assessment_citations")),
        sa.UniqueConstraint(
            "assessment_id",
            "evidence_link_id",
            name="uq_cccite_assessment_evidence_link_id",
        ),
        sa.UniqueConstraint(
            "assessment_id",
            "citation_order",
            name="uq_cccite_assessment_citation_order",
        ),
        sa.CheckConstraint("citation_order >= 0", name="citation_order_nn"),
    )
    op.create_index(
        "ix_cccite_assessment_id",
        "consistency_assessment_citations",
        ["assessment_id"],
        unique=False,
    )
    op.create_index(
        "ix_cccite_evidence_link_id",
        "consistency_assessment_citations",
        ["evidence_link_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_cccite_evidence_link_id", table_name="consistency_assessment_citations")
    op.drop_index("ix_cccite_assessment_id", table_name="consistency_assessment_citations")
    op.drop_table("consistency_assessment_citations")

    op.drop_index("ix_ccasmt_batch_index", table_name="consistency_assessments")
    op.drop_index(
        "ix_ccasmt_source_consistency_candidate_id",
        table_name="consistency_assessments",
    )
    op.drop_index(
        "ix_ccasmt_source_consistency_application_id",
        table_name="consistency_assessments",
    )
    op.drop_index(
        "ix_ccasmt_consistency_check_application_id",
        table_name="consistency_assessments",
    )
    op.drop_table("consistency_assessments")

    op.drop_index("ix_ccbatch_inference_run_id", table_name="consistency_check_batches")
    op.drop_index("ix_ccbatch_input_batch_id", table_name="consistency_check_batches")
    op.drop_index(
        "ix_ccbatch_consistency_check_application_id",
        table_name="consistency_check_batches",
    )
    op.drop_table("consistency_check_batches")

    op.drop_index(
        "ix_ccapp_consistency_application_id",
        table_name="consistency_check_applications",
    )
    op.drop_index("ix_ccapp_project_id", table_name="consistency_check_applications")
    op.drop_table("consistency_check_applications")
