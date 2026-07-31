"""Add fact extraction orchestration tables.

Revision ID: 202607312200
Revises: 202607311900
Create Date: 2026-07-31 22:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "202607312200"
down_revision: str | None = "202607311900"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "fact_extraction_orchestrations",
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("extraction_run_id", sa.Uuid(), nullable=False),
        sa.Column("attempt_no", sa.Integer(), nullable=False),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("plan_hash", sa.String(length=64), nullable=False),
        sa.Column("plan_json_hash", sa.String(length=64), nullable=False),
        sa.Column("plan_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("coordinator_name", sa.String(length=64), nullable=False),
        sa.Column("coordinator_version", sa.String(length=32), nullable=False),
        sa.Column("planner_name", sa.String(length=64), nullable=False),
        sa.Column("planner_version", sa.String(length=32), nullable=False),
        sa.Column("agent_name", sa.String(length=100), nullable=False),
        sa.Column("agent_version", sa.String(length=32), nullable=False),
        sa.Column("prompt_name", sa.String(length=100), nullable=False),
        sa.Column("prompt_version", sa.String(length=32), nullable=False),
        sa.Column("prompt_contract_hash", sa.String(length=64), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("requested_model", sa.String(length=128), nullable=False),
        sa.Column("executor_name", sa.String(length=64), nullable=False),
        sa.Column("executor_version", sa.String(length=32), nullable=False),
        sa.Column("persistence_name", sa.String(length=64), nullable=False),
        sa.Column("persistence_version", sa.String(length=32), nullable=False),
        sa.Column("entity_resolution_policy_name", sa.String(length=64), nullable=False),
        sa.Column("entity_resolution_policy_version", sa.String(length=32), nullable=False),
        sa.Column("batch_count", sa.Integer(), nullable=False),
        sa.Column("completed_batch_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("failed_batch_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("proposal_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("created_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("reused_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("withheld_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("failure_code", sa.String(length=64), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(["extraction_run_id"], ["extraction_runs.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "extraction_run_id",
            "request_hash",
            "attempt_no",
            name="uq_feo_extraction_run_request_attempt",
        ),
        sa.CheckConstraint("status IN ('planned', 'running', 'completed', 'partial', 'failed')", name="feo_status_valid"),
        sa.CheckConstraint("attempt_no > 0", name="feo_attempt_no_positive"),
        sa.CheckConstraint("batch_count > 0", name="feo_batch_count_positive"),
        sa.CheckConstraint("completed_batch_count >= 0", name="feo_completed_batch_count_non_negative"),
        sa.CheckConstraint("failed_batch_count >= 0", name="feo_failed_batch_count_non_negative"),
        sa.CheckConstraint("proposal_count >= 0", name="feo_proposal_count_non_negative"),
        sa.CheckConstraint("created_count >= 0", name="feo_created_count_non_negative"),
        sa.CheckConstraint("reused_count >= 0", name="feo_reused_count_non_negative"),
        sa.CheckConstraint("withheld_count >= 0", name="feo_withheld_count_non_negative"),
        sa.CheckConstraint("request_hash ~ '^[0-9a-f]{64}$'", name="feo_request_hash_format"),
        sa.CheckConstraint("plan_hash ~ '^[0-9a-f]{64}$'", name="feo_plan_hash_format"),
        sa.CheckConstraint("plan_json_hash ~ '^[0-9a-f]{64}$'", name="feo_plan_json_hash_format"),
        sa.CheckConstraint("prompt_contract_hash ~ '^[0-9a-f]{64}$'", name="feo_prompt_contract_hash_format"),
        sa.CheckConstraint("jsonb_typeof(plan_json) = 'object'", name="feo_plan_json_is_object"),
        sa.CheckConstraint(
            "status <> 'planned' OR (started_at IS NULL AND completed_at IS NULL AND failure_code IS NULL AND completed_batch_count = 0 AND failed_batch_count = 0 AND proposal_count = 0 AND created_count = 0 AND reused_count = 0 AND withheld_count = 0)",
            name="feo_planned_shape",
        ),
        sa.CheckConstraint(
            "status <> 'running' OR (started_at IS NOT NULL AND completed_at IS NULL AND failure_code IS NULL)",
            name="feo_running_shape",
        ),
        sa.CheckConstraint(
            "status <> 'completed' OR (started_at IS NOT NULL AND completed_at IS NOT NULL AND completed_batch_count = batch_count AND failed_batch_count = 0 AND failure_code IS NULL)",
            name="feo_completed_shape",
        ),
        sa.CheckConstraint(
            "status <> 'partial' OR (started_at IS NOT NULL AND completed_at IS NOT NULL AND completed_batch_count > 0 AND failed_batch_count > 0)",
            name="feo_partial_shape",
        ),
        sa.CheckConstraint(
            "status <> 'failed' OR (started_at IS NOT NULL AND completed_at IS NOT NULL AND completed_batch_count = 0 AND failed_batch_count > 0 AND failure_code IS NOT NULL)",
            name="feo_failed_shape",
        ),
    )
    op.create_index(op.f("ix_fact_extraction_orchestrations_project_id"), "fact_extraction_orchestrations", ["project_id"], unique=False)
    op.create_index(op.f("ix_fact_extraction_orchestrations_extraction_run_id"), "fact_extraction_orchestrations", ["extraction_run_id"], unique=False)
    op.create_index(
        "uq_feo_active_request",
        "fact_extraction_orchestrations",
        ["extraction_run_id", "request_hash"],
        unique=True,
        postgresql_where=sa.text("status IN ('planned', 'running')"),
    )

    op.create_table(
        "fact_extraction_orch_batches",
        sa.Column("orchestration_id", sa.Uuid(), nullable=False),
        sa.Column("batch_index", sa.Integer(), nullable=False),
        sa.Column("batch_plan_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("current_input_batch_id", sa.Uuid(), nullable=True),
        sa.Column("current_inference_run_id", sa.Uuid(), nullable=True),
        sa.Column("application_id", sa.Uuid(), nullable=True),
        sa.Column("lease_token", sa.Uuid(), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("proposal_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("created_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("reused_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("withheld_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("failure_code", sa.String(length=64), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(["application_id"], ["fact_extraction_batch_applications.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["current_inference_run_id"], ["inference_runs.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["current_input_batch_id"], ["inference_input_batches.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["orchestration_id"], ["fact_extraction_orchestrations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("application_id", name="uq_feob_application_id"),
        sa.UniqueConstraint("orchestration_id", "batch_index", name="uq_feob_orchestration_batch_index"),
        sa.UniqueConstraint("orchestration_id", "current_inference_run_id", name="uq_feob_orchestration_inference_run"),
        sa.CheckConstraint("status IN ('pending', 'running', 'completed', 'failed')", name="feob_status_valid"),
        sa.CheckConstraint("batch_index >= 0", name="feob_batch_index_non_negative"),
        sa.CheckConstraint("attempt_count >= 0", name="feob_attempt_count_non_negative"),
        sa.CheckConstraint("proposal_count >= 0", name="feob_proposal_count_non_negative"),
        sa.CheckConstraint("created_count >= 0", name="feob_created_count_non_negative"),
        sa.CheckConstraint("reused_count >= 0", name="feob_reused_count_non_negative"),
        sa.CheckConstraint("withheld_count >= 0", name="feob_withheld_count_non_negative"),
        sa.CheckConstraint("batch_plan_hash ~ '^[0-9a-f]{64}$'", name="feob_batch_plan_hash_format"),
        sa.CheckConstraint(
            "(lease_token IS NULL AND lease_expires_at IS NULL) OR (lease_token IS NOT NULL AND lease_expires_at IS NOT NULL)",
            name="feob_lease_pair",
        ),
        sa.CheckConstraint(
            "status <> 'pending' OR (lease_token IS NULL AND lease_expires_at IS NULL AND application_id IS NULL AND completed_at IS NULL AND failure_code IS NULL)",
            name="feob_pending_shape",
        ),
        sa.CheckConstraint(
            "status <> 'running' OR (lease_token IS NOT NULL AND lease_expires_at IS NOT NULL AND started_at IS NOT NULL AND application_id IS NULL AND completed_at IS NULL)",
            name="feob_running_shape",
        ),
        sa.CheckConstraint(
            "status <> 'completed' OR (current_input_batch_id IS NOT NULL AND current_inference_run_id IS NOT NULL AND application_id IS NOT NULL AND started_at IS NOT NULL AND completed_at IS NOT NULL AND lease_token IS NULL AND lease_expires_at IS NULL AND failure_code IS NULL)",
            name="feob_completed_shape",
        ),
        sa.CheckConstraint(
            "status <> 'failed' OR (started_at IS NOT NULL AND completed_at IS NOT NULL AND failure_code IS NOT NULL AND lease_token IS NULL AND lease_expires_at IS NULL)",
            name="feob_failed_shape",
        ),
    )
    op.create_index(op.f("ix_fact_extraction_orch_batches_orchestration_id"), "fact_extraction_orch_batches", ["orchestration_id"], unique=False)
    op.create_index(op.f("ix_fact_extraction_orch_batches_current_input_batch_id"), "fact_extraction_orch_batches", ["current_input_batch_id"], unique=False)
    op.create_index(op.f("ix_fact_extraction_orch_batches_current_inference_run_id"), "fact_extraction_orch_batches", ["current_inference_run_id"], unique=False)
    op.create_index(op.f("ix_fact_extraction_orch_batches_application_id"), "fact_extraction_orch_batches", ["application_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_fact_extraction_orch_batches_application_id"), table_name="fact_extraction_orch_batches")
    op.drop_index(op.f("ix_fact_extraction_orch_batches_current_inference_run_id"), table_name="fact_extraction_orch_batches")
    op.drop_index(op.f("ix_fact_extraction_orch_batches_current_input_batch_id"), table_name="fact_extraction_orch_batches")
    op.drop_index(op.f("ix_fact_extraction_orch_batches_orchestration_id"), table_name="fact_extraction_orch_batches")
    op.drop_table("fact_extraction_orch_batches")

    op.drop_index("uq_feo_active_request", table_name="fact_extraction_orchestrations")
    op.drop_index(op.f("ix_fact_extraction_orchestrations_extraction_run_id"), table_name="fact_extraction_orchestrations")
    op.drop_index(op.f("ix_fact_extraction_orchestrations_project_id"), table_name="fact_extraction_orchestrations")
    op.drop_table("fact_extraction_orchestrations")
