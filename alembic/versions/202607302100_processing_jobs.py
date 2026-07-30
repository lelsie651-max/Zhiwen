"""processing job task layer"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "202607302100"
down_revision: str | None = "202607301800"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


ACTIVE_PROCESSING_JOB_INDEX_PREDICATE = "status IN ('queued','running')"


def upgrade() -> None:
    op.create_table(
        "processing_jobs",
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("revision_id", sa.Uuid(), nullable=False),
        sa.Column("job_type", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("trigger_kind", sa.String(length=16), nullable=False),
        sa.Column("attempt_no", sa.Integer(), nullable=False),
        sa.Column("requested_by_id", sa.Uuid(), nullable=True),
        sa.Column("lease_token", sa.Uuid(), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("result_extraction_run_id", sa.Uuid(), nullable=True),
        sa.Column("failure_code", sa.String(length=64), nullable=True),
        sa.Column("failure_message", sa.Text(), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
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
        sa.CheckConstraint(
            "attempt_no > 0",
            name=op.f("ck_processing_jobs_processing_jobs_attempt_no_positive"),
        ),
        sa.CheckConstraint(
            "job_type IN ('revision_extraction')",
            name=op.f("ck_processing_jobs_processing_jobs_job_type_valid"),
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'failed', 'cancelled')",
            name=op.f("ck_processing_jobs_processing_jobs_status_valid"),
        ),
        sa.CheckConstraint(
            "trigger_kind IN ('automatic', 'manual', 'retry', 'recovery')",
            name=op.f("ck_processing_jobs_processing_jobs_trigger_kind_valid"),
        ),
        sa.CheckConstraint(
            "(trigger_kind NOT IN ('manual', 'retry') OR requested_by_id IS NOT NULL)",
            name=op.f("ck_processing_jobs_processing_jobs_manual_retry_requires_requester"),
        ),
        sa.CheckConstraint(
            "lease_expires_at IS NULL OR started_at IS NULL OR lease_expires_at >= started_at",
            name=op.f("ck_processing_jobs_processing_jobs_lease_after_started"),
        ),
        sa.CheckConstraint(
            "completed_at IS NULL OR started_at IS NULL OR completed_at >= started_at",
            name=op.f("ck_processing_jobs_processing_jobs_completed_after_started"),
        ),
        sa.CheckConstraint(
            "("
            "status <> 'queued' OR "
            "("
            "lease_token IS NULL AND lease_expires_at IS NULL AND started_at IS NULL AND "
            "completed_at IS NULL AND result_extraction_run_id IS NULL AND "
            "failure_code IS NULL AND failure_message IS NULL"
            ")"
            ")",
            name=op.f("ck_processing_jobs_processing_jobs_queued_state_shape"),
        ),
        sa.CheckConstraint(
            "("
            "status <> 'running' OR "
            "("
            "lease_token IS NOT NULL AND lease_expires_at IS NOT NULL AND started_at IS NOT NULL AND "
            "completed_at IS NULL AND result_extraction_run_id IS NULL AND "
            "failure_code IS NULL AND failure_message IS NULL"
            ")"
            ")",
            name=op.f("ck_processing_jobs_processing_jobs_running_state_shape"),
        ),
        sa.CheckConstraint(
            "("
            "status <> 'completed' OR "
            "("
            "lease_token IS NULL AND lease_expires_at IS NULL AND started_at IS NOT NULL AND "
            "completed_at IS NOT NULL AND result_extraction_run_id IS NOT NULL AND "
            "failure_code IS NULL AND failure_message IS NULL"
            ")"
            ")",
            name=op.f("ck_processing_jobs_processing_jobs_completed_state_shape"),
        ),
        sa.CheckConstraint(
            "("
            "status <> 'failed' OR "
            "("
            "lease_token IS NULL AND lease_expires_at IS NULL AND started_at IS NOT NULL AND "
            "completed_at IS NOT NULL AND result_extraction_run_id IS NULL AND "
            "failure_code IS NOT NULL"
            ")"
            ")",
            name=op.f("ck_processing_jobs_processing_jobs_failed_state_shape"),
        ),
        sa.CheckConstraint(
            "("
            "status <> 'cancelled' OR "
            "("
            "lease_token IS NULL AND lease_expires_at IS NULL AND completed_at IS NOT NULL AND "
            "result_extraction_run_id IS NULL AND failure_code IS NULL AND failure_message IS NULL"
            ")"
            ")",
            name=op.f("ck_processing_jobs_processing_jobs_cancelled_state_shape"),
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_processing_jobs_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["revision_id"],
            ["document_revisions.id"],
            name=op.f("fk_processing_jobs_revision_id_document_revisions"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["requested_by_id"],
            ["users.id"],
            name=op.f("fk_processing_jobs_requested_by_id_users"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["result_extraction_run_id"],
            ["extraction_runs.id"],
            name=op.f("fk_processing_jobs_result_extraction_run_id_extraction_runs"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_processing_jobs")),
        sa.UniqueConstraint(
            "revision_id",
            "job_type",
            "attempt_no",
            name="uq_processing_jobs_revision_id_job_type_attempt_no",
        ),
    )
    op.create_index(op.f("ix_processing_jobs_project_id"), "processing_jobs", ["project_id"], unique=False)
    op.create_index(op.f("ix_processing_jobs_revision_id"), "processing_jobs", ["revision_id"], unique=False)
    op.create_index(
        op.f("ix_processing_jobs_requested_by_id"),
        "processing_jobs",
        ["requested_by_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_processing_jobs_result_extraction_run_id"),
        "processing_jobs",
        ["result_extraction_run_id"],
        unique=False,
    )
    op.create_index(
        "uq_processing_jobs_active_revision_id_job_type",
        "processing_jobs",
        ["revision_id", "job_type"],
        unique=True,
        postgresql_where=sa.text(ACTIVE_PROCESSING_JOB_INDEX_PREDICATE),
    )


def downgrade() -> None:
    op.drop_index("uq_processing_jobs_active_revision_id_job_type", table_name="processing_jobs")
    op.drop_index(op.f("ix_processing_jobs_result_extraction_run_id"), table_name="processing_jobs")
    op.drop_index(op.f("ix_processing_jobs_requested_by_id"), table_name="processing_jobs")
    op.drop_index(op.f("ix_processing_jobs_revision_id"), table_name="processing_jobs")
    op.drop_index(op.f("ix_processing_jobs_project_id"), table_name="processing_jobs")
    op.drop_table("processing_jobs")
