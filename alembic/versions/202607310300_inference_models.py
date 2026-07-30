"""inference input batch, block, and run models"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "202607310300"
down_revision: str | None = "202607302100"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "inference_input_batches",
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("task_type", sa.String(length=32), nullable=False),
        sa.Column("selection_strategy", sa.String(length=64), nullable=False),
        sa.Column(
            "selection_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("block_count", sa.Integer(), nullable=False),
        sa.Column("character_count", sa.Integer(), nullable=False),
        sa.Column("snapshot_hash", sa.String(length=64), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint(
            "task_type IN ('fact_extraction', 'schema_inference', 'consistency_check')",
            name=op.f("ck_inference_input_batches_inference_input_batches_task_type_valid"),
        ),
        sa.CheckConstraint(
            "block_count > 0",
            name=op.f("ck_inference_input_batches_inference_input_batches_block_count_positive"),
        ),
        sa.CheckConstraint(
            "character_count > 0",
            name=op.f("ck_inference_input_batches_inference_input_batches_character_count_positive"),
        ),
        sa.CheckConstraint(
            "snapshot_hash ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_inference_input_batches_inference_input_batches_snapshot_hash_format"),
        ),
        sa.CheckConstraint(
            "char_length(selection_strategy) BETWEEN 1 AND 64",
            name=op.f("ck_inference_input_batches_inference_input_batches_selection_strategy_length"),
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_inference_input_batches_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_inference_input_batches")),
        sa.UniqueConstraint(
            "project_id",
            "task_type",
            "snapshot_hash",
            name="uq_inference_input_batches_project_id_task_type_snapshot_hash",
        ),
    )
    op.create_index(
        op.f("ix_inference_input_batches_project_id"),
        "inference_input_batches",
        ["project_id"],
        unique=False,
    )

    op.create_table(
        "inference_input_blocks",
        sa.Column("batch_id", sa.Uuid(), nullable=False),
        sa.Column("source_order", sa.Integer(), nullable=False),
        sa.Column("block_ref", sa.String(length=16), nullable=False),
        sa.Column("document_block_id", sa.Uuid(), nullable=True),
        sa.Column("source_block_id_snapshot", sa.Uuid(), nullable=False),
        sa.Column("extraction_run_id_snapshot", sa.Uuid(), nullable=False),
        sa.Column("block_type", sa.String(length=16), nullable=False),
        sa.Column("location_key", sa.String(length=255), nullable=False),
        sa.Column("anchor_hash", sa.String(length=64), nullable=False),
        sa.Column("page_no", sa.Integer(), nullable=True),
        sa.Column("start_line", sa.Integer(), nullable=True),
        sa.Column("end_line", sa.Integer(), nullable=True),
        sa.Column(
            "heading_path",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("content_text", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint(
            "source_order >= 0",
            name=op.f("ck_inference_input_blocks_inference_input_blocks_source_order_non_negative"),
        ),
        sa.CheckConstraint(
            "block_ref ~ '^B[0-9]{4,}$'",
            name=op.f("ck_inference_input_blocks_inference_input_blocks_block_ref_format"),
        ),
        sa.CheckConstraint(
            "block_type IN ('heading', 'paragraph', 'list_item', 'table_row', 'code', 'page_text')",
            name=op.f("ck_inference_input_blocks_inference_input_blocks_block_type_valid"),
        ),
        sa.CheckConstraint(
            "char_length(location_key) > 0",
            name=op.f("ck_inference_input_blocks_inference_input_blocks_location_key_non_empty"),
        ),
        sa.CheckConstraint(
            "anchor_hash ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_inference_input_blocks_inference_input_blocks_anchor_hash_format"),
        ),
        sa.CheckConstraint(
            "char_length(content_text) > 0",
            name=op.f("ck_inference_input_blocks_inference_input_blocks_content_text_non_empty"),
        ),
        sa.CheckConstraint(
            "content_hash ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_inference_input_blocks_inference_input_blocks_content_hash_format"),
        ),
        sa.CheckConstraint(
            "page_no IS NULL OR page_no > 0",
            name=op.f("ck_inference_input_blocks_inference_input_blocks_page_no_positive"),
        ),
        sa.CheckConstraint(
            "start_line IS NULL OR start_line > 0",
            name=op.f("ck_inference_input_blocks_inference_input_blocks_start_line_positive"),
        ),
        sa.CheckConstraint(
            "end_line IS NULL OR end_line > 0",
            name=op.f("ck_inference_input_blocks_inference_input_blocks_end_line_positive"),
        ),
        sa.ForeignKeyConstraint(
            ["batch_id"],
            ["inference_input_batches.id"],
            name=op.f("fk_inference_input_blocks_batch_id_inference_input_batches"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["document_block_id"],
            ["document_blocks.id"],
            name=op.f("fk_inference_input_blocks_document_block_id_document_blocks"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_inference_input_blocks")),
        sa.UniqueConstraint(
            "batch_id",
            "source_order",
            name="uq_inference_input_blocks_batch_id_source_order",
        ),
        sa.UniqueConstraint(
            "batch_id",
            "block_ref",
            name="uq_inference_input_blocks_batch_id_block_ref",
        ),
        sa.UniqueConstraint(
            "batch_id",
            "source_block_id_snapshot",
            name="uq_inference_input_blocks_batch_id_source_block_id_snapshot",
        ),
    )
    op.create_index(
        op.f("ix_inference_input_blocks_batch_id"),
        "inference_input_blocks",
        ["batch_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_inference_input_blocks_document_block_id"),
        "inference_input_blocks",
        ["document_block_id"],
        unique=False,
    )

    op.create_table(
        "inference_runs",
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("input_batch_id", sa.Uuid(), nullable=False),
        sa.Column("task_type", sa.String(length=32), nullable=False),
        sa.Column("attempt_no", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("agent_name", sa.String(length=100), nullable=False),
        sa.Column("agent_version", sa.String(length=32), nullable=False),
        sa.Column("prompt_name", sa.String(length=100), nullable=False),
        sa.Column("prompt_version", sa.String(length=32), nullable=False),
        sa.Column("prompt_contract_hash", sa.String(length=64), nullable=True),
        sa.Column("request_hash", sa.String(length=64), nullable=True),
        sa.Column(
            "request_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("requested_model", sa.String(length=128), nullable=False),
        sa.Column("response_model", sa.String(length=128), nullable=True),
        sa.Column("response_id", sa.String(length=128), nullable=True),
        sa.Column("system_fingerprint", sa.String(length=128), nullable=True),
        sa.Column("finish_reason", sa.String(length=32), nullable=True),
        sa.Column("temperature", sa.Float(), nullable=False),
        sa.Column("max_output_tokens", sa.Integer(), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("prompt_cache_hit_tokens", sa.Integer(), nullable=True),
        sa.Column("prompt_cache_miss_tokens", sa.Integer(), nullable=True),
        sa.Column("reasoning_tokens", sa.Integer(), nullable=True),
        sa.Column("response_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("response_hash", sa.String(length=64), nullable=True),
        sa.Column("failure_code", sa.String(length=64), nullable=True),
        sa.Column("failure_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name=op.f("ck_inference_runs_inference_runs_status_valid"),
        ),
        sa.CheckConstraint(
            "task_type IN ('fact_extraction', 'schema_inference', 'consistency_check')",
            name=op.f("ck_inference_runs_inference_runs_task_type_valid"),
        ),
        sa.CheckConstraint(
            "attempt_no > 0",
            name=op.f("ck_inference_runs_inference_runs_attempt_no_positive"),
        ),
        sa.CheckConstraint(
            "attempt_count >= 0",
            name=op.f("ck_inference_runs_inference_runs_attempt_count_non_negative"),
        ),
        sa.CheckConstraint(
            "prompt_tokens IS NULL OR prompt_tokens >= 0",
            name=op.f("ck_inference_runs_inference_runs_prompt_tokens_non_negative"),
        ),
        sa.CheckConstraint(
            "completion_tokens IS NULL OR completion_tokens >= 0",
            name=op.f("ck_inference_runs_inference_runs_completion_tokens_non_negative"),
        ),
        sa.CheckConstraint(
            "total_tokens IS NULL OR total_tokens >= 0",
            name=op.f("ck_inference_runs_inference_runs_total_tokens_non_negative"),
        ),
        sa.CheckConstraint(
            "prompt_cache_hit_tokens IS NULL OR prompt_cache_hit_tokens >= 0",
            name=op.f("ck_inference_runs_inference_runs_prompt_cache_hit_tokens_non_negative"),
        ),
        sa.CheckConstraint(
            "prompt_cache_miss_tokens IS NULL OR prompt_cache_miss_tokens >= 0",
            name=op.f("ck_inference_runs_inference_runs_prompt_cache_miss_tokens_non_negative"),
        ),
        sa.CheckConstraint(
            "reasoning_tokens IS NULL OR reasoning_tokens >= 0",
            name=op.f("ck_inference_runs_inference_runs_reasoning_tokens_non_negative"),
        ),
        sa.CheckConstraint(
            "temperature >= 0 AND temperature <= 2",
            name=op.f("ck_inference_runs_inference_runs_temperature_range"),
        ),
        sa.CheckConstraint(
            "max_output_tokens > 0",
            name=op.f("ck_inference_runs_inference_runs_max_output_tokens_positive"),
        ),
        sa.CheckConstraint(
            "prompt_contract_hash IS NULL OR prompt_contract_hash ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_inference_runs_inference_runs_prompt_contract_hash_format"),
        ),
        sa.CheckConstraint(
            "request_hash IS NULL OR request_hash ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_inference_runs_inference_runs_request_hash_format"),
        ),
        sa.CheckConstraint(
            "response_hash IS NULL OR response_hash ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_inference_runs_inference_runs_response_hash_format"),
        ),
        sa.CheckConstraint(
            "response_json IS NULL OR jsonb_typeof(response_json) = 'object'",
            name=op.f("ck_inference_runs_inference_runs_response_json_is_object"),
        ),
        sa.CheckConstraint(
            "completed_at IS NULL OR started_at IS NULL OR completed_at >= started_at",
            name=op.f("ck_inference_runs_inference_runs_completed_after_started"),
        ),
        sa.CheckConstraint(
            "status <> 'pending' OR ("
            "started_at IS NULL AND completed_at IS NULL AND response_json IS NULL "
            "AND response_hash IS NULL AND failure_code IS NULL AND failure_message IS NULL "
            "AND attempt_count = 0)",
            name=op.f("ck_inference_runs_inference_runs_pending_shape"),
        ),
        sa.CheckConstraint(
            "status <> 'running' OR ("
            "started_at IS NOT NULL AND completed_at IS NULL AND response_json IS NULL "
            "AND response_hash IS NULL AND failure_code IS NULL AND failure_message IS NULL "
            "AND attempt_count = 0)",
            name=op.f("ck_inference_runs_inference_runs_running_shape"),
        ),
        sa.CheckConstraint(
            "status <> 'completed' OR ("
            "started_at IS NOT NULL AND completed_at IS NOT NULL AND finish_reason = 'stop' "
            "AND response_model IS NOT NULL AND response_json IS NOT NULL AND response_hash IS NOT NULL "
            "AND attempt_count > 0 AND failure_code IS NULL AND failure_message IS NULL)",
            name=op.f("ck_inference_runs_inference_runs_completed_shape"),
        ),
        sa.CheckConstraint(
            "status <> 'failed' OR ("
            "started_at IS NOT NULL AND completed_at IS NOT NULL AND failure_code IS NOT NULL "
            "AND response_json IS NULL AND response_hash IS NULL)",
            name=op.f("ck_inference_runs_inference_runs_failed_shape"),
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_inference_runs_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["input_batch_id"],
            ["inference_input_batches.id"],
            name=op.f("fk_inference_runs_input_batch_id_inference_input_batches"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_inference_runs")),
        sa.UniqueConstraint(
            "input_batch_id",
            "agent_name",
            "prompt_version",
            "attempt_no",
            name="uq_inference_runs_batch_agent_prompt_attempt",
        ),
    )
    op.create_index(
        op.f("ix_inference_runs_project_id"),
        "inference_runs",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_inference_runs_input_batch_id"),
        "inference_runs",
        ["input_batch_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_inference_runs_input_batch_id"), table_name="inference_runs")
    op.drop_index(op.f("ix_inference_runs_project_id"), table_name="inference_runs")
    op.drop_table("inference_runs")
    op.drop_index(
        op.f("ix_inference_input_blocks_document_block_id"),
        table_name="inference_input_blocks",
    )
    op.drop_index(
        op.f("ix_inference_input_blocks_batch_id"),
        table_name="inference_input_blocks",
    )
    op.drop_table("inference_input_blocks")
    op.drop_index(
        op.f("ix_inference_input_batches_project_id"),
        table_name="inference_input_batches",
    )
    op.drop_table("inference_input_batches")
