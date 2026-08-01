"""Add inference response_json_hash and batch application ledger.

Revision ID: 202607311900
Revises: 202607311800
Create Date: 2026-07-31 19:00:00
"""

from __future__ import annotations

import hashlib
import json

from alembic import context, op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "202607311900"
down_revision: str | None = "202607311800"
branch_labels = None
depends_on = None


_RUN_PENDING_SHAPE_SQL = (
    "status <> 'pending' OR ("
    "started_at IS NULL AND completed_at IS NULL "
    "AND failure_code IS NULL AND failure_message IS NULL "
    "AND attempt_count = 0 "
    "AND response_json IS NULL AND response_hash IS NULL AND response_json_hash IS NULL "
    "AND response_model IS NULL AND response_id IS NULL "
    "AND system_fingerprint IS NULL AND finish_reason IS NULL "
    "AND prompt_tokens IS NULL AND completion_tokens IS NULL "
    "AND total_tokens IS NULL AND prompt_cache_hit_tokens IS NULL "
    "AND prompt_cache_miss_tokens IS NULL AND reasoning_tokens IS NULL)"
)
_RUN_RUNNING_SHAPE_SQL = (
    "status <> 'running' OR ("
    "started_at IS NOT NULL AND completed_at IS NULL "
    "AND failure_code IS NULL AND failure_message IS NULL "
    "AND attempt_count = 0 "
    "AND response_json IS NULL AND response_hash IS NULL AND response_json_hash IS NULL "
    "AND response_model IS NULL AND response_id IS NULL "
    "AND system_fingerprint IS NULL AND finish_reason IS NULL "
    "AND prompt_tokens IS NULL AND completion_tokens IS NULL "
    "AND total_tokens IS NULL AND prompt_cache_hit_tokens IS NULL "
    "AND prompt_cache_miss_tokens IS NULL AND reasoning_tokens IS NULL)"
)
_RUN_COMPLETED_SHAPE_SQL = (
    "status <> 'completed' OR ("
    "started_at IS NOT NULL AND completed_at IS NOT NULL AND finish_reason = 'stop' "
    "AND response_model IS NOT NULL AND response_json IS NOT NULL "
    "AND response_hash IS NOT NULL AND response_json_hash IS NOT NULL "
    "AND attempt_count > 0 AND failure_code IS NULL AND failure_message IS NULL)"
)
_RUN_FAILED_SHAPE_SQL = (
    "status <> 'failed' OR ("
    "started_at IS NOT NULL AND completed_at IS NOT NULL AND failure_code IS NOT NULL "
    "AND response_json IS NULL AND response_hash IS NULL AND response_json_hash IS NULL)"
)


def _build_canonical_json_hash(response_json: dict) -> str:
    if type(response_json) is not dict:
        raise RuntimeError("Cannot backfill response_json_hash: completed inference run response_json must be a JSON object.")
    for key in response_json.keys():
        if not isinstance(key, str):
            raise RuntimeError("Cannot backfill response_json_hash: response_json object keys must be strings.")
    payload = json.dumps(
        response_json,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def upgrade() -> None:
    if context.is_offline_mode():
        op.add_column(
            "inference_runs",
            sa.Column("response_json_hash", sa.String(length=64), nullable=True),
        )
        op.execute(
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM inference_runs
                    WHERE status = 'completed'
                ) THEN
                    RAISE EXCEPTION
                        'Cannot offline-apply 202607311900: completed inference runs require online response_json_hash backfill.';
                END IF;
            END
            $$;
            """
        )
    else:
        bind = op.get_bind()

        op.add_column(
            "inference_runs",
            sa.Column("response_json_hash", sa.String(length=64), nullable=True),
        )

        rows = bind.execute(
            sa.text(
                """
                SELECT id, response_json
                FROM inference_runs
                WHERE status = 'completed'
                """
            )
        ).mappings()
        for row in rows:
            response_json = row["response_json"]
            if response_json is None:
                raise RuntimeError(
                    "Cannot backfill response_json_hash: completed inference run is missing response_json."
                )
            response_json_hash = _build_canonical_json_hash(response_json)
            bind.execute(
                sa.text(
                    """
                    UPDATE inference_runs
                    SET response_json_hash = :response_json_hash
                    WHERE id = :run_id
                    """
                ),
                {
                    "response_json_hash": response_json_hash,
                    "run_id": row["id"],
                },
            )

    if context.is_offline_mode():
        op.execute(
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM fact_values
                    WHERE inference_run_id IS NOT NULL
                ) THEN
                    RAISE EXCEPTION
                        'Cannot offline-apply 202607311900: existing AI FactValue rows require explicit application backfill.';
                END IF;
            END
            $$;
            """
        )
    else:
        bind = op.get_bind()
        existing_ai_fact_value = bind.execute(
            sa.text(
                """
                SELECT 1
                FROM fact_values
                WHERE inference_run_id IS NOT NULL
                LIMIT 1
                """
            )
        ).first()
        if existing_ai_fact_value is not None:
            raise RuntimeError(
                "Cannot create fact_extraction_batch_applications: existing AI FactValue rows require explicit application backfill."
            )

    op.drop_constraint("inference_runs_pending_shape", "inference_runs", type_="check")
    op.drop_constraint("inference_runs_running_shape", "inference_runs", type_="check")
    op.drop_constraint("inference_runs_completed_shape", "inference_runs", type_="check")
    op.drop_constraint("inference_runs_failed_shape", "inference_runs", type_="check")

    op.create_check_constraint(
        "inference_runs_response_json_hash_format",
        "inference_runs",
        "response_json_hash IS NULL OR response_json_hash ~ '^[0-9a-f]{64}$'",
    )
    op.create_check_constraint(
        "inference_runs_pending_shape",
        "inference_runs",
        _RUN_PENDING_SHAPE_SQL,
    )
    op.create_check_constraint(
        "inference_runs_running_shape",
        "inference_runs",
        _RUN_RUNNING_SHAPE_SQL,
    )
    op.create_check_constraint(
        "inference_runs_completed_shape",
        "inference_runs",
        _RUN_COMPLETED_SHAPE_SQL,
    )
    op.create_check_constraint(
        "inference_runs_failed_shape",
        "inference_runs",
        _RUN_FAILED_SHAPE_SQL,
    )

    op.create_table(
        "fact_extraction_batch_applications",
        sa.Column("inference_run_id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("extraction_run_id", sa.Uuid(), nullable=False),
        sa.Column("input_batch_id", sa.Uuid(), nullable=False),
        sa.Column("response_hash", sa.String(length=64), nullable=False),
        sa.Column("response_json_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("persistence_name", sa.String(length=64), nullable=False),
        sa.Column("persistence_version", sa.String(length=32), nullable=False),
        sa.Column("entity_resolution_policy_name", sa.String(length=64), nullable=False),
        sa.Column("entity_resolution_policy_version", sa.String(length=32), nullable=False),
        sa.Column("result_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("result_hash", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(["extraction_run_id"], ["extraction_runs.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["inference_run_id"], ["inference_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["input_batch_id"], ["inference_input_batches.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("inference_run_id", name="uq_feba_inference_run_id"),
        sa.CheckConstraint("status IN ('applying', 'completed')", name="feba_status_valid"),
        sa.CheckConstraint("response_hash ~ '^[0-9a-f]{64}$'", name="feba_response_hash_format"),
        sa.CheckConstraint("response_json_hash ~ '^[0-9a-f]{64}$'", name="feba_response_json_hash_fmt"),
        sa.CheckConstraint(
            "result_hash IS NULL OR result_hash ~ '^[0-9a-f]{64}$'",
            name="feba_result_hash_format",
        ),
        sa.CheckConstraint(
            "result_json IS NULL OR jsonb_typeof(result_json) = 'object'",
            name="feba_result_json_is_object",
        ),
        sa.CheckConstraint(
            "status <> 'applying' OR (result_json IS NULL AND result_hash IS NULL AND completed_at IS NULL)",
            name="feba_applying_shape",
        ),
        sa.CheckConstraint(
            "status <> 'completed' OR (result_json IS NOT NULL AND result_hash IS NOT NULL AND completed_at IS NOT NULL)",
            name="feba_completed_shape",
        ),
    )
    op.create_index(
        op.f("ix_fact_extraction_batch_applications_inference_run_id"),
        "fact_extraction_batch_applications",
        ["inference_run_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_fact_extraction_batch_applications_project_id"),
        "fact_extraction_batch_applications",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_fact_extraction_batch_applications_extraction_run_id"),
        "fact_extraction_batch_applications",
        ["extraction_run_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_fact_extraction_batch_applications_input_batch_id"),
        "fact_extraction_batch_applications",
        ["input_batch_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_fact_extraction_batch_applications_input_batch_id"),
        table_name="fact_extraction_batch_applications",
    )
    op.drop_index(
        op.f("ix_fact_extraction_batch_applications_extraction_run_id"),
        table_name="fact_extraction_batch_applications",
    )
    op.drop_index(
        op.f("ix_fact_extraction_batch_applications_project_id"),
        table_name="fact_extraction_batch_applications",
    )
    op.drop_index(
        op.f("ix_fact_extraction_batch_applications_inference_run_id"),
        table_name="fact_extraction_batch_applications",
    )
    op.drop_table("fact_extraction_batch_applications")

    op.drop_constraint("inference_runs_pending_shape", "inference_runs", type_="check")
    op.drop_constraint("inference_runs_running_shape", "inference_runs", type_="check")
    op.drop_constraint("inference_runs_completed_shape", "inference_runs", type_="check")
    op.drop_constraint("inference_runs_failed_shape", "inference_runs", type_="check")
    op.drop_constraint("inference_runs_response_json_hash_format", "inference_runs", type_="check")

    op.create_check_constraint(
        "inference_runs_pending_shape",
        "inference_runs",
        "status <> 'pending' OR ("
        "started_at IS NULL AND completed_at IS NULL "
        "AND failure_code IS NULL AND failure_message IS NULL "
        "AND attempt_count = 0 "
        "AND response_json IS NULL AND response_hash IS NULL "
        "AND response_model IS NULL AND response_id IS NULL "
        "AND system_fingerprint IS NULL AND finish_reason IS NULL "
        "AND prompt_tokens IS NULL AND completion_tokens IS NULL "
        "AND total_tokens IS NULL AND prompt_cache_hit_tokens IS NULL "
        "AND prompt_cache_miss_tokens IS NULL AND reasoning_tokens IS NULL)",
    )
    op.create_check_constraint(
        "inference_runs_running_shape",
        "inference_runs",
        "status <> 'running' OR ("
        "started_at IS NOT NULL AND completed_at IS NULL "
        "AND failure_code IS NULL AND failure_message IS NULL "
        "AND attempt_count = 0 "
        "AND response_json IS NULL AND response_hash IS NULL "
        "AND response_model IS NULL AND response_id IS NULL "
        "AND system_fingerprint IS NULL AND finish_reason IS NULL "
        "AND prompt_tokens IS NULL AND completion_tokens IS NULL "
        "AND total_tokens IS NULL AND prompt_cache_hit_tokens IS NULL "
        "AND prompt_cache_miss_tokens IS NULL AND reasoning_tokens IS NULL)",
    )
    op.create_check_constraint(
        "inference_runs_completed_shape",
        "inference_runs",
        "status <> 'completed' OR ("
        "started_at IS NOT NULL AND completed_at IS NOT NULL AND finish_reason = 'stop' "
        "AND response_model IS NOT NULL AND response_json IS NOT NULL AND response_hash IS NOT NULL "
        "AND attempt_count > 0 AND failure_code IS NULL AND failure_message IS NULL)",
    )
    op.create_check_constraint(
        "inference_runs_failed_shape",
        "inference_runs",
        "status <> 'failed' OR ("
        "started_at IS NOT NULL AND completed_at IS NOT NULL AND failure_code IS NOT NULL "
        "AND response_json IS NULL AND response_hash IS NULL)",
    )
    op.drop_column("inference_runs", "response_json_hash")
