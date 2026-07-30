"""inference invariant hardening: strict pending/running shapes and jsonb types"""

from collections.abc import Sequence

from alembic import op


revision: str = "202607310330"
down_revision: str | None = "202607310300"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


_RESPONSE_NULLS = (
    "response_json IS NULL AND response_hash IS NULL "
    "AND response_model IS NULL AND response_id IS NULL "
    "AND system_fingerprint IS NULL AND finish_reason IS NULL "
    "AND prompt_tokens IS NULL AND completion_tokens IS NULL "
    "AND total_tokens IS NULL AND prompt_cache_hit_tokens IS NULL "
    "AND prompt_cache_miss_tokens IS NULL AND reasoning_tokens IS NULL"
)
_NEW_PENDING = (
    "status <> 'pending' OR ("
    "started_at IS NULL AND completed_at IS NULL "
    "AND failure_code IS NULL AND failure_message IS NULL "
    "AND attempt_count = 0 "
    f"AND {_RESPONSE_NULLS})"
)
_NEW_RUNNING = (
    "status <> 'running' OR ("
    "started_at IS NOT NULL AND completed_at IS NULL "
    "AND failure_code IS NULL AND failure_message IS NULL "
    "AND attempt_count = 0 "
    f"AND {_RESPONSE_NULLS})"
)
_OLD_PENDING = (
    "status <> 'pending' OR ("
    "started_at IS NULL AND completed_at IS NULL AND response_json IS NULL "
    "AND response_hash IS NULL AND failure_code IS NULL AND failure_message IS NULL "
    "AND attempt_count = 0)"
)
_OLD_RUNNING = (
    "status <> 'running' OR ("
    "started_at IS NOT NULL AND completed_at IS NULL AND response_json IS NULL "
    "AND response_hash IS NULL AND failure_code IS NULL AND failure_message IS NULL "
    "AND attempt_count = 0)"
)

_PENDING_NAME = "ck_inference_runs_inference_runs_pending_shape"
_RUNNING_NAME = "ck_inference_runs_inference_runs_running_shape"
_BATCH_META_NAME = (
    "ck_inference_input_batches_inference_input_batches_selection_metadata_is_object"
)
_RUN_META_NAME = "ck_inference_runs_inference_runs_request_metadata_is_object"
_HEADING_NAME = (
    "ck_inference_input_blocks_inference_input_blocks_heading_path_is_array"
)


def upgrade() -> None:
    # Tighten the non-terminal state shapes so pending/running rows can never
    # carry response identity or token accounting.
    op.drop_constraint(op.f(_PENDING_NAME), "inference_runs", type_="check")
    op.drop_constraint(op.f(_RUNNING_NAME), "inference_runs", type_="check")
    op.create_check_constraint(op.f(_PENDING_NAME), "inference_runs", _NEW_PENDING)
    op.create_check_constraint(op.f(_RUNNING_NAME), "inference_runs", _NEW_RUNNING)

    # Enforce JSONB container types at the database level.
    op.create_check_constraint(
        op.f(_BATCH_META_NAME),
        "inference_input_batches",
        "jsonb_typeof(selection_metadata) = 'object'",
    )
    op.create_check_constraint(
        op.f(_RUN_META_NAME),
        "inference_runs",
        "jsonb_typeof(request_metadata) = 'object'",
    )
    op.create_check_constraint(
        op.f(_HEADING_NAME),
        "inference_input_blocks",
        "jsonb_typeof(heading_path) = 'array'",
    )


def downgrade() -> None:
    op.drop_constraint(op.f(_HEADING_NAME), "inference_input_blocks", type_="check")
    op.drop_constraint(op.f(_RUN_META_NAME), "inference_runs", type_="check")
    op.drop_constraint(op.f(_BATCH_META_NAME), "inference_input_batches", type_="check")

    op.drop_constraint(op.f(_RUNNING_NAME), "inference_runs", type_="check")
    op.drop_constraint(op.f(_PENDING_NAME), "inference_runs", type_="check")
    op.create_check_constraint(op.f(_PENDING_NAME), "inference_runs", _OLD_PENDING)
    op.create_check_constraint(op.f(_RUNNING_NAME), "inference_runs", _OLD_RUNNING)
