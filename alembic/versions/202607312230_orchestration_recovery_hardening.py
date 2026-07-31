"""Harden orchestration recovery constraints.

Revision ID: 202607312230
Revises: 202607312200
Create Date: 2026-08-01 00:30:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision: str = "202607312230"
down_revision: str | None = "202607312200"
branch_labels = None
depends_on = None


_FEO_PENDING_SQL = (
    "status <> 'partial' OR (started_at IS NOT NULL AND completed_at IS NOT NULL "
    "AND completed_batch_count > 0 AND failed_batch_count > 0 "
    "AND completed_batch_count + failed_batch_count = batch_count)"
)
_FEO_FAILED_SQL = (
    "status <> 'failed' OR (started_at IS NOT NULL AND completed_at IS NOT NULL "
    "AND completed_batch_count = 0 AND failed_batch_count = batch_count "
    "AND failure_code IS NOT NULL)"
)
_FEOB_PENDING_SQL = (
    "status <> 'pending' OR (current_inference_run_id IS NULL AND application_id IS NULL "
    "AND lease_token IS NULL AND lease_expires_at IS NULL "
    "AND completed_at IS NULL AND failure_code IS NULL)"
)


def _ensure_no_incompatible_orchestration_rows() -> None:
    connection = op.get_bind()
    incompatible_orchestration = connection.execute(
        sa.text(
            """
            SELECT id
            FROM fact_extraction_orchestrations
            WHERE completed_batch_count + failed_batch_count > batch_count
               OR (status = 'partial' AND (
                    completed_batch_count <= 0
                    OR failed_batch_count <= 0
                    OR completed_batch_count + failed_batch_count <> batch_count
               ))
               OR (status = 'failed' AND (
                    completed_batch_count <> 0
                    OR failed_batch_count <> batch_count
               ))
            LIMIT 1
            """
        )
    ).first()
    if incompatible_orchestration is not None:
        raise RuntimeError("Cannot apply 202607312230: incompatible orchestration terminal counts exist.")

    incompatible_batch = connection.execute(
        sa.text(
            """
            SELECT id
            FROM fact_extraction_orch_batches
            WHERE status = 'pending'
              AND current_inference_run_id IS NOT NULL
            LIMIT 1
            """
        )
    ).first()
    if incompatible_batch is not None:
        raise RuntimeError("Cannot apply 202607312230: pending batch rows still reference current_inference_run_id.")


def upgrade() -> None:
    _ensure_no_incompatible_orchestration_rows()

    op.drop_constraint("feo_partial_shape", "fact_extraction_orchestrations", type_="check")
    op.drop_constraint("feo_failed_shape", "fact_extraction_orchestrations", type_="check")
    op.create_check_constraint(
        "feo_terminal_batch_counts_within_batch_count",
        "fact_extraction_orchestrations",
        "completed_batch_count + failed_batch_count <= batch_count",
    )
    op.create_check_constraint(
        "feo_partial_shape",
        "fact_extraction_orchestrations",
        _FEO_PENDING_SQL,
    )
    op.create_check_constraint(
        "feo_failed_shape",
        "fact_extraction_orchestrations",
        _FEO_FAILED_SQL,
    )

    op.drop_constraint("feob_pending_shape", "fact_extraction_orch_batches", type_="check")
    op.create_check_constraint(
        "feob_pending_shape",
        "fact_extraction_orch_batches",
        _FEOB_PENDING_SQL,
    )


def downgrade() -> None:
    op.drop_constraint("feob_pending_shape", "fact_extraction_orch_batches", type_="check")
    op.create_check_constraint(
        "feob_pending_shape",
        "fact_extraction_orch_batches",
        "status <> 'pending' OR (lease_token IS NULL AND lease_expires_at IS NULL "
        "AND application_id IS NULL AND completed_at IS NULL AND failure_code IS NULL)",
    )

    op.drop_constraint("feo_failed_shape", "fact_extraction_orchestrations", type_="check")
    op.drop_constraint("feo_partial_shape", "fact_extraction_orchestrations", type_="check")
    op.drop_constraint(
        "feo_terminal_batch_counts_within_batch_count",
        "fact_extraction_orchestrations",
        type_="check",
    )
    op.create_check_constraint(
        "feo_partial_shape",
        "fact_extraction_orchestrations",
        "status <> 'partial' OR (started_at IS NOT NULL AND completed_at IS NOT NULL "
        "AND completed_batch_count > 0 AND failed_batch_count > 0)",
    )
    op.create_check_constraint(
        "feo_failed_shape",
        "fact_extraction_orchestrations",
        "status <> 'failed' OR (started_at IS NOT NULL AND completed_at IS NOT NULL "
        "AND completed_batch_count = 0 AND failed_batch_count > 0 "
        "AND failure_code IS NOT NULL)",
    )
