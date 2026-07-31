"""add active inference request partial unique index"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "202607311700"
down_revision: str | None = "202607311600"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    duplicate_active = bind.execute(
        sa.text(
            """
            SELECT input_batch_id, request_hash
            FROM inference_runs
            WHERE status IN ('pending', 'running')
            GROUP BY input_batch_id, request_hash
            HAVING count(*) > 1
            LIMIT 1
            """
        )
    ).first()
    if duplicate_active is not None:
        raise RuntimeError(
            "Cannot create uq_ir_active_request: "
            "multiple active inference runs share the same input_batch_id and request_hash."
        )

    op.create_index(
        "uq_ir_active_request",
        "inference_runs",
        ["input_batch_id", "request_hash"],
        unique=True,
        postgresql_where=sa.text("status IN ('pending', 'running')"),
    )


def downgrade() -> None:
    op.drop_index("uq_ir_active_request", table_name="inference_runs")
