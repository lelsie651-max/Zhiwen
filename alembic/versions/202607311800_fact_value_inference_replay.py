"""Add unique replay constraint for AI fact values.

Revision ID: 202607311800
Revises: 202607311700
Create Date: 2026-07-31 18:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision: str = "202607311800"
down_revision: str | None = "202607311700"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM fact_values
                WHERE inference_run_id IS NOT NULL
                GROUP BY fact_id, inference_run_id, value_hash
                HAVING count(*) > 1
            ) THEN
                RAISE EXCEPTION
                    'Cannot create uq_fv_fact_ir_value_hash: duplicate AI fact values share the same fact_id, inference_run_id, and value_hash.';
            END IF;
        END
        $$;
        """
    )

    op.create_unique_constraint(
        "uq_fv_fact_ir_value_hash",
        "fact_values",
        ["fact_id", "inference_run_id", "value_hash"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_fv_fact_ir_value_hash", "fact_values", type_="unique")
