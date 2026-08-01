"""add inference run provenance to fact values"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "202607311030"
down_revision: str | None = "202607310330"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


_OLD_AI_CHECK = "ck_fact_values_fact_values_ai_requires_extraction_run"
_NEW_AI_CHECK = "ck_fact_values_fact_values_ai_requires_extraction_and_inference_run"
_NON_AI_CHECK = "ck_fact_values_fact_values_non_ai_forbids_inference_run"


def upgrade() -> None:
    op.add_column(
        "fact_values",
        sa.Column("inference_run_id", sa.Uuid(), nullable=True),
    )
    op.create_index(
        op.f("ix_fact_values_inference_run_id"),
        "fact_values",
        ["inference_run_id"],
        unique=False,
    )
    op.create_foreign_key(
        op.f("fk_fact_values_inference_run_id_inference_runs"),
        "fact_values",
        "inference_runs",
        ["inference_run_id"],
        ["id"],
        ondelete="RESTRICT",
    )

    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM fact_values
                WHERE source_kind = 'ai'
            ) THEN
                RAISE EXCEPTION
                    'Cannot add fact_value inference provenance: existing AI fact_values require explicit inference_run_id backfill.';
            END IF;
        END
        $$;
        """
    )

    op.drop_constraint(op.f(_OLD_AI_CHECK), "fact_values", type_="check")
    op.create_check_constraint(
        op.f(_NEW_AI_CHECK),
        "fact_values",
        "(source_kind <> 'ai' OR (extraction_run_id IS NOT NULL AND inference_run_id IS NOT NULL))",
    )
    op.create_check_constraint(
        op.f(_NON_AI_CHECK),
        "fact_values",
        "(source_kind = 'ai' OR inference_run_id IS NULL)",
    )


def downgrade() -> None:
    op.drop_constraint(op.f(_NON_AI_CHECK), "fact_values", type_="check")
    op.drop_constraint(op.f(_NEW_AI_CHECK), "fact_values", type_="check")
    op.create_check_constraint(
        op.f(_OLD_AI_CHECK),
        "fact_values",
        "(source_kind <> 'ai' OR extraction_run_id IS NOT NULL)",
    )
    op.drop_constraint(
        op.f("fk_fact_values_inference_run_id_inference_runs"),
        "fact_values",
        type_="foreignkey",
    )
    op.drop_index(op.f("ix_fact_values_inference_run_id"), table_name="fact_values")
    op.drop_column("fact_values", "inference_run_id")
