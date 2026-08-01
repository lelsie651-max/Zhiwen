"""harden processing job extraction result links"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "202607311600"
down_revision: str | None = "202607311430"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM processing_jobs
                WHERE result_extraction_run_id IS NOT NULL
                GROUP BY result_extraction_run_id
                HAVING count(*) > 1
            ) THEN
                RAISE EXCEPTION
                    'Cannot add unique processing job result links: multiple processing jobs reference the same extraction run.';
            END IF;
        END
        $$;
        """
    )

    op.create_unique_constraint(
        "uq_pj_result_run_id",
        "processing_jobs",
        ["result_extraction_run_id"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_pj_result_run_id", "processing_jobs", type_="unique")
