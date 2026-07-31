"""add fact to entity link columns"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "202607311430"
down_revision: str | None = "202607311230"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


_ENTITY_REF_PAIR_CHECK = "ck_fv_entity_ref_pair"


def upgrade() -> None:
    op.add_column(
        "facts",
        sa.Column("subject_entity_id", sa.Uuid(), nullable=True),
    )
    op.create_index(
        op.f("ix_facts_subject_entity_id"),
        "facts",
        ["subject_entity_id"],
        unique=False,
    )
    op.create_foreign_key(
        op.f("fk_facts_subject_entity_id_entities"),
        "facts",
        "entities",
        ["subject_entity_id"],
        ["id"],
        ondelete="RESTRICT",
    )

    op.add_column(
        "fact_values",
        sa.Column("referenced_entity_id", sa.Uuid(), nullable=True),
    )
    op.create_index(
        op.f("ix_fact_values_referenced_entity_id"),
        "fact_values",
        ["referenced_entity_id"],
        unique=False,
    )
    op.create_foreign_key(
        op.f("fk_fact_values_referenced_entity_id_entities"),
        "fact_values",
        "entities",
        ["referenced_entity_id"],
        ["id"],
        ondelete="RESTRICT",
    )

    bind = op.get_bind()
    existing_entity_ref_rows = bind.execute(
        sa.text("SELECT count(*) FROM fact_values WHERE value_type = 'entity_ref'")
    ).scalar_one()
    if existing_entity_ref_rows:
        raise RuntimeError(
            "Cannot add fact entity links: existing entity_ref fact_values "
            "require explicit referenced_entity_id backfill."
        )

    op.create_check_constraint(
        op.f(_ENTITY_REF_PAIR_CHECK),
        "fact_values",
        "((value_type = 'entity_ref' AND referenced_entity_id IS NOT NULL) OR "
        "(value_type <> 'entity_ref' AND referenced_entity_id IS NULL))",
    )


def downgrade() -> None:
    op.drop_constraint(op.f(_ENTITY_REF_PAIR_CHECK), "fact_values", type_="check")
    op.drop_constraint(
        op.f("fk_fact_values_referenced_entity_id_entities"),
        "fact_values",
        type_="foreignkey",
    )
    op.drop_constraint(
        op.f("fk_facts_subject_entity_id_entities"),
        "facts",
        type_="foreignkey",
    )
    op.drop_index(op.f("ix_fact_values_referenced_entity_id"), table_name="fact_values")
    op.drop_index(op.f("ix_facts_subject_entity_id"), table_name="facts")
    op.drop_column("fact_values", "referenced_entity_id")
    op.drop_column("facts", "subject_entity_id")
