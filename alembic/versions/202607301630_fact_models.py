"""fact, fact value, and fact evidence link models"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "202607301630"
down_revision: str | None = "202607301430"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "facts",
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("subject_kind", sa.String(length=64), nullable=False),
        sa.Column("subject_key", sa.String(length=255), nullable=False),
        sa.Column("predicate_key", sa.String(length=255), nullable=False),
        sa.Column("scope_key", sa.String(length=255), nullable=True),
        sa.Column("identity_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'active'"),
        ),
        sa.Column("current_value_id", sa.Uuid(), nullable=True),
        sa.Column("created_by_id", sa.Uuid(), nullable=True),
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
            "char_length(subject_kind) BETWEEN 1 AND 64",
            name=op.f("ck_facts_facts_subject_kind_length"),
        ),
        sa.CheckConstraint(
            "char_length(subject_key) BETWEEN 1 AND 255",
            name=op.f("ck_facts_facts_subject_key_length"),
        ),
        sa.CheckConstraint(
            "char_length(predicate_key) BETWEEN 1 AND 255",
            name=op.f("ck_facts_facts_predicate_key_length"),
        ),
        sa.CheckConstraint(
            "scope_key IS NULL OR char_length(scope_key) BETWEEN 1 AND 255",
            name=op.f("ck_facts_facts_scope_key_length"),
        ),
        sa.CheckConstraint(
            "identity_hash ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_facts_facts_identity_hash_format"),
        ),
        sa.CheckConstraint(
            "status IN ('active', 'retired')",
            name=op.f("ck_facts_facts_status_valid"),
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_facts_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["created_by_id"],
            ["users.id"],
            name=op.f("fk_facts_created_by_id_users"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_facts")),
        sa.UniqueConstraint(
            "project_id",
            "identity_hash",
            name="uq_facts_project_id_identity_hash",
        ),
        sa.UniqueConstraint(
            "current_value_id",
            name="uq_facts_current_value_id",
        ),
    )
    op.create_index(
        op.f("ix_facts_project_id"),
        "facts",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_facts_current_value_id"),
        "facts",
        ["current_value_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_facts_created_by_id"),
        "facts",
        ["created_by_id"],
        unique=False,
    )

    op.create_table(
        "fact_values",
        sa.Column("fact_id", sa.Uuid(), nullable=False),
        sa.Column("version_no", sa.Integer(), nullable=False),
        sa.Column("value_type", sa.String(length=16), nullable=False),
        sa.Column("value_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("normalized_value_text", sa.String(length=2000), nullable=True),
        sa.Column("value_hash", sa.String(length=64), nullable=False),
        sa.Column("language_code", sa.String(length=32), nullable=True),
        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'proposed'"),
        ),
        sa.Column("source_kind", sa.String(length=16), nullable=False),
        sa.Column("extraction_run_id", sa.Uuid(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("created_by_id", sa.Uuid(), nullable=True),
        sa.Column("decided_by_id", sa.Uuid(), nullable=True),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint(
            "version_no > 0",
            name=op.f("ck_fact_values_fact_values_version_no_positive"),
        ),
        sa.CheckConstraint(
            "value_type IN ('string', 'number', 'boolean', 'date', 'datetime', 'entity_ref', 'list', 'object', 'null')",
            name=op.f("ck_fact_values_fact_values_value_type_valid"),
        ),
        sa.CheckConstraint(
            "status IN ('proposed', 'accepted', 'rejected', 'superseded')",
            name=op.f("ck_fact_values_fact_values_status_valid"),
        ),
        sa.CheckConstraint(
            "source_kind IN ('ai', 'human', 'import', 'system')",
            name=op.f("ck_fact_values_fact_values_source_kind_valid"),
        ),
        sa.CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name=op.f("ck_fact_values_fact_values_confidence_range"),
        ),
        sa.CheckConstraint(
            "value_hash ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_fact_values_fact_values_value_hash_format"),
        ),
        sa.CheckConstraint(
            "((value_type = 'null' AND value_json IS NULL) OR (value_type <> 'null' AND value_json IS NOT NULL))",
            name=op.f("ck_fact_values_fact_values_value_json_null_rule"),
        ),
        sa.CheckConstraint(
            "((decided_by_id IS NULL AND decided_at IS NULL) OR (decided_by_id IS NOT NULL AND decided_at IS NOT NULL))",
            name=op.f("ck_fact_values_fact_values_decision_actor_time_pair"),
        ),
        sa.CheckConstraint(
            "(status <> 'proposed' OR (decided_by_id IS NULL AND decided_at IS NULL))",
            name=op.f("ck_fact_values_fact_values_proposed_has_no_decision"),
        ),
        sa.CheckConstraint(
            "(status NOT IN ('accepted', 'rejected') OR (decided_by_id IS NOT NULL AND decided_at IS NOT NULL))",
            name=op.f("ck_fact_values_fact_values_decided_status_requires_actor_and_time"),
        ),
        sa.CheckConstraint(
            "(source_kind <> 'human' OR created_by_id IS NOT NULL)",
            name=op.f("ck_fact_values_fact_values_human_requires_created_by"),
        ),
        sa.CheckConstraint(
            "(source_kind <> 'ai' OR extraction_run_id IS NOT NULL)",
            name=op.f("ck_fact_values_fact_values_ai_requires_extraction_run"),
        ),
        sa.ForeignKeyConstraint(
            ["fact_id"],
            ["facts.id"],
            name=op.f("fk_fact_values_fact_id_facts"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["extraction_run_id"],
            ["extraction_runs.id"],
            name=op.f("fk_fact_values_extraction_run_id_extraction_runs"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["created_by_id"],
            ["users.id"],
            name=op.f("fk_fact_values_created_by_id_users"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["decided_by_id"],
            ["users.id"],
            name=op.f("fk_fact_values_decided_by_id_users"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_fact_values")),
        sa.UniqueConstraint(
            "fact_id",
            "version_no",
            name="uq_fact_values_fact_id_version_no",
        ),
    )
    op.create_index(
        op.f("ix_fact_values_fact_id"),
        "fact_values",
        ["fact_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_fact_values_extraction_run_id"),
        "fact_values",
        ["extraction_run_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_fact_values_created_by_id"),
        "fact_values",
        ["created_by_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_fact_values_decided_by_id"),
        "fact_values",
        ["decided_by_id"],
        unique=False,
    )

    op.create_table(
        "fact_evidence_links",
        sa.Column("fact_value_id", sa.Uuid(), nullable=False),
        sa.Column("evidence_id", sa.Uuid(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column(
            "is_primary",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "source_order",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint(
            "role IN ('supporting', 'contradicting', 'context')",
            name=op.f("ck_fact_evidence_links_fact_evidence_links_role_valid"),
        ),
        sa.CheckConstraint(
            "source_order >= 0",
            name=op.f("ck_fact_evidence_links_fact_evidence_links_source_order_non_negative"),
        ),
        sa.ForeignKeyConstraint(
            ["fact_value_id"],
            ["fact_values.id"],
            name=op.f("fk_fact_evidence_links_fact_value_id_fact_values"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["evidence_id"],
            ["source_evidences.id"],
            name=op.f("fk_fact_evidence_links_evidence_id_source_evidences"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_fact_evidence_links")),
        sa.UniqueConstraint(
            "fact_value_id",
            "evidence_id",
            "role",
            name="uq_fact_evidence_links_fact_value_id_evidence_id_role",
        ),
    )
    op.create_index(
        op.f("ix_fact_evidence_links_fact_value_id"),
        "fact_evidence_links",
        ["fact_value_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_fact_evidence_links_evidence_id"),
        "fact_evidence_links",
        ["evidence_id"],
        unique=False,
    )

    op.create_foreign_key(
        op.f("fk_facts_current_value_id_fact_values"),
        "facts",
        "fact_values",
        ["current_value_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(op.f("fk_facts_current_value_id_fact_values"), "facts", type_="foreignkey")
    op.drop_index(op.f("ix_fact_evidence_links_evidence_id"), table_name="fact_evidence_links")
    op.drop_index(op.f("ix_fact_evidence_links_fact_value_id"), table_name="fact_evidence_links")
    op.drop_table("fact_evidence_links")
    op.drop_index(op.f("ix_fact_values_decided_by_id"), table_name="fact_values")
    op.drop_index(op.f("ix_fact_values_created_by_id"), table_name="fact_values")
    op.drop_index(op.f("ix_fact_values_extraction_run_id"), table_name="fact_values")
    op.drop_index(op.f("ix_fact_values_fact_id"), table_name="fact_values")
    op.drop_table("fact_values")
    op.drop_index(op.f("ix_facts_created_by_id"), table_name="facts")
    op.drop_index(op.f("ix_facts_current_value_id"), table_name="facts")
    op.drop_index(op.f("ix_facts_project_id"), table_name="facts")
    op.drop_table("facts")
