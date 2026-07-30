"""dynamic schema, version, and field models"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "202607301800"
down_revision: str | None = "202607301630"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "dynamic_schemas",
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("schema_key", sa.String(length=120), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("subject_kind", sa.String(length=64), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'active'"),
        ),
        sa.Column("current_version_id", sa.Uuid(), nullable=True),
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
            "char_length(schema_key) BETWEEN 1 AND 120",
            name=op.f("ck_dynamic_schemas_dynamic_schemas_schema_key_length"),
        ),
        sa.CheckConstraint(
            "schema_key ~ '^[a-z0-9._-]+$'",
            name=op.f("ck_dynamic_schemas_dynamic_schemas_schema_key_format"),
        ),
        sa.CheckConstraint(
            "char_length(name) BETWEEN 1 AND 200",
            name=op.f("ck_dynamic_schemas_dynamic_schemas_name_length"),
        ),
        sa.CheckConstraint(
            "char_length(subject_kind) BETWEEN 1 AND 64",
            name=op.f("ck_dynamic_schemas_dynamic_schemas_subject_kind_length"),
        ),
        sa.CheckConstraint(
            "status IN ('active', 'archived')",
            name=op.f("ck_dynamic_schemas_dynamic_schemas_status_valid"),
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_dynamic_schemas_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["created_by_id"],
            ["users.id"],
            name=op.f("fk_dynamic_schemas_created_by_id_users"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_dynamic_schemas")),
        sa.UniqueConstraint(
            "project_id",
            "schema_key",
            name="uq_dynamic_schemas_project_id_schema_key",
        ),
        sa.UniqueConstraint(
            "current_version_id",
            name="uq_dynamic_schemas_current_version_id",
        ),
    )
    op.create_index(
        op.f("ix_dynamic_schemas_project_id"),
        "dynamic_schemas",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_dynamic_schemas_current_version_id"),
        "dynamic_schemas",
        ["current_version_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_dynamic_schemas_created_by_id"),
        "dynamic_schemas",
        ["created_by_id"],
        unique=False,
    )

    op.create_table(
        "dynamic_schema_versions",
        sa.Column("schema_id", sa.Uuid(), nullable=False),
        sa.Column("version_no", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'draft'"),
        ),
        sa.Column("source_kind", sa.String(length=16), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column(
            "layout_config",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("created_by_id", sa.Uuid(), nullable=True),
        sa.Column("activated_by_id", sa.Uuid(), nullable=True),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint(
            "version_no > 0",
            name=op.f("ck_dynamic_schema_versions_dynamic_schema_versions_version_no_positive"),
        ),
        sa.CheckConstraint(
            "status IN ('draft', 'proposed', 'active', 'retired')",
            name=op.f("ck_dynamic_schema_versions_dynamic_schema_versions_status_valid"),
        ),
        sa.CheckConstraint(
            "source_kind IN ('ai', 'human', 'import', 'system')",
            name=op.f("ck_dynamic_schema_versions_dynamic_schema_versions_source_kind_valid"),
        ),
        sa.CheckConstraint(
            "((activated_by_id IS NULL AND activated_at IS NULL) OR (activated_by_id IS NOT NULL AND activated_at IS NOT NULL))",
            name=op.f("ck_dynamic_schema_versions_dynamic_schema_versions_activation_actor_time_pair"),
        ),
        sa.CheckConstraint(
            "(status NOT IN ('draft', 'proposed') OR (activated_by_id IS NULL AND activated_at IS NULL))",
            name=op.f("ck_dynamic_schema_versions_dynamic_schema_versions_inactive_status_has_no_activation"),
        ),
        sa.CheckConstraint(
            "(status NOT IN ('active', 'retired') OR (activated_by_id IS NOT NULL AND activated_at IS NOT NULL))",
            name=op.f("ck_dynamic_schema_versions_dynamic_schema_versions_active_status_requires_activation"),
        ),
        sa.CheckConstraint(
            "(source_kind <> 'human' OR created_by_id IS NOT NULL)",
            name=op.f("ck_dynamic_schema_versions_dynamic_schema_versions_human_requires_created_by"),
        ),
        sa.ForeignKeyConstraint(
            ["schema_id"],
            ["dynamic_schemas.id"],
            name=op.f("fk_dynamic_schema_versions_schema_id_dynamic_schemas"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["created_by_id"],
            ["users.id"],
            name=op.f("fk_dynamic_schema_versions_created_by_id_users"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["activated_by_id"],
            ["users.id"],
            name=op.f("fk_dynamic_schema_versions_activated_by_id_users"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_dynamic_schema_versions")),
        sa.UniqueConstraint(
            "schema_id",
            "version_no",
            name="uq_dynamic_schema_versions_schema_id_version_no",
        ),
    )
    op.create_index(
        op.f("ix_dynamic_schema_versions_schema_id"),
        "dynamic_schema_versions",
        ["schema_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_dynamic_schema_versions_created_by_id"),
        "dynamic_schema_versions",
        ["created_by_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_dynamic_schema_versions_activated_by_id"),
        "dynamic_schema_versions",
        ["activated_by_id"],
        unique=False,
    )

    op.create_table(
        "dynamic_schema_fields",
        sa.Column("schema_version_id", sa.Uuid(), nullable=False),
        sa.Column("field_key", sa.String(length=120), nullable=False),
        sa.Column("label", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("predicate_key", sa.String(length=255), nullable=False),
        sa.Column("scope_key", sa.String(length=255), nullable=True),
        sa.Column(
            "expected_value_type",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'any'"),
        ),
        sa.Column(
            "cardinality",
            sa.String(length=8),
            nullable=False,
            server_default=sa.text("'one'"),
        ),
        sa.Column(
            "is_required",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "is_title",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "is_summary",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "is_hidden",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("group_key", sa.String(length=120), nullable=True),
        sa.Column(
            "display_order",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "display_config",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "validation_rules",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint(
            "char_length(field_key) BETWEEN 1 AND 120",
            name=op.f("ck_dynamic_schema_fields_dynamic_schema_fields_field_key_length"),
        ),
        sa.CheckConstraint(
            "field_key ~ '^[a-z0-9._-]+$'",
            name=op.f("ck_dynamic_schema_fields_dynamic_schema_fields_field_key_format"),
        ),
        sa.CheckConstraint(
            "char_length(label) BETWEEN 1 AND 200",
            name=op.f("ck_dynamic_schema_fields_dynamic_schema_fields_label_length"),
        ),
        sa.CheckConstraint(
            "char_length(predicate_key) BETWEEN 1 AND 255",
            name=op.f("ck_dynamic_schema_fields_dynamic_schema_fields_predicate_key_length"),
        ),
        sa.CheckConstraint(
            "scope_key IS NULL OR char_length(scope_key) BETWEEN 1 AND 255",
            name=op.f("ck_dynamic_schema_fields_dynamic_schema_fields_scope_key_length"),
        ),
        sa.CheckConstraint(
            "expected_value_type IN ('any', 'string', 'number', 'boolean', 'date', 'datetime', 'entity_ref', 'list', 'object', 'null')",
            name=op.f("ck_dynamic_schema_fields_dynamic_schema_fields_expected_value_type_valid"),
        ),
        sa.CheckConstraint(
            "cardinality IN ('one', 'many')",
            name=op.f("ck_dynamic_schema_fields_dynamic_schema_fields_cardinality_valid"),
        ),
        sa.CheckConstraint(
            "group_key IS NULL OR char_length(group_key) BETWEEN 1 AND 120",
            name=op.f("ck_dynamic_schema_fields_dynamic_schema_fields_group_key_length"),
        ),
        sa.CheckConstraint(
            "display_order >= 0",
            name=op.f("ck_dynamic_schema_fields_dynamic_schema_fields_display_order_non_negative"),
        ),
        sa.ForeignKeyConstraint(
            ["schema_version_id"],
            ["dynamic_schema_versions.id"],
            name=op.f("fk_dynamic_schema_fields_schema_version_id_dynamic_schema_versions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_dynamic_schema_fields")),
        sa.UniqueConstraint(
            "schema_version_id",
            "field_key",
            name="uq_dynamic_schema_fields_schema_version_id_field_key",
        ),
    )
    op.create_index(
        op.f("ix_dynamic_schema_fields_schema_version_id"),
        "dynamic_schema_fields",
        ["schema_version_id"],
        unique=False,
    )

    op.create_foreign_key(
        op.f("fk_dynamic_schemas_current_version_id_dynamic_schema_versions"),
        "dynamic_schemas",
        "dynamic_schema_versions",
        ["current_version_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("fk_dynamic_schemas_current_version_id_dynamic_schema_versions"),
        "dynamic_schemas",
        type_="foreignkey",
    )
    op.drop_index(op.f("ix_dynamic_schema_fields_schema_version_id"), table_name="dynamic_schema_fields")
    op.drop_table("dynamic_schema_fields")
    op.drop_index(op.f("ix_dynamic_schema_versions_activated_by_id"), table_name="dynamic_schema_versions")
    op.drop_index(op.f("ix_dynamic_schema_versions_created_by_id"), table_name="dynamic_schema_versions")
    op.drop_index(op.f("ix_dynamic_schema_versions_schema_id"), table_name="dynamic_schema_versions")
    op.drop_table("dynamic_schema_versions")
    op.drop_index(op.f("ix_dynamic_schemas_created_by_id"), table_name="dynamic_schemas")
    op.drop_index(op.f("ix_dynamic_schemas_current_version_id"), table_name="dynamic_schemas")
    op.drop_index(op.f("ix_dynamic_schemas_project_id"), table_name="dynamic_schemas")
    op.drop_table("dynamic_schemas")
