"""add entity and alias models"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "202607311230"
down_revision: str | None = "202607311030"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "entities",
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("entity_type", sa.String(length=64), nullable=False),
        sa.Column("canonical_key", sa.String(length=255), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column("identity_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="active"),
        sa.Column("merged_into_entity_id", sa.Uuid(), nullable=True),
        sa.Column("created_by_id", sa.Uuid(), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.CheckConstraint("char_length(entity_type) BETWEEN 1 AND 64", name=op.f("ck_entities_ent_type_len")),
        sa.CheckConstraint("char_length(canonical_key) BETWEEN 1 AND 255", name=op.f("ck_entities_ent_key_len")),
        sa.CheckConstraint("char_length(display_name) BETWEEN 1 AND 255", name=op.f("ck_entities_ent_name_len")),
        sa.CheckConstraint("identity_hash ~ '^[0-9a-f]{64}$'", name=op.f("ck_entities_ent_hash_fmt")),
        sa.CheckConstraint("status IN ('active', 'merged', 'archived')", name=op.f("ck_entities_ent_status_ok")),
        sa.CheckConstraint(
            "((status = 'merged' AND merged_into_entity_id IS NOT NULL) OR "
            "(status <> 'merged' AND merged_into_entity_id IS NULL))",
            name=op.f("ck_entities_ent_merge_pair"),
        ),
        sa.CheckConstraint(
            "(merged_into_entity_id IS NULL OR merged_into_entity_id <> id)",
            name=op.f("ck_entities_ent_not_self_merge"),
        ),
        sa.ForeignKeyConstraint(["created_by_id"], ["users.id"], name=op.f("fk_entities_created_by_id_users"), ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["merged_into_entity_id"], ["entities.id"], name=op.f("fk_entities_merged_into_entity_id_entities"), ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], name=op.f("fk_entities_project_id_projects"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_entities")),
        sa.UniqueConstraint("project_id", "entity_type", "canonical_key", name="uq_ent_proj_type_key"),
        sa.UniqueConstraint("project_id", "identity_hash", name="uq_ent_proj_hash"),
    )
    op.create_index(op.f("ix_entities_created_by_id"), "entities", ["created_by_id"], unique=False)
    op.create_index(op.f("ix_entities_identity_hash"), "entities", ["identity_hash"], unique=False)
    op.create_index(op.f("ix_entities_merged_into_entity_id"), "entities", ["merged_into_entity_id"], unique=False)
    op.create_index(op.f("ix_entities_project_id"), "entities", ["project_id"], unique=False)
    op.create_index("ix_entities_project_type", "entities", ["project_id", "entity_type"], unique=False)

    op.create_table(
        "entity_aliases",
        sa.Column("entity_id", sa.Uuid(), nullable=False),
        sa.Column("alias_text", sa.String(length=255), nullable=False),
        sa.Column("normalized_alias", sa.String(length=255), nullable=False),
        sa.Column("language_code", sa.String(length=32), nullable=False, server_default="und"),
        sa.Column("alias_kind", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="active"),
        sa.Column("is_primary", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_by_id", sa.Uuid(), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.CheckConstraint("char_length(alias_text) BETWEEN 1 AND 255", name=op.f("ck_entity_aliases_ea_text_len")),
        sa.CheckConstraint("char_length(normalized_alias) BETWEEN 1 AND 255", name=op.f("ck_entity_aliases_ea_norm_len")),
        sa.CheckConstraint("char_length(language_code) BETWEEN 1 AND 32", name=op.f("ck_entity_aliases_ea_lang_len")),
        sa.CheckConstraint(
            "alias_kind IN ('canonical', 'alternate', 'abbreviation', 'transliteration')",
            name=op.f("ck_entity_aliases_ea_kind_ok"),
        ),
        sa.CheckConstraint("status IN ('active', 'retired')", name=op.f("ck_entity_aliases_ea_status_ok")),
        sa.CheckConstraint(
            "((NOT is_primary) OR (alias_kind = 'canonical' AND status = 'active'))",
            name=op.f("ck_entity_aliases_ea_primary_rule"),
        ),
        sa.CheckConstraint(
            "(status <> 'retired' OR NOT is_primary)",
            name=op.f("ck_entity_aliases_ea_retired_no_primary"),
        ),
        sa.ForeignKeyConstraint(["created_by_id"], ["users.id"], name=op.f("fk_entity_aliases_created_by_id_users"), ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["entity_id"], ["entities.id"], name=op.f("fk_entity_aliases_entity_id_entities"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_entity_aliases")),
        sa.UniqueConstraint("entity_id", "normalized_alias", "language_code", name="uq_ea_ent_norm_lang"),
    )
    op.create_index(op.f("ix_entity_aliases_created_by_id"), "entity_aliases", ["created_by_id"], unique=False)
    op.create_index(op.f("ix_entity_aliases_entity_id"), "entity_aliases", ["entity_id"], unique=False)
    op.create_index(
        "uq_ea_active_primary",
        "entity_aliases",
        ["entity_id"],
        unique=True,
        postgresql_where=sa.text("status = 'active' AND is_primary = true"),
    )


def downgrade() -> None:
    op.drop_index("uq_ea_active_primary", table_name="entity_aliases", postgresql_where=sa.text("status = 'active' AND is_primary = true"))
    op.drop_index(op.f("ix_entity_aliases_entity_id"), table_name="entity_aliases")
    op.drop_index(op.f("ix_entity_aliases_created_by_id"), table_name="entity_aliases")
    op.drop_table("entity_aliases")
    op.drop_index("ix_entities_project_type", table_name="entities")
    op.drop_index(op.f("ix_entities_project_id"), table_name="entities")
    op.drop_index(op.f("ix_entities_merged_into_entity_id"), table_name="entities")
    op.drop_index(op.f("ix_entities_identity_hash"), table_name="entities")
    op.drop_index(op.f("ix_entities_created_by_id"), table_name="entities")
    op.drop_table("entities")
