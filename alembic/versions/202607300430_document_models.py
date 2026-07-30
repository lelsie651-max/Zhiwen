"""document and document revision models"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "202607300430"
down_revision: str | None = "202607300201"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'active'"),
        ),
        sa.Column("created_by_id", sa.Uuid(), nullable=False),
        sa.Column("current_revision_id", sa.Uuid(), nullable=True),
        sa.Column("logical_order_kind", sa.String(length=32), nullable=True),
        sa.Column("logical_order_value", sa.Text(), nullable=True),
        sa.Column("logical_order_index", sa.Numeric(18, 6), nullable=True),
        sa.Column("effective_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("effective_to", sa.DateTime(timezone=True), nullable=True),
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
            "char_length(title) BETWEEN 1 AND 200",
            name=op.f("ck_documents_documents_title_length"),
        ),
        sa.CheckConstraint(
            "status IN ('active', 'archived')",
            name=op.f("ck_documents_documents_status_valid"),
        ),
        sa.CheckConstraint(
            "effective_from IS NULL OR effective_to IS NULL OR effective_to >= effective_from",
            name=op.f("ck_documents_documents_effective_range_valid"),
        ),
        sa.ForeignKeyConstraint(
            ["created_by_id"],
            ["users.id"],
            name=op.f("fk_documents_created_by_id_users"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_documents_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_documents")),
        sa.UniqueConstraint(
            "current_revision_id",
            name="uq_documents_current_revision_id",
        ),
    )
    op.create_index(op.f("ix_documents_created_by_id"), "documents", ["created_by_id"], unique=False)
    op.create_index(
        op.f("ix_documents_current_revision_id"),
        "documents",
        ["current_revision_id"],
        unique=False,
    )
    op.create_index(op.f("ix_documents_project_id"), "documents", ["project_id"], unique=False)

    op.create_table(
        "document_revisions",
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("revision_no", sa.Integer(), nullable=False),
        sa.Column("upload_intent", sa.String(length=16), nullable=False),
        sa.Column("supersedes_revision_id", sa.Uuid(), nullable=True),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("storage_key", sa.String(length=255), nullable=False),
        sa.Column("mime_type", sa.String(length=255), nullable=False),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("detected_language", sa.String(length=32), nullable=True),
        sa.Column("language_confidence", sa.Float(), nullable=True),
        sa.Column(
            "source_authority",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column(
            "status",
            sa.String(length=24),
            nullable=False,
            server_default=sa.text("'uploaded'"),
        ),
        sa.Column("uploaded_by_id", sa.Uuid(), nullable=False),
        sa.Column(
            "uploaded_at",
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
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.CheckConstraint(
            "revision_no > 0",
            name=op.f("ck_document_revisions_document_revisions_revision_no_positive"),
        ),
        sa.CheckConstraint(
            "((revision_no = 1 AND upload_intent = 'new_document') OR (revision_no > 1 AND upload_intent = 'revision'))",
            name=op.f("ck_document_revisions_document_revisions_revision_intent_valid"),
        ),
        sa.CheckConstraint(
            "((revision_no = 1 AND supersedes_revision_id IS NULL) OR (revision_no > 1 AND supersedes_revision_id IS NOT NULL))",
            name=op.f("ck_document_revisions_document_revisions_revision_supersedes_required"),
        ),
        sa.CheckConstraint(
            "char_length(original_filename) BETWEEN 1 AND 255",
            name=op.f("ck_document_revisions_document_revisions_original_filename_length"),
        ),
        sa.CheckConstraint(
            "char_length(storage_key) BETWEEN 1 AND 255",
            name=op.f("ck_document_revisions_document_revisions_storage_key_length"),
        ),
        sa.CheckConstraint(
            "storage_key <> original_filename",
            name=op.f("ck_document_revisions_document_revisions_storage_key_not_filename"),
        ),
        sa.CheckConstraint(
            "sha256 ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_document_revisions_document_revisions_sha256_format"),
        ),
        sa.CheckConstraint(
            "file_size_bytes > 0",
            name=op.f("ck_document_revisions_document_revisions_file_size_positive"),
        ),
        sa.CheckConstraint(
            "language_confidence IS NULL OR (language_confidence >= 0 AND language_confidence <= 1)",
            name=op.f("ck_document_revisions_document_revisions_language_confidence_range"),
        ),
        sa.CheckConstraint(
            "upload_intent IN ('new_document', 'revision')",
            name=op.f("ck_document_revisions_document_revisions_upload_intent_valid"),
        ),
        sa.CheckConstraint(
            "source_authority IN ('pending', 'formal', 'reference')",
            name=op.f("ck_document_revisions_document_revisions_source_authority_valid"),
        ),
        sa.CheckConstraint(
            "status IN ('uploaded', 'preflight_checking', 'rejected', 'needs_confirmation', 'quarantined', 'accepted', 'parsing', 'extracting', 'awaiting_review', 'completed', 'failed')",
            name=op.f("ck_document_revisions_document_revisions_status_valid"),
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_document_revisions_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["supersedes_revision_id"],
            ["document_revisions.id"],
            name=op.f("fk_document_revisions_supersedes_revision_id_document_revisions"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["uploaded_by_id"],
            ["users.id"],
            name=op.f("fk_document_revisions_uploaded_by_id_users"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_document_revisions")),
        sa.UniqueConstraint(
            "document_id",
            "revision_no",
            name="uq_document_revisions_document_id_revision_no",
        ),
        sa.UniqueConstraint("storage_key", name="uq_document_revisions_storage_key"),
        sa.UniqueConstraint(
            "supersedes_revision_id",
            name="uq_document_revisions_supersedes_revision_id",
        ),
    )
    op.create_index(
        op.f("ix_document_revisions_document_id"),
        "document_revisions",
        ["document_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_document_revisions_uploaded_by_id"),
        "document_revisions",
        ["uploaded_by_id"],
        unique=False,
    )
    op.create_foreign_key(
        op.f("fk_documents_current_revision_id_document_revisions"),
        "documents",
        "document_revisions",
        ["current_revision_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("fk_documents_current_revision_id_document_revisions"),
        "documents",
        type_="foreignkey",
    )
    op.drop_index(op.f("ix_document_revisions_uploaded_by_id"), table_name="document_revisions")
    op.drop_index(op.f("ix_document_revisions_document_id"), table_name="document_revisions")
    op.drop_table("document_revisions")
    op.drop_index(op.f("ix_documents_project_id"), table_name="documents")
    op.drop_index(op.f("ix_documents_current_revision_id"), table_name="documents")
    op.drop_index(op.f("ix_documents_created_by_id"), table_name="documents")
    op.drop_table("documents")
