"""document extraction run, block, and evidence models"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "202607301430"
down_revision: str | None = "202607300620"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "extraction_runs",
        sa.Column("revision_id", sa.Uuid(), nullable=False),
        sa.Column("attempt_no", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("outcome", sa.String(length=16), nullable=False),
        sa.Column("extractor_name", sa.String(length=100), nullable=False),
        sa.Column("extractor_version", sa.String(length=32), nullable=False),
        sa.Column("detected_format", sa.String(length=16), nullable=False),
        sa.Column("detected_encoding", sa.String(length=64), nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("character_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("block_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "warnings",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("failure_code", sa.String(length=64), nullable=True),
        sa.Column("failure_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint(
            "attempt_no > 0",
            name=op.f("ck_extraction_runs_extraction_runs_attempt_no_positive"),
        ),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'failed')",
            name=op.f("ck_extraction_runs_extraction_runs_status_valid"),
        ),
        sa.CheckConstraint(
            "outcome IN ('success', 'partial', 'needs_ocr', 'failed')",
            name=op.f("ck_extraction_runs_extraction_runs_outcome_valid"),
        ),
        sa.CheckConstraint(
            "character_count >= 0",
            name=op.f("ck_extraction_runs_extraction_runs_character_count_non_negative"),
        ),
        sa.CheckConstraint(
            "block_count >= 0",
            name=op.f("ck_extraction_runs_extraction_runs_block_count_non_negative"),
        ),
        sa.CheckConstraint(
            "completed_at IS NULL OR started_at IS NULL OR completed_at >= started_at",
            name=op.f("ck_extraction_runs_extraction_runs_completed_after_started"),
        ),
        sa.ForeignKeyConstraint(
            ["revision_id"],
            ["document_revisions.id"],
            name=op.f("fk_extraction_runs_revision_id_document_revisions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_extraction_runs")),
        sa.UniqueConstraint(
            "revision_id",
            "attempt_no",
            name="uq_extraction_runs_revision_id_attempt_no",
        ),
    )
    op.create_index(
        op.f("ix_extraction_runs_revision_id"),
        "extraction_runs",
        ["revision_id"],
        unique=False,
    )

    op.create_table(
        "document_blocks",
        sa.Column("extraction_run_id", sa.Uuid(), nullable=False),
        sa.Column("source_order", sa.Integer(), nullable=False),
        sa.Column("block_type", sa.String(length=16), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("normalized_text", sa.Text(), nullable=False),
        sa.Column("location_key", sa.String(length=255), nullable=False),
        sa.Column("anchor_hash", sa.String(length=64), nullable=False),
        sa.Column("page_no", sa.Integer(), nullable=True),
        sa.Column("block_index", sa.Integer(), nullable=False),
        sa.Column("heading_level", sa.Integer(), nullable=True),
        sa.Column(
            "heading_path",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("start_line", sa.Integer(), nullable=True),
        sa.Column("end_line", sa.Integer(), nullable=True),
        sa.Column("table_index", sa.Integer(), nullable=True),
        sa.Column("row_index", sa.Integer(), nullable=True),
        sa.Column(
            "metadata",
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
            "block_type IN ('heading', 'paragraph', 'list_item', 'table_row', 'code', 'page_text')",
            name=op.f("ck_document_blocks_document_blocks_block_type_valid"),
        ),
        sa.CheckConstraint(
            "source_order >= 0",
            name=op.f("ck_document_blocks_document_blocks_source_order_non_negative"),
        ),
        sa.CheckConstraint(
            "block_index >= 0",
            name=op.f("ck_document_blocks_document_blocks_block_index_non_negative"),
        ),
        sa.CheckConstraint(
            "char_length(raw_text) > 0",
            name=op.f("ck_document_blocks_document_blocks_raw_text_non_empty"),
        ),
        sa.CheckConstraint(
            "char_length(normalized_text) > 0",
            name=op.f("ck_document_blocks_document_blocks_normalized_text_non_empty"),
        ),
        sa.CheckConstraint(
            "char_length(location_key) > 0",
            name=op.f("ck_document_blocks_document_blocks_location_key_non_empty"),
        ),
        sa.CheckConstraint(
            "anchor_hash ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_document_blocks_document_blocks_anchor_hash_format"),
        ),
        sa.CheckConstraint(
            "page_no IS NULL OR page_no > 0",
            name=op.f("ck_document_blocks_document_blocks_page_no_positive"),
        ),
        sa.CheckConstraint(
            "heading_level IS NULL OR heading_level > 0",
            name=op.f("ck_document_blocks_document_blocks_heading_level_positive"),
        ),
        sa.CheckConstraint(
            "start_line IS NULL OR start_line > 0",
            name=op.f("ck_document_blocks_document_blocks_start_line_positive"),
        ),
        sa.CheckConstraint(
            "end_line IS NULL OR end_line > 0",
            name=op.f("ck_document_blocks_document_blocks_end_line_positive"),
        ),
        sa.CheckConstraint(
            "table_index IS NULL OR table_index > 0",
            name=op.f("ck_document_blocks_document_blocks_table_index_positive"),
        ),
        sa.CheckConstraint(
            "row_index IS NULL OR row_index > 0",
            name=op.f("ck_document_blocks_document_blocks_row_index_positive"),
        ),
        sa.ForeignKeyConstraint(
            ["extraction_run_id"],
            ["extraction_runs.id"],
            name=op.f("fk_document_blocks_extraction_run_id_extraction_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_document_blocks")),
        sa.UniqueConstraint(
            "extraction_run_id",
            "source_order",
            name="uq_document_blocks_extraction_run_id_source_order",
        ),
        sa.UniqueConstraint(
            "extraction_run_id",
            "location_key",
            name="uq_document_blocks_extraction_run_id_location_key",
        ),
        sa.UniqueConstraint(
            "extraction_run_id",
            "anchor_hash",
            name="uq_document_blocks_extraction_run_id_anchor_hash",
        ),
    )
    op.create_index(
        op.f("ix_document_blocks_extraction_run_id"),
        "document_blocks",
        ["extraction_run_id"],
        unique=False,
    )

    op.create_table(
        "source_evidences",
        sa.Column("block_id", sa.Uuid(), nullable=False),
        sa.Column("start_offset", sa.Integer(), nullable=False),
        sa.Column("end_offset", sa.Integer(), nullable=False),
        sa.Column("excerpt", sa.Text(), nullable=False),
        sa.Column("excerpt_hash", sa.String(length=64), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint(
            "start_offset >= 0",
            name=op.f("ck_source_evidences_source_evidences_start_offset_non_negative"),
        ),
        sa.CheckConstraint(
            "end_offset > start_offset",
            name=op.f("ck_source_evidences_source_evidences_end_offset_after_start"),
        ),
        sa.CheckConstraint(
            "char_length(excerpt) > 0",
            name=op.f("ck_source_evidences_source_evidences_excerpt_non_empty"),
        ),
        sa.CheckConstraint(
            "excerpt_hash ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_source_evidences_source_evidences_excerpt_hash_format"),
        ),
        sa.ForeignKeyConstraint(
            ["block_id"],
            ["document_blocks.id"],
            name=op.f("fk_source_evidences_block_id_document_blocks"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_source_evidences")),
        sa.UniqueConstraint(
            "block_id",
            "start_offset",
            "end_offset",
            name="uq_source_evidences_block_id_start_offset_end_offset",
        ),
    )
    op.create_index(
        op.f("ix_source_evidences_block_id"),
        "source_evidences",
        ["block_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_source_evidences_block_id"), table_name="source_evidences")
    op.drop_table("source_evidences")
    op.drop_index(op.f("ix_document_blocks_extraction_run_id"), table_name="document_blocks")
    op.drop_table("document_blocks")
    op.drop_index(op.f("ix_extraction_runs_revision_id"), table_name="extraction_runs")
    op.drop_table("extraction_runs")
