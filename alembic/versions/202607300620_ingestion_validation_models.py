"""ingestion validation report and risk flag models"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "202607300620"
down_revision: str | None = "202607300430"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ingestion_validation_reports",
        sa.Column("revision_id", sa.Uuid(), nullable=False),
        sa.Column("attempt_no", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column(
            "outcome",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column("content_kind", sa.String(length=64), nullable=True),
        sa.Column("content_confidence", sa.Float(), nullable=True),
        sa.Column("recommended_authority", sa.String(length=16), nullable=True),
        sa.Column("detected_language", sa.String(length=32), nullable=True),
        sa.Column("language_confidence", sa.Float(), nullable=True),
        sa.Column(
            "character_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("printable_ratio", sa.Float(), nullable=True),
        sa.Column("garbled_ratio", sa.Float(), nullable=True),
        sa.Column("repetition_ratio", sa.Float(), nullable=True),
        sa.Column(
            "file_checks",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "text_checks",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "classification_details",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("failure_code", sa.String(length=64), nullable=True),
        sa.Column("failure_message", sa.Text(), nullable=True),
        sa.Column("user_decision", sa.String(length=24), nullable=True),
        sa.Column("decided_by_id", sa.Uuid(), nullable=True),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
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
            "attempt_no > 0",
            name=op.f("ck_ingestion_validation_reports_ingestion_validation_reports_attempt_no_positive"),
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name=op.f("ck_ingestion_validation_reports_ingestion_validation_reports_status_valid"),
        ),
        sa.CheckConstraint(
            "outcome IN ('pending', 'pass', 'warning', 'reject', 'quarantine')",
            name=op.f("ck_ingestion_validation_reports_ingestion_validation_reports_outcome_valid"),
        ),
        sa.CheckConstraint(
            "recommended_authority IS NULL OR recommended_authority IN ('formal', 'reference')",
            name=op.f("ck_ingestion_validation_reports_ingestion_validation_reports_recommended_authority_valid"),
        ),
        sa.CheckConstraint(
            "user_decision IS NULL OR user_decision IN ('continue_formal', 'continue_reference', 'cancel')",
            name=op.f("ck_ingestion_validation_reports_ingestion_validation_reports_user_decision_valid"),
        ),
        sa.CheckConstraint(
            "content_confidence IS NULL OR (content_confidence >= 0 AND content_confidence <= 1)",
            name=op.f("ck_ingestion_validation_reports_ingestion_validation_reports_content_confidence_range"),
        ),
        sa.CheckConstraint(
            "language_confidence IS NULL OR (language_confidence >= 0 AND language_confidence <= 1)",
            name=op.f("ck_ingestion_validation_reports_ingestion_validation_reports_language_confidence_range"),
        ),
        sa.CheckConstraint(
            "printable_ratio IS NULL OR (printable_ratio >= 0 AND printable_ratio <= 1)",
            name=op.f("ck_ingestion_validation_reports_ingestion_validation_reports_printable_ratio_range"),
        ),
        sa.CheckConstraint(
            "garbled_ratio IS NULL OR (garbled_ratio >= 0 AND garbled_ratio <= 1)",
            name=op.f("ck_ingestion_validation_reports_ingestion_validation_reports_garbled_ratio_range"),
        ),
        sa.CheckConstraint(
            "repetition_ratio IS NULL OR (repetition_ratio >= 0 AND repetition_ratio <= 1)",
            name=op.f("ck_ingestion_validation_reports_ingestion_validation_reports_repetition_ratio_range"),
        ),
        sa.CheckConstraint(
            "character_count >= 0",
            name=op.f("ck_ingestion_validation_reports_ingestion_validation_reports_character_count_non_negative"),
        ),
        sa.CheckConstraint(
            "((decided_by_id IS NULL AND decided_at IS NULL) OR (decided_by_id IS NOT NULL AND decided_at IS NOT NULL))",
            name=op.f("ck_ingestion_validation_reports_ingestion_validation_reports_decision_actor_time_pair"),
        ),
        sa.CheckConstraint(
            "((user_decision IS NULL AND decided_by_id IS NULL AND decided_at IS NULL) OR (user_decision IS NOT NULL AND decided_by_id IS NOT NULL AND decided_at IS NOT NULL))",
            name=op.f("ck_ingestion_validation_reports_ingestion_validation_reports_user_decision_triplet"),
        ),
        sa.ForeignKeyConstraint(
            ["revision_id"],
            ["document_revisions.id"],
            name=op.f("fk_ingestion_validation_reports_revision_id_document_revisions"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["decided_by_id"],
            ["users.id"],
            name=op.f("fk_ingestion_validation_reports_decided_by_id_users"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ingestion_validation_reports")),
        sa.UniqueConstraint(
            "revision_id",
            "attempt_no",
            name="uq_ingestion_validation_reports_revision_id_attempt_no",
        ),
    )
    op.create_index(
        op.f("ix_ingestion_validation_reports_revision_id"),
        "ingestion_validation_reports",
        ["revision_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_ingestion_validation_reports_decided_by_id"),
        "ingestion_validation_reports",
        ["decided_by_id"],
        unique=False,
    )

    op.create_table(
        "ingestion_risk_flags",
        sa.Column("report_id", sa.Uuid(), nullable=False),
        sa.Column("category", sa.String(length=32), nullable=False),
        sa.Column("kind", sa.String(length=80), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("detector", sa.String(length=16), nullable=False),
        sa.Column("message", sa.String(length=500), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("location", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "details",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("evidence_excerpt", sa.Text(), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint(
            "category IN ('file', 'text_quality', 'content', 'privacy', 'credential', 'prompt_injection', 'security', 'other')",
            name=op.f("ck_ingestion_risk_flags_ingestion_risk_flags_category_valid"),
        ),
        sa.CheckConstraint(
            "severity IN ('info', 'warning', 'blocker')",
            name=op.f("ck_ingestion_risk_flags_ingestion_risk_flags_severity_valid"),
        ),
        sa.CheckConstraint(
            "detector IN ('deterministic', 'ai', 'manual')",
            name=op.f("ck_ingestion_risk_flags_ingestion_risk_flags_detector_valid"),
        ),
        sa.CheckConstraint(
            "char_length(kind) BETWEEN 1 AND 80",
            name=op.f("ck_ingestion_risk_flags_ingestion_risk_flags_kind_length"),
        ),
        sa.CheckConstraint(
            "char_length(message) BETWEEN 1 AND 500",
            name=op.f("ck_ingestion_risk_flags_ingestion_risk_flags_message_length"),
        ),
        sa.CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name=op.f("ck_ingestion_risk_flags_ingestion_risk_flags_confidence_range"),
        ),
        sa.CheckConstraint(
            "evidence_excerpt IS NULL OR char_length(evidence_excerpt) <= 500",
            name=op.f("ck_ingestion_risk_flags_ingestion_risk_flags_evidence_excerpt_length"),
        ),
        sa.ForeignKeyConstraint(
            ["report_id"],
            ["ingestion_validation_reports.id"],
            name=op.f("fk_ingestion_risk_flags_report_id_ingestion_validation_reports"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ingestion_risk_flags")),
    )
    op.create_index(
        op.f("ix_ingestion_risk_flags_report_id"),
        "ingestion_risk_flags",
        ["report_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_ingestion_risk_flags_report_id"), table_name="ingestion_risk_flags")
    op.drop_table("ingestion_risk_flags")
    op.drop_index(
        op.f("ix_ingestion_validation_reports_decided_by_id"),
        table_name="ingestion_validation_reports",
    )
    op.drop_index(
        op.f("ix_ingestion_validation_reports_revision_id"),
        table_name="ingestion_validation_reports",
    )
    op.drop_table("ingestion_validation_reports")
