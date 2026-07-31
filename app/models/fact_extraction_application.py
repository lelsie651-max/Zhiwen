from __future__ import annotations

from datetime import datetime
import re
import uuid
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.models.base import Base, UUIDPrimaryKeyMixin, utc_now
from app.utils.validation import normalize_text


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class FactExtractionBatchApplication(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "fact_extraction_batch_applications"
    __table_args__ = (
        UniqueConstraint("inference_run_id", name="uq_feba_inference_run_id"),
        CheckConstraint(
            "status IN ('applying', 'completed')",
            name="feba_status_valid",
        ),
        CheckConstraint(
            "response_hash ~ '^[0-9a-f]{64}$'",
            name="feba_response_hash_format",
        ),
        CheckConstraint(
            "response_json_hash ~ '^[0-9a-f]{64}$'",
            name="feba_response_json_hash_fmt",
        ),
        CheckConstraint(
            "result_hash IS NULL OR result_hash ~ '^[0-9a-f]{64}$'",
            name="feba_result_hash_format",
        ),
        CheckConstraint(
            "result_json IS NULL OR jsonb_typeof(result_json) = 'object'",
            name="feba_result_json_is_object",
        ),
        CheckConstraint(
            "status <> 'applying' OR (result_json IS NULL AND result_hash IS NULL AND completed_at IS NULL)",
            name="feba_applying_shape",
        ),
        CheckConstraint(
            "status <> 'completed' OR (result_json IS NOT NULL AND result_hash IS NOT NULL AND completed_at IS NOT NULL)",
            name="feba_completed_shape",
        ),
    )

    inference_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("inference_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    extraction_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("extraction_runs.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    input_batch_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("inference_input_batches.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    response_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    response_json_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    persistence_name: Mapped[str] = mapped_column(String(64), nullable=False)
    persistence_version: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_resolution_policy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_resolution_policy_version: Mapped[str] = mapped_column(String(32), nullable=False)
    result_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    result_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    inference_run = relationship(
        "InferenceRun",
        back_populates="batch_application",
        foreign_keys=[inference_run_id],
        uselist=False,
    )

    @validates(
        "status",
        "persistence_name",
        "persistence_version",
        "entity_resolution_policy_name",
        "entity_resolution_policy_version",
    )
    def validate_text(self, _key: str, value: str) -> str:
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @validates("response_hash", "response_json_hash", "result_hash")
    def validate_hash(self, _key: str, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.lower()
        if not SHA256_PATTERN.fullmatch(normalized):
            raise ValueError("hash must be a 64-character lowercase hexadecimal string")
        return normalized
