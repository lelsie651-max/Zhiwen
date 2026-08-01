from __future__ import annotations

from datetime import datetime
import re
import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, String, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.models.base import Base, UUIDPrimaryKeyMixin, utc_now
from app.utils.validation import normalize_text

if TYPE_CHECKING:
    from app.models.document_content import ExtractionRun
    from app.models.fact_extraction_application import FactExtractionBatchApplication
    from app.models.inference import InferenceInputBatch, InferenceRun
    from app.models.project import Project


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_ORCH_STATUS_SQL = "('planned', 'running', 'completed', 'partial', 'failed')"
_BATCH_STATUS_SQL = "('pending', 'running', 'completed', 'failed')"


class FactExtractionOrchestration(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "fact_extraction_orchestrations"
    __table_args__ = (
        UniqueConstraint(
            "id",
            "project_id",
            name="uq_feo_id_project",
        ),
        UniqueConstraint(
            "id",
            "extraction_run_id",
            name="uq_feo_id_extraction_run",
        ),
        UniqueConstraint(
            "extraction_run_id",
            "request_hash",
            "attempt_no",
            name="uq_feo_extraction_run_request_attempt",
        ),
        Index(
            "uq_feo_active_request",
            "extraction_run_id",
            "request_hash",
            unique=True,
            postgresql_where=text("status IN ('planned', 'running')"),
        ),
        CheckConstraint(f"status IN {_ORCH_STATUS_SQL}", name="feo_status_valid"),
        CheckConstraint("attempt_no > 0", name="feo_attempt_no_positive"),
        CheckConstraint("batch_count > 0", name="feo_batch_count_positive"),
        CheckConstraint("completed_batch_count >= 0", name="feo_completed_batch_count_non_negative"),
        CheckConstraint("failed_batch_count >= 0", name="feo_failed_batch_count_non_negative"),
        CheckConstraint(
            "completed_batch_count + failed_batch_count <= batch_count",
            name="feo_terminal_batch_counts_within_batch_count",
        ),
        CheckConstraint("proposal_count >= 0", name="feo_proposal_count_non_negative"),
        CheckConstraint("created_count >= 0", name="feo_created_count_non_negative"),
        CheckConstraint("reused_count >= 0", name="feo_reused_count_non_negative"),
        CheckConstraint("withheld_count >= 0", name="feo_withheld_count_non_negative"),
        CheckConstraint("request_hash ~ '^[0-9a-f]{64}$'", name="feo_request_hash_format"),
        CheckConstraint("plan_hash ~ '^[0-9a-f]{64}$'", name="feo_plan_hash_format"),
        CheckConstraint("plan_json_hash ~ '^[0-9a-f]{64}$'", name="feo_plan_json_hash_format"),
        CheckConstraint(
            "prompt_contract_hash ~ '^[0-9a-f]{64}$'",
            name="feo_prompt_contract_hash_format",
        ),
        CheckConstraint("jsonb_typeof(plan_json) = 'object'", name="feo_plan_json_is_object"),
        CheckConstraint(
            "status <> 'planned' OR ("
            "started_at IS NULL AND completed_at IS NULL AND failure_code IS NULL "
            "AND completed_batch_count = 0 AND failed_batch_count = 0 "
            "AND proposal_count = 0 AND created_count = 0 AND reused_count = 0 AND withheld_count = 0)",
            name="feo_planned_shape",
        ),
        CheckConstraint(
            "status <> 'running' OR (started_at IS NOT NULL AND completed_at IS NULL AND failure_code IS NULL)",
            name="feo_running_shape",
        ),
        CheckConstraint(
            "status <> 'completed' OR ("
            "started_at IS NOT NULL AND completed_at IS NOT NULL "
            "AND completed_batch_count = batch_count AND failed_batch_count = 0 "
            "AND failure_code IS NULL)",
            name="feo_completed_shape",
        ),
        CheckConstraint(
            "status <> 'partial' OR ("
            "started_at IS NOT NULL AND completed_at IS NOT NULL "
            "AND completed_batch_count > 0 AND failed_batch_count > 0 "
            "AND completed_batch_count + failed_batch_count = batch_count)",
            name="feo_partial_shape",
        ),
        CheckConstraint(
            "status <> 'failed' OR ("
            "started_at IS NOT NULL AND completed_at IS NOT NULL "
            "AND completed_batch_count = 0 AND failed_batch_count = batch_count "
            "AND failure_code IS NOT NULL)",
            name="feo_failed_shape",
        ),
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
    attempt_no: Mapped[int] = mapped_column(Integer, nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    plan_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    plan_json_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    plan_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    coordinator_name: Mapped[str] = mapped_column(String(64), nullable=False)
    coordinator_version: Mapped[str] = mapped_column(String(32), nullable=False)
    planner_name: Mapped[str] = mapped_column(String(64), nullable=False)
    planner_version: Mapped[str] = mapped_column(String(32), nullable=False)
    agent_name: Mapped[str] = mapped_column(String(100), nullable=False)
    agent_version: Mapped[str] = mapped_column(String(32), nullable=False)
    prompt_name: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False)
    prompt_contract_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    requested_model: Mapped[str] = mapped_column(String(128), nullable=False)
    executor_name: Mapped[str] = mapped_column(String(64), nullable=False)
    executor_version: Mapped[str] = mapped_column(String(32), nullable=False)
    persistence_name: Mapped[str] = mapped_column(String(64), nullable=False)
    persistence_version: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_resolution_policy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_resolution_policy_version: Mapped[str] = mapped_column(String(32), nullable=False)
    batch_count: Mapped[int] = mapped_column(Integer, nullable=False)
    completed_batch_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    failed_batch_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    proposal_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    created_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    reused_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    withheld_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    failure_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    project: Mapped["Project"] = relationship(foreign_keys=[project_id])
    extraction_run: Mapped["ExtractionRun"] = relationship(foreign_keys=[extraction_run_id])
    batches: Mapped[list["FactExtractionOrchestrationBatch"]] = relationship(
        back_populates="orchestration",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="FactExtractionOrchestrationBatch.batch_index",
    )

    @validates(
        "status",
        "coordinator_name",
        "coordinator_version",
        "planner_name",
        "planner_version",
        "agent_name",
        "agent_version",
        "prompt_name",
        "prompt_version",
        "provider",
        "requested_model",
        "executor_name",
        "executor_version",
        "persistence_name",
        "persistence_version",
        "entity_resolution_policy_name",
        "entity_resolution_policy_version",
        "failure_code",
    )
    def validate_text(self, _key: str, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @validates("request_hash", "plan_hash", "plan_json_hash", "prompt_contract_hash")
    def validate_hash(self, _key: str, value: str) -> str:
        normalized = value.lower()
        if not SHA256_PATTERN.fullmatch(normalized):
            raise ValueError("hash must be a 64-character lowercase hexadecimal string")
        return normalized


class FactExtractionOrchestrationBatch(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "fact_extraction_orch_batches"
    __table_args__ = (
        UniqueConstraint(
            "orchestration_id",
            "batch_index",
            name="uq_feob_orchestration_batch_index",
        ),
        UniqueConstraint(
            "id",
            "orchestration_id",
            name="uq_feob_id_orchestration",
        ),
        UniqueConstraint(
            "orchestration_id",
            "current_inference_run_id",
            name="uq_feob_orchestration_inference_run",
        ),
        UniqueConstraint("application_id", name="uq_feob_application_id"),
        CheckConstraint(f"status IN {_BATCH_STATUS_SQL}", name="feob_status_valid"),
        CheckConstraint("batch_index >= 0", name="feob_batch_index_non_negative"),
        CheckConstraint("attempt_count >= 0", name="feob_attempt_count_non_negative"),
        CheckConstraint("proposal_count >= 0", name="feob_proposal_count_non_negative"),
        CheckConstraint("created_count >= 0", name="feob_created_count_non_negative"),
        CheckConstraint("reused_count >= 0", name="feob_reused_count_non_negative"),
        CheckConstraint("withheld_count >= 0", name="feob_withheld_count_non_negative"),
        CheckConstraint("batch_plan_hash ~ '^[0-9a-f]{64}$'", name="feob_batch_plan_hash_format"),
        CheckConstraint(
            "(lease_token IS NULL AND lease_expires_at IS NULL) OR (lease_token IS NOT NULL AND lease_expires_at IS NOT NULL)",
            name="feob_lease_pair",
        ),
        CheckConstraint(
            "status <> 'pending' OR (current_inference_run_id IS NULL AND application_id IS NULL "
            "AND lease_token IS NULL AND lease_expires_at IS NULL "
            "AND completed_at IS NULL AND failure_code IS NULL)",
            name="feob_pending_shape",
        ),
        CheckConstraint(
            "status <> 'running' OR (lease_token IS NOT NULL AND lease_expires_at IS NOT NULL AND started_at IS NOT NULL AND application_id IS NULL AND completed_at IS NULL)",
            name="feob_running_shape",
        ),
        CheckConstraint(
            "status <> 'completed' OR (current_input_batch_id IS NOT NULL AND current_inference_run_id IS NOT NULL AND application_id IS NOT NULL AND started_at IS NOT NULL AND completed_at IS NOT NULL AND lease_token IS NULL AND lease_expires_at IS NULL AND failure_code IS NULL)",
            name="feob_completed_shape",
        ),
        CheckConstraint(
            "status <> 'failed' OR (started_at IS NOT NULL AND completed_at IS NOT NULL AND failure_code IS NOT NULL AND lease_token IS NULL AND lease_expires_at IS NULL)",
            name="feob_failed_shape",
        ),
    )

    orchestration_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("fact_extraction_orchestrations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    batch_index: Mapped[int] = mapped_column(Integer, nullable=False)
    batch_plan_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    current_input_batch_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("inference_input_batches.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    current_inference_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("inference_runs.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    application_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("fact_extraction_batch_applications.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    lease_token: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    proposal_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    created_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    reused_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    withheld_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    failure_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    orchestration: Mapped["FactExtractionOrchestration"] = relationship(
        back_populates="batches",
        foreign_keys=[orchestration_id],
    )
    current_input_batch: Mapped["InferenceInputBatch | None"] = relationship(
        foreign_keys=[current_input_batch_id],
    )
    current_inference_run: Mapped["InferenceRun | None"] = relationship(
        foreign_keys=[current_inference_run_id],
    )
    application: Mapped["FactExtractionBatchApplication | None"] = relationship(
        foreign_keys=[application_id],
    )

    @validates("status", "failure_code")
    def validate_text(self, _key: str, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @validates("batch_plan_hash")
    def validate_hash(self, _key: str, value: str) -> str:
        normalized = value.lower()
        if not SHA256_PATTERN.fullmatch(normalized):
            raise ValueError("hash must be a 64-character lowercase hexadecimal string")
        return normalized
