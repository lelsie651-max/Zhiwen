from __future__ import annotations

from datetime import datetime
import re
import uuid
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.models.base import Base, UUIDPrimaryKeyMixin, utc_now
from app.utils.validation import normalize_text

if TYPE_CHECKING:
    from app.models.document_content import DocumentBlock
    from app.models.fact_extraction_application import FactExtractionBatchApplication
    from app.models.fact import FactValue
    from app.models.project import Project


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
BLOCK_REF_PATTERN = re.compile(r"^B[0-9]{4,}$")


class InferenceTaskType(StrEnum):
    FACT_EXTRACTION = "fact_extraction"
    SCHEMA_INFERENCE = "schema_inference"
    CONSISTENCY_CHECK = "consistency_check"


class InferenceRunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


_TASK_TYPE_SQL = "('fact_extraction', 'schema_inference', 'consistency_check')"
_RUN_STATUS_SQL = "('pending', 'running', 'completed', 'failed')"
_BLOCK_TYPE_SQL = "('heading', 'paragraph', 'list_item', 'table_row', 'code', 'page_text')"

# Non-terminal runs (pending/running) must not carry any response identity or
# token accounting. Shared here so the ORM and the migration stay identical.
_RUN_RESPONSE_NULLS_SQL = (
    "response_json IS NULL AND response_hash IS NULL AND response_json_hash IS NULL "
    "AND response_model IS NULL AND response_id IS NULL "
    "AND system_fingerprint IS NULL AND finish_reason IS NULL "
    "AND prompt_tokens IS NULL AND completion_tokens IS NULL "
    "AND total_tokens IS NULL AND prompt_cache_hit_tokens IS NULL "
    "AND prompt_cache_miss_tokens IS NULL AND reasoning_tokens IS NULL"
)
_RUN_PENDING_SHAPE_SQL = (
    "status <> 'pending' OR ("
    "started_at IS NULL AND completed_at IS NULL "
    "AND failure_code IS NULL AND failure_message IS NULL "
    "AND attempt_count = 0 "
    f"AND {_RUN_RESPONSE_NULLS_SQL})"
)
_RUN_RUNNING_SHAPE_SQL = (
    "status <> 'running' OR ("
    "started_at IS NOT NULL AND completed_at IS NULL "
    "AND failure_code IS NULL AND failure_message IS NULL "
    "AND attempt_count = 0 "
    f"AND {_RUN_RESPONSE_NULLS_SQL})"
)


class InferenceInputBatch(UUIDPrimaryKeyMixin, Base):
    """An immutable, ordered snapshot of the blocks fed into one AI task.

    The batch and its blocks are written once and never mutated. All statistics
    and hashes are computed server-side so a run's inputs stay reproducible.
    """

    __tablename__ = "inference_input_batches"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "task_type",
            "snapshot_hash",
            name="uq_inference_input_batches_project_id_task_type_snapshot_hash",
        ),
        CheckConstraint(
            f"task_type IN {_TASK_TYPE_SQL}",
            name="inference_input_batches_task_type_valid",
        ),
        CheckConstraint("block_count > 0", name="inference_input_batches_block_count_positive"),
        CheckConstraint(
            "character_count > 0",
            name="inference_input_batches_character_count_positive",
        ),
        CheckConstraint(
            "snapshot_hash ~ '^[0-9a-f]{64}$'",
            name="inference_input_batches_snapshot_hash_format",
        ),
        CheckConstraint(
            "char_length(selection_strategy) BETWEEN 1 AND 64",
            name="inference_input_batches_selection_strategy_length",
        ),
        CheckConstraint(
            "jsonb_typeof(selection_metadata) = 'object'",
            name="inference_input_batches_selection_metadata_is_object",
        ),
    )

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    task_type: Mapped[str] = mapped_column(String(32), nullable=False)
    selection_strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    selection_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    block_count: Mapped[int] = mapped_column(Integer, nullable=False)
    character_count: Mapped[int] = mapped_column(Integer, nullable=False)
    snapshot_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    project: Mapped["Project"] = relationship(
        back_populates="inference_input_batches",
        foreign_keys=[project_id],
    )
    blocks: Mapped[list["InferenceInputBlock"]] = relationship(
        back_populates="batch",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="InferenceInputBlock.source_order",
    )
    runs: Mapped[list["InferenceRun"]] = relationship(
        back_populates="input_batch",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    @validates("selection_strategy")
    def validate_selection_strategy(self, _key: str, value: str) -> str:
        normalized = normalize_text(value)
        if not 1 <= len(normalized) <= 64:
            raise ValueError("selection_strategy must be 1-64 characters")
        return normalized

    @validates("snapshot_hash")
    def validate_snapshot_hash(self, _key: str, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("snapshot_hash must be a string")
        normalized = value.lower()
        if not SHA256_PATTERN.fullmatch(normalized):
            raise ValueError("snapshot_hash must be a 64-character lowercase hexadecimal string")
        return normalized


class InferenceInputBlock(UUIDPrimaryKeyMixin, Base):
    """One verbatim block inside an :class:`InferenceInputBatch`.

    ``content_text`` is a character-for-character copy of the source
    ``DocumentBlock.raw_text`` at snapshot time and must never be normalized.
    ``source_block_id_snapshot`` preserves the original block id even if the live
    FK is later nulled by a delete.
    """

    __tablename__ = "inference_input_blocks"
    __table_args__ = (
        UniqueConstraint(
            "batch_id",
            "source_order",
            name="uq_inference_input_blocks_batch_id_source_order",
        ),
        UniqueConstraint(
            "batch_id",
            "block_ref",
            name="uq_inference_input_blocks_batch_id_block_ref",
        ),
        UniqueConstraint(
            "batch_id",
            "source_block_id_snapshot",
            name="uq_inference_input_blocks_batch_id_source_block_id_snapshot",
        ),
        CheckConstraint("source_order >= 0", name="inference_input_blocks_source_order_non_negative"),
        CheckConstraint(
            "block_ref ~ '^B[0-9]{4,}$'",
            name="inference_input_blocks_block_ref_format",
        ),
        CheckConstraint(
            f"block_type IN {_BLOCK_TYPE_SQL}",
            name="inference_input_blocks_block_type_valid",
        ),
        CheckConstraint(
            "char_length(location_key) > 0",
            name="inference_input_blocks_location_key_non_empty",
        ),
        CheckConstraint(
            "anchor_hash ~ '^[0-9a-f]{64}$'",
            name="inference_input_blocks_anchor_hash_format",
        ),
        CheckConstraint(
            "char_length(content_text) > 0",
            name="inference_input_blocks_content_text_non_empty",
        ),
        CheckConstraint(
            "content_hash ~ '^[0-9a-f]{64}$'",
            name="inference_input_blocks_content_hash_format",
        ),
        CheckConstraint("page_no IS NULL OR page_no > 0", name="inference_input_blocks_page_no_positive"),
        CheckConstraint("start_line IS NULL OR start_line > 0", name="inference_input_blocks_start_line_positive"),
        CheckConstraint("end_line IS NULL OR end_line > 0", name="inference_input_blocks_end_line_positive"),
        CheckConstraint(
            "jsonb_typeof(heading_path) = 'array'",
            name="inference_input_blocks_heading_path_is_array",
        ),
    )

    batch_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("inference_input_batches.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_order: Mapped[int] = mapped_column(Integer, nullable=False)
    block_ref: Mapped[str] = mapped_column(String(16), nullable=False)
    document_block_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("document_blocks.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    source_block_id_snapshot: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    extraction_run_id_snapshot: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    block_type: Mapped[str] = mapped_column(String(16), nullable=False)
    location_key: Mapped[str] = mapped_column(String(255), nullable=False)
    anchor_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    page_no: Mapped[int | None] = mapped_column(Integer, nullable=True)
    start_line: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_line: Mapped[int | None] = mapped_column(Integer, nullable=True)
    heading_path: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )
    content_text: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    batch: Mapped["InferenceInputBatch"] = relationship(
        back_populates="blocks",
        foreign_keys=[batch_id],
    )
    document_block: Mapped["DocumentBlock | None"] = relationship(
        back_populates="inference_input_references",
        foreign_keys=[document_block_id],
    )

    @validates("block_ref")
    def validate_block_ref(self, _key: str, value: str) -> str:
        if not isinstance(value, str) or not BLOCK_REF_PATTERN.fullmatch(value):
            raise ValueError("block_ref must match the pattern B0001")
        return value

    @validates("location_key")
    def validate_location_key(self, _key: str, value: str) -> str:
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("location_key must not be empty")
        return normalized

    @validates("content_text")
    def validate_content_text(self, _key: str, value: str) -> str:
        # Snapshot fidelity: never strip, normalize, or rewrite the block text.
        if not isinstance(value, str):
            raise ValueError("content_text must be a string")
        if value == "":
            raise ValueError("content_text must not be empty")
        return value

    @validates("anchor_hash", "content_hash")
    def validate_hash_field(self, _key: str, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("hash must be a string")
        normalized = value.lower()
        if not SHA256_PATTERN.fullmatch(normalized):
            raise ValueError("hash must be a 64-character lowercase hexadecimal string")
        return normalized


class InferenceRun(UUIDPrimaryKeyMixin, Base):
    """One recorded LLM call for a given input batch, agent, and prompt version.

    Carries the full, traceable identity of the call (provider, models, response
    id, finish reason, token usage) plus a strict-parsed ``response_json``. It
    never stores API keys, raw prompts, or the raw model reply string.
    """

    __tablename__ = "inference_runs"
    __table_args__ = (
        UniqueConstraint(
            "input_batch_id",
            "agent_name",
            "prompt_version",
            "attempt_no",
            # Kept <= 63 chars for PostgreSQL's identifier limit (the descriptive
            # name would be truncated non-deterministically otherwise).
            name="uq_inference_runs_batch_agent_prompt_attempt",
        ),
        Index(
            "uq_ir_active_request",
            "input_batch_id",
            "request_hash",
            unique=True,
            postgresql_where=text("status IN ('pending', 'running')"),
        ),
        CheckConstraint(
            f"status IN {_RUN_STATUS_SQL}",
            name="inference_runs_status_valid",
        ),
        CheckConstraint(
            f"task_type IN {_TASK_TYPE_SQL}",
            name="inference_runs_task_type_valid",
        ),
        CheckConstraint("attempt_no > 0", name="inference_runs_attempt_no_positive"),
        CheckConstraint("attempt_count >= 0", name="inference_runs_attempt_count_non_negative"),
        CheckConstraint(
            "prompt_tokens IS NULL OR prompt_tokens >= 0",
            name="inference_runs_prompt_tokens_non_negative",
        ),
        CheckConstraint(
            "completion_tokens IS NULL OR completion_tokens >= 0",
            name="inference_runs_completion_tokens_non_negative",
        ),
        CheckConstraint(
            "total_tokens IS NULL OR total_tokens >= 0",
            name="inference_runs_total_tokens_non_negative",
        ),
        CheckConstraint(
            "prompt_cache_hit_tokens IS NULL OR prompt_cache_hit_tokens >= 0",
            name="inference_runs_prompt_cache_hit_tokens_non_negative",
        ),
        CheckConstraint(
            "prompt_cache_miss_tokens IS NULL OR prompt_cache_miss_tokens >= 0",
            name="inference_runs_prompt_cache_miss_tokens_non_negative",
        ),
        CheckConstraint(
            "reasoning_tokens IS NULL OR reasoning_tokens >= 0",
            name="inference_runs_reasoning_tokens_non_negative",
        ),
        CheckConstraint(
            "temperature >= 0 AND temperature <= 2",
            name="inference_runs_temperature_range",
        ),
        CheckConstraint("max_output_tokens > 0", name="inference_runs_max_output_tokens_positive"),
        CheckConstraint(
            "prompt_contract_hash IS NULL OR prompt_contract_hash ~ '^[0-9a-f]{64}$'",
            name="inference_runs_prompt_contract_hash_format",
        ),
        CheckConstraint(
            "request_hash IS NULL OR request_hash ~ '^[0-9a-f]{64}$'",
            name="inference_runs_request_hash_format",
        ),
        CheckConstraint(
            "response_hash IS NULL OR response_hash ~ '^[0-9a-f]{64}$'",
            name="inference_runs_response_hash_format",
        ),
        CheckConstraint(
            "response_json_hash IS NULL OR response_json_hash ~ '^[0-9a-f]{64}$'",
            name="inference_runs_response_json_hash_format",
        ),
        CheckConstraint(
            "response_json IS NULL OR jsonb_typeof(response_json) = 'object'",
            name="inference_runs_response_json_is_object",
        ),
        CheckConstraint(
            "completed_at IS NULL OR started_at IS NULL OR completed_at >= started_at",
            name="inference_runs_completed_after_started",
        ),
        # State-shape invariants, enforced at the database level.
        CheckConstraint(_RUN_PENDING_SHAPE_SQL, name="inference_runs_pending_shape"),
        CheckConstraint(_RUN_RUNNING_SHAPE_SQL, name="inference_runs_running_shape"),
        CheckConstraint(
            "jsonb_typeof(request_metadata) = 'object'",
            name="inference_runs_request_metadata_is_object",
        ),
        CheckConstraint(
            "status <> 'completed' OR ("
            "started_at IS NOT NULL AND completed_at IS NOT NULL AND finish_reason = 'stop' "
            "AND response_model IS NOT NULL AND response_json IS NOT NULL AND response_hash IS NOT NULL "
            "AND response_json_hash IS NOT NULL "
            "AND attempt_count > 0 AND failure_code IS NULL AND failure_message IS NULL)",
            name="inference_runs_completed_shape",
        ),
        CheckConstraint(
            "status <> 'failed' OR ("
            "started_at IS NOT NULL AND completed_at IS NOT NULL AND failure_code IS NOT NULL "
            "AND response_json IS NULL AND response_hash IS NULL AND response_json_hash IS NULL)",
            name="inference_runs_failed_shape",
        ),
    )

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    input_batch_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("inference_input_batches.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    task_type: Mapped[str] = mapped_column(String(32), nullable=False)
    attempt_no: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=InferenceRunStatus.PENDING.value,
        server_default=InferenceRunStatus.PENDING.value,
    )
    agent_name: Mapped[str] = mapped_column(String(100), nullable=False)
    agent_version: Mapped[str] = mapped_column(String(32), nullable=False)
    prompt_name: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False)
    prompt_contract_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    request_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    request_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    requested_model: Mapped[str] = mapped_column(String(128), nullable=False)
    response_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    response_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    system_fingerprint: Mapped[str | None] = mapped_column(String(128), nullable=True)
    finish_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    temperature: Mapped[float] = mapped_column(Float, nullable=False)
    max_output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    attempt_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    prompt_cache_hit_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    prompt_cache_miss_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reasoning_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    response_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    response_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    response_json_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    failure_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    failure_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    project: Mapped["Project"] = relationship(
        back_populates="inference_runs",
        foreign_keys=[project_id],
    )
    input_batch: Mapped["InferenceInputBatch"] = relationship(
        back_populates="runs",
        foreign_keys=[input_batch_id],
    )
    fact_values: Mapped[list["FactValue"]] = relationship(
        back_populates="inference_run",
        foreign_keys="FactValue.inference_run_id",
    )
    batch_application: Mapped["FactExtractionBatchApplication | None"] = relationship(
        back_populates="inference_run",
        foreign_keys="FactExtractionBatchApplication.inference_run_id",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    @validates(
        "agent_name",
        "agent_version",
        "prompt_name",
        "prompt_version",
        "provider",
        "requested_model",
    )
    def validate_required_text(self, _key: str, value: str) -> str:
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @validates(
        "response_model",
        "response_id",
        "system_fingerprint",
        "finish_reason",
        "failure_code",
        "failure_message",
    )
    def normalize_optional_text(self, _key: str, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_text(value)
        return normalized or None

    @validates("prompt_contract_hash", "request_hash", "response_hash", "response_json_hash")
    def validate_optional_hash(self, _key: str, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.lower()
        if not SHA256_PATTERN.fullmatch(normalized):
            raise ValueError("hash must be a 64-character lowercase hexadecimal string")
        return normalized
