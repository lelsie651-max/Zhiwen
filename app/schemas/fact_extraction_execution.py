from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.schemas.agent_fact_extraction import FactExtractionResponse


class InferenceInputBlockSnapshot(BaseModel):
    id: uuid.UUID
    batch_id: uuid.UUID
    source_order: int
    block_ref: str
    document_block_id: uuid.UUID | None = None
    source_block_id_snapshot: uuid.UUID
    extraction_run_id_snapshot: uuid.UUID
    block_type: str
    location_key: str
    anchor_hash: str
    page_no: int | None = None
    start_line: int | None = None
    end_line: int | None = None
    heading_path: tuple[Any, ...] = ()
    content_text: str
    content_hash: str

    model_config = ConfigDict(frozen=True, extra="forbid")


class MaterializedFactExtractionBatch(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    task_type: str
    selection_strategy: str
    selection_metadata: dict[str, Any]
    block_count: int
    character_count: int
    snapshot_hash: str
    blocks: tuple[InferenceInputBlockSnapshot, ...]

    model_config = ConfigDict(frozen=True, extra="forbid")


class FactExtractionBatchExecutionResult(BaseModel):
    project_id: uuid.UUID
    extraction_run_id: uuid.UUID

    plan_hash: str
    batch_index: int
    batch_plan_hash: str

    input_batch_id: uuid.UUID
    inference_run_id: uuid.UUID

    request_hash: str
    message_content_hash: str

    reused_completed_run: bool
    response: FactExtractionResponse

    response_model: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )
