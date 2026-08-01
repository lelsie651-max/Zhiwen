from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.schemas.agent_consistency_check import (
    ConsistencyCheckAssessment,
    ConsistencyCheckResponse,
)


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


class MaterializedConsistencyCheckBatch(BaseModel):
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


class ConsistencyCheckBatchExecutionResult(BaseModel):
    project_id: uuid.UUID
    consistency_application_id: uuid.UUID
    source_result_manifest_hash: str
    plan_manifest_hash: str
    batch_index: int
    batch_manifest_hash: str

    input_batch_id: uuid.UUID | None
    inference_run_id: uuid.UUID | None

    request_hash: str | None
    message_content_hash: str | None

    skipped_empty: bool
    reused_completed_run: bool
    response: ConsistencyCheckResponse

    response_model: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )


class ConsistencyCheckPlanExecutionResult(BaseModel):
    project_id: uuid.UUID
    consistency_application_id: uuid.UUID
    source_result_manifest_hash: str
    plan_manifest_hash: str

    batch_count: int
    executed_batch_count: int
    skipped_empty_batch_count: int

    inference_run_ids: tuple[uuid.UUID | None, ...]
    assessments: tuple[ConsistencyCheckAssessment, ...]
    result_manifest_hash: str

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )
