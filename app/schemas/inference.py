from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.inference import InferenceRunStatus, InferenceTaskType


class InferenceInputBatchRead(BaseModel):
    """Safe view of an input batch. Carries statistics but no block content."""

    id: uuid.UUID
    project_id: uuid.UUID
    task_type: InferenceTaskType
    selection_strategy: str
    selection_metadata: dict[str, Any] = Field(default_factory=dict)
    block_count: int
    character_count: int
    snapshot_hash: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class InferenceInputBlockRead(BaseModel):
    """Safe view of an input block. Never exposes the snapshot ``content_text``."""

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
    heading_path: list[Any] = Field(default_factory=list)
    content_hash: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class InferenceInputBlockInternalRead(InferenceInputBlockRead):
    """Internal view that additionally exposes the verbatim snapshot text."""

    content_text: str


class InferenceRunRead(BaseModel):
    """Safe view of a run. Hides ``response_json`` and ``failure_message``."""

    id: uuid.UUID
    project_id: uuid.UUID
    input_batch_id: uuid.UUID
    task_type: InferenceTaskType
    attempt_no: int
    status: InferenceRunStatus
    agent_name: str
    agent_version: str
    prompt_name: str
    prompt_version: str
    prompt_contract_hash: str | None = None
    request_hash: str | None = None
    request_metadata: dict[str, Any] = Field(default_factory=dict)
    provider: str
    requested_model: str
    response_model: str | None = None
    response_id: str | None = None
    system_fingerprint: str | None = None
    finish_reason: str | None = None
    temperature: float
    max_output_tokens: int
    attempt_count: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    prompt_cache_hit_tokens: int | None = None
    prompt_cache_miss_tokens: int | None = None
    reasoning_tokens: int | None = None
    response_hash: str | None = None
    response_json_hash: str | None = None
    failure_code: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class InferenceRunInternalRead(InferenceRunRead):
    """Internal view that additionally exposes the strict response object and
    the failure message. Still never carries API keys, prompts, or raw replies."""

    response_json: dict[str, Any] | None = None
    failure_message: str | None = None
