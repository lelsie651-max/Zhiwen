from __future__ import annotations

import re
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _require_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_hash(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.lower()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ValueError(f"{field_name} must be a 64-character lowercase hexadecimal string")
    return normalized


class FactExtractionPlannerConfig(BaseModel):
    target_message_characters: int = 24000
    max_message_characters: int = 30000
    max_blocks_per_batch: int = 40
    overlap_block_count: int = 1
    include_preceding_heading: bool = True

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("include_preceding_heading", mode="before")
    @classmethod
    def _validate_include_preceding_heading(cls, value: Any) -> bool:
        if not isinstance(value, bool):
            raise ValueError("include_preceding_heading must be a bool")
        return value

    @field_validator(
        "target_message_characters",
        "max_message_characters",
        "max_blocks_per_batch",
        "overlap_block_count",
        mode="before",
    )
    @classmethod
    def _validate_int_fields(cls, value: Any, info) -> int:
        return _require_int(value, field_name=info.field_name)

    @model_validator(mode="after")
    def validate_ranges(self) -> "FactExtractionPlannerConfig":
        if self.target_message_characters <= 0:
            raise ValueError("target_message_characters must be positive")
        if self.max_message_characters <= 0:
            raise ValueError("max_message_characters must be positive")
        if self.target_message_characters > self.max_message_characters:
            raise ValueError("target_message_characters must not exceed max_message_characters")
        if self.max_blocks_per_batch <= 0:
            raise ValueError("max_blocks_per_batch must be positive")
        if self.overlap_block_count < 0:
            raise ValueError("overlap_block_count must be greater than or equal to 0")
        if self.overlap_block_count >= self.max_blocks_per_batch:
            raise ValueError("overlap_block_count must be less than max_blocks_per_batch")
        return self


class FactExtractionBatchPlan(BaseModel):
    batch_index: int = Field(ge=0)
    block_ids: tuple[uuid.UUID, ...]
    block_refs: tuple[str, ...]
    primary_block_ids: tuple[uuid.UUID, ...]
    overlap_block_ids: tuple[uuid.UUID, ...] = ()
    context_block_ids: tuple[uuid.UUID, ...] = ()
    estimated_message_characters: int = Field(gt=0)
    content_character_count: int = Field(gt=0)
    message_template_hash: str
    plan_hash: str

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator(
        "batch_index",
        "estimated_message_characters",
        "content_character_count",
        mode="before",
    )
    @classmethod
    def _validate_int_fields(cls, value: Any, info) -> int:
        return _require_int(value, field_name=info.field_name)

    @field_validator("message_template_hash", "plan_hash")
    @classmethod
    def _validate_hash_fields(cls, value: str, info) -> str:
        return _require_hash(value, field_name=info.field_name)


class FactExtractionPlan(BaseModel):
    extraction_run_id: uuid.UUID
    prompt_contract_hash: str
    planner_name: str
    planner_version: str
    config: FactExtractionPlannerConfig
    batches: tuple[FactExtractionBatchPlan, ...]
    source_block_count: int = Field(gt=0)
    source_character_count: int = Field(gt=0)
    plan_hash: str

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("source_block_count", "source_character_count", mode="before")
    @classmethod
    def _validate_int_fields(cls, value: Any, info) -> int:
        return _require_int(value, field_name=info.field_name)

    @field_validator("prompt_contract_hash", "plan_hash")
    @classmethod
    def _validate_hash_fields(cls, value: str, info) -> str:
        return _require_hash(value, field_name=info.field_name)
