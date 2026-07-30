from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.document_content import DocumentBlockType, ExtractionRunOutcome, ExtractionRunStatus
from app.schemas.file_ingestion import DetectedFileFormat
from app.utils.validation import normalize_text


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class DocumentBlockRead(BaseModel):
    id: uuid.UUID
    extraction_run_id: uuid.UUID
    source_order: int
    block_type: DocumentBlockType
    raw_text: str
    normalized_text: str
    location_key: str
    anchor_hash: str
    page_no: int | None = None
    block_index: int
    heading_level: int | None = None
    heading_path: list[str] = Field(default_factory=list)
    start_line: int | None = None
    end_line: int | None = None
    table_index: int | None = None
    row_index: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict, validation_alias="block_metadata")
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class ExtractionRunRead(BaseModel):
    id: uuid.UUID
    revision_id: uuid.UUID
    attempt_no: int
    status: ExtractionRunStatus
    outcome: ExtractionRunOutcome
    extractor_name: str
    extractor_version: str
    detected_format: DetectedFileFormat
    detected_encoding: str | None = None
    page_count: int | None = None
    character_count: int
    block_count: int
    warnings: list[Any] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict, validation_alias="content_metadata")
    failure_code: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class ExtractionRunInternalRead(ExtractionRunRead):
    failure_message: str | None = None


class SourceEvidenceCreate(BaseModel):
    block_id: uuid.UUID
    start_offset: int
    end_offset: int

    model_config = ConfigDict(extra="forbid")

    @field_validator("start_offset")
    @classmethod
    def validate_start_offset_field(cls, value: int) -> int:
        if value < 0:
            raise ValueError("start_offset must be greater than or equal to 0")
        return value

    @field_validator("end_offset")
    @classmethod
    def validate_end_offset_field(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("end_offset must be greater than 0")
        return value

    @field_validator("end_offset")
    @classmethod
    def validate_end_offset_after_start(cls, value: int, info) -> int:
        start_offset = info.data.get("start_offset")
        if isinstance(start_offset, int) and value <= start_offset:
            raise ValueError("end_offset must be greater than start_offset")
        return value


class SourceEvidenceRead(BaseModel):
    id: uuid.UUID
    block_id: uuid.UUID
    start_offset: int
    end_offset: int
    excerpt: str
    excerpt_hash: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

    @field_validator("excerpt", mode="before")
    @classmethod
    def validate_excerpt_field(cls, value: str) -> str:
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("excerpt must not be empty")
        return value

    @field_validator("excerpt_hash", mode="before")
    @classmethod
    def validate_excerpt_hash_field(cls, value: str) -> str:
        normalized = normalize_text(value).lower()
        if not SHA256_PATTERN.fullmatch(normalized):
            raise ValueError("excerpt_hash must be a 64-character lowercase hexadecimal string")
        return normalized
