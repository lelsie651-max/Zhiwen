from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.file_ingestion import DetectedFileFormat
from app.utils.validation import normalize_text


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ExtractionOutcome(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    NEEDS_OCR = "needs_ocr"
    FAILED = "failed"


class ExtractedBlockType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE_ROW = "table_row"
    CODE = "code"
    PAGE_TEXT = "page_text"


class ExtractedBlock(BaseModel):
    source_order: int
    block_type: ExtractedBlockType
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
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @field_validator("source_order", "block_index")
    @classmethod
    def validate_non_negative_integer(cls, value: int) -> int:
        if value < 0:
            raise ValueError("value must be greater than or equal to 0")
        return value

    @field_validator("raw_text", "normalized_text", "location_key", mode="before")
    @classmethod
    def validate_text_fields(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("value must be a string")
        normalized = value.strip() if value != "" else value
        if not normalized and value == "":
            raise ValueError("value must not be empty")
        if not normalized and value.strip() == "":
            raise ValueError("value must not be blank")
        return value

    @field_validator("anchor_hash", mode="before")
    @classmethod
    def validate_anchor_hash_field(cls, value: str) -> str:
        normalized = normalize_text(value).lower()
        if not SHA256_PATTERN.fullmatch(normalized):
            raise ValueError("anchor_hash must be a 64-character lowercase hexadecimal string")
        return normalized

    @field_validator("page_no", "start_line", "end_line", "table_index", "row_index", "heading_level")
    @classmethod
    def validate_positive_optional_integer(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("value must be greater than 0")
        return value


class ExtractedDocument(BaseModel):
    outcome: ExtractionOutcome
    detected_format: DetectedFileFormat
    detected_encoding: str | None = None
    page_count: int | None = None
    character_count: int
    block_count: int
    blocks: list[ExtractedBlock] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @field_validator("page_count")
    @classmethod
    def validate_page_count_field(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("page_count must be greater than 0")
        return value

    @field_validator("character_count", "block_count")
    @classmethod
    def validate_non_negative_field(cls, value: int) -> int:
        if value < 0:
            raise ValueError("value must be greater than or equal to 0")
        return value
