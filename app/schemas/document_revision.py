from __future__ import annotations

import re
import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from app.models.document_revision import DocumentRevisionStatus, SourceAuthority, UploadIntent
from app.utils.validation import normalize_text


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class DocumentRevisionCreate(BaseModel):
    document_id: uuid.UUID
    revision_no: int
    upload_intent: UploadIntent
    supersedes_revision_id: uuid.UUID | None = None
    original_filename: str
    storage_key: str
    mime_type: str
    file_size_bytes: int
    sha256: str
    detected_language: str | None = None
    language_confidence: float | None = None
    source_authority: SourceAuthority = SourceAuthority.PENDING
    status: DocumentRevisionStatus = DocumentRevisionStatus.UPLOADED

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    @field_validator("revision_no")
    @classmethod
    def validate_revision_no_field(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("revision_no must be greater than 0")
        return value

    @field_validator("original_filename", mode="before")
    @classmethod
    def validate_original_filename_field(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("original_filename must not contain NUL characters")
        normalized = normalize_text(value)
        if not normalized or not 1 <= len(normalized) <= 255:
            raise ValueError("original_filename must be between 1 and 255 characters")
        return normalized

    @field_validator("storage_key", "mime_type", "detected_language", mode="before")
    @classmethod
    def normalize_optional_text_field(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @field_validator("file_size_bytes")
    @classmethod
    def validate_file_size_field(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("file_size_bytes must be greater than 0")
        return value

    @field_validator("sha256", mode="before")
    @classmethod
    def validate_sha256_field(cls, value: str) -> str:
        normalized = normalize_text(value).lower()
        if not SHA256_PATTERN.fullmatch(normalized):
            raise ValueError("sha256 must be a 64-character lowercase hexadecimal string")
        return normalized

    @field_validator("language_confidence")
    @classmethod
    def validate_language_confidence_field(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not 0 <= value <= 1:
            raise ValueError("language_confidence must be between 0 and 1")
        return value

    @field_validator("storage_key")
    @classmethod
    def validate_storage_key_field(cls, value: str, info) -> str:
        original_filename = info.data.get("original_filename")
        if original_filename and value == original_filename:
            raise ValueError("storage_key must not reuse original_filename")
        return value

    @model_validator(mode="after")
    def validate_revision_chain(self) -> "DocumentRevisionCreate":
        if self.revision_no == 1:
            if self.upload_intent != UploadIntent.NEW_DOCUMENT:
                raise ValueError("revision_no=1 must use upload_intent=new_document")
            if self.supersedes_revision_id is not None:
                raise ValueError("revision_no=1 must not set supersedes_revision_id")
            return self

        if self.upload_intent != UploadIntent.REVISION:
            raise ValueError("revision_no>1 must use upload_intent=revision")
        if self.supersedes_revision_id is None:
            raise ValueError("revision_no>1 must set supersedes_revision_id")
        return self


class DocumentRevisionRead(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    revision_no: int
    upload_intent: UploadIntent
    supersedes_revision_id: uuid.UUID | None = None
    original_filename: str
    mime_type: str
    file_size_bytes: int
    sha256: str
    detected_language: str | None = None
    language_confidence: float | None = None
    source_authority: SourceAuthority
    status: DocumentRevisionStatus
    uploaded_by_id: uuid.UUID
    uploaded_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentRevisionInternalRead(DocumentRevisionRead):
    storage_key: str
