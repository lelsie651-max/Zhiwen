from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from app.models.document_revision import UploadIntent
from app.utils.validation import normalize_text


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = normalize_text(value)
    return normalized or None


class DocumentUploadSubmit(BaseModel):
    upload_intent: UploadIntent
    title: str | None = None
    description: str | None = None
    logical_order_kind: str | None = None
    logical_order_value: str | None = None
    logical_order_index: Decimal | None = None
    effective_from: datetime | None = None
    effective_to: datetime | None = None
    document_id: uuid.UUID | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator(
        "title",
        "description",
        "logical_order_kind",
        "logical_order_value",
        "logical_order_index",
        "effective_from",
        "effective_to",
        "document_id",
        mode="before",
    )
    @classmethod
    def normalize_blank_values(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = normalize_text(value)
            return normalized or None
        return value

    @field_validator("title", mode="after")
    @classmethod
    def validate_title_field(cls, value: str | None) -> str | None:
        normalized = _normalize_optional_text(value)
        if normalized is None:
            return None
        if not 1 <= len(normalized) <= 200:
            raise ValueError("title must be between 1 and 200 characters")
        return normalized

    @field_validator("description", "logical_order_kind", "logical_order_value", mode="after")
    @classmethod
    def normalize_optional_text_field(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value)

    @model_validator(mode="after")
    def validate_intent_fields(self) -> "DocumentUploadSubmit":
        if self.upload_intent == UploadIntent.NEW_DOCUMENT:
            if self.title is None:
                raise ValueError("title is required when upload_intent=new_document")
            if self.document_id is not None:
                raise ValueError("document_id must be empty when upload_intent=new_document")
        else:
            if self.document_id is None:
                raise ValueError("document_id is required when upload_intent=revision")

        if (
            self.effective_from is not None
            and self.effective_to is not None
            and self.effective_to < self.effective_from
        ):
            raise ValueError("effective_to must not be earlier than effective_from")

        return self
