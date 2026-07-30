from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from app.models.document import DocumentStatus
from app.utils.validation import normalize_text


class DocumentCreate(BaseModel):
    title: str
    description: str | None = None
    status: DocumentStatus = DocumentStatus.ACTIVE
    logical_order_kind: str | None = None
    logical_order_value: str | None = None
    logical_order_index: Decimal | None = None
    effective_from: datetime | None = None
    effective_to: datetime | None = None

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    @field_validator("title", mode="before")
    @classmethod
    def validate_title_field(cls, value: str) -> str:
        normalized = normalize_text(value)
        if not normalized or not 1 <= len(normalized) <= 200:
            raise ValueError("title must be between 1 and 200 characters")
        return normalized

    @field_validator("description", "logical_order_kind", "logical_order_value", mode="before")
    @classmethod
    def normalize_optional_text_field(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_text(value)
        return normalized or None

    @model_validator(mode="after")
    def validate_effective_range(self) -> "DocumentCreate":
        if (
            self.effective_from is not None
            and self.effective_to is not None
            and self.effective_to < self.effective_from
        ):
            raise ValueError("effective_to must not be earlier than effective_from")
        return self


class DocumentRead(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    title: str
    description: str | None = None
    status: DocumentStatus
    created_by_id: uuid.UUID
    current_revision_id: uuid.UUID | None = None
    logical_order_kind: str | None = None
    logical_order_value: str | None = None
    logical_order_index: Decimal | None = None
    effective_from: datetime | None = None
    effective_to: datetime | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
