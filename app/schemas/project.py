from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator

from app.models.project import ProjectStatus, ProjectVisibility
from app.utils.validation import normalize_text, validate_slug


class ProjectCreate(BaseModel):
    name: str
    slug: str
    description: str | None = None
    visibility: ProjectVisibility = ProjectVisibility.PRIVATE

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name_field(cls, value: str) -> str:
        normalized = normalize_text(value)
        if not 1 <= len(normalized) <= 100:
            raise ValueError("name must be between 1 and 100 characters")
        return normalized

    @field_validator("slug", mode="before")
    @classmethod
    def validate_slug_field(cls, value: str) -> str:
        return validate_slug(value)

    @field_validator("description", mode="before")
    @classmethod
    def normalize_description_field(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_text(value)
        return normalized or None


class ProjectRead(BaseModel):
    id: uuid.UUID
    name: str
    slug: str
    description: str | None = None
    visibility: ProjectVisibility
    status: ProjectStatus
    created_by_id: uuid.UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
