from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.entity import EntityAliasKind, EntityAliasStatus, EntityStatus, normalize_entity_alias
from app.utils.validation import normalize_text


def _normalize_required_text(value: str, *, field_name: str, max_length: int) -> str:
    normalized = normalize_text(value)
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if len(normalized) > max_length:
        raise ValueError(f"{field_name} must be at most {max_length} characters")
    return normalized


class EntityCreateInput(BaseModel):
    entity_type: str
    canonical_key: str
    display_name: str

    model_config = ConfigDict(extra="forbid")

    @field_validator("entity_type")
    @classmethod
    def _validate_entity_type(cls, value: str) -> str:
        return _normalize_required_text(value, field_name="entity_type", max_length=64)

    @field_validator("canonical_key")
    @classmethod
    def _validate_canonical_key(cls, value: str) -> str:
        return normalize_entity_alias(value)

    @field_validator("display_name")
    @classmethod
    def _validate_display_name(cls, value: str) -> str:
        return _normalize_required_text(value, field_name="display_name", max_length=255)


class EntityAliasCreateInput(BaseModel):
    alias_text: str
    language_code: str = "und"
    alias_kind: EntityAliasKind = EntityAliasKind.ALTERNATE
    is_primary: bool = False

    model_config = ConfigDict(extra="forbid")

    @field_validator("alias_text")
    @classmethod
    def _validate_alias_text(cls, value: str) -> str:
        return _normalize_required_text(value, field_name="alias_text", max_length=255)

    @field_validator("language_code")
    @classmethod
    def _validate_language_code(cls, value: str) -> str:
        return _normalize_required_text(value, field_name="language_code", max_length=32)


class EntityRead(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    entity_type: str
    canonical_key: str
    display_name: str
    identity_hash: str
    status: EntityStatus
    merged_into_entity_id: uuid.UUID | None = None
    created_by_id: uuid.UUID | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, extra="forbid")


class EntityAliasRead(BaseModel):
    id: uuid.UUID
    entity_id: uuid.UUID
    alias_text: str
    normalized_alias: str
    language_code: str
    alias_kind: EntityAliasKind
    status: EntityAliasStatus
    is_primary: bool
    created_by_id: uuid.UUID | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, extra="forbid")


class ResolvedEntityAliasCandidate(BaseModel):
    entity: EntityRead
    alias: EntityAliasRead

    model_config = ConfigDict(extra="forbid")


class EntityAliasResolutionRead(BaseModel):
    candidates: tuple[ResolvedEntityAliasCandidate, ...] = Field(default_factory=tuple)

    model_config = ConfigDict(extra="forbid")
