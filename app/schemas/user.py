from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, field_validator

from app.models.user import UserStatus
from app.utils.validation import normalize_optional_email, normalize_text, validate_handle


class UserCreate(BaseModel):
    handle: str
    display_name: str
    email: EmailStr | None = None
    avatar_url: str | None = None
    locale: str = "zh-CN"
    timezone: str = "Asia/Shanghai"
    status: UserStatus = UserStatus.ACTIVE

    model_config = ConfigDict(from_attributes=True)

    @field_validator("handle", mode="before")
    @classmethod
    def validate_handle_field(cls, value: str) -> str:
        return validate_handle(value)

    @field_validator("display_name", "locale", "timezone", mode="before")
    @classmethod
    def normalize_text_field(cls, value: str) -> str:
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email_field(cls, value: str | None) -> str | None:
        return normalize_optional_email(value)


class UserRead(BaseModel):
    id: uuid.UUID
    handle: str
    display_name: str
    email: EmailStr | None = None
    avatar_url: str | None = None
    locale: str
    timezone: str
    status: UserStatus
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
