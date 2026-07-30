from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.project_member import ProjectMemberRole


class ProjectMemberCreate(BaseModel):
    project_id: uuid.UUID
    user_id: uuid.UUID
    role: ProjectMemberRole

    model_config = ConfigDict(from_attributes=True)


class ProjectMemberRead(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    user_id: uuid.UUID
    role: ProjectMemberRole
    joined_at: datetime

    model_config = ConfigDict(from_attributes=True)
