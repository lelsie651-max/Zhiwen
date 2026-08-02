from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.project import Project
from app.models.project_version import ProjectVersion
from app.models.user import User


async def get_project_for_update(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
) -> Project | None:
    result = await session.execute(
        select(Project).where(Project.id == project_id).with_for_update(of=Project)
    )
    return result.scalar_one_or_none()


async def get_project_by_id(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
) -> Project | None:
    result = await session.execute(select(Project).where(Project.id == project_id))
    return result.scalar_one_or_none()


async def get_user_by_id(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
) -> User | None:
    return await session.get(User, user_id)


async def get_project_version_by_id(
    session: AsyncSession,
    *,
    project_version_id: uuid.UUID,
) -> ProjectVersion | None:
    result = await session.execute(
        select(ProjectVersion).where(ProjectVersion.id == project_version_id)
    )
    return result.scalar_one_or_none()


async def get_project_version_by_id_for_update(
    session: AsyncSession,
    *,
    project_version_id: uuid.UUID,
) -> ProjectVersion | None:
    result = await session.execute(
        select(ProjectVersion)
        .where(ProjectVersion.id == project_version_id)
        .with_for_update(of=ProjectVersion)
    )
    return result.scalar_one_or_none()


async def get_project_version_for_project(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    project_version_id: uuid.UUID,
) -> ProjectVersion | None:
    result = await session.execute(
        select(ProjectVersion).where(
            ProjectVersion.project_id == project_id,
            ProjectVersion.id == project_version_id,
        )
    )
    return result.scalar_one_or_none()


async def get_max_project_version_no(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
) -> int:
    result = await session.execute(
        select(func.max(ProjectVersion.version_no)).where(
            ProjectVersion.project_id == project_id
        )
    )
    current_max = result.scalar_one_or_none()
    return int(current_max or 0)


async def create_project_version(
    session: AsyncSession,
    project_version: ProjectVersion,
) -> ProjectVersion:
    session.add(project_version)
    await session.flush()
    return project_version
