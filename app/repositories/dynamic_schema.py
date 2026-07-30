from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload

from app.models.dynamic_schema import DynamicSchema, DynamicSchemaField, DynamicSchemaVersion
from app.models.project import Project
from app.models.project_member import ProjectMember


async def get_project_by_id(
    session: AsyncSession,
    project_id: uuid.UUID,
) -> Project | None:
    result = await session.execute(select(Project).where(Project.id == project_id))
    return result.scalar_one_or_none()


async def get_project_member_for_project(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    user_id: uuid.UUID,
) -> ProjectMember | None:
    result = await session.execute(
        select(ProjectMember).where(
            ProjectMember.project_id == project_id,
            ProjectMember.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()


async def get_dynamic_schema_by_key(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    schema_key: str,
) -> DynamicSchema | None:
    result = await session.execute(
        select(DynamicSchema).where(
            DynamicSchema.project_id == project_id,
            DynamicSchema.schema_key == schema_key,
        )
    )
    return result.scalar_one_or_none()


async def get_dynamic_schema_by_key_for_update(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    schema_key: str,
) -> DynamicSchema | None:
    result = await session.execute(
        select(DynamicSchema)
        .where(
            DynamicSchema.project_id == project_id,
            DynamicSchema.schema_key == schema_key,
        )
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def get_dynamic_schema_by_id_for_update(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    schema_id: uuid.UUID,
) -> DynamicSchema | None:
    result = await session.execute(
        select(DynamicSchema)
        .where(
            DynamicSchema.project_id == project_id,
            DynamicSchema.id == schema_id,
        )
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def create_dynamic_schema(
    session: AsyncSession,
    dynamic_schema: DynamicSchema,
) -> DynamicSchema:
    session.add(dynamic_schema)
    await session.flush()
    return dynamic_schema


async def get_next_dynamic_schema_version_no(
    session: AsyncSession,
    schema_id: uuid.UUID,
) -> int:
    result = await session.execute(
        select(func.max(DynamicSchemaVersion.version_no)).where(
            DynamicSchemaVersion.schema_id == schema_id
        )
    )
    current_max = result.scalar_one_or_none()
    return (current_max or 0) + 1


async def create_dynamic_schema_version(
    session: AsyncSession,
    schema_version: DynamicSchemaVersion,
) -> DynamicSchemaVersion:
    session.add(schema_version)
    await session.flush()
    return schema_version


async def create_dynamic_schema_fields(
    session: AsyncSession,
    fields: list[DynamicSchemaField],
) -> list[DynamicSchemaField]:
    session.add_all(fields)
    await session.flush()
    return fields


async def get_dynamic_schema_version_for_update(
    session: AsyncSession,
    *,
    schema_id: uuid.UUID,
    version_id: uuid.UUID,
) -> DynamicSchemaVersion | None:
    result = await session.execute(
        select(DynamicSchemaVersion)
        .where(
            DynamicSchemaVersion.schema_id == schema_id,
            DynamicSchemaVersion.id == version_id,
        )
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def list_active_dynamic_schema_versions_for_update(
    session: AsyncSession,
    *,
    schema_id: uuid.UUID,
) -> list[DynamicSchemaVersion]:
    result = await session.execute(
        select(DynamicSchemaVersion)
        .where(
            DynamicSchemaVersion.schema_id == schema_id,
            DynamicSchemaVersion.status == "active",
        )
        .with_for_update()
    )
    return list(result.scalars().all())


async def get_dynamic_schema_version_with_fields(
    session: AsyncSession,
    version_id: uuid.UUID,
) -> DynamicSchemaVersion | None:
    result = await session.execute(
        select(DynamicSchemaVersion)
        .where(DynamicSchemaVersion.id == version_id)
        .options(
            joinedload(DynamicSchemaVersion.schema),
            selectinload(DynamicSchemaVersion.fields),
        )
    )
    return result.scalar_one_or_none()
