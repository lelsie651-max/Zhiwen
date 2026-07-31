from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.models.entity import Entity, EntityAlias
from app.models.project import Project
from app.models.project_member import ProjectMember


async def get_project_by_id(session: AsyncSession, project_id: uuid.UUID) -> Project | None:
    return await session.get(Project, project_id)


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


async def get_entity_by_identity_for_update(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    entity_type: str,
    canonical_key: str,
) -> Entity | None:
    result = await session.execute(
        select(Entity)
        .where(
            Entity.project_id == project_id,
            Entity.entity_type == entity_type,
            Entity.canonical_key == canonical_key,
        )
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def get_entity_by_identity_hash_for_update(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    identity_hash: str,
) -> Entity | None:
    result = await session.execute(
        select(Entity)
        .where(
            Entity.project_id == project_id,
            Entity.identity_hash == identity_hash,
        )
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def get_entity_by_id_for_update(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    entity_id: uuid.UUID,
) -> Entity | None:
    result = await session.execute(
        select(Entity)
        .where(
            Entity.project_id == project_id,
            Entity.id == entity_id,
        )
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def get_entity_alias_by_id_for_update(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    alias_id: uuid.UUID,
) -> EntityAlias | None:
    result = await session.execute(
        select(EntityAlias)
        .join(Entity, EntityAlias.entity_id == Entity.id)
        .where(
            Entity.project_id == project_id,
            EntityAlias.id == alias_id,
        )
        .options(joinedload(EntityAlias.entity))
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def create_entity(
    session: AsyncSession,
    entity: Entity,
) -> Entity:
    session.add(entity)
    await session.flush()
    return entity


async def create_entity_alias(
    session: AsyncSession,
    alias: EntityAlias,
) -> EntityAlias:
    session.add(alias)
    await session.flush()
    return alias


async def update_entity_alias(
    session: AsyncSession,
    alias: EntityAlias,
) -> EntityAlias:
    session.add(alias)
    await session.flush()
    return alias


async def resolve_entity_alias(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    entity_type: str,
    normalized_alias: str,
) -> list[EntityAlias]:
    result = await session.execute(
        select(EntityAlias)
        .join(Entity, EntityAlias.entity_id == Entity.id)
        .where(
            Entity.project_id == project_id,
            Entity.entity_type == entity_type,
            EntityAlias.normalized_alias == normalized_alias,
            EntityAlias.status == "active",
        )
        .options(joinedload(EntityAlias.entity))
        .order_by(Entity.created_at.asc(), Entity.id.asc(), EntityAlias.created_at.asc(), EntityAlias.id.asc())
    )
    return list(result.scalars().unique().all())
