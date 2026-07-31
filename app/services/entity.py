from __future__ import annotations

import hashlib
import json
import uuid

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity import (
    Entity,
    EntityAlias,
    EntityAliasKind,
    EntityAliasStatus,
    EntityStatus,
    normalize_entity_alias,
)
from app.models.project import ProjectStatus
from app.models.project_member import ProjectMemberRole
from app.models.user import UserStatus
from app.repositories import entity as entity_repository
from app.repositories import user as user_repository
from app.schemas.entity import EntityAliasCreateInput, EntityCreateInput
from app.utils.validation import normalize_text


class EntityServiceError(Exception):
    """Raised when entity operations fail."""


class EntityPermissionError(EntityServiceError):
    """Raised when the actor lacks permission to manage entities."""


class EntityProjectNotFoundError(EntityServiceError):
    """Raised when the target project does not exist or is not active."""


class EntityNotFoundError(EntityServiceError):
    """Raised when the target entity cannot be found within the project."""


class EntityAliasNotFoundError(EntityServiceError):
    """Raised when the target alias cannot be found within the project."""


class EntityStateError(EntityServiceError):
    """Raised when the target entity state does not allow the operation."""


class PrimaryEntityAliasRetireError(EntityServiceError):
    """Raised when attempting to retire the active primary alias."""


_ENTITY_UNIQUE_CONSTRAINT_NAMES = {
    "uq_ent_proj_type_key",
    "uq_ent_proj_hash",
}


async def create_entity_with_primary_alias(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    actor_id: uuid.UUID,
    payload: EntityCreateInput,
) -> Entity:
    try:
        actor = await _require_entity_actor(session, project_id=project_id, actor_id=actor_id)
        entity_type = _normalize_entity_type(payload.entity_type)
        canonical_key = normalize_entity_alias(payload.canonical_key)
        display_name = _normalize_display_name(payload.display_name)
        identity_hash = build_entity_identity_hash(
            project_id=project_id,
            entity_type=entity_type,
            canonical_key=canonical_key,
        )

        entity, created = await _get_or_create_entity_for_update(
            session,
            project_id=project_id,
            entity_type=entity_type,
            canonical_key=canonical_key,
            display_name=display_name,
            identity_hash=identity_hash,
            created_by_id=actor.id,
        )
        if not created:
            await session.commit()
            return entity

        primary_alias = EntityAlias(
            entity_id=entity.id,
            alias_text=display_name,
            normalized_alias=normalize_entity_alias(display_name),
            language_code="und",
            alias_kind=EntityAliasKind.CANONICAL.value,
            status=EntityAliasStatus.ACTIVE.value,
            is_primary=True,
            created_by_id=actor.id,
        )
        await entity_repository.create_entity_alias(session, primary_alias)
        entity.aliases.append(primary_alias)
        await session.commit()
        return entity
    except BaseException:
        await session.rollback()
        raise


async def add_entity_alias(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    actor_id: uuid.UUID,
    entity_id: uuid.UUID,
    payload: EntityAliasCreateInput,
) -> EntityAlias:
    try:
        actor = await _require_entity_actor(session, project_id=project_id, actor_id=actor_id)
        entity = await entity_repository.get_entity_by_id_for_update(
            session,
            project_id=project_id,
            entity_id=entity_id,
        )
        if entity is None:
            raise EntityNotFoundError("Entity must belong to the target project.")
        if entity.status != EntityStatus.ACTIVE.value:
            raise EntityStateError("Merged or archived entities cannot accept new aliases.")

        alias = EntityAlias(
            entity_id=entity.id,
            alias_text=_normalize_display_name(payload.alias_text),
            normalized_alias=normalize_entity_alias(payload.alias_text),
            language_code=_normalize_language_code(payload.language_code),
            alias_kind=payload.alias_kind.value,
            status=EntityAliasStatus.ACTIVE.value,
            is_primary=payload.is_primary,
            created_by_id=actor.id,
        )
        await entity_repository.create_entity_alias(session, alias)
        await session.commit()
        return alias
    except BaseException:
        await session.rollback()
        raise


async def retire_entity_alias(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    actor_id: uuid.UUID,
    alias_id: uuid.UUID,
) -> EntityAlias:
    try:
        await _require_entity_actor(session, project_id=project_id, actor_id=actor_id)
        alias = await entity_repository.get_entity_alias_by_id_for_update(
            session,
            project_id=project_id,
            alias_id=alias_id,
        )
        if alias is None:
            raise EntityAliasNotFoundError("Entity alias must belong to the target project.")
        if alias.is_primary and alias.status == EntityAliasStatus.ACTIVE.value:
            raise PrimaryEntityAliasRetireError("The active primary alias cannot be retired directly.")
        if alias.status == EntityAliasStatus.RETIRED.value:
            await session.commit()
            return alias

        alias.status = EntityAliasStatus.RETIRED.value
        alias.is_primary = False
        await entity_repository.update_entity_alias(session, alias)
        await session.commit()
        return alias
    except BaseException:
        await session.rollback()
        raise


async def resolve_entity_alias(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    entity_type: str,
    alias_text: str,
) -> tuple[EntityAlias, ...]:
    await _require_project_available(session, project_id=project_id)
    normalized_entity_type = _normalize_entity_type(entity_type)
    normalized_alias = normalize_entity_alias(alias_text)
    matches = await entity_repository.resolve_entity_alias(
        session,
        project_id=project_id,
        entity_type=normalized_entity_type,
        normalized_alias=normalized_alias,
    )
    return tuple(matches)


def build_entity_identity_hash(
    *,
    project_id: uuid.UUID,
    entity_type: str,
    canonical_key: str,
) -> str:
    payload = {
        "canonical_key": canonical_key,
        "entity_type": entity_type,
        "project_id": str(project_id),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


async def _require_project_available(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
) -> None:
    project = await entity_repository.get_project_by_id(session, project_id)
    if project is None or project.status != ProjectStatus.ACTIVE.value:
        raise EntityProjectNotFoundError("Project must exist and be active.")


async def _require_entity_actor(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    actor_id: uuid.UUID,
):
    await _require_project_available(session, project_id=project_id)
    actor = await user_repository.get_user_by_id(session, actor_id)
    if actor is None or actor.status != UserStatus.ACTIVE.value:
        raise EntityPermissionError("Actor must be an active user.")

    project_member = await entity_repository.get_project_member_for_project(
        session,
        project_id=project_id,
        user_id=actor_id,
    )
    if project_member is None:
        raise EntityPermissionError("Actor must belong to the target project.")
    if project_member.role not in {
        ProjectMemberRole.OWNER.value,
        ProjectMemberRole.EDITOR.value,
    }:
        raise EntityPermissionError("Actor does not have permission to manage entities.")

    return actor


async def _get_or_create_entity_for_update(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    entity_type: str,
    canonical_key: str,
    display_name: str,
    identity_hash: str,
    created_by_id: uuid.UUID | None,
) -> tuple[Entity, bool]:
    entity = await entity_repository.get_entity_by_identity_for_update(
        session,
        project_id=project_id,
        entity_type=entity_type,
        canonical_key=canonical_key,
    )
    if entity is not None:
        return entity, False

    savepoint = await session.begin_nested()
    try:
        entity = Entity(
            project_id=project_id,
            entity_type=entity_type,
            canonical_key=canonical_key,
            display_name=display_name,
            identity_hash=identity_hash,
            status=EntityStatus.ACTIVE.value,
            merged_into_entity_id=None,
            created_by_id=created_by_id,
        )
        await entity_repository.create_entity(session, entity)
    except IntegrityError as exc:
        if not _is_entity_identity_integrity_error(exc):
            await savepoint.rollback()
            raise
        await savepoint.rollback()
        entity = await entity_repository.get_entity_by_identity_for_update(
            session,
            project_id=project_id,
            entity_type=entity_type,
            canonical_key=canonical_key,
        )
        if entity is None:
            entity = await entity_repository.get_entity_by_identity_hash_for_update(
                session,
                project_id=project_id,
                identity_hash=identity_hash,
            )
        if entity is None:
            raise
        return entity, False
    else:
        await savepoint.commit()
        return entity, True


def _is_entity_identity_integrity_error(exc: IntegrityError) -> bool:
    message = str(exc.orig or exc).lower()
    return any(name in message for name in _ENTITY_UNIQUE_CONSTRAINT_NAMES)


def _normalize_entity_type(value: str) -> str:
    normalized = normalize_text(value)
    if not normalized:
        raise ValueError("entity_type must not be empty")
    if len(normalized) > 64:
        raise ValueError("entity_type must be at most 64 characters")
    return normalized


def _normalize_display_name(value: str) -> str:
    normalized = normalize_text(value)
    if not normalized:
        raise ValueError("display_name must not be empty")
    if len(normalized) > 255:
        raise ValueError("display_name must be at most 255 characters")
    return normalized


def _normalize_language_code(value: str) -> str:
    normalized = normalize_text(value)
    if not normalized:
        raise ValueError("language_code must not be empty")
    if len(normalized) > 32:
        raise ValueError("language_code must be at most 32 characters")
    return normalized
