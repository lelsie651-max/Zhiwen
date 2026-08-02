from __future__ import annotations

from datetime import datetime, timezone
import uuid

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import set_committed_value

from app.models.dynamic_schema import (
    DynamicSchema,
    DynamicSchemaField,
    DynamicSchemaStatus,
    DynamicSchemaVersion,
    DynamicSchemaVersionSourceKind,
    DynamicSchemaVersionStatus,
)
from app.models.project import ProjectStatus
from app.models.project_member import ProjectMemberRole
from app.models.user import UserStatus
from app.repositories import dynamic_schema as dynamic_schema_repository
from app.repositories import user as user_repository
from app.schemas.dynamic_schema import DynamicSchemaIdentityInput, DynamicSchemaVersionInput
from app.schemas.dynamic_schema_commands import AISchemaProposalInput, HumanSchemaDraftInput


class DynamicSchemaServiceError(Exception):
    """Raised when dynamic schema operations fail."""


class DynamicSchemaPermissionError(DynamicSchemaServiceError):
    """Raised when the actor lacks permission to manage dynamic schemas."""


class DynamicSchemaProjectNotFoundError(DynamicSchemaServiceError):
    """Raised when the target project does not exist or is unavailable."""


class DynamicSchemaIdentityMismatchError(DynamicSchemaServiceError):
    """Raised when a schema key is reused with different identity fields."""


class ArchivedDynamicSchemaError(DynamicSchemaServiceError):
    """Raised when attempting to create a version for an archived schema."""


class DynamicSchemaNotFoundError(DynamicSchemaServiceError):
    """Raised when the target schema cannot be found within the project."""


class DynamicSchemaVersionNotFoundError(DynamicSchemaServiceError):
    """Raised when the target version cannot be found for the schema."""


class DynamicSchemaActivationTransitionError(DynamicSchemaServiceError):
    """Raised when the requested activation transition is not allowed."""


class DynamicSchemaStateCorruptionError(DynamicSchemaServiceError):
    """Raised when schema/version activation state is internally inconsistent."""


def _assign_uuid_if_missing(model: object) -> None:
    if getattr(model, "id", None) is None:
        model.id = uuid.uuid4()


async def create_human_schema_draft(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    actor_id: uuid.UUID,
    payload: HumanSchemaDraftInput,
) -> DynamicSchemaVersion:
    actor = await _require_schema_actor(session, project_id=project_id, actor_id=actor_id)
    schema = await _get_or_create_schema_for_update(
        session,
        project_id=project_id,
        identity=payload.identity,
        created_by_id=actor.id,
    )
    if schema.status == DynamicSchemaStatus.ARCHIVED.value:
        raise ArchivedDynamicSchemaError("Archived schemas cannot accept new versions.")

    version = DynamicSchemaVersion(
        schema_id=schema.id,
        version_no=await dynamic_schema_repository.get_next_dynamic_schema_version_no(session, schema.id),
        status=DynamicSchemaVersionStatus.DRAFT.value,
        source_kind=DynamicSchemaVersionSourceKind.HUMAN.value,
        summary=payload.version.summary,
        layout_config=payload.version.layout_config,
        created_by_id=actor.id,
        activated_by_id=None,
        activated_at=None,
    )
    _assign_uuid_if_missing(version)

    try:
        await dynamic_schema_repository.create_dynamic_schema_version(session, version)
        fields = _build_schema_fields(schema_version_id=version.id, version=payload.version)
        if fields:
            await dynamic_schema_repository.create_dynamic_schema_fields(session, fields)
            set_committed_value(version, "fields", list(fields))
        await session.commit()
    except Exception:
        await session.rollback()
        raise

    return version


async def propose_ai_schema_version(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    payload: AISchemaProposalInput,
) -> DynamicSchemaVersion:
    await _require_project_available(session, project_id=project_id)
    schema = await _get_or_create_schema_for_update(
        session,
        project_id=project_id,
        identity=payload.identity,
        created_by_id=None,
    )
    if schema.status == DynamicSchemaStatus.ARCHIVED.value:
        raise ArchivedDynamicSchemaError("Archived schemas cannot accept new versions.")

    version = DynamicSchemaVersion(
        schema_id=schema.id,
        version_no=await dynamic_schema_repository.get_next_dynamic_schema_version_no(session, schema.id),
        status=DynamicSchemaVersionStatus.PROPOSED.value,
        source_kind=DynamicSchemaVersionSourceKind.AI.value,
        summary=payload.version.summary,
        layout_config=payload.version.layout_config,
        created_by_id=None,
        activated_by_id=None,
        activated_at=None,
    )
    _assign_uuid_if_missing(version)

    try:
        await dynamic_schema_repository.create_dynamic_schema_version(session, version)
        fields = _build_schema_fields(schema_version_id=version.id, version=payload.version)
        if fields:
            await dynamic_schema_repository.create_dynamic_schema_fields(session, fields)
            set_committed_value(version, "fields", list(fields))
        await session.commit()
    except Exception:
        await session.rollback()
        raise

    return version


async def activate_dynamic_schema_version(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    actor_id: uuid.UUID,
    schema_id: uuid.UUID,
    version_id: uuid.UUID,
) -> DynamicSchemaVersion:
    actor = await _require_schema_actor(session, project_id=project_id, actor_id=actor_id)

    schema = await dynamic_schema_repository.get_dynamic_schema_by_id_for_update(
        session,
        project_id=project_id,
        schema_id=schema_id,
    )
    if schema is None:
        raise DynamicSchemaNotFoundError("Schema must belong to the target project.")
    if schema.status != DynamicSchemaStatus.ACTIVE.value:
        raise ArchivedDynamicSchemaError("Archived schemas cannot be activated.")

    target = await dynamic_schema_repository.get_dynamic_schema_version_for_update(
        session,
        schema_id=schema.id,
        version_id=version_id,
    )
    if target is None:
        raise DynamicSchemaVersionNotFoundError("Version must belong to the target schema.")

    active_versions = await dynamic_schema_repository.list_active_dynamic_schema_versions_for_update(
        session,
        schema_id=schema.id,
    )

    current_version = None
    if schema.current_version_id is not None:
        current_version = await dynamic_schema_repository.get_dynamic_schema_version_for_update(
            session,
            schema_id=schema.id,
            version_id=schema.current_version_id,
        )
        if current_version is None:
            raise DynamicSchemaStateCorruptionError(
                "current_version_id points to a missing or foreign schema version."
            )

    _validate_activation_state(
        schema=schema,
        target=target,
        active_versions=active_versions,
        current_version=current_version,
    )

    if target.status == DynamicSchemaVersionStatus.ACTIVE.value:
        return target

    activation_time = datetime.now(timezone.utc)
    target.status = DynamicSchemaVersionStatus.ACTIVE.value
    target.activated_by_id = actor.id
    target.activated_at = activation_time

    if current_version is not None and current_version.id != target.id:
        current_version.status = DynamicSchemaVersionStatus.RETIRED.value

    schema.current_version_id = target.id

    try:
        await session.flush()
        await session.commit()
    except Exception:
        await session.rollback()
        raise

    return target


async def _require_project_available(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
) -> None:
    project = await dynamic_schema_repository.get_project_by_id(session, project_id)
    if project is None or project.status != ProjectStatus.ACTIVE.value:
        raise DynamicSchemaProjectNotFoundError("Project must exist and be active.")


async def _require_schema_actor(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    actor_id: uuid.UUID,
):
    await _require_project_available(session, project_id=project_id)
    actor = await user_repository.get_user_by_id(session, actor_id)
    if actor is None or actor.status != UserStatus.ACTIVE.value:
        raise DynamicSchemaPermissionError("Actor must be an active user.")

    project_member = await dynamic_schema_repository.get_project_member_for_project(
        session,
        project_id=project_id,
        user_id=actor_id,
    )
    if project_member is None:
        raise DynamicSchemaPermissionError("Actor must belong to the target project.")
    if project_member.role not in {
        ProjectMemberRole.OWNER.value,
        ProjectMemberRole.EDITOR.value,
    }:
        raise DynamicSchemaPermissionError("Actor does not have permission to manage dynamic schemas.")

    return actor


async def _get_or_create_schema_for_update(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    identity: DynamicSchemaIdentityInput,
    created_by_id: uuid.UUID | None,
) -> DynamicSchema:
    schema = await dynamic_schema_repository.get_dynamic_schema_by_key_for_update(
        session,
        project_id=project_id,
        schema_key=identity.schema_key,
    )
    if schema is not None:
        _assert_schema_identity_matches(schema, identity)
        return schema

    savepoint = await session.begin_nested()
    try:
        schema = DynamicSchema(
            project_id=project_id,
            schema_key=identity.schema_key,
            name=identity.name,
            subject_kind=identity.subject_kind,
            description=identity.description,
            status=DynamicSchemaStatus.ACTIVE.value,
            current_version_id=None,
            created_by_id=created_by_id,
        )
        _assign_uuid_if_missing(schema)
        await dynamic_schema_repository.create_dynamic_schema(session, schema)
    except IntegrityError:
        await savepoint.rollback()
        schema = await dynamic_schema_repository.get_dynamic_schema_by_key_for_update(
            session,
            project_id=project_id,
            schema_key=identity.schema_key,
        )
        if schema is None:
            raise
        _assert_schema_identity_matches(schema, identity)
    else:
        await savepoint.commit()
    return schema


def _assert_schema_identity_matches(
    schema: DynamicSchema,
    identity: DynamicSchemaIdentityInput,
) -> None:
    if (
        schema.subject_kind != identity.subject_kind
        or schema.name != identity.name
        or schema.description != identity.description
    ):
        raise DynamicSchemaIdentityMismatchError(
            "Existing schema identity fields must exactly match the provided identity."
        )


def _build_schema_fields(
    *,
    schema_version_id: uuid.UUID,
    version: DynamicSchemaVersionInput,
) -> list[DynamicSchemaField]:
    fields = [
        DynamicSchemaField(
            schema_version_id=schema_version_id,
            field_key=field.field_key,
            label=field.label,
            description=field.description,
            predicate_key=field.predicate_key,
            scope_key=field.scope_key,
            expected_value_type=field.expected_value_type.value,
            cardinality=field.cardinality.value,
            is_required=field.is_required,
            is_title=field.is_title,
            is_summary=field.is_summary,
            is_hidden=field.is_hidden,
            group_key=field.group_key,
            display_order=field.display_order,
            display_config=field.display_config,
            validation_rules=field.validation_rules,
        )
        for field in version.fields
    ]
    for field in fields:
        _assign_uuid_if_missing(field)
    return fields


def _validate_activation_state(
    *,
    schema: DynamicSchema,
    target: DynamicSchemaVersion,
    active_versions: list[DynamicSchemaVersion],
    current_version: DynamicSchemaVersion | None,
) -> None:
    if len(active_versions) > 1:
        raise DynamicSchemaStateCorruptionError("A schema cannot have multiple active versions.")

    active_version = active_versions[0] if active_versions else None

    if schema.current_version_id is None:
        if active_version is not None:
            raise DynamicSchemaStateCorruptionError(
                "Schema has an active version while current_version_id is null."
            )
    else:
        if current_version is None:
            raise DynamicSchemaStateCorruptionError("current_version_id could not be resolved.")
        if current_version.schema_id != schema.id:
            raise DynamicSchemaStateCorruptionError("Current version does not belong to the target schema.")
        if current_version.status != DynamicSchemaVersionStatus.ACTIVE.value:
            raise DynamicSchemaStateCorruptionError("Current version must have active status.")
        if active_version is None or active_version.id != current_version.id:
            raise DynamicSchemaStateCorruptionError(
                "Schema current version does not match the stored active version set."
            )

    if target.status == DynamicSchemaVersionStatus.RETIRED.value:
        raise DynamicSchemaActivationTransitionError("Retired versions cannot be reactivated.")
    if target.status == DynamicSchemaVersionStatus.ACTIVE.value:
        if schema.current_version_id != target.id:
            raise DynamicSchemaActivationTransitionError(
                "An active version that is not current cannot be reactivated."
            )
        if active_version is None or active_version.id != target.id:
            raise DynamicSchemaStateCorruptionError(
                "Target version is active/current but active version tracking is inconsistent."
            )
        return

    if target.status not in {
        DynamicSchemaVersionStatus.DRAFT.value,
        DynamicSchemaVersionStatus.PROPOSED.value,
    }:
        raise DynamicSchemaActivationTransitionError("Only draft or proposed versions can be activated.")
