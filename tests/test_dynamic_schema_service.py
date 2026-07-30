import asyncio
import inspect
from datetime import datetime, timezone
import uuid

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import joinedload as sa_joinedload
from sqlalchemy.orm import selectinload as sa_selectinload

from app.models.dynamic_schema import DynamicSchema, DynamicSchemaField, DynamicSchemaVersion
from app.models.project import Project
from app.models.project_member import ProjectMember
from app.models.user import User
from app.schemas.dynamic_schema import (
    DynamicSchemaFieldInput,
    DynamicSchemaIdentityInput,
    DynamicSchemaVersionInput,
)
from app.schemas.dynamic_schema_commands import AISchemaProposalInput, HumanSchemaDraftInput
from app.services import dynamic_schema as dynamic_schema_service


def run_async(awaitable):
    return asyncio.run(awaitable)


class FakeSavepoint:
    def __init__(self) -> None:
        self.commit_called = False
        self.rollback_called = False

    async def commit(self) -> None:
        self.commit_called = True

    async def rollback(self) -> None:
        self.rollback_called = True


class FakeSession:
    def __init__(self) -> None:
        self.commit_called = False
        self.rollback_called = False
        self.flush_called = False
        self.savepoints: list[FakeSavepoint] = []

    async def commit(self) -> None:
        self.commit_called = True

    async def rollback(self) -> None:
        self.rollback_called = True

    async def flush(self) -> None:
        self.flush_called = True

    async def begin_nested(self) -> FakeSavepoint:
        savepoint = FakeSavepoint()
        self.savepoints.append(savepoint)
        return savepoint


def build_project(*, project_id: uuid.UUID | None = None, status: str = "active") -> Project:
    actual_project_id = project_id or uuid.uuid4()
    return Project(
        id=actual_project_id,
        name="Schema Project",
        slug=f"schema-{str(actual_project_id)[:8]}",
        created_by_id=uuid.uuid4(),
        status=status,
    )


def build_user(*, user_id: uuid.UUID | None = None, status: str = "active") -> User:
    actual_user_id = user_id or uuid.uuid4()
    return User(
        id=actual_user_id,
        handle=f"user-{str(actual_user_id)[:8]}",
        display_name="Schema Actor",
        status=status,
    )


def build_project_member(*, project_id: uuid.UUID, user_id: uuid.UUID, role: str) -> ProjectMember:
    return ProjectMember(
        id=uuid.uuid4(),
        project_id=project_id,
        user_id=user_id,
        role=role,
    )


def build_identity(
    *,
    schema_key: str = "profile.main",
    name: str = "Profile Main",
    subject_kind: str = "person",
    description: str | None = "Main person profile",
) -> DynamicSchemaIdentityInput:
    return DynamicSchemaIdentityInput(
        schema_key=schema_key,
        name=name,
        subject_kind=subject_kind,
        description=description,
    )


def build_version_input(
    *,
    summary: str | None = "Profile layout",
    layout_config: dict | None = None,
) -> DynamicSchemaVersionInput:
    return DynamicSchemaVersionInput(
        summary=summary,
        layout_config=layout_config or {"sections": ["main", "meta"]},
        fields=[
            DynamicSchemaFieldInput(
                field_key="title",
                label="Title",
                predicate_key="person.name",
                display_order=0,
                is_title=True,
                display_config={"width": "full"},
                validation_rules={"max_length": 100},
            ),
            DynamicSchemaFieldInput(
                field_key="summary",
                label="Summary",
                predicate_key="person.summary",
                display_order=1,
                is_summary=True,
                group_key="meta",
                display_config={"multiline": True},
                validation_rules={},
            ),
        ],
    )


def build_human_payload(
    *,
    identity: DynamicSchemaIdentityInput | None = None,
    version: DynamicSchemaVersionInput | None = None,
) -> HumanSchemaDraftInput:
    return HumanSchemaDraftInput(
        identity=identity or build_identity(),
        version=version or build_version_input(),
    )


def build_ai_payload(
    *,
    identity: DynamicSchemaIdentityInput | None = None,
    version: DynamicSchemaVersionInput | None = None,
) -> AISchemaProposalInput:
    return AISchemaProposalInput(
        identity=identity or build_identity(),
        version=version or build_version_input(),
    )


def build_dynamic_schema(
    *,
    project_id: uuid.UUID,
    identity: DynamicSchemaIdentityInput | None = None,
    status: str = "active",
    current_version_id: uuid.UUID | None = None,
    created_by_id: uuid.UUID | None = None,
) -> DynamicSchema:
    actual_identity = identity or build_identity()
    return DynamicSchema(
        id=uuid.uuid4(),
        project_id=project_id,
        schema_key=actual_identity.schema_key,
        name=actual_identity.name,
        subject_kind=actual_identity.subject_kind,
        description=actual_identity.description,
        status=status,
        current_version_id=current_version_id,
        created_by_id=created_by_id,
    )


def build_dynamic_schema_version(
    *,
    schema_id: uuid.UUID,
    version_id: uuid.UUID | None = None,
    version_no: int = 1,
    status: str = "draft",
    source_kind: str = "human",
    created_by_id: uuid.UUID | None = None,
    activated_by_id: uuid.UUID | None = None,
    activated_at=None,
) -> DynamicSchemaVersion:
    return DynamicSchemaVersion(
        id=version_id or uuid.uuid4(),
        schema_id=schema_id,
        version_no=version_no,
        status=status,
        source_kind=source_kind,
        summary="Version summary",
        layout_config={"sections": ["main"]},
        created_by_id=created_by_id,
        activated_by_id=activated_by_id,
        activated_at=activated_at,
    )


class StubSchemaWithoutRelations:
    def __init__(
        self,
        *,
        project_id: uuid.UUID,
        identity: DynamicSchemaIdentityInput | None = None,
        status: str = "active",
        current_version_id: uuid.UUID | None = None,
        created_by_id: uuid.UUID | None = None,
    ) -> None:
        actual_identity = identity or build_identity()
        self.id = uuid.uuid4()
        self.project_id = project_id
        self.schema_key = actual_identity.schema_key
        self.name = actual_identity.name
        self.subject_kind = actual_identity.subject_kind
        self.description = actual_identity.description
        self.status = status
        self.current_version_id = current_version_id
        self.created_by_id = created_by_id

    @property
    def versions(self):
        raise AssertionError("schema.versions must not be accessed")

    @property
    def current_version(self):
        raise AssertionError("schema.current_version must not be accessed")


def patch_project_and_actor(
    monkeypatch,
    *,
    project: Project,
    actor: User | None = None,
    role: str | None = None,
) -> None:
    async def fake_get_project_by_id(_session, project_id):
        if project_id == project.id:
            return project
        return None

    async def fake_get_user_by_id(_session, user_id):
        if actor is not None and user_id == actor.id:
            return actor
        return None

    async def fake_get_project_member_for_project(_session, *, project_id, user_id):
        if actor is None or role is None:
            return None
        if project_id == project.id and user_id == actor.id:
            return build_project_member(project_id=project.id, user_id=actor.id, role=role)
        return None

    monkeypatch.setattr(
        dynamic_schema_service.dynamic_schema_repository,
        "get_project_by_id",
        fake_get_project_by_id,
    )
    monkeypatch.setattr(
        dynamic_schema_service.user_repository,
        "get_user_by_id",
        fake_get_user_by_id,
    )
    monkeypatch.setattr(
        dynamic_schema_service.dynamic_schema_repository,
        "get_project_member_for_project",
        fake_get_project_member_for_project,
    )


def patch_schema_creation_dependencies(
    monkeypatch,
    *,
    existing_schema: DynamicSchema | None = None,
    next_version_no: int = 1,
    create_fields_error: Exception | None = None,
):
    captured: dict[str, object] = {}

    async def fake_get_dynamic_schema_by_key_for_update(_session, **_kwargs):
        return existing_schema

    async def fake_create_dynamic_schema(_session, schema):
        captured["schema"] = schema
        return schema

    async def fake_get_next_dynamic_schema_version_no(_session, _schema_id):
        captured["next_version_schema_id"] = _schema_id
        return next_version_no

    async def fake_create_dynamic_schema_version(_session, version):
        captured["version"] = version
        return version

    async def fake_create_dynamic_schema_fields(_session, fields):
        if create_fields_error is not None:
            raise create_fields_error
        captured["fields"] = fields
        return fields

    monkeypatch.setattr(
        dynamic_schema_service.dynamic_schema_repository,
        "get_dynamic_schema_by_key_for_update",
        fake_get_dynamic_schema_by_key_for_update,
    )
    monkeypatch.setattr(
        dynamic_schema_service.dynamic_schema_repository,
        "create_dynamic_schema",
        fake_create_dynamic_schema,
    )
    monkeypatch.setattr(
        dynamic_schema_service.dynamic_schema_repository,
        "get_next_dynamic_schema_version_no",
        fake_get_next_dynamic_schema_version_no,
    )
    monkeypatch.setattr(
        dynamic_schema_service.dynamic_schema_repository,
        "create_dynamic_schema_version",
        fake_create_dynamic_schema_version,
    )
    monkeypatch.setattr(
        dynamic_schema_service.dynamic_schema_repository,
        "create_dynamic_schema_fields",
        fake_create_dynamic_schema_fields,
    )
    return captured


def patch_activation_dependencies(
    monkeypatch,
    *,
    schema,
    target_version: DynamicSchemaVersion | None,
    active_versions: list[DynamicSchemaVersion],
    current_version: DynamicSchemaVersion | None = None,
    flush_error: Exception | None = None,
):
    call_order: list[str] = []

    async def fake_get_dynamic_schema_by_id_for_update(_session, *, project_id, schema_id):
        call_order.append("lock_schema")
        if project_id == schema.project_id and schema_id == schema.id:
            return schema
        return None

    async def fake_get_dynamic_schema_version_for_update(_session, *, schema_id, version_id):
        if version_id == target_version.id if target_version is not None else False:
            call_order.append("lock_target_version")
            return target_version
        if current_version is not None and version_id == current_version.id:
            call_order.append("load_current_version")
            return current_version
        return None

    async def fake_list_active_dynamic_schema_versions_for_update(_session, *, schema_id):
        call_order.append("lock_active_versions")
        if schema_id == schema.id:
            return active_versions
        return []

    monkeypatch.setattr(
        dynamic_schema_service.dynamic_schema_repository,
        "get_dynamic_schema_by_id_for_update",
        fake_get_dynamic_schema_by_id_for_update,
    )
    monkeypatch.setattr(
        dynamic_schema_service.dynamic_schema_repository,
        "get_dynamic_schema_version_for_update",
        fake_get_dynamic_schema_version_for_update,
    )
    monkeypatch.setattr(
        dynamic_schema_service.dynamic_schema_repository,
        "list_active_dynamic_schema_versions_for_update",
        fake_list_active_dynamic_schema_versions_for_update,
    )

    return call_order


def test_owner_and_editor_can_create_human_schema_draft(monkeypatch, role: str = "owner") -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    payload = build_human_payload()

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role=role)
    captured = patch_schema_creation_dependencies(monkeypatch)

    result = run_async(
        dynamic_schema_service.create_human_schema_draft(
            session,
            project_id=project.id,
            actor_id=actor.id,
            payload=payload,
        )
    )

    assert result.status == "draft"
    assert result.source_kind == "human"
    assert result.created_by_id == actor.id
    assert result.activated_by_id is None
    assert result.activated_at is None
    assert captured["version"].version_no == 1


@pytest.mark.parametrize("role", ["owner", "editor"])
def test_owner_editor_roles_both_work_for_human_draft(monkeypatch, role: str) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role=role)
    patch_schema_creation_dependencies(monkeypatch)

    result = run_async(
        dynamic_schema_service.create_human_schema_draft(
            session,
            project_id=project.id,
            actor_id=actor.id,
            payload=build_human_payload(),
        )
    )

    assert result.status == "draft"


def test_viewer_cannot_create_human_schema_draft(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="viewer")
    patch_schema_creation_dependencies(monkeypatch)

    with pytest.raises(dynamic_schema_service.DynamicSchemaPermissionError):
        run_async(
            dynamic_schema_service.create_human_schema_draft(
                session,
                project_id=project.id,
                actor_id=actor.id,
                payload=build_human_payload(),
            )
        )


def test_human_draft_creates_new_schema_identity(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    payload = build_human_payload()

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")
    captured = patch_schema_creation_dependencies(monkeypatch)

    result = run_async(
        dynamic_schema_service.create_human_schema_draft(
            session,
            project_id=project.id,
            actor_id=actor.id,
            payload=payload,
        )
    )

    created_schema = captured["schema"]
    assert created_schema.project_id == project.id
    assert created_schema.schema_key == payload.identity.schema_key
    assert created_schema.created_by_id == actor.id
    assert created_schema.current_version_id is None
    assert result.schema_id == created_schema.id


def test_existing_schema_is_reused(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    existing_schema = build_dynamic_schema(project_id=project.id, created_by_id=actor.id)

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")
    captured = patch_schema_creation_dependencies(
        monkeypatch,
        existing_schema=existing_schema,
        next_version_no=3,
    )

    result = run_async(
        dynamic_schema_service.create_human_schema_draft(
            session,
            project_id=project.id,
            actor_id=actor.id,
            payload=build_human_payload(),
        )
    )

    assert "schema" not in captured
    assert result.schema_id == existing_schema.id
    assert result.version_no == 3


def test_identity_mismatch_is_rejected(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    existing_schema = build_dynamic_schema(project_id=project.id)
    mismatched_identity = build_identity(name="Different Name")

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")
    patch_schema_creation_dependencies(monkeypatch, existing_schema=existing_schema)

    with pytest.raises(dynamic_schema_service.DynamicSchemaIdentityMismatchError):
        run_async(
            dynamic_schema_service.create_human_schema_draft(
                session,
                project_id=project.id,
                actor_id=actor.id,
                payload=build_human_payload(identity=mismatched_identity),
            )
        )


def test_archived_schema_rejects_new_versions(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    archived_schema = build_dynamic_schema(project_id=project.id, status="archived")

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")
    patch_schema_creation_dependencies(monkeypatch, existing_schema=archived_schema)

    with pytest.raises(dynamic_schema_service.ArchivedDynamicSchemaError):
        run_async(
            dynamic_schema_service.create_human_schema_draft(
                session,
                project_id=project.id,
                actor_id=actor.id,
                payload=build_human_payload(),
            )
        )


def test_version_number_is_computed_under_schema_lock(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    existing_schema = build_dynamic_schema(project_id=project.id)
    call_order: list[str] = []

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")

    async def fake_get_dynamic_schema_by_key_for_update(_session, **_kwargs):
        call_order.append("lock_schema")
        return existing_schema

    async def fake_get_next_dynamic_schema_version_no(_session, _schema_id):
        call_order.append("next_version")
        return 5

    async def fake_create_dynamic_schema_version(_session, version):
        return version

    async def fake_create_dynamic_schema_fields(_session, fields):
        return fields

    monkeypatch.setattr(
        dynamic_schema_service.dynamic_schema_repository,
        "get_dynamic_schema_by_key_for_update",
        fake_get_dynamic_schema_by_key_for_update,
    )
    monkeypatch.setattr(
        dynamic_schema_service.dynamic_schema_repository,
        "get_next_dynamic_schema_version_no",
        fake_get_next_dynamic_schema_version_no,
    )
    monkeypatch.setattr(
        dynamic_schema_service.dynamic_schema_repository,
        "create_dynamic_schema_version",
        fake_create_dynamic_schema_version,
    )
    monkeypatch.setattr(
        dynamic_schema_service.dynamic_schema_repository,
        "create_dynamic_schema_fields",
        fake_create_dynamic_schema_fields,
    )

    result = run_async(
        dynamic_schema_service.create_human_schema_draft(
            session,
            project_id=project.id,
            actor_id=actor.id,
            payload=build_human_payload(),
        )
    )

    assert result.version_no == 5
    assert call_order == ["lock_schema", "next_version"]


def test_fields_are_fully_copied_and_ordered(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    payload = build_human_payload()

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")
    captured = patch_schema_creation_dependencies(monkeypatch)

    result = run_async(
        dynamic_schema_service.create_human_schema_draft(
            session,
            project_id=project.id,
            actor_id=actor.id,
            payload=payload,
        )
    )

    fields = captured["fields"]
    assert [field.field_key for field in fields] == ["title", "summary"]
    assert [field.display_order for field in fields] == [0, 1]
    assert fields[0].predicate_key == "person.name"
    assert fields[1].group_key == "meta"
    assert result.fields[0].field_key == "title"


def test_current_version_id_remains_unchanged(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    current_version_id = uuid.uuid4()
    existing_schema = build_dynamic_schema(
        project_id=project.id,
        current_version_id=current_version_id,
        created_by_id=actor.id,
    )

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")
    patch_schema_creation_dependencies(monkeypatch, existing_schema=existing_schema, next_version_no=2)

    run_async(
        dynamic_schema_service.create_human_schema_draft(
            session,
            project_id=project.id,
            actor_id=actor.id,
            payload=build_human_payload(),
        )
    )

    assert existing_schema.current_version_id == current_version_id


def test_ai_schema_version_is_always_proposed_without_actor(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    payload = build_ai_payload()

    async def fake_get_project_by_id(_session, project_id):
        if project_id == project.id:
            return project
        return None

    monkeypatch.setattr(
        dynamic_schema_service.dynamic_schema_repository,
        "get_project_by_id",
        fake_get_project_by_id,
    )
    captured = patch_schema_creation_dependencies(monkeypatch)

    result = run_async(
        dynamic_schema_service.propose_ai_schema_version(
            session,
            project_id=project.id,
            payload=payload,
        )
    )

    assert result.status == "proposed"
    assert result.source_kind == "ai"
    assert result.created_by_id is None
    assert result.activated_by_id is None
    assert result.activated_at is None
    assert captured["schema"].created_by_id is None


def test_concurrent_create_reuses_same_schema(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    existing_schema = build_dynamic_schema(project_id=project.id)
    lookup_count = {"count": 0}

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")

    async def fake_get_dynamic_schema_by_key_for_update(_session, **_kwargs):
        lookup_count["count"] += 1
        if lookup_count["count"] == 1:
            return None
        return existing_schema

    async def fake_create_dynamic_schema(_session, _schema):
        raise IntegrityError("insert", {}, Exception("duplicate key"))

    async def fake_get_next_dynamic_schema_version_no(_session, _schema_id):
        return 2

    async def fake_create_dynamic_schema_version(_session, version):
        return version

    async def fake_create_dynamic_schema_fields(_session, fields):
        return fields

    monkeypatch.setattr(
        dynamic_schema_service.dynamic_schema_repository,
        "get_dynamic_schema_by_key_for_update",
        fake_get_dynamic_schema_by_key_for_update,
    )
    monkeypatch.setattr(
        dynamic_schema_service.dynamic_schema_repository,
        "create_dynamic_schema",
        fake_create_dynamic_schema,
    )
    monkeypatch.setattr(
        dynamic_schema_service.dynamic_schema_repository,
        "get_next_dynamic_schema_version_no",
        fake_get_next_dynamic_schema_version_no,
    )
    monkeypatch.setattr(
        dynamic_schema_service.dynamic_schema_repository,
        "create_dynamic_schema_version",
        fake_create_dynamic_schema_version,
    )
    monkeypatch.setattr(
        dynamic_schema_service.dynamic_schema_repository,
        "create_dynamic_schema_fields",
        fake_create_dynamic_schema_fields,
    )

    result = run_async(
        dynamic_schema_service.create_human_schema_draft(
            session,
            project_id=project.id,
            actor_id=actor.id,
            payload=build_human_payload(),
        )
    )

    assert result.schema_id == existing_schema.id
    assert session.rollback_called is False
    assert session.savepoints[0].rollback_called is True


def test_failure_rolls_back_whole_transaction(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")
    patch_schema_creation_dependencies(
        monkeypatch,
        create_fields_error=RuntimeError("field insert failed"),
    )

    with pytest.raises(RuntimeError):
        run_async(
            dynamic_schema_service.create_human_schema_draft(
                session,
                project_id=project.id,
                actor_id=actor.id,
                payload=build_human_payload(),
            )
        )

    assert session.rollback_called is True
    assert session.commit_called is False


def test_archived_project_rejects_creation(monkeypatch) -> None:
    session = FakeSession()
    project = build_project(status="archived")
    actor = build_user()

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")
    patch_schema_creation_dependencies(monkeypatch)

    with pytest.raises(dynamic_schema_service.DynamicSchemaProjectNotFoundError):
        run_async(
            dynamic_schema_service.create_human_schema_draft(
                session,
                project_id=project.id,
                actor_id=actor.id,
                payload=build_human_payload(),
            )
        )


def test_service_does_not_write_fact_models(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")
    captured = patch_schema_creation_dependencies(monkeypatch)

    result = run_async(
        dynamic_schema_service.create_human_schema_draft(
            session,
            project_id=project.id,
            actor_id=actor.id,
            payload=build_human_payload(),
        )
    )

    assert isinstance(captured["schema"], DynamicSchema)
    assert isinstance(result, DynamicSchemaVersion)
    assert all(isinstance(field, DynamicSchemaField) for field in result.fields)


def test_human_draft_reuses_existing_schema_without_accessing_versions(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    existing_schema = StubSchemaWithoutRelations(project_id=project.id, created_by_id=actor.id)

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")
    patch_schema_creation_dependencies(monkeypatch, existing_schema=existing_schema, next_version_no=2)

    result = run_async(
        dynamic_schema_service.create_human_schema_draft(
            session,
            project_id=project.id,
            actor_id=actor.id,
            payload=build_human_payload(),
        )
    )

    assert result.schema_id == existing_schema.id
    assert [field.field_key for field in result.fields] == ["title", "summary"]


def test_ai_proposed_reuses_existing_schema_without_accessing_versions(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    existing_schema = StubSchemaWithoutRelations(project_id=project.id)

    async def fake_get_project_by_id(_session, project_id):
        if project_id == project.id:
            return project
        return None

    monkeypatch.setattr(
        dynamic_schema_service.dynamic_schema_repository,
        "get_project_by_id",
        fake_get_project_by_id,
    )
    patch_schema_creation_dependencies(monkeypatch, existing_schema=existing_schema, next_version_no=4)

    result = run_async(
        dynamic_schema_service.propose_ai_schema_version(
            session,
            project_id=project.id,
            payload=build_ai_payload(),
        )
    )

    assert result.schema_id == existing_schema.id
    assert result.status == "proposed"
    assert [field.field_key for field in result.fields] == ["title", "summary"]


def test_get_dynamic_schema_version_with_fields_uses_joinedload_for_schema_and_selectinload_for_fields(
    monkeypatch,
) -> None:
    from app.repositories import dynamic_schema as dynamic_schema_repository

    recorded: dict[str, object] = {}

    class FakeResult:
        def scalar_one_or_none(self):
            return "loaded-version"

    class FakeSessionForRepo:
        async def execute(self, statement):
            recorded["statement"] = statement
            return FakeResult()

    def tracked_joinedload(attribute):
        recorded["joinedload_attr"] = attribute
        return sa_joinedload(attribute)

    def tracked_selectinload(attribute):
        recorded["selectinload_attr"] = attribute
        return sa_selectinload(attribute)

    monkeypatch.setattr(dynamic_schema_repository, "joinedload", tracked_joinedload)
    monkeypatch.setattr(dynamic_schema_repository, "selectinload", tracked_selectinload)

    result = run_async(
        dynamic_schema_repository.get_dynamic_schema_version_with_fields(
            FakeSessionForRepo(),
            uuid.uuid4(),
        )
    )

    assert result == "loaded-version"
    assert recorded["joinedload_attr"] is DynamicSchemaVersion.schema
    assert recorded["selectinload_attr"] is DynamicSchemaVersion.fields
    source = inspect.getsource(dynamic_schema_repository.get_dynamic_schema_version_with_fields)
    assert "joinedload(DynamicSchemaVersion.fields)" not in source
    assert "selectinload(DynamicSchemaVersion.fields)" in source


@pytest.mark.parametrize("role", ["owner", "editor"])
def test_owner_and_editor_can_activate_schema_version(monkeypatch, role: str) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    schema = StubSchemaWithoutRelations(project_id=project.id)
    target = build_dynamic_schema_version(schema_id=schema.id, status="draft", version_no=2)

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role=role)
    patch_activation_dependencies(
        monkeypatch,
        schema=schema,
        target_version=target,
        active_versions=[],
    )

    result = run_async(
        dynamic_schema_service.activate_dynamic_schema_version(
            session,
            project_id=project.id,
            actor_id=actor.id,
            schema_id=schema.id,
            version_id=target.id,
        )
    )

    assert result.status == "active"
    assert result.activated_by_id == actor.id
    assert result.activated_at.tzinfo == timezone.utc
    assert schema.current_version_id == target.id
    assert session.flush_called is True
    assert session.commit_called is True


def test_viewer_cannot_activate_schema_version(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    schema = StubSchemaWithoutRelations(project_id=project.id)
    target = build_dynamic_schema_version(schema_id=schema.id, status="draft")

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="viewer")
    patch_activation_dependencies(
        monkeypatch,
        schema=schema,
        target_version=target,
        active_versions=[],
    )

    with pytest.raises(dynamic_schema_service.DynamicSchemaPermissionError):
        run_async(
            dynamic_schema_service.activate_dynamic_schema_version(
                session,
                project_id=project.id,
                actor_id=actor.id,
                schema_id=schema.id,
                version_id=target.id,
            )
        )


def test_proposed_version_can_be_activated_by_human(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    schema = StubSchemaWithoutRelations(project_id=project.id)
    target = build_dynamic_schema_version(schema_id=schema.id, status="proposed", source_kind="ai")

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")
    patch_activation_dependencies(
        monkeypatch,
        schema=schema,
        target_version=target,
        active_versions=[],
    )

    result = run_async(
        dynamic_schema_service.activate_dynamic_schema_version(
            session,
            project_id=project.id,
            actor_id=actor.id,
            schema_id=schema.id,
            version_id=target.id,
        )
    )

    assert result.status == "active"
    assert result.source_kind == "ai"
    assert result.activated_by_id == actor.id


def test_activation_without_old_current_sets_current_directly(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    schema = StubSchemaWithoutRelations(project_id=project.id)
    target = build_dynamic_schema_version(schema_id=schema.id, status="draft")

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")
    patch_activation_dependencies(
        monkeypatch,
        schema=schema,
        target_version=target,
        active_versions=[],
    )

    run_async(
        dynamic_schema_service.activate_dynamic_schema_version(
            session,
            project_id=project.id,
            actor_id=actor.id,
            schema_id=schema.id,
            version_id=target.id,
        )
    )

    assert schema.current_version_id == target.id


def test_old_active_version_becomes_retired_and_keeps_activation_metadata(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    old_actor_id = uuid.uuid4()
    old_time = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)
    old_current = build_dynamic_schema_version(
        schema_id=uuid.uuid4(),
        status="active",
        version_no=1,
        activated_by_id=old_actor_id,
        activated_at=old_time,
    )
    schema = StubSchemaWithoutRelations(
        project_id=project.id,
        current_version_id=old_current.id,
    )
    old_current.schema_id = schema.id
    target = build_dynamic_schema_version(schema_id=schema.id, status="draft", version_no=2)

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")
    patch_activation_dependencies(
        monkeypatch,
        schema=schema,
        target_version=target,
        active_versions=[old_current],
        current_version=old_current,
    )

    result = run_async(
        dynamic_schema_service.activate_dynamic_schema_version(
            session,
            project_id=project.id,
            actor_id=actor.id,
            schema_id=schema.id,
            version_id=target.id,
        )
    )

    assert old_current.status == "retired"
    assert old_current.activated_by_id == old_actor_id
    assert old_current.activated_at == old_time
    assert result.activated_by_id == actor.id
    assert result.activated_at.tzinfo == timezone.utc
    assert schema.current_version_id == target.id


def test_already_current_active_version_is_idempotent(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    original_time = datetime(2026, 7, 30, 8, 0, tzinfo=timezone.utc)
    target = build_dynamic_schema_version(
        schema_id=uuid.uuid4(),
        status="active",
        version_no=1,
        activated_by_id=uuid.uuid4(),
        activated_at=original_time,
    )
    schema = StubSchemaWithoutRelations(project_id=project.id, current_version_id=target.id)
    target.schema_id = schema.id

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")
    patch_activation_dependencies(
        monkeypatch,
        schema=schema,
        target_version=target,
        active_versions=[target],
        current_version=target,
    )

    result = run_async(
        dynamic_schema_service.activate_dynamic_schema_version(
            session,
            project_id=project.id,
            actor_id=actor.id,
            schema_id=schema.id,
            version_id=target.id,
        )
    )

    assert result is target
    assert result.activated_at == original_time
    assert session.flush_called is False
    assert session.commit_called is False


def test_retired_target_is_rejected(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    schema = StubSchemaWithoutRelations(project_id=project.id)
    target = build_dynamic_schema_version(schema_id=schema.id, status="retired", version_no=2)

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")
    patch_activation_dependencies(monkeypatch, schema=schema, target_version=target, active_versions=[])

    with pytest.raises(dynamic_schema_service.DynamicSchemaActivationTransitionError):
        run_async(
            dynamic_schema_service.activate_dynamic_schema_version(
                session,
                project_id=project.id,
                actor_id=actor.id,
                schema_id=schema.id,
                version_id=target.id,
            )
        )


def test_active_but_noncurrent_target_is_rejected(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    current = build_dynamic_schema_version(
        schema_id=uuid.uuid4(),
        status="active",
        activated_by_id=uuid.uuid4(),
        activated_at=datetime(2026, 7, 30, 7, 0, tzinfo=timezone.utc),
    )
    schema = StubSchemaWithoutRelations(project_id=project.id, current_version_id=current.id)
    current.schema_id = schema.id
    target = build_dynamic_schema_version(
        schema_id=schema.id,
        status="active",
        version_no=2,
        activated_by_id=uuid.uuid4(),
        activated_at=datetime(2026, 7, 30, 9, 0, tzinfo=timezone.utc),
    )

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")
    patch_activation_dependencies(
        monkeypatch,
        schema=schema,
        target_version=target,
        active_versions=[current],
        current_version=current,
    )

    with pytest.raises(dynamic_schema_service.DynamicSchemaActivationTransitionError):
        run_async(
            dynamic_schema_service.activate_dynamic_schema_version(
                session,
                project_id=project.id,
                actor_id=actor.id,
                schema_id=schema.id,
                version_id=target.id,
            )
        )


def test_current_missing_or_foreign_is_rejected(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    schema = StubSchemaWithoutRelations(project_id=project.id, current_version_id=uuid.uuid4())
    target = build_dynamic_schema_version(schema_id=schema.id, status="draft")

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")
    patch_activation_dependencies(
        monkeypatch,
        schema=schema,
        target_version=target,
        active_versions=[],
        current_version=None,
    )

    with pytest.raises(dynamic_schema_service.DynamicSchemaStateCorruptionError):
        run_async(
            dynamic_schema_service.activate_dynamic_schema_version(
                session,
                project_id=project.id,
                actor_id=actor.id,
                schema_id=schema.id,
                version_id=target.id,
            )
        )


def test_current_non_active_or_multiple_active_versions_are_rejected(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    current = build_dynamic_schema_version(schema_id=uuid.uuid4(), status="draft", version_no=1)
    schema = StubSchemaWithoutRelations(project_id=project.id, current_version_id=current.id)
    current.schema_id = schema.id
    other_active = build_dynamic_schema_version(
        schema_id=schema.id,
        status="active",
        version_no=2,
        activated_by_id=uuid.uuid4(),
        activated_at=datetime(2026, 7, 30, 11, 0, tzinfo=timezone.utc),
    )
    target = build_dynamic_schema_version(schema_id=schema.id, status="proposed", version_no=3)

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")
    patch_activation_dependencies(
        monkeypatch,
        schema=schema,
        target_version=target,
        active_versions=[other_active],
        current_version=current,
    )

    with pytest.raises(dynamic_schema_service.DynamicSchemaStateCorruptionError):
        run_async(
            dynamic_schema_service.activate_dynamic_schema_version(
                session,
                project_id=project.id,
                actor_id=actor.id,
                schema_id=schema.id,
                version_id=target.id,
            )
        )


def test_current_null_but_active_version_exists_is_rejected(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    schema = StubSchemaWithoutRelations(project_id=project.id, current_version_id=None)
    active_version = build_dynamic_schema_version(
        schema_id=schema.id,
        status="active",
        activated_by_id=uuid.uuid4(),
        activated_at=datetime(2026, 7, 30, 13, 0, tzinfo=timezone.utc),
    )
    target = build_dynamic_schema_version(schema_id=schema.id, status="draft", version_no=2)

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")
    patch_activation_dependencies(
        monkeypatch,
        schema=schema,
        target_version=target,
        active_versions=[active_version],
    )

    with pytest.raises(dynamic_schema_service.DynamicSchemaStateCorruptionError):
        run_async(
            dynamic_schema_service.activate_dynamic_schema_version(
                session,
                project_id=project.id,
                actor_id=actor.id,
                schema_id=schema.id,
                version_id=target.id,
            )
        )


def test_activation_lock_order_is_schema_then_target_then_active_versions(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    schema = StubSchemaWithoutRelations(project_id=project.id)
    target = build_dynamic_schema_version(schema_id=schema.id, status="draft")

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")
    call_order = patch_activation_dependencies(
        monkeypatch,
        schema=schema,
        target_version=target,
        active_versions=[],
    )

    run_async(
        dynamic_schema_service.activate_dynamic_schema_version(
            session,
            project_id=project.id,
            actor_id=actor.id,
            schema_id=schema.id,
            version_id=target.id,
        )
    )

    assert call_order == ["lock_schema", "lock_target_version", "lock_active_versions"]


def test_activation_failure_rolls_back_whole_transaction(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    schema = StubSchemaWithoutRelations(project_id=project.id)
    target = build_dynamic_schema_version(schema_id=schema.id, status="draft")

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")
    patch_activation_dependencies(
        monkeypatch,
        schema=schema,
        target_version=target,
        active_versions=[],
    )

    async def failing_flush():
        session.flush_called = True
        raise RuntimeError("flush failed")

    session.flush = failing_flush

    with pytest.raises(RuntimeError):
        run_async(
            dynamic_schema_service.activate_dynamic_schema_version(
                session,
                project_id=project.id,
                actor_id=actor.id,
                schema_id=schema.id,
                version_id=target.id,
            )
        )

    assert session.rollback_called is True
    assert session.commit_called is False


def test_activation_service_does_not_import_or_write_fact_models() -> None:
    source = inspect.getsource(dynamic_schema_service)
    assert "app.models.fact" not in source
    assert "from app.models.fact" not in source
