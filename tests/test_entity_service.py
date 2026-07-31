import asyncio
import inspect
import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from app.models.entity import Entity, EntityAlias
from app.models.project import Project
from app.models.project_member import ProjectMember
from app.models.user import User
from app.schemas.entity import EntityAliasCreateInput, EntityCreateInput
from app.services import entity as entity_service


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
        self.commit_count = 0
        self.rollback_count = 0
        self.savepoints: list[FakeSavepoint] = []

    async def commit(self) -> None:
        self.commit_called = True
        self.commit_count += 1

    async def rollback(self) -> None:
        self.rollback_called = True
        self.rollback_count += 1

    async def begin_nested(self) -> FakeSavepoint:
        savepoint = FakeSavepoint()
        self.savepoints.append(savepoint)
        return savepoint


def build_project(*, project_id: uuid.UUID | None = None, status: str = "active") -> Project:
    actual_project_id = project_id or uuid.uuid4()
    return Project(
        id=actual_project_id,
        name="Entity Project",
        slug=f"entity-{str(actual_project_id)[:8]}",
        created_by_id=uuid.uuid4(),
        status=status,
    )


def build_user(*, user_id: uuid.UUID | None = None, status: str = "active") -> User:
    actual_user_id = user_id or uuid.uuid4()
    return User(
        id=actual_user_id,
        handle=f"user-{str(actual_user_id)[:8]}",
        display_name="Entity Actor",
        status=status,
    )


def build_project_member(*, project_id: uuid.UUID, user_id: uuid.UUID, role: str) -> ProjectMember:
    return ProjectMember(
        id=uuid.uuid4(),
        project_id=project_id,
        user_id=user_id,
        role=role,
    )


def build_entity(
    *,
    project_id: uuid.UUID,
    entity_id: uuid.UUID | None = None,
    entity_type: str = "person",
    canonical_key: str = "zhang san",
    display_name: str = "张三",
    status: str = "active",
    created_by_id: uuid.UUID | None = None,
) -> Entity:
    return Entity(
        id=entity_id or uuid.uuid4(),
        project_id=project_id,
        entity_type=entity_type,
        canonical_key=canonical_key,
        display_name=display_name,
        identity_hash=entity_service.build_entity_identity_hash(
            project_id=project_id,
            entity_type=entity_type,
            canonical_key=canonical_key,
        ),
        status=status,
        merged_into_entity_id=None,
        created_by_id=created_by_id,
    )


def build_alias(
    *,
    entity_id: uuid.UUID,
    alias_id: uuid.UUID | None = None,
    alias_text: str = "张三",
    normalized_alias: str = "张三",
    language_code: str = "und",
    alias_kind: str = "alternate",
    status: str = "active",
    is_primary: bool = False,
    created_by_id: uuid.UUID | None = None,
) -> EntityAlias:
    return EntityAlias(
        id=alias_id or uuid.uuid4(),
        entity_id=entity_id,
        alias_text=alias_text,
        normalized_alias=normalized_alias,
        language_code=language_code,
        alias_kind=alias_kind,
        status=status,
        is_primary=is_primary,
        created_by_id=created_by_id,
    )


class StubEntityWithoutRelations:
    def __init__(self, *, project_id: uuid.UUID, entity_id: uuid.UUID | None = None, status: str = "active") -> None:
        self.id = entity_id or uuid.uuid4()
        self.project_id = project_id
        self.entity_type = "person"
        self.canonical_key = "zhang san"
        self.display_name = "张三"
        self.identity_hash = entity_service.build_entity_identity_hash(
            project_id=project_id,
            entity_type="person",
            canonical_key="zhang san",
        )
        self.status = status
        self.created_by_id = None

    @property
    def aliases(self):
        raise AssertionError("entity.aliases must not be accessed")


def patch_project_and_actor(monkeypatch, *, project: Project, actor: User | None = None, role: str | None = None) -> None:
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

    monkeypatch.setattr(entity_service.entity_repository, "get_project_by_id", fake_get_project_by_id)
    monkeypatch.setattr(entity_service.user_repository, "get_user_by_id", fake_get_user_by_id)
    monkeypatch.setattr(
        entity_service.entity_repository,
        "get_project_member_for_project",
        fake_get_project_member_for_project,
    )


class FakeDiag:
    def __init__(self, constraint_name: str | None) -> None:
        self.constraint_name = constraint_name


class FakeOrigError(Exception):
    def __init__(self, message: str, *, constraint_name: str | None = None) -> None:
        super().__init__(message)
        if constraint_name is not None:
            self.diag = FakeDiag(constraint_name)


def make_integrity_error(message: str, *, constraint_name: str | None = None) -> IntegrityError:
    return IntegrityError("insert", {}, FakeOrigError(message, constraint_name=constraint_name))


def test_create_entity_with_primary_alias_creates_both_records(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    payload = EntityCreateInput(entity_type="person", canonical_key="zhang san", display_name="张三")
    captured: dict[str, object] = {}

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")

    async def fake_get_entity_by_identity_for_update(_session, **_kwargs):
        return None

    async def fake_create_entity(_session, entity):
        captured["entity"] = entity
        return entity

    async def fake_create_entity_alias(_session, alias):
        captured["alias"] = alias
        return alias

    monkeypatch.setattr(entity_service.entity_repository, "get_entity_by_identity_for_update", fake_get_entity_by_identity_for_update)
    monkeypatch.setattr(entity_service.entity_repository, "create_entity", fake_create_entity)
    monkeypatch.setattr(entity_service.entity_repository, "create_entity_alias", fake_create_entity_alias)

    result = run_async(
        entity_service.create_entity_with_primary_alias(
            session,
            project_id=project.id,
            actor_id=actor.id,
            payload=payload,
        )
    )

    assert result is captured["entity"]
    assert captured["entity"].created_by_id == actor.id
    assert captured["alias"].entity_id == captured["entity"].id
    assert captured["alias"].alias_kind == "canonical"
    assert captured["alias"].status == "active"
    assert captured["alias"].is_primary is True
    assert session.commit_count == 1
    assert session.rollback_count == 0


def test_create_entity_is_idempotent_for_same_identity(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    existing = StubEntityWithoutRelations(project_id=project.id)

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")

    async def fake_get_entity_by_identity_for_update(_session, **_kwargs):
        return existing

    async def unexpected_create_alias(_session, _alias):
        raise AssertionError("primary alias must not be recreated for existing entity")

    monkeypatch.setattr(entity_service.entity_repository, "get_entity_by_identity_for_update", fake_get_entity_by_identity_for_update)
    monkeypatch.setattr(entity_service.entity_repository, "create_entity_alias", unexpected_create_alias)

    result = run_async(
        entity_service.create_entity_with_primary_alias(
            session,
            project_id=project.id,
            actor_id=actor.id,
            payload=EntityCreateInput(entity_type="person", canonical_key="zhang san", display_name="张三"),
        )
    )

    assert result is existing
    assert session.commit_count == 1
    assert session.rollback_count == 0


def test_concurrent_identity_create_returns_existing_entity(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    existing = build_entity(project_id=project.id)
    lookup_count = {"count": 0}

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")

    async def fake_get_entity_by_identity_for_update(_session, **_kwargs):
        lookup_count["count"] += 1
        if lookup_count["count"] == 1:
            return None
        return existing

    async def fake_get_entity_by_identity_hash_for_update(_session, **_kwargs):
        return existing

    async def fake_create_entity(_session, _entity):
        raise make_integrity_error("insert failed", constraint_name="uq_ent_proj_type_key")

    async def unexpected_create_alias(_session, _alias):
        raise AssertionError("alias creation must not run after idempotent conflict reuse")

    monkeypatch.setattr(entity_service.entity_repository, "get_entity_by_identity_for_update", fake_get_entity_by_identity_for_update)
    monkeypatch.setattr(entity_service.entity_repository, "get_entity_by_identity_hash_for_update", fake_get_entity_by_identity_hash_for_update)
    monkeypatch.setattr(entity_service.entity_repository, "create_entity", fake_create_entity)
    monkeypatch.setattr(entity_service.entity_repository, "create_entity_alias", unexpected_create_alias)

    result = run_async(
        entity_service.create_entity_with_primary_alias(
            session,
            project_id=project.id,
            actor_id=actor.id,
            payload=EntityCreateInput(entity_type="person", canonical_key="zhang san", display_name="张三"),
        )
    )

    assert result.id == existing.id
    assert session.savepoints[0].rollback_called is True
    assert session.commit_count == 1


def test_non_target_integrity_error_is_re_raised(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")

    async def fake_get_entity_by_identity_for_update(_session, **_kwargs):
        return None

    async def fake_create_entity(_session, _entity):
        raise make_integrity_error("uq_ent_proj_type_key appears in message", constraint_name="uq_some_other_constraint")

    monkeypatch.setattr(entity_service.entity_repository, "get_entity_by_identity_for_update", fake_get_entity_by_identity_for_update)
    monkeypatch.setattr(entity_service.entity_repository, "create_entity", fake_create_entity)

    with pytest.raises(IntegrityError):
        run_async(
            entity_service.create_entity_with_primary_alias(
                session,
                project_id=project.id,
                actor_id=actor.id,
                payload=EntityCreateInput(entity_type="person", canonical_key="zhang san", display_name="张三"),
            )
        )

    assert session.rollback_count == 1


def test_different_entities_can_share_same_alias_text(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    first = build_entity(project_id=project.id, canonical_key="zhang san")
    second = build_entity(project_id=project.id, canonical_key="li si")
    first_alias = build_alias(entity_id=first.id, normalized_alias="acme", alias_text="ACME")
    second_alias = build_alias(entity_id=second.id, normalized_alias="acme", alias_text="ＡＣＭＥ")
    first_alias.entity = first
    second_alias.entity = second

    async def fake_get_project_by_id(_session, project_id):
        return project if project_id == project.id else None

    async def fake_resolve_entity_alias(_session, **_kwargs):
        return [first_alias, second_alias]

    monkeypatch.setattr(entity_service.entity_repository, "get_project_by_id", fake_get_project_by_id)
    monkeypatch.setattr(entity_service.entity_repository, "resolve_entity_alias", fake_resolve_entity_alias)

    result = run_async(
        entity_service.resolve_entity_alias(
            session,
            project_id=project.id,
            entity_type="person",
            alias_text="acme",
        )
    )

    assert len(result) == 2
    assert {alias.entity_id for alias in result} == {first.id, second.id}


def test_same_entity_duplicate_alias_is_rejected(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    entity = StubEntityWithoutRelations(project_id=project.id)

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")

    async def fake_get_entity_by_id_for_update(_session, **_kwargs):
        return entity

    async def fake_create_entity_alias(_session, _alias):
        raise make_integrity_error("uq_ea_ent_norm_lang", constraint_name="uq_ea_ent_norm_lang")

    monkeypatch.setattr(entity_service.entity_repository, "get_entity_by_id_for_update", fake_get_entity_by_id_for_update)
    monkeypatch.setattr(entity_service.entity_repository, "create_entity_alias", fake_create_entity_alias)

    with pytest.raises(IntegrityError):
        run_async(
            entity_service.add_entity_alias(
                session,
                project_id=project.id,
                actor_id=actor.id,
                entity_id=entity.id,
                payload=EntityAliasCreateInput(alias_text="张三"),
            )
        )

    assert session.rollback_count == 1


def test_ambiguous_alias_returns_multiple_candidates_without_auto_selection(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    first = build_entity(project_id=project.id, canonical_key="zhang san")
    second = build_entity(project_id=project.id, canonical_key="zhang san 2")
    first_alias = build_alias(entity_id=first.id, normalized_alias="zhang san", alias_text="张三")
    second_alias = build_alias(entity_id=second.id, normalized_alias="zhang san", alias_text="張三")
    first_alias.entity = first
    second_alias.entity = second

    async def fake_get_project_by_id(_session, project_id):
        return project if project_id == project.id else None

    async def fake_resolve_entity_alias(_session, **_kwargs):
        return [first_alias, second_alias]

    monkeypatch.setattr(entity_service.entity_repository, "get_project_by_id", fake_get_project_by_id)
    monkeypatch.setattr(entity_service.entity_repository, "resolve_entity_alias", fake_resolve_entity_alias)

    result = run_async(
        entity_service.resolve_entity_alias(
            session,
            project_id=project.id,
            entity_type="person",
            alias_text="张三",
        )
    )

    assert tuple(alias.id for alias in result) == (first_alias.id, second_alias.id)


def test_get_integrity_constraint_name_reads_only_diag_constraint_name() -> None:
    error = make_integrity_error("message with uq_ent_proj_type_key", constraint_name="uq_ent_proj_hash")

    assert entity_service._get_integrity_constraint_name(error) == "uq_ent_proj_hash"


def test_get_integrity_constraint_name_ignores_text_without_target_diag() -> None:
    error = make_integrity_error("uq_ent_proj_type_key appears in text", constraint_name="uq_other")

    assert entity_service._get_integrity_constraint_name(error) == "uq_other"


def test_get_integrity_constraint_name_without_diag_returns_none() -> None:
    error = IntegrityError("insert", {}, Exception("uq_ent_proj_type_key"))

    assert entity_service._get_integrity_constraint_name(error) is None


def test_target_integrity_conflict_without_reloaded_entity_reraises_original(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    original_error = make_integrity_error("insert failed", constraint_name="uq_ent_proj_hash")

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")

    async def fake_get_entity_by_identity_for_update(_session, **_kwargs):
        return None

    async def fake_get_entity_by_identity_hash_for_update(_session, **_kwargs):
        return None

    async def fake_create_entity(_session, _entity):
        raise original_error

    monkeypatch.setattr(entity_service.entity_repository, "get_entity_by_identity_for_update", fake_get_entity_by_identity_for_update)
    monkeypatch.setattr(entity_service.entity_repository, "get_entity_by_identity_hash_for_update", fake_get_entity_by_identity_hash_for_update)
    monkeypatch.setattr(entity_service.entity_repository, "create_entity", fake_create_entity)

    with pytest.raises(IntegrityError) as exc_info:
        run_async(
            entity_service.create_entity_with_primary_alias(
                session,
                project_id=project.id,
                actor_id=actor.id,
                payload=EntityCreateInput(entity_type="person", canonical_key="zhang san", display_name="张三"),
            )
        )

    assert exc_info.value is original_error
    assert session.rollback_count == 1


def test_existing_entity_with_identity_hash_mismatch_is_rejected(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    existing = StubEntityWithoutRelations(project_id=project.id)
    existing.identity_hash = "0" * 64

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")

    async def fake_get_entity_by_identity_for_update(_session, **_kwargs):
        return existing

    monkeypatch.setattr(entity_service.entity_repository, "get_entity_by_identity_for_update", fake_get_entity_by_identity_for_update)

    with pytest.raises(entity_service.EntityIdentityConflictError):
        run_async(
            entity_service.create_entity_with_primary_alias(
                session,
                project_id=project.id,
                actor_id=actor.id,
                payload=EntityCreateInput(entity_type="person", canonical_key="zhang san", display_name="张三"),
            )
        )

    assert session.rollback_count == 1


def test_hash_reload_with_different_identity_is_rejected(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    mismatched = StubEntityWithoutRelations(project_id=project.id)
    mismatched.canonical_key = "li si"

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")

    async def fake_get_entity_by_identity_for_update(_session, **_kwargs):
        return None

    async def fake_get_entity_by_identity_hash_for_update(_session, **_kwargs):
        return mismatched

    async def fake_create_entity(_session, _entity):
        raise make_integrity_error("insert failed", constraint_name="uq_ent_proj_hash")

    monkeypatch.setattr(entity_service.entity_repository, "get_entity_by_identity_for_update", fake_get_entity_by_identity_for_update)
    monkeypatch.setattr(entity_service.entity_repository, "get_entity_by_identity_hash_for_update", fake_get_entity_by_identity_hash_for_update)
    monkeypatch.setattr(entity_service.entity_repository, "create_entity", fake_create_entity)

    with pytest.raises(entity_service.EntityIdentityConflictError):
        run_async(
            entity_service.create_entity_with_primary_alias(
                session,
                project_id=project.id,
                actor_id=actor.id,
                payload=EntityCreateInput(entity_type="person", canonical_key="zhang san", display_name="张三"),
            )
        )

    assert session.rollback_count == 1


@pytest.mark.parametrize("status", ["merged", "archived"])
def test_merged_or_archived_entity_rejects_new_alias(monkeypatch, status: str) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    entity = StubEntityWithoutRelations(project_id=project.id, status=status)

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")

    async def fake_get_entity_by_id_for_update(_session, **_kwargs):
        return entity

    monkeypatch.setattr(entity_service.entity_repository, "get_entity_by_id_for_update", fake_get_entity_by_id_for_update)

    with pytest.raises(entity_service.EntityStateError):
        run_async(
            entity_service.add_entity_alias(
                session,
                project_id=project.id,
                actor_id=actor.id,
                entity_id=entity.id,
                payload=EntityAliasCreateInput(alias_text="张三"),
            )
        )

    assert session.rollback_count == 1


def test_active_primary_alias_cannot_be_retired(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    entity = build_entity(project_id=project.id)
    alias = build_alias(
        entity_id=entity.id,
        alias_kind="canonical",
        status="active",
        is_primary=True,
    )
    alias.entity = entity

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")

    async def fake_get_entity_alias_by_id_for_update(_session, **_kwargs):
        return alias

    monkeypatch.setattr(entity_service.entity_repository, "get_entity_alias_by_id_for_update", fake_get_entity_alias_by_id_for_update)

    with pytest.raises(entity_service.PrimaryEntityAliasRetireError):
        run_async(
            entity_service.retire_entity_alias(
                session,
                project_id=project.id,
                actor_id=actor.id,
                alias_id=alias.id,
            )
        )

    assert session.rollback_count == 1


def test_resolve_entity_alias_commits_once_on_success(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    entity = build_entity(project_id=project.id)
    alias = build_alias(entity_id=entity.id, alias_text="ACME", normalized_alias="acme")
    alias.entity = entity

    async def fake_get_project_by_id(_session, project_id):
        return project if project_id == project.id else None

    async def fake_resolve_entity_alias(_session, **_kwargs):
        return [alias]

    monkeypatch.setattr(entity_service.entity_repository, "get_project_by_id", fake_get_project_by_id)
    monkeypatch.setattr(entity_service.entity_repository, "resolve_entity_alias", fake_resolve_entity_alias)

    result = run_async(
        entity_service.resolve_entity_alias(
            session,
            project_id=project.id,
            entity_type="person",
            alias_text="ＡＣＭＥ",
        )
    )

    assert result == (alias,)
    assert session.commit_count == 1
    assert session.rollback_count == 0


def test_resolve_entity_alias_failure_rolls_back(monkeypatch) -> None:
    session = FakeSession()
    project = build_project(status="archived")

    async def fake_get_project_by_id(_session, project_id):
        return project if project_id == project.id else None

    monkeypatch.setattr(entity_service.entity_repository, "get_project_by_id", fake_get_project_by_id)

    with pytest.raises(entity_service.EntityProjectNotFoundError):
        run_async(
            entity_service.resolve_entity_alias(
                session,
                project_id=project.id,
                entity_type="person",
                alias_text="acme",
            )
        )

    assert session.commit_count == 0
    assert session.rollback_count == 1


def test_resolve_entity_alias_cancelled_error_rolls_back_and_propagates(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()

    async def fake_get_project_by_id(_session, project_id):
        return project if project_id == project.id else None

    async def fake_resolve_entity_alias(_session, **_kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(entity_service.entity_repository, "get_project_by_id", fake_get_project_by_id)
    monkeypatch.setattr(entity_service.entity_repository, "resolve_entity_alias", fake_resolve_entity_alias)

    with pytest.raises(asyncio.CancelledError):
        run_async(
            entity_service.resolve_entity_alias(
                session,
                project_id=project.id,
                entity_type="person",
                alias_text="acme",
            )
        )

    assert session.commit_count == 0
    assert session.rollback_count == 1


def test_owner_and_editor_can_write_entities(monkeypatch, role: str = "owner") -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    captured: dict[str, object] = {}

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role=role)

    async def fake_get_entity_by_identity_for_update(_session, **_kwargs):
        return None

    async def fake_create_entity(_session, entity):
        captured["entity"] = entity
        return entity

    async def fake_create_entity_alias(_session, alias):
        captured["alias"] = alias
        return alias

    monkeypatch.setattr(entity_service.entity_repository, "get_entity_by_identity_for_update", fake_get_entity_by_identity_for_update)
    monkeypatch.setattr(entity_service.entity_repository, "create_entity", fake_create_entity)
    monkeypatch.setattr(entity_service.entity_repository, "create_entity_alias", fake_create_entity_alias)

    result = run_async(
        entity_service.create_entity_with_primary_alias(
            session,
            project_id=project.id,
            actor_id=actor.id,
            payload=EntityCreateInput(entity_type="person", canonical_key="zhang san", display_name="张三"),
        )
    )

    assert result.id == captured["entity"].id


@pytest.mark.parametrize("role", ["owner", "editor"])
def test_owner_and_editor_roles_both_work(monkeypatch, role: str) -> None:
    test_owner_and_editor_can_write_entities(monkeypatch, role=role)


def test_viewer_cannot_create_entity(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="viewer")

    with pytest.raises(entity_service.EntityPermissionError):
        run_async(
            entity_service.create_entity_with_primary_alias(
                session,
                project_id=project.id,
                actor_id=actor.id,
                payload=EntityCreateInput(entity_type="person", canonical_key="zhang san", display_name="张三"),
            )
        )

    assert session.rollback_count == 1


def test_archived_project_is_rejected(monkeypatch) -> None:
    session = FakeSession()
    project = build_project(status="archived")
    actor = build_user()

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")

    with pytest.raises(entity_service.EntityProjectNotFoundError):
        run_async(
            entity_service.create_entity_with_primary_alias(
                session,
                project_id=project.id,
                actor_id=actor.id,
                payload=EntityCreateInput(entity_type="person", canonical_key="zhang san", display_name="张三"),
            )
        )

    assert session.rollback_count == 1


def test_successful_add_alias_commits_once_without_rollback(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    entity = StubEntityWithoutRelations(project_id=project.id)
    captured: dict[str, object] = {}

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")

    async def fake_get_entity_by_id_for_update(_session, **_kwargs):
        return entity

    async def fake_create_entity_alias(_session, alias):
        captured["alias"] = alias
        return alias

    monkeypatch.setattr(entity_service.entity_repository, "get_entity_by_id_for_update", fake_get_entity_by_id_for_update)
    monkeypatch.setattr(entity_service.entity_repository, "create_entity_alias", fake_create_entity_alias)

    result = run_async(
        entity_service.add_entity_alias(
            session,
            project_id=project.id,
            actor_id=actor.id,
            entity_id=entity.id,
            payload=EntityAliasCreateInput(alias_text="  ＡＣＭＥ  ", alias_kind="alternate"),
        )
    )

    assert result is captured["alias"]
    assert captured["alias"].normalized_alias == "acme"
    assert session.commit_count == 1
    assert session.rollback_count == 0


def test_add_entity_alias_rejects_primary_change_before_repository_write(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    entity = StubEntityWithoutRelations(project_id=project.id)
    create_calls = {"count": 0}

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")

    async def fake_get_entity_by_id_for_update(_session, **_kwargs):
        return entity

    async def fake_create_entity_alias(_session, _alias):
        create_calls["count"] += 1
        return _alias

    monkeypatch.setattr(entity_service.entity_repository, "get_entity_by_id_for_update", fake_get_entity_by_id_for_update)
    monkeypatch.setattr(entity_service.entity_repository, "create_entity_alias", fake_create_entity_alias)

    with pytest.raises(entity_service.PrimaryEntityAliasChangeError):
        run_async(
            entity_service.add_entity_alias(
                session,
                project_id=project.id,
                actor_id=actor.id,
                entity_id=entity.id,
                payload=EntityAliasCreateInput(alias_text="张三", is_primary=True),
            )
        )

    assert create_calls["count"] == 0
    assert session.commit_count == 0
    assert session.rollback_count == 1


def test_cancelled_error_rolls_back_and_propagates(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()

    patch_project_and_actor(monkeypatch, project=project, actor=actor, role="owner")

    async def fake_get_entity_by_identity_for_update(_session, **_kwargs):
        return None

    async def fake_create_entity(_session, entity):
        return entity

    async def fake_create_entity_alias(_session, _alias):
        raise asyncio.CancelledError()

    monkeypatch.setattr(entity_service.entity_repository, "get_entity_by_identity_for_update", fake_get_entity_by_identity_for_update)
    monkeypatch.setattr(entity_service.entity_repository, "create_entity", fake_create_entity)
    monkeypatch.setattr(entity_service.entity_repository, "create_entity_alias", fake_create_entity_alias)

    with pytest.raises(asyncio.CancelledError):
        run_async(
            entity_service.create_entity_with_primary_alias(
                session,
                project_id=project.id,
                actor_id=actor.id,
                payload=EntityCreateInput(entity_type="person", canonical_key="zhang san", display_name="张三"),
            )
        )

    assert session.rollback_count == 1
    assert session.commit_count == 0


def test_resolve_entity_alias_uses_explicit_join_and_no_lazy_loading() -> None:
    from app.repositories import entity as entity_repository

    source = inspect.getsource(entity_repository.resolve_entity_alias)
    assert ".join(Entity, EntityAlias.entity_id == Entity.id)" in source
    assert "joinedload(EntityAlias.entity)" in source


def test_entity_service_does_not_import_fact_or_call_llm() -> None:
    source = inspect.getsource(entity_service)
    assert "app.models.fact" not in source
    assert "from app.models.fact" not in source
    assert "llm" not in source.lower()
