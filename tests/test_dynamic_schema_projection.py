from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import inspect
from types import SimpleNamespace
import uuid

import pytest

from app.services import dynamic_schema_projection as projection_service


def run_async(awaitable):
    import asyncio

    return asyncio.run(awaitable)


def _uuid(seed: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"dynamic-schema-projection:{seed}")


class FakeSession:
    def __init__(self) -> None:
        self.commit_count = 0
        self.rollback_count = 0

    async def commit(self) -> None:
        self.commit_count += 1

    async def rollback(self) -> None:
        self.rollback_count += 1

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class SessionFactory:
    def __init__(self) -> None:
        self.sessions: list[FakeSession] = []

    def __call__(self) -> FakeSession:
        session = FakeSession()
        self.sessions.append(session)
        return session


class SentinelObject:
    def __init__(self, sentinel: str) -> None:
        self.sentinel = sentinel

    def __repr__(self) -> str:
        return self.sentinel


def _project(project_id: uuid.UUID | None = None):
    return SimpleNamespace(id=project_id or _uuid("project"))


def _schema(
    *,
    project_id: uuid.UUID,
    schema_id: uuid.UUID | None = None,
    current_version_id: uuid.UUID | None = None,
    status: str = "active",
    schema_key: str = "profile.main",
    name: str = "Profile Main",
    subject_kind: str = "person",
    description: str | None = "Main profile",
):
    return SimpleNamespace(
        id=schema_id or _uuid("schema"),
        project_id=project_id,
        schema_key=schema_key,
        name=name,
        subject_kind=subject_kind,
        description=description,
        status=status,
        current_version_id=current_version_id,
    )


def _version(
    *,
    schema_id: uuid.UUID,
    version_id: uuid.UUID | None = None,
    version_no: int = 1,
    status: str = "draft",
    source_kind: str = "human",
    summary: str | None = "Schema summary",
    layout_config: object | None = None,
    created_by_id: uuid.UUID | None = None,
    activated_by_id: uuid.UUID | None = None,
    activated_at: datetime | None = None,
):
    return SimpleNamespace(
        id=version_id or _uuid(f"version:{version_no}:{status}"),
        schema_id=schema_id,
        version_no=version_no,
        status=status,
        source_kind=source_kind,
        summary=summary,
        layout_config={} if layout_config is None else layout_config,
        created_by_id=created_by_id,
        activated_by_id=activated_by_id,
        activated_at=activated_at,
    )


def _field(
    *,
    schema_version_id: uuid.UUID,
    seed: str,
    field_key: str,
    display_order: int,
    label: str | None = None,
    description: str | None = None,
    predicate_key: str | None = None,
    scope_key: str | None = None,
    expected_value_type: str = "string",
    cardinality: str = "one",
    is_required: bool = False,
    is_title: bool = False,
    is_summary: bool = False,
    is_hidden: bool = False,
    group_key: str | None = None,
    display_config: object | None = None,
    validation_rules: object | None = None,
):
    return SimpleNamespace(
        id=_uuid(f"field:{seed}"),
        schema_version_id=schema_version_id,
        field_key=field_key,
        label=label or field_key.title(),
        description=description,
        predicate_key=predicate_key or f"person.{field_key}",
        scope_key=scope_key,
        expected_value_type=expected_value_type,
        cardinality=cardinality,
        is_required=is_required,
        is_title=is_title,
        is_summary=is_summary,
        is_hidden=is_hidden,
        group_key=group_key,
        display_order=display_order,
        display_config={} if display_config is None else display_config,
        validation_rules={} if validation_rules is None else validation_rules,
        created_at=datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc),
    )


def _fixture(
    *,
    requested_status: str = "active",
    current_status: str | None = None,
    current_points_to_requested: bool = True,
):
    project = _project()
    requested_version_id = _uuid(f"requested:{requested_status}")
    current_version_id = requested_version_id if current_points_to_requested else _uuid("current")
    schema = _schema(
        project_id=project.id,
        current_version_id=current_version_id if current_status is not None else None,
    )
    requested_version = _version(
        schema_id=schema.id,
        version_id=requested_version_id,
        version_no=2,
        status=requested_status,
        source_kind="human",
        created_by_id=_uuid("created-by"),
        activated_by_id=(
            _uuid("activated-by")
            if requested_status in {"active", "retired"}
            else None
        ),
        activated_at=(
            datetime(2026, 8, 2, 10, 0, tzinfo=timezone.utc)
            if requested_status in {"active", "retired"}
            else None
        ),
        layout_config={"sections": ["main", "meta"]},
    )
    current_version = None
    active_versions: list[object] = []
    if current_status is not None:
        current_version = _version(
            schema_id=schema.id,
            version_id=schema.current_version_id,
            version_no=99,
            status=current_status,
            source_kind="human",
            created_by_id=_uuid("current-created-by"),
            activated_by_id=_uuid("current-activated-by"),
            activated_at=datetime(2026, 8, 2, 9, 0, tzinfo=timezone.utc),
        )
        active_versions = [current_version]
    fields = [
        _field(
            schema_version_id=requested_version.id,
            seed="summary",
            field_key="summary",
            display_order=1,
            is_summary=True,
            group_key="meta",
            display_config={"multiline": True},
        ),
        _field(
            schema_version_id=requested_version.id,
            seed="title",
            field_key="title",
            display_order=0,
            is_title=True,
            is_required=True,
            validation_rules={"max_length": 100},
        ),
    ]
    return {
        "project": project,
        "schema": schema,
        "requested_version": requested_version,
        "current_version": current_version,
        "fields": fields,
        "active_versions": active_versions,
    }


def _install_repository(
    monkeypatch: pytest.MonkeyPatch,
    *,
    project=None,
    schema=None,
    version=None,
    fields=None,
    active_versions=None,
    capture: dict[str, object] | None = None,
) -> None:
    async def fake_get_project_by_id(_session, *, project_id):
        if capture is not None:
            capture.setdefault("project_ids", []).append(project_id)
        if project is not None and project_id == project.id:
            return project
        return None

    async def fake_get_dynamic_schema_by_id(_session, *, schema_id):
        if capture is not None:
            capture.setdefault("schema_ids", []).append(schema_id)
        if schema is not None and schema_id == schema.id:
            return schema
        return None

    async def fake_get_dynamic_schema_version_by_id(_session, *, schema_version_id):
        if capture is not None:
            capture.setdefault("version_ids", []).append(schema_version_id)
        if version is not None and schema_version_id == version.id:
            return version
        return None

    async def fake_list_fields(_session, *, schema_version_id):
        if capture is not None:
            capture.setdefault("field_version_ids", []).append(schema_version_id)
        if version is not None and schema_version_id == version.id:
            return list(fields or [])
        return []

    async def fake_list_active_versions(_session, *, schema_id):
        if capture is not None:
            capture.setdefault("active_schema_ids", []).append(schema_id)
        if schema is not None and schema_id == schema.id:
            return list(active_versions or [])
        return []

    monkeypatch.setattr(
        projection_service.projection_repository,
        "get_project_by_id",
        fake_get_project_by_id,
    )
    monkeypatch.setattr(
        projection_service.projection_repository,
        "get_dynamic_schema_by_id",
        fake_get_dynamic_schema_by_id,
    )
    monkeypatch.setattr(
        projection_service.projection_repository,
        "get_dynamic_schema_version_by_id",
        fake_get_dynamic_schema_version_by_id,
    )
    monkeypatch.setattr(
        projection_service.projection_repository,
        "list_dynamic_schema_fields_by_version_id",
        fake_list_fields,
    )
    monkeypatch.setattr(
        projection_service.projection_repository,
        "list_active_dynamic_schema_versions",
        fake_list_active_versions,
    )


@pytest.mark.parametrize(
    ("requested_status", "current_status", "current_points_to_requested", "expected_is_current"),
    [
        ("draft", None, False, False),
        ("proposed", None, False, False),
        ("active", "active", True, True),
        ("retired", "active", False, False),
    ],
)
def test_get_dynamic_schema_definition_snapshot_reads_requested_version_status(
    monkeypatch: pytest.MonkeyPatch,
    requested_status: str,
    current_status: str | None,
    current_points_to_requested: bool,
    expected_is_current: bool,
) -> None:
    fixture = _fixture(
        requested_status=requested_status,
        current_status=current_status,
        current_points_to_requested=current_points_to_requested,
    )
    factory = SessionFactory()
    _install_repository(
        monkeypatch,
        project=fixture["project"],
        schema=fixture["schema"],
        version=fixture["requested_version"],
        fields=fixture["fields"],
        active_versions=fixture["active_versions"],
    )

    snapshot = run_async(
        projection_service.get_dynamic_schema_definition_snapshot(
            factory,
            project_id=fixture["project"].id,
            schema_id=fixture["schema"].id,
            schema_version_id=fixture["requested_version"].id,
        )
    )

    assert snapshot.version_status == requested_status
    assert snapshot.is_current is expected_is_current
    assert snapshot.field_count == 2
    assert [field.field_key for field in snapshot.fields] == ["title", "summary"]


def test_get_current_dynamic_schema_definition_snapshot_follows_current_version_id_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(
        requested_status="active",
        current_status="active",
        current_points_to_requested=False,
    )
    higher_version = _version(
        schema_id=fixture["schema"].id,
        version_id=_uuid("higher-version"),
        version_no=999,
        status="retired",
        source_kind="human",
        created_by_id=_uuid("higher-created"),
        activated_by_id=_uuid("higher-activated"),
        activated_at=datetime(2026, 8, 1, 8, 0, tzinfo=timezone.utc),
    )
    current_fields = [
        _field(
            schema_version_id=fixture["current_version"].id,
            seed="current-title",
            field_key="title",
            display_order=0,
            is_title=True,
        ),
        _field(
            schema_version_id=fixture["current_version"].id,
            seed="current-summary",
            field_key="summary",
            display_order=1,
            is_summary=True,
        ),
    ]
    capture: dict[str, object] = {}
    factory = SessionFactory()
    _install_repository(
        monkeypatch,
        project=fixture["project"],
        schema=fixture["schema"],
        version=fixture["current_version"],
        fields=current_fields,
        active_versions=[fixture["current_version"]],
        capture=capture,
    )

    snapshot = run_async(
        projection_service.get_current_dynamic_schema_definition_snapshot(
            factory,
            project_id=fixture["project"].id,
            schema_id=fixture["schema"].id,
        )
    )

    assert snapshot.schema_version_id == fixture["current_version"].id
    assert snapshot.schema_version_id != higher_version.id
    assert capture["version_ids"] == [fixture["current_version"].id]


@pytest.mark.parametrize(
    ("project_present", "schema_present", "version_present", "expected_code"),
    [
        (False, True, True, "dynamic_schema_definition_snapshot_project_not_found"),
        (True, False, True, "dynamic_schema_definition_snapshot_schema_not_found"),
        (True, True, False, "dynamic_schema_definition_snapshot_version_not_found"),
    ],
)
def test_get_dynamic_schema_definition_snapshot_rejects_unknown_or_cross_source_ids(
    monkeypatch: pytest.MonkeyPatch,
    project_present: bool,
    schema_present: bool,
    version_present: bool,
    expected_code: str,
) -> None:
    fixture = _fixture(requested_status="draft")
    factory = SessionFactory()
    _install_repository(
        monkeypatch,
        project=fixture["project"] if project_present else None,
        schema=fixture["schema"] if schema_present else None,
        version=fixture["requested_version"] if version_present else None,
        fields=fixture["fields"],
        active_versions=[],
    )

    with pytest.raises(
        projection_service.DynamicSchemaDefinitionSnapshotError,
        match=expected_code,
    ):
        run_async(
            projection_service.get_dynamic_schema_definition_snapshot(
                factory,
                project_id=fixture["project"].id,
                schema_id=fixture["schema"].id,
                schema_version_id=fixture["requested_version"].id,
            )
        )


@pytest.mark.parametrize(
    ("current_version_id", "active_versions", "requested_status"),
    [
        (_uuid("current"), [], "draft"),
        (None, [_version(schema_id=_uuid("schema"), status="active", activated_by_id=_uuid("a"), activated_at=datetime(2026, 8, 2, 8, 0, tzinfo=timezone.utc))], "draft"),
        (_uuid("current"), [
            _version(schema_id=_uuid("schema"), version_id=_uuid("active-1"), status="active", activated_by_id=_uuid("a1"), activated_at=datetime(2026, 8, 2, 8, 0, tzinfo=timezone.utc)),
            _version(schema_id=_uuid("schema"), version_id=_uuid("active-2"), status="active", activated_by_id=_uuid("a2"), activated_at=datetime(2026, 8, 2, 9, 0, tzinfo=timezone.utc)),
        ], "draft"),
    ],
)
def test_get_dynamic_schema_definition_snapshot_rejects_current_pointer_or_active_set_drift(
    monkeypatch: pytest.MonkeyPatch,
    current_version_id: uuid.UUID | None,
    active_versions: list[object],
    requested_status: str,
) -> None:
    fixture = _fixture(requested_status=requested_status)
    fixture["schema"].current_version_id = current_version_id
    for active_version in active_versions:
        active_version.schema_id = fixture["schema"].id
    factory = SessionFactory()
    _install_repository(
        monkeypatch,
        project=fixture["project"],
        schema=fixture["schema"],
        version=fixture["requested_version"],
        fields=fixture["fields"],
        active_versions=active_versions,
    )

    with pytest.raises(
        projection_service.DynamicSchemaDefinitionSnapshotInvariantError,
        match="dynamic_schema_definition_snapshot_current_state_invalid",
    ):
        run_async(
            projection_service.get_dynamic_schema_definition_snapshot(
                factory,
                project_id=fixture["project"].id,
                schema_id=fixture["schema"].id,
                schema_version_id=fixture["requested_version"].id,
            )
        )


@pytest.mark.parametrize(
    ("mutator", "expected_code"),
    [
        (
            lambda fixture: fixture["fields"].append(
                _field(
                    schema_version_id=fixture["requested_version"].id,
                    seed="dup-key",
                    field_key="title",
                    display_order=2,
                )
            ),
            "dynamic_schema_definition_snapshot_version_invalid",
        ),
        (
            lambda fixture: fixture["fields"].append(
                _field(
                    schema_version_id=fixture["requested_version"].id,
                    seed="dup-order",
                    field_key="extra",
                    display_order=0,
                )
            ),
            "dynamic_schema_definition_snapshot_version_invalid",
        ),
        (
            lambda fixture: setattr(fixture["fields"][0], "expected_value_type", "bogus"),
            "dynamic_schema_definition_snapshot_field_invalid",
        ),
        (
            lambda fixture: setattr(fixture["fields"][0], "is_hidden", True)
            or setattr(fixture["fields"][0], "is_title", True),
            "dynamic_schema_definition_snapshot_field_invalid",
        ),
        (
            lambda fixture: setattr(
                fixture["fields"][0],
                "schema_version_id",
                _uuid("foreign-version"),
            ),
            "dynamic_schema_definition_snapshot_field_invalid",
        ),
    ],
)
def test_get_dynamic_schema_definition_snapshot_rejects_invalid_field_contracts(
    monkeypatch: pytest.MonkeyPatch,
    mutator,
    expected_code: str,
) -> None:
    fixture = _fixture(requested_status="draft")
    mutator(fixture)
    factory = SessionFactory()
    _install_repository(
        monkeypatch,
        project=fixture["project"],
        schema=fixture["schema"],
        version=fixture["requested_version"],
        fields=fixture["fields"],
        active_versions=[],
    )

    with pytest.raises(
        projection_service.DynamicSchemaDefinitionSnapshotInvariantError,
        match=expected_code,
    ):
        run_async(
            projection_service.get_dynamic_schema_definition_snapshot(
                factory,
                project_id=fixture["project"].id,
                schema_id=fixture["schema"].id,
                schema_version_id=fixture["requested_version"].id,
            )
        )


@pytest.mark.parametrize(
    ("status", "activated_by_id", "activated_at", "expected_code"),
    [
        (
            "draft",
            _uuid("bad-activator"),
            datetime(2026, 8, 2, 8, 0, tzinfo=timezone.utc),
            "dynamic_schema_definition_snapshot_version_invalid",
        ),
        (
            "proposed",
            _uuid("bad-activator-proposed"),
            datetime(2026, 8, 2, 8, 30, tzinfo=timezone.utc),
            "dynamic_schema_definition_snapshot_version_invalid",
        ),
        (
            "active",
            None,
            None,
            "dynamic_schema_definition_snapshot_version_invalid",
        ),
    ],
)
def test_get_dynamic_schema_definition_snapshot_rejects_invalid_version_activation_shape(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    activated_by_id: uuid.UUID | None,
    activated_at: datetime | None,
    expected_code: str,
) -> None:
    fixture = _fixture(
        requested_status=status,
        current_status="active" if status == "active" else None,
        current_points_to_requested=status == "active",
    )
    fixture["requested_version"].activated_by_id = activated_by_id
    fixture["requested_version"].activated_at = activated_at
    factory = SessionFactory()
    _install_repository(
        monkeypatch,
        project=fixture["project"],
        schema=fixture["schema"],
        version=fixture["requested_version"],
        fields=fixture["fields"],
        active_versions=fixture["active_versions"],
    )

    with pytest.raises(
        projection_service.DynamicSchemaDefinitionSnapshotInvariantError,
        match=expected_code,
    ):
        run_async(
            projection_service.get_dynamic_schema_definition_snapshot(
                factory,
                project_id=fixture["project"].id,
                schema_id=fixture["schema"].id,
                schema_version_id=fixture["requested_version"].id,
            )
        )


@pytest.mark.parametrize(
    "invalid_config",
    [
        {"bad": float("nan")},
        {"bad": float("inf")},
        {"bad": b"bytes"},
        {"bad": datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc)},
        {1: "bad-key"},
        {"bad": SentinelObject("SENSITIVE_DYNAMIC_SCHEMA_SENTINEL")},
    ],
)
def test_get_dynamic_schema_definition_snapshot_rejects_invalid_json_configs_without_leaking_sentinel(
    monkeypatch: pytest.MonkeyPatch,
    invalid_config: object,
) -> None:
    fixture = _fixture(requested_status="draft")
    fixture["requested_version"].layout_config = invalid_config
    factory = SessionFactory()
    _install_repository(
        monkeypatch,
        project=fixture["project"],
        schema=fixture["schema"],
        version=fixture["requested_version"],
        fields=fixture["fields"],
        active_versions=[],
    )

    with pytest.raises(
        projection_service.DynamicSchemaDefinitionSnapshotInvariantError,
        match="dynamic_schema_definition_snapshot_json_config_invalid",
    ) as exc_info:
        run_async(
            projection_service.get_dynamic_schema_definition_snapshot(
                factory,
                project_id=fixture["project"].id,
                schema_id=fixture["schema"].id,
                schema_version_id=fixture["requested_version"].id,
            )
        )

    assert "SENSITIVE_DYNAMIC_SCHEMA_SENTINEL" not in str(exc_info.value)


def test_get_dynamic_schema_definition_snapshot_orders_fields_and_manifest_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(requested_status="active", current_status="active")
    factory = SessionFactory()
    _install_repository(
        monkeypatch,
        project=fixture["project"],
        schema=fixture["schema"],
        version=fixture["requested_version"],
        fields=list(reversed(fixture["fields"])),
        active_versions=[fixture["requested_version"]],
    )

    first = run_async(
        projection_service.get_dynamic_schema_definition_snapshot(
            factory,
            project_id=fixture["project"].id,
            schema_id=fixture["schema"].id,
            schema_version_id=fixture["requested_version"].id,
        )
    )
    second = run_async(
        projection_service.get_dynamic_schema_definition_snapshot(
            factory,
            project_id=fixture["project"].id,
            schema_id=fixture["schema"].id,
            schema_version_id=fixture["requested_version"].id,
        )
    )

    assert [field.field_key for field in first.fields] == ["title", "summary"]
    assert first.definition_manifest_hash == second.definition_manifest_hash


@pytest.mark.parametrize("mutation", ["schema", "version", "field", "config", "algorithm"])
def test_get_dynamic_schema_definition_snapshot_manifest_changes_when_definition_changes(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    fixture = _fixture(requested_status="active", current_status="active")
    factory = SessionFactory()
    _install_repository(
        monkeypatch,
        project=fixture["project"],
        schema=fixture["schema"],
        version=fixture["requested_version"],
        fields=fixture["fields"],
        active_versions=[fixture["requested_version"]],
    )
    baseline = run_async(
        projection_service.get_dynamic_schema_definition_snapshot(
            factory,
            project_id=fixture["project"].id,
            schema_id=fixture["schema"].id,
            schema_version_id=fixture["requested_version"].id,
        )
    )

    if mutation == "schema":
        fixture["schema"].name = "Changed Name"
    elif mutation == "version":
        fixture["requested_version"].summary = "Changed Summary"
    elif mutation == "field":
        fixture["fields"][0].label = "Changed Label"
    elif mutation == "config":
        fixture["requested_version"].layout_config = {"sections": ["changed"]}
    else:
        monkeypatch.setattr(
            projection_service,
            "DYNAMIC_SCHEMA_DEFINITION_SNAPSHOT_ALGORITHM_VERSION",
            "1.0.1",
        )

    changed = run_async(
        projection_service.get_dynamic_schema_definition_snapshot(
            factory,
            project_id=fixture["project"].id,
            schema_id=fixture["schema"].id,
            schema_version_id=fixture["requested_version"].id,
        )
    )

    assert baseline.definition_manifest_hash != changed.definition_manifest_hash


def test_get_dynamic_schema_definition_snapshot_returns_frozen_dto_without_orm_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(requested_status="active", current_status="active")
    factory = SessionFactory()
    _install_repository(
        monkeypatch,
        project=fixture["project"],
        schema=fixture["schema"],
        version=fixture["requested_version"],
        fields=fixture["fields"],
        active_versions=[fixture["requested_version"]],
    )

    snapshot = run_async(
        projection_service.get_dynamic_schema_definition_snapshot(
            factory,
            project_id=fixture["project"].id,
            schema_id=fixture["schema"].id,
            schema_version_id=fixture["requested_version"].id,
        )
    )

    with pytest.raises(FrozenInstanceError):
        snapshot.name = "mutated"
    assert isinstance(snapshot.fields, tuple)
    assert not hasattr(snapshot, "_sa_instance_state")
    assert not hasattr(snapshot.fields[0], "_sa_instance_state")


def test_get_dynamic_schema_definition_snapshot_rolls_back_without_commit_or_for_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(requested_status="draft")
    factory = SessionFactory()
    _install_repository(
        monkeypatch,
        project=fixture["project"],
        schema=fixture["schema"],
        version=fixture["requested_version"],
        fields=fixture["fields"],
        active_versions=[],
    )

    run_async(
        projection_service.get_dynamic_schema_definition_snapshot(
            factory,
            project_id=fixture["project"].id,
            schema_id=fixture["schema"].id,
            schema_version_id=fixture["requested_version"].id,
        )
    )

    assert len(factory.sessions) == 1
    assert factory.sessions[0].commit_count == 0
    assert factory.sessions[0].rollback_count == 1

    repository_source = inspect.getsource(projection_service.projection_repository)
    service_source = inspect.getsource(projection_service)
    assert "with_for_update" not in repository_source
    assert "app.services.llm" not in service_source
