from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import json
from types import SimpleNamespace
import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from app.models.project import ProjectStatus
from app.models.project_version import ProjectVersion
from app.schemas.project_version import ProjectVersionCreateResult
from app.schemas.dynamic_schema_knowledge_view import (
    DynamicSchemaKnowledgeField,
    DynamicSchemaKnowledgeRecord,
    DynamicSchemaKnowledgeSection,
    DynamicSchemaKnowledgeView,
)
from app.schemas.dynamic_schema_review_projection import DynamicSchemaReviewedFact
from app.schemas.dynamic_schema_ufl_projection import DynamicSchemaUFLProjectedField
from app.schemas.ufl_fact_snapshot import (
    UFLFactEvidenceLocator,
    UFLFactEvidenceSnapshot,
    UFLFactSnapshot,
    UFLFactValueGroupSnapshot,
    UFLFactValueSnapshot,
)
from app.services import project_version as project_version_service
from app.utils.deterministic_json import freeze_deterministic_json_value


def run_async(awaitable):
    return asyncio.run(awaitable)


def _uuid(seed: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"project-version-service:{seed}")


def _hash(seed: str) -> str:
    return project_version_service.duplicate_grouping_service.hash_deterministic_payload(
        {"seed": seed}
    )


def _integrity_error(constraint_name: str) -> IntegrityError:
    return IntegrityError(
        statement=None,
        params=None,
        orig=SimpleNamespace(diag=SimpleNamespace(constraint_name=constraint_name)),
    )


class FakeSession:
    def __init__(self, store: "Store") -> None:
        self.store = store
        self.commit_count = 0
        self.rollback_count = 0
        self.flush_count = 0
        self.pending_versions: list[ProjectVersion] = []
        self.created_orm_versions: list[ProjectVersion] = []
        self.project = self._clone_project(store.project)

    @staticmethod
    def _clone_project(project: SimpleNamespace | None) -> SimpleNamespace | None:
        if project is None:
            return None
        return SimpleNamespace(
            id=project.id,
            status=project.status,
            current_version_id=project.current_version_id,
        )

    async def commit(self) -> None:
        self.commit_count += 1
        if self.store.on_commit is not None:
            self.store.on_commit(self)
        if self.store.commit_error is not None:
            raise self.store.commit_error
        self.store.versions.extend(self.pending_versions)
        self.pending_versions = []
        if self.project is not None and self.store.project is not None:
            self.store.project.current_version_id = self.project.current_version_id

    async def rollback(self) -> None:
        self.rollback_count += 1
        self.pending_versions = []
        self.project = self._clone_project(self.store.project)

    async def flush(self) -> None:
        self.flush_count += 1


class SessionFactory:
    def __init__(self, store: "Store") -> None:
        self.store = store
        self.sessions: list[FakeSession] = []
        self.open_count = 0

    def __call__(self):
        factory = self

        class _Context:
            async def __aenter__(self_inner):
                factory.open_count += 1
                session = FakeSession(factory.store)
                factory.sessions.append(session)
                return session

            async def __aexit__(self_inner, exc_type, exc, tb):
                factory.open_count -= 1
                return False

        return _Context()


class Store:
    def __init__(self) -> None:
        self.project = SimpleNamespace(
            id=_uuid("project"),
            status=ProjectStatus.ACTIVE.value,
            current_version_id=None,
        )
        self.users: dict[uuid.UUID, object] = {
            _uuid("creator"): SimpleNamespace(id=_uuid("creator")),
            _uuid("creator-2"): SimpleNamespace(id=_uuid("creator-2")),
        }
        self.versions: list[ProjectVersion] = []
        self.call_log: list[str] = []
        self.build_calls: list[dict[str, object]] = []
        self.auth_calls: list[dict[str, object]] = []
        self.knowledge_view = _build_knowledge_view()
        self.create_error: Exception | None = None
        self.commit_error: Exception | None = None
        self.on_create_version = None
        self.on_commit = None


def _clone_project_version_row(project_version: ProjectVersion) -> SimpleNamespace:
    payload = {
        column.name: getattr(project_version, column.name)
        for column in ProjectVersion.__table__.columns
    }
    payload["snapshot_json"] = json.loads(
        project_version_service.duplicate_grouping_service.canonicalize_deterministic_payload(
            payload["snapshot_json"]
        ).decode("utf-8")
    )
    return SimpleNamespace(**payload)


def _build_knowledge_view() -> DynamicSchemaKnowledgeView:
    fact_value_id = _uuid("fact-value")
    fact = UFLFactSnapshot(
        fact_id=_uuid("fact"),
        identity_hash=_hash("fact-identity"),
        subject_kind="person",
        subject_key="alpha",
        subject_entity_id=None,
        predicate_key="title",
        scope_key=None,
        semantic_group_count=1,
        fact_value_count=1,
        value_groups=(
            UFLFactValueGroupSnapshot(
                semantic_key_hash=_hash("semantic"),
                value_type="string",
                value_json=freeze_deterministic_json_value({"text": "Alice"}),
                referenced_entity_id=None,
                fact_value_ids=(fact_value_id,),
                values=(
                    UFLFactValueSnapshot(
                        fact_value_id=fact_value_id,
                        source_batch_id=_uuid("batch"),
                        source_application_id=_uuid("source-application"),
                        proposal_index=0,
                        normalized_value_text="Alice",
                        value_hash=_hash("value"),
                        language_code="zh",
                        confidence=0.9,
                    ),
                ),
                evidences=(
                    UFLFactEvidenceSnapshot(
                        evidence_link_id=_uuid("evidence-link"),
                        evidence_id=_uuid("evidence"),
                        document_revision_id=_uuid("document-revision"),
                        document_block_id=_uuid("document-block"),
                        locator=UFLFactEvidenceLocator(
                            location_key="loc:alpha",
                            page_no=1,
                            start_line=2,
                            end_line=2,
                            table_index=None,
                            row_index=None,
                        ),
                        excerpt="Alice excerpt",
                        excerpt_hash=_hash("excerpt"),
                        content_hash=_hash("content"),
                        role="supporting",
                        is_primary=True,
                        source_order=0,
                    ),
                ),
            ),
        ),
    )
    source_field = DynamicSchemaUFLProjectedField(
        field_id=_uuid("field"),
        schema_version_id=_uuid("schema-version"),
        field_key="title",
        label="Title",
        description=None,
        predicate_key="title",
        scope_key=None,
        expected_value_type="string",
        cardinality="one",
        is_required=False,
        is_title=True,
        is_summary=False,
        is_hidden=False,
        group_key=None,
        display_order=0,
        display_config=freeze_deterministic_json_value({"kind": "text"}),
        validation_rules=freeze_deterministic_json_value({"max_length": 100}),
        created_at=datetime(2026, 8, 3, 10, 0, tzinfo=timezone.utc),
        matched_facts=(fact,),
        matched_fact_count=1,
        semantic_value_count=1,
        is_missing=False,
        type_compatible=True,
        issues=(),
    )
    reviewed_fact = DynamicSchemaReviewedFact(
        fact=fact,
        review_state="resolved",
        candidate_id=_uuid("candidate"),
        assessment_id=_uuid("assessment"),
        resolution_basis="human_selection",
        current_decision_id=_uuid("decision"),
        current_decision_kind="select_one",
        effective_fact_value_ids=(fact_value_id,),
        requires_review=False,
    )
    field = DynamicSchemaKnowledgeField(
        source_field=source_field,
        reviewed_facts=(reviewed_fact,),
        knowledge_state="resolved",
        effective_fact_value_ids=(fact_value_id,),
        observed_fact_value_count=1,
        semantic_value_count=1,
        has_schema_issues=False,
    )
    section = DynamicSchemaKnowledgeSection(
        group_key=None,
        display_order=0,
        fields=(field,),
    )
    record = DynamicSchemaKnowledgeRecord(
        subject_key="alpha",
        title_field_key="title",
        has_review_required=False,
        issue_count=0,
        sections=(section,),
    )
    return DynamicSchemaKnowledgeView(
        project_id=_uuid("project"),
        schema_id=_uuid("schema"),
        schema_version_id=_uuid("schema-version"),
        orchestration_id=_uuid("orchestration"),
        extraction_run_id=_uuid("extraction-run"),
        consistency_check_application_id=_uuid("consistency-check-application"),
        source_consistency_application_id=_uuid("source-consistency-application"),
        schema_definition_manifest_hash=_hash("schema-manifest"),
        ufl_source_manifest_hash=_hash("ufl-manifest"),
        consistency_result_manifest_hash=_hash("consistency-manifest"),
        raw_projection_manifest_hash=_hash("raw-manifest"),
        reviewed_projection_manifest_hash=_hash("reviewed-manifest"),
        comparison_quality="complete",
        algorithm_name=project_version_service.knowledge_view_service.DYNAMIC_SCHEMA_KNOWLEDGE_VIEW_ALGORITHM_NAME,
        algorithm_version=project_version_service.knowledge_view_service.DYNAMIC_SCHEMA_KNOWLEDGE_VIEW_ALGORITHM_VERSION,
        record_count=1,
        section_count=1,
        field_count=1,
        missing_field_count=0,
        review_required_field_count=0,
        resolved_field_count=1,
        observation_only_field_count=0,
        mixed_field_count=0,
        records=(record,),
        knowledge_view_manifest_hash=_hash("knowledge-view-manifest"),
    )


def _base_kwargs(store: Store) -> dict[str, object]:
    project_id = store.project.id if store.project is not None else _uuid("project")
    return {
        "project_version_id": _uuid("project-version-1"),
        "project_id": project_id,
        "schema_id": store.knowledge_view.schema_id,
        "schema_version_id": store.knowledge_view.schema_version_id,
        "orchestration_id": store.knowledge_view.orchestration_id,
        "consistency_check_application_id": (
            store.knowledge_view.consistency_check_application_id
        ),
        "created_by_id": _uuid("creator"),
        "creation_kind": "manual",
        "reason": "initial snapshot",
    }


def _thaw_snapshot_json(snapshot_json: object) -> dict[str, object]:
    return json.loads(
        project_version_service.duplicate_grouping_service.canonicalize_deterministic_payload(
            snapshot_json
        ).decode("utf-8")
    )


def _mutable_snapshot(
    snapshot: project_version_service.ProjectVersionSnapshot,
) -> project_version_service.ProjectVersionSnapshot:
    return replace(snapshot, snapshot_json=_thaw_snapshot_json(snapshot.snapshot_json))


def _install_repository_patches(
    monkeypatch: pytest.MonkeyPatch,
    store: Store,
) -> None:
    async def fake_get_project_for_update(session: FakeSession, *, project_id):
        store.call_log.append("project_lock")
        if session.project is None or session.project.id != project_id:
            return None
        return session.project

    async def fake_get_project_by_id(_session: FakeSession, *, project_id):
        store.call_log.append("project_read")
        if store.project is None or store.project.id != project_id:
            return None
        return SimpleNamespace(
            id=store.project.id,
            status=store.project.status,
            current_version_id=store.project.current_version_id,
        )

    async def fake_get_user_by_id(_session: FakeSession, *, user_id):
        store.call_log.append("user_read")
        return store.users.get(user_id)

    async def fake_get_project_version_by_id(_session: FakeSession, *, project_version_id):
        store.call_log.append("version_by_id")
        for version in store.versions:
            if version.id == project_version_id:
                return version
        return None

    async def fake_get_project_version_for_project(
        _session: FakeSession,
        *,
        project_id,
        project_version_id,
    ):
        store.call_log.append("version_by_project")
        for version in store.versions:
            if version.project_id == project_id and version.id == project_version_id:
                return version
        return None

    async def fake_get_max_project_version_no(_session: FakeSession, *, project_id):
        store.call_log.append("max_version_no")
        matching = [
            version.version_no for version in store.versions if version.project_id == project_id
        ]
        return max(matching, default=0)

    async def fake_create_project_version(session: FakeSession, project_version: ProjectVersion):
        store.call_log.append("create_project_version")
        session.pending_versions.append(_clone_project_version_row(project_version))
        session.created_orm_versions.append(project_version)
        await session.flush()
        if store.on_create_version is not None:
            store.on_create_version(project_version, session)
        if store.create_error is not None:
            raise store.create_error
        return project_version

    monkeypatch.setattr(
        project_version_service.project_version_repository,
        "get_project_for_update",
        fake_get_project_for_update,
    )
    monkeypatch.setattr(
        project_version_service.project_version_repository,
        "get_project_by_id",
        fake_get_project_by_id,
    )
    monkeypatch.setattr(
        project_version_service.project_version_repository,
        "get_user_by_id",
        fake_get_user_by_id,
    )
    monkeypatch.setattr(
        project_version_service.project_version_repository,
        "get_project_version_by_id",
        fake_get_project_version_by_id,
    )
    monkeypatch.setattr(
        project_version_service.project_version_repository,
        "get_project_version_for_project",
        fake_get_project_version_for_project,
    )
    monkeypatch.setattr(
        project_version_service.project_version_repository,
        "get_max_project_version_no",
        fake_get_max_project_version_no,
    )
    monkeypatch.setattr(
        project_version_service.project_version_repository,
        "create_project_version",
        fake_create_project_version,
    )


def _install_knowledge_view_patches(
    monkeypatch: pytest.MonkeyPatch,
    store: Store,
) -> None:
    async def fake_build_dynamic_schema_knowledge_view(
        _session_factory,
        *,
        project_id,
        schema_id,
        schema_version_id,
        orchestration_id,
        consistency_check_application_id,
        subject_keys=None,
    ):
        store.build_calls.append(
            {
                "project_id": project_id,
                "schema_id": schema_id,
                "schema_version_id": schema_version_id,
                "orchestration_id": orchestration_id,
                "consistency_check_application_id": consistency_check_application_id,
                "subject_keys": subject_keys,
            }
        )
        return store.knowledge_view

    def fake_authenticate_dynamic_schema_knowledge_view(view, *, subject_keys):
        store.auth_calls.append({"subject_keys": subject_keys, "view": view})
        return view

    monkeypatch.setattr(
        project_version_service.knowledge_view_service,
        "build_dynamic_schema_knowledge_view",
        fake_build_dynamic_schema_knowledge_view,
    )
    monkeypatch.setattr(
        project_version_service.knowledge_view_service,
        "authenticate_dynamic_schema_knowledge_view",
        fake_authenticate_dynamic_schema_knowledge_view,
    )


def _install_all_patches(
    monkeypatch: pytest.MonkeyPatch,
    store: Store,
) -> None:
    _install_repository_patches(monkeypatch, store)
    _install_knowledge_view_patches(monkeypatch, store)


def test_create_project_version_assigns_strictly_increasing_versions_and_locks_before_max(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = Store()
    session_factory = SessionFactory(store)
    _install_all_patches(monkeypatch, store)
    first_kwargs = _base_kwargs(store)
    second_kwargs = {**_base_kwargs(store), "project_version_id": _uuid("project-version-2")}

    first = run_async(project_version_service.create_project_version(session_factory, **first_kwargs))
    second = run_async(project_version_service.create_project_version(session_factory, **second_kwargs))

    assert first.created_new is True
    assert first.version_no == 1
    assert first.is_current is True
    assert second.created_new is True
    assert second.version_no == 2
    assert second.is_current is True
    assert store.project.current_version_id == second.id
    assert first.snapshot_json_hash == second.snapshot_json_hash
    assert first.version_manifest_hash != second.version_manifest_hash
    assert store.call_log.index("project_lock") < store.call_log.index("max_version_no")
    assert store.build_calls[0]["subject_keys"] is None
    assert all(call["subject_keys"] is None for call in store.auth_calls)


def test_create_project_version_authenticates_before_commit_and_returns_precommitted_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = Store()
    session_factory = SessionFactory(store)
    _install_all_patches(monkeypatch, store)
    call_order: list[str] = []
    original_authenticate = project_version_service.authenticate_project_version_snapshot
    sentinel_result: ProjectVersionCreateResult | None = None

    def on_commit(_session: FakeSession) -> None:
        call_order.append("commit")

    def tracking_authenticate(snapshot):
        nonlocal sentinel_result
        if isinstance(snapshot, ProjectVersionCreateResult) and snapshot.created_new is True:
            call_order.append("authenticate")
            assert session_factory.sessions[-1].commit_count == 0
            sentinel_result = replace(original_authenticate(snapshot), is_current=False)
            return sentinel_result
        return original_authenticate(snapshot)

    store.on_commit = on_commit
    monkeypatch.setattr(
        project_version_service,
        "authenticate_project_version_snapshot",
        tracking_authenticate,
    )

    created = run_async(
        project_version_service.create_project_version(session_factory, **_base_kwargs(store))
    )

    assert sentinel_result is not None
    assert created is sentinel_result
    assert call_order == ["authenticate", "commit"]


@pytest.mark.parametrize(
    ("mutate_store", "error_code"),
    [
        (
            lambda store: setattr(store, "project", None),
            "project_version_project_not_found",
        ),
        (
            lambda store: setattr(
                store.project,
                "status",
                ProjectStatus.ARCHIVED.value,
            ),
            "project_version_project_archived",
        ),
        (
            lambda store: store.users.pop(_uuid("creator")),
            "project_version_created_by_not_found",
        ),
    ],
)
def test_create_project_version_rejects_missing_archived_project_and_unknown_user(
    monkeypatch: pytest.MonkeyPatch,
    mutate_store,
    error_code: str,
) -> None:
    store = Store()
    session_factory = SessionFactory(store)
    _install_all_patches(monkeypatch, store)
    mutate_store(store)

    with pytest.raises(project_version_service.ProjectVersionStateError, match=error_code):
        run_async(project_version_service.create_project_version(session_factory, **_base_kwargs(store)))


def test_create_project_version_rejects_rollback_creation_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = Store()
    session_factory = SessionFactory(store)
    _install_all_patches(monkeypatch, store)
    kwargs = {**_base_kwargs(store), "creation_kind": "rollback"}

    with pytest.raises(
        project_version_service.ProjectVersionStateError,
        match="project_version_creation_kind_unsupported",
    ):
        run_async(project_version_service.create_project_version(session_factory, **kwargs))


def test_create_project_version_idempotent_retry_does_not_rewind_current_pointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = Store()
    session_factory = SessionFactory(store)
    _install_all_patches(monkeypatch, store)
    first_kwargs = _base_kwargs(store)
    second_kwargs = {**_base_kwargs(store), "project_version_id": _uuid("project-version-2")}

    first = run_async(project_version_service.create_project_version(session_factory, **first_kwargs))
    second = run_async(project_version_service.create_project_version(session_factory, **second_kwargs))
    retried = run_async(project_version_service.create_project_version(session_factory, **first_kwargs))

    assert second.id == store.project.current_version_id
    assert retried.created_new is False
    assert retried.id == first.id
    assert retried.version_no == 1
    assert retried.is_current is False
    assert store.project.current_version_id == second.id


@pytest.mark.parametrize(
    "mutated_kwargs",
    [
        {"reason": "changed reason"},
        {"created_by_id": _uuid("creator-2")},
        {"schema_id": _uuid("other-schema")},
    ],
)
def test_create_project_version_rejects_same_id_when_request_changes(
    monkeypatch: pytest.MonkeyPatch,
    mutated_kwargs: dict[str, object],
) -> None:
    store = Store()
    session_factory = SessionFactory(store)
    _install_all_patches(monkeypatch, store)
    base = _base_kwargs(store)
    run_async(project_version_service.create_project_version(session_factory, **base))

    with pytest.raises(
        project_version_service.ProjectVersionInvariantError,
        match="project_version_idempotency_mismatch",
    ):
        run_async(
            project_version_service.create_project_version(
                session_factory,
                **{**base, **mutated_kwargs},
            )
        )


def test_create_project_version_recovers_same_request_after_identity_integrity_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = Store()
    session_factory = SessionFactory(store)
    _install_all_patches(monkeypatch, store)

    def concurrent_winner(project_version: ProjectVersion, _session: FakeSession) -> None:
        if store.versions:
            return
        store.versions.append(_clone_project_version_row(project_version))
        store.project.current_version_id = project_version.id
        store.create_error = _integrity_error("pk_project_versions")

    store.on_create_version = concurrent_winner
    result = run_async(
        project_version_service.create_project_version(session_factory, **_base_kwargs(store))
    )

    assert result.created_new is False
    assert result.version_no == 1
    assert len(store.versions) == 1
    assert session_factory.sessions[0].rollback_count == 1
    assert session_factory.sessions[1].commit_count == 0
    assert session_factory.sessions[1].rollback_count == 1


def test_create_project_version_allows_different_ids_with_same_snapshot_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = Store()
    session_factory = SessionFactory(store)
    _install_all_patches(monkeypatch, store)
    first = run_async(project_version_service.create_project_version(session_factory, **_base_kwargs(store)))
    second = run_async(
        project_version_service.create_project_version(
            session_factory,
            **{**_base_kwargs(store), "project_version_id": _uuid("project-version-2")},
        )
    )

    assert first.id != second.id
    assert first.snapshot_json_hash == second.snapshot_json_hash
    assert first.version_no == 1
    assert second.version_no == 2


def test_create_project_version_rolls_back_version_and_pointer_on_write_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = Store()
    session_factory = SessionFactory(store)
    _install_all_patches(monkeypatch, store)
    store.create_error = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_async(project_version_service.create_project_version(session_factory, **_base_kwargs(store)))

    assert store.versions == []
    assert store.project.current_version_id is None
    assert session_factory.sessions[0].rollback_count == 1


def test_create_project_version_rolls_back_when_precommit_authentication_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = Store()
    session_factory = SessionFactory(store)
    _install_all_patches(monkeypatch, store)
    original_authenticate = project_version_service.authenticate_project_version_snapshot

    def failing_authenticate(snapshot):
        if isinstance(snapshot, ProjectVersionCreateResult) and snapshot.created_new is True:
            raise project_version_service.ProjectVersionInvariantError(
                "project_version_snapshot_invalid"
            )
        return original_authenticate(snapshot)

    monkeypatch.setattr(
        project_version_service,
        "authenticate_project_version_snapshot",
        failing_authenticate,
    )

    with pytest.raises(
        project_version_service.ProjectVersionInvariantError,
        match="project_version_snapshot_invalid",
    ):
        run_async(
            project_version_service.create_project_version(
                session_factory,
                **_base_kwargs(store),
            )
        )

    assert session_factory.sessions[0].commit_count == 0
    assert session_factory.sessions[0].rollback_count == 1
    assert store.versions == []
    assert store.project.current_version_id is None


def test_create_project_version_sanitizes_unknown_integrity_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = Store()
    session_factory = SessionFactory(store)
    _install_all_patches(monkeypatch, store)
    store.create_error = _integrity_error("unknown_constraint")

    with pytest.raises(
        project_version_service.ProjectVersionInvariantError,
        match="project_version_write_integrity_error",
    ):
        run_async(project_version_service.create_project_version(session_factory, **_base_kwargs(store)))


def test_create_project_version_does_not_access_orm_or_reauthenticate_after_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = Store()
    session_factory = SessionFactory(store)
    _install_all_patches(monkeypatch, store)
    after_commit = False
    original_authenticate = project_version_service.authenticate_project_version_snapshot
    original_build_snapshot = project_version_service._build_snapshot_from_row

    def on_commit(_session: FakeSession) -> None:
        nonlocal after_commit
        after_commit = True

    def tracking_authenticate(snapshot):
        if after_commit:
            raise AssertionError("post_commit_authenticate_called")
        return original_authenticate(snapshot)

    def tracking_build_snapshot(row, *, is_current):
        if after_commit:
            raise AssertionError("post_commit_orm_access")
        return original_build_snapshot(row, is_current=is_current)

    store.on_commit = on_commit
    monkeypatch.setattr(
        project_version_service,
        "authenticate_project_version_snapshot",
        tracking_authenticate,
    )
    monkeypatch.setattr(
        project_version_service,
        "_build_snapshot_from_row",
        tracking_build_snapshot,
    )

    created = run_async(
        project_version_service.create_project_version(session_factory, **_base_kwargs(store))
    )

    assert created.created_new is True
    assert after_commit is True


def test_create_project_version_recovers_when_commit_raises_integrity_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = Store()
    session_factory = SessionFactory(store)
    _install_all_patches(monkeypatch, store)

    def on_commit(session: FakeSession) -> None:
        if store.versions:
            return
        store.versions.extend(session.pending_versions)
        store.project.current_version_id = session.project.current_version_id

    store.on_commit = on_commit
    store.commit_error = _integrity_error("pk_project_versions")

    recovered = run_async(
        project_version_service.create_project_version(session_factory, **_base_kwargs(store))
    )

    assert recovered.created_new is False
    assert recovered.version_no == 1
    assert recovered.snapshot_json_hash
    assert session_factory.sessions[0].rollback_count == 1


def test_create_project_version_rolls_back_when_commit_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = Store()
    session_factory = SessionFactory(store)
    _install_all_patches(monkeypatch, store)
    store.commit_error = RuntimeError("commit boom")

    with pytest.raises(RuntimeError, match="commit boom"):
        run_async(
            project_version_service.create_project_version(
                session_factory,
                **_base_kwargs(store),
            )
        )

    assert session_factory.sessions[0].commit_count == 1
    assert session_factory.sessions[0].rollback_count == 1
    assert store.versions == []
    assert store.project.current_version_id is None


def test_get_project_version_snapshot_reads_exact_pair_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = Store()
    session_factory = SessionFactory(store)
    _install_all_patches(monkeypatch, store)
    created = run_async(project_version_service.create_project_version(session_factory, **_base_kwargs(store)))
    store.call_log.clear()

    snapshot = run_async(
        project_version_service.get_project_version_snapshot(
            session_factory,
            project_id=store.project.id,
            project_version_id=created.id,
        )
    )

    assert snapshot.id == created.id
    assert snapshot.is_current is True
    assert "project_read" in store.call_log
    assert "version_by_project" in store.call_log
    assert session_factory.sessions[-1].commit_count == 0
    assert session_factory.sessions[-1].rollback_count == 1


def test_get_project_version_snapshot_rejects_cross_project_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = Store()
    session_factory = SessionFactory(store)
    _install_all_patches(monkeypatch, store)
    created = run_async(project_version_service.create_project_version(session_factory, **_base_kwargs(store)))

    with pytest.raises(project_version_service.ProjectVersionStateError, match="project_version_not_found"):
        run_async(
            project_version_service.get_project_version_snapshot(
                session_factory,
                project_id=_uuid("other-project"),
                project_version_id=created.id,
            )
        )


def test_snapshot_json_preserves_fact_values_and_evidence_and_is_deeply_immutable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = Store()
    session_factory = SessionFactory(store)
    source_excerpt = (
        store.knowledge_view.records[0]
        .sections[0]
        .fields[0]
        .reviewed_facts[0]
        .fact.value_groups[0]
        .evidences[0]
        .excerpt
    )
    _install_all_patches(monkeypatch, store)

    created = run_async(project_version_service.create_project_version(session_factory, **_base_kwargs(store)))
    record = created.snapshot_json["records"][0]
    section = record["sections"][0]
    field = section["fields"][0]
    reviewed_fact = field["reviewed_facts"][0]
    value_group = reviewed_fact["fact"]["value_groups"][0]
    value = value_group["values"][0]
    evidence = value_group["evidences"][0]

    assert value["fact_value_id"] == str(_uuid("fact-value"))
    assert value["normalized_value_text"] == "Alice"
    assert evidence["evidence_id"] == str(_uuid("evidence"))
    assert evidence["excerpt"] == "Alice excerpt"
    assert evidence["locator"]["location_key"] == "loc:alpha"
    assert (
        store.knowledge_view.records[0]
        .sections[0]
        .fields[0]
        .reviewed_facts[0]
        .fact.value_groups[0]
        .evidences[0]
        .excerpt
        == source_excerpt
    )
    with pytest.raises(TypeError):
        created.snapshot_json["new_key"] = "blocked"  # type: ignore[index]
    with pytest.raises(TypeError):
        created.snapshot_json["records"][0] = "blocked"  # type: ignore[index]


def test_create_project_version_stores_plain_json_payload_for_jsonb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = Store()
    session_factory = SessionFactory(store)
    _install_all_patches(monkeypatch, store)

    run_async(project_version_service.create_project_version(session_factory, **_base_kwargs(store)))

    orm_version = session_factory.sessions[0].created_orm_versions[0]
    source_field = orm_version.snapshot_json["records"][0]["sections"][0]["fields"][0]["source_field"]
    value_group = (
        orm_version.snapshot_json["records"][0]["sections"][0]["fields"][0]["reviewed_facts"][0]["fact"]["value_groups"][0]
    )

    assert isinstance(orm_version.snapshot_json, dict)
    assert isinstance(source_field["display_config"], dict)
    assert isinstance(source_field["validation_rules"], dict)
    assert source_field["display_config"] == {"kind": "text"}
    assert source_field["validation_rules"] == {"max_length": 100}
    assert isinstance(value_group["value_json"], dict)
    assert value_group["value_json"] == {"text": "Alice"}


def test_authenticate_project_version_snapshot_returns_frozen_copy_for_mutable_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = Store()
    session_factory = SessionFactory(store)
    _install_all_patches(monkeypatch, store)
    created = run_async(
        project_version_service.create_project_version(session_factory, **_base_kwargs(store))
    )
    mutable_input = _mutable_snapshot(created)
    original_snapshot_json = mutable_input.snapshot_json

    authenticated = project_version_service.authenticate_project_version_snapshot(
        mutable_input
    )

    assert isinstance(authenticated, ProjectVersionCreateResult)
    assert authenticated.created_new is True
    assert authenticated.snapshot_json_hash == created.snapshot_json_hash
    assert authenticated.version_manifest_hash == created.version_manifest_hash
    assert authenticated.snapshot_json is not original_snapshot_json
    mutable_input.snapshot_json["project_id"] = str(_uuid("mutated-project"))  # type: ignore[index]
    mutable_input.snapshot_json["records"][0]["sections"][0]["fields"][0]["reviewed_facts"][0]["fact"]["value_groups"][0]["values"][0]["normalized_value_text"] = "mutated"  # type: ignore[index]
    assert authenticated.snapshot_json["project_id"] == str(created.project_id)
    assert (
        authenticated.snapshot_json["records"][0]["sections"][0]["fields"][0]["reviewed_facts"][0]["fact"]["value_groups"][0]["values"][0]["normalized_value_text"]  # type: ignore[index]
        == "Alice"
    )
    with pytest.raises(TypeError):
        authenticated.snapshot_json["project_id"] = "blocked"  # type: ignore[index]


def test_authenticate_project_version_snapshot_rejects_non_bool_created_new(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = Store()
    session_factory = SessionFactory(store)
    _install_all_patches(monkeypatch, store)
    created = run_async(
        project_version_service.create_project_version(session_factory, **_base_kwargs(store))
    )
    mutable_input = replace(
        _mutable_snapshot(created),
        created_new=1,  # type: ignore[arg-type]
    )

    with pytest.raises(
        project_version_service.ProjectVersionInvariantError,
        match="project_version_snapshot_invalid",
    ):
        project_version_service.authenticate_project_version_snapshot(mutable_input)


@pytest.mark.parametrize(
    "mutate_snapshot",
    [
        lambda snapshot_json: snapshot_json.pop("project_id"),
        lambda snapshot_json: snapshot_json["records"][0]["sections"][0]["fields"][0]["reviewed_facts"][0]["fact"].pop("fact_id"),
        lambda snapshot_json: snapshot_json["records"][0]["sections"][0]["fields"][0]["reviewed_facts"][0]["fact"]["value_groups"][0]["values"][0].pop("normalized_value_text"),
        lambda snapshot_json: snapshot_json["records"][0]["sections"][0]["fields"][0]["reviewed_facts"][0]["fact"]["value_groups"][0]["evidences"][0].pop("excerpt"),
    ],
)
def test_authenticate_project_version_snapshot_maps_missing_keys_to_fixed_error(
    monkeypatch: pytest.MonkeyPatch,
    mutate_snapshot,
) -> None:
    store = Store()
    session_factory = SessionFactory(store)
    _install_all_patches(monkeypatch, store)
    created = run_async(
        project_version_service.create_project_version(session_factory, **_base_kwargs(store))
    )
    mutated_snapshot_json = _thaw_snapshot_json(created.snapshot_json)
    mutate_snapshot(mutated_snapshot_json)

    with pytest.raises(
        project_version_service.ProjectVersionInvariantError,
        match="project_version_snapshot_invalid",
    ):
        project_version_service.authenticate_project_version_snapshot(
            replace(created, snapshot_json=mutated_snapshot_json)
        )


@pytest.mark.parametrize(
    "mutate_snapshot",
    [
        lambda created, snapshot_json: snapshot_json.__setitem__("records", {"not": "a-list"}),
        lambda created, snapshot_json: snapshot_json.__setitem__("project_id", "bad-uuid"),
        lambda created, snapshot_json: snapshot_json["records"][0]["sections"][0]["fields"][0]["source_field"].__setitem__("created_at", "bad-datetime"),
        lambda created, snapshot_json: snapshot_json["records"][0]["sections"][0]["fields"][0]["reviewed_facts"][0]["fact"].__setitem__("semantic_group_count", "bad-int"),
        lambda created, snapshot_json: replace(created, snapshot_json=snapshot_json, record_count=True),
    ],
)
def test_authenticate_project_version_snapshot_maps_malformed_shapes_to_fixed_error(
    monkeypatch: pytest.MonkeyPatch,
    mutate_snapshot,
) -> None:
    store = Store()
    session_factory = SessionFactory(store)
    _install_all_patches(monkeypatch, store)
    created = run_async(
        project_version_service.create_project_version(session_factory, **_base_kwargs(store))
    )
    mutated_snapshot_json = _thaw_snapshot_json(created.snapshot_json)
    mutated = mutate_snapshot(created, mutated_snapshot_json)
    mutated_snapshot = (
        mutated
        if isinstance(mutated, project_version_service.ProjectVersionSnapshot)
        else replace(created, snapshot_json=mutated_snapshot_json)
    )

    with pytest.raises(
        project_version_service.ProjectVersionInvariantError,
        match="project_version_snapshot_invalid",
    ):
        project_version_service.authenticate_project_version_snapshot(mutated_snapshot)


def test_authenticate_project_version_snapshot_redacts_sensitive_sentinel_in_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = Store()
    session_factory = SessionFactory(store)
    _install_all_patches(monkeypatch, store)
    created = run_async(
        project_version_service.create_project_version(session_factory, **_base_kwargs(store))
    )
    sentinel = "SENSITIVE_SENTINEL_13_2_1"
    mutated_snapshot_json = _thaw_snapshot_json(created.snapshot_json)
    mutated_snapshot_json["records"][0]["sections"][0]["fields"][0]["reviewed_facts"][0]["fact"]["value_groups"][0]["evidences"][0]["excerpt"] = sentinel
    mutated_snapshot_json.pop("project_id")

    with pytest.raises(project_version_service.ProjectVersionInvariantError) as exc_info:
        project_version_service.authenticate_project_version_snapshot(
            replace(created, snapshot_json=mutated_snapshot_json, reason=sentinel)
        )

    assert str(exc_info.value) == "project_version_snapshot_invalid"
    assert sentinel not in str(exc_info.value)


def test_create_get_idempotent_and_recovery_paths_use_authenticated_return_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_authenticate = project_version_service.authenticate_project_version_snapshot

    def tracking_authenticate(snapshot):
        authenticated = original_authenticate(snapshot)
        return replace(authenticated, is_current=False)

    monkeypatch.setattr(
        project_version_service,
        "authenticate_project_version_snapshot",
        tracking_authenticate,
    )

    create_store = Store()
    create_session_factory = SessionFactory(create_store)
    _install_all_patches(monkeypatch, create_store)
    created = run_async(
        project_version_service.create_project_version(
            create_session_factory,
            **_base_kwargs(create_store),
        )
    )
    fetched = run_async(
        project_version_service.get_project_version_snapshot(
            create_session_factory,
            project_id=create_store.project.id,
            project_version_id=created.id,
        )
    )
    retried = run_async(
        project_version_service.create_project_version(
            create_session_factory,
            **_base_kwargs(create_store),
        )
    )

    recovery_store = Store()
    recovery_session_factory = SessionFactory(recovery_store)
    _install_all_patches(monkeypatch, recovery_store)

    def concurrent_winner(project_version: ProjectVersion, _session: FakeSession) -> None:
        if recovery_store.versions:
            return
        recovery_store.versions.append(_clone_project_version_row(project_version))
        recovery_store.project.current_version_id = project_version.id
        recovery_store.create_error = _integrity_error("pk_project_versions")

    recovery_store.on_create_version = concurrent_winner
    recovered = run_async(
        project_version_service.create_project_version(
            recovery_session_factory,
            **_base_kwargs(recovery_store),
        )
    )

    assert created.is_current is False
    assert fetched.is_current is False
    assert retried.is_current is False
    assert recovered.is_current is False


def test_authenticate_project_version_snapshot_rejects_column_json_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = Store()
    session_factory = SessionFactory(store)
    _install_all_patches(monkeypatch, store)
    created = run_async(project_version_service.create_project_version(session_factory, **_base_kwargs(store)))

    with pytest.raises(
        project_version_service.ProjectVersionInvariantError,
        match="project_version_snapshot_invalid",
    ):
        project_version_service.authenticate_project_version_snapshot(
            replace(created, schema_id=_uuid("drifted-schema"))
        )


def test_authenticate_project_version_snapshot_rejects_resigned_snapshot_json_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = Store()
    session_factory = SessionFactory(store)
    _install_all_patches(monkeypatch, store)
    created = run_async(project_version_service.create_project_version(session_factory, **_base_kwargs(store)))
    mutated_snapshot_json = _thaw_snapshot_json(created.snapshot_json)
    mutated_snapshot_json["project_id"] = str(_uuid("drifted-project"))
    resigned_snapshot_json = freeze_deterministic_json_value(mutated_snapshot_json)
    resigned_snapshot_hash = project_version_service.duplicate_grouping_service.hash_deterministic_payload(
        mutated_snapshot_json
    )
    resigned_manifest_hash = project_version_service._build_version_manifest_hash(
        project_version_id=created.id,
        project_id=created.project_id,
        version_no=created.version_no,
        created_by_id=created.created_by_id,
        creation_kind=created.creation_kind,
        reason=created.reason,
        copied_from_version_id=created.copied_from_version_id,
        schema_id=created.schema_id,
        schema_version_id=created.schema_version_id,
        orchestration_id=created.orchestration_id,
        extraction_run_id=created.extraction_run_id,
        consistency_check_application_id=created.consistency_check_application_id,
        source_consistency_application_id=created.source_consistency_application_id,
        schema_definition_manifest_hash=created.schema_definition_manifest_hash,
        ufl_source_manifest_hash=created.ufl_source_manifest_hash,
        consistency_result_manifest_hash=created.consistency_result_manifest_hash,
        raw_projection_manifest_hash=created.raw_projection_manifest_hash,
        reviewed_projection_manifest_hash=created.reviewed_projection_manifest_hash,
        knowledge_view_manifest_hash=created.knowledge_view_manifest_hash,
        knowledge_view_algorithm_name=created.knowledge_view_algorithm_name,
        knowledge_view_algorithm_version=created.knowledge_view_algorithm_version,
        snapshot_json_hash=resigned_snapshot_hash,
        record_count=created.record_count,
        section_count=created.section_count,
        field_count=created.field_count,
        missing_field_count=created.missing_field_count,
        review_required_field_count=created.review_required_field_count,
        resolved_field_count=created.resolved_field_count,
        observation_only_field_count=created.observation_only_field_count,
        mixed_field_count=created.mixed_field_count,
        created_at=created.created_at,
    )

    with pytest.raises(
        project_version_service.ProjectVersionInvariantError,
        match="project_version_snapshot_invalid",
    ):
        project_version_service.authenticate_project_version_snapshot(
            replace(
                created,
                snapshot_json=resigned_snapshot_json,
                snapshot_json_hash=resigned_snapshot_hash,
                version_manifest_hash=resigned_manifest_hash,
            )
        )


def test_authenticate_project_version_snapshot_calls_public_knowledge_view_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = Store()
    session_factory = SessionFactory(store)
    _install_all_patches(monkeypatch, store)
    created = run_async(project_version_service.create_project_version(session_factory, **_base_kwargs(store)))
    calls: list[int] = []
    original_authenticate = (
        project_version_service.knowledge_view_service.authenticate_dynamic_schema_knowledge_view
    )

    def tracking_authenticate(view, *, subject_keys):
        calls.append(1)
        return original_authenticate(view, subject_keys=subject_keys)

    monkeypatch.setattr(
        project_version_service.knowledge_view_service,
        "authenticate_dynamic_schema_knowledge_view",
        tracking_authenticate,
    )

    authenticated = project_version_service.authenticate_project_version_snapshot(created)

    assert authenticated == created
    assert calls == [1, 1]
