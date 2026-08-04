from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timezone
import ast
from pathlib import Path
from types import SimpleNamespace
import uuid

from pydantic import SecretStr
import pytest
from starlette.testclient import TestClient

from app.core.config import Settings
from app.core.database import get_async_session_factory, get_db_session
from app.dependencies.auth import require_current_user
from app.models.project import Project, ProjectStatus, ProjectVisibility
from app.models.project_member import ProjectMember, ProjectMemberRole
from app.models.user import User, UserStatus
from app.repositories import project as project_repository
from app.schemas.review_query import (
    ReviewQueryItemDetailResult,
    ReviewQueryItemsResult,
    ReviewQueryItemSummary,
    VersionRecordQueryResult,
)
from app.services import bailian_review_tools as bailian_tools_service
from app.services import consistency_review as consistency_review_service
from app.services import frontend_api as frontend_api_service
from app.services import review_query as review_query_service
import app.dependencies.bailian_tools as bailian_tools_dependency
import app.main as main_module


REPO_ROOT = Path(__file__).resolve().parents[1]


def _uuid(seed: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"platform-boundaries:{seed}")


def _hash(seed: str) -> str:
    return review_query_service.duplicate_grouping_service.hash_deterministic_payload(
        {"seed": seed}
    )


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return imported


def _build_app(monkeypatch: pytest.MonkeyPatch, *, enabled: bool):
    settings = Settings(bailian_integration_enabled=enabled)
    monkeypatch.setattr(main_module, "settings", settings)
    return main_module.create_app()


def _build_user() -> User:
    now = datetime.now(timezone.utc)
    return User(
        id=_uuid("user"),
        handle="boundary-user",
        display_name="边界用户",
        email=None,
        avatar_url=None,
        locale="zh-CN",
        timezone="Asia/Shanghai",
        status=UserStatus.ACTIVE.value,
        created_at=now,
        updated_at=now,
    )


def _build_project(user_id: uuid.UUID) -> Project:
    now = datetime.now(timezone.utc)
    return Project(
        id=_uuid("project"),
        name="边界项目",
        slug="platform-boundary-project",
        description=None,
        visibility=ProjectVisibility.PRIVATE.value,
        status=ProjectStatus.ACTIVE.value,
        created_by_id=user_id,
        created_at=now,
        updated_at=now,
    )


def _build_membership(project: Project, user: User) -> ProjectMember:
    membership = ProjectMember(
        id=_uuid("membership"),
        project_id=project.id,
        user_id=user.id,
        role=ProjectMemberRole.EDITOR.value,
        joined_at=datetime.now(timezone.utc),
    )
    membership.project = project
    membership.user = user
    return membership


def _build_review_items_result() -> ReviewQueryItemsResult:
    payload = ReviewQueryItemsResult(
        source_manifest_hash=_hash("review-manifest"),
        payload_hash="",
        project_id=_uuid("project"),
        schema_id=_uuid("schema"),
        schema_version_id=_uuid("schema-version"),
        orchestration_id=_uuid("orchestration"),
        consistency_check_application_id=_uuid("consistency-app"),
        reviewed_projection_manifest_hash=_hash("review-manifest"),
        state="review_required",
        limit=20,
        item_count=1,
        items=(
            ReviewQueryItemSummary(
                fact_id=_uuid("fact"),
                subject_kind="person",
                subject_key="alpha",
                predicate_key="title",
                scope_key=None,
                matched_field_keys=("title",),
                review_state="pending_review",
                resolution_basis="none",
                requires_review=True,
                semantic_value_count=1,
                fact_value_count=2,
                evidence_count=1,
            ),
        ),
    )
    request_identity = {
        "project_id": str(payload.project_id),
        "schema_id": str(payload.schema_id),
        "schema_version_id": str(payload.schema_version_id),
        "orchestration_id": str(payload.orchestration_id),
        "consistency_check_application_id": str(payload.consistency_check_application_id),
        "state": payload.state,
        "limit": payload.limit,
    }
    return review_query_service.authenticate_review_items_result(
        payload.model_copy(
            update={
                "payload_hash": review_query_service._build_payload_hash(
                    payload,
                    tool_name="review_items_query",
                    request_identity=request_identity,
                )
            }
        ),
        request_identity=request_identity,
    )


def _build_review_item_detail_result() -> ReviewQueryItemDetailResult:
    payload = ReviewQueryItemDetailResult(
        source_manifest_hash=_hash("review-manifest"),
        payload_hash="",
        project_id=_uuid("project"),
        schema_id=_uuid("schema"),
        schema_version_id=_uuid("schema-version"),
        orchestration_id=_uuid("orchestration"),
        extraction_run_id=_uuid("extraction-run"),
        consistency_check_application_id=_uuid("consistency-app"),
        source_consistency_application_id=_uuid("source-consistency-app"),
        reviewed_projection_manifest_hash=_hash("review-manifest"),
        fact_id=_uuid("fact"),
        identity_hash=_hash("fact-identity"),
        subject_kind="person",
        subject_key="alpha",
        subject_entity_id=None,
        predicate_key="title",
        scope_key=None,
        semantic_value_count=1,
        fact_value_count=1,
        matched_field_keys=("title",),
        review_state="pending_review",
        resolution_basis="none",
        current_decision_id=None,
        current_decision_kind=None,
        effective_fact_value_ids=(),
        requires_review=True,
        value_groups=(
            {
                "semantic_key_hash": _hash("semantic"),
                "value_type": "string",
                "value_json": {"text": "Alice"},
                "referenced_entity_id": None,
                "fact_value_ids": [str(_uuid("fact-value"))],
                "values": [{"fact_value_id": str(_uuid("fact-value"))}],
                "evidences": [{"excerpt": "alpha evidence"}],
            },
        ),
    )
    request_identity = {
        "project_id": str(payload.project_id),
        "schema_id": str(payload.schema_id),
        "schema_version_id": str(payload.schema_version_id),
        "orchestration_id": str(payload.orchestration_id),
        "consistency_check_application_id": str(payload.consistency_check_application_id),
        "fact_id": str(payload.fact_id),
    }
    return review_query_service.authenticate_review_item_detail_result(
        payload.model_copy(
            update={
                "payload_hash": review_query_service._build_payload_hash(
                    payload,
                    tool_name="review_item_detail_query",
                    request_identity=request_identity,
                )
            }
        ),
        request_identity=request_identity,
    )


def _build_version_record_result() -> VersionRecordQueryResult:
    payload = VersionRecordQueryResult(
        source_manifest_hash=_hash("knowledge-manifest"),
        payload_hash="",
        project_id=_uuid("project"),
        project_version_id=_uuid("project-version"),
        version_no=2,
        is_current=True,
        schema_id=_uuid("schema"),
        schema_version_id=_uuid("schema-version"),
        orchestration_id=_uuid("orchestration"),
        extraction_run_id=_uuid("extraction-run"),
        consistency_check_application_id=_uuid("consistency-app"),
        source_consistency_application_id=_uuid("source-consistency-app"),
        knowledge_view_manifest_hash=_hash("knowledge-manifest"),
        subject_key="alpha",
        record_json={"subject_key": "alpha", "sections": []},
    )
    request_identity = {
        "project_id": str(payload.project_id),
        "project_version_id": str(payload.project_version_id),
        "subject_key": payload.subject_key,
    }
    return review_query_service.authenticate_version_record_response(
        payload.model_copy(
            update={
                "payload_hash": review_query_service._build_payload_hash(
                    payload,
                    tool_name="version_record_query",
                    request_identity=request_identity,
                )
            }
        ),
        request_identity=request_identity,
    )


def test_frontend_modules_do_not_import_bailian_modules() -> None:
    frontend_files = [
        REPO_ROOT / "app" / "routers" / "frontend_api.py",
        REPO_ROOT / "app" / "services" / "frontend_api.py",
        REPO_ROOT / "app" / "schemas" / "frontend_api.py",
    ]
    for path in frontend_files:
        imported = _imported_modules(path)
        assert all("bailian" not in module for module in imported), path.name


def test_core_services_do_not_import_routers_or_dependencies() -> None:
    service_files = [
        path
        for path in (REPO_ROOT / "app" / "services").glob("*.py")
        if not path.name.startswith("bailian_")
    ]
    for path in service_files:
        imported = _imported_modules(path)
        assert all(
            not module.startswith("app.routers") and not module.startswith("app.dependencies")
            for module in imported
        ), path.name


def test_app_creates_without_token_when_bailian_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_app(monkeypatch, enabled=False)
    assert app is not None


def test_bailian_endpoints_return_404_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_app(monkeypatch, enabled=False)
    with TestClient(app) as client:
        responses = [
            client.get(f"/api/v1/integrations/bailian/projects/{_uuid('project')}/review-items"),
            client.get(f"/api/v1/integrations/bailian/projects/{_uuid('project')}/review-items/{_uuid('fact')}"),
            client.get(
                f"/api/v1/integrations/bailian/projects/{_uuid('project')}/versions/{_uuid('project-version')}/records/alpha"
            ),
        ]
    assert [response.status_code for response in responses] == [404, 404, 404]


def test_frontend_operation_ids_remain_available_when_bailian_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_app(monkeypatch, enabled=False)
    openapi = app.openapi()
    operation_ids = {
        operation["operationId"]
        for path_item in openapi["paths"].values()
        for operation in path_item.values()
    }
    assert {
        "getCurrentUser",
        "frontendListReviewItems",
        "frontendGetReviewItemDetail",
        "frontendAppendReviewDecision",
        "frontendGetVersionRecord",
    }.issubset(operation_ids)
    assert "bailianListReviewItems" not in operation_ids


def test_bailian_endpoints_work_when_enabled_with_correct_token(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_app(monkeypatch, enabled=True)
    monkeypatch.setattr(
        bailian_tools_dependency,
        "get_settings",
        lambda: SimpleNamespace(bailian_tool_token=SecretStr("server-secret")),
    )
    app.dependency_overrides[get_async_session_factory] = lambda: (lambda: object())
    review_items = _build_review_items_result()
    review_detail = _build_review_item_detail_result()
    version_record = _build_version_record_result()

    async def fake_list(*args, **kwargs):
        return review_items

    async def fake_detail(*args, **kwargs):
        return review_detail

    async def fake_record(*args, **kwargs):
        return version_record

    monkeypatch.setattr(bailian_tools_service, "list_review_items", fake_list)
    monkeypatch.setattr(bailian_tools_service, "get_review_item_detail", fake_detail)
    monkeypatch.setattr(bailian_tools_service, "get_version_record", fake_record)

    with TestClient(app) as client:
        headers = {"Authorization": "Bearer server-secret"}
        list_response = client.get(
            f"/api/v1/integrations/bailian/projects/{_uuid('project')}/review-items",
            params={
                "schema_id": str(_uuid("schema")),
                "schema_version_id": str(_uuid("schema-version")),
                "orchestration_id": str(_uuid("orchestration")),
                "consistency_check_application_id": str(_uuid("consistency-app")),
            },
            headers=headers,
        )
        detail_response = client.get(
            f"/api/v1/integrations/bailian/projects/{_uuid('project')}/review-items/{_uuid('fact')}",
            params={
                "schema_id": str(_uuid("schema")),
                "schema_version_id": str(_uuid("schema-version")),
                "orchestration_id": str(_uuid("orchestration")),
                "consistency_check_application_id": str(_uuid("consistency-app")),
            },
            headers=headers,
        )
        record_response = client.get(
            f"/api/v1/integrations/bailian/projects/{_uuid('project')}/versions/{_uuid('project-version')}/records/alpha",
            headers=headers,
        )

    assert list_response.status_code == 200
    assert detail_response.status_code == 200
    assert record_response.status_code == 200


def test_bailian_wrong_token_stays_fail_closed_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_app(monkeypatch, enabled=True)
    monkeypatch.setattr(
        bailian_tools_dependency,
        "get_settings",
        lambda: SimpleNamespace(bailian_tool_token=SecretStr("server-secret")),
    )
    with TestClient(app) as client:
        response = client.get(
            f"/api/v1/integrations/bailian/projects/{_uuid('project')}/review-items",
            headers={"Authorization": "Bearer wrong-token"},
        )
    assert response.status_code == 401
    assert response.json() == {"detail": "bailian_tool_unauthorized"}


def test_frontend_and_bailian_list_routes_share_same_query_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_app(monkeypatch, enabled=True)
    result = _build_review_items_result()
    user = _build_user()
    project = _build_project(user.id)
    membership = _build_membership(project, user)

    async def fake_db() -> AsyncIterator[object]:
        yield object()

    async def fake_list(*args, **kwargs):
        return result

    async def fake_membership(*args, **kwargs):
        return membership

    app.dependency_overrides[get_db_session] = fake_db
    app.dependency_overrides[get_async_session_factory] = lambda: (lambda: object())
    app.dependency_overrides[require_current_user] = lambda: user
    monkeypatch.setattr(
        bailian_tools_dependency,
        "get_settings",
        lambda: SimpleNamespace(bailian_tool_token=SecretStr("server-secret")),
    )
    monkeypatch.setattr(review_query_service, "list_review_items", fake_list)
    monkeypatch.setattr(
        project_repository,
        "get_project_member_for_user_by_project_id",
        fake_membership,
    )

    with TestClient(app) as client:
        frontend_response = client.get(
            f"/api/v1/projects/{project.id}/review-items",
            params={
                "schema_id": str(result.schema_id),
                "schema_version_id": str(result.schema_version_id),
                "orchestration_id": str(result.orchestration_id),
                "consistency_check_application_id": str(result.consistency_check_application_id),
                "state": result.state,
                "limit": result.limit,
            },
        )
        bailian_response = client.get(
            f"/api/v1/integrations/bailian/projects/{result.project_id}/review-items",
            params={
                "schema_id": str(result.schema_id),
                "schema_version_id": str(result.schema_version_id),
                "orchestration_id": str(result.orchestration_id),
                "consistency_check_application_id": str(result.consistency_check_application_id),
                "state": result.state,
                "limit": result.limit,
            },
            headers={"Authorization": "Bearer server-secret"},
        )

    assert frontend_response.status_code == 200
    assert bailian_response.status_code == 200
    assert frontend_response.json()["items"][0]["fact_id"] == bailian_response.json()["items"][0]["fact_id"]
    assert frontend_response.json()["items"][0]["subject_key"] == bailian_response.json()["items"][0]["subject_key"]


def test_frontend_decision_readback_does_not_call_bailian_service(monkeypatch: pytest.MonkeyPatch) -> None:
    detail_result = _build_review_item_detail_result()

    async def fake_context(*args, **kwargs):
        return frontend_api_service._FrontendReviewFactContext(
            fact_id=detail_result.fact_id,
            assessment_id=_uuid("assessment"),
            requires_review=True,
        )

    async def fake_append(*args, **kwargs):
        return SimpleNamespace(
            decision_id=_uuid("decision"),
            decision_no=1,
            supersedes_decision_id=None,
            decision_manifest_hash="a" * 64,
            selected_fact_value_ids=(),
            created_new=True,
        )

    async def fake_detail(*args, **kwargs):
        return detail_result

    async def unexpected_bailian(*args, **kwargs):
        raise AssertionError("bailian_service_should_not_be_used")

    monkeypatch.setattr(frontend_api_service, "_get_review_fact_context", fake_context)
    monkeypatch.setattr(
        consistency_review_service,
        "append_consistency_review_decision",
        fake_append,
    )
    monkeypatch.setattr(review_query_service, "get_review_item_detail", fake_detail)
    monkeypatch.setattr(bailian_tools_service, "get_review_item_detail", unexpected_bailian)

    response = frontend_api_service.write_review_decision(
        lambda: object(),
        project_id=_uuid("project"),
        fact_id=detail_result.fact_id,
        schema_id=detail_result.schema_id,
        schema_version_id=detail_result.schema_version_id,
        orchestration_id=detail_result.orchestration_id,
        consistency_check_application_id=detail_result.consistency_check_application_id,
        assessment_id=_uuid("assessment"),
        actor_id=_uuid("actor"),
        expected_current_decision_id=None,
        decision_kind="defer",
        selected_fact_value_ids=(),
        comment=None,
    )
    write_result = __import__("asyncio").run(response)
    assert write_result.current_state.fact_id == detail_result.fact_id


def test_openapi_switches_bailian_operation_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    disabled_app = _build_app(monkeypatch, enabled=False)
    disabled_ids = {
        operation["operationId"]
        for path_item in disabled_app.openapi()["paths"].values()
        for operation in path_item.values()
    }
    enabled_app = _build_app(monkeypatch, enabled=True)
    enabled_ids = {
        operation["operationId"]
        for path_item in enabled_app.openapi()["paths"].values()
        for operation in path_item.values()
    }

    assert "bailianListReviewItems" not in disabled_ids
    assert {
        "bailianListReviewItems",
        "bailianGetReviewItemDetail",
        "bailianGetVersionRecord",
    }.issubset(enabled_ids)
    assert len(enabled_ids) == len(set(enabled_ids))
