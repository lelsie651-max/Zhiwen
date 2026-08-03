from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timezone
import re
from types import SimpleNamespace
import uuid

from pydantic import SecretStr, ValidationError
import pytest
from starlette.testclient import TestClient

from app.core.config import Settings
from app.core.database import get_async_session_factory, get_db_session
import app.dependencies.bailian_tools as bailian_tools_dependency
from app.main import app, create_app
from app.models.project import Project, ProjectStatus, ProjectVisibility
from app.models.project_member import ProjectMember, ProjectMemberRole
from app.models.user import User, UserStatus
from app.repositories import project as project_repository
from app.schemas.bailian_review_tools import (
    BailianReviewItemDetailResponse,
    BailianReviewItemsResponse,
    BailianReviewItemSummary,
    BailianVersionRecordResponse,
)
from app.schemas.frontend_api import FrontendReviewDecisionWriteResponse
from app.services import bailian_review_tools as bailian_tools_service
from app.services import consistency_review as consistency_review_service
from app.services import frontend_api as frontend_api_service
from app.services import identity as identity_service
import app.main as main_module


def extract_csrf_token(html: str) -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', html)
    assert match is not None
    return match.group(1)


def _uuid(seed: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"frontend-api:{seed}")


def _hash(seed: str) -> str:
    return bailian_tools_service.duplicate_grouping_service.hash_deterministic_payload(
        {"seed": seed}
    )


def build_user() -> User:
    now = datetime.now(timezone.utc)
    return User(
        id=_uuid("user"),
        handle="writer01",
        display_name="织文用户",
        email=None,
        avatar_url=None,
        locale="zh-CN",
        timezone="Asia/Shanghai",
        status=UserStatus.ACTIVE.value,
        created_at=now,
        updated_at=now,
    )


def build_project(user_id: uuid.UUID) -> Project:
    now = datetime.now(timezone.utc)
    return Project(
        id=_uuid("project"),
        name="织文项目",
        slug="zhiwen-project",
        description="项目说明",
        visibility=ProjectVisibility.PRIVATE.value,
        status=ProjectStatus.ACTIVE.value,
        created_by_id=user_id,
        created_at=now,
        updated_at=now,
    )


def build_membership(project: Project, user: User) -> ProjectMember:
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


def _build_list_response() -> BailianReviewItemsResponse:
    response = BailianReviewItemsResponse(
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
            BailianReviewItemSummary(
                fact_id=_uuid("fact"),
                subject_kind="person",
                subject_key="alpha",
                predicate_key="title",
                scope_key=None,
                matched_field_keys=("title",),
                review_state="pending_review",
                resolution_basis="none",
                requires_review=True,
                semantic_value_count=2,
                fact_value_count=2,
                evidence_count=1,
            ),
        ),
    )
    request_identity = {
        "project_id": str(response.project_id),
        "schema_id": str(response.schema_id),
        "schema_version_id": str(response.schema_version_id),
        "orchestration_id": str(response.orchestration_id),
        "consistency_check_application_id": str(
            response.consistency_check_application_id
        ),
        "state": response.state,
        "limit": response.limit,
    }
    return bailian_tools_service.authenticate_bailian_review_items_response(
        response.model_copy(
            update={
                "payload_hash": bailian_tools_service._build_payload_hash(
                    response,
                    tool_name="bailian_review_items",
                    request_identity=request_identity,
                )
            }
        ),
        request_identity=request_identity,
    )


def _build_detail_response() -> BailianReviewItemDetailResponse:
    return BailianReviewItemDetailResponse(
        source_manifest_hash=_hash("review-manifest"),
        payload_hash=_hash("detail-payload"),
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
        semantic_value_count=2,
        fact_value_count=3,
        matched_field_keys=("title",),
        review_state="pending_review",
        resolution_basis="none",
        current_decision_id=None,
        current_decision_kind=None,
        effective_fact_value_ids=(_uuid("fact-value-2"),),
        requires_review=True,
        value_groups=(
            {
                "semantic_key_hash": _hash("semantic-1"),
                "value_type": "string",
                "value_json": {"text": "Alice"},
                "referenced_entity_id": None,
                "fact_value_ids": [str(_uuid("fact-value-1"))],
                "values": [
                    {
                        "fact_value_id": str(_uuid("fact-value-1")),
                        "source_batch_id": str(_uuid("batch-1")),
                        "source_application_id": str(_uuid("source-app-1")),
                        "proposal_index": 0,
                        "normalized_value_text": "Alice",
                        "value_hash": _hash("value-1"),
                        "language_code": "zh",
                        "confidence": 0.8,
                    }
                ],
                "evidences": [
                    {
                        "evidence_link_id": str(_uuid("evidence-link-1")),
                        "evidence_id": str(_uuid("evidence-1")),
                        "document_revision_id": str(_uuid("doc-rev-1")),
                        "document_block_id": str(_uuid("doc-block-1")),
                        "locator": {
                            "location_key": "loc:1",
                            "page_no": 1,
                            "start_line": 1,
                            "end_line": 1,
                            "table_index": None,
                            "row_index": None,
                        },
                        "excerpt": "alpha evidence",
                        "excerpt_hash": _hash("excerpt-1"),
                        "content_hash": _hash("content-1"),
                        "role": "supporting",
                        "is_primary": True,
                        "source_order": 0,
                    }
                ],
            },
            {
                "semantic_key_hash": _hash("semantic-2"),
                "value_type": "string",
                "value_json": {"text": "Alicia"},
                "referenced_entity_id": None,
                "fact_value_ids": [str(_uuid("fact-value-2")), str(_uuid("fact-value-3"))],
                "values": [
                    {
                        "fact_value_id": str(_uuid("fact-value-2")),
                        "source_batch_id": str(_uuid("batch-2")),
                        "source_application_id": str(_uuid("source-app-2")),
                        "proposal_index": 0,
                        "normalized_value_text": "Alicia",
                        "value_hash": _hash("value-2"),
                        "language_code": "zh",
                        "confidence": 0.9,
                    },
                    {
                        "fact_value_id": str(_uuid("fact-value-3")),
                        "source_batch_id": str(_uuid("batch-3")),
                        "source_application_id": str(_uuid("source-app-3")),
                        "proposal_index": 1,
                        "normalized_value_text": "Alicia",
                        "value_hash": _hash("value-3"),
                        "language_code": "en",
                        "confidence": 0.7,
                    },
                ],
                "evidences": [
                    {
                        "evidence_link_id": str(_uuid("evidence-link-2")),
                        "evidence_id": str(_uuid("evidence-2")),
                        "document_revision_id": str(_uuid("doc-rev-2")),
                        "document_block_id": str(_uuid("doc-block-2")),
                        "locator": {
                            "location_key": "loc:2",
                            "page_no": 2,
                            "start_line": 3,
                            "end_line": 4,
                            "table_index": None,
                            "row_index": None,
                        },
                        "excerpt": "beta evidence",
                        "excerpt_hash": _hash("excerpt-2"),
                        "content_hash": _hash("content-2"),
                        "role": "supporting",
                        "is_primary": False,
                        "source_order": 1,
                    }
                ],
            },
        ),
    )


def _build_record_response() -> BailianVersionRecordResponse:
    response = BailianVersionRecordResponse(
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
        subject_key="织文/企业知识库",
        record_json={
            "subject_key": "织文/企业知识库",
            "title_field_key": "title",
            "has_review_required": False,
            "issue_count": 0,
            "sections": [],
        },
    )
    request_identity = {
        "project_id": str(response.project_id),
        "project_version_id": str(response.project_version_id),
        "subject_key": response.subject_key,
    }
    return bailian_tools_service.authenticate_bailian_version_record_response(
        response.model_copy(
            update={
                "payload_hash": bailian_tools_service._build_payload_hash(
                    response,
                    tool_name="bailian_version_record",
                    request_identity=request_identity,
                )
            }
        ),
        request_identity=request_identity,
    )


@pytest.fixture(autouse=True)
def override_dependencies() -> object:
    fake_db_session = object()

    async def _override_db() -> AsyncIterator[object]:
        yield fake_db_session

    def _override_factory():
        return "frontend-session-factory"

    app.dependency_overrides[get_db_session] = _override_db
    app.dependency_overrides[get_async_session_factory] = _override_factory
    yield fake_db_session
    app.dependency_overrides.clear()


def login_client(client: TestClient, monkeypatch: pytest.MonkeyPatch, user: User) -> None:
    async def fake_register_user(_session: object, _payload) -> User:
        return user

    async def fake_get_active_user_by_id(_session: object, user_id: uuid.UUID) -> User | None:
        if user_id == user.id:
            return user
        return None

    monkeypatch.setattr(identity_service, "register_user", fake_register_user)
    monkeypatch.setattr(identity_service, "get_active_user_by_id", fake_get_active_user_by_id)

    setup_page = client.get("/setup")
    csrf_token = extract_csrf_token(setup_page.text)
    response = client.post(
        "/setup",
        data={
            "handle": user.handle,
            "display_name": user.display_name,
            "email": "",
            "locale": "zh-CN",
            "timezone": "Asia/Shanghai",
            "csrf_token": csrf_token,
        },
        follow_redirects=False,
    )
    assert response.status_code == 303


def test_frontend_me_requires_authentication() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/me")

    assert response.status_code == 401
    assert response.json() == {"detail": "authentication_required"}


def test_frontend_me_returns_current_user_and_csrf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    user = build_user()

    with TestClient(app) as client:
        login_client(client, monkeypatch, user)
        response = client.get("/api/v1/me")

    assert response.status_code == 200
    body = response.json()
    assert body["user"]["id"] == str(user.id)
    assert body["user"]["handle"] == user.handle
    assert isinstance(body["csrf_token"], str)
    assert body["csrf_token"]


def test_frontend_review_list_rejects_without_project_access(monkeypatch: pytest.MonkeyPatch) -> None:
    user = build_user()

    async def fake_membership(*args, **kwargs):
        return None

    monkeypatch.setattr(
        project_repository,
        "get_project_member_for_user_by_project_id",
        fake_membership,
    )

    with TestClient(app) as client:
        login_client(client, monkeypatch, user)
        response = client.get(
            f"/api/v1/projects/{_uuid('project')}/review-items",
            params={
                "schema_id": str(_uuid("schema")),
                "schema_version_id": str(_uuid("schema-version")),
                "orchestration_id": str(_uuid("orchestration")),
                "consistency_check_application_id": str(_uuid("consistency-app")),
            },
        )

    assert response.status_code == 403
    assert response.json() == {"detail": "project_access_denied"}


def test_frontend_review_list_returns_authenticated_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    user = build_user()
    project = build_project(user.id)
    membership = build_membership(project, user)
    expected = _build_list_response()

    async def fake_membership(*args, **kwargs):
        return membership

    async def fake_list(session_factory, **kwargs):
        assert session_factory == "frontend-session-factory"
        assert kwargs["project_id"] == project.id
        assert kwargs["schema_id"] == _uuid("schema")
        assert kwargs["state"] == "review_required"
        assert kwargs["limit"] == 20
        return expected

    monkeypatch.setattr(
        project_repository,
        "get_project_member_for_user_by_project_id",
        fake_membership,
    )
    monkeypatch.setattr(frontend_api_service, "list_review_items", fake_list)

    with TestClient(app) as client:
        login_client(client, monkeypatch, user)
        response = client.get(
            f"/api/v1/projects/{project.id}/review-items",
            params={
                "schema_id": str(_uuid("schema")),
                "schema_version_id": str(_uuid("schema-version")),
                "orchestration_id": str(_uuid("orchestration")),
                "consistency_check_application_id": str(_uuid("consistency-app")),
                "state": "review_required",
                "limit": 20,
            },
        )

    assert response.status_code == 200
    assert response.json()["item_count"] == 1
    assert response.json()["items"][0]["fact_id"] == str(_uuid("fact"))


def test_frontend_review_detail_preserves_evidence_and_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = build_user()
    project = build_project(user.id)
    membership = build_membership(project, user)
    expected = _build_detail_response()

    async def fake_membership(*args, **kwargs):
        return membership

    async def fake_detail(session_factory, **kwargs):
        assert session_factory == "frontend-session-factory"
        assert kwargs["fact_id"] == _uuid("fact")
        return expected

    monkeypatch.setattr(
        project_repository,
        "get_project_member_for_user_by_project_id",
        fake_membership,
    )
    monkeypatch.setattr(frontend_api_service, "get_review_item_detail", fake_detail)

    with TestClient(app) as client:
        login_client(client, monkeypatch, user)
        response = client.get(
            f"/api/v1/projects/{project.id}/review-items/{_uuid('fact')}",
            params={
                "schema_id": str(_uuid("schema")),
                "schema_version_id": str(_uuid("schema-version")),
                "orchestration_id": str(_uuid("orchestration")),
                "consistency_check_application_id": str(_uuid("consistency-app")),
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert len(body["value_groups"]) == 2
    assert len(body["value_groups"][1]["values"]) == 2
    assert body["value_groups"][0]["evidences"][0]["excerpt"] == "alpha evidence"
    assert body["effective_fact_value_ids"] == [str(_uuid("fact-value-2"))]


def test_frontend_decision_write_requires_csrf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    user = build_user()
    project = build_project(user.id)
    membership = build_membership(project, user)

    async def fake_membership(*args, **kwargs):
        return membership

    monkeypatch.setattr(
        project_repository,
        "get_project_member_for_user_by_project_id",
        fake_membership,
    )

    with TestClient(app) as client:
        login_client(client, monkeypatch, user)
        response = client.post(
            f"/api/v1/projects/{project.id}/review-items/{_uuid('fact')}/decisions",
            json={
                "schema_id": str(_uuid("schema")),
                "schema_version_id": str(_uuid("schema-version")),
                "orchestration_id": str(_uuid("orchestration")),
                "consistency_check_application_id": str(_uuid("consistency-app")),
                "assessment_id": str(_uuid("assessment")),
                "expected_current_decision_id": None,
                "decision_kind": "select_one",
                "selected_fact_value_ids": [str(_uuid("fact-value-2"))],
                "comment": "保留人工选择",
            },
        )

    assert response.status_code == 403
    assert response.json() == {"detail": "Invalid CSRF token."}


def test_frontend_decision_write_supports_idempotent_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = build_user()
    project = build_project(user.id)
    membership = build_membership(project, user)
    detail = _build_detail_response()
    calls = {"count": 0}

    async def fake_membership(*args, **kwargs):
        return membership

    async def fake_write(session_factory, **kwargs):
        assert session_factory == "frontend-session-factory"
        assert kwargs["actor_id"] == user.id
        calls["count"] += 1
        return FrontendReviewDecisionWriteResponse(
            decision_id=_uuid("decision"),
            decision_no=1,
            supersedes_decision_id=None,
            decision_manifest_hash="a" * 64,
            selected_fact_value_ids=(_uuid("fact-value-2"),),
            created_new=calls["count"] == 1,
            current_state=detail,
        )

    monkeypatch.setattr(
        project_repository,
        "get_project_member_for_user_by_project_id",
        fake_membership,
    )
    monkeypatch.setattr(frontend_api_service, "write_review_decision", fake_write)

    with TestClient(app) as client:
        login_client(client, monkeypatch, user)
        me = client.get("/api/v1/me")
        csrf_token = me.json()["csrf_token"]
        payload = {
            "schema_id": str(_uuid("schema")),
            "schema_version_id": str(_uuid("schema-version")),
            "orchestration_id": str(_uuid("orchestration")),
            "consistency_check_application_id": str(_uuid("consistency-app")),
            "assessment_id": str(_uuid("assessment")),
            "expected_current_decision_id": None,
            "decision_kind": "select_one",
            "selected_fact_value_ids": [str(_uuid("fact-value-2"))],
            "comment": "保留人工选择",
        }
        first = client.post(
            f"/api/v1/projects/{project.id}/review-items/{_uuid('fact')}/decisions",
            json=payload,
            headers={"X-CSRF-Token": csrf_token},
        )
        second = client.post(
            f"/api/v1/projects/{project.id}/review-items/{_uuid('fact')}/decisions",
            json=payload,
            headers={"X-CSRF-Token": csrf_token},
        )

    assert first.status_code == 200
    assert first.json()["created_new"] is True
    assert second.status_code == 200
    assert second.json()["created_new"] is False


def test_frontend_decision_write_rejects_invalid_selection_and_masks_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = build_user()
    project = build_project(user.id)
    membership = build_membership(project, user)

    async def fake_membership(*args, **kwargs):
        return membership

    async def fake_invalid(*args, **kwargs):
        raise consistency_review_service.ConsistencyReviewStateError(
            "consistency_review_selected_fact_value_ids_invalid"
        )

    async def fake_invariant(*args, **kwargs):
        raise consistency_review_service.ConsistencyReviewInvariantError(
            "sensitive-sentinel-should-not-leak"
        )

    monkeypatch.setattr(
        project_repository,
        "get_project_member_for_user_by_project_id",
        fake_membership,
    )

    with TestClient(app) as client:
        login_client(client, monkeypatch, user)
        csrf_token = client.get("/api/v1/me").json()["csrf_token"]
        payload = {
            "schema_id": str(_uuid("schema")),
            "schema_version_id": str(_uuid("schema-version")),
            "orchestration_id": str(_uuid("orchestration")),
            "consistency_check_application_id": str(_uuid("consistency-app")),
            "assessment_id": str(_uuid("assessment")),
            "expected_current_decision_id": None,
            "decision_kind": "select_one",
            "selected_fact_value_ids": [str(_uuid("fact-value-2"))],
            "comment": "保留人工选择",
        }

        monkeypatch.setattr(frontend_api_service, "write_review_decision", fake_invalid)
        invalid = client.post(
            f"/api/v1/projects/{project.id}/review-items/{_uuid('fact')}/decisions",
            json=payload,
            headers={"X-CSRF-Token": csrf_token},
        )

        monkeypatch.setattr(frontend_api_service, "write_review_decision", fake_invariant)
        invariant = client.post(
            f"/api/v1/projects/{project.id}/review-items/{_uuid('fact')}/decisions",
            json=payload,
            headers={"X-CSRF-Token": csrf_token},
        )

    assert invalid.status_code == 422
    assert invalid.json() == {"detail": "consistency_review_selected_fact_value_ids_invalid"}
    assert invariant.status_code == 409
    assert invariant.json() == {"detail": "frontend_review_decision_source_invalid"}


def test_frontend_version_record_is_exact_and_no_implicit_latest_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = build_user()
    project = build_project(user.id)
    membership = build_membership(project, user)
    expected = _build_record_response()

    async def fake_membership(*args, **kwargs):
        return membership

    async def fake_record(session_factory, **kwargs):
        assert session_factory == "frontend-session-factory"
        assert kwargs["project_version_id"] == _uuid("project-version")
        return expected

    monkeypatch.setattr(
        project_repository,
        "get_project_member_for_user_by_project_id",
        fake_membership,
    )
    monkeypatch.setattr(frontend_api_service, "get_version_record", fake_record)

    with TestClient(app) as client:
        login_client(client, monkeypatch, user)
        exact = client.get(
            f"/api/v1/projects/{project.id}/versions/{_uuid('project-version')}/records/%E7%BB%87%E6%96%87%2F%E4%BC%81%E4%B8%9A%E7%9F%A5%E8%AF%86%E5%BA%93"
        )
        implicit_latest = client.get(
            f"/api/v1/projects/{project.id}/versions/records/%E7%BB%87%E6%96%87%2F%E4%BC%81%E4%B8%9A%E7%9F%A5%E8%AF%86%E5%BA%93"
        )

    assert exact.status_code == 200
    assert exact.json()["project_version_id"] == str(_uuid("project-version"))
    assert implicit_latest.status_code == 404


def test_frontend_version_record_masks_invariant_error(monkeypatch: pytest.MonkeyPatch) -> None:
    user = build_user()
    project = build_project(user.id)
    membership = build_membership(project, user)

    async def fake_membership(*args, **kwargs):
        return membership

    async def fake_record(*args, **kwargs):
        raise bailian_tools_service.BailianReviewToolInvariantError(
            "sensitive-sentinel-should-not-leak"
        )

    monkeypatch.setattr(
        project_repository,
        "get_project_member_for_user_by_project_id",
        fake_membership,
    )
    monkeypatch.setattr(frontend_api_service, "get_version_record", fake_record)

    with TestClient(app) as client:
        login_client(client, monkeypatch, user)
        response = client.get(
            f"/api/v1/projects/{project.id}/versions/{_uuid('project-version')}/records/%E7%BB%87%E6%96%87%2F%E4%BC%81%E4%B8%9A%E7%9F%A5%E8%AF%86%E5%BA%93"
        )

    assert response.status_code == 409
    assert response.json() == {"detail": "frontend_api_source_invalid"}


def test_openapi_operation_ids_are_unique_and_bailian_ids_unchanged() -> None:
    openapi = app.openapi()
    operation_ids: list[str] = []
    for path_item in openapi["paths"].values():
        for operation in path_item.values():
            operation_ids.append(operation["operationId"])

    assert len(operation_ids) == len(set(operation_ids))
    assert openapi["paths"]["/api/v1/integrations/bailian/projects/{project_id}/review-items"]["get"][
        "operationId"
    ] == "bailianListReviewItems"
    assert openapi["paths"]["/api/v1/projects/{project_id}/review-items"]["get"][
        "operationId"
    ] == "frontendListReviewItems"
    assert openapi["paths"]["/api/v1/projects/{project_id}/review-items/{fact_id}/decisions"]["post"][
        "operationId"
    ] == "frontendAppendReviewDecision"


def test_frontend_origins_normalize_and_reject_invalid_config() -> None:
    settings = Settings(frontend_origins="http://localhost:3000/, https://example.com")
    assert settings.frontend_origins == (
        "http://localhost:3000",
        "https://example.com",
    )

    with pytest.raises(ValidationError):
        Settings(frontend_origins="*,http://localhost:3000")


def test_cors_whitelist_allows_configured_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    custom_settings = Settings(frontend_origins="http://localhost:3000")
    monkeypatch.setattr(main_module, "settings", custom_settings)
    local_app = create_app()

    with TestClient(local_app) as client:
        response = client.options(
            "/api/v1/me",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert response.headers["access-control-allow-credentials"] == "true"
    assert response.headers["access-control-allow-origin"] != "*"


def test_cors_rejects_unconfigured_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    custom_settings = Settings(frontend_origins="http://localhost:3000")
    monkeypatch.setattr(main_module, "settings", custom_settings)
    local_app = create_app()

    with TestClient(local_app) as client:
        response = client.options(
            "/api/v1/me",
            headers={
                "Origin": "http://malicious.local",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert "access-control-allow-origin" not in response.headers
