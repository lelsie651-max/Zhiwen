from __future__ import annotations

from types import SimpleNamespace
import uuid

from pydantic import SecretStr
import pytest
from starlette.testclient import TestClient

from app.core.config import Settings
from app.core.database import get_async_session_factory
from app.schemas.bailian_review_tools import (
    BailianReviewItemDetailResponse,
    BailianReviewItemsResponse,
    BailianReviewItemSummary,
    BailianVersionRecordResponse,
)
from app.services import bailian_review_tools as bailian_tools_service
import app.dependencies.bailian_tools as bailian_tools_dependency
import app.main as main_module


def _uuid(seed: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"bailian-tools-router:{seed}")


def _hash(seed: str) -> str:
    return bailian_tools_service.duplicate_grouping_service.hash_deterministic_payload(
        {"seed": seed}
    )


@pytest.fixture(autouse=True)
def configure_local_app(monkeypatch: pytest.MonkeyPatch):
    settings = Settings(bailian_integration_enabled=True)
    monkeypatch.setattr(main_module, "settings", settings)
    monkeypatch.setattr(main_module, "app", main_module.create_app())


@pytest.fixture(autouse=True)
def override_async_session_factory():
    def _factory():
        raise AssertionError("unexpected db factory access")

    main_module.app.dependency_overrides[get_async_session_factory] = lambda: _factory
    yield
    main_module.app.dependency_overrides.clear()


def _set_tool_token(monkeypatch: pytest.MonkeyPatch, token: str) -> None:
    monkeypatch.setattr(
        bailian_tools_dependency,
        "get_settings",
        lambda: SimpleNamespace(bailian_tool_token=SecretStr(token)),
    )


def _bearer_headers(token: str, *, scheme: str = "Bearer") -> dict[str, str]:
    return {"Authorization": f"{scheme} {token}"}


def _build_list_response() -> BailianReviewItemsResponse:
    response = BailianReviewItemsResponse(
        source_manifest_hash=_hash("reviewed-manifest"),
        payload_hash="",
        project_id=_uuid("project"),
        schema_id=_uuid("schema"),
        schema_version_id=_uuid("schema-version"),
        orchestration_id=_uuid("orchestration"),
        consistency_check_application_id=_uuid("consistency-app"),
        reviewed_projection_manifest_hash=_hash("reviewed-manifest"),
        state="all",
        limit=50,
        item_count=1,
        items=(
            BailianReviewItemSummary(
                fact_id=_uuid("fact"),
                subject_kind="person",
                subject_key="alpha",
                predicate_key="title",
                scope_key=None,
                matched_field_keys=("title", "alias"),
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
        source_manifest_hash=_hash("reviewed-manifest"),
        payload_hash=_hash("detail-payload"),
        project_id=_uuid("project"),
        schema_id=_uuid("schema"),
        schema_version_id=_uuid("schema-version"),
        orchestration_id=_uuid("orchestration"),
        extraction_run_id=_uuid("extraction-run"),
        consistency_check_application_id=_uuid("consistency-app"),
        source_consistency_application_id=_uuid("source-consistency-app"),
        reviewed_projection_manifest_hash=_hash("reviewed-manifest"),
        fact_id=_uuid("fact"),
        identity_hash=_hash("fact-identity"),
        subject_kind="person",
        subject_key="alpha",
        subject_entity_id=None,
        predicate_key="title",
        scope_key=None,
        semantic_value_count=1,
        fact_value_count=2,
        matched_field_keys=("title", "alias"),
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
                "values": [
                    {
                        "fact_value_id": str(_uuid("fact-value")),
                        "source_batch_id": str(_uuid("batch")),
                        "source_application_id": str(_uuid("source-app")),
                        "proposal_index": 0,
                        "normalized_value_text": "Alice",
                        "value_hash": _hash("value"),
                        "language_code": "zh",
                        "confidence": 0.9,
                    }
                ],
                "evidences": [
                    {
                        "evidence_link_id": str(_uuid("evidence-link")),
                        "evidence_id": str(_uuid("evidence")),
                        "document_revision_id": str(_uuid("doc-rev")),
                        "document_block_id": str(_uuid("doc-block")),
                        "locator": {
                            "location_key": "loc:alpha",
                            "page_no": 1,
                            "start_line": 1,
                            "end_line": 1,
                            "table_index": None,
                            "row_index": None,
                        },
                        "excerpt": "alpha evidence",
                        "excerpt_hash": _hash("excerpt"),
                        "content_hash": _hash("content"),
                        "role": "supporting",
                        "is_primary": True,
                        "source_order": 0,
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
        subject_key="alpha",
        record_json={
            "subject_key": "alpha",
            "title_field_key": "title",
            "has_review_required": True,
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


def test_bailian_router_rejects_unconfigured_token(monkeypatch) -> None:
    _set_tool_token(monkeypatch, "")
    with TestClient(main_module.app) as client:
        response = client.get(
            f"/api/v1/integrations/bailian/projects/{_uuid('project')}/review-items"
        )
    assert response.status_code == 503
    assert response.json() == {"detail": "bailian_tool_unconfigured"}


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"X-Zhiwen-Tool-Token": "server-secret"},
        {"Authorization": "Basic server-secret"},
        {"Authorization": "AppCode server-secret"},
        {"Authorization": "Bearer"},
        {"Authorization": "Bearer "},
        {"Authorization": "Bearer wrong-token"},
        {"Authorization": "malformed"},
    ],
)
def test_bailian_router_rejects_missing_or_invalid_authorization(
    monkeypatch: pytest.MonkeyPatch,
    headers: dict[str, str],
) -> None:
    _set_tool_token(monkeypatch, "server-secret")
    with TestClient(main_module.app) as client:
        response = client.get(
            f"/api/v1/integrations/bailian/projects/{_uuid('project')}/review-items",
            headers=headers,
        )
    assert response.status_code == 401
    assert response.json() == {"detail": "bailian_tool_unauthorized"}
    assert "server-secret" not in response.text


def test_bailian_router_accepts_correct_token_and_calls_services(monkeypatch) -> None:
    _set_tool_token(monkeypatch, "server-secret")
    calls = {"list": 0, "detail": 0, "record": 0}

    async def fake_list(*args, **kwargs):
        calls["list"] += 1
        return _build_list_response()

    async def fake_detail(*args, **kwargs):
        calls["detail"] += 1
        return _build_detail_response()

    async def fake_record(*args, **kwargs):
        calls["record"] += 1
        return _build_record_response()

    monkeypatch.setattr(bailian_tools_service, "list_review_items", fake_list)
    monkeypatch.setattr(bailian_tools_service, "get_review_item_detail", fake_detail)
    monkeypatch.setattr(bailian_tools_service, "get_version_record", fake_record)

    headers = _bearer_headers("server-secret")
    with TestClient(main_module.app) as client:
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
    assert calls == {"list": 1, "detail": 1, "record": 1}


@pytest.mark.parametrize("scheme", ["Bearer", "bearer", "BEARER"])
def test_bailian_router_accepts_case_insensitive_bearer_scheme(
    monkeypatch: pytest.MonkeyPatch,
    scheme: str,
) -> None:
    _set_tool_token(monkeypatch, "server-secret")

    async def fake_list(*args, **kwargs):
        return _build_list_response()

    monkeypatch.setattr(bailian_tools_service, "list_review_items", fake_list)

    with TestClient(main_module.app) as client:
        response = client.get(
            f"/api/v1/integrations/bailian/projects/{_uuid('project')}/review-items",
            params={
                "schema_id": str(_uuid("schema")),
                "schema_version_id": str(_uuid("schema-version")),
                "orchestration_id": str(_uuid("orchestration")),
                "consistency_check_application_id": str(_uuid("consistency-app")),
            },
            headers=_bearer_headers("server-secret", scheme=scheme),
        )

    assert response.status_code == 200


def test_bailian_router_serializes_frozen_detail_and_record_json(monkeypatch) -> None:
    _set_tool_token(monkeypatch, "server-secret")

    async def fake_detail(*args, **kwargs):
        return _build_detail_response()

    async def fake_record(*args, **kwargs):
        return _build_record_response()

    monkeypatch.setattr(bailian_tools_service, "get_review_item_detail", fake_detail)
    monkeypatch.setattr(bailian_tools_service, "get_version_record", fake_record)
    headers = _bearer_headers("server-secret")
    with TestClient(main_module.app) as client:
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

    detail_json = detail_response.json()
    record_json = record_response.json()
    assert isinstance(detail_json["value_groups"], list)
    assert isinstance(detail_json["value_groups"][0], dict)
    assert isinstance(detail_json["value_groups"][0]["values"], list)
    assert isinstance(record_json["record_json"], dict)
    assert isinstance(record_json["record_json"]["sections"], list)


def test_bailian_router_supports_subject_key_path_segments(monkeypatch) -> None:
    _set_tool_token(monkeypatch, "server-secret")
    captured: list[str] = []

    async def fake_record(*args, **kwargs):
        captured.append(kwargs["subject_key"])
        return _build_record_response().model_copy(
            update={"subject_key": kwargs["subject_key"], "record_json": {"subject_key": kwargs["subject_key"], "sections": []}}
        )

    monkeypatch.setattr(bailian_tools_service, "get_version_record", fake_record)
    headers = _bearer_headers("server-secret")
    with TestClient(main_module.app) as client:
        response = client.get(
            f"/api/v1/integrations/bailian/projects/{_uuid('project')}/versions/{_uuid('project-version')}/records/%E7%A0%94%E5%8F%91%E9%83%A8/%E5%9F%BA%E7%A1%80%E5%B9%B3%E5%8F%B0%E7%BB%84",
            headers=headers,
        )

    assert response.status_code == 200
    assert captured == ["研发部/基础平台组"]
    assert response.json()["subject_key"] == "研发部/基础平台组"


@pytest.mark.parametrize(
    ("service_name", "error", "status_code", "detail"),
    [
        (
            "list_review_items",
            bailian_tools_service.BailianReviewToolStateError(
                "bailian_review_tool_state_invalid"
            ),
            400,
            "bailian_review_tool_state_invalid",
        ),
        (
            "get_review_item_detail",
            bailian_tools_service.BailianReviewToolNotFoundError(
                "bailian_review_item_not_found"
            ),
            404,
            "bailian_review_item_not_found",
        ),
        (
            "get_version_record",
            bailian_tools_service.BailianReviewToolInvariantError(
                "bailian_review_tool_source_mismatch"
            ),
            409,
            "bailian_review_tool_source_mismatch",
        ),
    ],
)
def test_bailian_router_maps_domain_errors(
    monkeypatch: pytest.MonkeyPatch,
    service_name: str,
    error: Exception,
    status_code: int,
    detail: str,
) -> None:
    _set_tool_token(monkeypatch, "server-secret")

    async def fake_service(*args, **kwargs):
        raise error

    monkeypatch.setattr(bailian_tools_service, service_name, fake_service)
    headers = _bearer_headers("server-secret")
    with TestClient(main_module.app) as client:
        if service_name == "list_review_items":
            response = client.get(
                f"/api/v1/integrations/bailian/projects/{_uuid('project')}/review-items",
                params={
                    "schema_id": str(_uuid("schema")),
                    "schema_version_id": str(_uuid("schema-version")),
                    "orchestration_id": str(_uuid("orchestration")),
                    "consistency_check_application_id": str(
                        _uuid("consistency-app")
                    ),
                },
                headers=headers,
            )
        elif service_name == "get_review_item_detail":
            response = client.get(
                f"/api/v1/integrations/bailian/projects/{_uuid('project')}/review-items/{_uuid('fact')}",
                params={
                    "schema_id": str(_uuid("schema")),
                    "schema_version_id": str(_uuid("schema-version")),
                    "orchestration_id": str(_uuid("orchestration")),
                    "consistency_check_application_id": str(
                        _uuid("consistency-app")
                    ),
                },
                headers=headers,
            )
        else:
            response = client.get(
                f"/api/v1/integrations/bailian/projects/{_uuid('project')}/versions/{_uuid('project-version')}/records/alpha",
                headers=headers,
            )

    assert response.status_code == status_code
    assert response.json() == {"detail": detail}


def test_bailian_router_openapi_exposes_bearer_security_and_operation_ids(monkeypatch) -> None:
    secret = "server-secret"
    _set_tool_token(monkeypatch, secret)

    with TestClient(main_module.app) as client:
        openapi = client.get("/openapi.json").json()

    security_scheme = openapi["components"]["securitySchemes"]["BailianToolBearer"]
    assert security_scheme["type"] == "http"
    assert security_scheme["scheme"] == "bearer"
    assert security_scheme["bearerFormat"] == "opaque shared token"
    assert "BailianToolToken" not in openapi["components"]["securitySchemes"]
    assert (
        openapi["paths"]["/api/v1/integrations/bailian/projects/{project_id}/review-items"]["get"]["operationId"]
        == "bailianListReviewItems"
    )
    assert (
        openapi["paths"]["/api/v1/integrations/bailian/projects/{project_id}/review-items/{fact_id}"]["get"]["operationId"]
        == "bailianGetReviewItemDetail"
    )
    assert (
        openapi["paths"]["/api/v1/integrations/bailian/projects/{project_id}/versions/{project_version_id}/records/{subject_key}"]["get"]["operationId"]
        == "bailianGetVersionRecord"
    )
    assert openapi["paths"]["/api/v1/integrations/bailian/projects/{project_id}/review-items"]["get"]["security"] == [
        {"BailianToolBearer": []}
    ]
    assert secret not in str(openapi)
    assert "X-Zhiwen-Tool-Token" not in str(openapi)
