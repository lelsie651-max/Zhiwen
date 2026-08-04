from __future__ import annotations

import asyncio
import uuid

import pytest

from app.schemas.review_query import (
    ReviewQueryItemDetailResult,
    ReviewQueryItemsResult,
    ReviewQueryItemSummary,
    VersionRecordQueryResult,
)
from app.services import bailian_review_tools as bailian_tools_service
from app.services import review_query as review_query_service


def run_async(awaitable):
    return asyncio.run(awaitable)


def _uuid(seed: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"bailian-adapter:{seed}")


def _hash(seed: str) -> str:
    return review_query_service.duplicate_grouping_service.hash_deterministic_payload(
        {"seed": seed}
    )


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
        state="all",
        limit=20,
        item_count=1,
        items=(
            ReviewQueryItemSummary(
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


def test_bailian_adapter_resigns_review_items_with_legacy_tool_name(monkeypatch) -> None:
    query_result = _build_review_items_result()

    async def fake_list(*args, **kwargs):
        return query_result

    monkeypatch.setattr(review_query_service, "list_review_items", fake_list)

    response = run_async(
        bailian_tools_service.list_review_items(
            object(),
            project_id=str(query_result.project_id),
            schema_id=str(query_result.schema_id),
            schema_version_id=str(query_result.schema_version_id),
            orchestration_id=str(query_result.orchestration_id),
            consistency_check_application_id=str(query_result.consistency_check_application_id),
            state=query_result.state,
            limit=query_result.limit,
        )
    )

    request_identity = {
        "project_id": str(query_result.project_id),
        "schema_id": str(query_result.schema_id),
        "schema_version_id": str(query_result.schema_version_id),
        "orchestration_id": str(query_result.orchestration_id),
        "consistency_check_application_id": str(query_result.consistency_check_application_id),
        "state": query_result.state,
        "limit": query_result.limit,
    }
    expected_hash = bailian_tools_service._build_payload_hash(
        response,
        tool_name="bailian_review_items",
        request_identity=request_identity,
    )

    assert response.payload_hash == expected_hash
    assert response.payload_hash != query_result.payload_hash


def test_bailian_adapter_resigns_detail_and_version_payloads(monkeypatch) -> None:
    detail_result = _build_review_item_detail_result()
    record_result = _build_version_record_result()

    async def fake_detail(*args, **kwargs):
        return detail_result

    async def fake_record(*args, **kwargs):
        return record_result

    monkeypatch.setattr(review_query_service, "get_review_item_detail", fake_detail)
    monkeypatch.setattr(review_query_service, "get_version_record", fake_record)

    detail_response = run_async(
        bailian_tools_service.get_review_item_detail(
            object(),
            project_id=str(detail_result.project_id),
            fact_id=str(detail_result.fact_id),
            schema_id=str(detail_result.schema_id),
            schema_version_id=str(detail_result.schema_version_id),
            orchestration_id=str(detail_result.orchestration_id),
            consistency_check_application_id=str(detail_result.consistency_check_application_id),
        )
    )
    record_response = run_async(
        bailian_tools_service.get_version_record(
            object(),
            project_id=str(record_result.project_id),
            project_version_id=str(record_result.project_version_id),
            subject_key=record_result.subject_key,
        )
    )

    assert detail_response.payload_hash != detail_result.payload_hash
    assert record_response.payload_hash != record_result.payload_hash
    assert detail_response.subject_key == detail_result.subject_key
    assert record_response.record_json["subject_key"] == record_result.subject_key


@pytest.mark.parametrize(
    ("service_name", "error", "expected_detail"),
    [
        (
            "list_review_items",
            review_query_service.ReviewQueryStateError("review_query_state_invalid"),
            "bailian_review_tool_state_invalid",
        ),
        (
            "get_review_item_detail",
            review_query_service.ReviewQueryNotFoundError("review_item_not_found"),
            "bailian_review_item_not_found",
        ),
        (
            "get_version_record",
            review_query_service.ReviewQueryInvariantError("review_query_source_mismatch"),
            "bailian_review_tool_source_mismatch",
        ),
    ],
)
def test_bailian_adapter_maps_review_query_errors(
    monkeypatch: pytest.MonkeyPatch,
    service_name: str,
    error: Exception,
    expected_detail: str,
) -> None:
    async def fake_service(*args, **kwargs):
        raise error

    monkeypatch.setattr(review_query_service, service_name, fake_service)

    if service_name == "list_review_items":
        awaitable = bailian_tools_service.list_review_items(
            object(),
            project_id=str(_uuid("project")),
            schema_id=str(_uuid("schema")),
            schema_version_id=str(_uuid("schema-version")),
            orchestration_id=str(_uuid("orchestration")),
            consistency_check_application_id=str(_uuid("consistency-app")),
        )
    elif service_name == "get_review_item_detail":
        awaitable = bailian_tools_service.get_review_item_detail(
            object(),
            project_id=str(_uuid("project")),
            fact_id=str(_uuid("fact")),
            schema_id=str(_uuid("schema")),
            schema_version_id=str(_uuid("schema-version")),
            orchestration_id=str(_uuid("orchestration")),
            consistency_check_application_id=str(_uuid("consistency-app")),
        )
    else:
        awaitable = bailian_tools_service.get_version_record(
            object(),
            project_id=str(_uuid("project")),
            project_version_id=str(_uuid("project-version")),
            subject_key="alpha",
        )

    with pytest.raises(type(bailian_tools_service._translate_review_query_error(error)), match=expected_detail):
        run_async(awaitable)
