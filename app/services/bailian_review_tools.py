from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
import uuid
from typing import Any

from pydantic import BaseModel

from app.schemas.bailian_review_tools import (
    BAILIAN_READ_TOOL_PAYLOAD_ALGORITHM_NAME,
    BAILIAN_READ_TOOL_PAYLOAD_ALGORITHM_VERSION,
    BailianReviewItemDetailResponse,
    BailianReviewItemsResponse,
    BailianReviewItemSummary,
    BailianVersionRecordResponse,
)
from app.schemas.dynamic_schema_review_projection import (
    DynamicSchemaReviewProjection,
    DynamicSchemaReviewedFact,
)
import app.services.dynamic_schema_review_projection as review_projection_service
import app.services.dynamic_schema_ufl_projection as raw_projection_service
import app.services.fact_value_duplicate_grouping as duplicate_grouping_service
import app.services.project_version as project_version_service


class BailianReviewToolError(Exception):
    """Base error for Bailian read tools."""


class BailianReviewToolStateError(BailianReviewToolError):
    """Raised when Bailian read tool inputs are invalid."""


class BailianReviewToolNotFoundError(BailianReviewToolError):
    """Raised when the requested Bailian read tool resource is missing."""


class BailianReviewToolInvariantError(BailianReviewToolError):
    """Raised when authenticated source projections drift or conflict."""


def _require_uuid(value: object, *, field_name: str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    if not isinstance(value, str):
        raise BailianReviewToolStateError(f"bailian_review_tool_{field_name}_invalid")
    try:
        return uuid.UUID(value)
    except (TypeError, ValueError):
        raise BailianReviewToolStateError(
            f"bailian_review_tool_{field_name}_invalid"
        ) from None


def _normalize_review_item_state(value: object) -> str:
    if not isinstance(value, str):
        raise BailianReviewToolStateError("bailian_review_tool_state_invalid")
    normalized = value.strip()
    if normalized not in {"review_required", "resolved", "observation_only", "all"}:
        raise BailianReviewToolStateError("bailian_review_tool_state_invalid")
    return normalized


def _normalize_limit(value: object) -> int:
    if isinstance(value, bool):
        raise BailianReviewToolStateError("bailian_review_tool_limit_invalid")
    if isinstance(value, int):
        normalized = value
    elif isinstance(value, str):
        try:
            normalized = int(value.strip())
        except ValueError:
            raise BailianReviewToolStateError(
                "bailian_review_tool_limit_invalid"
            ) from None
    else:
        raise BailianReviewToolStateError("bailian_review_tool_limit_invalid")
    if not 1 <= normalized <= 100:
        raise BailianReviewToolStateError("bailian_review_tool_limit_invalid")
    return normalized


def _normalize_subject_key(value: object) -> str:
    if not isinstance(value, str):
        raise BailianReviewToolStateError("bailian_review_tool_subject_key_invalid")
    try:
        return raw_projection_service.normalize_dynamic_schema_ufl_subject_keys([value])[0]
    except raw_projection_service.DynamicSchemaUFLProjectionStateError:
        raise BailianReviewToolStateError(
            "bailian_review_tool_subject_key_invalid"
        ) from None


def _review_item_state(reviewed_fact: DynamicSchemaReviewedFact) -> str:
    if reviewed_fact.requires_review:
        return "review_required"
    if reviewed_fact.review_state == "resolved":
        return "resolved"
    return "observation_only"


def _evidence_count(reviewed_fact: DynamicSchemaReviewedFact) -> int:
    return sum(len(value_group.evidences) for value_group in reviewed_fact.fact.value_groups)


def _serialize_reviewed_fact_values(
    reviewed_fact: DynamicSchemaReviewedFact,
) -> tuple[dict[str, object], ...]:
    fact_json = raw_projection_service.serialize_dynamic_schema_ufl_fact(reviewed_fact.fact)
    value_groups = fact_json.get("value_groups")
    if not isinstance(value_groups, list):
        raise BailianReviewToolInvariantError("bailian_review_tool_payload_invalid")
    return tuple(value_groups)


async def _get_authenticated_review_projection(
    session_factory,
    *,
    project_id: uuid.UUID,
    schema_id: uuid.UUID,
    schema_version_id: uuid.UUID,
    orchestration_id: uuid.UUID,
    consistency_check_application_id: uuid.UUID,
) -> DynamicSchemaReviewProjection:
    try:
        projection = await review_projection_service.project_reviewed_orchestration_ufl_to_dynamic_schema(
            session_factory,
            project_id=project_id,
            schema_id=schema_id,
            schema_version_id=schema_version_id,
            orchestration_id=orchestration_id,
            consistency_check_application_id=consistency_check_application_id,
            subject_keys=None,
        )
        return review_projection_service.authenticate_dynamic_schema_review_projection(
            projection,
            subject_keys=None,
        )
    except review_projection_service.DynamicSchemaReviewProjectionStateError:
        raise BailianReviewToolStateError("bailian_review_tool_source_invalid") from None
    except review_projection_service.DynamicSchemaReviewProjectionInvariantError:
        raise BailianReviewToolInvariantError(
            "bailian_review_tool_source_mismatch"
        ) from None


def _build_review_item_index(
    projection: DynamicSchemaReviewProjection,
) -> dict[uuid.UUID, dict[str, object]]:
    indexed: dict[uuid.UUID, dict[str, object]] = {}
    for record in projection.records:
        for field in record.fields:
            for reviewed_fact in field.reviewed_facts:
                current = indexed.get(reviewed_fact.fact.fact_id)
                if current is None:
                    indexed[reviewed_fact.fact.fact_id] = {
                        "reviewed_fact": reviewed_fact,
                        "matched_field_keys": [field.source_field.field_key],
                    }
                    continue
                if current["reviewed_fact"] != reviewed_fact:
                    raise BailianReviewToolInvariantError(
                        "bailian_review_tool_source_mismatch"
                    )
                matched_field_keys = current["matched_field_keys"]
                if field.source_field.field_key not in matched_field_keys:
                    matched_field_keys.append(field.source_field.field_key)
    return indexed


def _summary_from_index_entry(
    *,
    reviewed_fact: DynamicSchemaReviewedFact,
    matched_field_keys: Sequence[str],
) -> BailianReviewItemSummary:
    return BailianReviewItemSummary(
        fact_id=reviewed_fact.fact.fact_id,
        subject_kind=reviewed_fact.fact.subject_kind,
        subject_key=reviewed_fact.fact.subject_key,
        predicate_key=reviewed_fact.fact.predicate_key,
        scope_key=reviewed_fact.fact.scope_key,
        matched_field_keys=tuple(matched_field_keys),
        review_state=reviewed_fact.review_state,
        resolution_basis=reviewed_fact.resolution_basis,
        requires_review=reviewed_fact.requires_review,
        semantic_value_count=reviewed_fact.fact.semantic_group_count,
        fact_value_count=reviewed_fact.fact.fact_value_count,
        evidence_count=_evidence_count(reviewed_fact),
    )


def _sort_review_items(
    items: Sequence[BailianReviewItemSummary],
) -> tuple[BailianReviewItemSummary, ...]:
    return tuple(
        sorted(
            items,
            key=lambda item: (
                not item.requires_review,
                item.subject_key,
                item.predicate_key,
                item.scope_key is not None,
                item.scope_key or "",
                str(item.fact_id),
            ),
        )
    )


def _freeze_payload_hash(
    payload_model: BaseModel,
    *,
    tool_name: str,
    request_identity: Mapping[str, object],
) -> str:
    return duplicate_grouping_service.hash_deterministic_payload(
        {
            "tool_name": tool_name,
            "request_identity": dict(request_identity),
            "algorithm": {
                "name": BAILIAN_READ_TOOL_PAYLOAD_ALGORITHM_NAME,
                "version": BAILIAN_READ_TOOL_PAYLOAD_ALGORITHM_VERSION,
            },
            "content": payload_model.model_dump(mode="json", exclude={"payload_hash"}),
        }
    )


def _finalize_payload(
    payload_model: BaseModel,
    *,
    tool_name: str,
    request_identity: Mapping[str, object],
):
    return payload_model.model_copy(
        update={
            "payload_hash": _freeze_payload_hash(
                payload_model,
                tool_name=tool_name,
                request_identity=request_identity,
            )
        }
    )


def _thaw_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_thaw_json_value(item) for item in value]
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


async def list_review_items(
    session_factory,
    *,
    project_id: object,
    schema_id: object,
    schema_version_id: object,
    orchestration_id: object,
    consistency_check_application_id: object,
    state: object = "all",
    limit: object = 50,
) -> BailianReviewItemsResponse:
    normalized_project_id = _require_uuid(project_id, field_name="project_id")
    normalized_schema_id = _require_uuid(schema_id, field_name="schema_id")
    normalized_schema_version_id = _require_uuid(
        schema_version_id,
        field_name="schema_version_id",
    )
    normalized_orchestration_id = _require_uuid(
        orchestration_id,
        field_name="orchestration_id",
    )
    normalized_application_id = _require_uuid(
        consistency_check_application_id,
        field_name="consistency_check_application_id",
    )
    normalized_state = _normalize_review_item_state(state)
    normalized_limit = _normalize_limit(limit)

    projection = await _get_authenticated_review_projection(
        session_factory,
        project_id=normalized_project_id,
        schema_id=normalized_schema_id,
        schema_version_id=normalized_schema_version_id,
        orchestration_id=normalized_orchestration_id,
        consistency_check_application_id=normalized_application_id,
    )
    indexed = _build_review_item_index(projection)
    items = [
        _summary_from_index_entry(
            reviewed_fact=entry["reviewed_fact"],
            matched_field_keys=entry["matched_field_keys"],
        )
        for entry in indexed.values()
    ]
    if normalized_state != "all":
        items = [
            item for item in items if _review_item_state(indexed[item.fact_id]["reviewed_fact"]) == normalized_state
        ]
    sorted_items = _sort_review_items(items)[:normalized_limit]
    payload = BailianReviewItemsResponse(
        source_manifest_hash=projection.reviewed_projection_manifest_hash,
        payload_hash="",
        project_id=projection.project_id,
        schema_id=projection.schema_id,
        schema_version_id=projection.schema_version_id,
        orchestration_id=projection.orchestration_id,
        consistency_check_application_id=projection.consistency_check_application_id,
        reviewed_projection_manifest_hash=projection.reviewed_projection_manifest_hash,
        state=normalized_state,
        limit=normalized_limit,
        item_count=len(sorted_items),
        items=sorted_items,
    )
    return _finalize_payload(
        payload,
        tool_name="bailian_review_items",
        request_identity={
            "project_id": str(normalized_project_id),
            "schema_id": str(normalized_schema_id),
            "schema_version_id": str(normalized_schema_version_id),
            "orchestration_id": str(normalized_orchestration_id),
            "consistency_check_application_id": str(normalized_application_id),
            "state": normalized_state,
            "limit": normalized_limit,
        },
    )


async def get_review_item_detail(
    session_factory,
    *,
    project_id: object,
    fact_id: object,
    schema_id: object,
    schema_version_id: object,
    orchestration_id: object,
    consistency_check_application_id: object,
) -> BailianReviewItemDetailResponse:
    normalized_project_id = _require_uuid(project_id, field_name="project_id")
    normalized_fact_id = _require_uuid(fact_id, field_name="fact_id")
    normalized_schema_id = _require_uuid(schema_id, field_name="schema_id")
    normalized_schema_version_id = _require_uuid(
        schema_version_id,
        field_name="schema_version_id",
    )
    normalized_orchestration_id = _require_uuid(
        orchestration_id,
        field_name="orchestration_id",
    )
    normalized_application_id = _require_uuid(
        consistency_check_application_id,
        field_name="consistency_check_application_id",
    )

    projection = await _get_authenticated_review_projection(
        session_factory,
        project_id=normalized_project_id,
        schema_id=normalized_schema_id,
        schema_version_id=normalized_schema_version_id,
        orchestration_id=normalized_orchestration_id,
        consistency_check_application_id=normalized_application_id,
    )
    indexed = _build_review_item_index(projection)
    entry = indexed.get(normalized_fact_id)
    if entry is None:
        raise BailianReviewToolNotFoundError("bailian_review_item_not_found")
    reviewed_fact = entry["reviewed_fact"]
    payload = BailianReviewItemDetailResponse(
        source_manifest_hash=projection.reviewed_projection_manifest_hash,
        payload_hash="",
        project_id=projection.project_id,
        schema_id=projection.schema_id,
        schema_version_id=projection.schema_version_id,
        orchestration_id=projection.orchestration_id,
        extraction_run_id=projection.extraction_run_id,
        consistency_check_application_id=projection.consistency_check_application_id,
        source_consistency_application_id=projection.source_consistency_application_id,
        reviewed_projection_manifest_hash=projection.reviewed_projection_manifest_hash,
        fact_id=reviewed_fact.fact.fact_id,
        identity_hash=reviewed_fact.fact.identity_hash,
        subject_kind=reviewed_fact.fact.subject_kind,
        subject_key=reviewed_fact.fact.subject_key,
        subject_entity_id=reviewed_fact.fact.subject_entity_id,
        predicate_key=reviewed_fact.fact.predicate_key,
        scope_key=reviewed_fact.fact.scope_key,
        semantic_value_count=reviewed_fact.fact.semantic_group_count,
        fact_value_count=reviewed_fact.fact.fact_value_count,
        matched_field_keys=tuple(entry["matched_field_keys"]),
        review_state=reviewed_fact.review_state,
        resolution_basis=reviewed_fact.resolution_basis,
        current_decision_id=reviewed_fact.current_decision_id,
        current_decision_kind=reviewed_fact.current_decision_kind,
        effective_fact_value_ids=reviewed_fact.effective_fact_value_ids,
        requires_review=reviewed_fact.requires_review,
        value_groups=_serialize_reviewed_fact_values(reviewed_fact),
    )
    return _finalize_payload(
        payload,
        tool_name="bailian_review_item_detail",
        request_identity={
            "project_id": str(normalized_project_id),
            "schema_id": str(normalized_schema_id),
            "schema_version_id": str(normalized_schema_version_id),
            "orchestration_id": str(normalized_orchestration_id),
            "consistency_check_application_id": str(normalized_application_id),
            "fact_id": str(normalized_fact_id),
        },
    )


async def get_version_record(
    session_factory,
    *,
    project_id: object,
    project_version_id: object,
    subject_key: object,
) -> BailianVersionRecordResponse:
    normalized_project_id = _require_uuid(project_id, field_name="project_id")
    normalized_project_version_id = _require_uuid(
        project_version_id,
        field_name="project_version_id",
    )
    normalized_subject_key = _normalize_subject_key(subject_key)

    try:
        snapshot = await project_version_service.get_project_version_snapshot(
            session_factory,
            project_id=normalized_project_id,
            project_version_id=normalized_project_version_id,
        )
        authenticated_snapshot = project_version_service.authenticate_project_version_snapshot(
            snapshot
        )
    except project_version_service.ProjectVersionStateError as exc:
        if exc.args and exc.args[0] == "project_version_not_found":
            raise BailianReviewToolNotFoundError("bailian_version_record_not_found") from None
        raise BailianReviewToolStateError("bailian_review_tool_source_invalid") from None
    except project_version_service.ProjectVersionInvariantError:
        raise BailianReviewToolInvariantError(
            "bailian_review_tool_source_mismatch"
        ) from None

    for record in authenticated_snapshot.snapshot_json["records"]:
        if not isinstance(record, Mapping):
            raise BailianReviewToolInvariantError("bailian_review_tool_payload_invalid")
        if record.get("subject_key") == normalized_subject_key:
            record_json = _thaw_json_value(record)
            if not isinstance(record_json, dict):
                raise BailianReviewToolInvariantError(
                    "bailian_review_tool_payload_invalid"
                )
            payload = BailianVersionRecordResponse(
                source_manifest_hash=authenticated_snapshot.knowledge_view_manifest_hash,
                payload_hash="",
                project_id=authenticated_snapshot.project_id,
                project_version_id=authenticated_snapshot.id,
                version_no=authenticated_snapshot.version_no,
                is_current=authenticated_snapshot.is_current,
                schema_id=authenticated_snapshot.schema_id,
                schema_version_id=authenticated_snapshot.schema_version_id,
                orchestration_id=authenticated_snapshot.orchestration_id,
                extraction_run_id=authenticated_snapshot.extraction_run_id,
                consistency_check_application_id=authenticated_snapshot.consistency_check_application_id,
                source_consistency_application_id=authenticated_snapshot.source_consistency_application_id,
                knowledge_view_manifest_hash=authenticated_snapshot.knowledge_view_manifest_hash,
                subject_key=normalized_subject_key,
                record_json=record_json,
            )
            return _finalize_payload(
                payload,
                tool_name="bailian_version_record",
                request_identity={
                    "project_id": str(normalized_project_id),
                    "project_version_id": str(normalized_project_version_id),
                    "subject_key": normalized_subject_key,
                },
            )

    raise BailianReviewToolNotFoundError("bailian_version_record_not_found")
