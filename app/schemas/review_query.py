from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer


REVIEW_QUERY_PAYLOAD_ALGORITHM_NAME = "bailian_read_tool_payload"
REVIEW_QUERY_PAYLOAD_ALGORITHM_VERSION = "1.0.0"


def _serialize_frozen_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _serialize_frozen_json(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_serialize_frozen_json(item) for item in value]
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    return value


class ReviewQueryPayloadBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    algorithm_name: Literal["bailian_read_tool_payload"] = (
        REVIEW_QUERY_PAYLOAD_ALGORITHM_NAME
    )
    algorithm_version: Literal["1.0.0"] = REVIEW_QUERY_PAYLOAD_ALGORITHM_VERSION
    source_manifest_hash: str
    payload_hash: str


class ReviewQueryItemSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fact_id: uuid.UUID
    subject_kind: str
    subject_key: str
    predicate_key: str
    scope_key: str | None
    matched_field_keys: tuple[str, ...]
    review_state: str
    resolution_basis: str
    requires_review: bool
    semantic_value_count: int
    fact_value_count: int
    evidence_count: int


class ReviewQueryItemsResult(ReviewQueryPayloadBase):
    project_id: uuid.UUID
    schema_id: uuid.UUID
    schema_version_id: uuid.UUID
    orchestration_id: uuid.UUID
    consistency_check_application_id: uuid.UUID
    reviewed_projection_manifest_hash: str
    state: Literal["review_required", "resolved", "observation_only", "all"]
    limit: int = Field(ge=1, le=100)
    item_count: int = Field(ge=0)
    items: tuple[ReviewQueryItemSummary, ...]


class ReviewQueryItemDetailResult(ReviewQueryPayloadBase):
    project_id: uuid.UUID
    schema_id: uuid.UUID
    schema_version_id: uuid.UUID
    orchestration_id: uuid.UUID
    extraction_run_id: uuid.UUID
    consistency_check_application_id: uuid.UUID
    source_consistency_application_id: uuid.UUID
    reviewed_projection_manifest_hash: str
    fact_id: uuid.UUID
    identity_hash: str
    subject_kind: str
    subject_key: str
    subject_entity_id: uuid.UUID | None
    predicate_key: str
    scope_key: str | None
    semantic_value_count: int = Field(ge=0)
    fact_value_count: int = Field(ge=0)
    matched_field_keys: tuple[str, ...]
    review_state: str
    resolution_basis: str
    current_decision_id: uuid.UUID | None
    current_decision_kind: str | None
    effective_fact_value_ids: tuple[uuid.UUID, ...]
    requires_review: bool
    value_groups: tuple[Mapping[str, Any], ...]

    @field_serializer("value_groups", when_used="json")
    def serialize_value_groups(self, value: tuple[Mapping[str, Any], ...]) -> object:
        return _serialize_frozen_json(value)


class VersionRecordQueryResult(ReviewQueryPayloadBase):
    project_id: uuid.UUID
    project_version_id: uuid.UUID
    version_no: int = Field(gt=0)
    is_current: bool
    schema_id: uuid.UUID
    schema_version_id: uuid.UUID
    orchestration_id: uuid.UUID
    extraction_run_id: uuid.UUID
    consistency_check_application_id: uuid.UUID
    source_consistency_application_id: uuid.UUID
    knowledge_view_manifest_hash: str
    subject_key: str
    record_json: Mapping[str, Any]

    @field_serializer("record_json", when_used="json")
    def serialize_record_json(self, value: Mapping[str, Any]) -> object:
        return _serialize_frozen_json(value)
