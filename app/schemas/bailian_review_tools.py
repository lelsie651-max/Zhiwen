from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


BAILIAN_READ_TOOL_PAYLOAD_ALGORITHM_NAME = "bailian_read_tool_payload"
BAILIAN_READ_TOOL_PAYLOAD_ALGORITHM_VERSION = "1.0.0"


class BailianToolPayloadBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    algorithm_name: Literal["bailian_read_tool_payload"] = (
        BAILIAN_READ_TOOL_PAYLOAD_ALGORITHM_NAME
    )
    algorithm_version: Literal["1.0.0"] = BAILIAN_READ_TOOL_PAYLOAD_ALGORITHM_VERSION
    source_manifest_hash: str
    payload_hash: str


class BailianReviewItemSummary(BaseModel):
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


class BailianReviewItemsResponse(BailianToolPayloadBase):
    project_id: uuid.UUID
    schema_id: uuid.UUID
    schema_version_id: uuid.UUID
    orchestration_id: uuid.UUID
    consistency_check_application_id: uuid.UUID
    reviewed_projection_manifest_hash: str
    state: Literal["review_required", "resolved", "observation_only", "all"]
    limit: int = Field(ge=1, le=100)
    item_count: int = Field(ge=0)
    items: tuple[BailianReviewItemSummary, ...]


class BailianReviewItemDetailResponse(BailianToolPayloadBase):
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
    value_groups: tuple[dict[str, Any], ...]


class BailianVersionRecordResponse(BailianToolPayloadBase):
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
    record_json: dict[str, Any]
