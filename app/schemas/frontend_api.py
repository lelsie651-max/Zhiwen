from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field

from app.models.consistency_review import ConsistencyReviewDecisionKind
from app.schemas.bailian_review_tools import BailianReviewItemDetailResponse
from app.schemas.user import UserRead


class FrontendCurrentUserResponse(BaseModel):
    user: UserRead
    csrf_token: str = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")


class FrontendReviewDecisionWriteRequest(BaseModel):
    schema_id: uuid.UUID
    schema_version_id: uuid.UUID
    orchestration_id: uuid.UUID
    consistency_check_application_id: uuid.UUID
    assessment_id: uuid.UUID
    expected_current_decision_id: uuid.UUID | None = None
    decision_kind: ConsistencyReviewDecisionKind
    selected_fact_value_ids: tuple[uuid.UUID, ...] = ()
    comment: str | None = None

    model_config = ConfigDict(extra="forbid")


class FrontendReviewDecisionWriteResponse(BaseModel):
    decision_id: uuid.UUID
    decision_no: int = Field(gt=0)
    supersedes_decision_id: uuid.UUID | None = None
    decision_manifest_hash: str = Field(min_length=64, max_length=64)
    selected_fact_value_ids: tuple[uuid.UUID, ...]
    created_new: bool
    current_state: BailianReviewItemDetailResponse

    model_config = ConfigDict(extra="forbid", frozen=True)
