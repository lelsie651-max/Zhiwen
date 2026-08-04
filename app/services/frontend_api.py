from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.frontend_api import (
    FrontendReviewDecisionWriteResponse,
    FrontendReviewItemDetailResponse,
    FrontendReviewItemsResponse,
    FrontendVersionRecordResponse,
)
import app.services.consistency_review as consistency_review_service
import app.services.dynamic_schema_review_projection as review_projection_service
import app.services.review_query as review_query_service


class FrontendAPIError(Exception):
    """Base error for frontend API composition helpers."""


class FrontendAPIStateError(FrontendAPIError):
    """Raised when the frontend request shape is invalid."""


class FrontendAPINotFoundError(FrontendAPIError):
    """Raised when the requested frontend resource is not found."""


class FrontendAPIInvariantError(FrontendAPIError):
    """Raised when authenticated projection bindings drift."""


@dataclass(frozen=True, slots=True)
class _FrontendReviewFactContext:
    fact_id: uuid.UUID
    assessment_id: uuid.UUID
    requires_review: bool


async def list_review_items(
    session_factory: Callable[[], AsyncSession],
    *,
    project_id: uuid.UUID,
    schema_id: uuid.UUID,
    schema_version_id: uuid.UUID,
    orchestration_id: uuid.UUID,
    consistency_check_application_id: uuid.UUID,
    state: str,
    limit: int,
) -> FrontendReviewItemsResponse:
    return await review_query_service.list_review_items(
        session_factory,
        project_id=project_id,
        schema_id=schema_id,
        schema_version_id=schema_version_id,
        orchestration_id=orchestration_id,
        consistency_check_application_id=consistency_check_application_id,
        state=state,
        limit=limit,
    )


async def get_review_item_detail(
    session_factory: Callable[[], AsyncSession],
    *,
    project_id: uuid.UUID,
    fact_id: uuid.UUID,
    schema_id: uuid.UUID,
    schema_version_id: uuid.UUID,
    orchestration_id: uuid.UUID,
    consistency_check_application_id: uuid.UUID,
) -> FrontendReviewItemDetailResponse:
    return await review_query_service.get_review_item_detail(
        session_factory,
        project_id=project_id,
        fact_id=fact_id,
        schema_id=schema_id,
        schema_version_id=schema_version_id,
        orchestration_id=orchestration_id,
        consistency_check_application_id=consistency_check_application_id,
    )


async def write_review_decision(
    session_factory: Callable[[], AsyncSession],
    *,
    project_id: uuid.UUID,
    fact_id: uuid.UUID,
    schema_id: uuid.UUID,
    schema_version_id: uuid.UUID,
    orchestration_id: uuid.UUID,
    consistency_check_application_id: uuid.UUID,
    assessment_id: uuid.UUID,
    actor_id: uuid.UUID,
    expected_current_decision_id: uuid.UUID | None,
    decision_kind: str,
    selected_fact_value_ids: tuple[uuid.UUID, ...],
    comment: str | None,
) -> FrontendReviewDecisionWriteResponse:
    fact_context = await _get_review_fact_context(
        session_factory,
        project_id=project_id,
        fact_id=fact_id,
        schema_id=schema_id,
        schema_version_id=schema_version_id,
        orchestration_id=orchestration_id,
        consistency_check_application_id=consistency_check_application_id,
    )
    if not fact_context.requires_review:
        raise FrontendAPIStateError("frontend_review_item_decision_target_invalid")
    if fact_context.assessment_id != assessment_id:
        raise FrontendAPIStateError("frontend_review_item_assessment_mismatch")

    decision_result = await consistency_review_service.append_consistency_review_decision(
        session_factory,
        project_id=project_id,
        consistency_check_application_id=consistency_check_application_id,
        assessment_id=assessment_id,
        actor_id=actor_id,
        expected_current_decision_id=expected_current_decision_id,
        decision_kind=decision_kind,
        selected_fact_value_ids=selected_fact_value_ids,
        comment=comment,
    )
    current_state = await get_review_item_detail(
        session_factory,
        project_id=project_id,
        fact_id=fact_id,
        schema_id=schema_id,
        schema_version_id=schema_version_id,
        orchestration_id=orchestration_id,
        consistency_check_application_id=consistency_check_application_id,
    )
    return FrontendReviewDecisionWriteResponse(
        decision_id=decision_result.decision_id,
        decision_no=decision_result.decision_no,
        supersedes_decision_id=decision_result.supersedes_decision_id,
        decision_manifest_hash=decision_result.decision_manifest_hash,
        selected_fact_value_ids=decision_result.selected_fact_value_ids,
        created_new=decision_result.created_new,
        current_state=current_state,
    )


async def get_version_record(
    session_factory: Callable[[], AsyncSession],
    *,
    project_id: uuid.UUID,
    project_version_id: uuid.UUID,
    subject_key: str,
) -> FrontendVersionRecordResponse:
    return await review_query_service.get_version_record(
        session_factory,
        project_id=project_id,
        project_version_id=project_version_id,
        subject_key=subject_key,
    )


async def _get_review_fact_context(
    session_factory: Callable[[], AsyncSession],
    *,
    project_id: uuid.UUID,
    fact_id: uuid.UUID,
    schema_id: uuid.UUID,
    schema_version_id: uuid.UUID,
    orchestration_id: uuid.UUID,
    consistency_check_application_id: uuid.UUID,
) -> _FrontendReviewFactContext:
    try:
        projection = await review_projection_service.project_reviewed_orchestration_ufl_to_dynamic_schema(
            session_factory,
            project_id=project_id,
            schema_id=schema_id,
            schema_version_id=schema_version_id,
            orchestration_id=orchestration_id,
            consistency_check_application_id=consistency_check_application_id,
        )
        authenticated_projection = (
            review_projection_service.authenticate_dynamic_schema_review_projection(
                projection
            )
        )
    except review_projection_service.DynamicSchemaReviewProjectionStateError as exc:
        raise FrontendAPIStateError(str(exc)) from None
    except review_projection_service.DynamicSchemaReviewProjectionInvariantError as exc:
        raise FrontendAPIInvariantError(str(exc)) from None

    matched_context: _FrontendReviewFactContext | None = None
    for record in authenticated_projection.records:
        for field in record.fields:
            for reviewed_fact in field.reviewed_facts:
                if reviewed_fact.fact.fact_id != fact_id:
                    continue
                if reviewed_fact.assessment_id is None:
                    raise FrontendAPIStateError(
                        "frontend_review_item_decision_target_invalid"
                    )
                candidate = _FrontendReviewFactContext(
                    fact_id=reviewed_fact.fact.fact_id,
                    assessment_id=reviewed_fact.assessment_id,
                    requires_review=reviewed_fact.requires_review,
                )
                if matched_context is not None and matched_context != candidate:
                    raise FrontendAPIInvariantError(
                        "frontend_review_item_projection_mismatch"
                    )
                matched_context = candidate

    if matched_context is None:
        raise FrontendAPINotFoundError("frontend_review_item_not_found")
    return matched_context
