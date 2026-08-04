from __future__ import annotations

from dataclasses import dataclass
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.database import get_async_session_factory, get_db_session
from app.dependencies.auth import require_current_user, verify_api_csrf_token
from app.models.project_member import ProjectMember
from app.models.user import User
from app.repositories import project as project_repository
from app.schemas.frontend_api import (
    FrontendCurrentUserResponse,
    FrontendReviewItemDetailResponse,
    FrontendReviewItemsResponse,
    FrontendReviewDecisionWriteRequest,
    FrontendReviewDecisionWriteResponse,
    FrontendVersionRecordResponse,
)
from app.schemas.user import UserRead
from app.services import consistency_review as consistency_review_service
from app.services import frontend_api as frontend_api_service
import app.services.review_query as review_query_service
from app.utils.csrf import ensure_csrf_token


router = APIRouter(prefix="/api/v1", tags=["frontend-api"])


@dataclass(frozen=True, slots=True)
class ProjectAccessContext:
    current_user: User
    membership: ProjectMember


async def require_project_access(
    project_id: uuid.UUID,
    current_user: User = Depends(require_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> ProjectAccessContext:
    membership = await project_repository.get_project_member_for_user_by_project_id(
        session,
        user_id=current_user.id,
        project_id=project_id,
    )
    if membership is None:
        raise HTTPException(status_code=403, detail="project_access_denied")
    return ProjectAccessContext(current_user=current_user, membership=membership)


def _map_review_query_error(error: Exception) -> HTTPException:
    detail = str(error)
    if isinstance(error, review_query_service.ReviewQueryStateError):
        return HTTPException(status_code=422, detail=detail)
    if isinstance(error, review_query_service.ReviewQueryNotFoundError):
        return HTTPException(status_code=404, detail=detail)
    if isinstance(error, review_query_service.ReviewQueryInvariantError):
        return HTTPException(status_code=409, detail="frontend_api_source_invalid")
    raise TypeError("unsupported error type")


def _map_consistency_review_error(error: Exception) -> HTTPException:
    detail = str(error)
    if isinstance(error, consistency_review_service.ConsistencyReviewInvariantError):
        return HTTPException(
            status_code=409,
            detail="frontend_review_decision_source_invalid",
        )
    if isinstance(error, consistency_review_service.ConsistencyReviewStateError):
        if detail in {
            "consistency_review_actor_not_found",
            "consistency_review_actor_membership_not_found",
            "consistency_review_actor_permission_denied",
        }:
            return HTTPException(status_code=403, detail=detail)
        if detail == "consistency_review_stale_decision":
            return HTTPException(status_code=409, detail=detail)
        return HTTPException(status_code=422, detail=detail)
    raise TypeError("unsupported error type")


def _map_frontend_api_error(error: Exception) -> HTTPException:
    detail = str(error)
    if isinstance(error, frontend_api_service.FrontendAPINotFoundError):
        return HTTPException(status_code=404, detail=detail)
    if isinstance(error, frontend_api_service.FrontendAPIInvariantError):
        return HTTPException(status_code=409, detail="frontend_api_source_invalid")
    if isinstance(error, frontend_api_service.FrontendAPIStateError):
        if detail in {
            "frontend_review_item_decision_target_invalid",
            "frontend_review_item_assessment_mismatch",
        }:
            return HTTPException(status_code=409, detail=detail)
        return HTTPException(status_code=422, detail=detail)
    raise TypeError("unsupported error type")


@router.get(
    "/me",
    response_model=FrontendCurrentUserResponse,
    operation_id="getCurrentUser",
)
async def get_current_user(
    request: Request,
    current_user: User = Depends(require_current_user),
) -> FrontendCurrentUserResponse:
    return FrontendCurrentUserResponse(
        user=UserRead.model_validate(current_user),
        csrf_token=ensure_csrf_token(request),
    )


@router.get(
    "/projects/{project_id}/review-items",
    response_model=FrontendReviewItemsResponse,
    operation_id="frontendListReviewItems",
)
async def list_review_items(
    project_id: uuid.UUID,
    schema_id: uuid.UUID,
    schema_version_id: uuid.UUID,
    orchestration_id: uuid.UUID,
    consistency_check_application_id: uuid.UUID,
    state: Literal["review_required", "resolved", "observation_only", "all"] = Query(
        default="all"
    ),
    limit: int = Query(default=50, ge=1, le=100),
    _access: ProjectAccessContext = Depends(require_project_access),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_async_session_factory),
) -> FrontendReviewItemsResponse:
    try:
        return await frontend_api_service.list_review_items(
            session_factory,
            project_id=project_id,
            schema_id=schema_id,
            schema_version_id=schema_version_id,
            orchestration_id=orchestration_id,
            consistency_check_application_id=consistency_check_application_id,
            state=state,
            limit=limit,
        )
    except review_query_service.ReviewQueryError as exc:
        raise _map_review_query_error(exc) from None


@router.get(
    "/projects/{project_id}/review-items/{fact_id}",
    response_model=FrontendReviewItemDetailResponse,
    operation_id="frontendGetReviewItemDetail",
)
async def get_review_item_detail(
    project_id: uuid.UUID,
    fact_id: uuid.UUID,
    schema_id: uuid.UUID,
    schema_version_id: uuid.UUID,
    orchestration_id: uuid.UUID,
    consistency_check_application_id: uuid.UUID,
    _access: ProjectAccessContext = Depends(require_project_access),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_async_session_factory),
) -> FrontendReviewItemDetailResponse:
    try:
        return await frontend_api_service.get_review_item_detail(
            session_factory,
            project_id=project_id,
            fact_id=fact_id,
            schema_id=schema_id,
            schema_version_id=schema_version_id,
            orchestration_id=orchestration_id,
            consistency_check_application_id=consistency_check_application_id,
        )
    except review_query_service.ReviewQueryError as exc:
        raise _map_review_query_error(exc) from None


@router.post(
    "/projects/{project_id}/review-items/{fact_id}/decisions",
    response_model=FrontendReviewDecisionWriteResponse,
    operation_id="frontendAppendReviewDecision",
    dependencies=[Depends(verify_api_csrf_token)],
)
async def append_review_decision(
    project_id: uuid.UUID,
    fact_id: uuid.UUID,
    payload: FrontendReviewDecisionWriteRequest,
    access: ProjectAccessContext = Depends(require_project_access),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_async_session_factory),
) -> FrontendReviewDecisionWriteResponse:
    try:
        return await frontend_api_service.write_review_decision(
            session_factory,
            project_id=project_id,
            fact_id=fact_id,
            schema_id=payload.schema_id,
            schema_version_id=payload.schema_version_id,
            orchestration_id=payload.orchestration_id,
            consistency_check_application_id=payload.consistency_check_application_id,
            assessment_id=payload.assessment_id,
            actor_id=access.current_user.id,
            expected_current_decision_id=payload.expected_current_decision_id,
            decision_kind=payload.decision_kind.value,
            selected_fact_value_ids=payload.selected_fact_value_ids,
            comment=payload.comment,
        )
    except frontend_api_service.FrontendAPIError as exc:
        raise _map_frontend_api_error(exc) from None
    except consistency_review_service.ConsistencyReviewError as exc:
        raise _map_consistency_review_error(exc) from None
    except review_query_service.ReviewQueryError as exc:
        raise _map_review_query_error(exc) from None


@router.get(
    "/projects/{project_id}/versions/{project_version_id}/records/{subject_key:path}",
    response_model=FrontendVersionRecordResponse,
    operation_id="frontendGetVersionRecord",
)
async def get_version_record(
    project_id: uuid.UUID,
    project_version_id: uuid.UUID,
    subject_key: str,
    _access: ProjectAccessContext = Depends(require_project_access),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_async_session_factory),
) -> FrontendVersionRecordResponse:
    try:
        return await frontend_api_service.get_version_record(
            session_factory,
            project_id=project_id,
            project_version_id=project_version_id,
            subject_key=subject_key,
        )
    except review_query_service.ReviewQueryError as exc:
        raise _map_review_query_error(exc) from None
