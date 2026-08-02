from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.database import get_async_session_factory
from app.dependencies.bailian_tools import authorize_bailian_tool_request
from app.schemas.bailian_review_tools import (
    BailianReviewItemDetailResponse,
    BailianReviewItemsResponse,
    BailianVersionRecordResponse,
)
import app.services.bailian_review_tools as bailian_review_tools_service


router = APIRouter(
    prefix="/api/v1/integrations/bailian",
    tags=["bailian-tools"],
    dependencies=[Depends(authorize_bailian_tool_request)],
)


def _translate_bailian_tool_error(error: Exception) -> HTTPException:
    if isinstance(error, bailian_review_tools_service.BailianReviewToolStateError):
        return HTTPException(status_code=400, detail=error.args[0])
    if isinstance(error, bailian_review_tools_service.BailianReviewToolNotFoundError):
        return HTTPException(status_code=404, detail=error.args[0])
    if isinstance(error, bailian_review_tools_service.BailianReviewToolInvariantError):
        return HTTPException(status_code=409, detail=error.args[0])
    raise error


@router.get(
    "/projects/{project_id}/review-items",
    response_model=BailianReviewItemsResponse,
    operation_id="bailianListReviewItems",
)
async def list_review_items(
    project_id: str,
    schema_id: str,
    schema_version_id: str,
    orchestration_id: str,
    consistency_check_application_id: str,
    state: str = "all",
    limit: str = "50",
    session_factory=Depends(get_async_session_factory),
) -> BailianReviewItemsResponse:
    try:
        return await bailian_review_tools_service.list_review_items(
            session_factory,
            project_id=project_id,
            schema_id=schema_id,
            schema_version_id=schema_version_id,
            orchestration_id=orchestration_id,
            consistency_check_application_id=consistency_check_application_id,
            state=state,
            limit=limit,
        )
    except (
        bailian_review_tools_service.BailianReviewToolStateError,
        bailian_review_tools_service.BailianReviewToolNotFoundError,
        bailian_review_tools_service.BailianReviewToolInvariantError,
    ) as exc:
        raise _translate_bailian_tool_error(exc) from None


@router.get(
    "/projects/{project_id}/review-items/{fact_id}",
    response_model=BailianReviewItemDetailResponse,
    operation_id="bailianGetReviewItemDetail",
)
async def get_review_item_detail(
    project_id: str,
    fact_id: str,
    schema_id: str,
    schema_version_id: str,
    orchestration_id: str,
    consistency_check_application_id: str,
    session_factory=Depends(get_async_session_factory),
) -> BailianReviewItemDetailResponse:
    try:
        return await bailian_review_tools_service.get_review_item_detail(
            session_factory,
            project_id=project_id,
            fact_id=fact_id,
            schema_id=schema_id,
            schema_version_id=schema_version_id,
            orchestration_id=orchestration_id,
            consistency_check_application_id=consistency_check_application_id,
        )
    except (
        bailian_review_tools_service.BailianReviewToolStateError,
        bailian_review_tools_service.BailianReviewToolNotFoundError,
        bailian_review_tools_service.BailianReviewToolInvariantError,
    ) as exc:
        raise _translate_bailian_tool_error(exc) from None


@router.get(
    "/projects/{project_id}/versions/{project_version_id}/records/{subject_key:path}",
    response_model=BailianVersionRecordResponse,
    operation_id="bailianGetVersionRecord",
)
async def get_version_record(
    project_id: str,
    project_version_id: str,
    subject_key: str,
    session_factory=Depends(get_async_session_factory),
) -> BailianVersionRecordResponse:
    try:
        return await bailian_review_tools_service.get_version_record(
            session_factory,
            project_id=project_id,
            project_version_id=project_version_id,
            subject_key=subject_key,
        )
    except (
        bailian_review_tools_service.BailianReviewToolStateError,
        bailian_review_tools_service.BailianReviewToolNotFoundError,
        bailian_review_tools_service.BailianReviewToolInvariantError,
    ) as exc:
        raise _translate_bailian_tool_error(exc) from None
