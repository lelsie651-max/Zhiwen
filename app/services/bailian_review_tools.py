from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel

from app.schemas.bailian_review_tools import (
    BailianReviewItemDetailResponse,
    BailianReviewItemsResponse,
    BailianVersionRecordResponse,
)
import app.services.review_query as review_query_service

duplicate_grouping_service = review_query_service.duplicate_grouping_service


class BailianReviewToolError(Exception):
    """Base error for Bailian adapter reads."""


class BailianReviewToolStateError(BailianReviewToolError):
    """Raised when Bailian adapter inputs are invalid."""


class BailianReviewToolNotFoundError(BailianReviewToolError):
    """Raised when the requested Bailian adapter resource is missing."""


class BailianReviewToolInvariantError(BailianReviewToolError):
    """Raised when authenticated source projections drift or conflict."""


def _translate_review_query_error(error: Exception) -> BailianReviewToolError:
    detail = str(error)
    if isinstance(error, review_query_service.ReviewQueryStateError):
        if detail == "review_item_not_found":
            return BailianReviewToolNotFoundError("bailian_review_item_not_found")
        if detail == "version_record_not_found":
            return BailianReviewToolNotFoundError("bailian_version_record_not_found")
        if detail.startswith("review_query_"):
            return BailianReviewToolStateError(
                detail.replace("review_query_", "bailian_review_tool_", 1)
            )
        return BailianReviewToolStateError(detail)
    if isinstance(error, review_query_service.ReviewQueryNotFoundError):
        if detail == "review_item_not_found":
            return BailianReviewToolNotFoundError("bailian_review_item_not_found")
        if detail == "version_record_not_found":
            return BailianReviewToolNotFoundError("bailian_version_record_not_found")
        return BailianReviewToolNotFoundError(detail)
    if isinstance(error, review_query_service.ReviewQueryInvariantError):
        if detail.startswith("review_query_"):
            return BailianReviewToolInvariantError(
                detail.replace("review_query_", "bailian_review_tool_", 1)
            )
        if detail.startswith("review_item_"):
            return BailianReviewToolInvariantError(
                detail.replace("review_item_", "bailian_review_item_", 1)
            )
        if detail.startswith("version_record_"):
            return BailianReviewToolInvariantError(
                detail.replace("version_record_", "bailian_version_record_", 1)
            )
        return BailianReviewToolInvariantError(detail)
    raise TypeError("unsupported error type")


def _build_payload_hash(
    payload_model: BaseModel,
    *,
    tool_name: str,
    request_identity: Mapping[str, object],
) -> str:
    return review_query_service._build_payload_hash(
        payload_model,
        tool_name=tool_name,
        request_identity=request_identity,
    )


def authenticate_bailian_review_items_response(
    response: BailianReviewItemsResponse,
    *,
    request_identity: Mapping[str, object],
) -> BailianReviewItemsResponse:
    try:
        return review_query_service.authenticate_review_items_result(
            response,
            request_identity=request_identity,
            tool_name="bailian_review_items",
        )
    except review_query_service.ReviewQueryError as exc:
        raise _translate_review_query_error(exc) from None


def authenticate_bailian_review_item_detail_response(
    response: BailianReviewItemDetailResponse,
    *,
    request_identity: Mapping[str, object],
) -> BailianReviewItemDetailResponse:
    try:
        return review_query_service.authenticate_review_item_detail_result(
            response,
            request_identity=request_identity,
            tool_name="bailian_review_item_detail",
        )
    except review_query_service.ReviewQueryError as exc:
        raise _translate_review_query_error(exc) from None


def authenticate_bailian_version_record_response(
    response: BailianVersionRecordResponse,
    *,
    request_identity: Mapping[str, object],
) -> BailianVersionRecordResponse:
    try:
        return review_query_service.authenticate_version_record_response(
            response,
            request_identity=request_identity,
            tool_name="bailian_version_record",
        )
    except review_query_service.ReviewQueryError as exc:
        raise _translate_review_query_error(exc) from None


def _resign_for_bailian(
    payload_model: BaseModel,
    *,
    tool_name: str,
    request_identity: Mapping[str, object],
) -> BaseModel:
    return payload_model.model_copy(
        update={
            "payload_hash": _build_payload_hash(
                payload_model,
                tool_name=tool_name,
                request_identity=request_identity,
            )
        }
    )


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
    try:
        result = await review_query_service.list_review_items(
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
        raise _translate_review_query_error(exc) from None

    request_identity = {
        "project_id": str(result.project_id),
        "schema_id": str(result.schema_id),
        "schema_version_id": str(result.schema_version_id),
        "orchestration_id": str(result.orchestration_id),
        "consistency_check_application_id": str(result.consistency_check_application_id),
        "state": result.state,
        "limit": result.limit,
    }
    resigned = _resign_for_bailian(
        result,
        tool_name="bailian_review_items",
        request_identity=request_identity,
    )
    return authenticate_bailian_review_items_response(
        resigned,
        request_identity=request_identity,
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
    try:
        result = await review_query_service.get_review_item_detail(
            session_factory,
            project_id=project_id,
            fact_id=fact_id,
            schema_id=schema_id,
            schema_version_id=schema_version_id,
            orchestration_id=orchestration_id,
            consistency_check_application_id=consistency_check_application_id,
        )
    except review_query_service.ReviewQueryError as exc:
        raise _translate_review_query_error(exc) from None

    request_identity = {
        "project_id": str(result.project_id),
        "schema_id": str(result.schema_id),
        "schema_version_id": str(result.schema_version_id),
        "orchestration_id": str(result.orchestration_id),
        "consistency_check_application_id": str(result.consistency_check_application_id),
        "fact_id": str(result.fact_id),
    }
    resigned = _resign_for_bailian(
        result,
        tool_name="bailian_review_item_detail",
        request_identity=request_identity,
    )
    return authenticate_bailian_review_item_detail_response(
        resigned,
        request_identity=request_identity,
    )


async def get_version_record(
    session_factory,
    *,
    project_id: object,
    project_version_id: object,
    subject_key: object,
) -> BailianVersionRecordResponse:
    try:
        result = await review_query_service.get_version_record(
            session_factory,
            project_id=project_id,
            project_version_id=project_version_id,
            subject_key=subject_key,
        )
    except review_query_service.ReviewQueryError as exc:
        raise _translate_review_query_error(exc) from None

    request_identity = {
        "project_id": str(result.project_id),
        "project_version_id": str(result.project_version_id),
        "subject_key": result.subject_key,
    }
    resigned = _resign_for_bailian(
        result,
        tool_name="bailian_version_record",
        request_identity=request_identity,
    )
    return authenticate_bailian_version_record_response(
        resigned,
        request_identity=request_identity,
    )
