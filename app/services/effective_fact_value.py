from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.consistency_projection import (
    ConsistencyReviewProjection,
    ConsistencyReviewProjectionItem,
    ConsistencyReviewProjectionMember,
)
from app.schemas.effective_fact_value import (
    EffectiveFactValueProjection,
    EffectiveFactValueProjectionItem,
)
from app.services import consistency_projection as projection_service


class EffectiveFactValueProjectionError(Exception):
    """Base class for effective fact value projection failures."""


class EffectiveFactValueProjectionStateError(EffectiveFactValueProjectionError):
    """Raised when the requested application or projection input is invalid."""


class EffectiveFactValueProjectionInvariantError(EffectiveFactValueProjectionError):
    """Raised when the read-only projection contract diverges from authenticated data."""


def _require_uuid(value: uuid.UUID, *, field_name: str) -> uuid.UUID:
    if not isinstance(value, uuid.UUID):
        raise EffectiveFactValueProjectionStateError(
            f"effective_fact_value_projection_{field_name}_invalid"
        )
    return value


def _map_projection_error(error: Exception) -> Exception:
    if isinstance(error, projection_service.ConsistencyProjectionInvariantError):
        return EffectiveFactValueProjectionInvariantError(
            "effective_fact_value_projection_immutable_ledger_mismatch"
        )
    if isinstance(error, projection_service.ConsistencyProjectionStateError):
        message = str(error)
        if message == "consistency_review_projection_application_not_found":
            return EffectiveFactValueProjectionStateError(
                "effective_fact_value_projection_application_not_found"
            )
        if message == "consistency_review_projection_project_id_mismatch":
            return EffectiveFactValueProjectionStateError(
                "effective_fact_value_projection_project_id_mismatch"
            )
        return EffectiveFactValueProjectionStateError(
            "effective_fact_value_projection_source_not_authenticated"
        )
    return error


def _immutable_mismatch() -> None:
    raise EffectiveFactValueProjectionInvariantError(
        "effective_fact_value_projection_immutable_ledger_mismatch"
    )


def _build_member_index(
    members: Sequence[ConsistencyReviewProjectionMember],
) -> tuple[dict[uuid.UUID, int], set[uuid.UUID]]:
    order_by_fact_value_id: dict[uuid.UUID, int] = {}
    seen_fact_value_ids: set[uuid.UUID] = set()
    for index, member in enumerate(members):
        if member.fact_value_id in seen_fact_value_ids:
            _immutable_mismatch()
        seen_fact_value_ids.add(member.fact_value_id)
        order_by_fact_value_id[member.fact_value_id] = index
    return order_by_fact_value_id, seen_fact_value_ids


def _validate_projection_item_contract(
    item: ConsistencyReviewProjectionItem,
) -> tuple[dict[uuid.UUID, int], set[uuid.UUID]]:
    order_by_fact_value_id, member_fact_value_ids = _build_member_index(item.members)
    current_decision = item.current_decision
    history = item.decision_history
    if current_decision is None:
        if history:
            _immutable_mismatch()
    else:
        if not history or history[-1] != current_decision:
            _immutable_mismatch()

    explicit_selected_ids = item.selected_fact_value_ids
    seen_selected_ids: set[uuid.UUID] = set()
    for selected_id in explicit_selected_ids:
        if selected_id in seen_selected_ids:
            _immutable_mismatch()
        if selected_id not in member_fact_value_ids:
            _immutable_mismatch()
        seen_selected_ids.add(selected_id)

    selected_member_ids = tuple(
        member.fact_value_id
        for member in item.members
        if member.selected_by_current_decision
    )
    selected_order_pairs = tuple(
        (member.current_selection_order, member.fact_value_id)
        for member in item.members
        if member.current_selection_order is not None
    )
    if len(selected_order_pairs) != len(selected_member_ids):
        _immutable_mismatch()
    if selected_order_pairs:
        sorted_pairs = tuple(sorted(selected_order_pairs))
        expected_orders = tuple(range(len(sorted_pairs)))
        if tuple(order for order, _fact_value_id in sorted_pairs) != expected_orders:
            _immutable_mismatch()
        if tuple(fact_value_id for _order, fact_value_id in sorted_pairs) != explicit_selected_ids:
            _immutable_mismatch()
    elif explicit_selected_ids:
        _immutable_mismatch()

    if current_decision is None:
        if explicit_selected_ids:
            _immutable_mismatch()
        return order_by_fact_value_id, member_fact_value_ids

    if current_decision.decision_kind in {"confirm_compatible", "defer"}:
        if explicit_selected_ids:
            _immutable_mismatch()
        if selected_member_ids:
            _immutable_mismatch()
    elif current_decision.decision_kind in {"select_one", "keep_multiple"}:
        if current_decision.selected_fact_value_ids != explicit_selected_ids:
            _immutable_mismatch()
    else:
        _immutable_mismatch()

    return order_by_fact_value_id, member_fact_value_ids


def _resolve_effective_fact_value_ids(
    item: ConsistencyReviewProjectionItem,
    *,
    member_order_by_fact_value_id: dict[uuid.UUID, int],
) -> tuple[str, str, tuple[uuid.UUID, ...]]:
    current_decision = item.current_decision
    if current_decision is None:
        if item.verdict == "compatible":
            return "unreviewed_compatible", "none", ()
        return "pending_review", "none", ()

    if current_decision.decision_kind == "select_one":
        return "resolved", "human_selection", item.selected_fact_value_ids
    if current_decision.decision_kind == "keep_multiple":
        return "resolved", "human_selection", item.selected_fact_value_ids
    if current_decision.decision_kind == "confirm_compatible":
        return (
            "resolved",
            "human_confirmed_compatibility",
            tuple(
                member.fact_value_id
                for member in sorted(
                    item.members,
                    key=lambda member: member_order_by_fact_value_id[member.fact_value_id],
                )
            ),
        )
    if current_decision.decision_kind == "defer":
        return "deferred", "none", ()
    _immutable_mismatch()


async def get_effective_fact_value_projection(
    session_factory: Callable[[], AsyncSession],
    *,
    project_id: uuid.UUID,
    consistency_check_application_id: uuid.UUID,
) -> EffectiveFactValueProjection:
    project_id = _require_uuid(project_id, field_name="project_id")
    consistency_check_application_id = _require_uuid(
        consistency_check_application_id,
        field_name="consistency_check_application_id",
    )
    try:
        review_projection = await projection_service.get_consistency_review_projection(
            session_factory,
            project_id=project_id,
            consistency_check_application_id=consistency_check_application_id,
        )
    except Exception as error:
        mapped = _map_projection_error(error)
        if mapped is error:
            raise
        raise mapped from None

    return _build_effective_fact_value_projection(review_projection)


def _build_effective_fact_value_projection(
    review_projection: ConsistencyReviewProjection,
) -> EffectiveFactValueProjection:
    items: list[EffectiveFactValueProjectionItem] = []
    seen_fact_ids: set[uuid.UUID] = set()
    for review_item in review_projection.items:
        if review_item.fact_id in seen_fact_ids:
            _immutable_mismatch()
        seen_fact_ids.add(review_item.fact_id)
        member_order_by_fact_value_id, _member_fact_value_ids = (
            _validate_projection_item_contract(review_item)
        )
        resolution_status, resolution_basis, effective_fact_value_ids = (
            _resolve_effective_fact_value_ids(
                review_item,
                member_order_by_fact_value_id=member_order_by_fact_value_id,
            )
        )
        if len(effective_fact_value_ids) != len(set(effective_fact_value_ids)):
            _immutable_mismatch()
        if any(
            fact_value_id not in member_order_by_fact_value_id
            for fact_value_id in effective_fact_value_ids
        ):
            _immutable_mismatch()
        items.append(
            EffectiveFactValueProjectionItem(
                fact_id=review_item.fact_id,
                candidate_id=review_item.candidate_id,
                assessment_id=review_item.assessment_id,
                agent_verdict=review_item.verdict,
                review_status=review_item.review_status,
                resolution_status=resolution_status,
                resolution_basis=resolution_basis,
                current_decision_id=(
                    None
                    if review_item.current_decision is None
                    else review_item.current_decision.decision_id
                ),
                current_decision_kind=(
                    None
                    if review_item.current_decision is None
                    else review_item.current_decision.decision_kind
                ),
                effective_fact_value_ids=effective_fact_value_ids,
                candidate_members=review_item.members,
            )
        )

    resolved_count = sum(
        1 for item in items if item.resolution_status == "resolved"
    )
    pending_count = sum(
        1
        for item in items
        if item.resolution_status in {"pending_review", "unreviewed_compatible"}
    )
    deferred_count = sum(
        1 for item in items if item.resolution_status == "deferred"
    )
    return EffectiveFactValueProjection(
        project_id=review_projection.project_id,
        consistency_check_application_id=review_projection.consistency_check_application_id,
        source_consistency_application_id=review_projection.source_consistency_application_id,
        result_manifest_hash=review_projection.result_manifest_hash,
        fact_count=len(items),
        resolved_count=resolved_count,
        pending_count=pending_count,
        deferred_count=deferred_count,
        items=tuple(items),
    )
