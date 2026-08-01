from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import uuid

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.consistency_review import (
    ConsistencyReviewDecision,
    ConsistencyReviewDecisionKind,
    ConsistencyReviewDecisionSelection,
)
from app.models.project_member import ProjectMemberRole
from app.repositories import consistency_review as consistency_review_repository
from app.schemas.consistency_review import (
    AppendConsistencyReviewDecisionResult,
    ConsistencyReviewCandidateMemberRecord,
    ConsistencyReviewDecisionLedgerRecord,
    ConsistencyReviewDecisionSelectionLedgerRecord,
)
from app.services import consistency_check_persistence as persistence_service
from app.services import fact_value_duplicate_grouping as duplicate_grouping_service
from app.utils.validation import normalize_text


_HANDLED_DECISION_CONSTRAINTS = frozenset(
    {
        "uq_ccrevd_manifest_hash",
        "uq_ccrevd_asmt_dec_no",
        "uq_ccrevd_supersedes_id",
    }
)
_ALLOWED_ROLES = frozenset(
    {
        ProjectMemberRole.OWNER.value,
        ProjectMemberRole.EDITOR.value,
    }
)


class ConsistencyReviewError(Exception):
    """Base class for consistency review append failures."""


class ConsistencyReviewStateError(ConsistencyReviewError):
    """Raised when inputs, permissions, or current ledger state are invalid."""


class ConsistencyReviewInvariantError(ConsistencyReviewError):
    """Raised when immutable ledgers diverge from the authoritative chain."""


@dataclass(frozen=True, slots=True)
class _NormalizedRequest:
    project_id: uuid.UUID
    consistency_check_application_id: uuid.UUID
    assessment_id: uuid.UUID
    actor_id: uuid.UUID
    expected_current_decision_id: uuid.UUID | None
    decision_kind: str
    selected_fact_value_ids: tuple[uuid.UUID, ...]
    comment: str | None


@dataclass(frozen=True, slots=True)
class _ValidatedDecisionChainEntry:
    decision: ConsistencyReviewDecisionLedgerRecord
    selected_fact_value_ids: tuple[uuid.UUID, ...]


@dataclass(frozen=True, slots=True)
class _DecisionWritePlan:
    project_id: uuid.UUID
    consistency_check_application_id: uuid.UUID
    assessment_id: uuid.UUID
    source_consistency_application_id: uuid.UUID
    source_consistency_candidate_id: uuid.UUID
    actor_id: uuid.UUID
    decision_no: int
    supersedes_decision_id: uuid.UUID | None
    decision_kind: str
    selected_fact_value_ids: tuple[uuid.UUID, ...]
    comment: str | None
    decision_manifest_hash: str


def _get_integrity_constraint_name(error: IntegrityError) -> str | None:
    diag = getattr(error.orig, "diag", None)
    constraint_name = getattr(diag, "constraint_name", None)
    if constraint_name is None or not isinstance(constraint_name, str):
        return None
    return constraint_name


def _immutable_mismatch() -> None:
    raise ConsistencyReviewInvariantError(
        "consistency_review_immutable_ledger_mismatch"
    )


def _require_uuid(value: object, *, field_name: str) -> uuid.UUID:
    if not isinstance(value, uuid.UUID):
        raise ConsistencyReviewStateError(f"consistency_review_{field_name}_invalid")
    return value


def _normalize_decision_kind(value: object) -> str:
    if not isinstance(value, str):
        raise ConsistencyReviewStateError("consistency_review_decision_kind_invalid")
    normalized = normalize_text(value)
    if not normalized:
        raise ConsistencyReviewStateError("consistency_review_decision_kind_invalid")
    allowed = {kind.value for kind in ConsistencyReviewDecisionKind}
    if normalized not in allowed:
        raise ConsistencyReviewStateError("consistency_review_decision_kind_invalid")
    return normalized


def _normalize_selected_fact_value_ids(value: object) -> tuple[uuid.UUID, ...]:
    if not isinstance(value, (list, tuple)):
        raise ConsistencyReviewStateError(
            "consistency_review_selected_fact_value_ids_invalid"
        )
    normalized: list[uuid.UUID] = []
    seen: set[uuid.UUID] = set()
    for item in value:
        if not isinstance(item, uuid.UUID):
            raise ConsistencyReviewStateError(
                "consistency_review_selected_fact_value_ids_invalid"
            )
        if item in seen:
            raise ConsistencyReviewStateError(
                "consistency_review_selected_fact_value_ids_duplicate"
            )
        seen.add(item)
        normalized.append(item)
    return tuple(normalized)


def _normalize_comment(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConsistencyReviewStateError("consistency_review_comment_invalid")
    normalized = normalize_text(value)
    if not normalized:
        return None
    if len(normalized) > 2000:
        raise ConsistencyReviewStateError("consistency_review_comment_invalid")
    return normalized


def _normalize_request(
    *,
    project_id: object,
    consistency_check_application_id: object,
    assessment_id: object,
    actor_id: object,
    expected_current_decision_id: object,
    decision_kind: object,
    selected_fact_value_ids: object,
    comment: object,
) -> _NormalizedRequest:
    if expected_current_decision_id is not None and not isinstance(
        expected_current_decision_id,
        uuid.UUID,
    ):
        raise ConsistencyReviewStateError(
            "consistency_review_expected_current_decision_id_invalid"
        )
    return _NormalizedRequest(
        project_id=_require_uuid(project_id, field_name="project_id"),
        consistency_check_application_id=_require_uuid(
            consistency_check_application_id,
            field_name="consistency_check_application_id",
        ),
        assessment_id=_require_uuid(assessment_id, field_name="assessment_id"),
        actor_id=_require_uuid(actor_id, field_name="actor_id"),
        expected_current_decision_id=expected_current_decision_id,
        decision_kind=_normalize_decision_kind(decision_kind),
        selected_fact_value_ids=_normalize_selected_fact_value_ids(
            selected_fact_value_ids
        ),
        comment=_normalize_comment(comment),
    )


def _validate_selection_shape(
    *,
    decision_kind: str,
    selected_fact_value_ids: Sequence[uuid.UUID],
) -> None:
    count = len(selected_fact_value_ids)
    if decision_kind == ConsistencyReviewDecisionKind.SELECT_ONE.value:
        if count != 1:
            raise ConsistencyReviewStateError(
                "consistency_review_selected_fact_value_ids_shape_invalid"
            )
        return
    if decision_kind == ConsistencyReviewDecisionKind.KEEP_MULTIPLE.value:
        if count < 2 or count > 200:
            raise ConsistencyReviewStateError(
                "consistency_review_selected_fact_value_ids_shape_invalid"
            )
        return
    if count != 0:
        raise ConsistencyReviewStateError(
            "consistency_review_selected_fact_value_ids_shape_invalid"
        )


def _build_member_snapshot(
    members: Sequence[object],
) -> tuple[ConsistencyReviewCandidateMemberRecord, ...]:
    return tuple(
        ConsistencyReviewCandidateMemberRecord(
            consistency_application_id=member.consistency_application_id,
            candidate_id=member.candidate_id,
            fact_value_id=member.fact_value_id,
            source_batch_id=member.source_batch_id,
            semantic_key_hash=member.semantic_key_hash,
        )
        for member in members
    )


def _build_decision_manifest_hash(
    *,
    project_id: uuid.UUID,
    consistency_check_application_id: uuid.UUID,
    assessment_id: uuid.UUID,
    source_consistency_application_id: uuid.UUID,
    source_consistency_candidate_id: uuid.UUID,
    actor_id: uuid.UUID,
    decision_no: int,
    supersedes_decision_id: uuid.UUID | None,
    decision_kind: str,
    comment: str | None,
    selected_fact_value_ids: Sequence[uuid.UUID],
) -> str:
    return duplicate_grouping_service.hash_deterministic_payload(
        {
            "project_id": str(project_id),
            "consistency_check_application_id": str(consistency_check_application_id),
            "assessment_id": str(assessment_id),
            "source_consistency_application_id": str(source_consistency_application_id),
            "source_consistency_candidate_id": str(source_consistency_candidate_id),
            "actor_id": str(actor_id),
            "decision_no": decision_no,
            "supersedes_decision_id": (
                None if supersedes_decision_id is None else str(supersedes_decision_id)
            ),
            "decision_kind": decision_kind,
            "comment": comment,
            "selected_fact_value_ids": [str(value_id) for value_id in selected_fact_value_ids],
        }
    )


def _build_result(
    *,
    decision: ConsistencyReviewDecisionLedgerRecord,
    selected_fact_value_ids: Sequence[uuid.UUID],
    created_new: bool,
) -> AppendConsistencyReviewDecisionResult:
    return AppendConsistencyReviewDecisionResult(
        decision_id=decision.id,
        decision_no=decision.decision_no,
        supersedes_decision_id=decision.supersedes_decision_id,
        decision_manifest_hash=decision.decision_manifest_hash,
        selected_fact_value_ids=tuple(selected_fact_value_ids),
        created_new=created_new,
    )


def _map_persistence_error(error: Exception) -> Exception:
    if isinstance(error, persistence_service.ConsistencyCheckPersistenceInvariantError):
        return ConsistencyReviewInvariantError(
            "consistency_review_immutable_ledger_mismatch"
        )
    if isinstance(error, persistence_service.ConsistencyCheckPersistenceStateError):
        code = str(error)
        if code.endswith("application_not_found"):
            return ConsistencyReviewStateError(
                "consistency_review_application_not_found"
            )
        if code.endswith("project_id_mismatch"):
            return ConsistencyReviewStateError(
                "consistency_review_project_id_mismatch"
            )
        return ConsistencyReviewStateError("consistency_review_application_invalid")
    return error


async def _authenticate_authoritative_application(
    session_factory: Callable[[], AsyncSession],
    *,
    request: _NormalizedRequest,
):
    try:
        return await persistence_service.authenticate_persisted_consistency_check_application(
            session_factory,
            project_id=request.project_id,
            consistency_check_application_id=request.consistency_check_application_id,
        )
    except (persistence_service.ConsistencyCheckPersistenceInvariantError, persistence_service.ConsistencyCheckPersistenceStateError) as error:
        raise _map_persistence_error(error) from None


async def _authorize_actor(
    session_factory: Callable[[], AsyncSession],
    *,
    project_id: uuid.UUID,
    actor_id: uuid.UUID,
) -> None:
    async with session_factory() as read_session:
        try:
            actor = await consistency_review_repository.get_active_user_by_id(
                read_session,
                user_id=actor_id,
            )
            membership = await consistency_review_repository.get_project_member_for_project(
                read_session,
                project_id=project_id,
                user_id=actor_id,
            )
            _assert_actor_permission(actor=actor, membership=membership)
        except BaseException:
            await read_session.rollback()
            raise
        else:
            await read_session.rollback()


def _assert_actor_permission(
    *,
    actor: object | None,
    membership: object | None,
) -> None:
    if actor is None:
        raise ConsistencyReviewStateError("consistency_review_actor_not_found")
    if membership is None:
        raise ConsistencyReviewStateError(
            "consistency_review_actor_membership_not_found"
        )
    if membership.role not in _ALLOWED_ROLES:
        raise ConsistencyReviewStateError(
            "consistency_review_actor_permission_denied"
        )


async def _reauthorize_actor_in_transaction(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    actor_id: uuid.UUID,
) -> None:
    actor = await consistency_review_repository.get_active_user_by_id_for_update(
        session,
        user_id=actor_id,
    )
    membership = (
        await consistency_review_repository.get_project_member_for_project_for_update(
            session,
            project_id=project_id,
            user_id=actor_id,
        )
    )
    _assert_actor_permission(actor=actor, membership=membership)


def _resolve_authoritative_target(
    *,
    authenticated_context,
    assessment_id: uuid.UUID,
) -> tuple[object, object, tuple[ConsistencyReviewCandidateMemberRecord, ...]]:
    assessment_by_id = {
        assessment.id: assessment for assessment in authenticated_context.assessments
    }
    assessment = assessment_by_id.get(assessment_id)
    if assessment is None:
        raise ConsistencyReviewStateError("consistency_review_assessment_not_found")

    candidate_by_id = {
        candidate.candidate_id: candidate
        for candidate in authenticated_context.candidate_bundles
    }
    candidate = candidate_by_id.get(assessment.source_consistency_candidate_id)
    if candidate is None:
        raise ConsistencyReviewInvariantError(
            "consistency_review_immutable_ledger_mismatch"
        )
    return assessment, candidate, _build_member_snapshot(candidate.members)


def _validate_selected_fact_value_ids(
    *,
    selected_fact_value_ids: Sequence[uuid.UUID],
    candidate_members: Sequence[ConsistencyReviewCandidateMemberRecord],
) -> None:
    allowed_fact_value_ids = {member.fact_value_id for member in candidate_members}
    for fact_value_id in selected_fact_value_ids:
        if fact_value_id not in allowed_fact_value_ids:
            raise ConsistencyReviewStateError(
                "consistency_review_selected_fact_value_ids_invalid"
            )


def _assert_application_snapshot_matches(
    current_application: object | None,
    *,
    authenticated_context,
) -> None:
    if current_application is None:
        raise ConsistencyReviewStateError("consistency_review_application_not_found")
    application = authenticated_context.application
    if current_application.id != application.id:
        _immutable_mismatch()
    if current_application.project_id != application.project_id:
        _immutable_mismatch()
    if current_application.consistency_application_id != application.consistency_application_id:
        _immutable_mismatch()
    if current_application.orchestration_id != application.orchestration_id:
        _immutable_mismatch()
    if current_application.source_result_manifest_hash != application.source_result_manifest_hash:
        _immutable_mismatch()
    if current_application.plan_manifest_hash != application.plan_manifest_hash:
        _immutable_mismatch()
    if current_application.execution_identity_hash != application.execution_identity_hash:
        _immutable_mismatch()
    if current_application.result_manifest_hash != application.result_manifest_hash:
        _immutable_mismatch()
    if current_application.prompt_contract_hash != application.prompt_contract_hash:
        _immutable_mismatch()
    if current_application.provider != application.provider:
        _immutable_mismatch()
    if current_application.requested_model != application.requested_model:
        _immutable_mismatch()
    if current_application.executor_name != application.executor_name:
        _immutable_mismatch()
    if current_application.executor_version != application.executor_version:
        _immutable_mismatch()
    if current_application.batch_count != application.batch_count:
        _immutable_mismatch()
    if current_application.executed_batch_count != application.executed_batch_count:
        _immutable_mismatch()
    if (
        current_application.skipped_empty_batch_count
        != application.skipped_empty_batch_count
    ):
        _immutable_mismatch()
    if current_application.inference_run_count != application.inference_run_count:
        _immutable_mismatch()
    if current_application.assessment_count != application.assessment_count:
        _immutable_mismatch()


def _assert_assessment_snapshot_matches(
    current_assessment: object | None,
    *,
    authoritative_assessment,
    consistency_check_application_id: uuid.UUID,
) -> None:
    if current_assessment is None:
        raise ConsistencyReviewStateError("consistency_review_assessment_not_found")
    if current_assessment.id != authoritative_assessment.id:
        _immutable_mismatch()
    if current_assessment.consistency_check_application_id != consistency_check_application_id:
        _immutable_mismatch()
    if (
        current_assessment.source_consistency_application_id
        != authoritative_assessment.source_consistency_application_id
    ):
        _immutable_mismatch()
    if (
        current_assessment.source_consistency_candidate_id
        != authoritative_assessment.source_consistency_candidate_id
    ):
        _immutable_mismatch()
    if current_assessment.batch_index != authoritative_assessment.batch_index:
        _immutable_mismatch()
    if current_assessment.assessment_manifest_hash != authoritative_assessment.assessment_manifest_hash:
        _immutable_mismatch()


def _assert_candidate_member_snapshot_matches(
    current_members: Sequence[ConsistencyReviewCandidateMemberRecord],
    *,
    authoritative_members: Sequence[ConsistencyReviewCandidateMemberRecord],
) -> None:
    current_keys = tuple(
        (
            member.consistency_application_id,
            member.candidate_id,
            member.fact_value_id,
            member.source_batch_id,
            member.semantic_key_hash,
        )
        for member in current_members
    )
    authoritative_keys = tuple(
        (
            member.consistency_application_id,
            member.candidate_id,
            member.fact_value_id,
            member.source_batch_id,
            member.semantic_key_hash,
        )
        for member in authoritative_members
    )
    if current_keys != authoritative_keys:
        _immutable_mismatch()


def _validate_existing_chain(
    *,
    decisions: Sequence[ConsistencyReviewDecisionLedgerRecord],
    selections: Sequence[ConsistencyReviewDecisionSelectionLedgerRecord],
    request: _NormalizedRequest,
    source_consistency_application_id: uuid.UUID,
    source_consistency_candidate_id: uuid.UUID,
    candidate_members: Sequence[ConsistencyReviewCandidateMemberRecord],
) -> tuple[_ValidatedDecisionChainEntry, ...]:
    member_fact_value_ids = {member.fact_value_id for member in candidate_members}
    selections_by_decision_id: dict[uuid.UUID, list[ConsistencyReviewDecisionSelectionLedgerRecord]] = {}
    decision_ids = {decision.id for decision in decisions}
    for selection in selections:
        if selection.decision_id not in decision_ids:
            _immutable_mismatch()
        selections_by_decision_id.setdefault(selection.decision_id, []).append(selection)

    validated: list[_ValidatedDecisionChainEntry] = []
    previous_decision_id: uuid.UUID | None = None
    for index, decision in enumerate(decisions, start=1):
        if decision.project_id != request.project_id:
            _immutable_mismatch()
        if decision.consistency_check_application_id != request.consistency_check_application_id:
            _immutable_mismatch()
        if decision.assessment_id != request.assessment_id:
            _immutable_mismatch()
        if (
            decision.source_consistency_application_id
            != source_consistency_application_id
        ):
            _immutable_mismatch()
        if (
            decision.source_consistency_candidate_id
            != source_consistency_candidate_id
        ):
            _immutable_mismatch()
        if decision.decision_no != index:
            _immutable_mismatch()
        if index == 1:
            if decision.supersedes_decision_id is not None:
                _immutable_mismatch()
        elif decision.supersedes_decision_id != previous_decision_id:
            _immutable_mismatch()

        stored_comment = _normalize_comment(decision.comment)
        if stored_comment != decision.comment:
            _immutable_mismatch()
        try:
            stored_kind = _normalize_decision_kind(decision.decision_kind)
        except ConsistencyReviewStateError:
            _immutable_mismatch()
        else:
            if stored_kind != decision.decision_kind:
                _immutable_mismatch()

        decision_selections = selections_by_decision_id.get(decision.id, [])
        ordered_fact_value_ids: list[uuid.UUID] = []
        seen_fact_value_ids: set[uuid.UUID] = set()
        for selection_order, selection in enumerate(decision_selections):
            if selection.assessment_id != decision.assessment_id:
                _immutable_mismatch()
            if (
                selection.source_consistency_application_id
                != decision.source_consistency_application_id
            ):
                _immutable_mismatch()
            if (
                selection.source_consistency_candidate_id
                != decision.source_consistency_candidate_id
            ):
                _immutable_mismatch()
            if selection.selection_order != selection_order:
                _immutable_mismatch()
            if selection.fact_value_id in seen_fact_value_ids:
                _immutable_mismatch()
            if selection.fact_value_id not in member_fact_value_ids:
                _immutable_mismatch()
            seen_fact_value_ids.add(selection.fact_value_id)
            ordered_fact_value_ids.append(selection.fact_value_id)

        if decision.selected_value_count != len(ordered_fact_value_ids):
            _immutable_mismatch()
        _validate_selection_shape(
            decision_kind=decision.decision_kind,
            selected_fact_value_ids=ordered_fact_value_ids,
        )
        expected_manifest = _build_decision_manifest_hash(
            project_id=decision.project_id,
            consistency_check_application_id=decision.consistency_check_application_id,
            assessment_id=decision.assessment_id,
            source_consistency_application_id=decision.source_consistency_application_id,
            source_consistency_candidate_id=decision.source_consistency_candidate_id,
            actor_id=decision.actor_id,
            decision_no=decision.decision_no,
            supersedes_decision_id=decision.supersedes_decision_id,
            decision_kind=decision.decision_kind,
            comment=decision.comment,
            selected_fact_value_ids=ordered_fact_value_ids,
        )
        if expected_manifest != decision.decision_manifest_hash:
            _immutable_mismatch()

        validated.append(
            _ValidatedDecisionChainEntry(
                decision=decision,
                selected_fact_value_ids=tuple(ordered_fact_value_ids),
            )
        )
        previous_decision_id = decision.id
    return tuple(validated)


def _find_existing_matching_decision(
    *,
    chain: Sequence[_ValidatedDecisionChainEntry],
    request: _NormalizedRequest,
    source_consistency_application_id: uuid.UUID,
    source_consistency_candidate_id: uuid.UUID,
) -> _ValidatedDecisionChainEntry | None:
    decision_no_by_id = {entry.decision.id: entry.decision.decision_no for entry in chain}
    if request.expected_current_decision_id is None:
        expected_decision_no = 1
    else:
        predecessor_no = decision_no_by_id.get(request.expected_current_decision_id)
        if predecessor_no is None:
            return None
        expected_decision_no = predecessor_no + 1
    expected_manifest = _build_decision_manifest_hash(
        project_id=request.project_id,
        consistency_check_application_id=request.consistency_check_application_id,
        assessment_id=request.assessment_id,
        source_consistency_application_id=source_consistency_application_id,
        source_consistency_candidate_id=source_consistency_candidate_id,
        actor_id=request.actor_id,
        decision_no=expected_decision_no,
        supersedes_decision_id=request.expected_current_decision_id,
        decision_kind=request.decision_kind,
        comment=request.comment,
        selected_fact_value_ids=request.selected_fact_value_ids,
    )
    for entry in chain:
        decision = entry.decision
        if decision.decision_manifest_hash != expected_manifest:
            continue
        if decision.actor_id != request.actor_id:
            _immutable_mismatch()
        if decision.decision_kind != request.decision_kind:
            _immutable_mismatch()
        if decision.comment != request.comment:
            _immutable_mismatch()
        if entry.selected_fact_value_ids != request.selected_fact_value_ids:
            _immutable_mismatch()
        if decision.supersedes_decision_id != request.expected_current_decision_id:
            _immutable_mismatch()
        if decision.decision_no != expected_decision_no:
            _immutable_mismatch()
        return entry
    return None


def _build_write_plan(
    *,
    request: _NormalizedRequest,
    source_consistency_application_id: uuid.UUID,
    source_consistency_candidate_id: uuid.UUID,
    chain: Sequence[_ValidatedDecisionChainEntry],
) -> _DecisionWritePlan:
    current_leaf = chain[-1].decision if chain else None
    expected_leaf_id = None if current_leaf is None else current_leaf.id
    if request.expected_current_decision_id != expected_leaf_id:
        raise ConsistencyReviewStateError("consistency_review_stale_decision")

    decision_no = 1 if current_leaf is None else current_leaf.decision_no + 1
    supersedes_decision_id = None if current_leaf is None else current_leaf.id
    return _DecisionWritePlan(
        project_id=request.project_id,
        consistency_check_application_id=request.consistency_check_application_id,
        assessment_id=request.assessment_id,
        source_consistency_application_id=source_consistency_application_id,
        source_consistency_candidate_id=source_consistency_candidate_id,
        actor_id=request.actor_id,
        decision_no=decision_no,
        supersedes_decision_id=supersedes_decision_id,
        decision_kind=request.decision_kind,
        selected_fact_value_ids=request.selected_fact_value_ids,
        comment=request.comment,
        decision_manifest_hash=_build_decision_manifest_hash(
            project_id=request.project_id,
            consistency_check_application_id=request.consistency_check_application_id,
            assessment_id=request.assessment_id,
            source_consistency_application_id=source_consistency_application_id,
            source_consistency_candidate_id=source_consistency_candidate_id,
            actor_id=request.actor_id,
            decision_no=decision_no,
            supersedes_decision_id=supersedes_decision_id,
            decision_kind=request.decision_kind,
            comment=request.comment,
            selected_fact_value_ids=request.selected_fact_value_ids,
        ),
    )


async def _load_validated_chain_in_session(
    session: AsyncSession,
    *,
    request: _NormalizedRequest,
    authenticated_context,
    authoritative_assessment,
    authoritative_members: Sequence[ConsistencyReviewCandidateMemberRecord],
) -> tuple[_ValidatedDecisionChainEntry, ...]:
    current_application = await consistency_review_repository.get_consistency_check_application_by_id(
        session,
        consistency_check_application_id=request.consistency_check_application_id,
    )
    _assert_application_snapshot_matches(
        current_application,
        authenticated_context=authenticated_context,
    )
    current_assessment = await consistency_review_repository.get_consistency_assessment_for_update(
        session,
        consistency_check_application_id=request.consistency_check_application_id,
        assessment_id=request.assessment_id,
    )
    _assert_assessment_snapshot_matches(
        current_assessment,
        authoritative_assessment=authoritative_assessment,
        consistency_check_application_id=request.consistency_check_application_id,
    )
    await _reauthorize_actor_in_transaction(
        session,
        project_id=request.project_id,
        actor_id=request.actor_id,
    )
    current_members = await consistency_review_repository.list_candidate_member_records(
        session,
        source_consistency_application_id=authoritative_assessment.source_consistency_application_id,
        source_consistency_candidate_id=authoritative_assessment.source_consistency_candidate_id,
    )
    _assert_candidate_member_snapshot_matches(
        current_members,
        authoritative_members=authoritative_members,
    )

    decisions = await consistency_review_repository.list_decision_ledgers(
        session,
        assessment_id=request.assessment_id,
    )
    selections = await consistency_review_repository.list_selection_ledgers(
        session,
        assessment_id=request.assessment_id,
    )
    return _validate_existing_chain(
        decisions=decisions,
        selections=selections,
        request=request,
        source_consistency_application_id=authoritative_assessment.source_consistency_application_id,
        source_consistency_candidate_id=authoritative_assessment.source_consistency_candidate_id,
        candidate_members=current_members,
    )


async def _read_existing_matching_decision(
    session_factory: Callable[[], AsyncSession],
    *,
    request: _NormalizedRequest,
    authenticated_context,
    authoritative_assessment,
    authoritative_members: Sequence[ConsistencyReviewCandidateMemberRecord],
) -> AppendConsistencyReviewDecisionResult | None:
    async with session_factory() as read_session:
        try:
            chain = await _load_validated_chain_in_session(
                read_session,
                request=request,
                authenticated_context=authenticated_context,
                authoritative_assessment=authoritative_assessment,
                authoritative_members=authoritative_members,
            )
            existing_entry = _find_existing_matching_decision(
                chain=chain,
                request=request,
                source_consistency_application_id=authoritative_assessment.source_consistency_application_id,
                source_consistency_candidate_id=authoritative_assessment.source_consistency_candidate_id,
            )
        except BaseException:
            await read_session.rollback()
            raise
        else:
            await read_session.rollback()
    if existing_entry is None:
        return None
    return _build_result(
        decision=existing_entry.decision,
        selected_fact_value_ids=existing_entry.selected_fact_value_ids,
        created_new=False,
    )


async def append_consistency_review_decision(
    session_factory: Callable[[], AsyncSession],
    *,
    project_id,
    consistency_check_application_id,
    assessment_id,
    actor_id,
    expected_current_decision_id,
    decision_kind,
    selected_fact_value_ids,
    comment=None,
) -> AppendConsistencyReviewDecisionResult:
    request = _normalize_request(
        project_id=project_id,
        consistency_check_application_id=consistency_check_application_id,
        assessment_id=assessment_id,
        actor_id=actor_id,
        expected_current_decision_id=expected_current_decision_id,
        decision_kind=decision_kind,
        selected_fact_value_ids=selected_fact_value_ids,
        comment=comment,
    )
    _validate_selection_shape(
        decision_kind=request.decision_kind,
        selected_fact_value_ids=request.selected_fact_value_ids,
    )

    authenticated_context = await _authenticate_authoritative_application(
        session_factory,
        request=request,
    )
    await _authorize_actor(
        session_factory,
        project_id=request.project_id,
        actor_id=request.actor_id,
    )
    authoritative_assessment, _authoritative_candidate, authoritative_members = (
        _resolve_authoritative_target(
            authenticated_context=authenticated_context,
            assessment_id=request.assessment_id,
        )
    )
    _validate_selected_fact_value_ids(
        selected_fact_value_ids=request.selected_fact_value_ids,
        candidate_members=authoritative_members,
    )

    async with session_factory() as write_session:
        try:
            chain = await _load_validated_chain_in_session(
                write_session,
                request=request,
                authenticated_context=authenticated_context,
                authoritative_assessment=authoritative_assessment,
                authoritative_members=authoritative_members,
            )
            existing_entry = _find_existing_matching_decision(
                chain=chain,
                request=request,
                source_consistency_application_id=authoritative_assessment.source_consistency_application_id,
                source_consistency_candidate_id=authoritative_assessment.source_consistency_candidate_id,
            )
            if existing_entry is not None:
                await write_session.commit()
                return _build_result(
                    decision=existing_entry.decision,
                    selected_fact_value_ids=existing_entry.selected_fact_value_ids,
                    created_new=False,
                )

            write_plan = _build_write_plan(
                request=request,
                source_consistency_application_id=authoritative_assessment.source_consistency_application_id,
                source_consistency_candidate_id=authoritative_assessment.source_consistency_candidate_id,
                chain=chain,
            )
            decision = ConsistencyReviewDecision(
                id=uuid.uuid4(),
                project_id=write_plan.project_id,
                consistency_check_application_id=write_plan.consistency_check_application_id,
                assessment_id=write_plan.assessment_id,
                source_consistency_application_id=write_plan.source_consistency_application_id,
                source_consistency_candidate_id=write_plan.source_consistency_candidate_id,
                actor_id=write_plan.actor_id,
                decision_no=write_plan.decision_no,
                supersedes_decision_id=write_plan.supersedes_decision_id,
                decision_kind=write_plan.decision_kind,
                selected_value_count=len(write_plan.selected_fact_value_ids),
                comment=write_plan.comment,
                decision_manifest_hash=write_plan.decision_manifest_hash,
            )
            await consistency_review_repository.create_decision(
                write_session,
                decision,
            )
            selection_rows = [
                ConsistencyReviewDecisionSelection(
                    id=uuid.uuid4(),
                    decision_id=decision.id,
                    assessment_id=write_plan.assessment_id,
                    source_consistency_application_id=write_plan.source_consistency_application_id,
                    source_consistency_candidate_id=write_plan.source_consistency_candidate_id,
                    fact_value_id=fact_value_id,
                    selection_order=selection_order,
                )
                for selection_order, fact_value_id in enumerate(
                    write_plan.selected_fact_value_ids
                )
            ]
            if selection_rows:
                await consistency_review_repository.create_selections(
                    write_session,
                    selection_rows,
                )
            await write_session.commit()
            return AppendConsistencyReviewDecisionResult(
                decision_id=decision.id,
                decision_no=decision.decision_no,
                supersedes_decision_id=decision.supersedes_decision_id,
                decision_manifest_hash=decision.decision_manifest_hash,
                selected_fact_value_ids=write_plan.selected_fact_value_ids,
                created_new=True,
            )
        except IntegrityError as error:
            constraint_name = _get_integrity_constraint_name(error)
            await write_session.rollback()
            if constraint_name not in _HANDLED_DECISION_CONSTRAINTS:
                raise
        except BaseException:
            await write_session.rollback()
            raise

    existing_result = await _read_existing_matching_decision(
        session_factory,
        request=request,
        authenticated_context=authenticated_context,
        authoritative_assessment=authoritative_assessment,
        authoritative_members=authoritative_members,
    )
    if existing_result is not None:
        return existing_result
    raise ConsistencyReviewStateError("consistency_review_stale_decision")
