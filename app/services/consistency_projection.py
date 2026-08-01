from __future__ import annotations

import uuid
from collections.abc import Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.consistency_projection import (
    ConsistencyReviewProjection,
    ConsistencyReviewProjectionDecision,
    ConsistencyReviewProjectionEvidence,
    ConsistencyReviewProjectionItem,
    ConsistencyReviewProjectionMember,
)
from app.schemas.consistency_review import (
    AuthenticatedConsistencyReviewDecisionChainEntry,
    ConsistencyReviewCandidateMemberRecord,
)
from app.repositories import consistency_review as consistency_review_repository
from app.services import consistency_check_persistence as persistence_service
from app.services import consistency_review as review_service


class ConsistencyProjectionError(Exception):
    """Base class for consistency review projection failures."""


class ConsistencyProjectionStateError(ConsistencyProjectionError):
    """Raised when the requested application or source chain is invalid."""


class ConsistencyProjectionInvariantError(ConsistencyProjectionError):
    """Raised when persisted immutable ledgers diverge from the authenticated source."""


def _require_uuid(value: uuid.UUID, *, field_name: str) -> uuid.UUID:
    if not isinstance(value, uuid.UUID):
        raise ConsistencyProjectionStateError(f"{field_name}_invalid")
    return value


def _map_persistence_error(error: Exception) -> Exception:
    if isinstance(error, persistence_service.ConsistencyCheckPersistenceInvariantError):
        return ConsistencyProjectionInvariantError(
            "consistency_review_projection_immutable_ledger_mismatch"
        )
    if isinstance(error, persistence_service.ConsistencyCheckPersistenceStateError):
        message = str(error)
        if message == "consistency_check_persistence_application_not_found":
            return ConsistencyProjectionStateError(
                "consistency_review_projection_application_not_found"
            )
        if message == "consistency_check_persistence_project_id_mismatch":
            return ConsistencyProjectionStateError(
                "consistency_review_projection_project_id_mismatch"
            )
        return ConsistencyProjectionStateError(
            "consistency_review_projection_source_not_authenticated"
        )
    return error


def _review_status_for_verdict(verdict: str) -> str:
    if verdict == "compatible":
        return "not_required"
    return "pending_review"


def _review_status_for_item(
    *,
    verdict: str,
    current_decision: ConsistencyReviewProjectionDecision | None,
) -> str:
    if current_decision is not None:
        if current_decision.decision_kind == "defer":
            return "deferred"
        return "reviewed"
    return _review_status_for_verdict(verdict)


def _build_unique_map(
    records,
    *,
    key_builder,
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for record in records:
        key = key_builder(record)
        if key in mapping:
            raise ConsistencyProjectionInvariantError(
                "consistency_review_projection_immutable_ledger_mismatch"
            )
        mapping[key] = record
    return mapping


def _build_representative_map(
    records,
    *,
    key_builder,
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for record in records:
        key = key_builder(record)
        mapping.setdefault(key, record)
    return mapping


def _require_source_row(mapping: dict[uuid.UUID, object], *, key: uuid.UUID) -> object:
    row = mapping.get(key)
    if row is None:
        raise ConsistencyProjectionInvariantError(
            "consistency_review_projection_immutable_ledger_mismatch"
        )
    return row


def _require_uuid_field(value: object) -> uuid.UUID:
    if not isinstance(value, uuid.UUID):
        raise ConsistencyProjectionInvariantError(
            "consistency_review_projection_immutable_ledger_mismatch"
        )
    return value


def _build_decision_projection(
    entry: AuthenticatedConsistencyReviewDecisionChainEntry,
) -> ConsistencyReviewProjectionDecision:
    decision = entry.decision
    return ConsistencyReviewProjectionDecision(
        decision_id=decision.id,
        decision_no=decision.decision_no,
        supersedes_decision_id=decision.supersedes_decision_id,
        actor_id=decision.actor_id,
        decision_kind=decision.decision_kind,
        selected_fact_value_ids=entry.selected_fact_value_ids,
        comment=decision.comment,
        decision_manifest_hash=decision.decision_manifest_hash,
        created_at=decision.created_at,
    )


def _raise_projection_immutable_mismatch() -> None:
    raise ConsistencyProjectionInvariantError(
        "consistency_review_projection_immutable_ledger_mismatch"
    )


async def _load_authenticated_decision_chains(
    session_factory: Callable[[], AsyncSession],
    *,
    authenticated,
) -> dict[uuid.UUID, tuple[AuthenticatedConsistencyReviewDecisionChainEntry, ...]]:
    candidate_members_by_candidate_id: dict[
        uuid.UUID, tuple[ConsistencyReviewCandidateMemberRecord, ...]
    ] = {
        candidate.candidate_id: tuple(
            ConsistencyReviewCandidateMemberRecord(
                consistency_application_id=authenticated.application.consistency_application_id,
                candidate_id=candidate.candidate_id,
                fact_value_id=member.fact_value_id,
                source_batch_id=member.source_batch_id,
                semantic_key_hash=member.semantic_key_hash,
            )
            for member in candidate.members
        )
        for candidate in authenticated.candidate_bundles
    }
    assessments_by_id = _build_unique_map(
        authenticated.assessments,
        key_builder=lambda assessment: assessment.id,
    )
    assessment_by_candidate_id = _build_unique_map(
        authenticated.assessments,
        key_builder=lambda assessment: assessment.source_consistency_candidate_id,
    )
    async with session_factory() as read_session:
        try:
            decisions = await consistency_review_repository.list_decision_ledgers_by_application(
                read_session,
                consistency_check_application_id=authenticated.application.id,
            )
            selections = await consistency_review_repository.list_selection_ledgers_by_application(
                read_session,
                consistency_check_application_id=authenticated.application.id,
            )
        except BaseException:
            await read_session.rollback()
            raise
        else:
            await read_session.rollback()

    decisions_by_assessment: dict[uuid.UUID, list[object]] = {}
    for decision in decisions:
        assessment = assessments_by_id.get(decision.assessment_id)
        if assessment is None:
            raise ConsistencyProjectionInvariantError(
                "consistency_review_projection_immutable_ledger_mismatch"
            )
        if decision.consistency_check_application_id != authenticated.application.id:
            raise ConsistencyProjectionInvariantError(
                "consistency_review_projection_immutable_ledger_mismatch"
            )
        if decision.source_consistency_application_id != authenticated.application.consistency_application_id:
            raise ConsistencyProjectionInvariantError(
                "consistency_review_projection_immutable_ledger_mismatch"
            )
        if (
            decision.source_consistency_candidate_id
            != assessment.source_consistency_candidate_id
        ):
            raise ConsistencyProjectionInvariantError(
                "consistency_review_projection_immutable_ledger_mismatch"
            )
        decisions_by_assessment.setdefault(decision.assessment_id, []).append(decision)

    selections_by_assessment: dict[uuid.UUID, list[object]] = {}
    for selection in selections:
        assessment = assessments_by_id.get(selection.assessment_id)
        if assessment is None:
            raise ConsistencyProjectionInvariantError(
                "consistency_review_projection_immutable_ledger_mismatch"
            )
        if (
            selection.source_consistency_application_id
            != authenticated.application.consistency_application_id
        ):
            raise ConsistencyProjectionInvariantError(
                "consistency_review_projection_immutable_ledger_mismatch"
            )
        if (
            selection.source_consistency_candidate_id
            != assessment.source_consistency_candidate_id
        ):
            raise ConsistencyProjectionInvariantError(
                "consistency_review_projection_immutable_ledger_mismatch"
            )
        selections_by_assessment.setdefault(selection.assessment_id, []).append(selection)

    authenticated_chains: dict[
        uuid.UUID, tuple[AuthenticatedConsistencyReviewDecisionChainEntry, ...]
    ] = {}
    for assessment in authenticated.assessments:
        candidate_members = candidate_members_by_candidate_id.get(
            assessment.source_consistency_candidate_id
        )
        authoritative_assessment = assessment_by_candidate_id.get(
            assessment.source_consistency_candidate_id
        )
        if candidate_members is None or authoritative_assessment is None:
            _raise_projection_immutable_mismatch()
        try:
            authenticated_chains[assessment.id] = (
                review_service.authenticate_consistency_review_decision_chain(
                    decisions=tuple(decisions_by_assessment.get(assessment.id, ())),
                    selections=tuple(selections_by_assessment.get(assessment.id, ())),
                    project_id=authenticated.application.project_id,
                    consistency_check_application_id=authenticated.application.id,
                    assessment_id=assessment.id,
                    source_consistency_application_id=assessment.source_consistency_application_id,
                    source_consistency_candidate_id=assessment.source_consistency_candidate_id,
                    candidate_members=candidate_members,
                )
            )
        except (
            review_service.ConsistencyReviewInvariantError,
            review_service.ConsistencyReviewStateError,
        ):
            _raise_projection_immutable_mismatch()
    if len(decisions) != sum(len(chain) for chain in authenticated_chains.values()):
        raise ConsistencyProjectionInvariantError(
            "consistency_review_projection_immutable_ledger_mismatch"
        )
    if len(selections) != sum(
        len(entry.selected_fact_value_ids)
        for chain in authenticated_chains.values()
        for entry in chain
    ):
        raise ConsistencyProjectionInvariantError(
            "consistency_review_projection_immutable_ledger_mismatch"
        )
    return authenticated_chains


async def get_consistency_review_projection(
    session_factory: Callable[[], AsyncSession],
    *,
    project_id: uuid.UUID,
    consistency_check_application_id: uuid.UUID,
) -> ConsistencyReviewProjection:
    project_id = _require_uuid(project_id, field_name="project_id")
    consistency_check_application_id = _require_uuid(
        consistency_check_application_id,
        field_name="consistency_check_application_id",
    )
    try:
        authenticated = await persistence_service.authenticate_persisted_consistency_check_application(
            session_factory,
            project_id=project_id,
            consistency_check_application_id=consistency_check_application_id,
        )
    except Exception as error:
        mapped = _map_persistence_error(error)
        if mapped is error:
            raise
        raise mapped from None

    application = authenticated.application
    assessment_by_candidate_id = _build_unique_map(
        authenticated.assessments,
        key_builder=lambda assessment: assessment.source_consistency_candidate_id,
    )
    authenticated_decision_chains = await _load_authenticated_decision_chains(
        session_factory,
        authenticated=authenticated,
    )
    source_row_by_fact_value_id = _build_representative_map(
        authenticated.source_rows,
        key_builder=lambda row: row.member_fact_value_id,
    )
    source_row_by_evidence_link_id = _build_unique_map(
        (
            row
            for row in authenticated.source_rows
            if row.evidence_link_id is not None
        ),
        key_builder=lambda row: row.evidence_link_id,
    )
    citations_by_assessment_id: dict[uuid.UUID, set[uuid.UUID]] = {}
    for citation in authenticated.citations:
        citations_by_assessment_id.setdefault(citation.assessment_id, set()).add(
            citation.evidence_link_id
        )

    items: list[ConsistencyReviewProjectionItem] = []
    for batch_index in range(application.batch_count):
        for candidate in authenticated.candidate_bundles:
            assessment = assessment_by_candidate_id.get(candidate.candidate_id)
            if assessment is None or assessment.batch_index != batch_index:
                continue
            decision_chain = authenticated_decision_chains.get(assessment.id, ())
            decision_history = tuple(
                _build_decision_projection(entry) for entry in decision_chain
            )
            current_decision = (
                decision_history[-1] if decision_history else None
            )
            current_selection_order_by_fact_value_id = (
                {
                    fact_value_id: selection_order
                    for selection_order, fact_value_id in enumerate(
                        current_decision.selected_fact_value_ids
                    )
                }
                if current_decision is not None
                else {}
            )
            selected_fact_value_ids = (
                current_decision.selected_fact_value_ids
                if current_decision is not None
                and current_decision.decision_kind in {"select_one", "keep_multiple"}
                else ()
            )
            cited_evidence_ids = citations_by_assessment_id.get(assessment.id, set())
            members = tuple(
                ConsistencyReviewProjectionMember(
                    fact_value_id=member.fact_value_id,
                    value_type=member.value_type,
                    value_json=member.value_json,
                    normalized_value_text=_require_source_row(
                        source_row_by_fact_value_id,
                        key=member.fact_value_id,
                    ).fact_value_normalized_value_text,
                    referenced_entity_id=member.referenced_entity_id,
                    selected_by_current_decision=member.fact_value_id
                    in current_selection_order_by_fact_value_id,
                    current_selection_order=current_selection_order_by_fact_value_id.get(
                        member.fact_value_id
                    ),
                    evidences=tuple(
                        ConsistencyReviewProjectionEvidence(
                            evidence_link_id=evidence.evidence_link_id,
                            evidence_id=evidence.evidence_id,
                            document_revision_id=_require_uuid_field(
                                _require_source_row(
                                    source_row_by_evidence_link_id,
                                    key=evidence.evidence_link_id,
                                ).document_revision_id
                            ),
                            document_block_id=evidence.document_block_id,
                            location_key=evidence.location_key,
                            page_no=evidence.page_no,
                            start_line=evidence.start_line,
                            end_line=evidence.end_line,
                            start_offset=evidence.start_offset,
                            end_offset=evidence.end_offset,
                            excerpt=evidence.excerpt,
                            excerpt_hash=evidence.evidence_content_hash,
                            content_hash=evidence.evidence_content_hash,
                            cited_by_assessment=evidence.evidence_link_id in cited_evidence_ids,
                        )
                        for evidence in member.evidences
                    ),
                )
                for member in candidate.members
            )
            items.append(
                ConsistencyReviewProjectionItem(
                    assessment_id=assessment.id,
                    candidate_id=candidate.candidate_id,
                    batch_index=assessment.batch_index,
                    verdict=assessment.verdict,
                    severity=assessment.severity,
                    confidence=assessment.confidence,
                    explanation=assessment.explanation,
                    impact=assessment.impact_json,
                    recommended_actions=assessment.recommended_actions_json,
                    review_status=_review_status_for_item(
                        verdict=assessment.verdict,
                        current_decision=current_decision,
                    ),
                    current_decision=current_decision,
                    decision_history=decision_history,
                    selected_fact_value_ids=selected_fact_value_ids,
                    members=members,
                )
            )

    if len(items) != application.assessment_count:
        raise ConsistencyProjectionInvariantError(
            "consistency_review_projection_immutable_ledger_mismatch"
        )

    conflict_count = sum(1 for item in items if item.verdict == "conflict")
    compatible_count = sum(1 for item in items if item.verdict == "compatible")
    insufficient_evidence_count = sum(
        1 for item in items if item.verdict == "insufficient_evidence"
    )
    red_count = sum(1 for item in items if item.severity == "red")
    yellow_count = sum(1 for item in items if item.severity == "yellow")
    pending_review_count = sum(
        1 for item in items if item.review_status == "pending_review"
    )
    reviewed_count = sum(1 for item in items if item.review_status == "reviewed")
    deferred_count = sum(1 for item in items if item.review_status == "deferred")
    not_required_count = sum(
        1 for item in items if item.review_status == "not_required"
    )
    decision_count = sum(len(item.decision_history) for item in items)
    return ConsistencyReviewProjection(
        project_id=application.project_id,
        consistency_check_application_id=application.id,
        source_consistency_application_id=application.consistency_application_id,
        plan_manifest_hash=application.plan_manifest_hash,
        result_manifest_hash=application.result_manifest_hash,
        assessment_count=application.assessment_count,
        conflict_count=conflict_count,
        compatible_count=compatible_count,
        insufficient_evidence_count=insufficient_evidence_count,
        red_count=red_count,
        yellow_count=yellow_count,
        pending_review_count=pending_review_count,
        reviewed_count=reviewed_count,
        deferred_count=deferred_count,
        not_required_count=not_required_count,
        decision_count=decision_count,
        items=tuple(items),
    )
