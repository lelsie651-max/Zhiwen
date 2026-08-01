from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import uuid

import pytest

from app.schemas.consistency_projection import (
    ConsistencyReviewProjection,
    ConsistencyReviewProjectionDecision,
    ConsistencyReviewProjectionEvidence,
    ConsistencyReviewProjectionItem,
    ConsistencyReviewProjectionMember,
)
from app.services import effective_fact_value as effective_fact_value_service


def run_async(awaitable):
    return asyncio.run(awaitable)


class FakeSession:
    def __init__(self) -> None:
        self.commit_count = 0
        self.rollback_count = 0

    async def commit(self) -> None:
        self.commit_count += 1

    async def rollback(self) -> None:
        self.rollback_count += 1


class SessionFactory:
    def __init__(self) -> None:
        self.sessions: list[FakeSession] = []
        self.open_count = 0

    def __call__(self):
        factory = self

        class _Context:
            async def __aenter__(self_inner):
                factory.open_count += 1
                session = FakeSession()
                factory.sessions.append(session)
                return session

            async def __aexit__(self_inner, exc_type, exc, tb):
                factory.open_count -= 1
                return False

        return _Context()


def _uuid(seed: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, seed)


def _evidence(seed: str) -> ConsistencyReviewProjectionEvidence:
    return ConsistencyReviewProjectionEvidence(
        evidence_link_id=_uuid(f"{seed}-link"),
        evidence_id=_uuid(f"{seed}-evidence"),
        document_revision_id=_uuid(f"{seed}-revision"),
        document_block_id=_uuid(f"{seed}-block"),
        location_key=f"loc:{seed}",
        page_no=1,
        start_line=1,
        end_line=1,
        start_offset=0,
        end_offset=5,
        excerpt=f"excerpt-{seed}",
        excerpt_hash="a" * 64,
        content_hash="a" * 64,
        cited_by_assessment=True,
    )


def _member(
    *,
    fact_value_id: uuid.UUID,
    selected: bool = False,
    selection_order: int | None = None,
) -> ConsistencyReviewProjectionMember:
    return ConsistencyReviewProjectionMember(
        fact_value_id=fact_value_id,
        value_type="string",
        value_json=f"value-{fact_value_id}",
        normalized_value_text=f"text-{fact_value_id}",
        referenced_entity_id=None,
        selected_by_current_decision=selected,
        current_selection_order=selection_order,
        evidences=(_evidence(str(fact_value_id)),),
    )


def _decision(
    *,
    seed: str,
    decision_no: int,
    decision_kind: str,
    selected_fact_value_ids: tuple[uuid.UUID, ...],
    supersedes_decision_id: uuid.UUID | None = None,
) -> ConsistencyReviewProjectionDecision:
    return ConsistencyReviewProjectionDecision(
        decision_id=_uuid(f"{seed}-decision"),
        decision_no=decision_no,
        supersedes_decision_id=supersedes_decision_id,
        actor_id=_uuid(f"{seed}-actor"),
        decision_kind=decision_kind,
        selected_fact_value_ids=selected_fact_value_ids,
        comment=None,
        decision_manifest_hash="b" * 64,
        created_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
    )


def _item(
    *,
    seed: str,
    fact_id: uuid.UUID,
    assessment_id: uuid.UUID,
    candidate_id: uuid.UUID,
    verdict: str,
    review_status: str,
    members: tuple[ConsistencyReviewProjectionMember, ...],
    current_decision: ConsistencyReviewProjectionDecision | None = None,
    decision_history: tuple[ConsistencyReviewProjectionDecision, ...] = (),
    selected_fact_value_ids: tuple[uuid.UUID, ...] = (),
) -> ConsistencyReviewProjectionItem:
    return ConsistencyReviewProjectionItem(
        assessment_id=assessment_id,
        fact_id=fact_id,
        candidate_id=candidate_id,
        batch_index=0 if seed != "gamma" else 1,
        verdict=verdict,
        severity="yellow" if verdict == "conflict" else "none",
        confidence=0.9,
        explanation=f"explanation-{seed}",
        impact=(),
        recommended_actions=(),
        review_status=review_status,
        current_decision=current_decision,
        decision_history=decision_history,
        selected_fact_value_ids=selected_fact_value_ids,
        members=members,
    )


def _projection(
    items: tuple[ConsistencyReviewProjectionItem, ...],
) -> ConsistencyReviewProjection:
    return ConsistencyReviewProjection(
        project_id=_uuid("project"),
        consistency_check_application_id=_uuid("cc-app"),
        source_consistency_application_id=_uuid("source-app"),
        plan_manifest_hash="c" * 64,
        result_manifest_hash="d" * 64,
        assessment_count=len(items),
        conflict_count=sum(1 for item in items if item.verdict == "conflict"),
        compatible_count=sum(1 for item in items if item.verdict == "compatible"),
        insufficient_evidence_count=sum(
            1 for item in items if item.verdict == "insufficient_evidence"
        ),
        red_count=0,
        yellow_count=sum(1 for item in items if item.severity == "yellow"),
        pending_review_count=sum(
            1 for item in items if item.review_status == "pending_review"
        ),
        reviewed_count=sum(1 for item in items if item.review_status == "reviewed"),
        deferred_count=sum(1 for item in items if item.review_status == "deferred"),
        not_required_count=sum(
            1 for item in items if item.review_status == "not_required"
        ),
        decision_count=sum(len(item.decision_history) for item in items),
        items=items,
    )


def _install_projection_monkeypatch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    projection: ConsistencyReviewProjection | Exception,
) -> None:
    async def fake_get_consistency_review_projection(
        session_factory,
        *,
        project_id,
        consistency_check_application_id,
    ):
        if isinstance(projection, Exception):
            raise projection
        return projection

    monkeypatch.setattr(
        effective_fact_value_service.projection_service,
        "get_consistency_review_projection",
        fake_get_consistency_review_projection,
    )


def test_get_effective_fact_value_projection_maps_select_one(monkeypatch) -> None:
    fv1 = _uuid("fv-1")
    fv2 = _uuid("fv-2")
    decision = _decision(
        seed="alpha",
        decision_no=1,
        decision_kind="select_one",
        selected_fact_value_ids=(fv2,),
    )
    projection = _projection(
        (
            _item(
                seed="alpha",
                fact_id=_uuid("fact-1"),
                assessment_id=_uuid("assessment-1"),
                candidate_id=_uuid("candidate-1"),
                verdict="conflict",
                review_status="reviewed",
                current_decision=decision,
                decision_history=(decision,),
                selected_fact_value_ids=(fv2,),
                members=(
                    _member(fact_value_id=fv1),
                    _member(fact_value_id=fv2, selected=True, selection_order=0),
                ),
            ),
        )
    )
    _install_projection_monkeypatch(monkeypatch, projection=projection)

    result = run_async(
        effective_fact_value_service.get_effective_fact_value_projection(
            SessionFactory(),
            project_id=projection.project_id,
            consistency_check_application_id=projection.consistency_check_application_id,
        )
    )

    assert result.fact_count == 1
    assert result.resolved_count == 1
    item = result.items[0]
    assert item.fact_id == _uuid("fact-1")
    assert item.resolution_status == "resolved"
    assert item.resolution_basis == "human_selection"
    assert item.current_decision_id == decision.decision_id
    assert item.current_decision_kind == "select_one"
    assert item.effective_fact_value_ids == (fv2,)


def test_get_effective_fact_value_projection_maps_keep_multiple_in_selection_order(
    monkeypatch,
) -> None:
    fv1 = _uuid("fv-1")
    fv2 = _uuid("fv-2")
    decision = _decision(
        seed="beta",
        decision_no=1,
        decision_kind="keep_multiple",
        selected_fact_value_ids=(fv2, fv1),
    )
    projection = _projection(
        (
            _item(
                seed="beta",
                fact_id=_uuid("fact-2"),
                assessment_id=_uuid("assessment-2"),
                candidate_id=_uuid("candidate-2"),
                verdict="conflict",
                review_status="reviewed",
                current_decision=decision,
                decision_history=(decision,),
                selected_fact_value_ids=(fv2, fv1),
                members=(
                    _member(fact_value_id=fv1, selected=True, selection_order=1),
                    _member(fact_value_id=fv2, selected=True, selection_order=0),
                ),
            ),
        )
    )
    _install_projection_monkeypatch(monkeypatch, projection=projection)

    result = run_async(
        effective_fact_value_service.get_effective_fact_value_projection(
            SessionFactory(),
            project_id=projection.project_id,
            consistency_check_application_id=projection.consistency_check_application_id,
        )
    )

    assert result.items[0].effective_fact_value_ids == (fv2, fv1)
    assert result.items[0].resolution_basis == "human_selection"


def test_get_effective_fact_value_projection_maps_confirm_compatible_without_fake_explicit_selection(
    monkeypatch,
) -> None:
    fv1 = _uuid("fv-1")
    fv2 = _uuid("fv-2")
    decision = _decision(
        seed="gamma",
        decision_no=1,
        decision_kind="confirm_compatible",
        selected_fact_value_ids=(),
    )
    members = (
        _member(fact_value_id=fv1),
        _member(fact_value_id=fv2),
    )
    original_members = deepcopy(members)
    projection = _projection(
        (
            _item(
                seed="gamma",
                fact_id=_uuid("fact-3"),
                assessment_id=_uuid("assessment-3"),
                candidate_id=_uuid("candidate-3"),
                verdict="compatible",
                review_status="reviewed",
                current_decision=decision,
                decision_history=(decision,),
                selected_fact_value_ids=(),
                members=members,
            ),
        )
    )
    _install_projection_monkeypatch(monkeypatch, projection=projection)

    result = run_async(
        effective_fact_value_service.get_effective_fact_value_projection(
            SessionFactory(),
            project_id=projection.project_id,
            consistency_check_application_id=projection.consistency_check_application_id,
        )
    )

    item = result.items[0]
    assert item.resolution_status == "resolved"
    assert item.resolution_basis == "human_confirmed_compatibility"
    assert item.effective_fact_value_ids == (fv1, fv2)
    assert item.candidate_members == original_members
    assert all(member.selected_by_current_decision is False for member in item.candidate_members)
    assert all(member.current_selection_order is None for member in item.candidate_members)


def test_get_effective_fact_value_projection_maps_defer_and_no_decision_to_empty_effective(
    monkeypatch,
) -> None:
    defer_decision = _decision(
        seed="delta",
        decision_no=1,
        decision_kind="defer",
        selected_fact_value_ids=(),
    )
    projection = _projection(
        (
            _item(
                seed="delta",
                fact_id=_uuid("fact-4"),
                assessment_id=_uuid("assessment-4"),
                candidate_id=_uuid("candidate-4"),
                verdict="conflict",
                review_status="deferred",
                current_decision=defer_decision,
                decision_history=(defer_decision,),
                selected_fact_value_ids=(),
                members=(_member(fact_value_id=_uuid("fv-4")),),
            ),
            _item(
                seed="epsilon",
                fact_id=_uuid("fact-5"),
                assessment_id=_uuid("assessment-5"),
                candidate_id=_uuid("candidate-5"),
                verdict="conflict",
                review_status="pending_review",
                members=(_member(fact_value_id=_uuid("fv-5")),),
            ),
        )
    )
    _install_projection_monkeypatch(monkeypatch, projection=projection)

    result = run_async(
        effective_fact_value_service.get_effective_fact_value_projection(
            SessionFactory(),
            project_id=projection.project_id,
            consistency_check_application_id=projection.consistency_check_application_id,
        )
    )

    assert result.deferred_count == 1
    assert result.pending_count == 1
    assert result.items[0].effective_fact_value_ids == ()
    assert result.items[0].resolution_status == "deferred"
    assert result.items[1].effective_fact_value_ids == ()
    assert result.items[1].resolution_status == "pending_review"


def test_get_effective_fact_value_projection_does_not_let_ai_compatible_generate_effective_values(
    monkeypatch,
) -> None:
    projection = _projection(
        (
            _item(
                seed="zeta",
                fact_id=_uuid("fact-6"),
                assessment_id=_uuid("assessment-6"),
                candidate_id=_uuid("candidate-6"),
                verdict="compatible",
                review_status="not_required",
                members=(
                    _member(fact_value_id=_uuid("fv-6a")),
                    _member(fact_value_id=_uuid("fv-6b")),
                ),
            ),
        )
    )
    _install_projection_monkeypatch(monkeypatch, projection=projection)

    result = run_async(
        effective_fact_value_service.get_effective_fact_value_projection(
            SessionFactory(),
            project_id=projection.project_id,
            consistency_check_application_id=projection.consistency_check_application_id,
        )
    )

    item = result.items[0]
    assert item.resolution_status == "unreviewed_compatible"
    assert item.resolution_basis == "none"
    assert item.effective_fact_value_ids == ()


def test_get_effective_fact_value_projection_uses_only_current_leaf_decision(
    monkeypatch,
) -> None:
    fv1 = _uuid("fv-7a")
    fv2 = _uuid("fv-7b")
    first = _decision(
        seed="eta-first",
        decision_no=1,
        decision_kind="select_one",
        selected_fact_value_ids=(fv1,),
    )
    second = _decision(
        seed="eta-second",
        decision_no=2,
        decision_kind="keep_multiple",
        selected_fact_value_ids=(fv2, fv1),
        supersedes_decision_id=first.decision_id,
    )
    projection = _projection(
        (
            _item(
                seed="eta",
                fact_id=_uuid("fact-7"),
                assessment_id=_uuid("assessment-7"),
                candidate_id=_uuid("candidate-7"),
                verdict="conflict",
                review_status="reviewed",
                current_decision=second,
                decision_history=(first, second),
                selected_fact_value_ids=(fv2, fv1),
                members=(
                    _member(fact_value_id=fv1, selected=True, selection_order=1),
                    _member(fact_value_id=fv2, selected=True, selection_order=0),
                ),
            ),
        )
    )
    _install_projection_monkeypatch(monkeypatch, projection=projection)

    result = run_async(
        effective_fact_value_service.get_effective_fact_value_projection(
            SessionFactory(),
            project_id=projection.project_id,
            consistency_check_application_id=projection.consistency_check_application_id,
        )
    )

    assert result.items[0].current_decision_id == second.decision_id
    assert result.items[0].effective_fact_value_ids == (fv2, fv1)


@pytest.mark.parametrize(
    "mutate_projection",
    [
        lambda projection: replace(
            projection,
            items=(
                replace(
                    projection.items[0],
                    selected_fact_value_ids=(_uuid("unknown-fv"),),
                    current_decision=replace(
                        projection.items[0].current_decision,
                        selected_fact_value_ids=(_uuid("unknown-fv"),),
                    ),
                ),
            ),
        ),
        lambda projection: replace(
            projection,
            items=(
                replace(
                    projection.items[0],
                    selected_fact_value_ids=(
                        projection.items[0].members[0].fact_value_id,
                        projection.items[0].members[0].fact_value_id,
                    ),
                    current_decision=replace(
                        projection.items[0].current_decision,
                        selected_fact_value_ids=(
                            projection.items[0].members[0].fact_value_id,
                            projection.items[0].members[0].fact_value_id,
                        ),
                    ),
                ),
            ),
        ),
        lambda projection: replace(
            projection,
            items=(
                replace(
                    projection.items[0],
                    selected_fact_value_ids=(projection.items[1].members[0].fact_value_id,),
                    current_decision=replace(
                        projection.items[0].current_decision,
                        selected_fact_value_ids=(projection.items[1].members[0].fact_value_id,),
                    ),
                ),
                projection.items[1],
            ),
        ),
    ],
)
def test_get_effective_fact_value_projection_fails_closed_on_invalid_effective_ids(
    monkeypatch,
    mutate_projection,
) -> None:
    first_decision = _decision(
        seed="theta-first",
        decision_no=1,
        decision_kind="select_one",
        selected_fact_value_ids=(_uuid("fv-8a"),),
    )
    projection = _projection(
        (
            _item(
                seed="theta",
                fact_id=_uuid("fact-8"),
                assessment_id=_uuid("assessment-8"),
                candidate_id=_uuid("candidate-8"),
                verdict="conflict",
                review_status="reviewed",
                current_decision=first_decision,
                decision_history=(first_decision,),
                selected_fact_value_ids=(_uuid("fv-8a"),),
                members=(
                    _member(
                        fact_value_id=_uuid("fv-8a"),
                        selected=True,
                        selection_order=0,
                    ),
                    _member(fact_value_id=_uuid("fv-8b")),
                ),
            ),
            _item(
                seed="theta-other",
                fact_id=_uuid("fact-9"),
                assessment_id=_uuid("assessment-9"),
                candidate_id=_uuid("candidate-9"),
                verdict="conflict",
                review_status="pending_review",
                members=(_member(fact_value_id=_uuid("fv-9a")),),
            ),
        )
    )
    _install_projection_monkeypatch(
        monkeypatch,
        projection=mutate_projection(projection),
    )

    with pytest.raises(
        effective_fact_value_service.EffectiveFactValueProjectionInvariantError,
        match="effective_fact_value_projection_immutable_ledger_mismatch",
    ):
        run_async(
            effective_fact_value_service.get_effective_fact_value_projection(
                SessionFactory(),
                project_id=projection.project_id,
                consistency_check_application_id=projection.consistency_check_application_id,
            )
        )


def test_get_effective_fact_value_projection_computes_stats_and_preserves_order(
    monkeypatch,
) -> None:
    select_decision = _decision(
        seed="iota",
        decision_no=1,
        decision_kind="select_one",
        selected_fact_value_ids=(_uuid("fv-10a"),),
    )
    defer_decision = _decision(
        seed="kappa",
        decision_no=1,
        decision_kind="defer",
        selected_fact_value_ids=(),
    )
    projection = _projection(
        (
            _item(
                seed="iota",
                fact_id=_uuid("fact-10"),
                assessment_id=_uuid("assessment-10"),
                candidate_id=_uuid("candidate-10"),
                verdict="conflict",
                review_status="reviewed",
                current_decision=select_decision,
                decision_history=(select_decision,),
                selected_fact_value_ids=(_uuid("fv-10a"),),
                members=(
                    _member(
                        fact_value_id=_uuid("fv-10a"),
                        selected=True,
                        selection_order=0,
                    ),
                ),
            ),
            _item(
                seed="kappa",
                fact_id=_uuid("fact-11"),
                assessment_id=_uuid("assessment-11"),
                candidate_id=_uuid("candidate-11"),
                verdict="conflict",
                review_status="deferred",
                current_decision=defer_decision,
                decision_history=(defer_decision,),
                selected_fact_value_ids=(),
                members=(_member(fact_value_id=_uuid("fv-11a")),),
            ),
            _item(
                seed="lambda",
                fact_id=_uuid("fact-12"),
                assessment_id=_uuid("assessment-12"),
                candidate_id=_uuid("candidate-12"),
                verdict="compatible",
                review_status="not_required",
                members=(_member(fact_value_id=_uuid("fv-12a")),),
            ),
            _item(
                seed="mu",
                fact_id=_uuid("fact-13"),
                assessment_id=_uuid("assessment-13"),
                candidate_id=_uuid("candidate-13"),
                verdict="insufficient_evidence",
                review_status="pending_review",
                members=(_member(fact_value_id=_uuid("fv-13a")),),
            ),
        )
    )
    _install_projection_monkeypatch(monkeypatch, projection=projection)

    result = run_async(
        effective_fact_value_service.get_effective_fact_value_projection(
            SessionFactory(),
            project_id=projection.project_id,
            consistency_check_application_id=projection.consistency_check_application_id,
        )
    )

    assert [item.fact_id for item in result.items] == [
        _uuid("fact-10"),
        _uuid("fact-11"),
        _uuid("fact-12"),
        _uuid("fact-13"),
    ]
    assert result.fact_count == 4
    assert result.resolved_count == 1
    assert result.deferred_count == 1
    assert result.pending_count == 2


def test_get_effective_fact_value_projection_zero_assessment_returns_empty_projection(
    monkeypatch,
) -> None:
    projection = _projection(())
    _install_projection_monkeypatch(monkeypatch, projection=projection)

    result = run_async(
        effective_fact_value_service.get_effective_fact_value_projection(
            SessionFactory(),
            project_id=projection.project_id,
            consistency_check_application_id=projection.consistency_check_application_id,
        )
    )

    assert result.items == ()
    assert result.fact_count == 0
    assert result.resolved_count == 0
    assert result.pending_count == 0
    assert result.deferred_count == 0


def test_get_effective_fact_value_projection_does_not_write_and_does_not_leak_sensitive_sentinel(
    monkeypatch,
) -> None:
    session_factory = SessionFactory()
    sentinel = "SENSITIVE_COMMENT_SENTINEL"
    _install_projection_monkeypatch(
        monkeypatch,
        projection=effective_fact_value_service.projection_service.ConsistencyProjectionInvariantError(
            "consistency_review_projection_immutable_ledger_mismatch"
        ),
    )

    with pytest.raises(
        effective_fact_value_service.EffectiveFactValueProjectionInvariantError,
        match="effective_fact_value_projection_immutable_ledger_mismatch",
    ) as exc_info:
        run_async(
            effective_fact_value_service.get_effective_fact_value_projection(
                session_factory,
                project_id=_uuid("project"),
                consistency_check_application_id=_uuid("cc-app"),
            )
        )

    assert session_factory.sessions == []
    assert sentinel not in str(exc_info.value)
