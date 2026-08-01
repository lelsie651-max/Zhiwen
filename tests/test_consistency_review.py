from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from app.models.project_member import ProjectMember, ProjectMemberRole
from app.models.user import User
from app.schemas.consistency_check_persistence import (
    ConsistencyAssessmentLedgerRecord,
    ConsistencyCheckApplicationLedgerRecord,
)
from app.schemas.consistency_review import (
    ConsistencyReviewCandidateMemberRecord,
    ConsistencyReviewDecisionLedgerRecord,
    ConsistencyReviewDecisionSelectionLedgerRecord,
)
from app.services import consistency_review as review_service


def run_async(awaitable):
    return asyncio.run(awaitable)


def _uuid(seed: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, seed)


def _integrity_error(constraint_name: str) -> IntegrityError:
    return IntegrityError(
        statement=None,
        params=None,
        orig=SimpleNamespace(diag=SimpleNamespace(constraint_name=constraint_name)),
    )


class FakeSession:
    def __init__(self, store: "LedgerStore") -> None:
        self.store = store
        self.commit_count = 0
        self.rollback_count = 0
        self.flush_count = 0
        self.pending_decisions: list[object] = []
        self.pending_selections: list[object] = []

    async def commit(self) -> None:
        self.commit_count += 1
        self.store.decisions.extend(self.pending_decisions)
        self.store.selections.extend(self.pending_selections)
        self.pending_decisions = []
        self.pending_selections = []

    async def rollback(self) -> None:
        self.rollback_count += 1
        self.pending_decisions = []
        self.pending_selections = []

    async def flush(self) -> None:
        self.flush_count += 1


class SessionFactory:
    def __init__(self, store: "LedgerStore") -> None:
        self.store = store
        self.sessions: list[FakeSession] = []
        self.open_count = 0

    def __call__(self):
        factory = self

        class _Context:
            async def __aenter__(self_inner):
                factory.open_count += 1
                session = FakeSession(factory.store)
                factory.sessions.append(session)
                return session

            async def __aexit__(self_inner, exc_type, exc, tb):
                factory.open_count -= 1
                return False

        return _Context()


class LedgerStore:
    def __init__(self) -> None:
        self.application: ConsistencyCheckApplicationLedgerRecord | None = None
        self.assessment: ConsistencyAssessmentLedgerRecord | None = None
        self.candidate_members: list[ConsistencyReviewCandidateMemberRecord] = []
        self.decisions: list[ConsistencyReviewDecisionLedgerRecord] = []
        self.selections: list[ConsistencyReviewDecisionSelectionLedgerRecord] = []
        self.actor: User | None = None
        self.membership: ProjectMember | None = None
        self.create_decision_error: Exception | None = None
        self.create_selection_error: Exception | None = None
        self.on_create_decision = None


def _build_actor(*, user_id: uuid.UUID, status: str = "active") -> User:
    return User(
        id=user_id,
        handle=f"user-{str(user_id)[:8]}",
        display_name="Reviewer",
        status=status,
    )


def _build_membership(*, project_id: uuid.UUID, user_id: uuid.UUID, role: str) -> ProjectMember:
    return ProjectMember(
        id=uuid.uuid4(),
        project_id=project_id,
        user_id=user_id,
        role=role,
    )


def _decision_manifest(
    *,
    project_id: uuid.UUID,
    application_id: uuid.UUID,
    assessment_id: uuid.UUID,
    source_application_id: uuid.UUID,
    source_candidate_id: uuid.UUID,
    actor_id: uuid.UUID,
    decision_no: int,
    supersedes_decision_id: uuid.UUID | None,
    decision_kind: str,
    comment: str | None,
    selected_fact_value_ids: tuple[uuid.UUID, ...],
) -> str:
    return review_service._build_decision_manifest_hash(
        project_id=project_id,
        consistency_check_application_id=application_id,
        assessment_id=assessment_id,
        source_consistency_application_id=source_application_id,
        source_consistency_candidate_id=source_candidate_id,
        actor_id=actor_id,
        decision_no=decision_no,
        supersedes_decision_id=supersedes_decision_id,
        decision_kind=decision_kind,
        comment=comment,
        selected_fact_value_ids=selected_fact_value_ids,
    )


def _build_store() -> tuple[LedgerStore, dict[str, uuid.UUID]]:
    store = LedgerStore()
    project_id = _uuid("project")
    app_id = _uuid("cc-app")
    assessment_id = _uuid("assessment")
    source_application_id = _uuid("source-app")
    source_candidate_id = _uuid("source-candidate")
    actor_id = _uuid("actor")
    created_at = datetime(2026, 8, 2, tzinfo=timezone.utc)

    store.application = ConsistencyCheckApplicationLedgerRecord(
        id=app_id,
        project_id=project_id,
        consistency_application_id=source_application_id,
        orchestration_id=_uuid("orchestration"),
        source_result_manifest_hash="a" * 64,
        plan_manifest_hash="b" * 64,
        execution_identity_hash="c" * 64,
        result_manifest_hash="d" * 64,
        prompt_contract_hash="e" * 64,
        provider="openai",
        requested_model="gpt-5",
        executor_name="consistency-check-executor",
        executor_version="1.0.0",
        batch_count=1,
        executed_batch_count=1,
        skipped_empty_batch_count=0,
        inference_run_count=1,
        assessment_count=1,
        created_at=created_at,
    )
    store.assessment = ConsistencyAssessmentLedgerRecord(
        id=assessment_id,
        consistency_check_application_id=app_id,
        source_consistency_application_id=source_application_id,
        source_consistency_candidate_id=source_candidate_id,
        batch_index=0,
        verdict="conflict",
        severity="yellow",
        confidence=0.8,
        explanation="needs human review",
        impact_json=("scope_review",),
        recommended_actions_json=("review_source_scope",),
        assessment_manifest_hash="f" * 64,
        created_at=created_at,
    )
    fv1 = _uuid("fv-1")
    fv2 = _uuid("fv-2")
    fv3 = _uuid("fv-3")
    store.candidate_members = [
        ConsistencyReviewCandidateMemberRecord(
            consistency_application_id=source_application_id,
            candidate_id=source_candidate_id,
            fact_value_id=fv1,
            source_batch_id=_uuid("batch-1"),
            semantic_key_hash="1" * 64,
        ),
        ConsistencyReviewCandidateMemberRecord(
            consistency_application_id=source_application_id,
            candidate_id=source_candidate_id,
            fact_value_id=fv2,
            source_batch_id=_uuid("batch-2"),
            semantic_key_hash="2" * 64,
        ),
        ConsistencyReviewCandidateMemberRecord(
            consistency_application_id=source_application_id,
            candidate_id=source_candidate_id,
            fact_value_id=fv3,
            source_batch_id=_uuid("batch-3"),
            semantic_key_hash="3" * 64,
        ),
    ]
    store.actor = _build_actor(user_id=actor_id)
    store.membership = _build_membership(
        project_id=project_id,
        user_id=actor_id,
        role=ProjectMemberRole.OWNER.value,
    )
    ids = {
        "project_id": project_id,
        "app_id": app_id,
        "assessment_id": assessment_id,
        "source_application_id": source_application_id,
        "source_candidate_id": source_candidate_id,
        "actor_id": actor_id,
        "fv1": fv1,
        "fv2": fv2,
        "fv3": fv3,
        "other_candidate_fv": _uuid("other-candidate-fv"),
    }
    return store, ids


def _build_authenticated_context(store: LedgerStore):
    assert store.application is not None
    assert store.assessment is not None
    candidate = SimpleNamespace(
        candidate_id=store.assessment.source_consistency_candidate_id,
        members=tuple(
            SimpleNamespace(
                consistency_application_id=member.consistency_application_id,
                candidate_id=member.candidate_id,
                fact_value_id=member.fact_value_id,
                source_batch_id=member.source_batch_id,
                semantic_key_hash=member.semantic_key_hash,
            )
            for member in store.candidate_members
        ),
    )
    return SimpleNamespace(
        application=store.application,
        authenticated_source=SimpleNamespace(),
        candidate_bundles=(candidate,),
        source_rows=(),
        batches=(),
        assessments=(store.assessment,),
        citations=(),
    )


def _committed_decision(
    store: LedgerStore,
    ids: dict[str, uuid.UUID],
    *,
    decision_no: int,
    supersedes_decision_id: uuid.UUID | None,
    decision_kind: str,
    actor_id: uuid.UUID | None = None,
    comment: str | None = None,
    selected_fact_value_ids: tuple[uuid.UUID, ...] = (),
    decision_id: uuid.UUID | None = None,
) -> tuple[ConsistencyReviewDecisionLedgerRecord, list[ConsistencyReviewDecisionSelectionLedgerRecord]]:
    assert store.application is not None
    assert store.assessment is not None
    actual_actor_id = actor_id or ids["actor_id"]
    actual_decision_id = decision_id or uuid.uuid4()
    decision = ConsistencyReviewDecisionLedgerRecord(
        id=actual_decision_id,
        project_id=ids["project_id"],
        consistency_check_application_id=ids["app_id"],
        assessment_id=ids["assessment_id"],
        source_consistency_application_id=ids["source_application_id"],
        source_consistency_candidate_id=ids["source_candidate_id"],
        actor_id=actual_actor_id,
        decision_no=decision_no,
        supersedes_decision_id=supersedes_decision_id,
        decision_kind=decision_kind,
        selected_value_count=len(selected_fact_value_ids),
        comment=comment,
        decision_manifest_hash=_decision_manifest(
            project_id=ids["project_id"],
            application_id=ids["app_id"],
            assessment_id=ids["assessment_id"],
            source_application_id=ids["source_application_id"],
            source_candidate_id=ids["source_candidate_id"],
            actor_id=actual_actor_id,
            decision_no=decision_no,
            supersedes_decision_id=supersedes_decision_id,
            decision_kind=decision_kind,
            comment=comment,
            selected_fact_value_ids=selected_fact_value_ids,
        ),
        created_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
    )
    selections = [
        ConsistencyReviewDecisionSelectionLedgerRecord(
            id=uuid.uuid4(),
            decision_id=decision.id,
            assessment_id=ids["assessment_id"],
            source_consistency_application_id=ids["source_application_id"],
            source_consistency_candidate_id=ids["source_candidate_id"],
            fact_value_id=fact_value_id,
            selection_order=selection_order,
            created_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
        )
        for selection_order, fact_value_id in enumerate(selected_fact_value_ids)
    ]
    return decision, selections


def _install_common_monkeypatches(monkeypatch, store: LedgerStore) -> None:
    async def fake_authenticate_persisted_consistency_check_application(
        _session_factory,
        *,
        project_id,
        consistency_check_application_id,
    ):
        assert store.application is not None
        if consistency_check_application_id != store.application.id:
            raise review_service.persistence_service.ConsistencyCheckPersistenceStateError(
                "consistency_check_persistence_application_not_found"
            )
        if project_id != store.application.project_id:
            raise review_service.persistence_service.ConsistencyCheckPersistenceStateError(
                "consistency_check_persistence_project_id_mismatch"
            )
        return _build_authenticated_context(store)

    async def fake_get_active_user_by_id(_session, *, user_id):
        if store.actor is not None and user_id == store.actor.id:
            return store.actor
        return None

    async def fake_get_project_member_for_project(_session, *, project_id, user_id):
        membership = store.membership
        if (
            membership is not None
            and membership.project_id == project_id
            and membership.user_id == user_id
        ):
            return membership
        return None

    async def fake_get_consistency_check_application_by_id(_session, *, consistency_check_application_id):
        if store.application is not None and consistency_check_application_id == store.application.id:
            return store.application
        return None

    async def fake_get_consistency_assessment_for_update(
        _session,
        *,
        consistency_check_application_id,
        assessment_id,
    ):
        if (
            store.assessment is not None
            and consistency_check_application_id == store.assessment.consistency_check_application_id
            and assessment_id == store.assessment.id
        ):
            return store.assessment
        return None

    async def fake_list_candidate_member_records(
        _session,
        *,
        source_consistency_application_id,
        source_consistency_candidate_id,
    ):
        if store.assessment is None:
            return ()
        if (
            source_consistency_application_id != store.assessment.source_consistency_application_id
            or source_consistency_candidate_id != store.assessment.source_consistency_candidate_id
        ):
            return ()
        return tuple(store.candidate_members)

    async def fake_list_decision_ledgers(_session, *, assessment_id):
        if store.assessment is None or assessment_id != store.assessment.id:
            return ()
        return tuple(sorted(store.decisions, key=lambda row: (row.decision_no, row.id)))

    async def fake_list_selection_ledgers(_session, *, assessment_id):
        if store.assessment is None or assessment_id != store.assessment.id:
            return ()
        decision_no_by_id = {decision.id: decision.decision_no for decision in store.decisions}
        return tuple(
            sorted(
                store.selections,
                key=lambda row: (
                    decision_no_by_id.get(row.decision_id, 10**9),
                    row.selection_order,
                    row.id,
                ),
            )
        )

    async def fake_create_decision(session: FakeSession, decision):
        if store.on_create_decision is not None:
            store.on_create_decision(decision)
        if store.create_decision_error is not None:
            raise store.create_decision_error
        session.pending_decisions.append(
            ConsistencyReviewDecisionLedgerRecord(
                id=decision.id,
                project_id=decision.project_id,
                consistency_check_application_id=decision.consistency_check_application_id,
                assessment_id=decision.assessment_id,
                source_consistency_application_id=decision.source_consistency_application_id,
                source_consistency_candidate_id=decision.source_consistency_candidate_id,
                actor_id=decision.actor_id,
                decision_no=decision.decision_no,
                supersedes_decision_id=decision.supersedes_decision_id,
                decision_kind=decision.decision_kind,
                selected_value_count=decision.selected_value_count,
                comment=decision.comment,
                decision_manifest_hash=decision.decision_manifest_hash,
                created_at=decision.created_at,
            )
        )
        await session.flush()
        return decision

    async def fake_create_selections(session: FakeSession, selections):
        if store.create_selection_error is not None:
            raise store.create_selection_error
        session.pending_selections.extend(
            ConsistencyReviewDecisionSelectionLedgerRecord(
                id=selection.id,
                decision_id=selection.decision_id,
                assessment_id=selection.assessment_id,
                source_consistency_application_id=selection.source_consistency_application_id,
                source_consistency_candidate_id=selection.source_consistency_candidate_id,
                fact_value_id=selection.fact_value_id,
                selection_order=selection.selection_order,
                created_at=selection.created_at,
            )
            for selection in selections
        )
        await session.flush()
        return selections

    monkeypatch.setattr(
        review_service.persistence_service,
        "authenticate_persisted_consistency_check_application",
        fake_authenticate_persisted_consistency_check_application,
    )
    monkeypatch.setattr(
        review_service.consistency_review_repository,
        "get_active_user_by_id",
        fake_get_active_user_by_id,
    )
    monkeypatch.setattr(
        review_service.consistency_review_repository,
        "get_project_member_for_project",
        fake_get_project_member_for_project,
    )
    monkeypatch.setattr(
        review_service.consistency_review_repository,
        "get_consistency_check_application_by_id",
        fake_get_consistency_check_application_by_id,
    )
    monkeypatch.setattr(
        review_service.consistency_review_repository,
        "get_consistency_assessment_for_update",
        fake_get_consistency_assessment_for_update,
    )
    monkeypatch.setattr(
        review_service.consistency_review_repository,
        "list_candidate_member_records",
        fake_list_candidate_member_records,
    )
    monkeypatch.setattr(
        review_service.consistency_review_repository,
        "list_decision_ledgers",
        fake_list_decision_ledgers,
    )
    monkeypatch.setattr(
        review_service.consistency_review_repository,
        "list_selection_ledgers",
        fake_list_selection_ledgers,
    )
    monkeypatch.setattr(
        review_service.consistency_review_repository,
        "create_decision",
        fake_create_decision,
    )
    monkeypatch.setattr(
        review_service.consistency_review_repository,
        "create_selections",
        fake_create_selections,
    )


@pytest.mark.parametrize("role", [ProjectMemberRole.OWNER.value, ProjectMemberRole.EDITOR.value])
def test_append_consistency_review_decision_allows_owner_and_editor(monkeypatch, role: str) -> None:
    store, ids = _build_store()
    store.membership = _build_membership(
        project_id=ids["project_id"],
        user_id=ids["actor_id"],
        role=role,
    )
    session_factory = SessionFactory(store)
    _install_common_monkeypatches(monkeypatch, store)

    result = run_async(
        review_service.append_consistency_review_decision(
            session_factory,
            project_id=ids["project_id"],
            consistency_check_application_id=ids["app_id"],
            assessment_id=ids["assessment_id"],
            actor_id=ids["actor_id"],
            expected_current_decision_id=None,
            decision_kind="select_one",
            selected_fact_value_ids=[ids["fv2"]],
            comment="  choose second  ",
        )
    )

    assert result.created_new is True
    assert result.decision_no == 1
    assert result.selected_fact_value_ids == (ids["fv2"],)
    assert store.decisions[0].comment == "choose second"


@pytest.mark.parametrize(
    ("membership", "error_code"),
    [
        (
            lambda ids: _build_membership(
                project_id=ids["project_id"],
                user_id=ids["actor_id"],
                role=ProjectMemberRole.VIEWER.value,
            ),
            "consistency_review_actor_permission_denied",
        ),
        (lambda _ids: None, "consistency_review_actor_membership_not_found"),
    ],
)
def test_append_consistency_review_decision_rejects_viewer_and_non_member(
    monkeypatch,
    membership,
    error_code: str,
) -> None:
    store, ids = _build_store()
    store.membership = membership(ids)
    session_factory = SessionFactory(store)
    _install_common_monkeypatches(monkeypatch, store)

    with pytest.raises(review_service.ConsistencyReviewStateError, match=error_code):
        run_async(
            review_service.append_consistency_review_decision(
                session_factory,
                project_id=ids["project_id"],
                consistency_check_application_id=ids["app_id"],
                assessment_id=ids["assessment_id"],
                actor_id=ids["actor_id"],
                expected_current_decision_id=None,
                decision_kind="defer",
                selected_fact_value_ids=[],
            )
        )


def test_append_consistency_review_decision_rejects_unknown_actor(monkeypatch) -> None:
    store, ids = _build_store()
    store.actor = None
    session_factory = SessionFactory(store)
    _install_common_monkeypatches(monkeypatch, store)

    with pytest.raises(review_service.ConsistencyReviewStateError, match="consistency_review_actor_not_found"):
        run_async(
            review_service.append_consistency_review_decision(
                session_factory,
                project_id=ids["project_id"],
                consistency_check_application_id=ids["app_id"],
                assessment_id=ids["assessment_id"],
                actor_id=ids["actor_id"],
                expected_current_decision_id=None,
                decision_kind="defer",
                selected_fact_value_ids=[],
            )
        )


@pytest.mark.parametrize(
    ("decision_kind", "selected_ids"),
    [
        ("select_one", ("fv1",)),
        ("keep_multiple", ("fv2", "fv1")),
        ("confirm_compatible", ()),
        ("defer", ()),
    ],
)
def test_append_consistency_review_decision_supports_all_decision_kinds(
    monkeypatch,
    decision_kind: str,
    selected_ids: tuple[str, ...],
) -> None:
    store, ids = _build_store()
    session_factory = SessionFactory(store)
    _install_common_monkeypatches(monkeypatch, store)

    result = run_async(
        review_service.append_consistency_review_decision(
            session_factory,
            project_id=ids["project_id"],
            consistency_check_application_id=ids["app_id"],
            assessment_id=ids["assessment_id"],
            actor_id=ids["actor_id"],
            expected_current_decision_id=None,
            decision_kind=decision_kind,
            selected_fact_value_ids=[ids[item] for item in selected_ids],
            comment="   " if decision_kind == "defer" else None,
        )
    )

    assert result.created_new is True
    assert result.selected_fact_value_ids == tuple(ids[item] for item in selected_ids)
    if decision_kind == "defer":
        assert store.decisions[0].comment is None


@pytest.mark.parametrize(
    ("decision_kind", "selected_ids"),
    [
        ("select_one", ()),
        ("select_one", ("fv1", "fv2")),
        ("keep_multiple", ("fv1",)),
        ("confirm_compatible", ("fv1",)),
        ("defer", ("fv1",)),
    ],
)
def test_append_consistency_review_decision_rejects_invalid_selection_shapes(
    monkeypatch,
    decision_kind: str,
    selected_ids: tuple[str, ...],
) -> None:
    store, ids = _build_store()
    session_factory = SessionFactory(store)
    _install_common_monkeypatches(monkeypatch, store)

    with pytest.raises(
        review_service.ConsistencyReviewStateError,
        match="consistency_review_selected_fact_value_ids_shape_invalid",
    ):
        run_async(
            review_service.append_consistency_review_decision(
                session_factory,
                project_id=ids["project_id"],
                consistency_check_application_id=ids["app_id"],
                assessment_id=ids["assessment_id"],
                actor_id=ids["actor_id"],
                expected_current_decision_id=None,
                decision_kind=decision_kind,
                selected_fact_value_ids=[ids[item] for item in selected_ids],
            )
        )


def test_append_consistency_review_decision_rejects_other_candidate_fact_value(monkeypatch) -> None:
    store, ids = _build_store()
    session_factory = SessionFactory(store)
    _install_common_monkeypatches(monkeypatch, store)

    with pytest.raises(
        review_service.ConsistencyReviewStateError,
        match="consistency_review_selected_fact_value_ids_invalid",
    ):
        run_async(
            review_service.append_consistency_review_decision(
                session_factory,
                project_id=ids["project_id"],
                consistency_check_application_id=ids["app_id"],
                assessment_id=ids["assessment_id"],
                actor_id=ids["actor_id"],
                expected_current_decision_id=None,
                decision_kind="select_one",
                selected_fact_value_ids=[ids["other_candidate_fv"]],
            )
        )


def test_append_consistency_review_decision_builds_linear_revision_chain(monkeypatch) -> None:
    store, ids = _build_store()
    session_factory = SessionFactory(store)
    _install_common_monkeypatches(monkeypatch, store)

    first = run_async(
        review_service.append_consistency_review_decision(
            session_factory,
            project_id=ids["project_id"],
            consistency_check_application_id=ids["app_id"],
            assessment_id=ids["assessment_id"],
            actor_id=ids["actor_id"],
            expected_current_decision_id=None,
            decision_kind="select_one",
            selected_fact_value_ids=[ids["fv1"]],
        )
    )
    second = run_async(
        review_service.append_consistency_review_decision(
            session_factory,
            project_id=ids["project_id"],
            consistency_check_application_id=ids["app_id"],
            assessment_id=ids["assessment_id"],
            actor_id=ids["actor_id"],
            expected_current_decision_id=first.decision_id,
            decision_kind="keep_multiple",
            selected_fact_value_ids=[ids["fv2"], ids["fv1"]],
            comment="keep both",
        )
    )

    assert first.decision_no == 1
    assert second.decision_no == 2
    assert second.supersedes_decision_id == first.decision_id
    assert [decision.decision_no for decision in store.decisions] == [1, 2]
    assert [decision.supersedes_decision_id for decision in store.decisions] == [None, first.decision_id]
    second_selection_rows = [row for row in store.selections if row.decision_id == second.decision_id]
    assert [row.fact_value_id for row in second_selection_rows] == [ids["fv2"], ids["fv1"]]
    assert [row.selection_order for row in second_selection_rows] == [0, 1]


@pytest.mark.parametrize(
    ("expected_current", "error_code"),
    [
        ("wrong", "consistency_review_stale_decision"),
        ("missing-on-empty", "consistency_review_stale_decision"),
    ],
)
def test_append_consistency_review_decision_rejects_wrong_or_stale_expected_current(
    monkeypatch,
    expected_current: str,
    error_code: str,
) -> None:
    store, ids = _build_store()
    if expected_current == "wrong":
        prior_decision, prior_selections = _committed_decision(
            store,
            ids,
            decision_no=1,
            supersedes_decision_id=None,
            decision_kind="select_one",
            selected_fact_value_ids=(ids["fv1"],),
        )
        store.decisions.append(prior_decision)
        store.selections.extend(prior_selections)
        expected_current_id = _uuid("wrong-current")
    else:
        expected_current_id = _uuid("missing-current")
    session_factory = SessionFactory(store)
    _install_common_monkeypatches(monkeypatch, store)

    with pytest.raises(review_service.ConsistencyReviewStateError, match=error_code):
        run_async(
            review_service.append_consistency_review_decision(
                session_factory,
                project_id=ids["project_id"],
                consistency_check_application_id=ids["app_id"],
                assessment_id=ids["assessment_id"],
                actor_id=ids["actor_id"],
                expected_current_decision_id=expected_current_id,
                decision_kind="defer",
                selected_fact_value_ids=[],
            )
        )


def test_append_consistency_review_decision_retries_same_request_idempotently(monkeypatch) -> None:
    store, ids = _build_store()
    session_factory = SessionFactory(store)
    _install_common_monkeypatches(monkeypatch, store)

    first = run_async(
        review_service.append_consistency_review_decision(
            session_factory,
            project_id=ids["project_id"],
            consistency_check_application_id=ids["app_id"],
            assessment_id=ids["assessment_id"],
            actor_id=ids["actor_id"],
            expected_current_decision_id=None,
            decision_kind="select_one",
            selected_fact_value_ids=[ids["fv1"]],
            comment="same request",
        )
    )
    second = run_async(
        review_service.append_consistency_review_decision(
            session_factory,
            project_id=ids["project_id"],
            consistency_check_application_id=ids["app_id"],
            assessment_id=ids["assessment_id"],
            actor_id=ids["actor_id"],
            expected_current_decision_id=None,
            decision_kind="select_one",
            selected_fact_value_ids=[ids["fv1"]],
            comment="same request",
        )
    )

    assert first.decision_id == second.decision_id
    assert second.created_new is False
    assert len(store.decisions) == 1


def test_append_consistency_review_decision_handles_same_request_concurrent_conflict(monkeypatch) -> None:
    store, ids = _build_store()
    session_factory = SessionFactory(store)

    def concurrent_winner(_pending_decision) -> None:
        if store.decisions:
            return
        winner, winner_selections = _committed_decision(
            store,
            ids,
            decision_no=1,
            supersedes_decision_id=None,
            decision_kind="select_one",
            comment="same request",
            selected_fact_value_ids=(ids["fv1"],),
        )
        store.decisions.append(winner)
        store.selections.extend(winner_selections)
        store.create_decision_error = _integrity_error("uq_ccrevd_manifest_hash")

    store.on_create_decision = concurrent_winner
    _install_common_monkeypatches(monkeypatch, store)

    result = run_async(
        review_service.append_consistency_review_decision(
            session_factory,
            project_id=ids["project_id"],
            consistency_check_application_id=ids["app_id"],
            assessment_id=ids["assessment_id"],
            actor_id=ids["actor_id"],
            expected_current_decision_id=None,
            decision_kind="select_one",
            selected_fact_value_ids=[ids["fv1"]],
            comment="same request",
        )
    )

    assert result.created_new is False
    assert len(store.decisions) == 1
    assert result.decision_id == store.decisions[0].id


def test_append_consistency_review_decision_rejects_different_request_after_concurrent_win(monkeypatch) -> None:
    store, ids = _build_store()
    session_factory = SessionFactory(store)

    def concurrent_winner(_pending_decision) -> None:
        if store.decisions:
            return
        winner, winner_selections = _committed_decision(
            store,
            ids,
            decision_no=1,
            supersedes_decision_id=None,
            decision_kind="select_one",
            selected_fact_value_ids=(ids["fv1"],),
        )
        store.decisions.append(winner)
        store.selections.extend(winner_selections)
        store.create_decision_error = _integrity_error("uq_ccrevd_supersedes_id")

    store.on_create_decision = concurrent_winner
    _install_common_monkeypatches(monkeypatch, store)

    with pytest.raises(review_service.ConsistencyReviewStateError, match="consistency_review_stale_decision"):
        run_async(
            review_service.append_consistency_review_decision(
                session_factory,
                project_id=ids["project_id"],
                consistency_check_application_id=ids["app_id"],
                assessment_id=ids["assessment_id"],
                actor_id=ids["actor_id"],
                expected_current_decision_id=None,
                decision_kind="select_one",
                selected_fact_value_ids=[ids["fv2"]],
            )
        )


@pytest.mark.parametrize(
    "mutate_store",
    [
        lambda store, ids: store.decisions.append(
            replace(
                _committed_decision(
                    store,
                    ids,
                    decision_no=2,
                    supersedes_decision_id=None,
                    decision_kind="defer",
                )[0],
                comment=None,
            )
        ),
        lambda store, ids: _mutate_branch(store, ids),
        lambda store, ids: _mutate_count_drift(store, ids),
        lambda store, ids: _mutate_selection_order_drift(store, ids),
        lambda store, ids: _mutate_manifest_drift(store, ids),
    ],
)
def test_append_consistency_review_decision_fails_closed_on_existing_chain_drift(
    monkeypatch,
    mutate_store,
) -> None:
    store, ids = _build_store()
    first, first_selections = _committed_decision(
        store,
        ids,
        decision_no=1,
        supersedes_decision_id=None,
        decision_kind="select_one",
        selected_fact_value_ids=(ids["fv1"],),
    )
    store.decisions.append(first)
    store.selections.extend(first_selections)
    mutate_store(store, ids)
    session_factory = SessionFactory(store)
    _install_common_monkeypatches(monkeypatch, store)

    with pytest.raises(
        review_service.ConsistencyReviewInvariantError,
        match="consistency_review_immutable_ledger_mismatch",
    ):
        run_async(
            review_service.append_consistency_review_decision(
                session_factory,
                project_id=ids["project_id"],
                consistency_check_application_id=ids["app_id"],
                assessment_id=ids["assessment_id"],
                actor_id=ids["actor_id"],
                expected_current_decision_id=first.id,
                decision_kind="defer",
                selected_fact_value_ids=[],
            )
        )


def _mutate_branch(store: LedgerStore, ids: dict[str, uuid.UUID]) -> None:
    first = store.decisions[0]
    second, second_selections = _committed_decision(
        store,
        ids,
        decision_no=2,
        supersedes_decision_id=_uuid("other-predecessor"),
        decision_kind="defer",
    )
    store.decisions.append(second)
    store.selections.extend(second_selections)
    assert first.supersedes_decision_id is None


def _mutate_count_drift(store: LedgerStore, ids: dict[str, uuid.UUID]) -> None:
    first = store.decisions[0]
    store.decisions[0] = replace(first, selected_value_count=2)


def _mutate_selection_order_drift(store: LedgerStore, ids: dict[str, uuid.UUID]) -> None:
    selection = store.selections[0]
    store.selections[0] = replace(selection, selection_order=1)


def _mutate_manifest_drift(store: LedgerStore, ids: dict[str, uuid.UUID]) -> None:
    first = store.decisions[0]
    store.decisions[0] = replace(first, decision_manifest_hash="0" * 64)


def test_append_consistency_review_decision_rolls_back_when_selection_write_fails(monkeypatch) -> None:
    store, ids = _build_store()
    store.create_selection_error = IntegrityError(
        statement=None,
        params=None,
        orig=Exception("selection failed"),
    )
    session_factory = SessionFactory(store)
    _install_common_monkeypatches(monkeypatch, store)

    with pytest.raises(IntegrityError):
        run_async(
            review_service.append_consistency_review_decision(
                session_factory,
                project_id=ids["project_id"],
                consistency_check_application_id=ids["app_id"],
                assessment_id=ids["assessment_id"],
                actor_id=ids["actor_id"],
                expected_current_decision_id=None,
                decision_kind="select_one",
                selected_fact_value_ids=[ids["fv1"]],
            )
        )

    assert store.decisions == []
    assert store.selections == []
    assert session_factory.sessions[-1].rollback_count >= 1


def test_append_consistency_review_decision_does_not_swallow_unknown_integrity_error(monkeypatch) -> None:
    store, ids = _build_store()
    store.create_decision_error = _integrity_error("uq_ccrevs_decision_order")
    session_factory = SessionFactory(store)
    _install_common_monkeypatches(monkeypatch, store)

    with pytest.raises(IntegrityError):
        run_async(
            review_service.append_consistency_review_decision(
                session_factory,
                project_id=ids["project_id"],
                consistency_check_application_id=ids["app_id"],
                assessment_id=ids["assessment_id"],
                actor_id=ids["actor_id"],
                expected_current_decision_id=None,
                decision_kind="select_one",
                selected_fact_value_ids=[ids["fv1"]],
            )
        )


def test_append_consistency_review_decision_rejects_bool_like_and_duplicate_inputs(monkeypatch) -> None:
    store, ids = _build_store()
    session_factory = SessionFactory(store)
    _install_common_monkeypatches(monkeypatch, store)

    with pytest.raises(
        review_service.ConsistencyReviewStateError,
        match="consistency_review_selected_fact_value_ids_invalid",
    ):
        run_async(
            review_service.append_consistency_review_decision(
                session_factory,
                project_id=ids["project_id"],
                consistency_check_application_id=ids["app_id"],
                assessment_id=ids["assessment_id"],
                actor_id=ids["actor_id"],
                expected_current_decision_id=None,
                decision_kind="select_one",
                selected_fact_value_ids=[True],
            )
        )
    with pytest.raises(
        review_service.ConsistencyReviewStateError,
        match="consistency_review_selected_fact_value_ids_duplicate",
    ):
        run_async(
            review_service.append_consistency_review_decision(
                session_factory,
                project_id=ids["project_id"],
                consistency_check_application_id=ids["app_id"],
                assessment_id=ids["assessment_id"],
                actor_id=ids["actor_id"],
                expected_current_decision_id=None,
                decision_kind="keep_multiple",
                selected_fact_value_ids=[ids["fv1"], ids["fv1"]],
            )
        )


def test_append_consistency_review_decision_does_not_leak_sensitive_sentinel(monkeypatch) -> None:
    store, ids = _build_store()
    session_factory = SessionFactory(store)
    _install_common_monkeypatches(monkeypatch, store)
    sentinel = "SECRET_COMMENT_SENTINEL"

    with pytest.raises(review_service.ConsistencyReviewStateError) as exc_info:
        run_async(
            review_service.append_consistency_review_decision(
                session_factory,
                project_id=ids["project_id"],
                consistency_check_application_id=ids["app_id"],
                assessment_id=ids["assessment_id"],
                actor_id=ids["actor_id"],
                expected_current_decision_id=None,
                decision_kind="select_one",
                selected_fact_value_ids=[ids["other_candidate_fv"]],
                comment=sentinel,
            )
        )

    assert sentinel not in str(exc_info.value)


def test_append_consistency_review_decision_does_not_modify_fact_or_agent2_ledgers(monkeypatch) -> None:
    store, ids = _build_store()
    initial_application = deepcopy(store.application)
    initial_assessment = deepcopy(store.assessment)
    initial_candidate_members = deepcopy(store.candidate_members)
    session_factory = SessionFactory(store)
    _install_common_monkeypatches(monkeypatch, store)

    run_async(
        review_service.append_consistency_review_decision(
            session_factory,
            project_id=ids["project_id"],
            consistency_check_application_id=ids["app_id"],
            assessment_id=ids["assessment_id"],
            actor_id=ids["actor_id"],
            expected_current_decision_id=None,
            decision_kind="select_one",
            selected_fact_value_ids=[ids["fv1"]],
        )
    )

    assert store.application == initial_application
    assert store.assessment == initial_assessment
    assert store.candidate_members == initial_candidate_members
