from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import IntegrityError

from app.agents.prompt_registry import get_prompt
from app.models.consistency_check import (
    ConsistencyAssessmentCitation,
    ConsistencyAssessmentLedger,
    ConsistencyCheckApplication,
    ConsistencyCheckBatchLedger,
)
from app.schemas.agent_consistency_check import (
    ConsistencyCheckAssessment,
    ConsistencyCheckResponse,
)
from app.schemas.consistency_check import (
    CONSISTENCY_CHECK_PLANNER_NAME,
    CONSISTENCY_CHECK_PLANNER_VERSION,
    ConsistencyCheckBatchPlan,
    ConsistencyCheckCandidateBundle,
    ConsistencyCheckEvidenceBundle,
    ConsistencyCheckMemberBundle,
    ConsistencyCheckPlan,
    ConsistencyCheckPlannerConfig,
)
from app.schemas.consistency_check_execution import ConsistencyCheckBatchExecutionResult
from app.schemas.consistency_check_persistence import (
    ConsistencyAssessmentCitationLedgerRecord,
    ConsistencyAssessmentLedgerRecord,
)
from app.schemas.fact_value_duplicate_grouping import (
    FactValueConsistencyCandidateApplicationLedger,
)
from app.services import consistency_check as consistency_check_service
from app.services import consistency_check_execution as execution_service
from app.services import consistency_check_persistence as persistence_service
from app.services import fact_value_duplicate_grouping as duplicate_grouping_service


PROMPT = get_prompt("agent2_consistency_check", "1.0.0")
CONFIG = ConsistencyCheckPlannerConfig(
    max_candidates_per_batch=8,
    max_evidence_characters_per_batch=500,
)


def run_async(awaitable):
    return asyncio.run(awaitable)


class FakeSession:
    def __init__(self) -> None:
        self.commit_count = 0
        self.rollback_count = 0
        self.flush_count = 0

    async def commit(self) -> None:
        self.commit_count += 1

    async def rollback(self) -> None:
        self.rollback_count += 1

    async def flush(self) -> None:
        self.flush_count += 1


class SessionFactory:
    def __init__(self) -> None:
        self.open_count = 0
        self.sessions: list[FakeSession] = []

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


class LedgerStore:
    def __init__(self) -> None:
        self.application: ConsistencyCheckApplication | None = None
        self.batches: list[ConsistencyCheckBatchLedger] = []
        self.assessments: list[ConsistencyAssessmentLedger] = []
        self.citations: list[ConsistencyAssessmentCitation] = []


def _evidence_bundle(
    *,
    evidence_link_id: uuid.UUID,
    excerpt: str = "evidence",
    source_order: int = 0,
) -> ConsistencyCheckEvidenceBundle:
    return ConsistencyCheckEvidenceBundle(
        evidence_link_id=evidence_link_id,
        evidence_id=uuid.uuid4(),
        role="supporting",
        is_primary=True,
        source_order=source_order,
        document_block_id=uuid.uuid4(),
        location_key=f"loc:{source_order}",
        page_no=1,
        start_line=1,
        end_line=1,
        start_offset=0,
        end_offset=len(excerpt),
        excerpt=excerpt,
        evidence_content_hash=duplicate_grouping_service.hash_deterministic_payload(excerpt),
    )


def _candidate_bundle(
    *,
    candidate_id: uuid.UUID,
    fact_id: uuid.UUID,
    fact_value_id: uuid.UUID,
    evidence_link_ids: tuple[uuid.UUID, ...],
) -> ConsistencyCheckCandidateBundle:
    evidences = tuple(
        _evidence_bundle(evidence_link_id=evidence_link_id, excerpt=f"excerpt-{index}", source_order=index)
        for index, evidence_link_id in enumerate(evidence_link_ids)
    )
    member = ConsistencyCheckMemberBundle(
        fact_value_id=fact_value_id,
        source_batch_id=uuid.uuid4(),
        semantic_key_hash="a" * 64,
        value_type="string",
        value_json="value",
        referenced_entity_id=None,
        evidences=evidences,
    )
    return ConsistencyCheckCandidateBundle(
        candidate_id=candidate_id,
        fact_id=fact_id,
        candidate_kind="multi_value",
        members=(member,),
    )


def _batch_plan(
    *,
    batch_index: int,
    candidates: tuple[ConsistencyCheckCandidateBundle, ...],
) -> ConsistencyCheckBatchPlan:
    candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
    return ConsistencyCheckBatchPlan(
        batch_index=batch_index,
        candidate_ids=candidate_ids,
        candidate_count=len(candidates),
        evidence_character_count=sum(
            len(evidence.excerpt)
            for candidate in candidates
            for member in candidate.members
            for evidence in member.evidences
        ),
        batch_manifest_hash=f"{batch_index + 1:064x}"[-64:],
        candidates=candidates,
    )


def _build_plan(
    *,
    project_id: uuid.UUID,
    consistency_application_id: uuid.UUID,
    source_result_manifest_hash: str,
    candidates: tuple[ConsistencyCheckCandidateBundle, ...],
    config: ConsistencyCheckPlannerConfig = CONFIG,
) -> ConsistencyCheckPlan:
    batches = consistency_check_service._build_consistency_check_batches(
        consistency_application_id=consistency_application_id,
        source_result_manifest_hash=source_result_manifest_hash,
        config=config,
        candidate_bundles=candidates,
    )
    plan_manifest_hash = duplicate_grouping_service.hash_deterministic_payload(
        {
            "project_id": str(project_id),
            "consistency_application_id": str(consistency_application_id),
            "source_result_manifest_hash": source_result_manifest_hash,
            "planner_name": CONSISTENCY_CHECK_PLANNER_NAME,
            "planner_version": CONSISTENCY_CHECK_PLANNER_VERSION,
            "config": {
                "max_candidates_per_batch": config.max_candidates_per_batch,
                "max_evidence_characters_per_batch": config.max_evidence_characters_per_batch,
            },
            "batches": [
                {
                    "batch_index": batch.batch_index,
                    "candidate_ids": [str(candidate_id) for candidate_id in batch.candidate_ids],
                    "candidate_count": batch.candidate_count,
                    "evidence_character_count": batch.evidence_character_count,
                    "batch_manifest_hash": batch.batch_manifest_hash,
                }
                for batch in batches
            ],
        }
    )
    return ConsistencyCheckPlan(
        project_id=project_id,
        consistency_application_id=consistency_application_id,
        source_result_manifest_hash=source_result_manifest_hash,
        planner_name=CONSISTENCY_CHECK_PLANNER_NAME,
        planner_version=CONSISTENCY_CHECK_PLANNER_VERSION,
        config=config,
        batches=batches,
        plan_manifest_hash=plan_manifest_hash,
    )


def _assessment(
    *,
    candidate_id: uuid.UUID,
    evidence_link_ids: tuple[uuid.UUID, ...],
    explanation: str = "assessment",
) -> ConsistencyCheckAssessment:
    return ConsistencyCheckAssessment(
        candidate_id=candidate_id,
        verdict="conflict",
        severity="yellow",
        confidence=0.75,
        explanation=explanation,
        cited_evidence_link_ids=list(evidence_link_ids),
        impact=["scope_review"],
        recommended_actions=["review_source_scope"],
    )


def _batch_result(
    *,
    plan: ConsistencyCheckPlan,
    batch: ConsistencyCheckBatchPlan,
    assessment: ConsistencyCheckAssessment | None,
) -> ConsistencyCheckBatchExecutionResult:
    response = ConsistencyCheckResponse(assessments=[] if assessment is None else [assessment])
    return ConsistencyCheckBatchExecutionResult(
        project_id=plan.project_id,
        consistency_application_id=plan.consistency_application_id,
        source_result_manifest_hash=plan.source_result_manifest_hash,
        plan_manifest_hash=plan.plan_manifest_hash,
        batch_index=batch.batch_index,
        batch_manifest_hash=batch.batch_manifest_hash,
        input_batch_id=None if batch.candidate_count == 0 else uuid.uuid4(),
        inference_run_id=None if batch.candidate_count == 0 else uuid.uuid5(uuid.NAMESPACE_URL, f"run:{batch.batch_index}"),
        request_hash=None if batch.candidate_count == 0 else f"{batch.batch_index + 10:064x}"[-64:],
        message_content_hash=None if batch.candidate_count == 0 else f"{batch.batch_index + 20:064x}"[-64:],
        skipped_empty=batch.candidate_count == 0,
        reused_completed_run=True,
        response=response,
        response_model="gpt-test" if batch.candidate_count else None,
        prompt_tokens=10 if batch.candidate_count else None,
        completion_tokens=5 if batch.candidate_count else None,
        total_tokens=15 if batch.candidate_count else None,
    )


def _plan_execution_result(
    *,
    plan: ConsistencyCheckPlan,
    batch_results: tuple[ConsistencyCheckBatchExecutionResult, ...],
):
    ordered_assessments = [
        execution_service._validate_and_order_batch_result(
            authoritative_plan=plan,
            batch=batch,
            result=batch_result,
        )
        for batch, batch_result in zip(plan.batches, batch_results, strict=True)
    ]
    return execution_service._build_plan_execution_result(
        authoritative_plan=plan,
        ordered_batch_results=batch_results,
        ordered_assessments_by_batch=ordered_assessments,
    )


def _build_authoritative_fixture():
    project_id = uuid.uuid4()
    consistency_application_id = uuid.uuid4()
    orchestration_id = uuid.uuid4()
    source_result_manifest_hash = "a" * 64
    candidate_a = _candidate_bundle(
        candidate_id=uuid.UUID("00000000-0000-0000-0000-000000000101"),
        fact_id=uuid.uuid4(),
        fact_value_id=uuid.uuid4(),
        evidence_link_ids=(
            uuid.UUID("00000000-0000-0000-0000-000000000201"),
            uuid.UUID("00000000-0000-0000-0000-000000000202"),
        ),
    )
    candidate_b = _candidate_bundle(
        candidate_id=uuid.UUID("00000000-0000-0000-0000-000000000102"),
        fact_id=uuid.uuid4(),
        fact_value_id=uuid.uuid4(),
        evidence_link_ids=(uuid.UUID("00000000-0000-0000-0000-000000000203"),),
    )
    plan = _build_plan(
        project_id=project_id,
        consistency_application_id=consistency_application_id,
        source_result_manifest_hash=source_result_manifest_hash,
        candidates=(candidate_a, candidate_b),
        config=ConsistencyCheckPlannerConfig(
            max_candidates_per_batch=1,
            max_evidence_characters_per_batch=500,
        ),
    )
    batch0, batch1 = plan.batches
    batch_results = (
        _batch_result(
            plan=plan,
            batch=batch0,
            assessment=_assessment(
                candidate_id=candidate_a.candidate_id,
                evidence_link_ids=(
                    candidate_a.members[0].evidences[1].evidence_link_id,
                    candidate_a.members[0].evidences[0].evidence_link_id,
                ),
            ),
        ),
        _batch_result(
            plan=plan,
            batch=batch1,
            assessment=_assessment(
                candidate_id=candidate_b.candidate_id,
                evidence_link_ids=(candidate_b.members[0].evidences[0].evidence_link_id,),
            ),
        ),
    )
    execution_result = _plan_execution_result(plan=plan, batch_results=batch_results)
    authenticated_source = SimpleNamespace(
        project_id=project_id,
        application=FactValueConsistencyCandidateApplicationLedger(
            id=consistency_application_id,
            duplicate_grouping_application_id=uuid.uuid4(),
            orchestration_id=orchestration_id,
            extraction_run_id=uuid.uuid4(),
            algorithm_version="cross_batch_multi_value_v1",
            input_manifest_hash="c" * 64,
            result_manifest_hash=source_result_manifest_hash,
            candidate_count=2,
            member_count=2,
            created_at=datetime.now(timezone.utc),
        ),
        write_plan=SimpleNamespace(),
        candidate_ledgers=(),
        member_ledgers=(),
    )
    return plan, batch_results, execution_result, authenticated_source


def _install_common_monkeypatches(
    monkeypatch: pytest.MonkeyPatch,
    *,
    store: LedgerStore,
    plan: ConsistencyCheckPlan,
    batch_results: tuple[ConsistencyCheckBatchExecutionResult, ...],
    authenticated_source,
) -> None:
    async def fake_build_plan(_session_factory, *, consistency_application_id, config):
        assert consistency_application_id == plan.consistency_application_id
        assert config == plan.config
        return plan

    async def fake_auth_source(_session_factory, *, consistency_application_id):
        assert consistency_application_id == plan.consistency_application_id
        return authenticated_source

    async def fake_auth_batch(
        _session_factory,
        *,
        authoritative_plan,
        batch,
        prompt,
        provider,
        requested_model,
        inference_run_id,
    ):
        assert authoritative_plan == plan
        assert prompt == PROMPT
        expected = batch_results[batch.batch_index]
        assert inference_run_id == expected.inference_run_id
        return expected

    async def fake_get_application_for_update(_session, *, execution_identity_hash):
        if store.application is None:
            return None
        if store.application.execution_identity_hash != execution_identity_hash:
            return None
        return store.application

    async def fake_get_application_by_execution_identity(_session, *, execution_identity_hash):
        if store.application is None:
            return None
        if store.application.execution_identity_hash != execution_identity_hash:
            return None
        return persistence_service.ConsistencyCheckApplicationLedgerRecord(
            id=store.application.id,
            project_id=store.application.project_id,
            consistency_application_id=store.application.consistency_application_id,
            orchestration_id=store.application.orchestration_id,
            source_result_manifest_hash=store.application.source_result_manifest_hash,
            plan_manifest_hash=store.application.plan_manifest_hash,
            execution_identity_hash=store.application.execution_identity_hash,
            result_manifest_hash=store.application.result_manifest_hash,
            prompt_contract_hash=store.application.prompt_contract_hash,
            provider=store.application.provider,
            requested_model=store.application.requested_model,
            executor_name=store.application.executor_name,
            executor_version=store.application.executor_version,
            batch_count=store.application.batch_count,
            executed_batch_count=store.application.executed_batch_count,
            skipped_empty_batch_count=store.application.skipped_empty_batch_count,
            inference_run_count=store.application.inference_run_count,
            assessment_count=store.application.assessment_count,
            created_at=store.application.created_at,
        )

    async def fake_create_application(_session, application):
        store.application = application
        return application

    async def fake_create_batches(_session, batches):
        store.batches.extend(batches)
        return batches

    async def fake_create_assessments(_session, assessments):
        store.assessments.extend(assessments)
        return assessments

    async def fake_create_citations(_session, citations):
        store.citations.extend(citations)
        return citations

    async def fake_list_batches(_session, *, consistency_check_application_id):
        assert store.application is not None
        assert consistency_check_application_id == store.application.id
        return tuple(
            persistence_service.ConsistencyCheckBatchLedgerRecord(
                id=item.id,
                consistency_check_application_id=item.consistency_check_application_id,
                batch_index=item.batch_index,
                batch_manifest_hash=item.batch_manifest_hash,
                skipped_empty=item.skipped_empty,
                input_batch_id=item.input_batch_id,
                inference_run_id=item.inference_run_id,
                request_hash=item.request_hash,
                message_content_hash=item.message_content_hash,
                created_at=item.created_at,
            )
            for item in sorted(store.batches, key=lambda value: (value.batch_index, str(value.id)))
        )

    async def fake_list_assessments(_session, *, consistency_check_application_id):
        assert store.application is not None
        assert consistency_check_application_id == store.application.id
        return tuple(
            ConsistencyAssessmentLedgerRecord(
                id=item.id,
                consistency_check_application_id=item.consistency_check_application_id,
                source_consistency_application_id=item.source_consistency_application_id,
                source_consistency_candidate_id=item.source_consistency_candidate_id,
                batch_index=item.batch_index,
                verdict=item.verdict,
                severity=item.severity,
                confidence=item.confidence,
                explanation=item.explanation,
                impact_json=tuple(item.impact_json),
                recommended_actions_json=tuple(item.recommended_actions_json),
                assessment_manifest_hash=item.assessment_manifest_hash,
                created_at=item.created_at,
            )
            for item in sorted(
                store.assessments,
                key=lambda value: (value.batch_index, str(value.source_consistency_candidate_id), str(value.id)),
            )
        )

    async def fake_list_citations(_session, *, consistency_check_application_id):
        assert store.application is not None
        assert consistency_check_application_id == store.application.id
        return tuple(
            ConsistencyAssessmentCitationLedgerRecord(
                id=item.id,
                assessment_id=item.assessment_id,
                source_consistency_application_id=item.source_consistency_application_id,
                source_consistency_candidate_id=item.source_consistency_candidate_id,
                source_fact_value_id=item.source_fact_value_id,
                evidence_link_id=item.evidence_link_id,
                citation_order=item.citation_order,
                created_at=item.created_at,
            )
            for item in sorted(
                store.citations,
                key=lambda value: (
                    next(
                        assessment.batch_index
                        for assessment in store.assessments
                        if assessment.id == value.assessment_id
                    ),
                    str(value.source_consistency_candidate_id),
                    value.citation_order,
                    str(value.id),
                ),
            )
        )

    monkeypatch.setattr(consistency_check_service, "build_consistency_check_plan", fake_build_plan)
    monkeypatch.setattr(
        duplicate_grouping_service,
        "authenticate_fact_value_consistency_candidate_application",
        fake_auth_source,
    )
    monkeypatch.setattr(
        execution_service,
        "authenticate_completed_consistency_check_batch_run",
        fake_auth_batch,
    )
    monkeypatch.setattr(
        persistence_service,
        "_reauthenticate_source_in_write_transaction",
        lambda *_args, **_kwargs: asyncio.sleep(0),
    )
    monkeypatch.setattr(
        persistence_service.consistency_check_repository,
        "get_consistency_check_application_for_update",
        fake_get_application_for_update,
    )
    monkeypatch.setattr(
        persistence_service.consistency_check_repository,
        "get_consistency_check_application_ledger_by_execution_identity",
        fake_get_application_by_execution_identity,
    )
    monkeypatch.setattr(
        persistence_service.consistency_check_repository,
        "create_consistency_check_application",
        fake_create_application,
    )
    monkeypatch.setattr(
        persistence_service.consistency_check_repository,
        "create_consistency_check_batches",
        fake_create_batches,
    )
    monkeypatch.setattr(
        persistence_service.consistency_check_repository,
        "create_consistency_assessments",
        fake_create_assessments,
    )
    monkeypatch.setattr(
        persistence_service.consistency_check_repository,
        "create_consistency_assessment_citations",
        fake_create_citations,
    )
    monkeypatch.setattr(
        persistence_service.consistency_check_repository,
        "list_consistency_check_batch_ledgers",
        fake_list_batches,
    )
    monkeypatch.setattr(
        persistence_service.consistency_check_repository,
        "list_consistency_assessment_ledgers",
        fake_list_assessments,
    )
    monkeypatch.setattr(
        persistence_service.consistency_check_repository,
        "list_consistency_assessment_citation_ledgers",
        fake_list_citations,
    )


def _make_integrity_error(constraint_name: str) -> IntegrityError:
    class _Diag:
        def __init__(self, name: str) -> None:
            self.constraint_name = name

    class _Orig(Exception):
        def __init__(self, name: str) -> None:
            self.diag = _Diag(name)

    return IntegrityError("stmt", {}, _Orig(constraint_name))


def test_persist_consistency_check_plan_result_writes_four_layer_ledgers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, batch_results, execution_result, authenticated_source = _build_authoritative_fixture()
    store = LedgerStore()
    session_factory = SessionFactory()
    _install_common_monkeypatches(
        monkeypatch,
        store=store,
        plan=plan,
        batch_results=batch_results,
        authenticated_source=authenticated_source,
    )

    result = run_async(
        persistence_service.persist_consistency_check_plan_result(
            session_factory,
            plan=plan,
            execution_result=execution_result,
            prompt=PROMPT,
            provider="openai",
            requested_model="gpt-4.1",
        )
    )

    assert result.created_new is True
    assert result.batch_count == 2
    assert result.assessment_count == 2
    assert store.application is not None
    assert len(store.batches) == 2
    assert len(store.assessments) == 2
    assert len(store.citations) == 3
    candidate_a = plan.batches[0].candidates[0]
    assert [citation.evidence_link_id for citation in store.citations[:2]] == [
        candidate_a.members[0].evidences[1].evidence_link_id,
        candidate_a.members[0].evidences[0].evidence_link_id,
    ]
    assert [citation.citation_order for citation in store.citations[:2]] == [0, 1]
    assert all(citation.source_fact_value_id == candidate_a.members[0].fact_value_id for citation in store.citations[:2])
    assert session_factory.open_count == 0


def test_persist_consistency_check_plan_result_persists_empty_plan_as_skipped_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id = uuid.uuid4()
    application_id = uuid.uuid4()
    source_result_manifest_hash = "a" * 64
    plan = _build_plan(
        project_id=project_id,
        consistency_application_id=application_id,
        source_result_manifest_hash=source_result_manifest_hash,
        candidates=(),
    )
    empty_batch = plan.batches[0]
    batch_result = _batch_result(plan=plan, batch=empty_batch, assessment=None)
    execution_result = _plan_execution_result(plan=plan, batch_results=(batch_result,))
    authenticated_source = SimpleNamespace(
        project_id=project_id,
        application=FactValueConsistencyCandidateApplicationLedger(
            id=application_id,
            duplicate_grouping_application_id=uuid.uuid4(),
            orchestration_id=uuid.uuid4(),
            extraction_run_id=uuid.uuid4(),
            algorithm_version="cross_batch_multi_value_v1",
            input_manifest_hash="c" * 64,
            result_manifest_hash=source_result_manifest_hash,
            candidate_count=0,
            member_count=0,
            created_at=datetime.now(timezone.utc),
        ),
        write_plan=SimpleNamespace(),
        candidate_ledgers=(),
        member_ledgers=(),
    )
    store = LedgerStore()
    session_factory = SessionFactory()
    _install_common_monkeypatches(
        monkeypatch,
        store=store,
        plan=plan,
        batch_results=(batch_result,),
        authenticated_source=authenticated_source,
    )

    result = run_async(
        persistence_service.persist_consistency_check_plan_result(
            session_factory,
            plan=plan,
            execution_result=execution_result,
            prompt=PROMPT,
            provider="openai",
            requested_model="gpt-4.1",
        )
    )

    assert result.created_new is True
    assert len(store.batches) == 1
    assert store.batches[0].skipped_empty is True
    assert store.batches[0].inference_run_id is None
    assert len(store.assessments) == 0
    assert len(store.citations) == 0


def test_persist_consistency_check_plan_result_rejects_authenticated_run_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, batch_results, execution_result, authenticated_source = _build_authoritative_fixture()
    store = LedgerStore()
    session_factory = SessionFactory()
    _install_common_monkeypatches(
        monkeypatch,
        store=store,
        plan=plan,
        batch_results=batch_results,
        authenticated_source=authenticated_source,
    )

    async def fake_auth_batch(*_args, **_kwargs):
        raise execution_service.ConsistencyCheckExecutionError(
            "consistency_check_persistence_run_metadata_mismatch"
        )

    monkeypatch.setattr(
        execution_service,
        "authenticate_completed_consistency_check_batch_run",
        fake_auth_batch,
    )

    with pytest.raises(
        execution_service.ConsistencyCheckExecutionError,
        match="consistency_check_persistence_run_metadata_mismatch",
    ):
        run_async(
            persistence_service.persist_consistency_check_plan_result(
                session_factory,
                plan=plan,
                execution_result=execution_result,
                prompt=PROMPT,
                provider="openai",
                requested_model="gpt-4.1",
            )
        )


def test_persist_consistency_check_plan_result_is_idempotent_and_returns_existing_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, batch_results, execution_result, authenticated_source = _build_authoritative_fixture()
    store = LedgerStore()
    session_factory = SessionFactory()
    _install_common_monkeypatches(
        monkeypatch,
        store=store,
        plan=plan,
        batch_results=batch_results,
        authenticated_source=authenticated_source,
    )

    first = run_async(
        persistence_service.persist_consistency_check_plan_result(
            session_factory,
            plan=plan,
            execution_result=execution_result,
            prompt=PROMPT,
            provider="openai",
            requested_model="gpt-4.1",
        )
    )
    second = run_async(
        persistence_service.persist_consistency_check_plan_result(
            session_factory,
            plan=plan,
            execution_result=execution_result,
            prompt=PROMPT,
            provider="openai",
            requested_model="gpt-4.1",
        )
    )

    assert first.created_new is True
    assert second.created_new is False
    assert second.consistency_check_application_id == first.consistency_check_application_id
    assert len(store.batches) == 2
    assert len(store.assessments) == 2
    assert len(store.citations) == 3


def test_persist_consistency_check_plan_result_detects_subledger_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "SENSITIVE_ASSESSMENT_SENTINEL"
    plan, batch_results, execution_result, authenticated_source = _build_authoritative_fixture()
    execution_result = execution_result.model_copy(
        update={
            "assessments": tuple(
                [
                    batch_results[0].response.assessments[0].model_copy(update={"explanation": sentinel}),
                    batch_results[1].response.assessments[0],
                ]
            )
        }
    )
    batch_results = (
        batch_results[0].model_copy(
            update={
                "response": ConsistencyCheckResponse(
                    assessments=[
                        batch_results[0].response.assessments[0].model_copy(
                            update={"explanation": sentinel}
                        )
                    ]
                )
            },
        ),
        batch_results[1],
    )
    execution_result = _plan_execution_result(plan=plan, batch_results=batch_results)
    store = LedgerStore()
    session_factory = SessionFactory()
    _install_common_monkeypatches(
        monkeypatch,
        store=store,
        plan=plan,
        batch_results=batch_results,
        authenticated_source=authenticated_source,
    )

    run_async(
        persistence_service.persist_consistency_check_plan_result(
            session_factory,
            plan=plan,
            execution_result=execution_result,
            prompt=PROMPT,
            provider="openai",
            requested_model="gpt-4.1",
        )
    )
    store.citations.pop()

    with pytest.raises(
        persistence_service.ConsistencyCheckPersistenceInvariantError,
        match="consistency_check_persistence_immutable_ledger_mismatch",
    ) as exc_info:
        run_async(
            persistence_service.persist_consistency_check_plan_result(
                session_factory,
                plan=plan,
                execution_result=execution_result,
                prompt=PROMPT,
                provider="openai",
                requested_model="gpt-4.1",
            )
        )

    assert sentinel not in str(exc_info.value)


def test_persist_consistency_check_plan_result_handles_unique_conflict_by_reloading_existing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, batch_results, execution_result, authenticated_source = _build_authoritative_fixture()
    store = LedgerStore()
    session_factory = SessionFactory()
    _install_common_monkeypatches(
        monkeypatch,
        store=store,
        plan=plan,
        batch_results=batch_results,
        authenticated_source=authenticated_source,
    )
    first = run_async(
        persistence_service.persist_consistency_check_plan_result(
            session_factory,
            plan=plan,
            execution_result=execution_result,
            prompt=PROMPT,
            provider="openai",
            requested_model="gpt-4.1",
        )
    )

    async def fake_get_for_update(_session, *, execution_identity_hash):
        return None

    async def fake_create_application(_session, application):
        raise _make_integrity_error("uq_ccapp_exec_identity_hash")

    monkeypatch.setattr(
        persistence_service.consistency_check_repository,
        "get_consistency_check_application_for_update",
        fake_get_for_update,
    )
    monkeypatch.setattr(
        persistence_service.consistency_check_repository,
        "create_consistency_check_application",
        fake_create_application,
    )

    second = run_async(
        persistence_service.persist_consistency_check_plan_result(
            session_factory,
            plan=plan,
            execution_result=execution_result,
            prompt=PROMPT,
            provider="openai",
            requested_model="gpt-4.1",
        )
    )

    assert first.consistency_check_application_id == second.consistency_check_application_id
    assert second.created_new is False


def test_persist_consistency_check_plan_result_does_not_swallow_unknown_integrity_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, batch_results, execution_result, authenticated_source = _build_authoritative_fixture()
    store = LedgerStore()
    session_factory = SessionFactory()
    _install_common_monkeypatches(
        monkeypatch,
        store=store,
        plan=plan,
        batch_results=batch_results,
        authenticated_source=authenticated_source,
    )

    async def fake_create_application(_session, application):
        raise _make_integrity_error("uq_some_other_constraint")

    monkeypatch.setattr(
        persistence_service.consistency_check_repository,
        "create_consistency_check_application",
        fake_create_application,
    )

    with pytest.raises(IntegrityError):
        run_async(
            persistence_service.persist_consistency_check_plan_result(
                session_factory,
                plan=plan,
                execution_result=execution_result,
                prompt=PROMPT,
                provider="openai",
                requested_model="gpt-4.1",
            )
        )


def test_persist_consistency_check_plan_result_fails_closed_on_source_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, batch_results, execution_result, authenticated_source = _build_authoritative_fixture()
    store = LedgerStore()
    session_factory = SessionFactory()
    _install_common_monkeypatches(
        monkeypatch,
        store=store,
        plan=plan,
        batch_results=batch_results,
        authenticated_source=authenticated_source,
    )

    async def fake_reauth(*_args, **_kwargs):
        raise persistence_service.ConsistencyCheckPersistenceStateError(
            "consistency_check_persistence_source_project_drift"
        )

    monkeypatch.setattr(
        persistence_service,
        "_reauthenticate_source_in_write_transaction",
        fake_reauth,
    )

    with pytest.raises(
        persistence_service.ConsistencyCheckPersistenceStateError,
        match="consistency_check_persistence_source_project_drift",
    ):
        run_async(
            persistence_service.persist_consistency_check_plan_result(
                session_factory,
                plan=plan,
                execution_result=execution_result,
                prompt=PROMPT,
                provider="openai",
                requested_model="gpt-4.1",
            )
        )


def test_persist_consistency_check_plan_result_rejects_execution_result_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, batch_results, execution_result, authenticated_source = _build_authoritative_fixture()
    store = LedgerStore()
    session_factory = SessionFactory()
    _install_common_monkeypatches(
        monkeypatch,
        store=store,
        plan=plan,
        batch_results=batch_results,
        authenticated_source=authenticated_source,
    )
    tampered = execution_result.model_copy(update={"result_manifest_hash": "f" * 64})

    with pytest.raises(
        execution_service.ConsistencyCheckExecutionError,
        match="consistency_check_plan_result_invalid",
    ):
        run_async(
            persistence_service.persist_consistency_check_plan_result(
                session_factory,
                plan=plan,
                execution_result=tampered,
                prompt=PROMPT,
                provider="openai",
                requested_model="gpt-4.1",
            )
        )
