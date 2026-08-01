from __future__ import annotations

import asyncio
import uuid
import pytest

from app.agents.prompt_registry import get_prompt
from app.schemas.agent_consistency_check import ConsistencyCheckAssessment
from app.schemas.consistency_check import (
    CONSISTENCY_CHECK_PLANNER_NAME,
    CONSISTENCY_CHECK_PLANNER_VERSION,
    ConsistencyCheckBatchPlan,
    ConsistencyCheckPlan,
    ConsistencyCheckPlannerConfig,
)
from app.schemas.consistency_check_execution import ConsistencyCheckPlanExecutionResult
from app.schemas.consistency_check_persistence import ConsistencyCheckPersistenceResult
from app.schemas.consistency_check_workflow import ConsistencyCheckWorkflowResult
from app.services import consistency_check_execution as execution_service
from app.services import consistency_check_persistence as persistence_service
from app.services import consistency_check_workflow as workflow_service


PROMPT = get_prompt("agent2_consistency_check", "1.0.0")
CONFIG = ConsistencyCheckPlannerConfig(
    max_candidates_per_batch=8,
    max_evidence_characters_per_batch=500,
)


def run_async(awaitable):
    return asyncio.run(awaitable)


class SessionFactory:
    def __init__(self) -> None:
        self.open_count = 0

    def __call__(self):
        factory = self

        class _Context:
            async def __aenter__(self_inner):
                factory.open_count += 1
                return object()

            async def __aexit__(self_inner, exc_type, exc, tb):
                factory.open_count -= 1
                return False

        return _Context()


def _plan(
    *,
    project_id: uuid.UUID | None = None,
    consistency_application_id: uuid.UUID | None = None,
    batch_count: int = 1,
    empty: bool = False,
) -> ConsistencyCheckPlan:
    resolved_project_id = project_id or uuid.uuid4()
    resolved_application_id = consistency_application_id or uuid.uuid4()
    batches = tuple(
        ConsistencyCheckBatchPlan(
            batch_index=index,
            candidate_ids=() if empty else (uuid.uuid5(uuid.NAMESPACE_URL, f"candidate:{index}"),),
            candidate_count=0 if empty else 1,
            evidence_character_count=0 if empty else 10,
            batch_manifest_hash=f"{index + 1:064x}"[-64:],
            candidates=(),
        )
        for index in range(batch_count)
    )
    return ConsistencyCheckPlan(
        project_id=resolved_project_id,
        consistency_application_id=resolved_application_id,
        source_result_manifest_hash="a" * 64,
        planner_name=CONSISTENCY_CHECK_PLANNER_NAME,
        planner_version=CONSISTENCY_CHECK_PLANNER_VERSION,
        config=CONFIG,
        batches=batches,
        plan_manifest_hash="b" * 64,
    )


def _execution_result(
    *,
    plan: ConsistencyCheckPlan,
    assessment_count: int = 1,
) -> ConsistencyCheckPlanExecutionResult:
    assessments = tuple(
        ConsistencyCheckAssessment(
            candidate_id=uuid.uuid5(uuid.NAMESPACE_URL, f"assessment:{index}"),
            verdict="conflict",
            severity="yellow",
            confidence=0.8,
            explanation="ok",
            cited_evidence_link_ids=[uuid.uuid5(uuid.NAMESPACE_URL, f"evidence:{index}")],
            impact=["scope_review"],
            recommended_actions=["review_source_scope"],
        )
        for index in range(assessment_count)
    )
    return ConsistencyCheckPlanExecutionResult(
        project_id=plan.project_id,
        consistency_application_id=plan.consistency_application_id,
        source_result_manifest_hash=plan.source_result_manifest_hash,
        plan_manifest_hash=plan.plan_manifest_hash,
        batch_count=len(plan.batches),
        executed_batch_count=len(plan.batches),
        skipped_empty_batch_count=sum(1 for batch in plan.batches if batch.candidate_count == 0),
        inference_run_ids=tuple(
            None if batch.candidate_count == 0 else uuid.uuid5(uuid.NAMESPACE_URL, f"run:{batch.batch_index}")
            for batch in plan.batches
        ),
        assessments=assessments,
        result_manifest_hash="c" * 64,
    )


def _persistence_result(
    *,
    created_new: bool = True,
    batch_count: int = 1,
    assessment_count: int = 1,
    consistency_check_application_id: uuid.UUID | None = None,
) -> ConsistencyCheckPersistenceResult:
    return ConsistencyCheckPersistenceResult(
        consistency_check_application_id=consistency_check_application_id or uuid.uuid4(),
        created_new=created_new,
        batch_count=batch_count,
        assessment_count=assessment_count,
    )


def test_run_consistency_check_workflow_calls_build_execute_persist_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(batch_count=2)
    execution_result = _execution_result(plan=plan, assessment_count=2)
    persistence_result = _persistence_result(
        created_new=True,
        batch_count=2,
        assessment_count=2,
    )
    session_factory = SessionFactory()
    llm_client = object()
    call_order: list[str] = []

    async def fake_build(_session_factory, *, consistency_application_id, config):
        assert _session_factory is session_factory
        assert consistency_application_id == plan.consistency_application_id
        assert config == CONFIG
        call_order.append("build")
        return plan

    async def fake_execute(
        _session_factory,
        *,
        project_id,
        plan: ConsistencyCheckPlan,
        prompt,
        llm_client,
        provider,
        requested_model,
    ):
        assert _session_factory is session_factory
        assert project_id == expected_plan.project_id
        assert plan == expected_plan
        assert prompt == PROMPT
        assert provider == "openai"
        assert requested_model == "gpt-4.1"
        call_order.append("execute")
        return execution_result

    async def fake_persist(
        _session_factory,
        *,
        plan: ConsistencyCheckPlan,
        execution_result: ConsistencyCheckPlanExecutionResult,
        prompt,
        provider,
        requested_model,
    ):
        assert _session_factory is session_factory
        assert plan == expected_plan
        assert execution_result == expected_execution_result
        assert prompt == PROMPT
        assert provider == "openai"
        assert requested_model == "gpt-4.1"
        call_order.append("persist")
        return persistence_result

    expected_plan = plan
    expected_execution_result = execution_result
    monkeypatch.setattr(workflow_service.consistency_check_service, "build_consistency_check_plan", fake_build)
    monkeypatch.setattr(workflow_service.execution_service, "execute_consistency_check_plan", fake_execute)
    monkeypatch.setattr(
        workflow_service.persistence_service,
        "persist_consistency_check_plan_result",
        fake_persist,
    )

    result = run_async(
        workflow_service.run_consistency_check_workflow(
            session_factory,
            project_id=plan.project_id,
            consistency_application_id=plan.consistency_application_id,
            config=CONFIG,
            prompt=PROMPT,
            llm_client=llm_client,
            provider="openai",
            requested_model="gpt-4.1",
        )
    )

    assert call_order == ["build", "execute", "persist"]
    assert result == ConsistencyCheckWorkflowResult(
        project_id=plan.project_id,
        consistency_application_id=plan.consistency_application_id,
        plan_manifest_hash=plan.plan_manifest_hash,
        execution_result_manifest_hash=execution_result.result_manifest_hash,
        consistency_check_application_id=persistence_result.consistency_check_application_id,
        created_new=True,
        batch_count=2,
        assessment_count=2,
    )
    assert session_factory.open_count == 0


def test_run_consistency_check_workflow_rejects_project_mismatch_before_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    session_factory = SessionFactory()

    async def fake_build(_session_factory, *, consistency_application_id, config):
        return plan

    async def fail_execute(*args, **kwargs):
        raise AssertionError("execute should not be called")

    async def fail_persist(*args, **kwargs):
        raise AssertionError("persist should not be called")

    monkeypatch.setattr(workflow_service.consistency_check_service, "build_consistency_check_plan", fake_build)
    monkeypatch.setattr(workflow_service.execution_service, "execute_consistency_check_plan", fail_execute)
    monkeypatch.setattr(
        workflow_service.persistence_service,
        "persist_consistency_check_plan_result",
        fail_persist,
    )

    with pytest.raises(
        workflow_service.ConsistencyCheckWorkflowStateError,
        match="consistency_check_workflow_project_id_mismatch",
    ):
        run_async(
            workflow_service.run_consistency_check_workflow(
                session_factory,
                project_id=uuid.uuid4(),
                consistency_application_id=plan.consistency_application_id,
                config=CONFIG,
                prompt=PROMPT,
                llm_client=object(),
                provider="openai",
                requested_model="gpt-4.1",
            )
        )

    assert session_factory.open_count == 0


def test_run_consistency_check_workflow_persists_empty_plan_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(empty=True)
    execution_result = _execution_result(plan=plan, assessment_count=0)
    persistence_result = _persistence_result(
        created_new=True,
        batch_count=1,
        assessment_count=0,
    )
    call_order: list[str] = []

    async def fake_build(*args, **kwargs):
        call_order.append("build")
        return plan

    async def fake_execute(*args, **kwargs):
        call_order.append("execute")
        return execution_result

    async def fake_persist(*args, **kwargs):
        call_order.append("persist")
        return persistence_result

    monkeypatch.setattr(workflow_service.consistency_check_service, "build_consistency_check_plan", fake_build)
    monkeypatch.setattr(workflow_service.execution_service, "execute_consistency_check_plan", fake_execute)
    monkeypatch.setattr(
        workflow_service.persistence_service,
        "persist_consistency_check_plan_result",
        fake_persist,
    )

    result = run_async(
        workflow_service.run_consistency_check_workflow(
            SessionFactory(),
            project_id=plan.project_id,
            consistency_application_id=plan.consistency_application_id,
            config=CONFIG,
            prompt=PROMPT,
            llm_client=object(),
            provider="openai",
            requested_model="gpt-4.1",
        )
    )

    assert call_order == ["build", "execute", "persist"]
    assert result.assessment_count == 0
    assert result.batch_count == 1


def test_run_consistency_check_workflow_skips_persist_when_execute_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()

    async def fake_build(*args, **kwargs):
        return plan

    async def fake_execute(*args, **kwargs):
        raise execution_service.ConsistencyCheckExecutionError("workflow_execute_failed")

    async def fail_persist(*args, **kwargs):
        raise AssertionError("persist should not be called")

    monkeypatch.setattr(workflow_service.consistency_check_service, "build_consistency_check_plan", fake_build)
    monkeypatch.setattr(workflow_service.execution_service, "execute_consistency_check_plan", fake_execute)
    monkeypatch.setattr(
        workflow_service.persistence_service,
        "persist_consistency_check_plan_result",
        fail_persist,
    )

    with pytest.raises(
        execution_service.ConsistencyCheckExecutionError,
        match="workflow_execute_failed",
    ):
        run_async(
            workflow_service.run_consistency_check_workflow(
                SessionFactory(),
                project_id=plan.project_id,
                consistency_application_id=plan.consistency_application_id,
                config=CONFIG,
                prompt=PROMPT,
                llm_client=object(),
                provider="openai",
                requested_model="gpt-4.1",
            )
        )


def test_run_consistency_check_workflow_retries_after_persist_failure_without_repeating_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    execution_result = _execution_result(plan=plan, assessment_count=1)
    success_persistence_result = _persistence_result(
        created_new=True,
        batch_count=1,
        assessment_count=1,
    )
    execute_call_count = 0
    llm_call_count = 0
    persist_call_count = 0

    async def fake_build(*args, **kwargs):
        return plan

    async def fake_execute(*args, **kwargs):
        nonlocal execute_call_count, llm_call_count
        execute_call_count += 1
        if execute_call_count == 1:
            llm_call_count += 1
        return execution_result

    async def fake_persist(*args, **kwargs):
        nonlocal persist_call_count
        persist_call_count += 1
        if persist_call_count == 1:
            raise persistence_service.ConsistencyCheckPersistenceStateError(
                "workflow_persist_failed"
            )
        return success_persistence_result

    monkeypatch.setattr(workflow_service.consistency_check_service, "build_consistency_check_plan", fake_build)
    monkeypatch.setattr(workflow_service.execution_service, "execute_consistency_check_plan", fake_execute)
    monkeypatch.setattr(
        workflow_service.persistence_service,
        "persist_consistency_check_plan_result",
        fake_persist,
    )

    with pytest.raises(
        persistence_service.ConsistencyCheckPersistenceStateError,
        match="workflow_persist_failed",
    ):
        run_async(
            workflow_service.run_consistency_check_workflow(
                SessionFactory(),
                project_id=plan.project_id,
                consistency_application_id=plan.consistency_application_id,
                config=CONFIG,
                prompt=PROMPT,
                llm_client=object(),
                provider="openai",
                requested_model="gpt-4.1",
            )
        )

    result = run_async(
        workflow_service.run_consistency_check_workflow(
            SessionFactory(),
            project_id=plan.project_id,
            consistency_application_id=plan.consistency_application_id,
            config=CONFIG,
            prompt=PROMPT,
            llm_client=object(),
            provider="openai",
            requested_model="gpt-4.1",
        )
    )

    assert execute_call_count == 2
    assert llm_call_count == 1
    assert result.consistency_check_application_id == success_persistence_result.consistency_check_application_id


def test_run_consistency_check_workflow_repeat_returns_same_persisted_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    execution_result = _execution_result(plan=plan, assessment_count=1)
    ledger_id = uuid.uuid4()
    persistence_results = iter(
        (
            _persistence_result(
                created_new=True,
                batch_count=1,
                assessment_count=1,
                consistency_check_application_id=ledger_id,
            ),
            _persistence_result(
                created_new=False,
                batch_count=1,
                assessment_count=1,
                consistency_check_application_id=ledger_id,
            ),
        )
    )

    async def fake_build(*args, **kwargs):
        return plan

    async def fake_execute(*args, **kwargs):
        return execution_result

    async def fake_persist(*args, **kwargs):
        return next(persistence_results)

    monkeypatch.setattr(workflow_service.consistency_check_service, "build_consistency_check_plan", fake_build)
    monkeypatch.setattr(workflow_service.execution_service, "execute_consistency_check_plan", fake_execute)
    monkeypatch.setattr(
        workflow_service.persistence_service,
        "persist_consistency_check_plan_result",
        fake_persist,
    )

    first = run_async(
        workflow_service.run_consistency_check_workflow(
            SessionFactory(),
            project_id=plan.project_id,
            consistency_application_id=plan.consistency_application_id,
            config=CONFIG,
            prompt=PROMPT,
            llm_client=object(),
            provider="openai",
            requested_model="gpt-4.1",
        )
    )
    second = run_async(
        workflow_service.run_consistency_check_workflow(
            SessionFactory(),
            project_id=plan.project_id,
            consistency_application_id=plan.consistency_application_id,
            config=CONFIG,
            prompt=PROMPT,
            llm_client=object(),
            provider="openai",
            requested_model="gpt-4.1",
        )
    )

    assert first.consistency_check_application_id == second.consistency_check_application_id
    assert first.created_new is True
    assert second.created_new is False


def test_run_consistency_check_workflow_normalizes_provider_and_model_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    execution_result = _execution_result(plan=plan, assessment_count=1)
    persistence_result = _persistence_result(
        created_new=True,
        batch_count=1,
        assessment_count=1,
    )
    seen_identities: list[tuple[str, str]] = []

    async def fake_build(*args, **kwargs):
        return plan

    async def fake_execute(*args, **kwargs):
        seen_identities.append((kwargs["provider"], kwargs["requested_model"]))
        return execution_result

    async def fake_persist(*args, **kwargs):
        seen_identities.append((kwargs["provider"], kwargs["requested_model"]))
        return persistence_result

    monkeypatch.setattr(workflow_service.consistency_check_service, "build_consistency_check_plan", fake_build)
    monkeypatch.setattr(workflow_service.execution_service, "execute_consistency_check_plan", fake_execute)
    monkeypatch.setattr(
        workflow_service.persistence_service,
        "persist_consistency_check_plan_result",
        fake_persist,
    )

    run_async(
        workflow_service.run_consistency_check_workflow(
            SessionFactory(),
            project_id=plan.project_id,
            consistency_application_id=plan.consistency_application_id,
            config=CONFIG,
            prompt=PROMPT,
            llm_client=object(),
            provider=" openai ",
            requested_model=" gpt-4.1 ",
        )
    )

    assert seen_identities == [("openai", "gpt-4.1"), ("openai", "gpt-4.1")]


def test_run_consistency_check_workflow_propagates_cancelled_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()

    async def fake_build(*args, **kwargs):
        return plan

    async def fake_execute(*args, **kwargs):
        raise asyncio.CancelledError()

    async def fail_persist(*args, **kwargs):
        raise AssertionError("persist should not be called")

    monkeypatch.setattr(workflow_service.consistency_check_service, "build_consistency_check_plan", fake_build)
    monkeypatch.setattr(workflow_service.execution_service, "execute_consistency_check_plan", fake_execute)
    monkeypatch.setattr(
        workflow_service.persistence_service,
        "persist_consistency_check_plan_result",
        fail_persist,
    )

    with pytest.raises(asyncio.CancelledError):
        run_async(
            workflow_service.run_consistency_check_workflow(
                SessionFactory(),
                project_id=plan.project_id,
                consistency_application_id=plan.consistency_application_id,
                config=CONFIG,
                prompt=PROMPT,
                llm_client=object(),
                provider="openai",
                requested_model="gpt-4.1",
            )
        )


def test_run_consistency_check_workflow_rejects_invalid_identity_without_leaking_sentinel() -> None:
    sentinel = "  SENSITIVE_MODEL_SENTINEL" * 20
    session_factory = SessionFactory()

    with pytest.raises(
        workflow_service.ConsistencyCheckWorkflowStateError,
        match="consistency_check_workflow_execution_identity_invalid",
    ) as exc_info:
        run_async(
            workflow_service.run_consistency_check_workflow(
                session_factory,
                project_id=uuid.uuid4(),
                consistency_application_id=uuid.uuid4(),
                config=CONFIG,
                prompt=PROMPT,
                llm_client=object(),
                provider="openai",
                requested_model=sentinel,
            )
        )

    assert sentinel.strip() not in str(exc_info.value)
    assert session_factory.open_count == 0
