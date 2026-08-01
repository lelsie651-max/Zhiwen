from __future__ import annotations

import asyncio
import uuid

import pytest

from app.schemas.consistency_check import ConsistencyCheckPlannerConfig
from app.schemas.consistency_check_workflow import ConsistencyCheckWorkflowResult
from app.schemas.consistency_pipeline import FactExtractionConsistencyPipelineResult
from app.schemas.fact_extraction_orchestration import (
    FactExtractionOrchestrationResult,
    FactExtractionOrchestrationStatus,
)
from app.schemas.fact_value_duplicate_grouping import (
    DuplicateGroupingResult,
    FactValueConsistencyCandidateResult,
)
from app.services import consistency_check_workflow as workflow_service
from app.services import consistency_pipeline as pipeline_service
from app.services import fact_extraction_orchestration as orchestration_service
from app.services import fact_value_duplicate_grouping as duplicate_grouping_service


CONFIG = ConsistencyCheckPlannerConfig(
    max_candidates_per_batch=8,
    max_evidence_characters_per_batch=500,
)


def run_async(awaitable):
    return asyncio.run(awaitable)


def _orchestration_result(
    *,
    orchestration_id: uuid.UUID | None = None,
    status: FactExtractionOrchestrationStatus,
) -> FactExtractionOrchestrationResult:
    return FactExtractionOrchestrationResult(
        orchestration_id=orchestration_id or uuid.uuid4(),
        attempt_no=1,
        request_hash="1" * 64,
        plan_hash="2" * 64,
        status=status,
        batch_count=1,
        completed_batch_count=1 if status != FactExtractionOrchestrationStatus.FAILED else 0,
        failed_batch_count=1 if status == FactExtractionOrchestrationStatus.FAILED else 0,
        proposal_count=1,
        created_count=1,
        reused_count=0,
        withheld_count=0,
        batches=(),
    )


def _grouping_result(
    *,
    grouping_application_id: uuid.UUID | None = None,
) -> DuplicateGroupingResult:
    return DuplicateGroupingResult(
        grouping_application_id=grouping_application_id or uuid.uuid4(),
        algorithm_version="cross_batch_exact_v2",
        input_fact_value_count=2,
        duplicate_group_count=1,
        duplicate_member_count=2,
        created_new=True,
    )


def _candidate_result(
    *,
    consistency_application_id: uuid.UUID | None = None,
    duplicate_grouping_application_id: uuid.UUID | None = None,
) -> FactValueConsistencyCandidateResult:
    return FactValueConsistencyCandidateResult(
        consistency_application_id=consistency_application_id or uuid.uuid4(),
        duplicate_grouping_application_id=duplicate_grouping_application_id or uuid.uuid4(),
        algorithm_version="cross_batch_multi_value_v1",
        candidate_count=1,
        member_count=2,
        created_new=True,
    )


def _workflow_result(
    *,
    project_id: uuid.UUID,
    consistency_application_id: uuid.UUID,
    consistency_check_application_id: uuid.UUID | None = None,
    assessment_count: int = 1,
    created_new: bool = True,
) -> ConsistencyCheckWorkflowResult:
    return ConsistencyCheckWorkflowResult(
        project_id=project_id,
        consistency_application_id=consistency_application_id,
        plan_manifest_hash="3" * 64,
        execution_result_manifest_hash="4" * 64,
        consistency_check_application_id=consistency_check_application_id or uuid.uuid4(),
        created_new=created_new,
        batch_count=1,
        assessment_count=assessment_count,
    )


def test_run_fact_extraction_consistency_pipeline_completed_runs_full_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id = uuid.uuid4()
    extraction_run_id = uuid.uuid4()
    worker_token = uuid.uuid4()
    extraction_plan = object()
    extraction_prompt = object()
    consistency_prompt = object()
    llm_client = object()
    orchestration_result = _orchestration_result(status=FactExtractionOrchestrationStatus.COMPLETED)
    grouping_result = _grouping_result()
    candidate_result = _candidate_result(
        duplicate_grouping_application_id=grouping_result.grouping_application_id,
    )
    workflow_result = _workflow_result(
        project_id=project_id,
        consistency_application_id=candidate_result.consistency_application_id,
    )
    call_order: list[str] = []

    async def fake_extract(*args, **kwargs):
        assert kwargs["project_id"] == project_id
        assert kwargs["extraction_run_id"] == extraction_run_id
        assert kwargs["plan"] is extraction_plan
        assert kwargs["prompt"] is extraction_prompt
        assert kwargs["llm_client"] is llm_client
        assert kwargs["provider"] == "agent1-provider"
        assert kwargs["requested_model"] == "agent1-model"
        assert kwargs["worker_token"] == worker_token
        call_order.append("extract")
        return orchestration_result

    async def fake_group(*args, **kwargs):
        assert kwargs["orchestration_id"] == orchestration_result.orchestration_id
        call_order.append("group")
        return grouping_result

    async def fake_candidate(*args, **kwargs):
        assert kwargs["duplicate_grouping_application_id"] == grouping_result.grouping_application_id
        call_order.append("candidate")
        return candidate_result

    async def fake_workflow(*args, **kwargs):
        assert kwargs["project_id"] == project_id
        assert kwargs["consistency_application_id"] == candidate_result.consistency_application_id
        assert kwargs["config"] == CONFIG
        assert kwargs["prompt"] is consistency_prompt
        assert kwargs["llm_client"] is llm_client
        assert kwargs["provider"] == "agent2-provider"
        assert kwargs["requested_model"] == "agent2-model"
        call_order.append("workflow")
        return workflow_result

    monkeypatch.setattr(
        pipeline_service.orchestration_service,
        "execute_fact_extraction_orchestration",
        fake_extract,
    )
    monkeypatch.setattr(
        pipeline_service.duplicate_grouping_service,
        "ensure_cross_batch_duplicate_grouping",
        fake_group,
    )
    monkeypatch.setattr(
        pipeline_service.duplicate_grouping_service,
        "ensure_cross_batch_multi_value_consistency_candidates",
        fake_candidate,
    )
    monkeypatch.setattr(
        pipeline_service.workflow_service,
        "run_consistency_check_workflow",
        fake_workflow,
    )

    result = run_async(
        pipeline_service.run_fact_extraction_consistency_pipeline(
            object(),
            project_id=project_id,
            extraction_run_id=extraction_run_id,
            plan=extraction_plan,
            prompt=extraction_prompt,
            llm_client=llm_client,
            provider="agent1-provider",
            requested_model="agent1-model",
            worker_token=worker_token,
            consistency_config=CONFIG,
            consistency_prompt=consistency_prompt,
            consistency_provider="agent2-provider",
            consistency_requested_model="agent2-model",
        )
    )

    assert call_order == ["extract", "group", "candidate", "workflow"]
    assert result == FactExtractionConsistencyPipelineResult(
        extraction_orchestration_id=orchestration_result.orchestration_id,
        extraction_status=FactExtractionOrchestrationStatus.COMPLETED,
        grouping_application_id=grouping_result.grouping_application_id,
        consistency_application_id=candidate_result.consistency_application_id,
        consistency_check_application_id=workflow_result.consistency_check_application_id,
        consistency_plan_manifest_hash=workflow_result.plan_manifest_hash,
        consistency_execution_result_manifest_hash=workflow_result.execution_result_manifest_hash,
        assessment_count=workflow_result.assessment_count,
        consistency_created_new=workflow_result.created_new,
        skipped_reason=None,
    )


def test_run_fact_extraction_consistency_pipeline_partial_also_runs_full_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestration_result = _orchestration_result(status=FactExtractionOrchestrationStatus.PARTIAL)
    grouping_result = _grouping_result()
    candidate_result = _candidate_result(
        duplicate_grouping_application_id=grouping_result.grouping_application_id,
    )
    workflow_result = _workflow_result(
        project_id=uuid.uuid4(),
        consistency_application_id=candidate_result.consistency_application_id,
    )
    call_order: list[str] = []

    async def fake_extract(*args, **kwargs):
        call_order.append("extract")
        return orchestration_result

    async def fake_group(*args, **kwargs):
        call_order.append("group")
        return grouping_result

    async def fake_candidate(*args, **kwargs):
        call_order.append("candidate")
        return candidate_result

    async def fake_workflow(*args, **kwargs):
        call_order.append("workflow")
        return workflow_result

    monkeypatch.setattr(
        pipeline_service.orchestration_service,
        "execute_fact_extraction_orchestration",
        fake_extract,
    )
    monkeypatch.setattr(
        pipeline_service.duplicate_grouping_service,
        "ensure_cross_batch_duplicate_grouping",
        fake_group,
    )
    monkeypatch.setattr(
        pipeline_service.duplicate_grouping_service,
        "ensure_cross_batch_multi_value_consistency_candidates",
        fake_candidate,
    )
    monkeypatch.setattr(
        pipeline_service.workflow_service,
        "run_consistency_check_workflow",
        fake_workflow,
    )

    result = run_async(
        pipeline_service.run_fact_extraction_consistency_pipeline(
            object(),
            project_id=workflow_result.project_id,
            extraction_run_id=uuid.uuid4(),
            plan=object(),
            prompt=object(),
            llm_client=object(),
            provider="agent1-provider",
            requested_model="agent1-model",
            worker_token=uuid.uuid4(),
            consistency_config=CONFIG,
            consistency_prompt=object(),
            consistency_provider="agent2-provider",
            consistency_requested_model="agent2-model",
        )
    )

    assert call_order == ["extract", "group", "candidate", "workflow"]
    assert result.extraction_status == FactExtractionOrchestrationStatus.PARTIAL


def test_run_fact_extraction_consistency_pipeline_failed_skips_consistency_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestration_result = _orchestration_result(status=FactExtractionOrchestrationStatus.FAILED)

    async def fake_extract(*args, **kwargs):
        return orchestration_result

    async def fail_group(*args, **kwargs):
        raise AssertionError("group should not be called")

    async def fail_candidate(*args, **kwargs):
        raise AssertionError("candidate should not be called")

    async def fail_workflow(*args, **kwargs):
        raise AssertionError("workflow should not be called")

    monkeypatch.setattr(
        pipeline_service.orchestration_service,
        "execute_fact_extraction_orchestration",
        fake_extract,
    )
    monkeypatch.setattr(
        pipeline_service.duplicate_grouping_service,
        "ensure_cross_batch_duplicate_grouping",
        fail_group,
    )
    monkeypatch.setattr(
        pipeline_service.duplicate_grouping_service,
        "ensure_cross_batch_multi_value_consistency_candidates",
        fail_candidate,
    )
    monkeypatch.setattr(
        pipeline_service.workflow_service,
        "run_consistency_check_workflow",
        fail_workflow,
    )

    result = run_async(
        pipeline_service.run_fact_extraction_consistency_pipeline(
            object(),
            project_id=uuid.uuid4(),
            extraction_run_id=uuid.uuid4(),
            plan=object(),
            prompt=object(),
            llm_client=object(),
            provider="agent1-provider",
            requested_model="agent1-model",
            worker_token=uuid.uuid4(),
            consistency_config=CONFIG,
            consistency_prompt=object(),
            consistency_provider="agent2-provider",
            consistency_requested_model="agent2-model",
        )
    )

    assert result == FactExtractionConsistencyPipelineResult(
        extraction_orchestration_id=orchestration_result.orchestration_id,
        extraction_status=FactExtractionOrchestrationStatus.FAILED,
        grouping_application_id=None,
        consistency_application_id=None,
        consistency_check_application_id=None,
        consistency_plan_manifest_hash=None,
        consistency_execution_result_manifest_hash=None,
        assessment_count=None,
        consistency_created_new=None,
        skipped_reason="extraction_failed",
    )


def test_run_fact_extraction_consistency_pipeline_reused_completed_is_fully_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id = uuid.uuid4()
    orchestration_id = uuid.uuid4()
    grouping_application_id = uuid.uuid4()
    consistency_application_id = uuid.uuid4()
    consistency_check_application_id = uuid.uuid4()
    extract_llm_count = 0
    workflow_llm_count = 0
    pipeline_results = []

    async def fake_extract(*args, **kwargs):
        nonlocal extract_llm_count
        if not pipeline_results:
            extract_llm_count += 1
        return _orchestration_result(
            orchestration_id=orchestration_id,
            status=FactExtractionOrchestrationStatus.COMPLETED,
        )

    async def fake_group(*args, **kwargs):
        return _grouping_result(grouping_application_id=grouping_application_id)

    async def fake_candidate(*args, **kwargs):
        return _candidate_result(
            consistency_application_id=consistency_application_id,
            duplicate_grouping_application_id=grouping_application_id,
        )

    async def fake_workflow(*args, **kwargs):
        nonlocal workflow_llm_count
        if not pipeline_results:
            workflow_llm_count += 1
            return _workflow_result(
                project_id=project_id,
                consistency_application_id=consistency_application_id,
                consistency_check_application_id=consistency_check_application_id,
                created_new=True,
            )
        return _workflow_result(
            project_id=project_id,
            consistency_application_id=consistency_application_id,
            consistency_check_application_id=consistency_check_application_id,
            created_new=False,
        )

    monkeypatch.setattr(
        pipeline_service.orchestration_service,
        "execute_fact_extraction_orchestration",
        fake_extract,
    )
    monkeypatch.setattr(
        pipeline_service.duplicate_grouping_service,
        "ensure_cross_batch_duplicate_grouping",
        fake_group,
    )
    monkeypatch.setattr(
        pipeline_service.duplicate_grouping_service,
        "ensure_cross_batch_multi_value_consistency_candidates",
        fake_candidate,
    )
    monkeypatch.setattr(
        pipeline_service.workflow_service,
        "run_consistency_check_workflow",
        fake_workflow,
    )

    for _ in range(2):
        pipeline_results.append(
            run_async(
                pipeline_service.run_fact_extraction_consistency_pipeline(
                    object(),
                    project_id=project_id,
                    extraction_run_id=uuid.uuid4(),
                    plan=object(),
                    prompt=object(),
                    llm_client=object(),
                    provider="agent1-provider",
                    requested_model="agent1-model",
                    worker_token=uuid.uuid4(),
                    consistency_config=CONFIG,
                    consistency_prompt=object(),
                    consistency_provider="agent2-provider",
                    consistency_requested_model="agent2-model",
                )
            )
        )

    assert extract_llm_count == 1
    assert workflow_llm_count == 1
    assert (
        pipeline_results[0].consistency_check_application_id
        == pipeline_results[1].consistency_check_application_id
    )
    assert pipeline_results[0].consistency_created_new is True
    assert pipeline_results[1].consistency_created_new is False


def test_run_fact_extraction_consistency_pipeline_zero_candidate_keeps_workflow_and_zero_assessment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id = uuid.uuid4()
    grouping_result = _grouping_result()
    candidate_result = _candidate_result(
        consistency_application_id=uuid.uuid4(),
        duplicate_grouping_application_id=grouping_result.grouping_application_id,
    )
    workflow_result = _workflow_result(
        project_id=project_id,
        consistency_application_id=candidate_result.consistency_application_id,
        assessment_count=0,
    )
    workflow_call_count = 0

    async def fake_extract(*args, **kwargs):
        return _orchestration_result(status=FactExtractionOrchestrationStatus.COMPLETED)

    async def fake_group(*args, **kwargs):
        return grouping_result

    async def fake_candidate(*args, **kwargs):
        return candidate_result

    async def fake_workflow(*args, **kwargs):
        nonlocal workflow_call_count
        workflow_call_count += 1
        return workflow_result

    monkeypatch.setattr(
        pipeline_service.orchestration_service,
        "execute_fact_extraction_orchestration",
        fake_extract,
    )
    monkeypatch.setattr(
        pipeline_service.duplicate_grouping_service,
        "ensure_cross_batch_duplicate_grouping",
        fake_group,
    )
    monkeypatch.setattr(
        pipeline_service.duplicate_grouping_service,
        "ensure_cross_batch_multi_value_consistency_candidates",
        fake_candidate,
    )
    monkeypatch.setattr(
        pipeline_service.workflow_service,
        "run_consistency_check_workflow",
        fake_workflow,
    )

    result = run_async(
        pipeline_service.run_fact_extraction_consistency_pipeline(
            object(),
            project_id=project_id,
            extraction_run_id=uuid.uuid4(),
            plan=object(),
            prompt=object(),
            llm_client=object(),
            provider="agent1-provider",
            requested_model="agent1-model",
            worker_token=uuid.uuid4(),
            consistency_config=CONFIG,
            consistency_prompt=object(),
            consistency_provider="agent2-provider",
            consistency_requested_model="agent2-model",
        )
    )

    assert workflow_call_count == 1
    assert result.assessment_count == 0


@pytest.mark.parametrize("stage", ["group", "candidate", "workflow"])
def test_run_fact_extraction_consistency_pipeline_stops_after_stage_failure(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    orchestration_result = _orchestration_result(status=FactExtractionOrchestrationStatus.COMPLETED)
    grouping_result = _grouping_result()
    candidate_result = _candidate_result(
        duplicate_grouping_application_id=grouping_result.grouping_application_id,
    )
    calls: list[str] = []

    async def fake_extract(*args, **kwargs):
        calls.append("extract")
        return orchestration_result

    async def fake_group(*args, **kwargs):
        calls.append("group")
        if stage == "group":
            raise duplicate_grouping_service.CrossBatchDuplicateGroupingStateError(
                "pipeline_group_failed"
            )
        return grouping_result

    async def fake_candidate(*args, **kwargs):
        calls.append("candidate")
        if stage == "candidate":
            raise duplicate_grouping_service.FactValueConsistencyCandidateStateError(
                "pipeline_candidate_failed"
            )
        return candidate_result

    async def fake_workflow(*args, **kwargs):
        calls.append("workflow")
        if stage == "workflow":
            raise workflow_service.ConsistencyCheckWorkflowStateError(
                "pipeline_workflow_failed"
            )
        raise AssertionError("workflow should not succeed in this test")

    monkeypatch.setattr(
        pipeline_service.orchestration_service,
        "execute_fact_extraction_orchestration",
        fake_extract,
    )
    monkeypatch.setattr(
        pipeline_service.duplicate_grouping_service,
        "ensure_cross_batch_duplicate_grouping",
        fake_group,
    )
    monkeypatch.setattr(
        pipeline_service.duplicate_grouping_service,
        "ensure_cross_batch_multi_value_consistency_candidates",
        fake_candidate,
    )
    monkeypatch.setattr(
        pipeline_service.workflow_service,
        "run_consistency_check_workflow",
        fake_workflow,
    )

    expected_error = {
        "group": duplicate_grouping_service.CrossBatchDuplicateGroupingStateError,
        "candidate": duplicate_grouping_service.FactValueConsistencyCandidateStateError,
        "workflow": workflow_service.ConsistencyCheckWorkflowStateError,
    }[stage]
    expected_match = {
        "group": "pipeline_group_failed",
        "candidate": "pipeline_candidate_failed",
        "workflow": "pipeline_workflow_failed",
    }[stage]

    with pytest.raises(expected_error, match=expected_match):
        run_async(
            pipeline_service.run_fact_extraction_consistency_pipeline(
                object(),
                project_id=uuid.uuid4(),
                extraction_run_id=uuid.uuid4(),
                plan=object(),
                prompt=object(),
                llm_client=object(),
                provider="agent1-provider",
                requested_model="agent1-model",
                worker_token=uuid.uuid4(),
                consistency_config=CONFIG,
                consistency_prompt=object(),
                consistency_provider="agent2-provider",
                consistency_requested_model="agent2-model",
            )
        )

    expected_calls = {
        "group": ["extract", "group"],
        "candidate": ["extract", "group", "candidate"],
        "workflow": ["extract", "group", "candidate", "workflow"],
    }[stage]
    assert calls == expected_calls


@pytest.mark.parametrize("stage", ["extract", "group", "candidate", "workflow"])
def test_run_fact_extraction_consistency_pipeline_propagates_cancelled_error(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    orchestration_result = _orchestration_result(status=FactExtractionOrchestrationStatus.COMPLETED)
    grouping_result = _grouping_result()
    candidate_result = _candidate_result(
        duplicate_grouping_application_id=grouping_result.grouping_application_id,
    )

    async def fake_extract(*args, **kwargs):
        if stage == "extract":
            raise asyncio.CancelledError()
        return orchestration_result

    async def fake_group(*args, **kwargs):
        if stage == "group":
            raise asyncio.CancelledError()
        return grouping_result

    async def fake_candidate(*args, **kwargs):
        if stage == "candidate":
            raise asyncio.CancelledError()
        return candidate_result

    async def fake_workflow(*args, **kwargs):
        if stage == "workflow":
            raise asyncio.CancelledError()
        return _workflow_result(
            project_id=uuid.uuid4(),
            consistency_application_id=candidate_result.consistency_application_id,
        )

    monkeypatch.setattr(
        pipeline_service.orchestration_service,
        "execute_fact_extraction_orchestration",
        fake_extract,
    )
    monkeypatch.setattr(
        pipeline_service.duplicate_grouping_service,
        "ensure_cross_batch_duplicate_grouping",
        fake_group,
    )
    monkeypatch.setattr(
        pipeline_service.duplicate_grouping_service,
        "ensure_cross_batch_multi_value_consistency_candidates",
        fake_candidate,
    )
    monkeypatch.setattr(
        pipeline_service.workflow_service,
        "run_consistency_check_workflow",
        fake_workflow,
    )

    with pytest.raises(asyncio.CancelledError):
        run_async(
            pipeline_service.run_fact_extraction_consistency_pipeline(
                object(),
                project_id=uuid.uuid4(),
                extraction_run_id=uuid.uuid4(),
                plan=object(),
                prompt=object(),
                llm_client=object(),
                provider="agent1-provider",
                requested_model="agent1-model",
                worker_token=uuid.uuid4(),
                consistency_config=CONFIG,
                consistency_prompt=object(),
                consistency_provider="agent2-provider",
                consistency_requested_model="agent2-model",
            )
        )


def test_run_fact_extraction_consistency_pipeline_result_uses_authoritative_subresults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id = uuid.uuid4()
    orchestration_result = _orchestration_result(
        orchestration_id=uuid.uuid4(),
        status=FactExtractionOrchestrationStatus.COMPLETED,
    )
    grouping_result = _grouping_result(grouping_application_id=uuid.uuid4())
    candidate_result = _candidate_result(
        consistency_application_id=uuid.uuid4(),
        duplicate_grouping_application_id=grouping_result.grouping_application_id,
    )
    workflow_result = _workflow_result(
        project_id=project_id,
        consistency_application_id=candidate_result.consistency_application_id,
        consistency_check_application_id=uuid.uuid4(),
        assessment_count=7,
        created_new=False,
    )

    async def fake_extract(*args, **kwargs):
        return orchestration_result

    async def fake_group(*args, **kwargs):
        return grouping_result

    async def fake_candidate(*args, **kwargs):
        return candidate_result

    async def fake_workflow(*args, **kwargs):
        return workflow_result

    monkeypatch.setattr(
        pipeline_service.orchestration_service,
        "execute_fact_extraction_orchestration",
        fake_extract,
    )
    monkeypatch.setattr(
        pipeline_service.duplicate_grouping_service,
        "ensure_cross_batch_duplicate_grouping",
        fake_group,
    )
    monkeypatch.setattr(
        pipeline_service.duplicate_grouping_service,
        "ensure_cross_batch_multi_value_consistency_candidates",
        fake_candidate,
    )
    monkeypatch.setattr(
        pipeline_service.workflow_service,
        "run_consistency_check_workflow",
        fake_workflow,
    )

    result = run_async(
        pipeline_service.run_fact_extraction_consistency_pipeline(
            object(),
            project_id=project_id,
            extraction_run_id=uuid.uuid4(),
            plan=object(),
            prompt=object(),
            llm_client=object(),
            provider="caller-agent1-provider",
            requested_model="caller-agent1-model",
            worker_token=uuid.uuid4(),
            consistency_config=CONFIG,
            consistency_prompt=object(),
            consistency_provider="caller-agent2-provider",
            consistency_requested_model="caller-agent2-model",
        )
    )

    assert result.extraction_orchestration_id == orchestration_result.orchestration_id
    assert result.grouping_application_id == grouping_result.grouping_application_id
    assert result.consistency_application_id == candidate_result.consistency_application_id
    assert (
        result.consistency_check_application_id
        == workflow_result.consistency_check_application_id
    )
    assert result.consistency_plan_manifest_hash == workflow_result.plan_manifest_hash
    assert (
        result.consistency_execution_result_manifest_hash
        == workflow_result.execution_result_manifest_hash
    )
    assert result.assessment_count == workflow_result.assessment_count
    assert result.consistency_created_new is workflow_result.created_new


def test_run_fact_extraction_consistency_pipeline_does_not_leak_sensitive_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "SENSITIVE_PIPELINE_SENTINEL"

    async def fake_extract(*args, **kwargs):
        return _orchestration_result(status=FactExtractionOrchestrationStatus.COMPLETED)

    async def fake_group(*args, **kwargs):
        raise duplicate_grouping_service.CrossBatchDuplicateGroupingStateError(
            "cross_batch_duplicate_grouping_not_ready"
        )

    async def fail_candidate(*args, **kwargs):
        raise AssertionError(sentinel)

    async def fail_workflow(*args, **kwargs):
        raise AssertionError(sentinel)

    monkeypatch.setattr(
        pipeline_service.orchestration_service,
        "execute_fact_extraction_orchestration",
        fake_extract,
    )
    monkeypatch.setattr(
        pipeline_service.duplicate_grouping_service,
        "ensure_cross_batch_duplicate_grouping",
        fake_group,
    )
    monkeypatch.setattr(
        pipeline_service.duplicate_grouping_service,
        "ensure_cross_batch_multi_value_consistency_candidates",
        fail_candidate,
    )
    monkeypatch.setattr(
        pipeline_service.workflow_service,
        "run_consistency_check_workflow",
        fail_workflow,
    )

    with pytest.raises(
        duplicate_grouping_service.CrossBatchDuplicateGroupingStateError,
        match="cross_batch_duplicate_grouping_not_ready",
    ) as exc_info:
        run_async(
            pipeline_service.run_fact_extraction_consistency_pipeline(
                object(),
                project_id=uuid.uuid4(),
                extraction_run_id=uuid.uuid4(),
                plan=object(),
                prompt=object(),
                llm_client=object(),
                provider="agent1-provider",
                requested_model="agent1-model",
                worker_token=uuid.uuid4(),
                consistency_config=CONFIG,
                consistency_prompt=object(),
                consistency_provider="agent2-provider",
                consistency_requested_model="agent2-model",
            )
        )

    assert sentinel not in str(exc_info.value)
