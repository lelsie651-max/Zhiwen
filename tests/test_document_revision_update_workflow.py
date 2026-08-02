from __future__ import annotations

import asyncio
from dataclasses import replace
import uuid

import pytest

from app.schemas.consistency_check import ConsistencyCheckPlannerConfig
from app.schemas.consistency_pipeline import FactExtractionConsistencyPipelineResult
from app.schemas.document_revision_fact_diff import (
    DocumentRevisionFactDiffFactSnapshot,
    DocumentRevisionFactDiffValueGroup,
)
from app.schemas.document_revision_update_impact import (
    DocumentRevisionUpdateImpact,
    DocumentRevisionUpdateImpactItem,
)
from app.schemas.document_revision_update_workflow import (
    DocumentRevisionUpdateWorkflowResult,
)
from app.schemas.fact_extraction_orchestration import (
    FactExtractionOrchestrationBatchResult,
    FactExtractionOrchestrationBatchStatus,
    FactExtractionOrchestrationResult,
    FactExtractionOrchestrationStatus,
)
from app.services import document_revision_update_workflow as workflow_service


CONFIG = ConsistencyCheckPlannerConfig(
    max_candidates_per_batch=8,
    max_evidence_characters_per_batch=500,
)


def run_async(awaitable):
    return asyncio.run(awaitable)


class ForbiddenSessionFactory:
    def __call__(self):
        raise AssertionError("workflow layer must not open sessions")


def _pipeline_result(
    *,
    orchestration_id: uuid.UUID,
    extraction_status: FactExtractionOrchestrationStatus,
    grouping_application_id: uuid.UUID | None = None,
    consistency_application_id: uuid.UUID | None = None,
    consistency_check_application_id: uuid.UUID | None = None,
    consistency_plan_manifest_hash: str | None = None,
    consistency_execution_result_manifest_hash: str | None = None,
    assessment_count: int | None = None,
    consistency_created_new: bool | None = None,
    skipped_reason: str | None = None,
) -> FactExtractionConsistencyPipelineResult:
    return FactExtractionConsistencyPipelineResult(
        extraction_orchestration_id=orchestration_id,
        extraction_status=extraction_status,
        grouping_application_id=grouping_application_id,
        consistency_application_id=consistency_application_id,
        consistency_check_application_id=consistency_check_application_id,
        consistency_plan_manifest_hash=consistency_plan_manifest_hash,
        consistency_execution_result_manifest_hash=consistency_execution_result_manifest_hash,
        assessment_count=assessment_count,
        consistency_created_new=consistency_created_new,
        skipped_reason=skipped_reason,
    )


def _terminal_result(
    *,
    orchestration_id: uuid.UUID,
    status: FactExtractionOrchestrationStatus,
) -> FactExtractionOrchestrationResult:
    return FactExtractionOrchestrationResult(
        orchestration_id=orchestration_id,
        attempt_no=1,
        request_hash="a" * 64,
        plan_hash="b" * 64,
        status=status,
        batch_count=1,
        completed_batch_count=1 if status == FactExtractionOrchestrationStatus.COMPLETED else 0,
        failed_batch_count=1 if status == FactExtractionOrchestrationStatus.FAILED else 0,
        proposal_count=0,
        created_count=0,
        reused_count=0,
        withheld_count=0,
        batches=(
            FactExtractionOrchestrationBatchResult(
                batch_index=0,
                batch_plan_hash="c" * 64,
                status=(
                    FactExtractionOrchestrationBatchStatus.COMPLETED
                    if status != FactExtractionOrchestrationStatus.FAILED
                    else FactExtractionOrchestrationBatchStatus.FAILED
                ),
                attempt_count=1,
                input_batch_id=None,
                inference_run_id=None,
                application_id=None,
                proposal_count=0,
                created_count=0,
                reused_count=0,
                withheld_count=0,
                failure_code=None,
            ),
        ),
    )


def _impact_result(
    *,
    project_id: uuid.UUID,
    document_id: uuid.UUID,
    base_revision_id: uuid.UUID,
    target_revision_id: uuid.UUID,
    base_extraction_run_id: uuid.UUID,
    target_extraction_run_id: uuid.UUID,
    base_orchestration_id: uuid.UUID,
    target_orchestration_id: uuid.UUID,
    base_consistency_check_application_id: uuid.UUID,
    comparison_quality: str = "complete",
    impact_manifest_hash: str = "f" * 64,
    fact_count: int = 3,
    review_required_count: int = 2,
) -> DocumentRevisionUpdateImpact:
    return DocumentRevisionUpdateImpact(
        project_id=project_id,
        document_id=document_id,
        base_revision_id=base_revision_id,
        target_revision_id=target_revision_id,
        base_extraction_run_id=base_extraction_run_id,
        target_extraction_run_id=target_extraction_run_id,
        base_orchestration_id=base_orchestration_id,
        target_orchestration_id=target_orchestration_id,
        base_consistency_check_application_id=base_consistency_check_application_id,
        base_source_consistency_application_id=uuid.uuid4(),
        comparison_quality=comparison_quality,
        block_diff_manifest_hash="a" * 64,
        fact_diff_manifest_hash="b" * 64,
        base_consistency_result_manifest_hash="c" * 64,
        impact_algorithm_name="document_revision_update_impact",
        impact_algorithm_version="1.0.0",
        fact_count=fact_count,
        review_required_count=review_required_count,
        unchanged_resolved_count=1,
        unchanged_no_review_context_count=0,
        unchanged_unresolved_count=0,
        modified_count=1,
        added_count=1,
        removed_count=0,
        items=(),
        impact_manifest_hash=impact_manifest_hash,
    )


def _workflow_kwargs() -> dict[str, object]:
    return {
        "project_id": uuid.uuid4(),
        "document_id": uuid.uuid4(),
        "base_revision_id": uuid.uuid4(),
        "target_revision_id": uuid.uuid4(),
        "base_extraction_run_id": uuid.uuid4(),
        "target_extraction_run_id": uuid.uuid4(),
        "base_orchestration_id": uuid.uuid4(),
        "base_consistency_check_application_id": uuid.uuid4(),
        "target_fact_extraction_plan": object(),
        "fact_extraction_prompt": object(),
        "llm_client": object(),
        "fact_extraction_provider": "agent1-provider",
        "fact_extraction_requested_model": "agent1-model",
        "worker_token": uuid.uuid4(),
        "consistency_config": CONFIG,
        "consistency_prompt": object(),
        "consistency_provider": "agent2-provider",
        "consistency_requested_model": "agent2-model",
    }


def _install_terminal_auth(
    monkeypatch: pytest.MonkeyPatch,
    *,
    terminal_result: FactExtractionOrchestrationResult | Exception,
) -> None:
    async def fake_terminal_auth(_session_factory, **_kwargs):
        if isinstance(terminal_result, Exception):
            raise terminal_result
        return terminal_result

    monkeypatch.setattr(
        workflow_service.orchestration_service,
        "authenticate_terminal_fact_extraction_orchestration",
        fake_terminal_auth,
    )


def _install_impact_auth(
    monkeypatch: pytest.MonkeyPatch,
    *,
    authenticated_impact: DocumentRevisionUpdateImpact | Exception | None = None,
) -> None:
    def fake_authenticate(impact):
        if isinstance(authenticated_impact, Exception):
            raise authenticated_impact
        if authenticated_impact is not None:
            return authenticated_impact
        return impact

    monkeypatch.setattr(
        workflow_service.impact_service,
        "authenticate_document_revision_update_impact_projection",
        fake_authenticate,
    )


def _fact_snapshot(seed: str) -> DocumentRevisionFactDiffFactSnapshot:
    fact_id = uuid.uuid5(uuid.NAMESPACE_URL, f"fact-{seed}")
    return DocumentRevisionFactDiffFactSnapshot(
        fact_id=fact_id,
        identity_hash="a" * 64,
        subject_kind="subject",
        subject_key=f"subject-{seed}",
        predicate_key=f"predicate-{seed}",
        scope_key=None,
        subject_entity_id=None,
    )


def _value_group(seed: str, *fact_value_ids: uuid.UUID) -> DocumentRevisionFactDiffValueGroup:
    return DocumentRevisionFactDiffValueGroup(
        semantic_key_hash="b" * 64,
        value_type="string",
        value_json=f"value-{seed}",
        referenced_entity_id=None,
        fact_value_ids=fact_value_ids,
        evidences=(),
    )


def test_run_document_revision_update_workflow_completed_runs_pipeline_then_impact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _workflow_kwargs()
    target_orchestration_id = uuid.uuid4()
    grouping_application_id = uuid.uuid4()
    consistency_application_id = uuid.uuid4()
    consistency_check_application_id = uuid.uuid4()
    pipeline_result = _pipeline_result(
        orchestration_id=target_orchestration_id,
        extraction_status=FactExtractionOrchestrationStatus.COMPLETED,
        grouping_application_id=grouping_application_id,
        consistency_application_id=consistency_application_id,
        consistency_check_application_id=consistency_check_application_id,
        consistency_plan_manifest_hash="1" * 64,
        consistency_execution_result_manifest_hash="2" * 64,
        assessment_count=7,
        consistency_created_new=True,
        skipped_reason=None,
    )
    impact_result = _impact_result(
        project_id=kwargs["project_id"],
        document_id=kwargs["document_id"],
        base_revision_id=kwargs["base_revision_id"],
        target_revision_id=kwargs["target_revision_id"],
        base_extraction_run_id=kwargs["base_extraction_run_id"],
        target_extraction_run_id=kwargs["target_extraction_run_id"],
        base_orchestration_id=kwargs["base_orchestration_id"],
        target_orchestration_id=target_orchestration_id,
        base_consistency_check_application_id=kwargs["base_consistency_check_application_id"],
    )
    call_order: list[str] = []

    async def fake_pipeline(session_factory, **pipeline_kwargs):
        assert session_factory is session_factory_sentinel
        assert pipeline_kwargs["project_id"] == kwargs["project_id"]
        assert pipeline_kwargs["extraction_run_id"] == kwargs["target_extraction_run_id"]
        assert pipeline_kwargs["plan"] is kwargs["target_fact_extraction_plan"]
        assert pipeline_kwargs["prompt"] is kwargs["fact_extraction_prompt"]
        assert pipeline_kwargs["llm_client"] is kwargs["llm_client"]
        assert pipeline_kwargs["provider"] == kwargs["fact_extraction_provider"]
        assert (
            pipeline_kwargs["requested_model"]
            == kwargs["fact_extraction_requested_model"]
        )
        assert pipeline_kwargs["worker_token"] == kwargs["worker_token"]
        call_order.append("pipeline")
        return pipeline_result

    async def fake_impact(session_factory, **impact_kwargs):
        assert session_factory is session_factory_sentinel
        assert impact_kwargs["project_id"] == kwargs["project_id"]
        assert impact_kwargs["document_id"] == kwargs["document_id"]
        assert impact_kwargs["base_revision_id"] == kwargs["base_revision_id"]
        assert impact_kwargs["target_revision_id"] == kwargs["target_revision_id"]
        assert impact_kwargs["base_extraction_run_id"] == kwargs["base_extraction_run_id"]
        assert impact_kwargs["target_extraction_run_id"] == kwargs["target_extraction_run_id"]
        assert impact_kwargs["base_orchestration_id"] == kwargs["base_orchestration_id"]
        assert impact_kwargs["target_orchestration_id"] == target_orchestration_id
        assert (
            impact_kwargs["base_consistency_check_application_id"]
            == kwargs["base_consistency_check_application_id"]
        )
        call_order.append("impact")
        return impact_result

    session_factory_sentinel = ForbiddenSessionFactory()
    monkeypatch.setattr(
        workflow_service.pipeline_service,
        "run_fact_extraction_consistency_pipeline",
        fake_pipeline,
    )
    _install_terminal_auth(
        monkeypatch,
        terminal_result=_terminal_result(
            orchestration_id=target_orchestration_id,
            status=FactExtractionOrchestrationStatus.COMPLETED,
        ),
    )
    monkeypatch.setattr(
        workflow_service.impact_service,
        "get_document_revision_update_impact",
        fake_impact,
    )
    _install_impact_auth(monkeypatch)

    result = run_async(
        workflow_service.run_document_revision_update_workflow(
            session_factory_sentinel,
            **kwargs,
        )
    )

    assert call_order == ["pipeline", "impact"]
    assert result == DocumentRevisionUpdateWorkflowResult(
        project_id=kwargs["project_id"],
        document_id=kwargs["document_id"],
        base_revision_id=kwargs["base_revision_id"],
        target_revision_id=kwargs["target_revision_id"],
        base_extraction_run_id=kwargs["base_extraction_run_id"],
        target_extraction_run_id=kwargs["target_extraction_run_id"],
        base_orchestration_id=kwargs["base_orchestration_id"],
        target_orchestration_id=target_orchestration_id,
        target_extraction_status=FactExtractionOrchestrationStatus.COMPLETED,
        target_grouping_application_id=grouping_application_id,
        target_consistency_application_id=consistency_application_id,
        target_consistency_check_application_id=consistency_check_application_id,
        target_consistency_plan_manifest_hash="1" * 64,
        target_consistency_execution_manifest_hash="2" * 64,
        target_assessment_count=7,
        target_consistency_created_new=True,
        comparison_quality="complete",
        impact_manifest_hash="f" * 64,
        fact_count=3,
        review_required_count=2,
        skipped_reason=None,
    )
    assert not hasattr(result, "base_current_decision_id")


def test_run_document_revision_update_workflow_partial_runs_impact_and_propagates_quality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _workflow_kwargs()
    target_orchestration_id = uuid.uuid4()

    async def fake_pipeline(_session_factory, **_kwargs):
        return _pipeline_result(
            orchestration_id=target_orchestration_id,
            extraction_status=FactExtractionOrchestrationStatus.PARTIAL,
            grouping_application_id=uuid.uuid4(),
            consistency_application_id=uuid.uuid4(),
            consistency_check_application_id=uuid.uuid4(),
            consistency_plan_manifest_hash="3" * 64,
            consistency_execution_result_manifest_hash="4" * 64,
            assessment_count=0,
            consistency_created_new=False,
            skipped_reason=None,
        )

    async def fake_impact(_session_factory, **_kwargs):
        return _impact_result(
            project_id=kwargs["project_id"],
            document_id=kwargs["document_id"],
            base_revision_id=kwargs["base_revision_id"],
            target_revision_id=kwargs["target_revision_id"],
            base_extraction_run_id=kwargs["base_extraction_run_id"],
            target_extraction_run_id=kwargs["target_extraction_run_id"],
            base_orchestration_id=kwargs["base_orchestration_id"],
            target_orchestration_id=target_orchestration_id,
            base_consistency_check_application_id=kwargs["base_consistency_check_application_id"],
            comparison_quality="partial",
        )

    monkeypatch.setattr(
        workflow_service.pipeline_service,
        "run_fact_extraction_consistency_pipeline",
        fake_pipeline,
    )
    _install_terminal_auth(
        monkeypatch,
        terminal_result=_terminal_result(
            orchestration_id=target_orchestration_id,
            status=FactExtractionOrchestrationStatus.PARTIAL,
        ),
    )
    monkeypatch.setattr(
        workflow_service.impact_service,
        "get_document_revision_update_impact",
        fake_impact,
    )
    _install_impact_auth(monkeypatch)

    result = run_async(
        workflow_service.run_document_revision_update_workflow(
            ForbiddenSessionFactory(),
            **kwargs,
        )
    )

    assert result.target_extraction_status == FactExtractionOrchestrationStatus.PARTIAL
    assert result.comparison_quality == "partial"
    assert result.skipped_reason is None


def test_run_document_revision_update_workflow_failed_skips_impact_and_returns_skipped_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _workflow_kwargs()
    target_orchestration_id = uuid.uuid4()

    async def fake_pipeline(_session_factory, **_kwargs):
        return _pipeline_result(
            orchestration_id=target_orchestration_id,
            extraction_status=FactExtractionOrchestrationStatus.FAILED,
            skipped_reason="extraction_failed",
        )

    async def fail_impact(*args, **kwargs):
        raise AssertionError("impact should not be called")

    monkeypatch.setattr(
        workflow_service.pipeline_service,
        "run_fact_extraction_consistency_pipeline",
        fake_pipeline,
    )
    _install_terminal_auth(
        monkeypatch,
        terminal_result=_terminal_result(
            orchestration_id=target_orchestration_id,
            status=FactExtractionOrchestrationStatus.FAILED,
        ),
    )
    monkeypatch.setattr(
        workflow_service.impact_service,
        "get_document_revision_update_impact",
        fail_impact,
    )

    result = run_async(
        workflow_service.run_document_revision_update_workflow(
            ForbiddenSessionFactory(),
            **kwargs,
        )
    )

    assert result.target_orchestration_id == target_orchestration_id
    assert result.target_extraction_status == FactExtractionOrchestrationStatus.FAILED
    assert result.target_grouping_application_id is None
    assert result.target_consistency_application_id is None
    assert result.target_consistency_check_application_id is None
    assert result.comparison_quality is None
    assert result.impact_manifest_hash is None
    assert result.fact_count is None
    assert result.review_required_count is None
    assert result.skipped_reason == "target_extraction_failed"


def test_run_document_revision_update_workflow_rejects_failed_pipeline_when_terminal_orchestration_source_mismatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _workflow_kwargs()
    target_orchestration_id = uuid.uuid4()

    async def fake_pipeline(_session_factory, **_kwargs):
        return _pipeline_result(
            orchestration_id=target_orchestration_id,
            extraction_status=FactExtractionOrchestrationStatus.FAILED,
            skipped_reason="extraction_failed",
        )

    monkeypatch.setattr(
        workflow_service.pipeline_service,
        "run_fact_extraction_consistency_pipeline",
        fake_pipeline,
    )
    _install_terminal_auth(
        monkeypatch,
        terminal_result=workflow_service.orchestration_service.FactExtractionOrchestrationStateError(
            "orchestration project mismatch"
        ),
    )

    with pytest.raises(
        workflow_service.orchestration_service.FactExtractionOrchestrationStateError,
        match="orchestration project mismatch",
    ):
        run_async(
            workflow_service.run_document_revision_update_workflow(
                ForbiddenSessionFactory(),
                **kwargs,
            )
        )


def test_run_document_revision_update_workflow_rejects_pipeline_and_terminal_status_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _workflow_kwargs()
    target_orchestration_id = uuid.uuid4()

    async def fake_pipeline(_session_factory, **_kwargs):
        return _pipeline_result(
            orchestration_id=target_orchestration_id,
            extraction_status=FactExtractionOrchestrationStatus.PARTIAL,
            grouping_application_id=uuid.uuid4(),
            consistency_application_id=uuid.uuid4(),
            consistency_check_application_id=uuid.uuid4(),
            consistency_plan_manifest_hash="1" * 64,
            consistency_execution_result_manifest_hash="2" * 64,
            assessment_count=1,
            consistency_created_new=True,
            skipped_reason=None,
        )

    monkeypatch.setattr(
        workflow_service.pipeline_service,
        "run_fact_extraction_consistency_pipeline",
        fake_pipeline,
    )
    _install_terminal_auth(
        monkeypatch,
        terminal_result=_terminal_result(
            orchestration_id=target_orchestration_id,
            status=FactExtractionOrchestrationStatus.COMPLETED,
        ),
    )

    with pytest.raises(
        workflow_service.DocumentRevisionUpdateWorkflowInvariantError,
        match="document_revision_update_workflow_terminal_orchestration_status_mismatch",
    ):
        run_async(
            workflow_service.run_document_revision_update_workflow(
                ForbiddenSessionFactory(),
                **kwargs,
            )
        )


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (
            "success_missing_grouping",
            "document_revision_update_workflow_target_grouping_application_id_invalid",
        ),
        (
            "success_bad_manifest",
            "document_revision_update_workflow_target_consistency_plan_manifest_hash_invalid",
        ),
        (
            "success_bad_assessment_count",
            "document_revision_update_workflow_target_assessment_count_invalid",
        ),
        (
            "failed_wrong_reason",
            "document_revision_update_workflow_pipeline_failed_shape_invalid",
        ),
        (
            "failed_non_null_field",
            "document_revision_update_workflow_pipeline_failed_shape_invalid",
        ),
    ],
)
def test_run_document_revision_update_workflow_rejects_pipeline_shape_drift(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_code: str,
) -> None:
    kwargs = _workflow_kwargs()
    pipeline_result = _pipeline_result(
        orchestration_id=uuid.uuid4(),
        extraction_status=FactExtractionOrchestrationStatus.COMPLETED,
        grouping_application_id=uuid.uuid4(),
        consistency_application_id=uuid.uuid4(),
        consistency_check_application_id=uuid.uuid4(),
        consistency_plan_manifest_hash="1" * 64,
        consistency_execution_result_manifest_hash="2" * 64,
        assessment_count=1,
        consistency_created_new=True,
        skipped_reason=None,
    )
    if mutation == "success_missing_grouping":
        pipeline_result = _pipeline_result(
            orchestration_id=pipeline_result.extraction_orchestration_id,
            extraction_status=pipeline_result.extraction_status,
            grouping_application_id=None,
            consistency_application_id=pipeline_result.consistency_application_id,
            consistency_check_application_id=pipeline_result.consistency_check_application_id,
            consistency_plan_manifest_hash=pipeline_result.consistency_plan_manifest_hash,
            consistency_execution_result_manifest_hash=(
                pipeline_result.consistency_execution_result_manifest_hash
            ),
            assessment_count=pipeline_result.assessment_count,
            consistency_created_new=pipeline_result.consistency_created_new,
            skipped_reason=None,
        )
    elif mutation == "success_bad_manifest":
        pipeline_result = _pipeline_result(
            orchestration_id=pipeline_result.extraction_orchestration_id,
            extraction_status=pipeline_result.extraction_status,
            grouping_application_id=pipeline_result.grouping_application_id,
            consistency_application_id=pipeline_result.consistency_application_id,
            consistency_check_application_id=pipeline_result.consistency_check_application_id,
            consistency_plan_manifest_hash="NOT_A_HASH",
            consistency_execution_result_manifest_hash=(
                pipeline_result.consistency_execution_result_manifest_hash
            ),
            assessment_count=pipeline_result.assessment_count,
            consistency_created_new=pipeline_result.consistency_created_new,
            skipped_reason=None,
        )
    elif mutation == "success_bad_assessment_count":
        pipeline_result = _pipeline_result(
            orchestration_id=pipeline_result.extraction_orchestration_id,
            extraction_status=pipeline_result.extraction_status,
            grouping_application_id=pipeline_result.grouping_application_id,
            consistency_application_id=pipeline_result.consistency_application_id,
            consistency_check_application_id=pipeline_result.consistency_check_application_id,
            consistency_plan_manifest_hash=pipeline_result.consistency_plan_manifest_hash,
            consistency_execution_result_manifest_hash=(
                pipeline_result.consistency_execution_result_manifest_hash
            ),
            assessment_count=True,
            consistency_created_new=pipeline_result.consistency_created_new,
            skipped_reason=None,
        )
    elif mutation == "failed_wrong_reason":
        pipeline_result = _pipeline_result(
            orchestration_id=uuid.uuid4(),
            extraction_status=FactExtractionOrchestrationStatus.FAILED,
            skipped_reason="bad_reason",
        )
    else:
        pipeline_result = _pipeline_result(
            orchestration_id=uuid.uuid4(),
            extraction_status=FactExtractionOrchestrationStatus.FAILED,
            grouping_application_id=uuid.uuid4(),
            skipped_reason="extraction_failed",
        )

    async def fake_pipeline(_session_factory, **_kwargs):
        return pipeline_result

    monkeypatch.setattr(
        workflow_service.pipeline_service,
        "run_fact_extraction_consistency_pipeline",
        fake_pipeline,
    )

    with pytest.raises(
        workflow_service.DocumentRevisionUpdateWorkflowError,
        match=expected_code,
    ):
        run_async(
            workflow_service.run_document_revision_update_workflow(
                ForbiddenSessionFactory(),
                **kwargs,
            )
        )


@pytest.mark.parametrize(
    ("mutation", "expected_code", "expected_exception"),
    [
        (
            "project",
            "document_revision_update_workflow_impact_source_mismatch",
            workflow_service.DocumentRevisionUpdateWorkflowInvariantError,
        ),
        (
            "target_orchestration",
            "document_revision_update_workflow_impact_source_mismatch",
            workflow_service.DocumentRevisionUpdateWorkflowInvariantError,
        ),
        (
            "comparison_quality",
            "document_revision_update_workflow_comparison_quality_invalid",
            workflow_service.impact_service.DocumentRevisionUpdateImpactInvariantError,
        ),
        (
            "impact_manifest_hash",
            "document_revision_update_workflow_impact_manifest_hash_invalid",
            workflow_service.impact_service.DocumentRevisionUpdateImpactInvariantError,
        ),
    ],
)
def test_run_document_revision_update_workflow_rejects_impact_shape_or_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_code: str,
    expected_exception: type[Exception],
) -> None:
    kwargs = _workflow_kwargs()
    target_orchestration_id = uuid.uuid4()
    pipeline_result = _pipeline_result(
        orchestration_id=target_orchestration_id,
        extraction_status=FactExtractionOrchestrationStatus.COMPLETED,
        grouping_application_id=uuid.uuid4(),
        consistency_application_id=uuid.uuid4(),
        consistency_check_application_id=uuid.uuid4(),
        consistency_plan_manifest_hash="1" * 64,
        consistency_execution_result_manifest_hash="2" * 64,
        assessment_count=1,
        consistency_created_new=True,
        skipped_reason=None,
    )
    impact_result = _impact_result(
        project_id=kwargs["project_id"],
        document_id=kwargs["document_id"],
        base_revision_id=kwargs["base_revision_id"],
        target_revision_id=kwargs["target_revision_id"],
        base_extraction_run_id=kwargs["base_extraction_run_id"],
        target_extraction_run_id=kwargs["target_extraction_run_id"],
        base_orchestration_id=kwargs["base_orchestration_id"],
        target_orchestration_id=target_orchestration_id,
        base_consistency_check_application_id=kwargs["base_consistency_check_application_id"],
    )
    if mutation == "project":
        impact_result = _impact_result(
            project_id=uuid.uuid4(),
            document_id=kwargs["document_id"],
            base_revision_id=kwargs["base_revision_id"],
            target_revision_id=kwargs["target_revision_id"],
            base_extraction_run_id=kwargs["base_extraction_run_id"],
            target_extraction_run_id=kwargs["target_extraction_run_id"],
            base_orchestration_id=kwargs["base_orchestration_id"],
            target_orchestration_id=target_orchestration_id,
            base_consistency_check_application_id=kwargs["base_consistency_check_application_id"],
        )
    elif mutation == "target_orchestration":
        impact_result = _impact_result(
            project_id=kwargs["project_id"],
            document_id=kwargs["document_id"],
            base_revision_id=kwargs["base_revision_id"],
            target_revision_id=kwargs["target_revision_id"],
            base_extraction_run_id=kwargs["base_extraction_run_id"],
            target_extraction_run_id=kwargs["target_extraction_run_id"],
            base_orchestration_id=kwargs["base_orchestration_id"],
            target_orchestration_id=uuid.uuid4(),
            base_consistency_check_application_id=kwargs["base_consistency_check_application_id"],
        )
    elif mutation == "comparison_quality":
        impact_result = _impact_result(
            project_id=kwargs["project_id"],
            document_id=kwargs["document_id"],
            base_revision_id=kwargs["base_revision_id"],
            target_revision_id=kwargs["target_revision_id"],
            base_extraction_run_id=kwargs["base_extraction_run_id"],
            target_extraction_run_id=kwargs["target_extraction_run_id"],
            base_orchestration_id=kwargs["base_orchestration_id"],
            target_orchestration_id=target_orchestration_id,
            base_consistency_check_application_id=kwargs["base_consistency_check_application_id"],
            comparison_quality="invalid",
        )
    else:
        impact_result = _impact_result(
            project_id=kwargs["project_id"],
            document_id=kwargs["document_id"],
            base_revision_id=kwargs["base_revision_id"],
            target_revision_id=kwargs["target_revision_id"],
            base_extraction_run_id=kwargs["base_extraction_run_id"],
            target_extraction_run_id=kwargs["target_extraction_run_id"],
            base_orchestration_id=kwargs["base_orchestration_id"],
            target_orchestration_id=target_orchestration_id,
            base_consistency_check_application_id=kwargs["base_consistency_check_application_id"],
            impact_manifest_hash="NOT_A_HASH",
        )

    async def fake_pipeline(_session_factory, **_kwargs):
        return pipeline_result

    async def fake_impact(_session_factory, **_kwargs):
        return impact_result

    monkeypatch.setattr(
        workflow_service.pipeline_service,
        "run_fact_extraction_consistency_pipeline",
        fake_pipeline,
    )
    _install_terminal_auth(
        monkeypatch,
        terminal_result=_terminal_result(
            orchestration_id=target_orchestration_id,
            status=FactExtractionOrchestrationStatus.COMPLETED,
        ),
    )
    monkeypatch.setattr(
        workflow_service.impact_service,
        "get_document_revision_update_impact",
        fake_impact,
    )
    if mutation in {"comparison_quality", "impact_manifest_hash"}:
        _install_impact_auth(
            monkeypatch,
            authenticated_impact=workflow_service.impact_service.DocumentRevisionUpdateImpactInvariantError(
                expected_code
            ),
        )
    else:
        _install_impact_auth(monkeypatch)

    with pytest.raises(expected_exception, match=expected_code):
        run_async(
            workflow_service.run_document_revision_update_workflow(
                ForbiddenSessionFactory(),
                **kwargs,
            )
        )


def test_run_document_revision_update_workflow_rejects_resigned_semantically_invalid_impact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _workflow_kwargs()
    target_orchestration_id = uuid.uuid4()
    base_fact = _fact_snapshot("workflow-base")
    target_fact = _fact_snapshot("workflow-base")
    base_fact_value_id = uuid.uuid4()
    invalid_impact = DocumentRevisionUpdateImpact(
        project_id=kwargs["project_id"],
        document_id=kwargs["document_id"],
        base_revision_id=kwargs["base_revision_id"],
        target_revision_id=kwargs["target_revision_id"],
        base_extraction_run_id=kwargs["base_extraction_run_id"],
        target_extraction_run_id=kwargs["target_extraction_run_id"],
        base_orchestration_id=kwargs["base_orchestration_id"],
        target_orchestration_id=target_orchestration_id,
        base_consistency_check_application_id=kwargs["base_consistency_check_application_id"],
        base_source_consistency_application_id=uuid.uuid4(),
        comparison_quality="complete",
        block_diff_manifest_hash="a" * 64,
        fact_diff_manifest_hash="b" * 64,
        base_consistency_result_manifest_hash="c" * 64,
        impact_algorithm_name="document_revision_update_impact",
        impact_algorithm_version="1.0.0",
        fact_count=1,
        review_required_count=0,
        unchanged_resolved_count=1,
        unchanged_no_review_context_count=0,
        unchanged_unresolved_count=0,
        modified_count=0,
        added_count=0,
        removed_count=0,
        items=(
            DocumentRevisionUpdateImpactItem(
                fact_id=base_fact.fact_id,
                fact_change_kind="unchanged",
                impact_kind="unchanged_resolved",
                requires_review=False,
                base_assessment_id=uuid.uuid4(),
                base_review_status="pending_review",
                base_resolution_status="pending_review",
                base_resolution_basis="none",
                base_current_decision_id=None,
                base_current_decision_kind=None,
                base_effective_fact_value_ids=(),
                base_fact=base_fact,
                base_value_groups=(
                    _value_group("workflow-base", base_fact_value_id),
                ),
                target_fact=target_fact,
                target_value_groups=(),
            ),
        ),
        impact_manifest_hash="",
    )
    invalid_impact = replace(
        invalid_impact,
        impact_manifest_hash=workflow_service.impact_service._build_manifest_hash(
            impact=invalid_impact
        ),
    )

    async def fake_pipeline(_session_factory, **_kwargs):
        return _pipeline_result(
            orchestration_id=target_orchestration_id,
            extraction_status=FactExtractionOrchestrationStatus.COMPLETED,
            grouping_application_id=uuid.uuid4(),
            consistency_application_id=uuid.uuid4(),
            consistency_check_application_id=uuid.uuid4(),
            consistency_plan_manifest_hash="1" * 64,
            consistency_execution_result_manifest_hash="2" * 64,
            assessment_count=1,
            consistency_created_new=False,
            skipped_reason=None,
        )

    async def fake_impact(_session_factory, **_kwargs):
        return invalid_impact

    monkeypatch.setattr(
        workflow_service.pipeline_service,
        "run_fact_extraction_consistency_pipeline",
        fake_pipeline,
    )
    _install_terminal_auth(
        monkeypatch,
        terminal_result=_terminal_result(
            orchestration_id=target_orchestration_id,
            status=FactExtractionOrchestrationStatus.COMPLETED,
        ),
    )
    monkeypatch.setattr(
        workflow_service.impact_service,
        "get_document_revision_update_impact",
        fake_impact,
    )

    with pytest.raises(
        workflow_service.impact_service.DocumentRevisionUpdateImpactInvariantError,
        match="document_revision_update_impact_base_review_context_invalid",
    ):
        run_async(
            workflow_service.run_document_revision_update_workflow(
                ForbiddenSessionFactory(),
                **kwargs,
            )
        )


def test_run_document_revision_update_workflow_repeated_calls_reuse_pipeline_and_ledgers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _workflow_kwargs()
    target_orchestration_id = uuid.uuid4()
    pipeline_llm_count = 0
    impact_call_count = 0

    async def fake_pipeline(_session_factory, **_kwargs):
        nonlocal pipeline_llm_count
        if pipeline_llm_count == 0:
            pipeline_llm_count += 1
            created_new = True
        else:
            created_new = False
        return _pipeline_result(
            orchestration_id=target_orchestration_id,
            extraction_status=FactExtractionOrchestrationStatus.COMPLETED,
            grouping_application_id=uuid.uuid4(),
            consistency_application_id=uuid.uuid4(),
            consistency_check_application_id=uuid.uuid4(),
            consistency_plan_manifest_hash="1" * 64,
            consistency_execution_result_manifest_hash="2" * 64,
            assessment_count=1,
            consistency_created_new=created_new,
            skipped_reason=None,
        )

    async def fake_impact(_session_factory, **_kwargs):
        nonlocal impact_call_count
        impact_call_count += 1
        return _impact_result(
            project_id=kwargs["project_id"],
            document_id=kwargs["document_id"],
            base_revision_id=kwargs["base_revision_id"],
            target_revision_id=kwargs["target_revision_id"],
            base_extraction_run_id=kwargs["base_extraction_run_id"],
            target_extraction_run_id=kwargs["target_extraction_run_id"],
            base_orchestration_id=kwargs["base_orchestration_id"],
            target_orchestration_id=target_orchestration_id,
            base_consistency_check_application_id=kwargs["base_consistency_check_application_id"],
        )

    monkeypatch.setattr(
        workflow_service.pipeline_service,
        "run_fact_extraction_consistency_pipeline",
        fake_pipeline,
    )
    _install_terminal_auth(
        monkeypatch,
        terminal_result=_terminal_result(
            orchestration_id=target_orchestration_id,
            status=FactExtractionOrchestrationStatus.COMPLETED,
        ),
    )
    monkeypatch.setattr(
        workflow_service.impact_service,
        "get_document_revision_update_impact",
        fake_impact,
    )
    _install_impact_auth(monkeypatch)

    first = run_async(
        workflow_service.run_document_revision_update_workflow(
            ForbiddenSessionFactory(),
            **kwargs,
        )
    )
    second = run_async(
        workflow_service.run_document_revision_update_workflow(
            ForbiddenSessionFactory(),
            **kwargs,
        )
    )

    assert pipeline_llm_count == 1
    assert impact_call_count == 2
    assert first.target_orchestration_id == second.target_orchestration_id
    assert first.target_consistency_created_new is True
    assert second.target_consistency_created_new is False


def test_run_document_revision_update_workflow_retries_after_impact_failure_without_repeating_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _workflow_kwargs()
    target_orchestration_id = uuid.uuid4()
    pipeline_llm_count = 0
    impact_attempts = 0

    async def fake_pipeline(_session_factory, **_kwargs):
        nonlocal pipeline_llm_count
        if pipeline_llm_count == 0:
            pipeline_llm_count += 1
            created_new = True
        else:
            created_new = False
        return _pipeline_result(
            orchestration_id=target_orchestration_id,
            extraction_status=FactExtractionOrchestrationStatus.COMPLETED,
            grouping_application_id=uuid.uuid4(),
            consistency_application_id=uuid.uuid4(),
            consistency_check_application_id=uuid.uuid4(),
            consistency_plan_manifest_hash="1" * 64,
            consistency_execution_result_manifest_hash="2" * 64,
            assessment_count=1,
            consistency_created_new=created_new,
            skipped_reason=None,
        )

    async def fake_impact(_session_factory, **_kwargs):
        nonlocal impact_attempts
        impact_attempts += 1
        if impact_attempts == 1:
            raise workflow_service.DocumentRevisionUpdateWorkflowStateError(
                "impact_retry_me"
            )
        return _impact_result(
            project_id=kwargs["project_id"],
            document_id=kwargs["document_id"],
            base_revision_id=kwargs["base_revision_id"],
            target_revision_id=kwargs["target_revision_id"],
            base_extraction_run_id=kwargs["base_extraction_run_id"],
            target_extraction_run_id=kwargs["target_extraction_run_id"],
            base_orchestration_id=kwargs["base_orchestration_id"],
            target_orchestration_id=target_orchestration_id,
            base_consistency_check_application_id=kwargs["base_consistency_check_application_id"],
        )

    monkeypatch.setattr(
        workflow_service.pipeline_service,
        "run_fact_extraction_consistency_pipeline",
        fake_pipeline,
    )
    _install_terminal_auth(
        monkeypatch,
        terminal_result=_terminal_result(
            orchestration_id=target_orchestration_id,
            status=FactExtractionOrchestrationStatus.COMPLETED,
        ),
    )
    monkeypatch.setattr(
        workflow_service.impact_service,
        "get_document_revision_update_impact",
        fake_impact,
    )
    _install_impact_auth(monkeypatch)

    with pytest.raises(
        workflow_service.DocumentRevisionUpdateWorkflowStateError,
        match="impact_retry_me",
    ):
        run_async(
            workflow_service.run_document_revision_update_workflow(
                ForbiddenSessionFactory(),
                **kwargs,
            )
        )

    recovered = run_async(
        workflow_service.run_document_revision_update_workflow(
            ForbiddenSessionFactory(),
            **kwargs,
        )
    )

    assert pipeline_llm_count == 1
    assert impact_attempts == 2
    assert recovered.target_consistency_created_new is False


@pytest.mark.parametrize("stage", ["pipeline", "impact"])
def test_run_document_revision_update_workflow_propagates_cancelled_error(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    kwargs = _workflow_kwargs()
    target_orchestration_id = uuid.uuid4()

    async def fake_pipeline(_session_factory, **_kwargs):
        if stage == "pipeline":
            raise asyncio.CancelledError()
        return _pipeline_result(
            orchestration_id=target_orchestration_id,
            extraction_status=FactExtractionOrchestrationStatus.COMPLETED,
            grouping_application_id=uuid.uuid4(),
            consistency_application_id=uuid.uuid4(),
            consistency_check_application_id=uuid.uuid4(),
            consistency_plan_manifest_hash="1" * 64,
            consistency_execution_result_manifest_hash="2" * 64,
            assessment_count=1,
            consistency_created_new=True,
            skipped_reason=None,
        )

    async def fake_impact(_session_factory, **_kwargs):
        if stage == "impact":
            raise asyncio.CancelledError()
        raise AssertionError("impact should not succeed in this test")

    monkeypatch.setattr(
        workflow_service.pipeline_service,
        "run_fact_extraction_consistency_pipeline",
        fake_pipeline,
    )
    if stage != "pipeline":
        _install_terminal_auth(
            monkeypatch,
            terminal_result=_terminal_result(
                orchestration_id=target_orchestration_id,
                status=FactExtractionOrchestrationStatus.COMPLETED,
            ),
        )
    monkeypatch.setattr(
        workflow_service.impact_service,
        "get_document_revision_update_impact",
        fake_impact,
    )
    _install_impact_auth(monkeypatch)

    with pytest.raises(asyncio.CancelledError):
        run_async(
            workflow_service.run_document_revision_update_workflow(
                ForbiddenSessionFactory(),
                **kwargs,
            )
        )


def test_run_document_revision_update_workflow_does_not_leak_sensitive_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _workflow_kwargs()
    sentinel = "SENSITIVE_WORKFLOW_SENTINEL"

    async def fake_pipeline(_session_factory, **_kwargs):
        return _pipeline_result(
            orchestration_id=uuid.uuid4(),
            extraction_status=FactExtractionOrchestrationStatus.COMPLETED,
            grouping_application_id=uuid.uuid4(),
            consistency_application_id=uuid.uuid4(),
            consistency_check_application_id=uuid.uuid4(),
            consistency_plan_manifest_hash=sentinel,
            consistency_execution_result_manifest_hash="2" * 64,
            assessment_count=1,
            consistency_created_new=True,
            skipped_reason=None,
        )

    async def fail_impact(*args, **kwargs):
        raise AssertionError(sentinel)

    monkeypatch.setattr(
        workflow_service.pipeline_service,
        "run_fact_extraction_consistency_pipeline",
        fake_pipeline,
    )
    _install_terminal_auth(
        monkeypatch,
        terminal_result=_terminal_result(
            orchestration_id=uuid.uuid4(),
            status=FactExtractionOrchestrationStatus.COMPLETED,
        ),
    )
    monkeypatch.setattr(
        workflow_service.impact_service,
        "get_document_revision_update_impact",
        fail_impact,
    )

    with pytest.raises(
        workflow_service.DocumentRevisionUpdateWorkflowInvariantError,
        match="document_revision_update_workflow_target_consistency_plan_manifest_hash_invalid",
    ) as exc_info:
        run_async(
            workflow_service.run_document_revision_update_workflow(
                ForbiddenSessionFactory(),
                **kwargs,
            )
        )

    assert sentinel not in str(exc_info.value)
