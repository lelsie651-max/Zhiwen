from __future__ import annotations

import asyncio
import hashlib
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import CheckConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex

from app.agents.fact_extraction_planner import plan_fact_extraction_batches
from app.agents.prompt_registry import get_prompt
from app.models import Base
from app.models.base import utc_now
from app.models.document_content import DocumentBlock
from app.models.fact_extraction_orchestration import (
    FactExtractionOrchestration,
    FactExtractionOrchestrationBatch,
)
from app.models.inference import InferenceRun, InferenceRunStatus
from app.schemas.fact_extraction_orchestration import (
    FactExtractionOrchestrationBatchStatus,
    FactExtractionOrchestrationResult,
    FactExtractionOrchestrationStatus,
    StaleInferenceRecoveryStatus,
)
from app.schemas.fact_extraction_plan import FactExtractionPlannerConfig
from app.services import fact_extraction_orchestration as orchestration_service


PROMPT = get_prompt("agent1_fact_extraction", "1.0.0")


def run_async(awaitable):
    return asyncio.run(awaitable)


def async_lambda(result):
    async def _call(*_args, **_kwargs):
        return result

    return _call


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FakeSession:
    def __init__(self, factory: "SessionFactory | None" = None) -> None:
        self.factory = factory
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
                session = FakeSession(factory)
                factory.sessions.append(session)
                return session

            async def __aexit__(self_inner, exc_type, exc, tb):
                factory.open_count -= 1
                return False

        return _Context()


def _source_block(*, source_order: int, raw_text: str, extraction_run_id: uuid.UUID) -> DocumentBlock:
    return DocumentBlock(
        id=uuid.uuid4(),
        extraction_run_id=extraction_run_id,
        source_order=source_order,
        block_type="paragraph",
        raw_text=raw_text,
        normalized_text=raw_text,
        location_key=f"loc-{source_order}",
        anchor_hash=sha256(f"anchor-{source_order}-{raw_text}"),
        block_index=source_order,
        heading_path=[],
    )


def _planned_fixture():
    extraction_run_id = uuid.uuid4()
    blocks = [
        _source_block(
            source_order=0,
            raw_text="蔷薇王国的首都是白蔷城。",
            extraction_run_id=extraction_run_id,
        ),
        _source_block(
            source_order=1,
            raw_text="白蔷城位于北境。",
            extraction_run_id=extraction_run_id,
        ),
        _source_block(
            source_order=2,
            raw_text="白蔷城有白石城墙。",
            extraction_run_id=extraction_run_id,
        ),
    ]
    return extraction_run_id, plan_fact_extraction_batches(
        extraction_run_id=extraction_run_id,
        blocks=blocks,
        prompt=PROMPT,
        config=FactExtractionPlannerConfig(
            target_message_characters=5000,
            max_message_characters=6000,
            max_blocks_per_batch=1,
            overlap_block_count=0,
        ),
    )


def _make_orchestration(*, status: str = "planned") -> FactExtractionOrchestration:
    started_at = utc_now() if status in {"running", "completed", "partial", "failed"} else None
    completed_at = utc_now() if status in {"completed", "partial", "failed"} else None
    failure_code = "llm_transport_error" if status == "failed" else None
    return FactExtractionOrchestration(
        id=uuid.uuid4(),
        project_id=uuid.uuid4(),
        extraction_run_id=uuid.uuid4(),
        attempt_no=1,
        request_hash="a" * 64,
        plan_hash="b" * 64,
        plan_json_hash="c" * 64,
        plan_json={"ok": True},
        status=status,
        coordinator_name=orchestration_service.FACT_EXTRACTION_COORDINATOR_NAME,
        coordinator_version=orchestration_service.FACT_EXTRACTION_COORDINATOR_VERSION,
        planner_name="deterministic_fact_block_planner",
        planner_version="1.0.1",
        agent_name=PROMPT.agent_name,
        agent_version=PROMPT.agent_version,
        prompt_name=PROMPT.prompt_name,
        prompt_version=PROMPT.prompt_version,
        prompt_contract_hash=PROMPT.contract_hash,
        provider="deepseek",
        requested_model="deepseek-v4-flash",
        executor_name="agent1_fact_extraction_batch_executor",
        executor_version="1.0.0",
        persistence_name="agent1_fact_persistence",
        persistence_version="1.0.0",
        entity_resolution_policy_name="canonical_then_unique_active_alias",
        entity_resolution_policy_version="1.0.0",
        batch_count=3,
        completed_batch_count=0,
        failed_batch_count=0,
        proposal_count=0,
        created_count=0,
        reused_count=0,
        withheld_count=0,
        failure_code=failure_code,
        started_at=started_at,
        completed_at=completed_at,
    )


def _make_batch(
    orchestration_id: uuid.UUID,
    *,
    batch_index: int,
    status: str = "pending",
    attempt_count: int = 0,
) -> FactExtractionOrchestrationBatch:
    running = status == "running"
    terminal = status in {"completed", "failed"}
    return FactExtractionOrchestrationBatch(
        id=uuid.uuid4(),
        orchestration_id=orchestration_id,
        batch_index=batch_index,
        batch_plan_hash=sha256(f"batch-{batch_index}"),
        status=status,
        attempt_count=attempt_count,
        current_input_batch_id=uuid.uuid4() if status == "completed" else None,
        current_inference_run_id=uuid.uuid4() if status == "completed" else None,
        application_id=uuid.uuid4() if status == "completed" else None,
        lease_token=uuid.uuid4() if running else None,
        lease_expires_at=utc_now() if running else None,
        proposal_count=0,
        created_count=0,
        reused_count=0,
        withheld_count=0,
        failure_code="llm_transport_error" if status == "failed" else None,
        started_at=utc_now() if running or terminal else None,
        completed_at=utc_now() if terminal else None,
    )


def test_orchestration_tables_constraints_and_active_index_exist() -> None:
    orch_table = Base.metadata.tables["fact_extraction_orchestrations"]
    batch_table = Base.metadata.tables["fact_extraction_orch_batches"]

    orch_checks = {constraint.name for constraint in orch_table.constraints if isinstance(constraint, CheckConstraint)}
    batch_checks = {constraint.name for constraint in batch_table.constraints if isinstance(constraint, CheckConstraint)}
    assert any(name.endswith("feo_status_valid") for name in orch_checks)
    assert any(name.endswith("feo_planned_shape") for name in orch_checks)
    assert any(name.endswith("feo_running_shape") for name in orch_checks)
    assert any(name.endswith("feo_completed_shape") for name in orch_checks)
    assert any(name.endswith("feo_partial_shape") for name in orch_checks)
    assert any(name.endswith("feo_failed_shape") for name in orch_checks)
    assert any(name.endswith("feob_status_valid") for name in batch_checks)
    assert any(name.endswith("feob_pending_shape") for name in batch_checks)
    assert any(name.endswith("feob_running_shape") for name in batch_checks)
    assert any(name.endswith("feob_completed_shape") for name in batch_checks)
    assert any(name.endswith("feob_failed_shape") for name in batch_checks)

    active_index = next(index for index in orch_table.indexes if index.name == "uq_feo_active_request")
    compiled = str(CreateIndex(active_index).compile(dialect=postgresql.dialect()))
    assert "WHERE status IN ('planned', 'running')" in compiled


def test_orchestration_request_hash_is_stable_and_changes_with_policy_inputs() -> None:
    extraction_run_id, plan = _planned_fixture()
    project_id = uuid.uuid4()

    first = orchestration_service.build_fact_extraction_orchestration_request_hash(
        project_id=project_id,
        extraction_run_id=extraction_run_id,
        plan_hash=plan.plan_hash,
        plan_json_hash="d" * 64,
        planner_name=plan.planner_name,
        planner_version=plan.planner_version,
        agent_name=PROMPT.agent_name,
        agent_version=PROMPT.agent_version,
        prompt_name=PROMPT.prompt_name,
        prompt_version=PROMPT.prompt_version,
        prompt_contract_hash=PROMPT.contract_hash,
        provider="deepseek",
        requested_model="deepseek-v4-flash",
        executor_name="agent1_fact_extraction_batch_executor",
        executor_version="1.0.0",
        persistence_name="agent1_fact_persistence",
        persistence_version="1.0.0",
        entity_resolution_policy_name="canonical_then_unique_active_alias",
        entity_resolution_policy_version="1.0.0",
        coordinator_name=orchestration_service.FACT_EXTRACTION_COORDINATOR_NAME,
        coordinator_version=orchestration_service.FACT_EXTRACTION_COORDINATOR_VERSION,
        max_batch_attempts=2,
        batch_lease_seconds=300,
        stale_inference_seconds=900,
    )
    second = orchestration_service.build_fact_extraction_orchestration_request_hash(
        project_id=project_id,
        extraction_run_id=extraction_run_id,
        plan_hash=plan.plan_hash,
        plan_json_hash="d" * 64,
        planner_name=plan.planner_name,
        planner_version=plan.planner_version,
        agent_name=PROMPT.agent_name,
        agent_version=PROMPT.agent_version,
        prompt_name=PROMPT.prompt_name,
        prompt_version=PROMPT.prompt_version,
        prompt_contract_hash=PROMPT.contract_hash,
        provider="deepseek",
        requested_model="deepseek-v4-flash",
        executor_name="agent1_fact_extraction_batch_executor",
        executor_version="1.0.0",
        persistence_name="agent1_fact_persistence",
        persistence_version="1.0.0",
        entity_resolution_policy_name="canonical_then_unique_active_alias",
        entity_resolution_policy_version="1.0.0",
        coordinator_name=orchestration_service.FACT_EXTRACTION_COORDINATOR_NAME,
        coordinator_version=orchestration_service.FACT_EXTRACTION_COORDINATOR_VERSION,
        max_batch_attempts=2,
        batch_lease_seconds=300,
        stale_inference_seconds=900,
    )
    changed = orchestration_service.build_fact_extraction_orchestration_request_hash(
        project_id=project_id,
        extraction_run_id=extraction_run_id,
        plan_hash=plan.plan_hash,
        plan_json_hash="d" * 64,
        planner_name=plan.planner_name,
        planner_version=plan.planner_version,
        agent_name=PROMPT.agent_name,
        agent_version=PROMPT.agent_version,
        prompt_name=PROMPT.prompt_name,
        prompt_version=PROMPT.prompt_version,
        prompt_contract_hash=PROMPT.contract_hash,
        provider="deepseek",
        requested_model="deepseek-v4-flash",
        executor_name="agent1_fact_extraction_batch_executor",
        executor_version="1.0.0",
        persistence_name="agent1_fact_persistence",
        persistence_version="1.0.0",
        entity_resolution_policy_name="canonical_then_unique_active_alias",
        entity_resolution_policy_version="1.0.0",
        coordinator_name=orchestration_service.FACT_EXTRACTION_COORDINATOR_NAME,
        coordinator_version=orchestration_service.FACT_EXTRACTION_COORDINATOR_VERSION,
        max_batch_attempts=2,
        batch_lease_seconds=301,
        stale_inference_seconds=900,
    )

    assert first == second
    assert changed != first


def test_prepare_orchestration_rejects_tampered_plan_hash_before_db_access() -> None:
    extraction_run_id, plan = _planned_fixture()
    bad_plan = plan.model_copy(update={"plan_hash": "f" * 64})

    with pytest.raises(orchestration_service.FactExtractionOrchestrationError, match="plan_hash"):
        run_async(
            orchestration_service.prepare_fact_extraction_orchestration(
                FakeSession(),
                project_id=uuid.uuid4(),
                extraction_run_id=extraction_run_id,
                plan=bad_plan,
                prompt=PROMPT,
                provider="deepseek",
                requested_model="deepseek-v4-flash",
                max_batch_attempts=2,
                batch_lease_seconds=300,
                stale_inference_seconds=900,
            )
        )


def test_claim_pending_batch_starts_running_and_sets_lease(monkeypatch) -> None:
    session = FakeSession()
    orchestration = _make_orchestration(status="planned")
    batch = _make_batch(orchestration.id, batch_index=0, status="pending")
    worker_token = uuid.uuid4()

    async def fake_get_orchestration(*_args, **_kwargs):
        return orchestration

    async def fake_get_batch(*_args, **_kwargs):
        return batch

    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_orchestration_for_update", fake_get_orchestration)
    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_batch_for_update", fake_get_batch)

    claim = run_async(
        orchestration_service.claim_fact_extraction_orchestration_batch(
            session,
            orchestration_id=orchestration.id,
            batch_index=0,
            worker_token=worker_token,
            lease_seconds=300,
            max_batch_attempts=2,
            stale_before=utc_now(),
        )
    )

    assert claim.claimed is True
    assert batch.status == "running"
    assert batch.attempt_count == 1
    assert batch.lease_token == worker_token
    assert batch.lease_expires_at is not None
    assert orchestration.status == "running"
    assert orchestration.started_at is not None


def test_claim_running_batch_respects_owner_and_expiry(monkeypatch) -> None:
    session = FakeSession()
    orchestration = _make_orchestration(status="running")
    batch = _make_batch(orchestration.id, batch_index=0, status="running", attempt_count=1)
    owner_token = batch.lease_token
    assert owner_token is not None

    async def fake_get_orchestration(*_args, **_kwargs):
        return orchestration

    async def fake_get_batch(*_args, **_kwargs):
        return batch

    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_orchestration_for_update", fake_get_orchestration)
    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_batch_for_update", fake_get_batch)

    same_worker = run_async(
        orchestration_service.claim_fact_extraction_orchestration_batch(
            session,
            orchestration_id=orchestration.id,
            batch_index=0,
            worker_token=owner_token,
            lease_seconds=300,
            max_batch_attempts=2,
            stale_before=utc_now(),
        )
    )
    assert same_worker.claimed is True

    other_worker = run_async(
        orchestration_service.claim_fact_extraction_orchestration_batch(
            session,
            orchestration_id=orchestration.id,
            batch_index=0,
            worker_token=uuid.uuid4(),
            lease_seconds=300,
            max_batch_attempts=2,
            stale_before=utc_now(),
        )
    )
    assert other_worker.claimed is False

    batch.lease_expires_at = utc_now().replace(year=2020)
    recovered = run_async(
        orchestration_service.claim_fact_extraction_orchestration_batch(
            session,
            orchestration_id=orchestration.id,
            batch_index=0,
            worker_token=uuid.uuid4(),
            lease_seconds=300,
            max_batch_attempts=2,
            stale_before=utc_now(),
        )
    )
    assert recovered.claimed is True
    assert batch.attempt_count == 1


def test_recover_stale_inference_run_marks_old_running_failed(monkeypatch) -> None:
    session = FakeSession()
    run = InferenceRun(
        id=uuid.uuid4(),
        project_id=uuid.uuid4(),
        input_batch_id=uuid.uuid4(),
        task_type="fact_extraction",
        attempt_no=1,
        status=InferenceRunStatus.RUNNING.value,
        agent_name=PROMPT.agent_name,
        agent_version=PROMPT.agent_version,
        prompt_name=PROMPT.prompt_name,
        prompt_version=PROMPT.prompt_version,
        prompt_contract_hash=PROMPT.contract_hash,
        request_hash="a" * 64,
        request_metadata={},
        provider="deepseek",
        requested_model="deepseek-v4-flash",
        temperature=PROMPT.temperature,
        max_output_tokens=PROMPT.max_output_tokens,
        attempt_count=1,
        started_at=utc_now().replace(year=2020),
    )

    async def fake_get_run(*_args, **_kwargs):
        return run

    monkeypatch.setattr(orchestration_service.inference_repository, "get_run_for_update", fake_get_run)

    result = run_async(
        orchestration_service.recover_stale_fact_extraction_inference_run(
            session,
            inference_run_id=run.id,
            stale_before=utc_now(),
        )
    )

    assert result.status == StaleInferenceRecoveryStatus.FAILED
    assert result.recovered_stale_run is True
    assert run.status == InferenceRunStatus.FAILED.value
    assert run.failure_code == "fact_extraction_execution_stale"
    assert run.completed_at is not None


def test_finalize_orchestration_aggregates_completed_application_counts(monkeypatch) -> None:
    session = FakeSession()
    orchestration = _make_orchestration(status="running")
    completed_a = _make_batch(orchestration.id, batch_index=0, status="completed")
    completed_b = _make_batch(orchestration.id, batch_index=1, status="completed")
    completed_a.proposal_count = 2
    completed_a.created_count = 1
    completed_a.reused_count = 0
    completed_a.withheld_count = 1
    completed_b.proposal_count = 1
    completed_b.created_count = 0
    completed_b.reused_count = 1
    completed_b.withheld_count = 0
    failed = _make_batch(orchestration.id, batch_index=2, status="failed")
    failed.failure_code = "llm_transport_error"
    app_a = SimpleNamespace(
        id=completed_a.application_id,
        status="completed",
        project_id=orchestration.project_id,
        extraction_run_id=orchestration.extraction_run_id,
        input_batch_id=completed_a.current_input_batch_id,
        inference_run_id=completed_a.current_inference_run_id,
        response_hash="a" * 64,
        persistence_name="agent1_fact_persistence",
        persistence_version="1.0.0",
        entity_resolution_policy_name="canonical_then_unique_active_alias",
        entity_resolution_policy_version="1.0.0",
        result_json={
            "application_id": str(completed_a.application_id),
            "replayed_application": False,
            "project_id": str(orchestration.project_id),
            "extraction_run_id": str(orchestration.extraction_run_id),
            "inference_run_id": str(completed_a.current_inference_run_id),
            "input_batch_id": str(completed_a.current_input_batch_id),
            "response_hash": "a" * 64,
            "persistence_name": "agent1_fact_persistence",
            "persistence_version": "1.0.0",
            "entity_resolution_policy_name": "canonical_then_unique_active_alias",
            "entity_resolution_policy_version": "1.0.0",
            "proposal_count": 2,
            "created_count": 1,
            "reused_count": 0,
            "withheld_count": 1,
            "items": [
                {
                    "proposal_index": 0,
                    "proposal_hash": "a" * 64,
                    "outcome": "created",
                    "withheld_reason": None,
                    "subject_resolution_status": "unresolved",
                    "referenced_resolution_status": None,
                    "fact_id": str(uuid.uuid4()),
                    "fact_value_id": str(uuid.uuid4()),
                    "subject_entity_id": None,
                    "referenced_entity_id": None,
                    "evidence_ids": [],
                },
                {
                    "proposal_index": 1,
                    "proposal_hash": "b" * 64,
                    "outcome": "withheld",
                    "withheld_reason": "subject_ambiguous",
                    "subject_resolution_status": "ambiguous",
                    "referenced_resolution_status": None,
                    "fact_id": None,
                    "fact_value_id": None,
                    "subject_entity_id": None,
                    "referenced_entity_id": None,
                    "evidence_ids": [],
                },
            ],
        },
        result_hash="",
    )
    app_a.result_hash = orchestration_service.build_fact_extraction_application_result_hash(app_a.result_json)
    app_b = SimpleNamespace(
        id=completed_b.application_id,
        status="completed",
        project_id=orchestration.project_id,
        extraction_run_id=orchestration.extraction_run_id,
        input_batch_id=completed_b.current_input_batch_id,
        inference_run_id=completed_b.current_inference_run_id,
        response_hash="b" * 64,
        persistence_name="agent1_fact_persistence",
        persistence_version="1.0.0",
        entity_resolution_policy_name="canonical_then_unique_active_alias",
        entity_resolution_policy_version="1.0.0",
        result_json={
            "application_id": str(completed_b.application_id),
            "replayed_application": False,
            "project_id": str(orchestration.project_id),
            "extraction_run_id": str(orchestration.extraction_run_id),
            "inference_run_id": str(completed_b.current_inference_run_id),
            "input_batch_id": str(completed_b.current_input_batch_id),
            "response_hash": "b" * 64,
            "persistence_name": "agent1_fact_persistence",
            "persistence_version": "1.0.0",
            "entity_resolution_policy_name": "canonical_then_unique_active_alias",
            "entity_resolution_policy_version": "1.0.0",
            "proposal_count": 1,
            "created_count": 0,
            "reused_count": 1,
            "withheld_count": 0,
            "items": [
                {
                    "proposal_index": 0,
                    "proposal_hash": "c" * 64,
                    "outcome": "reused",
                    "withheld_reason": None,
                    "subject_resolution_status": "unresolved",
                    "referenced_resolution_status": None,
                    "fact_id": str(uuid.uuid4()),
                    "fact_value_id": str(uuid.uuid4()),
                    "subject_entity_id": None,
                    "referenced_entity_id": None,
                    "evidence_ids": [],
                },
            ],
        },
        result_hash="",
    )
    app_b.result_hash = orchestration_service.build_fact_extraction_application_result_hash(app_b.result_json)

    async def fake_get_orchestration(*_args, **_kwargs):
        return orchestration

    async def fake_list_batches(*_args, **_kwargs):
        return [completed_a, completed_b, failed]

    async def fake_list_applications(*_args, **_kwargs):
        return [app_a, app_b]

    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_orchestration_for_update", fake_get_orchestration)
    monkeypatch.setattr(orchestration_service.orchestration_repository, "list_batches_for_orchestration_for_update", fake_list_batches)
    monkeypatch.setattr(orchestration_service.orchestration_repository, "list_applications", fake_list_applications)

    result = run_async(
        orchestration_service._finalize_orchestration(
            session,
            orchestration_id=orchestration.id,
        )
    )

    assert result.status == FactExtractionOrchestrationStatus.PARTIAL
    assert result.completed_batch_count == 2
    assert result.failed_batch_count == 1
    assert result.proposal_count == 3
    assert result.created_count == 1
    assert result.reused_count == 1
    assert result.withheld_count == 1


def test_execute_orchestration_skips_completed_batches_and_keeps_sessions_closed_during_llm(monkeypatch) -> None:
    session_factory = SessionFactory()
    prepared_calls: list[tuple[int, uuid.UUID, uuid.UUID]] = []
    executed_batches: list[int] = []
    finalized_success: list[int] = []
    extraction_run_id, plan = _planned_fixture()

    async def fake_prepare(*_args, **_kwargs):
        return orchestration_service.PreparedFactExtractionOrchestration(
            orchestration_id=uuid.uuid4(),
            attempt_no=1,
            request_hash="a" * 64,
            plan_hash=plan.plan_hash,
            reused_completed=False,
        )

    async def fake_list_batches(*_args, **_kwargs):
        return [SimpleNamespace(batch_index=0), SimpleNamespace(batch_index=1)]

    async def fake_claim(*_args, batch_index, **_kwargs):
        if batch_index == 0:
            return orchestration_service.FactExtractionBatchLeaseClaim(
                orchestration_id=uuid.uuid4(),
                batch_id=uuid.uuid4(),
                batch_index=0,
                batch_plan_hash=plan.batches[0].plan_hash,
                status="completed",
                claimed=False,
                attempt_count=1,
                current_input_batch_id=uuid.uuid4(),
                current_inference_run_id=uuid.uuid4(),
                application_id=uuid.uuid4(),
                lease_token=None,
                lease_expires_at=None,
                failure_code=None,
            )
        return orchestration_service.FactExtractionBatchLeaseClaim(
            orchestration_id=uuid.uuid4(),
            batch_id=uuid.uuid4(),
            batch_index=1,
            batch_plan_hash=plan.batches[1].plan_hash,
            status="running",
            claimed=True,
            attempt_count=1,
            current_input_batch_id=None,
            current_inference_run_id=None,
            application_id=None,
            lease_token=uuid.uuid4(),
            lease_expires_at=utc_now(),
            failure_code=None,
        )

    async def fake_record_notice(_session, *, notice, **_kwargs):
        prepared_calls.append((notice.batch_index, notice.input_batch_id, notice.inference_run_id))

    async def fake_execute_batch(
        factory,
        *,
        batch_index,
        prepared_run_observer,
        **_kwargs,
    ):
        assert factory.open_count == 0
        executed_batches.append(batch_index)
        input_batch_id = uuid.uuid4()
        inference_run_id = uuid.uuid4()
        await prepared_run_observer(
            orchestration_service.PreparedFactExtractionRunNotice(
                project_id=uuid.uuid4(),
                extraction_run_id=extraction_run_id,
                plan_hash=plan.plan_hash,
                batch_index=batch_index,
                batch_plan_hash=plan.batches[batch_index].plan_hash,
                input_batch_id=input_batch_id,
                inference_run_id=inference_run_id,
                inference_request_hash="b" * 64,
            )
        )
        return SimpleNamespace(
            batch_index=batch_index,
            batch_plan_hash=plan.batches[batch_index].plan_hash,
            input_batch_id=input_batch_id,
            inference_run_id=inference_run_id,
            project_id=uuid.uuid4(),
            extraction_run_id=extraction_run_id,
        )

    async def fake_persist(*_args, **kwargs):
        return SimpleNamespace(
            application_id=uuid.uuid4(),
            proposal_count=1,
            created_count=1,
            reused_count=0,
            withheld_count=0,
            inference_run_id=kwargs["inference_run_id"],
            input_batch_id=prepared_calls[-1][1],
            project_id=uuid.uuid4(),
            extraction_run_id=extraction_run_id,
        )

    async def fake_finalize_success(*_args, execution_result, **_kwargs):
        finalized_success.append(execution_result.batch_index)

    async def fake_finalize_orchestration(*_args, **_kwargs):
        return FactExtractionOrchestrationResult(
            orchestration_id=uuid.uuid4(),
            attempt_no=1,
            request_hash="a" * 64,
            plan_hash=plan.plan_hash,
            status=FactExtractionOrchestrationStatus.COMPLETED,
            batch_count=len(plan.batches),
            completed_batch_count=1,
            failed_batch_count=0,
            proposal_count=1,
            created_count=1,
            reused_count=0,
            withheld_count=0,
            batches=(
                orchestration_service.FactExtractionOrchestrationBatchResult(
                    batch_index=1,
                    batch_plan_hash=plan.batches[1].plan_hash,
                    status=FactExtractionOrchestrationBatchStatus.COMPLETED,
                    attempt_count=1,
                    input_batch_id=prepared_calls[-1][1],
                    inference_run_id=prepared_calls[-1][2],
                    application_id=uuid.uuid4(),
                    proposal_count=1,
                    created_count=1,
                    reused_count=0,
                    withheld_count=0,
                    failure_code=None,
                ),
            ),
        )

    monkeypatch.setattr(orchestration_service, "prepare_fact_extraction_orchestration", fake_prepare)
    monkeypatch.setattr(orchestration_service.orchestration_repository, "list_batches_for_orchestration", fake_list_batches)
    monkeypatch.setattr(orchestration_service, "claim_fact_extraction_orchestration_batch", fake_claim)
    monkeypatch.setattr(orchestration_service, "_record_prepared_run_notice", fake_record_notice)
    monkeypatch.setattr(orchestration_service, "execute_fact_extraction_batch", fake_execute_batch)
    monkeypatch.setattr(orchestration_service, "persist_completed_fact_extraction_batch", fake_persist)
    monkeypatch.setattr(orchestration_service, "_finalize_batch_success", fake_finalize_success)
    monkeypatch.setattr(orchestration_service, "_finalize_orchestration", fake_finalize_orchestration)
    async def fake_run_with_heartbeat(_factory, *, operation, **_kwargs):
        return await operation()

    monkeypatch.setattr(orchestration_service, "run_with_batch_lease_heartbeat", fake_run_with_heartbeat)
    monkeypatch.setattr(orchestration_service, "renew_fact_extraction_orchestration_batch_lease", async_lambda(None))

    result = run_async(
        orchestration_service.execute_fact_extraction_orchestration(
            session_factory,
            project_id=uuid.uuid4(),
            extraction_run_id=extraction_run_id,
            plan=plan,
            prompt=PROMPT,
            llm_client=SimpleNamespace(),
            provider="deepseek",
            requested_model="deepseek-v4-flash",
            worker_token=uuid.uuid4(),
        )
    )

    assert executed_batches == [1]
    assert finalized_success == [1]
    assert len(prepared_calls) == 1
    assert result.status == FactExtractionOrchestrationStatus.COMPLETED


def test_execute_orchestration_cancelled_cleans_up_and_reraises(monkeypatch) -> None:
    session_factory = SessionFactory()
    extraction_run_id, plan = _planned_fixture()
    cleanup_calls: list[tuple[int, str]] = []

    async def fake_prepare(*_args, **_kwargs):
        return orchestration_service.PreparedFactExtractionOrchestration(
            orchestration_id=uuid.uuid4(),
            attempt_no=1,
            request_hash="a" * 64,
            plan_hash=plan.plan_hash,
            reused_completed=False,
        )

    async def fake_list_batches(*_args, **_kwargs):
        return [SimpleNamespace(batch_index=0)]

    async def fake_claim(*_args, **_kwargs):
        return orchestration_service.FactExtractionBatchLeaseClaim(
            orchestration_id=uuid.uuid4(),
            batch_id=uuid.uuid4(),
            batch_index=0,
            batch_plan_hash=plan.batches[0].plan_hash,
            status="running",
            claimed=True,
            attempt_count=1,
            current_input_batch_id=None,
            current_inference_run_id=None,
            application_id=None,
            lease_token=uuid.uuid4(),
            lease_expires_at=utc_now(),
            failure_code=None,
        )

    async def fake_execute_batch(factory, **_kwargs):
        assert factory.open_count == 0
        raise asyncio.CancelledError()

    async def fake_finalize_failure(*_args, batch_index, failure_code, **_kwargs):
        cleanup_calls.append((batch_index, failure_code))

    monkeypatch.setattr(orchestration_service, "prepare_fact_extraction_orchestration", fake_prepare)
    monkeypatch.setattr(orchestration_service.orchestration_repository, "list_batches_for_orchestration", fake_list_batches)
    monkeypatch.setattr(orchestration_service, "claim_fact_extraction_orchestration_batch", fake_claim)
    monkeypatch.setattr(orchestration_service, "execute_fact_extraction_batch", fake_execute_batch)
    monkeypatch.setattr(orchestration_service, "_best_effort_finalize_cancelled_batch", async_lambda(None))

    async def fake_get_batch_for_update(*_args, **_kwargs):
        return SimpleNamespace(current_inference_run_id=None)

    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_batch_for_update", fake_get_batch_for_update)

    with pytest.raises(asyncio.CancelledError):
        run_async(
            orchestration_service.execute_fact_extraction_orchestration(
                session_factory,
                project_id=uuid.uuid4(),
                extraction_run_id=extraction_run_id,
                plan=plan,
                prompt=PROMPT,
                llm_client=SimpleNamespace(),
                provider="deepseek",
                requested_model="deepseek-v4-flash",
                worker_token=uuid.uuid4(),
            )
        )

    assert cleanup_calls == []


def test_orchestration_migration_is_latest_head_and_declares_tables() -> None:
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    migration = Path("alembic/versions/202607312230_orchestration_recovery_hardening.py").read_text(encoding="utf-8")
    assert "fact_extraction_orchestrations" in migration
    assert "fact_extraction_orch_batches" in migration
    assert "feo_terminal_batch_counts_within_batch_count" in migration

    root = Path(__file__).resolve().parents[1]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    script = ScriptDirectory.from_config(config)
    assert list(script.get_heads()) == ["202607312230"]
