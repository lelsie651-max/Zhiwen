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
from app.models.document_content import DocumentBlock, ExtractionRun as DocumentExtractionRun
from app.models.fact_extraction_orchestration import (
    FactExtractionOrchestration,
    FactExtractionOrchestrationBatch,
)
from app.models.inference import InferenceRun, InferenceRunStatus
from app.repositories import fact_extraction_orchestration as orchestration_repository
from app.repositories import inference as inference_repository
from app.schemas.fact_extraction_orchestration import (
    FactExtractionOrchestrationBatchStatus,
    FactExtractionOrchestrationResult,
    FactExtractionOrchestrationStatus,
    StaleInferenceRecoveryStatus,
)
from app.schemas.fact_extraction_plan import FactExtractionPlannerConfig
from app.services import fact_extraction_orchestration as orchestration_service
from app.services import fact_value_duplicate_grouping as duplicate_grouping_service


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


class ExecuteResult:
    def __init__(self, *, scalar=None, row=None, scalars=None) -> None:
        self._scalar = scalar
        self._row = row
        self._scalars = scalars or []

    def scalar_one_or_none(self):
        return self._scalar

    def one_or_none(self):
        return self._row

    def scalars(self):
        values = self._scalars

        class _Scalars:
            def all(self_inner):
                return list(values)

            def first(self_inner):
                return values[0] if values else None

        return _Scalars()


class StatementCaptureSession:
    def __init__(self, result: ExecuteResult | list[ExecuteResult]) -> None:
        self.result = result
        self.statements = []

    async def execute(self, statement):
        self.statements.append(statement)
        if isinstance(self.result, list):
            return self.result.pop(0)
        return self.result


class FakeExtractionRunRow:
    def __init__(self, extraction_run, project_id, revision_status) -> None:
        self._values = (extraction_run, project_id, revision_status)
        self.project_id = project_id
        self.revision_status = revision_status

    def __getitem__(self, index):
        return self._values[index]


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


def _make_orchestration_result(
    *,
    status: FactExtractionOrchestrationStatus = FactExtractionOrchestrationStatus.COMPLETED,
) -> FactExtractionOrchestrationResult:
    return FactExtractionOrchestrationResult(
        orchestration_id=uuid.uuid4(),
        attempt_no=1,
        request_hash="a" * 64,
        plan_hash="b" * 64,
        status=status,
        batch_count=1,
        completed_batch_count=1 if status != FactExtractionOrchestrationStatus.FAILED else 0,
        failed_batch_count=0 if status != FactExtractionOrchestrationStatus.FAILED else 1,
        proposal_count=1 if status != FactExtractionOrchestrationStatus.FAILED else 0,
        created_count=1 if status != FactExtractionOrchestrationStatus.FAILED else 0,
        reused_count=0,
        withheld_count=0,
        batches=(),
    )


def test_orchestration_tables_constraints_and_active_index_exist() -> None:
    orch_table = Base.metadata.tables["fact_extraction_orchestrations"]
    batch_table = Base.metadata.tables["fact_extraction_orch_batches"]

    orch_checks = {constraint.name for constraint in orch_table.constraints if isinstance(constraint, CheckConstraint)}
    orch_uniques = {
        tuple(constraint.columns.keys()): constraint.name
        for constraint in orch_table.constraints
        if not isinstance(constraint, CheckConstraint) and hasattr(constraint, "columns")
    }
    batch_checks = {constraint.name for constraint in batch_table.constraints if isinstance(constraint, CheckConstraint)}
    assert any(name.endswith("feo_status_valid") for name in orch_checks)
    assert any(name.endswith("feo_planned_shape") for name in orch_checks)
    assert any(name.endswith("feo_running_shape") for name in orch_checks)
    assert any(name.endswith("feo_completed_shape") for name in orch_checks)
    assert any(name.endswith("feo_partial_shape") for name in orch_checks)
    assert any(name.endswith("feo_failed_shape") for name in orch_checks)
    assert orch_uniques[("id", "project_id")] == "uq_feo_id_project"
    assert orch_uniques[("id", "extraction_run_id")] == "uq_feo_id_extraction_run"
    assert any(name.endswith("feob_status_valid") for name in batch_checks)
    assert any(name.endswith("feob_pending_shape") for name in batch_checks)
    assert any(name.endswith("feob_running_shape") for name in batch_checks)
    assert any(name.endswith("feob_completed_shape") for name in batch_checks)
    assert any(name.endswith("feob_failed_shape") for name in batch_checks)

    active_index = next(index for index in orch_table.indexes if index.name == "uq_feo_active_request")
    compiled = str(CreateIndex(active_index).compile(dialect=postgresql.dialect()))
    assert "WHERE status IN ('planned', 'running')" in compiled


def test_repository_explicitly_separates_document_and_inference_runs() -> None:
    repo_path = Path("app/repositories/fact_extraction_orchestration.py")
    source = repo_path.read_text(encoding="utf-8")

    assert "from app.models.document_content import ExtractionRun as DocumentExtractionRun" in source
    assert "from app.models.inference import InferenceRun" in source
    assert "get_batch_attempt_reconciliation_context_for_update" not in source
    assert "row.DocumentExtractionRun" not in source


def test_document_extraction_run_alias_keeps_original_class_name() -> None:
    assert DocumentExtractionRun.__name__ == "ExtractionRun"


def test_get_extraction_run_with_project_for_update_reads_entity_without_alias_attribute() -> None:
    extraction_run = DocumentExtractionRun(
        revision_id=uuid.uuid4(),
        attempt_no=1,
        status="completed",
        outcome="success",
        extractor_name="doc_extractor",
        extractor_version="1.0.0",
        detected_format="markdown",
        detected_encoding="utf-8",
        page_count=1,
        character_count=10,
        block_count=1,
        warnings=[],
        content_metadata={},
        failure_code=None,
        failure_message=None,
        started_at=utc_now(),
        completed_at=utc_now(),
    )
    project_id = uuid.uuid4()
    row = FakeExtractionRunRow(
        extraction_run=extraction_run,
        project_id=project_id,
        revision_status="awaiting_review",
    )
    session = StatementCaptureSession(ExecuteResult(row=row))

    context = run_async(
        orchestration_repository.get_extraction_run_with_project_for_update(
            session,
            extraction_run_id=uuid.uuid4(),
        )
    )

    assert context is not None
    assert context.extraction_run is extraction_run
    assert context.project_id == project_id
    assert context.revision_status == "awaiting_review"


def test_get_extraction_run_with_project_for_update_returns_none_when_missing() -> None:
    session = StatementCaptureSession(ExecuteResult(row=None))

    context = run_async(
        orchestration_repository.get_extraction_run_with_project_for_update(
            session,
            extraction_run_id=uuid.uuid4(),
        )
    )

    assert context is None


def test_get_extraction_run_with_project_for_update_compiles_postgresql_sql() -> None:
    session = StatementCaptureSession(ExecuteResult(row=None))

    run_async(
        orchestration_repository.get_extraction_run_with_project_for_update(
            session,
            extraction_run_id=uuid.uuid4(),
        )
    )

    sql = str(session.statements[0].compile(dialect=postgresql.dialect()))
    assert "FROM extraction_runs" in sql
    assert "JOIN document_revisions" in sql
    assert "JOIN documents" in sql
    assert "FOR UPDATE OF extraction_runs" in sql
    assert "FROM inference_runs" not in sql
    assert "extraction_runs.task_type" not in sql
    assert "extraction_runs.input_batch_id" not in sql


def test_inference_run_lock_query_compiles_with_inference_table_fields() -> None:
    run_id = uuid.uuid4()
    session = StatementCaptureSession(ExecuteResult(scalar=None))

    run_async(inference_repository.get_run_for_update(session, run_id))

    sql = str(session.statements[0].compile(dialect=postgresql.dialect()))
    assert "FROM inference_runs" in sql
    assert "inference_runs.project_id" in sql
    assert "inference_runs.task_type" in sql
    assert "inference_runs.input_batch_id" in sql
    assert "inference_runs.failure_code" in sql
    assert "extraction_runs.task_type" not in sql
    assert "FOR UPDATE" in sql


def test_application_by_inference_run_query_uses_where_and_for_update() -> None:
    run_id = uuid.uuid4()
    session = StatementCaptureSession(ExecuteResult(scalar=None))

    run_async(
        orchestration_repository.get_application_by_inference_run_for_update(
            session,
            inference_run_id=run_id,
        )
    )

    sql = str(session.statements[0].compile(dialect=postgresql.dialect()))
    assert "FROM fact_extraction_batch_applications" in sql
    assert "fact_extraction_batch_applications.inference_run_id" in sql
    assert "FOR UPDATE" in sql
    assert "LEFT OUTER JOIN" not in sql


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


@pytest.mark.parametrize(
    ("run_status", "expected_error"),
    [
        (InferenceRunStatus.COMPLETED.value, "completed_inference_requires_reconciliation"),
        (InferenceRunStatus.PENDING.value, "active_inference_requires_recovery"),
        (InferenceRunStatus.RUNNING.value, "active_inference_requires_recovery"),
    ],
)
def test_transition_after_failed_attempt_rejects_non_failed_runs(
    monkeypatch,
    run_status: str,
    expected_error: str,
) -> None:
    session = FakeSession()
    orchestration = _make_orchestration(status="running")
    batch = _make_batch(orchestration.id, batch_index=0, status="running", attempt_count=1)
    batch.current_input_batch_id = uuid.uuid4()
    batch.current_inference_run_id = uuid.uuid4()
    batch.lease_token = uuid.uuid4()
    run = SimpleNamespace(
        id=batch.current_inference_run_id,
        status=run_status,
        project_id=orchestration.project_id,
        task_type="fact_extraction",
        input_batch_id=batch.current_input_batch_id,
        failure_code="llm_transport_error",
    )

    async def fake_get_orchestration(*_args, **_kwargs):
        return orchestration

    async def fake_get_batch(*_args, **_kwargs):
        return batch

    async def fake_get_run(*_args, **_kwargs):
        return run

    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_orchestration_for_update", fake_get_orchestration)
    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_batch_for_update", fake_get_batch)
    monkeypatch.setattr(orchestration_service.inference_repository, "get_run_for_update", fake_get_run)

    with pytest.raises(orchestration_service.FactExtractionOrchestrationStateError, match=expected_error):
        run_async(
            orchestration_service.transition_batch_after_failed_inference_attempt(
                session,
                orchestration_id=orchestration.id,
                batch_index=0,
                worker_token=batch.lease_token,
                inference_run_id=batch.current_inference_run_id,
                failure_code="llm_transport_error",
                max_batch_attempts=2,
            )
        )


def test_transition_after_failed_attempt_accepts_failed_run_and_preserves_input_batch(monkeypatch) -> None:
    session = FakeSession()
    orchestration = _make_orchestration(status="running")
    batch = _make_batch(orchestration.id, batch_index=0, status="running", attempt_count=1)
    batch.current_input_batch_id = uuid.uuid4()
    batch.current_inference_run_id = uuid.uuid4()
    batch.lease_token = uuid.uuid4()
    run = SimpleNamespace(
        id=batch.current_inference_run_id,
        status=InferenceRunStatus.FAILED.value,
        project_id=orchestration.project_id,
        task_type="fact_extraction",
        input_batch_id=batch.current_input_batch_id,
        failure_code="llm_transport_error",
    )

    async def fake_get_orchestration(*_args, **_kwargs):
        return orchestration

    async def fake_get_batch(*_args, **_kwargs):
        return batch

    async def fake_get_run(*_args, **_kwargs):
        return run

    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_orchestration_for_update", fake_get_orchestration)
    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_batch_for_update", fake_get_batch)
    monkeypatch.setattr(orchestration_service.inference_repository, "get_run_for_update", fake_get_run)

    transition = run_async(
        orchestration_service.transition_batch_after_failed_inference_attempt(
            session,
            orchestration_id=orchestration.id,
            batch_index=0,
            worker_token=batch.lease_token,
            inference_run_id=batch.current_inference_run_id,
            failure_code="llm_transport_error",
            max_batch_attempts=2,
        )
    )

    assert transition.status == "pending"
    assert transition.current_input_batch_id == batch.current_input_batch_id
    assert transition.current_inference_run_id is None


def test_reconcile_completed_application_finalizes_batch_even_if_attempts_exhausted(monkeypatch) -> None:
    session = FakeSession()
    orchestration = _make_orchestration(status="running")
    batch = _make_batch(orchestration.id, batch_index=0, status="running", attempt_count=1)
    batch.current_input_batch_id = uuid.uuid4()
    batch.current_inference_run_id = uuid.uuid4()
    batch.lease_token = uuid.uuid4()
    context = orchestration_service.orchestration_repository.BatchAttemptReconciliationContext(
        orchestration_id=orchestration.id,
        orchestration_status="running",
        orchestration_project_id=orchestration.project_id,
        orchestration_extraction_run_id=orchestration.extraction_run_id,
        batch_id=batch.id,
        batch_index=0,
        batch_status="running",
        attempt_count=1,
        lease_token=batch.lease_token,
        input_batch_id=batch.current_input_batch_id,
        inference_run_id=batch.current_inference_run_id,
        inference_run_status=InferenceRunStatus.COMPLETED.value,
        inference_run_project_id=orchestration.project_id,
        inference_run_task_type="fact_extraction",
        inference_run_input_batch_id=batch.current_input_batch_id,
        inference_run_failure_code=None,
        application_id=uuid.uuid4(),
        application_status="completed",
        batch_application_id=None,
        batch_application_status=None,
        run_application_id=uuid.uuid4(),
        run_application_status="completed",
    )
    finalized_calls: list[tuple[uuid.UUID | None, uuid.UUID, uuid.UUID]] = []

    async def fake_lock_state(*_args, **_kwargs):
        return orchestration, batch, None, None, None, context

    async def fake_finalize(*_args, worker_token, inference_run_id, application_id, **_kwargs):
        finalized_calls.append((worker_token, inference_run_id, application_id))
        return orchestration_service.FactExtractionOrchestrationBatchResult(
            batch_index=0,
            batch_plan_hash=batch.batch_plan_hash,
            status=FactExtractionOrchestrationBatchStatus.COMPLETED,
            attempt_count=1,
            input_batch_id=batch.current_input_batch_id,
            inference_run_id=batch.current_inference_run_id,
            application_id=application_id,
            proposal_count=1,
            created_count=1,
            reused_count=0,
            withheld_count=0,
            failure_code=None,
        )

    monkeypatch.setattr(orchestration_service, "_lock_batch_attempt_reconciliation_state", fake_lock_state)
    monkeypatch.setattr(orchestration_service, "finalize_batch_from_completed_application", fake_finalize)

    result = run_async(
        orchestration_service.reconcile_fact_extraction_batch_after_interruption(
            session,
            orchestration_id=orchestration.id,
            batch_index=0,
            worker_token=batch.lease_token,
            failure_code="persistence_context_invalid",
            max_batch_attempts=1,
        )
    )

    assert result.batch_status == "completed"
    assert result.application_id == finalized_calls[0][2]
    assert finalized_calls == [(batch.lease_token, batch.current_inference_run_id, finalized_calls[0][2])]


def test_reconcile_completed_run_without_application_keeps_run_anchor_and_expires_lease(monkeypatch) -> None:
    session = FakeSession()
    orchestration = _make_orchestration(status="running")
    batch = _make_batch(orchestration.id, batch_index=0, status="running", attempt_count=1)
    batch.current_input_batch_id = uuid.uuid4()
    batch.current_inference_run_id = uuid.uuid4()
    batch.lease_token = uuid.uuid4()
    batch.lease_expires_at = utc_now()
    context = orchestration_service.orchestration_repository.BatchAttemptReconciliationContext(
        orchestration_id=orchestration.id,
        orchestration_status="running",
        orchestration_project_id=orchestration.project_id,
        orchestration_extraction_run_id=orchestration.extraction_run_id,
        batch_id=batch.id,
        batch_index=0,
        batch_status="running",
        attempt_count=1,
        lease_token=batch.lease_token,
        input_batch_id=batch.current_input_batch_id,
        inference_run_id=batch.current_inference_run_id,
        inference_run_status=InferenceRunStatus.COMPLETED.value,
        inference_run_project_id=orchestration.project_id,
        inference_run_task_type="fact_extraction",
        inference_run_input_batch_id=batch.current_input_batch_id,
        inference_run_failure_code=None,
        application_id=None,
        application_status=None,
        batch_application_id=None,
        batch_application_status=None,
        run_application_id=None,
        run_application_status=None,
    )

    run = SimpleNamespace(
        id=batch.current_inference_run_id,
        status=InferenceRunStatus.COMPLETED.value,
        project_id=orchestration.project_id,
        task_type="fact_extraction",
        input_batch_id=batch.current_input_batch_id,
        failure_code=None,
    )

    async def fake_lock_state(*_args, **_kwargs):
        return orchestration, batch, run, None, None, context

    monkeypatch.setattr(orchestration_service, "_lock_batch_attempt_reconciliation_state", fake_lock_state)

    result = run_async(
        orchestration_service.reconcile_fact_extraction_batch_after_interruption(
            session,
            orchestration_id=orchestration.id,
            batch_index=0,
            worker_token=batch.lease_token,
            failure_code="fact_extraction_execution_cancelled",
            max_batch_attempts=2,
        )
    )

    assert result.batch_status == "running"
    assert batch.current_inference_run_id == context.inference_run_id
    assert batch.current_input_batch_id == context.input_batch_id
    assert batch.lease_expires_at is not None and batch.lease_expires_at < utc_now()


def test_reconcile_completed_run_with_persistence_context_invalid_fails_batch_but_preserves_run(monkeypatch) -> None:
    session = FakeSession()
    orchestration = _make_orchestration(status="running")
    batch = _make_batch(orchestration.id, batch_index=0, status="running", attempt_count=1)
    batch.current_input_batch_id = uuid.uuid4()
    batch.current_inference_run_id = uuid.uuid4()
    batch.lease_token = uuid.uuid4()
    context = orchestration_service.orchestration_repository.BatchAttemptReconciliationContext(
        orchestration_id=orchestration.id,
        orchestration_status="running",
        orchestration_project_id=orchestration.project_id,
        orchestration_extraction_run_id=orchestration.extraction_run_id,
        batch_id=batch.id,
        batch_index=0,
        batch_status="running",
        attempt_count=1,
        lease_token=batch.lease_token,
        input_batch_id=batch.current_input_batch_id,
        inference_run_id=batch.current_inference_run_id,
        inference_run_status=InferenceRunStatus.COMPLETED.value,
        inference_run_project_id=orchestration.project_id,
        inference_run_task_type="fact_extraction",
        inference_run_input_batch_id=batch.current_input_batch_id,
        inference_run_failure_code=None,
        application_id=None,
        application_status=None,
        batch_application_id=None,
        batch_application_status=None,
        run_application_id=None,
        run_application_status=None,
    )

    run = SimpleNamespace(
        id=batch.current_inference_run_id,
        status=InferenceRunStatus.COMPLETED.value,
        project_id=orchestration.project_id,
        task_type="fact_extraction",
        input_batch_id=batch.current_input_batch_id,
        failure_code=None,
    )

    async def fake_lock_state(*_args, **_kwargs):
        return orchestration, batch, run, None, None, context

    monkeypatch.setattr(orchestration_service, "_lock_batch_attempt_reconciliation_state", fake_lock_state)

    result = run_async(
        orchestration_service.reconcile_fact_extraction_batch_after_interruption(
            session,
            orchestration_id=orchestration.id,
            batch_index=0,
            worker_token=batch.lease_token,
            failure_code="persistence_context_invalid",
            max_batch_attempts=2,
        )
    )

    assert result.batch_status == "failed"
    assert batch.current_inference_run_id == context.inference_run_id
    assert batch.failure_code == "persistence_context_invalid"


def test_reconcile_does_not_modify_batch_after_lease_is_taken_over(monkeypatch) -> None:
    session = FakeSession()
    orchestration = _make_orchestration(status="running")
    batch = _make_batch(orchestration.id, batch_index=0, status="running", attempt_count=1)
    batch.current_input_batch_id = uuid.uuid4()
    batch.current_inference_run_id = uuid.uuid4()
    batch.lease_token = uuid.uuid4()
    other_worker = uuid.uuid4()
    context = orchestration_service.orchestration_repository.BatchAttemptReconciliationContext(
        orchestration_id=orchestration.id,
        orchestration_status="running",
        orchestration_project_id=orchestration.project_id,
        orchestration_extraction_run_id=orchestration.extraction_run_id,
        batch_id=batch.id,
        batch_index=0,
        batch_status="running",
        attempt_count=1,
        lease_token=other_worker,
        input_batch_id=batch.current_input_batch_id,
        inference_run_id=batch.current_inference_run_id,
        inference_run_status=InferenceRunStatus.RUNNING.value,
        inference_run_project_id=orchestration.project_id,
        inference_run_task_type="fact_extraction",
        inference_run_input_batch_id=batch.current_input_batch_id,
        inference_run_failure_code=None,
        application_id=None,
        application_status=None,
        batch_application_id=None,
        batch_application_status=None,
        run_application_id=None,
        run_application_status=None,
    )

    run = SimpleNamespace(
        id=batch.current_inference_run_id,
        status=InferenceRunStatus.RUNNING.value,
        project_id=orchestration.project_id,
        task_type="fact_extraction",
        input_batch_id=batch.current_input_batch_id,
        failure_code=None,
    )

    async def fake_lock_state(*_args, **_kwargs):
        return orchestration, batch, run, None, None, context

    monkeypatch.setattr(orchestration_service, "_lock_batch_attempt_reconciliation_state", fake_lock_state)

    result = run_async(
        orchestration_service.reconcile_fact_extraction_batch_after_interruption(
            session,
            orchestration_id=orchestration.id,
            batch_index=0,
            worker_token=batch.lease_token,
            failure_code="fact_extraction_batch_lease_lost",
            max_batch_attempts=2,
        )
    )

    assert result.batch_status == "running"
    assert batch.lease_token != context.lease_token
    assert batch.current_inference_run_id == context.inference_run_id


def test_lock_batch_attempt_reconciliation_state_uses_ordered_lock_sequence(monkeypatch) -> None:
    session = FakeSession()
    orchestration = _make_orchestration(status="running")
    batch = _make_batch(orchestration.id, batch_index=0, status="running", attempt_count=1)
    batch.current_input_batch_id = uuid.uuid4()
    batch.current_inference_run_id = uuid.uuid4()
    batch.application_id = uuid.uuid4()
    run = SimpleNamespace(
        id=batch.current_inference_run_id,
        status=InferenceRunStatus.COMPLETED.value,
        project_id=orchestration.project_id,
        task_type="fact_extraction",
        input_batch_id=batch.current_input_batch_id,
        failure_code=None,
    )
    application = SimpleNamespace(id=batch.application_id, status="completed", inference_run_id=run.id)
    calls: list[str] = []

    async def fake_get_orchestration(*_args, **_kwargs):
        calls.append("orchestration")
        return orchestration

    async def fake_get_batch(*_args, **_kwargs):
        calls.append("batch")
        return batch

    async def fake_get_run(*_args, **_kwargs):
        calls.append("inference_run")
        return run

    async def fake_get_application(*_args, **_kwargs):
        calls.append("application")
        return application

    async def fake_get_application_by_run(*_args, **_kwargs):
        calls.append("run_application")
        return application

    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_orchestration_for_update", fake_get_orchestration)
    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_batch_for_update", fake_get_batch)
    monkeypatch.setattr(orchestration_service.inference_repository, "get_run_for_update", fake_get_run)
    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_application_for_update", fake_get_application)
    monkeypatch.setattr(
        orchestration_service.orchestration_repository,
        "get_application_by_inference_run_for_update",
        fake_get_application_by_run,
    )

    _, _, _, _, _, context = run_async(
        orchestration_service._lock_batch_attempt_reconciliation_state(
            session,
            orchestration_id=orchestration.id,
            batch_index=0,
        )
    )

    assert calls == ["orchestration", "batch", "inference_run", "application", "run_application"]
    assert context.inference_run_id == run.id
    assert context.application_id == application.id


def test_lock_batch_attempt_reconciliation_state_rejects_application_binding_conflict(monkeypatch) -> None:
    session = FakeSession()
    orchestration = _make_orchestration(status="running")
    batch = _make_batch(orchestration.id, batch_index=0, status="running", attempt_count=1)
    batch.current_input_batch_id = uuid.uuid4()
    batch.current_inference_run_id = uuid.uuid4()
    batch.application_id = uuid.uuid4()
    run = SimpleNamespace(
        id=batch.current_inference_run_id,
        status=InferenceRunStatus.COMPLETED.value,
        project_id=orchestration.project_id,
        task_type="fact_extraction",
        input_batch_id=batch.current_input_batch_id,
        failure_code=None,
    )

    async def fake_get_orchestration(*_args, **_kwargs):
        return orchestration

    async def fake_get_batch(*_args, **_kwargs):
        return batch

    async def fake_get_run(*_args, **_kwargs):
        return run

    async def fake_batch_application(*_args, **_kwargs):
        return SimpleNamespace(id=batch.application_id, status="completed", inference_run_id=run.id)

    async def fake_run_application(*_args, **_kwargs):
        return SimpleNamespace(id=uuid.uuid4(), status="completed", inference_run_id=run.id)

    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_orchestration_for_update", fake_get_orchestration)
    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_batch_for_update", fake_get_batch)
    monkeypatch.setattr(orchestration_service.inference_repository, "get_run_for_update", fake_get_run)
    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_application_for_update", fake_batch_application)
    monkeypatch.setattr(
        orchestration_service.orchestration_repository,
        "get_application_by_inference_run_for_update",
        fake_run_application,
    )

    with pytest.raises(orchestration_service.FactExtractionOrchestrationStateError, match="batch_application_binding_conflict"):
        run_async(
            orchestration_service._lock_batch_attempt_reconciliation_state(
                session,
                orchestration_id=orchestration.id,
                batch_index=0,
            )
        )


def test_lock_batch_attempt_reconciliation_state_rejects_missing_current_run(monkeypatch) -> None:
    session = FakeSession()
    orchestration = _make_orchestration(status="running")
    batch = _make_batch(orchestration.id, batch_index=0, status="running", attempt_count=1)
    batch.current_input_batch_id = uuid.uuid4()
    batch.current_inference_run_id = uuid.uuid4()

    async def fake_get_orchestration(*_args, **_kwargs):
        return orchestration

    async def fake_get_batch(*_args, **_kwargs):
        return batch

    async def fake_get_run(*_args, **_kwargs):
        return None

    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_orchestration_for_update", fake_get_orchestration)
    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_batch_for_update", fake_get_batch)
    monkeypatch.setattr(orchestration_service.inference_repository, "get_run_for_update", fake_get_run)

    with pytest.raises(orchestration_service.FactExtractionOrchestrationStateError, match="current_inference_run_missing"):
        run_async(
            orchestration_service._lock_batch_attempt_reconciliation_state(
                session,
                orchestration_id=orchestration.id,
                batch_index=0,
            )
        )


def test_lock_batch_attempt_reconciliation_state_rejects_run_project_mismatch(monkeypatch) -> None:
    session = FakeSession()
    orchestration = _make_orchestration(status="running")
    batch = _make_batch(orchestration.id, batch_index=0, status="running", attempt_count=1)
    batch.current_input_batch_id = uuid.uuid4()
    batch.current_inference_run_id = uuid.uuid4()
    run = SimpleNamespace(
        id=batch.current_inference_run_id,
        status=InferenceRunStatus.RUNNING.value,
        project_id=uuid.uuid4(),
        task_type="fact_extraction",
        input_batch_id=batch.current_input_batch_id,
        failure_code=None,
    )

    async def fake_get_orchestration(*_args, **_kwargs):
        return orchestration

    async def fake_get_batch(*_args, **_kwargs):
        return batch

    async def fake_get_run(*_args, **_kwargs):
        return run

    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_orchestration_for_update", fake_get_orchestration)
    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_batch_for_update", fake_get_batch)
    monkeypatch.setattr(orchestration_service.inference_repository, "get_run_for_update", fake_get_run)

    with pytest.raises(orchestration_service.FactExtractionOrchestrationStateError, match="project mismatch"):
        run_async(
            orchestration_service._lock_batch_attempt_reconciliation_state(
                session,
                orchestration_id=orchestration.id,
                batch_index=0,
            )
        )


def test_lock_batch_attempt_reconciliation_state_rejects_run_input_batch_mismatch(monkeypatch) -> None:
    session = FakeSession()
    orchestration = _make_orchestration(status="running")
    batch = _make_batch(orchestration.id, batch_index=0, status="running", attempt_count=1)
    batch.current_input_batch_id = uuid.uuid4()
    batch.current_inference_run_id = uuid.uuid4()
    run = SimpleNamespace(
        id=batch.current_inference_run_id,
        status=InferenceRunStatus.RUNNING.value,
        project_id=orchestration.project_id,
        task_type="fact_extraction",
        input_batch_id=uuid.uuid4(),
        failure_code=None,
    )

    async def fake_get_orchestration(*_args, **_kwargs):
        return orchestration

    async def fake_get_batch(*_args, **_kwargs):
        return batch

    async def fake_get_run(*_args, **_kwargs):
        return run

    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_orchestration_for_update", fake_get_orchestration)
    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_batch_for_update", fake_get_batch)
    monkeypatch.setattr(orchestration_service.inference_repository, "get_run_for_update", fake_get_run)

    with pytest.raises(orchestration_service.FactExtractionOrchestrationStateError, match="input_batch_id mismatch"):
        run_async(
            orchestration_service._lock_batch_attempt_reconciliation_state(
                session,
                orchestration_id=orchestration.id,
                batch_index=0,
            )
        )


def test_record_prepared_run_notice_rejects_failed_run_registration(monkeypatch) -> None:
    session = FakeSession()
    orchestration = _make_orchestration(status="running")
    batch = _make_batch(orchestration.id, batch_index=0, status="running", attempt_count=1)
    batch.lease_token = uuid.uuid4()
    notice = orchestration_service.PreparedFactExtractionRunNotice(
        project_id=orchestration.project_id,
        extraction_run_id=orchestration.extraction_run_id,
        plan_hash=orchestration.plan_hash,
        batch_index=0,
        batch_plan_hash=batch.batch_plan_hash,
        input_batch_id=uuid.uuid4(),
        inference_run_id=uuid.uuid4(),
        inference_request_hash="a" * 64,
    )
    registration_context = orchestration_service.inference_repository.PreparedInferenceRunRegistrationContext(
        inference_run_id=notice.inference_run_id,
        input_batch_id=notice.input_batch_id,
        project_id=orchestration.project_id,
        task_type="fact_extraction",
        status=InferenceRunStatus.FAILED.value,
        inference_request_hash=notice.inference_request_hash,
        agent_name=orchestration.agent_name,
        agent_version=orchestration.agent_version,
        prompt_name=orchestration.prompt_name,
        prompt_version=orchestration.prompt_version,
        prompt_contract_hash=orchestration.prompt_contract_hash,
        provider=orchestration.provider,
        requested_model=orchestration.requested_model,
        batch_project_id=orchestration.project_id,
        batch_task_type="fact_extraction",
        request_metadata={
            "extraction_run_id": str(orchestration.extraction_run_id),
            "plan_hash": orchestration.plan_hash,
            "batch_index": 0,
            "batch_plan_hash": batch.batch_plan_hash,
            "executor_name": orchestration.executor_name,
            "executor_version": orchestration.executor_version,
            "planner_name": orchestration.planner_name,
            "planner_version": orchestration.planner_version,
            "prompt_contract_hash": orchestration.prompt_contract_hash,
        },
    )

    async def fake_get_orchestration(*_args, **_kwargs):
        return orchestration

    async def fake_get_batch(*_args, **_kwargs):
        return batch

    async def fake_registration(*_args, **_kwargs):
        return registration_context

    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_orchestration_for_update", fake_get_orchestration)
    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_batch_for_update", fake_get_batch)
    monkeypatch.setattr(
        orchestration_service.inference_repository,
        "get_prepared_inference_run_registration_context",
        fake_registration,
    )

    with pytest.raises(orchestration_service.PreparedInferenceRunRegistrationError, match="status"):
        run_async(
            orchestration_service._record_prepared_run_notice(
                session,
                orchestration_id=orchestration.id,
                worker_token=batch.lease_token,
                notice=notice,
            )
        )


def test_record_prepared_run_notice_rejects_non_running_orchestration(monkeypatch) -> None:
    session = FakeSession()
    orchestration = _make_orchestration(status="failed")
    batch = _make_batch(orchestration.id, batch_index=0, status="running", attempt_count=1)
    batch.lease_token = uuid.uuid4()
    notice = orchestration_service.PreparedFactExtractionRunNotice(
        project_id=orchestration.project_id,
        extraction_run_id=orchestration.extraction_run_id,
        plan_hash=orchestration.plan_hash,
        batch_index=0,
        batch_plan_hash=batch.batch_plan_hash,
        input_batch_id=uuid.uuid4(),
        inference_run_id=uuid.uuid4(),
        inference_request_hash="a" * 64,
    )
    registration_context = orchestration_service.inference_repository.PreparedInferenceRunRegistrationContext(
        inference_run_id=notice.inference_run_id,
        input_batch_id=notice.input_batch_id,
        project_id=orchestration.project_id,
        task_type="fact_extraction",
        status=InferenceRunStatus.PENDING.value,
        inference_request_hash=notice.inference_request_hash,
        agent_name=orchestration.agent_name,
        agent_version=orchestration.agent_version,
        prompt_name=orchestration.prompt_name,
        prompt_version=orchestration.prompt_version,
        prompt_contract_hash=orchestration.prompt_contract_hash,
        provider=orchestration.provider,
        requested_model=orchestration.requested_model,
        batch_project_id=orchestration.project_id,
        batch_task_type="fact_extraction",
        request_metadata={
            "extraction_run_id": str(orchestration.extraction_run_id),
            "plan_hash": orchestration.plan_hash,
            "batch_index": 0,
            "batch_plan_hash": batch.batch_plan_hash,
            "executor_name": orchestration.executor_name,
            "executor_version": orchestration.executor_version,
            "planner_name": orchestration.planner_name,
            "planner_version": orchestration.planner_version,
            "prompt_contract_hash": orchestration.prompt_contract_hash,
        },
    )

    async def fake_get_orchestration(*_args, **_kwargs):
        return orchestration

    async def fake_get_batch(*_args, **_kwargs):
        return batch

    async def fake_registration(*_args, **_kwargs):
        return registration_context

    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_orchestration_for_update", fake_get_orchestration)
    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_batch_for_update", fake_get_batch)
    monkeypatch.setattr(
        orchestration_service.inference_repository,
        "get_prepared_inference_run_registration_context",
        fake_registration,
    )

    with pytest.raises(orchestration_service.PreparedInferenceRunRegistrationError, match="orchestration"):
        run_async(
            orchestration_service._record_prepared_run_notice(
                session,
                orchestration_id=orchestration.id,
                worker_token=batch.lease_token,
                notice=notice,
            )
        )


def test_classify_lease_lost_uses_dedicated_failure_code() -> None:
    error = orchestration_service.FactExtractionBatchLeaseLostError("lost")
    assert orchestration_service._classify_batch_failure(error) == "fact_extraction_batch_lease_lost"


def test_terminal_validation_is_shared_by_read_and_finalize(monkeypatch) -> None:
    session = FakeSession()
    orchestration = _make_orchestration(status="completed")
    orchestration.completed_batch_count = orchestration.batch_count
    batch = _make_batch(orchestration.id, batch_index=0, status="completed")

    async def fake_get_orchestration(*_args, **_kwargs):
        return orchestration

    async def fake_list_batches(*_args, **_kwargs):
        return [batch, _make_batch(orchestration.id, batch_index=1, status="completed"), _make_batch(orchestration.id, batch_index=2, status="completed")]

    async def fake_list_applications(*_args, **_kwargs):
        return []

    def sentinel(**_kwargs):
        raise orchestration_service.FactExtractionOrchestrationStateError("shared-terminal-validator")

    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_orchestration_for_update", fake_get_orchestration)
    monkeypatch.setattr(orchestration_service.orchestration_repository, "list_batches_for_orchestration_for_update", fake_list_batches)
    monkeypatch.setattr(orchestration_service.orchestration_repository, "list_applications", fake_list_applications)
    monkeypatch.setattr(orchestration_service, "_load_authenticated_completed_applications", lambda **_kwargs: {})
    monkeypatch.setattr(orchestration_service, "validate_terminal_orchestration_state", sentinel)

    with pytest.raises(orchestration_service.FactExtractionOrchestrationStateError, match="shared-terminal-validator"):
        run_async(orchestration_service._read_completed_orchestration_result(session, orchestration_id=orchestration.id))

    with pytest.raises(orchestration_service.FactExtractionOrchestrationStateError, match="shared-terminal-validator"):
        run_async(orchestration_service._finalize_orchestration(session, orchestration_id=orchestration.id))


def test_authenticate_terminal_fact_extraction_orchestration_rolls_back_without_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = SessionFactory()
    orchestration = _make_orchestration(status="failed")
    orchestration.completed_batch_count = 0
    orchestration.failed_batch_count = orchestration.batch_count
    batches = [
        _make_batch(orchestration.id, batch_index=0, status="failed", attempt_count=1),
        _make_batch(orchestration.id, batch_index=1, status="failed", attempt_count=1),
        _make_batch(orchestration.id, batch_index=2, status="failed", attempt_count=1),
    ]

    async def fake_get_orchestration(*_args, **_kwargs):
        return orchestration

    async def fake_list_batches(*_args, **_kwargs):
        return batches

    async def fake_list_applications(*_args, **_kwargs):
        return []

    monkeypatch.setattr(
        orchestration_service.orchestration_repository,
        "get_orchestration",
        fake_get_orchestration,
    )
    monkeypatch.setattr(
        orchestration_service.orchestration_repository,
        "list_batches_for_orchestration",
        fake_list_batches,
    )
    monkeypatch.setattr(
        orchestration_service.orchestration_repository,
        "list_applications",
        fake_list_applications,
    )
    monkeypatch.setattr(
        orchestration_service,
        "_load_authenticated_completed_applications",
        lambda **_kwargs: {},
    )

    result = run_async(
        orchestration_service.authenticate_terminal_fact_extraction_orchestration(
            factory,
            project_id=orchestration.project_id,
            extraction_run_id=orchestration.extraction_run_id,
            orchestration_id=orchestration.id,
        )
    )

    assert result.orchestration_id == orchestration.id
    assert result.status == FactExtractionOrchestrationStatus.FAILED
    assert factory.open_count == 0
    assert len(factory.sessions) == 1
    assert factory.sessions[0].commit_count == 0
    assert factory.sessions[0].rollback_count == 1


def test_finalize_batch_from_completed_application_locks_run_before_application(monkeypatch) -> None:
    session = FakeSession()
    orchestration = _make_orchestration(status="running")
    batch = _make_batch(orchestration.id, batch_index=0, status="running", attempt_count=1)
    batch.current_input_batch_id = uuid.uuid4()
    batch.current_inference_run_id = uuid.uuid4()
    batch.application_id = uuid.uuid4()
    batch.lease_token = uuid.uuid4()
    run = SimpleNamespace(
        id=batch.current_inference_run_id,
        status=InferenceRunStatus.COMPLETED.value,
        project_id=orchestration.project_id,
        task_type="fact_extraction",
        input_batch_id=batch.current_input_batch_id,
        failure_code=None,
    )
    application = SimpleNamespace(
        id=batch.application_id,
        status="completed",
        project_id=orchestration.project_id,
        extraction_run_id=orchestration.extraction_run_id,
        inference_run_id=run.id,
        input_batch_id=batch.current_input_batch_id,
    )
    ledger = SimpleNamespace(
        proposal_count=1,
        created_count=1,
        reused_count=0,
        withheld_count=0,
    )
    calls: list[str] = []

    async def fake_get_orchestration(*_args, **_kwargs):
        calls.append("orchestration")
        return orchestration

    async def fake_get_batch(*_args, **_kwargs):
        calls.append("batch")
        return batch

    async def fake_get_run(*_args, **_kwargs):
        calls.append("run")
        return run

    async def fake_get_application(*_args, **_kwargs):
        calls.append("application")
        return application

    async def fake_get_application_by_run(*_args, **_kwargs):
        calls.append("run_application")
        return application

    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_orchestration_for_update", fake_get_orchestration)
    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_batch_for_update", fake_get_batch)
    monkeypatch.setattr(orchestration_service.inference_repository, "get_run_for_update", fake_get_run)
    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_application_for_update", fake_get_application)
    monkeypatch.setattr(
        orchestration_service.orchestration_repository,
        "get_application_by_inference_run_for_update",
        fake_get_application_by_run,
    )
    monkeypatch.setattr(
        orchestration_service,
        "validate_fact_extraction_application_result_envelope",
        lambda *, application: ledger,
    )

    result = run_async(
        orchestration_service.finalize_batch_from_completed_application(
            session,
            orchestration_id=orchestration.id,
            batch_index=0,
            worker_token=batch.lease_token,
            inference_run_id=run.id,
            application_id=application.id,
        )
    )

    assert result.status == FactExtractionOrchestrationBatchStatus.COMPLETED
    assert calls.index("run") < calls.index("application")


def test_finalize_batch_success_locks_run_before_application(monkeypatch) -> None:
    session = FakeSession()
    orchestration = _make_orchestration(status="running")
    batch = _make_batch(orchestration.id, batch_index=0, status="running", attempt_count=1)
    batch.current_input_batch_id = uuid.uuid4()
    batch.current_inference_run_id = uuid.uuid4()
    batch.lease_token = uuid.uuid4()
    run = SimpleNamespace(
        id=batch.current_inference_run_id,
        status=InferenceRunStatus.COMPLETED.value,
        project_id=orchestration.project_id,
        task_type="fact_extraction",
        input_batch_id=batch.current_input_batch_id,
    )
    application = SimpleNamespace(
        id=uuid.uuid4(),
        status="completed",
        project_id=orchestration.project_id,
        extraction_run_id=orchestration.extraction_run_id,
        inference_run_id=run.id,
        input_batch_id=batch.current_input_batch_id,
    )
    calls: list[str] = []

    async def fake_get_orchestration(*_args, **_kwargs):
        calls.append("orchestration")
        return orchestration

    async def fake_get_batch(*_args, **_kwargs):
        calls.append("batch")
        return batch

    async def fake_get_run(*_args, **_kwargs):
        calls.append("run")
        return run

    async def fake_get_application(*_args, **_kwargs):
        calls.append("application")
        return application

    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_orchestration_for_update", fake_get_orchestration)
    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_batch_for_update", fake_get_batch)
    monkeypatch.setattr(orchestration_service.inference_repository, "get_run_for_update", fake_get_run)
    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_application_for_update", fake_get_application)

    persistence_result = SimpleNamespace(
        application_id=application.id,
        replayed_application=False,
        model_dump=lambda mode="json": {
            "application_id": str(application.id),
            "project_id": str(orchestration.project_id),
        },
    )
    ledger_result = SimpleNamespace(
        application_id=application.id,
        inference_run_id=run.id,
        input_batch_id=batch.current_input_batch_id,
        project_id=orchestration.project_id,
        extraction_run_id=orchestration.extraction_run_id,
        proposal_count=0,
        created_count=0,
        reused_count=0,
        withheld_count=0,
        model_dump=lambda mode="json": {
            "application_id": str(application.id),
            "project_id": str(orchestration.project_id),
        },
    )
    execution_result = SimpleNamespace(
        batch_index=0,
        batch_plan_hash=batch.batch_plan_hash,
        inference_run_id=run.id,
        input_batch_id=batch.current_input_batch_id,
        project_id=orchestration.project_id,
        extraction_run_id=orchestration.extraction_run_id,
    )

    monkeypatch.setattr(
        orchestration_service,
        "validate_fact_extraction_application_result_envelope",
        lambda *, application: ledger_result,
    )

    run_async(
        orchestration_service._finalize_batch_success(
            session,
            orchestration_id=orchestration.id,
            worker_token=batch.lease_token,
            execution_result=execution_result,
            persistence_result=persistence_result,
        )
    )

    assert calls.index("run") < calls.index("application")


@pytest.mark.parametrize(
    ("status", "batch_statuses"),
    [
        ("completed", ["completed", "pending", "completed"]),
        ("partial", ["completed", "running", "failed"]),
        ("failed", ["completed", "failed", "failed"]),
    ],
)
def test_terminal_orchestration_state_mismatch_rejects_without_mutation(
    monkeypatch,
    status: str,
    batch_statuses: list[str],
) -> None:
    session = FakeSession()
    orchestration = _make_orchestration(status=status)
    original_completed_at = orchestration.completed_at
    original_counts = (
        orchestration.completed_batch_count,
        orchestration.failed_batch_count,
        orchestration.proposal_count,
        orchestration.created_count,
        orchestration.reused_count,
        orchestration.withheld_count,
    )
    batches = [
        _make_batch(orchestration.id, batch_index=index, status=batch_status)
        for index, batch_status in enumerate(batch_statuses)
    ]

    async def fake_get_orchestration(*_args, **_kwargs):
        return orchestration

    async def fake_list_batches(*_args, **_kwargs):
        return batches

    async def fake_list_applications(*_args, **_kwargs):
        return []

    monkeypatch.setattr(orchestration_service.orchestration_repository, "get_orchestration_for_update", fake_get_orchestration)
    monkeypatch.setattr(orchestration_service.orchestration_repository, "list_batches_for_orchestration_for_update", fake_list_batches)
    monkeypatch.setattr(orchestration_service.orchestration_repository, "list_applications", fake_list_applications)
    monkeypatch.setattr(orchestration_service, "_load_authenticated_completed_applications", lambda **_kwargs: {})

    with pytest.raises(orchestration_service.FactExtractionOrchestrationStateError):
        run_async(orchestration_service._finalize_orchestration(session, orchestration_id=orchestration.id))

    assert orchestration.completed_at == original_completed_at
    assert (
        orchestration.completed_batch_count,
        orchestration.failed_batch_count,
        orchestration.proposal_count,
        orchestration.created_count,
        orchestration.reused_count,
        orchestration.withheld_count,
    ) == original_counts
    assert session.rollback_count == 1


def test_finalize_batch_failure_is_only_a_reconciliation_wrapper(monkeypatch) -> None:
    session = FakeSession()
    calls = []

    async def fake_reconcile(*_args, **_kwargs):
        calls.append(_kwargs)
        return orchestration_service.BatchInterruptionReconciliation(
            reconciliation_status=None,
            batch_status="failed",
            attempt_count=1,
            input_batch_id=None,
            inference_run_id=None,
            application_id=None,
            failure_code="llm_transport_error",
        )

    monkeypatch.setattr(orchestration_service, "reconcile_fact_extraction_batch_after_interruption", fake_reconcile)

    run_async(
        orchestration_service._finalize_batch_failure(
            session,
            orchestration_id=uuid.uuid4(),
            batch_index=0,
            worker_token=uuid.uuid4(),
            failure_code="llm_transport_error",
            max_batch_attempts=2,
        )
    )

    assert len(calls) == 1
    assert calls[0]["failure_code"] == "llm_transport_error"


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


@pytest.mark.parametrize(
    ("status", "expected_stage_calls"),
    [
        (
            FactExtractionOrchestrationStatus.COMPLETED,
            ("duplicate_grouping", "consistency_candidates"),
        ),
        (
            FactExtractionOrchestrationStatus.PARTIAL,
            ("duplicate_grouping", "consistency_candidates"),
        ),
    ],
)
def test_terminal_consistency_postprocessing_calls_grouping_then_candidates(
    monkeypatch,
    status: FactExtractionOrchestrationStatus,
    expected_stage_calls: tuple[str, str],
) -> None:
    session_factory = SessionFactory()
    orchestration_result = _make_orchestration_result(status=status)
    grouping_application_id = uuid.uuid4()
    calls: list[tuple[str, uuid.UUID]] = []

    async def fake_grouping(_session_factory, *, orchestration_id, algorithm_version="cross_batch_exact_v2"):
        assert algorithm_version == "cross_batch_exact_v2"
        calls.append(("duplicate_grouping", orchestration_id))
        return duplicate_grouping_service.DuplicateGroupingResult(
            grouping_application_id=grouping_application_id,
            algorithm_version=algorithm_version,
            input_fact_value_count=2,
            duplicate_group_count=0,
            duplicate_member_count=0,
            created_new=(status == FactExtractionOrchestrationStatus.COMPLETED),
        )

    async def fake_candidates(_session_factory, *, duplicate_grouping_application_id, algorithm_version="cross_batch_multi_value_v1"):
        assert algorithm_version == "cross_batch_multi_value_v1"
        calls.append(("consistency_candidates", duplicate_grouping_application_id))
        return SimpleNamespace(created_new=True)

    monkeypatch.setattr(
        orchestration_service.duplicate_grouping_service,
        "ensure_cross_batch_duplicate_grouping",
        fake_grouping,
    )
    monkeypatch.setattr(
        orchestration_service.duplicate_grouping_service,
        "ensure_cross_batch_multi_value_consistency_candidates",
        fake_candidates,
    )

    result = run_async(
        orchestration_service._maybe_run_terminal_consistency_postprocessing(
            session_factory,
            orchestration_result=orchestration_result,
        )
    )

    assert result == orchestration_result
    assert calls == [
        ("duplicate_grouping", orchestration_result.orchestration_id),
        ("consistency_candidates", grouping_application_id),
    ]
    assert tuple(item[0] for item in calls) == expected_stage_calls


def test_terminal_consistency_postprocessing_skips_non_terminal_failure_status(
    monkeypatch,
) -> None:
    session_factory = SessionFactory()
    orchestration_result = _make_orchestration_result(status=FactExtractionOrchestrationStatus.FAILED)
    calls: list[str] = []

    async def fake_grouping(*_args, **_kwargs):
        calls.append("grouping")
        return None

    async def fake_candidates(*_args, **_kwargs):
        calls.append("candidate")
        return None

    monkeypatch.setattr(
        orchestration_service.duplicate_grouping_service,
        "ensure_cross_batch_duplicate_grouping",
        fake_grouping,
    )
    monkeypatch.setattr(
        orchestration_service.duplicate_grouping_service,
        "ensure_cross_batch_multi_value_consistency_candidates",
        fake_candidates,
    )

    result = run_async(
        orchestration_service._maybe_run_terminal_consistency_postprocessing(
            session_factory,
            orchestration_result=orchestration_result,
        )
    )

    assert result == orchestration_result
    assert calls == []


def test_terminal_consistency_postprocessing_grouping_idempotent_hit_still_runs_candidates(
    monkeypatch,
) -> None:
    session_factory = SessionFactory()
    orchestration_result = _make_orchestration_result(status=FactExtractionOrchestrationStatus.COMPLETED)
    grouping_application_id = uuid.uuid4()
    calls: list[str] = []

    async def fake_grouping(_session_factory, *, orchestration_id, algorithm_version="cross_batch_exact_v2"):
        calls.append("grouping")
        return duplicate_grouping_service.DuplicateGroupingResult(
            grouping_application_id=grouping_application_id,
            algorithm_version=algorithm_version,
            input_fact_value_count=2,
            duplicate_group_count=1,
            duplicate_member_count=2,
            created_new=False,
        )

    async def fake_candidates(_session_factory, *, duplicate_grouping_application_id, algorithm_version="cross_batch_multi_value_v1"):
        assert duplicate_grouping_application_id == grouping_application_id
        calls.append("candidate")
        return SimpleNamespace(created_new=True)

    monkeypatch.setattr(
        orchestration_service.duplicate_grouping_service,
        "ensure_cross_batch_duplicate_grouping",
        fake_grouping,
    )
    monkeypatch.setattr(
        orchestration_service.duplicate_grouping_service,
        "ensure_cross_batch_multi_value_consistency_candidates",
        fake_candidates,
    )

    result = run_async(
        orchestration_service._maybe_run_terminal_consistency_postprocessing(
            session_factory,
            orchestration_result=orchestration_result,
        )
    )

    assert result == orchestration_result
    assert calls == ["grouping", "candidate"]


def test_terminal_consistency_postprocessing_grouping_failure_keeps_terminal_result_and_skips_candidates(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session_factory = SessionFactory()
    orchestration_result = _make_orchestration_result(status=FactExtractionOrchestrationStatus.PARTIAL)
    sentinel = "SENSITIVE_DUPLICATE_GROUPING_SENTINEL"
    candidate_called = False

    async def fake_grouping(_session_factory, *, orchestration_id, algorithm_version="cross_batch_exact_v2"):
        raise duplicate_grouping_service.CrossBatchDuplicateGroupingInvariantError(sentinel)

    async def fake_candidates(_session_factory, *, duplicate_grouping_application_id, algorithm_version="cross_batch_multi_value_v1"):
        nonlocal candidate_called
        candidate_called = True
        return SimpleNamespace(created_new=True)

    monkeypatch.setattr(
        orchestration_service.duplicate_grouping_service,
        "ensure_cross_batch_duplicate_grouping",
        fake_grouping,
    )
    monkeypatch.setattr(
        orchestration_service.duplicate_grouping_service,
        "ensure_cross_batch_multi_value_consistency_candidates",
        fake_candidates,
    )

    result = run_async(
        orchestration_service._maybe_run_terminal_consistency_postprocessing(
            session_factory,
            orchestration_result=orchestration_result,
        )
    )

    assert result == orchestration_result
    assert candidate_called is False
    assert "Terminal orchestration consistency postprocessing step failed" in caplog.text
    assert any(getattr(record, "stage", None) == "duplicate_grouping" for record in caplog.records)
    assert any(getattr(record, "error_type", None) == "CrossBatchDuplicateGroupingInvariantError" for record in caplog.records)
    assert sentinel not in caplog.text


def test_terminal_consistency_postprocessing_candidate_failure_keeps_terminal_result(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session_factory = SessionFactory()
    orchestration_result = _make_orchestration_result(status=FactExtractionOrchestrationStatus.COMPLETED)
    grouping_application_id = uuid.uuid4()
    sentinel = "SENSITIVE_CONSISTENCY_CANDIDATE_SENTINEL"

    async def fake_grouping(_session_factory, *, orchestration_id, algorithm_version="cross_batch_exact_v2"):
        return duplicate_grouping_service.DuplicateGroupingResult(
            grouping_application_id=grouping_application_id,
            algorithm_version=algorithm_version,
            input_fact_value_count=2,
            duplicate_group_count=1,
            duplicate_member_count=2,
            created_new=True,
        )

    async def fake_candidates(_session_factory, *, duplicate_grouping_application_id, algorithm_version="cross_batch_multi_value_v1"):
        raise duplicate_grouping_service.FactValueConsistencyCandidateInvariantError(sentinel)

    monkeypatch.setattr(
        orchestration_service.duplicate_grouping_service,
        "ensure_cross_batch_duplicate_grouping",
        fake_grouping,
    )
    monkeypatch.setattr(
        orchestration_service.duplicate_grouping_service,
        "ensure_cross_batch_multi_value_consistency_candidates",
        fake_candidates,
    )

    result = run_async(
        orchestration_service._maybe_run_terminal_consistency_postprocessing(
            session_factory,
            orchestration_result=orchestration_result,
        )
    )

    assert result == orchestration_result
    assert "Terminal orchestration consistency postprocessing step failed" in caplog.text
    assert any(getattr(record, "stage", None) == "consistency_candidates" for record in caplog.records)
    assert any(
        getattr(record, "grouping_application_id", None) == str(grouping_application_id)
        for record in caplog.records
    )
    assert any(
        getattr(record, "error_type", None) == "FactValueConsistencyCandidateInvariantError"
        for record in caplog.records
    )
    assert sentinel not in caplog.text


def test_terminal_consistency_postprocessing_grouping_cancelled_error_propagates(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session_factory = SessionFactory()
    orchestration_result = _make_orchestration_result(status=FactExtractionOrchestrationStatus.COMPLETED)

    async def fake_grouping(_session_factory, *, orchestration_id, algorithm_version="cross_batch_exact_v2"):
        raise asyncio.CancelledError()

    monkeypatch.setattr(
        orchestration_service.duplicate_grouping_service,
        "ensure_cross_batch_duplicate_grouping",
        fake_grouping,
    )

    with pytest.raises(asyncio.CancelledError):
        run_async(
            orchestration_service._maybe_run_terminal_consistency_postprocessing(
                session_factory,
                orchestration_result=orchestration_result,
            )
        )

    assert "Terminal orchestration consistency postprocessing step failed" not in caplog.text


def test_terminal_consistency_postprocessing_candidate_cancelled_error_propagates(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session_factory = SessionFactory()
    orchestration_result = _make_orchestration_result(status=FactExtractionOrchestrationStatus.COMPLETED)
    grouping_application_id = uuid.uuid4()

    async def fake_grouping(_session_factory, *, orchestration_id, algorithm_version="cross_batch_exact_v2"):
        return duplicate_grouping_service.DuplicateGroupingResult(
            grouping_application_id=grouping_application_id,
            algorithm_version=algorithm_version,
            input_fact_value_count=2,
            duplicate_group_count=1,
            duplicate_member_count=2,
            created_new=True,
        )

    async def fake_candidates(_session_factory, *, duplicate_grouping_application_id, algorithm_version="cross_batch_multi_value_v1"):
        raise asyncio.CancelledError()

    monkeypatch.setattr(
        orchestration_service.duplicate_grouping_service,
        "ensure_cross_batch_duplicate_grouping",
        fake_grouping,
    )
    monkeypatch.setattr(
        orchestration_service.duplicate_grouping_service,
        "ensure_cross_batch_multi_value_consistency_candidates",
        fake_candidates,
    )

    with pytest.raises(asyncio.CancelledError):
        run_async(
            orchestration_service._maybe_run_terminal_consistency_postprocessing(
                session_factory,
                orchestration_result=orchestration_result,
            )
        )

    assert "Terminal orchestration consistency postprocessing step failed" not in caplog.text


@pytest.mark.parametrize("reused_completed", [True, False])
def test_execute_orchestration_reused_completed_and_finalize_share_terminal_postprocessing_entry(
    monkeypatch,
    reused_completed: bool,
) -> None:
    session_factory = SessionFactory()
    extraction_run_id, plan = _planned_fixture()
    expected_result = _make_orchestration_result(status=FactExtractionOrchestrationStatus.COMPLETED)
    postprocess_calls: list[uuid.UUID] = []

    async def fake_prepare(*_args, **_kwargs):
        return orchestration_service.PreparedFactExtractionOrchestration(
            orchestration_id=expected_result.orchestration_id,
            attempt_no=1,
            request_hash=expected_result.request_hash,
            plan_hash=expected_result.plan_hash,
            reused_completed=reused_completed,
        )

    async def fake_postprocess(_session_factory, *, orchestration_result):
        postprocess_calls.append(orchestration_result.orchestration_id)
        return orchestration_result

    monkeypatch.setattr(orchestration_service, "prepare_fact_extraction_orchestration", fake_prepare)
    monkeypatch.setattr(
        orchestration_service,
        "_maybe_run_terminal_consistency_postprocessing",
        fake_postprocess,
    )

    if reused_completed:
        async def fake_read_completed(*_args, **_kwargs):
            return expected_result

        monkeypatch.setattr(orchestration_service, "_read_completed_orchestration_result", fake_read_completed)
    else:
        async def fake_list_batches(*_args, **_kwargs):
            return []

        async def fake_finalize(*_args, **_kwargs):
            return expected_result

        monkeypatch.setattr(
            orchestration_service.orchestration_repository,
            "list_batches_for_orchestration",
            fake_list_batches,
        )
        monkeypatch.setattr(orchestration_service, "_finalize_orchestration", fake_finalize)

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

    assert result == expected_result
    assert postprocess_calls == [expected_result.orchestration_id]


def test_orchestration_migration_is_latest_head_and_declares_tables() -> None:
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    migration = Path("alembic/versions/202607312230_orchestration_recovery_hardening.py").read_text(encoding="utf-8")
    assert "fact_extraction_orchestrations" in migration
    assert "fact_extraction_orch_batches" in migration
    assert "feo_terminal_batch_counts_within_batch_count" in migration
    historical_migration = Path("alembic/versions/202607312200_fact_extraction_orchestration.py").read_text(encoding="utf-8")
    assert "feo_terminal_batch_counts_within_batch_count" not in historical_migration
    assert "completed_batch_count + failed_batch_count = batch_count" not in historical_migration
    assert historical_migration.count("feo_partial_shape") == 1
    assert historical_migration.count("feo_failed_shape") == 1
    assert historical_migration.count("feob_pending_shape") == 1
    assert '"feo_terminal_batch_counts_within_batch_count",' in migration
    assert "DO $$" in migration
    assert "op.get_bind()" not in migration
    assert 'op.drop_constraint("feo_partial_shape"' in migration
    assert 'op.drop_constraint("feo_failed_shape"' in migration
    assert 'op.drop_constraint("feob_pending_shape"' in migration

    root = Path(__file__).resolve().parents[1]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    script = ScriptDirectory.from_config(config)
    assert list(script.get_heads()) == ["202608010500"]
