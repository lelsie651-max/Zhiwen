from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from pathlib import Path

import pytest

from app.agents.fact_extraction_planner import plan_fact_extraction_batches
from app.agents.prompt_registry import get_prompt
from app.models.base import utc_now
from app.models.document_content import DocumentBlock
from app.models.inference import InferenceInputBatch, InferenceInputBlock, InferenceRun, InferenceRunStatus
from app.schemas.agent_fact_extraction import FactExtractionResponse
from app.services import fact_extraction_execution as execution_service
from app.services.inference import InferenceRunClaim, PreparedInferenceRun
from app.services.llm import MockLLMClient, make_stub_completion


PROMPT = get_prompt("agent1_fact_extraction", "1.0.0")


def run_async(awaitable):
    return asyncio.run(awaitable)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FakeSession:
    def __init__(self, factory: "SessionFactory"):
        self.factory = factory
        self.commit_count = 0
        self.rollback_count = 0
        self.flush_count = 0

    async def commit(self):
        self.commit_count += 1

    async def rollback(self):
        self.rollback_count += 1

    async def flush(self):
        self.flush_count += 1


class SessionFactory:
    def __init__(self):
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


def _source_block(
    *,
    source_order: int,
    raw_text: str,
    extraction_run_id: uuid.UUID,
) -> DocumentBlock:
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
    project_id = uuid.uuid4()
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
    ]
    plan = plan_fact_extraction_batches(
        extraction_run_id=extraction_run_id,
        blocks=blocks,
        prompt=PROMPT,
    )
    batch_plan = plan.batches[0]
    return project_id, extraction_run_id, blocks, plan, batch_plan


def _materialized_orm_batch(
    *,
    project_id: uuid.UUID,
    extraction_run_id: uuid.UUID,
    blocks: list[DocumentBlock],
    batch_plan,
) -> InferenceInputBatch:
    batch = InferenceInputBatch(
        id=uuid.uuid4(),
        project_id=project_id,
        task_type="fact_extraction",
        selection_strategy="deterministic_fact_block_planner",
        selection_metadata={},
        block_count=len(batch_plan.block_ids),
        character_count=sum(len(block.raw_text) for block in blocks),
        snapshot_hash="c" * 64,
    )
    by_id = {block.id: block for block in blocks}
    batch.blocks = []
    for source_order, (block_id, block_ref) in enumerate(
        zip(batch_plan.block_ids, batch_plan.block_refs, strict=True)
    ):
        block = by_id[block_id]
        batch.blocks.append(
            InferenceInputBlock(
                id=uuid.uuid4(),
                batch_id=batch.id,
                source_order=source_order,
                block_ref=block_ref,
                document_block_id=block.id,
                source_block_id_snapshot=block.id,
                extraction_run_id_snapshot=extraction_run_id,
                block_type=block.block_type,
                location_key=block.location_key,
                anchor_hash=block.anchor_hash,
                page_no=block.page_no,
                start_line=block.start_line,
                end_line=block.end_line,
                heading_path=list(block.heading_path),
                content_text=block.raw_text,
                content_hash=sha256(block.raw_text),
            )
        )
    return batch


def _run(
    *,
    project_id: uuid.UUID,
    input_batch_id: uuid.UUID,
    request_hash: str,
    status: str,
    response_json: dict | None = None,
) -> InferenceRun:
    run = InferenceRun(
        id=uuid.uuid4(),
        project_id=project_id,
        input_batch_id=input_batch_id,
        task_type="fact_extraction",
        attempt_no=1,
        status=status,
        agent_name=PROMPT.agent_name,
        agent_version=PROMPT.agent_version,
        prompt_name=PROMPT.prompt_name,
        prompt_version=PROMPT.prompt_version,
        prompt_contract_hash=PROMPT.contract_hash,
        request_hash=request_hash,
        request_metadata={},
        provider="deepseek",
        requested_model="deepseek-v4-flash",
        temperature=PROMPT.temperature,
        max_output_tokens=PROMPT.max_output_tokens,
        attempt_count=0,
    )
    if status == InferenceRunStatus.RUNNING.value:
        run.started_at = utc_now()
    if status == InferenceRunStatus.COMPLETED.value:
        run.started_at = utc_now()
        run.completed_at = utc_now()
        run.finish_reason = "stop"
        run.response_model = "deepseek-v4-flash"
        run.attempt_count = 1
        run.response_json = response_json
        run.response_hash = sha256(json.dumps(response_json, sort_keys=True, ensure_ascii=False))
    return run


def _valid_response_payload():
    return {
        "facts": [
            {
                "subject_kind": "capital",
                "subject_key": "rose_kingdom",
                "predicate_key": "name",
                "value_type": "string",
                "value_json": "白蔷城",
                "confidence": 0.9,
                "evidence": [
                    {
                        "block_ref": "B0001",
                        "start_offset": 0,
                        "end_offset": 4,
                        "role": "supporting",
                    }
                ],
            }
        ],
        "batch_summary": "ok",
        "uncertainties": [],
    }


def test_execute_fact_extraction_batch_success(monkeypatch):
    session_factory = SessionFactory()
    project_id, extraction_run_id, blocks, plan, batch_plan = _planned_fixture()
    orm_batch = _materialized_orm_batch(
        project_id=project_id,
        extraction_run_id=extraction_run_id,
        blocks=blocks,
        batch_plan=batch_plan,
    )
    request_hash = sha256("req")
    pending_run = _run(
        project_id=project_id,
        input_batch_id=orm_batch.id,
        request_hash=request_hash,
        status=InferenceRunStatus.PENDING.value,
    )
    completed_run = _run(
        project_id=project_id,
        input_batch_id=orm_batch.id,
        request_hash=request_hash,
        status=InferenceRunStatus.COMPLETED.value,
        response_json=_valid_response_payload(),
    )
    completed_run.prompt_tokens = 12
    completed_run.completion_tokens = 34
    completed_run.total_tokens = 46

    monkeypatch.setattr(
        execution_service,
        "create_inference_input_batch",
        async_lambda(orm_batch),
    )
    monkeypatch.setattr(
        execution_service.inference_repository,
        "get_batch_by_identity",
        async_lambda(orm_batch),
    )
    monkeypatch.setattr(
        execution_service,
        "prepare_inference_run",
        async_lambda(PreparedInferenceRun(run=pending_run, created=True, reused_completed=False)),
    )
    monkeypatch.setattr(
        execution_service,
        "claim_inference_run_for_execution",
        async_lambda(
            InferenceRunClaim(
                run_id=pending_run.id,
                status=InferenceRunStatus.RUNNING.value,
                claimed=True,
            )
        ),
    )

    async def fake_complete(_session, *, run_id, completion):
        assert run_id == pending_run.id
        assert completion.finish_reason == "stop"
        return completed_run

    monkeypatch.setattr(execution_service, "complete_inference_run", fake_complete)

    def handler(messages):
        assert session_factory.open_count == 0
        assert len(messages) == 2
        return make_stub_completion(
            json.dumps(_valid_response_payload(), ensure_ascii=False),
            provider="deepseek",
            model="deepseek-v4-flash",
        )

    client = MockLLMClient(handler=handler)

    result = run_async(
        execution_service.execute_fact_extraction_batch(
            session_factory,
            project_id=project_id,
            extraction_run_id=extraction_run_id,
            plan=plan,
            batch_index=0,
            prompt=PROMPT,
            llm_client=client,
            provider="deepseek",
            requested_model="deepseek-v4-flash",
        )
    )

    assert result.project_id == project_id
    assert result.extraction_run_id == extraction_run_id
    assert result.input_batch_id == orm_batch.id
    assert result.inference_run_id == completed_run.id
    assert result.reused_completed_run is False
    assert result.response_model == "deepseek-v4-flash"
    assert result.total_tokens == 46
    assert result.response == FactExtractionResponse.model_validate(_valid_response_payload())
    assert len(client.calls) == 1
    assert client.calls[0].temperature == PROMPT.temperature
    assert client.calls[0].max_tokens == PROMPT.max_output_tokens
    assert client.calls[0].response_format_json is True


def test_execute_fact_extraction_batch_reuses_completed_run_without_llm(monkeypatch):
    session_factory = SessionFactory()
    project_id, extraction_run_id, blocks, plan, batch_plan = _planned_fixture()
    orm_batch = _materialized_orm_batch(
        project_id=project_id,
        extraction_run_id=extraction_run_id,
        blocks=blocks,
        batch_plan=batch_plan,
    )
    completed_run = _run(
        project_id=project_id,
        input_batch_id=orm_batch.id,
        request_hash=sha256("req"),
        status=InferenceRunStatus.COMPLETED.value,
        response_json=_valid_response_payload(),
    )

    monkeypatch.setattr(execution_service, "create_inference_input_batch", async_lambda(orm_batch))
    monkeypatch.setattr(
        execution_service.inference_repository,
        "get_batch_by_identity",
        async_lambda(orm_batch),
    )
    monkeypatch.setattr(
        execution_service,
        "prepare_inference_run",
        async_lambda(PreparedInferenceRun(run=completed_run, created=False, reused_completed=True)),
    )

    client = MockLLMClient(["{}"])

    result = run_async(
        execution_service.execute_fact_extraction_batch(
            session_factory,
            project_id=project_id,
            extraction_run_id=extraction_run_id,
            plan=plan,
            batch_index=0,
            prompt=PROMPT,
            llm_client=client,
            provider="deepseek",
            requested_model="deepseek-v4-flash",
        )
    )

    assert result.reused_completed_run is True
    assert client.calls == []


def test_execute_fact_extraction_batch_running_claim_does_not_call_llm_or_fail_run(monkeypatch):
    session_factory = SessionFactory()
    project_id, extraction_run_id, blocks, plan, batch_plan = _planned_fixture()
    orm_batch = _materialized_orm_batch(
        project_id=project_id,
        extraction_run_id=extraction_run_id,
        blocks=blocks,
        batch_plan=batch_plan,
    )
    pending_run = _run(
        project_id=project_id,
        input_batch_id=orm_batch.id,
        request_hash=sha256("req"),
        status=InferenceRunStatus.PENDING.value,
    )
    fail_calls: list[tuple[uuid.UUID, str]] = []

    monkeypatch.setattr(execution_service, "create_inference_input_batch", async_lambda(orm_batch))
    monkeypatch.setattr(
        execution_service.inference_repository,
        "get_batch_by_identity",
        async_lambda(orm_batch),
    )
    monkeypatch.setattr(
        execution_service,
        "prepare_inference_run",
        async_lambda(PreparedInferenceRun(run=pending_run, created=False, reused_completed=False)),
    )
    monkeypatch.setattr(
        execution_service,
        "claim_inference_run_for_execution",
        async_lambda(
            InferenceRunClaim(
                run_id=pending_run.id,
                status=InferenceRunStatus.RUNNING.value,
                claimed=False,
            )
        ),
    )

    async def fake_fail(_session, *, run_id, failure_code, failure_message=None):
        fail_calls.append((run_id, failure_code))
        return pending_run

    monkeypatch.setattr(execution_service, "fail_inference_run", fake_fail)
    client = MockLLMClient(["{}"])

    with pytest.raises(execution_service.FactExtractionRunAlreadyRunningError):
        run_async(
            execution_service.execute_fact_extraction_batch(
                session_factory,
                project_id=project_id,
                extraction_run_id=extraction_run_id,
                plan=plan,
                batch_index=0,
                prompt=PROMPT,
                llm_client=client,
                provider="deepseek",
                requested_model="deepseek-v4-flash",
            )
        )

    assert client.calls == []
    assert fail_calls == []


def test_execute_fact_extraction_batch_evidence_bounds_failure_records_safe_code(monkeypatch):
    session_factory = SessionFactory()
    project_id, extraction_run_id, blocks, plan, batch_plan = _planned_fixture()
    orm_batch = _materialized_orm_batch(
        project_id=project_id,
        extraction_run_id=extraction_run_id,
        blocks=blocks,
        batch_plan=batch_plan,
    )
    pending_run = _run(
        project_id=project_id,
        input_batch_id=orm_batch.id,
        request_hash=sha256("req"),
        status=InferenceRunStatus.PENDING.value,
    )
    fail_calls: list[tuple[uuid.UUID, str]] = []

    monkeypatch.setattr(execution_service, "create_inference_input_batch", async_lambda(orm_batch))
    monkeypatch.setattr(
        execution_service.inference_repository,
        "get_batch_by_identity",
        async_lambda(orm_batch),
    )
    monkeypatch.setattr(
        execution_service,
        "prepare_inference_run",
        async_lambda(PreparedInferenceRun(run=pending_run, created=True, reused_completed=False)),
    )
    monkeypatch.setattr(
        execution_service,
        "claim_inference_run_for_execution",
        async_lambda(
            InferenceRunClaim(
                run_id=pending_run.id,
                status=InferenceRunStatus.RUNNING.value,
                claimed=True,
            )
        ),
    )

    async def fake_fail(_session, *, run_id, failure_code, failure_message=None):
        fail_calls.append((run_id, failure_code))
        return pending_run

    monkeypatch.setattr(execution_service, "fail_inference_run", fake_fail)
    client = MockLLMClient(
        [
            json.dumps(
                {
                    "facts": [
                        {
                            "subject_kind": "capital",
                            "subject_key": "rose_kingdom",
                            "predicate_key": "name",
                            "value_type": "string",
                            "value_json": "白蔷城",
                            "confidence": 0.9,
                            "evidence": [
                                {
                                    "block_ref": "B0001",
                                    "start_offset": 0,
                                    "end_offset": 9999,
                                    "role": "supporting",
                                }
                            ],
                        }
                    ],
                    "batch_summary": "ok",
                    "uncertainties": [],
                },
                ensure_ascii=False,
            )
        ]
    )

    with pytest.raises(execution_service.FactExtractionEvidenceBoundsError):
        run_async(
            execution_service.execute_fact_extraction_batch(
                session_factory,
                project_id=project_id,
                extraction_run_id=extraction_run_id,
                plan=plan,
                batch_index=0,
                prompt=PROMPT,
                llm_client=client,
                provider="deepseek",
                requested_model="deepseek-v4-flash",
            )
        )

    assert fail_calls == [(pending_run.id, "agent_evidence_bounds_invalid")]


def test_execute_fact_extraction_batch_cancelled_records_safe_failure_and_reraises(monkeypatch):
    session_factory = SessionFactory()
    project_id, extraction_run_id, blocks, plan, batch_plan = _planned_fixture()
    orm_batch = _materialized_orm_batch(
        project_id=project_id,
        extraction_run_id=extraction_run_id,
        blocks=blocks,
        batch_plan=batch_plan,
    )
    pending_run = _run(
        project_id=project_id,
        input_batch_id=orm_batch.id,
        request_hash=sha256("req"),
        status=InferenceRunStatus.PENDING.value,
    )
    fail_calls: list[tuple[uuid.UUID, str]] = []

    monkeypatch.setattr(execution_service, "create_inference_input_batch", async_lambda(orm_batch))
    monkeypatch.setattr(
        execution_service.inference_repository,
        "get_batch_by_identity",
        async_lambda(orm_batch),
    )
    monkeypatch.setattr(
        execution_service,
        "prepare_inference_run",
        async_lambda(PreparedInferenceRun(run=pending_run, created=True, reused_completed=False)),
    )
    monkeypatch.setattr(
        execution_service,
        "claim_inference_run_for_execution",
        async_lambda(
            InferenceRunClaim(
                run_id=pending_run.id,
                status=InferenceRunStatus.RUNNING.value,
                claimed=True,
            )
        ),
    )

    async def fake_fail(_session, *, run_id, failure_code, failure_message=None):
        fail_calls.append((run_id, failure_code))
        return pending_run

    monkeypatch.setattr(execution_service, "fail_inference_run", fake_fail)

    class CancelClient:
        async def complete(self, *args, **kwargs):
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        run_async(
            execution_service.execute_fact_extraction_batch(
                session_factory,
                project_id=project_id,
                extraction_run_id=extraction_run_id,
                plan=plan,
                batch_index=0,
                prompt=PROMPT,
                llm_client=CancelClient(),
                provider="deepseek",
                requested_model="deepseek-v4-flash",
            )
        )

    assert fail_calls == [(pending_run.id, "fact_extraction_execution_cancelled")]


def test_execute_fact_extraction_batch_rejects_bool_batch_index():
    project_id, extraction_run_id, blocks, plan, _batch_plan = _planned_fixture()
    with pytest.raises(execution_service.FactExtractionExecutionError):
        run_async(
            execution_service.execute_fact_extraction_batch(
                SessionFactory(),
                project_id=project_id,
                extraction_run_id=extraction_run_id,
                plan=plan,
                batch_index=True,
                prompt=PROMPT,
                llm_client=MockLLMClient(["{}"]),
                provider="deepseek",
                requested_model="deepseek-v4-flash",
            )
        )


def test_execution_source_does_not_import_httpx_or_create_fact_records():
    source = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "services"
        / "fact_extraction_execution.py"
    ).read_text(encoding="utf-8")
    assert "httpx" not in source
    assert "Fact(" not in source
    assert "FactValue(" not in source
    assert "SourceEvidence(" not in source
    assert "Entity(" not in source


def async_lambda(result):
    async def _wrapper(*args, **kwargs):
        return result

    return _wrapper


def test_prepared_run_observer_runs_after_prepare_and_before_claim_and_llm(monkeypatch):
    session_factory = SessionFactory()
    project_id, extraction_run_id, blocks, plan, batch_plan = _planned_fixture()
    orm_batch = _materialized_orm_batch(
        project_id=project_id,
        extraction_run_id=extraction_run_id,
        blocks=blocks,
        batch_plan=batch_plan,
    )
    request_hash = sha256("req")
    pending_run = _run(
        project_id=project_id,
        input_batch_id=orm_batch.id,
        request_hash=request_hash,
        status=InferenceRunStatus.PENDING.value,
    )
    completed_run = _run(
        project_id=project_id,
        input_batch_id=orm_batch.id,
        request_hash=request_hash,
        status=InferenceRunStatus.COMPLETED.value,
        response_json=_valid_response_payload(),
    )
    events: list[str] = []

    monkeypatch.setattr(execution_service, "create_inference_input_batch", async_lambda(orm_batch))
    monkeypatch.setattr(
        execution_service.inference_repository,
        "get_batch_by_identity",
        async_lambda(orm_batch),
    )

    async def fake_prepare(*_args, **_kwargs):
        events.append("prepare")
        return PreparedInferenceRun(run=pending_run, created=True, reused_completed=False)

    async def fake_claim(*_args, **_kwargs):
        assert events == ["prepare", "observer"]
        events.append("claim")
        return InferenceRunClaim(
            run_id=pending_run.id,
            status=InferenceRunStatus.RUNNING.value,
            claimed=True,
        )

    async def fake_complete(*_args, **_kwargs):
        events.append("complete")
        return completed_run

    monkeypatch.setattr(execution_service, "prepare_inference_run", fake_prepare)
    monkeypatch.setattr(execution_service, "claim_inference_run_for_execution", fake_claim)
    monkeypatch.setattr(execution_service, "complete_inference_run", fake_complete)

    async def observer(notice):
        assert session_factory.open_count == 0
        assert notice.project_id == project_id
        assert notice.extraction_run_id == extraction_run_id
        assert notice.plan_hash == plan.plan_hash
        assert notice.batch_index == 0
        assert notice.batch_plan_hash == batch_plan.plan_hash
        assert notice.input_batch_id == orm_batch.id
        assert notice.inference_run_id == pending_run.id
        assert notice.inference_request_hash == request_hash
        assert not hasattr(notice, "messages")
        assert not hasattr(notice, "prompt")
        events.append("observer")

    client = MockLLMClient(
        handler=lambda _messages: (
            events.append("llm"),
            make_stub_completion(
                json.dumps(_valid_response_payload(), ensure_ascii=False),
                provider="deepseek",
                model="deepseek-v4-flash",
            ),
        )[1]
    )

    result = run_async(
        execution_service.execute_fact_extraction_batch(
            session_factory,
            project_id=project_id,
            extraction_run_id=extraction_run_id,
            plan=plan,
            batch_index=0,
            prompt=PROMPT,
            llm_client=client,
            provider="deepseek",
            requested_model="deepseek-v4-flash",
            prepared_run_observer=observer,
        )
    )

    assert result.inference_run_id == completed_run.id
    assert events == ["prepare", "observer", "claim", "llm", "complete"]


def test_prepared_run_observer_failure_prevents_claim_and_llm(monkeypatch):
    session_factory = SessionFactory()
    project_id, extraction_run_id, blocks, plan, batch_plan = _planned_fixture()
    orm_batch = _materialized_orm_batch(
        project_id=project_id,
        extraction_run_id=extraction_run_id,
        blocks=blocks,
        batch_plan=batch_plan,
    )
    pending_run = _run(
        project_id=project_id,
        input_batch_id=orm_batch.id,
        request_hash=sha256("req"),
        status=InferenceRunStatus.PENDING.value,
    )
    claim_calls = {"count": 0}

    monkeypatch.setattr(execution_service, "create_inference_input_batch", async_lambda(orm_batch))
    monkeypatch.setattr(
        execution_service.inference_repository,
        "get_batch_by_identity",
        async_lambda(orm_batch),
    )
    monkeypatch.setattr(
        execution_service,
        "prepare_inference_run",
        async_lambda(PreparedInferenceRun(run=pending_run, created=True, reused_completed=False)),
    )

    async def fake_claim(*_args, **_kwargs):
        claim_calls["count"] += 1
        return InferenceRunClaim(
            run_id=pending_run.id,
            status=InferenceRunStatus.RUNNING.value,
            claimed=True,
        )

    monkeypatch.setattr(execution_service, "claim_inference_run_for_execution", fake_claim)
    client = MockLLMClient(["{}"])

    async def observer(_notice):
        raise RuntimeError("observer failed")

    with pytest.raises(RuntimeError):
        run_async(
            execution_service.execute_fact_extraction_batch(
                session_factory,
                project_id=project_id,
                extraction_run_id=extraction_run_id,
                plan=plan,
                batch_index=0,
                prompt=PROMPT,
                llm_client=client,
                provider="deepseek",
                requested_model="deepseek-v4-flash",
                prepared_run_observer=observer,
            )
        )

    assert claim_calls["count"] == 0
    assert client.calls == []
