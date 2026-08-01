from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import uuid

import pytest

from app.agents.prompt_registry import get_prompt
from app.models.base import utc_now
from app.models.inference import InferenceInputBatch, InferenceInputBlock, InferenceRun, InferenceRunStatus
from app.schemas.agent_consistency_check import ConsistencyCheckResponse
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
from app.services import consistency_check as consistency_check_service
from app.services import consistency_check_execution as execution_service
from app.services import fact_value_duplicate_grouping as duplicate_grouping_service
from app.services import inference as inference_service
from app.services.inference import InferenceRunClaim, PreparedInferenceRun
from app.services.llm import MockLLMClient, make_stub_completion


PROMPT = get_prompt("agent2_consistency_check", "1.0.0")


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


def _evidence(
    seed: int,
    *,
    document_block_id: uuid.UUID,
    excerpt: str,
    source_order: int,
) -> ConsistencyCheckEvidenceBundle:
    return ConsistencyCheckEvidenceBundle(
        evidence_link_id=uuid.UUID(f"00000000-0000-0000-0000-{seed:012d}"),
        evidence_id=uuid.UUID(f"10000000-0000-0000-0000-{seed:012d}"),
        role="supporting",
        is_primary=True,
        source_order=source_order,
        document_block_id=document_block_id,
        location_key=f"loc:{seed}",
        page_no=1,
        start_line=seed,
        end_line=seed,
        start_offset=0,
        end_offset=len(excerpt),
        excerpt=excerpt,
        evidence_content_hash=sha256(excerpt),
    )


def _member(
    seed: int,
    *,
    semantic_key_hash: str,
    value_json: object,
    evidences: tuple[ConsistencyCheckEvidenceBundle, ...],
) -> ConsistencyCheckMemberBundle:
    return ConsistencyCheckMemberBundle(
        fact_value_id=uuid.UUID(f"20000000-0000-0000-0000-{seed:012d}"),
        source_batch_id=uuid.UUID(f"30000000-0000-0000-0000-{seed:012d}"),
        semantic_key_hash=semantic_key_hash,
        value_type="string",
        value_json=value_json,
        referenced_entity_id=None,
        evidences=evidences,
    )


def _candidate(
    seed: int,
    *,
    members: tuple[ConsistencyCheckMemberBundle, ...],
) -> ConsistencyCheckCandidateBundle:
    return ConsistencyCheckCandidateBundle(
        candidate_id=uuid.UUID(f"40000000-0000-0000-0000-{seed:012d}"),
        fact_id=uuid.UUID(f"50000000-0000-0000-0000-{seed:012d}"),
        candidate_kind="multi_value",
        members=members,
    )


def _plan(
    candidates: tuple[ConsistencyCheckCandidateBundle, ...],
    *,
    consistency_application_id: uuid.UUID | None = None,
    source_result_manifest_hash: str = "a" * 64,
    config: ConsistencyCheckPlannerConfig | None = None,
) -> tuple[ConsistencyCheckPlan, ConsistencyCheckBatchPlan]:
    resolved_config = config or ConsistencyCheckPlannerConfig(
        max_candidates_per_batch=10,
        max_evidence_characters_per_batch=10_000,
    )
    application_id = consistency_application_id or uuid.uuid4()
    batches = consistency_check_service._build_consistency_check_batches(
        consistency_application_id=application_id,
        source_result_manifest_hash=source_result_manifest_hash,
        config=resolved_config,
        candidate_bundles=candidates,
    )
    plan_manifest_hash = duplicate_grouping_service.hash_deterministic_payload(
        {
            "consistency_application_id": str(application_id),
            "source_result_manifest_hash": source_result_manifest_hash,
            "planner_name": CONSISTENCY_CHECK_PLANNER_NAME,
            "planner_version": CONSISTENCY_CHECK_PLANNER_VERSION,
            "config": {
                "max_candidates_per_batch": resolved_config.max_candidates_per_batch,
                "max_evidence_characters_per_batch": resolved_config.max_evidence_characters_per_batch,
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
    plan = ConsistencyCheckPlan(
        consistency_application_id=application_id,
        source_result_manifest_hash=source_result_manifest_hash,
        planner_name=CONSISTENCY_CHECK_PLANNER_NAME,
        planner_version=CONSISTENCY_CHECK_PLANNER_VERSION,
        config=resolved_config,
        batches=batches,
        plan_manifest_hash=plan_manifest_hash,
    )
    return plan, plan.batches[0]


def _orm_batch(
    *,
    project_id: uuid.UUID,
    source_block_ids: tuple[uuid.UUID, ...],
    texts_by_block_id: dict[uuid.UUID, str],
    snapshot_hash: str = "c" * 64,
) -> InferenceInputBatch:
    batch = InferenceInputBatch(
        id=uuid.uuid4(),
        project_id=project_id,
        task_type="consistency_check",
        selection_strategy=execution_service.CONSISTENCY_CHECK_EXECUTOR_NAME,
        selection_metadata={},
        block_count=len(source_block_ids),
        character_count=sum(len(texts_by_block_id[block_id]) for block_id in source_block_ids),
        snapshot_hash=snapshot_hash,
    )
    blocks: list[InferenceInputBlock] = []
    extraction_run_id = uuid.uuid4()
    for source_order, block_id in enumerate(source_block_ids):
        text = texts_by_block_id[block_id]
        blocks.append(
            InferenceInputBlock(
                id=uuid.uuid4(),
                batch_id=batch.id,
                source_order=source_order,
                block_ref=f"B{source_order + 1:04d}",
                document_block_id=block_id,
                source_block_id_snapshot=block_id,
                extraction_run_id_snapshot=extraction_run_id,
                block_type="paragraph",
                location_key=f"loc-{source_order}",
                anchor_hash=sha256(f"anchor-{source_order}"),
                page_no=1,
                start_line=source_order + 1,
                end_line=source_order + 1,
                heading_path=[],
                content_text=text,
                content_hash=sha256(text),
            )
        )
    batch.blocks = blocks
    batch.snapshot_hash = inference_service.build_inference_input_batch_snapshot_hash(
        [
            {
                "source_order": block.source_order,
                "block_ref": block.block_ref,
                "source_block_id": str(block.source_block_id_snapshot),
                "extraction_run_id": str(block.extraction_run_id_snapshot),
                "block_type": block.block_type,
                "location_key": block.location_key,
                "anchor_hash": block.anchor_hash,
                "page_no": block.page_no,
                "start_line": block.start_line,
                "end_line": block.end_line,
                "heading_path": list(block.heading_path),
                "content_hash": block.content_hash,
            }
            for block in blocks
        ]
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
        task_type="consistency_check",
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
        run.response_json_hash = sha256(json.dumps(response_json, sort_keys=True, ensure_ascii=False))
    return run


def _response_payload(batch: ConsistencyCheckBatchPlan) -> dict[str, object]:
    assessments: list[dict[str, object]] = []
    for candidate in batch.candidates:
        assessments.append(
            {
                "candidate_id": str(candidate.candidate_id),
                "verdict": "compatible",
                "severity": "none",
                "confidence": 0.5,
                "explanation": "Evidence is compatible.",
                "cited_evidence_link_ids": [str(candidate.members[0].evidences[0].evidence_link_id)],
                "impact": [],
                "recommended_actions": ["leave_as_is"],
            }
        )
    return {"assessments": assessments}


def async_lambda(result):
    async def _wrapper(*args, **kwargs):
        return result

    return _wrapper


def test_execute_consistency_check_batch_skips_empty_batch_without_run_or_llm(monkeypatch):
    session_factory = SessionFactory()
    project_id = uuid.uuid4()
    plan, _batch = _plan(())
    called = {"create_batch": 0}

    async def fake_build_plan(_session_factory, *, consistency_application_id, config):
        return plan

    async def fake_create_batch(*args, **kwargs):
        called["create_batch"] += 1
        raise AssertionError("should not create input batch for empty batch")

    monkeypatch.setattr(execution_service, "build_consistency_check_plan", fake_build_plan)
    monkeypatch.setattr(execution_service, "create_inference_input_batch", fake_create_batch)
    client = MockLLMClient(["{}"])

    result = run_async(
        execution_service.execute_consistency_check_batch(
            session_factory,
            project_id=project_id,
            plan=plan,
            batch_index=0,
            prompt=PROMPT,
            llm_client=client,
            provider="deepseek",
            requested_model="deepseek-v4-flash",
        )
    )

    assert result.skipped_empty is True
    assert result.reused_completed_run is False
    assert result.input_batch_id is None
    assert result.inference_run_id is None
    assert result.request_hash is None
    assert result.message_content_hash is None
    assert result.response == ConsistencyCheckResponse(assessments=[])
    assert called["create_batch"] == 0
    assert client.calls == []


def test_execute_consistency_check_batch_materializes_block_order_with_first_seen_dedup(monkeypatch):
    session_factory = SessionFactory()
    project_id = uuid.uuid4()
    block_a = uuid.uuid4()
    block_b = uuid.uuid4()
    block_c = uuid.uuid4()
    candidate = _candidate(
        1,
        members=(
            _member(
                1,
                semantic_key_hash="1" * 64,
                value_json="A",
                evidences=(
                    _evidence(1, document_block_id=block_a, excerpt="alpha", source_order=0),
                    _evidence(2, document_block_id=block_b, excerpt="beta", source_order=1),
                ),
            ),
            _member(
                2,
                semantic_key_hash="2" * 64,
                value_json="B",
                evidences=(
                    _evidence(3, document_block_id=block_a, excerpt="alpha again", source_order=0),
                    _evidence(4, document_block_id=block_c, excerpt="gamma", source_order=1),
                ),
            ),
        ),
    )
    plan, batch = _plan((candidate,))
    expected_block_ids = (block_a, block_b, block_c)
    orm_batch = _orm_batch(
        project_id=project_id,
        source_block_ids=expected_block_ids,
        texts_by_block_id={
            block_a: "Block A",
            block_b: "Block B",
            block_c: "Block C",
        },
    )
    captured_block_ids: list[uuid.UUID] = []
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
        response_json=_response_payload(batch),
    )

    async def fake_build_plan(_session_factory, *, consistency_application_id, config):
        return plan

    async def fake_create_batch(_session, **kwargs):
        captured_block_ids.extend(kwargs["block_ids"])
        return orm_batch

    monkeypatch.setattr(execution_service, "build_consistency_check_plan", fake_build_plan)
    monkeypatch.setattr(execution_service, "create_inference_input_batch", fake_create_batch)
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
    monkeypatch.setattr(execution_service, "complete_inference_run", async_lambda(completed_run))
    client = MockLLMClient([json.dumps(_response_payload(batch))])

    result = run_async(
        execution_service.execute_consistency_check_batch(
            session_factory,
            project_id=project_id,
            plan=plan,
            batch_index=0,
            prompt=PROMPT,
            llm_client=client,
            provider="deepseek",
            requested_model="deepseek-v4-flash",
        )
    )

    assert captured_block_ids == list(expected_block_ids)
    assert result.input_batch_id == orm_batch.id
    assert result.response == ConsistencyCheckResponse.model_validate(_response_payload(batch))


def test_execute_consistency_check_batch_rejects_plan_rebuild_mismatch(monkeypatch):
    session_factory = SessionFactory()
    project_id = uuid.uuid4()
    block_id = uuid.uuid4()
    candidate = _candidate(
        1,
        members=(
            _member(
                1,
                semantic_key_hash="1" * 64,
                value_json="A",
                evidences=(_evidence(1, document_block_id=block_id, excerpt="alpha", source_order=0),),
            ),
            _member(
                2,
                semantic_key_hash="2" * 64,
                value_json="B",
                evidences=(_evidence(2, document_block_id=block_id, excerpt="beta", source_order=1),),
            ),
        ),
    )
    plan, _batch = _plan((candidate,))
    provided_plan = replace(plan, plan_manifest_hash="f" * 64)
    create_calls = {"count": 0}

    async def fake_build_plan(_session_factory, *, consistency_application_id, config):
        return plan

    async def fake_create_batch(*args, **kwargs):
        create_calls["count"] += 1
        raise AssertionError("should not materialize batch after plan mismatch")

    monkeypatch.setattr(execution_service, "build_consistency_check_plan", fake_build_plan)
    monkeypatch.setattr(execution_service, "create_inference_input_batch", fake_create_batch)

    with pytest.raises(execution_service.ConsistencyCheckPlanMismatchError):
        run_async(
            execution_service.execute_consistency_check_batch(
                session_factory,
                project_id=project_id,
                plan=provided_plan,
                batch_index=0,
                prompt=PROMPT,
                llm_client=MockLLMClient(["{}"]),
                provider="deepseek",
                requested_model="deepseek-v4-flash",
            )
        )

    assert create_calls["count"] == 0


def test_execute_consistency_check_batch_calls_llm_only_after_sessions_close(monkeypatch):
    session_factory = SessionFactory()
    project_id = uuid.uuid4()
    block_id = uuid.uuid4()
    candidate = _candidate(
        1,
        members=(
            _member(
                1,
                semantic_key_hash="1" * 64,
                value_json="A",
                evidences=(_evidence(1, document_block_id=block_id, excerpt="alpha", source_order=0),),
            ),
            _member(
                2,
                semantic_key_hash="2" * 64,
                value_json="B",
                evidences=(_evidence(2, document_block_id=block_id, excerpt="beta", source_order=1),),
            ),
        ),
    )
    plan, batch = _plan((candidate,))
    orm_batch = _orm_batch(
        project_id=project_id,
        source_block_ids=(block_id,),
        texts_by_block_id={block_id: "Shared block"},
    )
    pending_run = _run(
        project_id=project_id,
        input_batch_id=orm_batch.id,
        request_hash=sha256("req"),
        status=InferenceRunStatus.PENDING.value,
    )
    completed_run = _run(
        project_id=project_id,
        input_batch_id=orm_batch.id,
        request_hash=sha256("req"),
        status=InferenceRunStatus.COMPLETED.value,
        response_json=_response_payload(batch),
    )

    async def fake_build_plan(_session_factory, *, consistency_application_id, config):
        return plan

    monkeypatch.setattr(execution_service, "build_consistency_check_plan", fake_build_plan)
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
    monkeypatch.setattr(execution_service, "complete_inference_run", async_lambda(completed_run))

    def handler(messages):
        assert session_factory.open_count == 0
        return make_stub_completion(
            json.dumps(_response_payload(batch), ensure_ascii=False),
            provider="deepseek",
            model="deepseek-v4-flash",
        )

    client = MockLLMClient(handler=handler)

    result = run_async(
        execution_service.execute_consistency_check_batch(
            session_factory,
            project_id=project_id,
            plan=plan,
            batch_index=0,
            prompt=PROMPT,
            llm_client=client,
            provider="deepseek",
            requested_model="deepseek-v4-flash",
        )
    )

    assert result.reused_completed_run is False
    assert len(client.calls) == 1


def test_execute_consistency_check_batch_success_completes_new_run(monkeypatch):
    session_factory = SessionFactory()
    project_id = uuid.uuid4()
    block_id = uuid.uuid4()
    candidate = _candidate(
        1,
        members=(
            _member(
                1,
                semantic_key_hash="1" * 64,
                value_json="A",
                evidences=(_evidence(1, document_block_id=block_id, excerpt="alpha", source_order=0),),
            ),
            _member(
                2,
                semantic_key_hash="2" * 64,
                value_json="B",
                evidences=(_evidence(2, document_block_id=block_id, excerpt="beta", source_order=1),),
            ),
        ),
    )
    plan, batch = _plan((candidate,))
    orm_batch = _orm_batch(
        project_id=project_id,
        source_block_ids=(block_id,),
        texts_by_block_id={block_id: "Shared block"},
    )
    request_hash = sha256("req-success")
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
        response_json=_response_payload(batch),
    )
    completed_run.prompt_tokens = 11
    completed_run.completion_tokens = 22
    completed_run.total_tokens = 33

    async def fake_build_plan(_session_factory, *, consistency_application_id, config):
        return plan

    monkeypatch.setattr(execution_service, "build_consistency_check_plan", fake_build_plan)
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
    monkeypatch.setattr(execution_service, "complete_inference_run", async_lambda(completed_run))
    client = MockLLMClient(
        [
            make_stub_completion(
                json.dumps(_response_payload(batch), ensure_ascii=False),
                provider="deepseek",
                model="deepseek-v4-flash",
            )
        ]
    )

    result = run_async(
        execution_service.execute_consistency_check_batch(
            session_factory,
            project_id=project_id,
            plan=plan,
            batch_index=0,
            prompt=PROMPT,
            llm_client=client,
            provider="deepseek",
            requested_model="deepseek-v4-flash",
        )
    )

    assert result.project_id == project_id
    assert result.consistency_application_id == plan.consistency_application_id
    assert result.input_batch_id == orm_batch.id
    assert result.inference_run_id == completed_run.id
    assert result.request_hash == request_hash
    assert result.skipped_empty is False
    assert result.reused_completed_run is False
    assert result.response == ConsistencyCheckResponse.model_validate(_response_payload(batch))
    assert result.total_tokens == 33


def test_execute_consistency_check_batch_reuses_completed_run_without_llm(monkeypatch):
    session_factory = SessionFactory()
    project_id = uuid.uuid4()
    block_id = uuid.uuid4()
    candidate = _candidate(
        1,
        members=(
            _member(
                1,
                semantic_key_hash="1" * 64,
                value_json="A",
                evidences=(_evidence(1, document_block_id=block_id, excerpt="alpha", source_order=0),),
            ),
            _member(
                2,
                semantic_key_hash="2" * 64,
                value_json="B",
                evidences=(_evidence(2, document_block_id=block_id, excerpt="beta", source_order=1),),
            ),
        ),
    )
    plan, batch = _plan((candidate,))
    orm_batch = _orm_batch(
        project_id=project_id,
        source_block_ids=(block_id,),
        texts_by_block_id={block_id: "Shared block"},
    )
    completed_run = _run(
        project_id=project_id,
        input_batch_id=orm_batch.id,
        request_hash=sha256("req-completed"),
        status=InferenceRunStatus.COMPLETED.value,
        response_json=_response_payload(batch),
    )

    async def fake_build_plan(_session_factory, *, consistency_application_id, config):
        return plan

    monkeypatch.setattr(execution_service, "build_consistency_check_plan", fake_build_plan)
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
        execution_service.execute_consistency_check_batch(
            session_factory,
            project_id=project_id,
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


def test_execute_consistency_check_batch_running_claim_does_not_call_llm(monkeypatch):
    session_factory = SessionFactory()
    project_id = uuid.uuid4()
    block_id = uuid.uuid4()
    candidate = _candidate(
        1,
        members=(
            _member(
                1,
                semantic_key_hash="1" * 64,
                value_json="A",
                evidences=(_evidence(1, document_block_id=block_id, excerpt="alpha", source_order=0),),
            ),
            _member(
                2,
                semantic_key_hash="2" * 64,
                value_json="B",
                evidences=(_evidence(2, document_block_id=block_id, excerpt="beta", source_order=1),),
            ),
        ),
    )
    plan, _batch = _plan((candidate,))
    orm_batch = _orm_batch(
        project_id=project_id,
        source_block_ids=(block_id,),
        texts_by_block_id={block_id: "Shared block"},
    )
    pending_run = _run(
        project_id=project_id,
        input_batch_id=orm_batch.id,
        request_hash=sha256("req-running"),
        status=InferenceRunStatus.PENDING.value,
    )
    fail_calls: list[tuple[uuid.UUID, str]] = []

    async def fake_build_plan(_session_factory, *, consistency_application_id, config):
        return plan

    async def fake_fail(_session, *, run_id, failure_code, failure_message=None):
        fail_calls.append((run_id, failure_code))
        return pending_run

    monkeypatch.setattr(execution_service, "build_consistency_check_plan", fake_build_plan)
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
    monkeypatch.setattr(execution_service, "fail_inference_run", fake_fail)
    client = MockLLMClient(["{}"])

    with pytest.raises(execution_service.ConsistencyCheckRunAlreadyRunningError):
        run_async(
            execution_service.execute_consistency_check_batch(
                session_factory,
                project_id=project_id,
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


def test_execute_consistency_check_batch_invalid_model_response_fails_claimed_run(monkeypatch):
    session_factory = SessionFactory()
    project_id = uuid.uuid4()
    block_id = uuid.uuid4()
    candidate = _candidate(
        1,
        members=(
            _member(
                1,
                semantic_key_hash="1" * 64,
                value_json="A",
                evidences=(_evidence(1, document_block_id=block_id, excerpt="alpha", source_order=0),),
            ),
            _member(
                2,
                semantic_key_hash="2" * 64,
                value_json="B",
                evidences=(_evidence(2, document_block_id=block_id, excerpt="beta", source_order=1),),
            ),
        ),
    )
    plan, batch = _plan((candidate,))
    orm_batch = _orm_batch(
        project_id=project_id,
        source_block_ids=(block_id,),
        texts_by_block_id={block_id: "Shared block"},
    )
    pending_run = _run(
        project_id=project_id,
        input_batch_id=orm_batch.id,
        request_hash=sha256("req-invalid"),
        status=InferenceRunStatus.PENDING.value,
    )
    fail_calls: list[tuple[uuid.UUID, str]] = []

    async def fake_build_plan(_session_factory, *, consistency_application_id, config):
        return plan

    async def fake_fail(_session, *, run_id, failure_code, failure_message=None):
        fail_calls.append((run_id, failure_code))
        return pending_run

    monkeypatch.setattr(execution_service, "build_consistency_check_plan", fake_build_plan)
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
    monkeypatch.setattr(execution_service, "fail_inference_run", fake_fail)
    client = MockLLMClient(
        [
            json.dumps(
                {
                    "assessments": [
                        {
                            "candidate_id": str(batch.candidates[0].candidate_id),
                            "verdict": "compatible",
                            "severity": "none",
                            "confidence": 0.5,
                            "explanation": "ok",
                            "cited_evidence_link_ids": [str(uuid.uuid4())],
                            "impact": [],
                            "recommended_actions": ["leave_as_is"],
                        }
                    ]
                }
            )
        ]
    )

    with pytest.raises(execution_service.AgentConsistencyCheckResponseError):
        run_async(
            execution_service.execute_consistency_check_batch(
                session_factory,
                project_id=project_id,
                plan=plan,
                batch_index=0,
                prompt=PROMPT,
                llm_client=client,
                provider="deepseek",
                requested_model="deepseek-v4-flash",
            )
        )

    assert fail_calls == [(pending_run.id, "consistency_check_response_invalid")]


def test_execute_consistency_check_batch_cancelled_records_failure_and_reraises(monkeypatch):
    session_factory = SessionFactory()
    project_id = uuid.uuid4()
    block_id = uuid.uuid4()
    candidate = _candidate(
        1,
        members=(
            _member(
                1,
                semantic_key_hash="1" * 64,
                value_json="A",
                evidences=(_evidence(1, document_block_id=block_id, excerpt="alpha", source_order=0),),
            ),
            _member(
                2,
                semantic_key_hash="2" * 64,
                value_json="B",
                evidences=(_evidence(2, document_block_id=block_id, excerpt="beta", source_order=1),),
            ),
        ),
    )
    plan, _batch = _plan((candidate,))
    orm_batch = _orm_batch(
        project_id=project_id,
        source_block_ids=(block_id,),
        texts_by_block_id={block_id: "Shared block"},
    )
    pending_run = _run(
        project_id=project_id,
        input_batch_id=orm_batch.id,
        request_hash=sha256("req-cancel"),
        status=InferenceRunStatus.PENDING.value,
    )
    fail_calls: list[tuple[uuid.UUID, str]] = []

    async def fake_build_plan(_session_factory, *, consistency_application_id, config):
        return plan

    async def fake_fail(_session, *, run_id, failure_code, failure_message=None):
        fail_calls.append((run_id, failure_code))
        return pending_run

    monkeypatch.setattr(execution_service, "build_consistency_check_plan", fake_build_plan)
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
    monkeypatch.setattr(execution_service, "fail_inference_run", fake_fail)

    class CancelClient:
        async def complete(self, *args, **kwargs):
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        run_async(
            execution_service.execute_consistency_check_batch(
                session_factory,
                project_id=project_id,
                plan=plan,
                batch_index=0,
                prompt=PROMPT,
                llm_client=CancelClient(),
                provider="deepseek",
                requested_model="deepseek-v4-flash",
            )
        )

    assert fail_calls == [(pending_run.id, "consistency_check_execution_cancelled")]


def test_execute_consistency_check_batch_request_metadata_is_complete_and_changes_request_identity(monkeypatch):
    session_factory = SessionFactory()
    project_id = uuid.uuid4()
    shared_application_id = uuid.uuid4()
    block_id = uuid.uuid4()
    candidate_a = _candidate(
        1,
        members=(
            _member(
                1,
                semantic_key_hash="1" * 64,
                value_json="A",
                evidences=(_evidence(1, document_block_id=block_id, excerpt="alpha", source_order=0),),
            ),
            _member(
                2,
                semantic_key_hash="2" * 64,
                value_json="B",
                evidences=(_evidence(2, document_block_id=block_id, excerpt="beta", source_order=1),),
            ),
        ),
    )
    candidate_b = replace(
        candidate_a,
        members=(
            replace(
                candidate_a.members[0],
                evidences=(
                    replace(
                        candidate_a.members[0].evidences[0],
                        excerpt="changed excerpt",
                        evidence_content_hash=sha256("changed excerpt"),
                    ),
                ),
            ),
            candidate_a.members[1],
        ),
    )
    plan_a, batch_a = _plan(
        (candidate_a,),
        consistency_application_id=shared_application_id,
        source_result_manifest_hash="a" * 64,
    )
    plan_b, batch_b = _plan(
        (candidate_b,),
        consistency_application_id=shared_application_id,
        source_result_manifest_hash="b" * 64,
    )
    orm_batch = _orm_batch(
        project_id=project_id,
        source_block_ids=(block_id,),
        texts_by_block_id={block_id: "Shared block"},
    )
    captured_metadata: list[dict[str, object]] = []
    request_hashes: list[str] = []

    monkeypatch.setattr(execution_service, "create_inference_input_batch", async_lambda(orm_batch))
    monkeypatch.setattr(
        execution_service.inference_repository,
        "get_batch_by_identity",
        async_lambda(orm_batch),
    )

    async def fake_claim(_session, *, run_id):
        return InferenceRunClaim(
            run_id=run_id,
            status=InferenceRunStatus.RUNNING.value,
            claimed=True,
        )

    async def fake_complete(_session, *, run_id, completion):
        response_json = json.loads(completion.content)
        request_hash = request_hashes[-1]
        return _run(
            project_id=project_id,
            input_batch_id=orm_batch.id,
            request_hash=request_hash,
            status=InferenceRunStatus.COMPLETED.value,
            response_json=response_json,
        )

    monkeypatch.setattr(execution_service, "claim_inference_run_for_execution", fake_claim)
    monkeypatch.setattr(execution_service, "complete_inference_run", fake_complete)

    current_plan = {"value": plan_a}

    async def fake_build_plan(_session_factory, *, consistency_application_id, config):
        return current_plan["value"]

    async def fake_prepare(
        _session,
        *,
        project_id,
        input_batch_id,
        task_type,
        agent_name,
        agent_version,
        prompt_name,
        prompt_version,
        prompt_contract_hash,
        provider,
        requested_model,
        temperature,
        max_output_tokens,
        request_metadata,
    ):
        metadata = dict(request_metadata)
        captured_metadata.append(metadata)
        request_hash = inference_service.build_inference_request_hash(
            snapshot_hash=orm_batch.snapshot_hash,
            task_type=task_type,
            agent_name=agent_name,
            agent_version=agent_version,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            prompt_contract_hash=prompt_contract_hash,
            provider=provider,
            requested_model=requested_model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            request_metadata=metadata,
        )
        request_hashes.append(request_hash)
        return PreparedInferenceRun(
            run=_run(
                project_id=project_id,
                input_batch_id=input_batch_id,
                request_hash=request_hash,
                status=InferenceRunStatus.PENDING.value,
            ),
            created=True,
            reused_completed=False,
        )

    monkeypatch.setattr(execution_service, "build_consistency_check_plan", fake_build_plan)
    monkeypatch.setattr(execution_service, "prepare_inference_run", fake_prepare)
    client = MockLLMClient(
        [
            json.dumps(_response_payload(batch_a), ensure_ascii=False),
            json.dumps(_response_payload(batch_b), ensure_ascii=False),
        ]
    )

    result_a = run_async(
        execution_service.execute_consistency_check_batch(
            session_factory,
            project_id=project_id,
            plan=plan_a,
            batch_index=0,
            prompt=PROMPT,
            llm_client=client,
            provider="deepseek",
            requested_model="deepseek-v4-flash",
        )
    )

    current_plan["value"] = plan_b
    result_b = run_async(
        execution_service.execute_consistency_check_batch(
            session_factory,
            project_id=project_id,
            plan=plan_b,
            batch_index=0,
            prompt=PROMPT,
            llm_client=client,
            provider="deepseek",
            requested_model="deepseek-v4-flash",
        )
    )

    first_metadata = captured_metadata[0]
    assert first_metadata == {
        "consistency_application_id": str(plan_a.consistency_application_id),
        "source_result_manifest_hash": plan_a.source_result_manifest_hash,
        "plan_manifest_hash": plan_a.plan_manifest_hash,
        "batch_index": batch_a.batch_index,
        "batch_manifest_hash": batch_a.batch_manifest_hash,
        "message_content_hash": result_a.message_content_hash,
        "executor_name": execution_service.CONSISTENCY_CHECK_EXECUTOR_NAME,
        "executor_version": execution_service.CONSISTENCY_CHECK_EXECUTOR_VERSION,
        "planner_name": plan_a.planner_name,
        "planner_version": plan_a.planner_version,
        "prompt_contract_hash": PROMPT.contract_hash,
    }
    assert result_a.request_hash != result_b.request_hash
    assert captured_metadata[0]["source_result_manifest_hash"] != captured_metadata[1]["source_result_manifest_hash"]
    assert captured_metadata[0]["plan_manifest_hash"] != captured_metadata[1]["plan_manifest_hash"]
    assert captured_metadata[0]["batch_manifest_hash"] != captured_metadata[1]["batch_manifest_hash"]
    assert captured_metadata[0]["message_content_hash"] != captured_metadata[1]["message_content_hash"]


def test_execute_consistency_check_batch_invalid_response_does_not_leak_sensitive_sentinel(monkeypatch):
    session_factory = SessionFactory()
    project_id = uuid.uuid4()
    block_id = uuid.uuid4()
    candidate = _candidate(
        1,
        members=(
            _member(
                1,
                semantic_key_hash="1" * 64,
                value_json="A",
                evidences=(_evidence(1, document_block_id=block_id, excerpt="alpha", source_order=0),),
            ),
            _member(
                2,
                semantic_key_hash="2" * 64,
                value_json="B",
                evidences=(_evidence(2, document_block_id=block_id, excerpt="beta", source_order=1),),
            ),
        ),
    )
    plan, _batch = _plan((candidate,))
    orm_batch = _orm_batch(
        project_id=project_id,
        source_block_ids=(block_id,),
        texts_by_block_id={block_id: "Shared block"},
    )
    pending_run = _run(
        project_id=project_id,
        input_batch_id=orm_batch.id,
        request_hash=sha256("req-sentinel"),
        status=InferenceRunStatus.PENDING.value,
    )
    sentinel = "SENSITIVE_MODEL_TEXT_SENTINEL"
    warning_messages: list[str] = []

    async def fake_build_plan(_session_factory, *, consistency_application_id, config):
        return plan

    async def fake_fail(_session, *, run_id, failure_code, failure_message=None):
        return pending_run

    monkeypatch.setattr(execution_service, "build_consistency_check_plan", fake_build_plan)
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
    monkeypatch.setattr(execution_service, "fail_inference_run", fake_fail)
    monkeypatch.setattr(
        execution_service.logger,
        "warning",
        lambda message, *args, **kwargs: warning_messages.append(str(message)),
    )
    client = MockLLMClient([f"{sentinel} {{\"assessments\": []}}"])

    with pytest.raises(execution_service.AgentConsistencyCheckResponseError) as exc_info:
        run_async(
            execution_service.execute_consistency_check_batch(
                session_factory,
                project_id=project_id,
                plan=plan,
                batch_index=0,
                prompt=PROMPT,
                llm_client=client,
                provider="deepseek",
                requested_model="deepseek-v4-flash",
            )
        )

    assert sentinel not in str(exc_info.value)
    assert all(sentinel not in message for message in warning_messages)


def test_execution_source_does_not_import_httpx_or_create_conflict_records():
    source = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "services"
        / "consistency_check_execution.py"
    ).read_text(encoding="utf-8")
    assert "httpx" not in source
    assert "Conflict(" not in source
