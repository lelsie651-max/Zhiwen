from __future__ import annotations

import asyncio
import hashlib
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import CheckConstraint, Enum as SAEnum, String, UniqueConstraint

from app.models import Base
from app.models.document_content import DocumentBlock
from app.models.inference import (
    InferenceInputBatch,
    InferenceInputBlock,
    InferenceRun,
    InferenceRunStatus,
)
from app.repositories import inference as inference_repository
from app.schemas.inference import (
    InferenceInputBatchRead,
    InferenceInputBlockInternalRead,
    InferenceInputBlockRead,
    InferenceRunInternalRead,
    InferenceRunRead,
)
from app.services import inference as inference_service
from app.services.llm import make_stub_completion, LLMUsage


def run_async(awaitable):
    return asyncio.run(awaitable)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_source_block(
    *,
    raw_text: str,
    location_key: str,
    extraction_run_id: uuid.UUID | None = None,
    block_id: uuid.UUID | None = None,
    page_no: int | None = None,
) -> DocumentBlock:
    return DocumentBlock(
        id=block_id or uuid.uuid4(),
        extraction_run_id=extraction_run_id or uuid.uuid4(),
        source_order=0,
        block_type="paragraph",
        raw_text=raw_text,
        normalized_text=raw_text.strip(),
        location_key=location_key,
        anchor_hash=sha256(location_key + raw_text),
        block_index=0,
        heading_path=["H1"],
        page_no=page_no,
    )


def fake_row(block: DocumentBlock, *, project_id: uuid.UUID, status="completed", outcome="success"):
    return SimpleNamespace(
        DocumentBlock=block,
        run_status=status,
        run_outcome=outcome,
        project_id=project_id,
    )


class FakeSession:
    def __init__(self):
        self.committed = 0
        self.rolled_back = 0
        self.flushed = 0

    async def commit(self):
        self.committed += 1

    async def rollback(self):
        self.rolled_back += 1

    async def flush(self):
        self.flushed += 1


# --------------------------------------------------------------------------- #
# Model / schema structure
# --------------------------------------------------------------------------- #


def test_inference_tables_registered():
    assert {
        "inference_input_batches",
        "inference_input_blocks",
        "inference_runs",
    } <= set(Base.metadata.tables)


def test_single_migration_head_is_inference():
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    root = Path(__file__).resolve().parents[1]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    script = ScriptDirectory.from_config(config)
    assert list(script.get_heads()) == ["202607310300"]


def test_status_uses_string_and_check_not_native_enum():
    runs = Base.metadata.tables["inference_runs"]
    assert isinstance(runs.c.status.type, String)
    assert not isinstance(runs.c.status.type, SAEnum)
    assert not isinstance(runs.c.task_type.type, SAEnum)
    check_sql = {
        str(c.sqltext) for c in runs.constraints if isinstance(c, CheckConstraint)
    }
    assert any("status IN (" in sql for sql in check_sql)


def test_run_composite_unique_constraint_exists():
    runs = Base.metadata.tables["inference_runs"]
    assert any(
        isinstance(c, UniqueConstraint)
        and tuple(c.columns.keys())
        == ("input_batch_id", "agent_name", "prompt_version", "attempt_no")
        for c in runs.constraints
    )


def test_batch_and_block_unique_constraints_exist():
    batches = Base.metadata.tables["inference_input_batches"]
    assert any(
        isinstance(c, UniqueConstraint)
        and tuple(c.columns.keys()) == ("project_id", "task_type", "snapshot_hash")
        for c in batches.constraints
    )
    blocks = Base.metadata.tables["inference_input_blocks"]
    triples = {
        tuple(c.columns.keys())
        for c in blocks.constraints
        if isinstance(c, UniqueConstraint)
    }
    assert ("batch_id", "source_order") in triples
    assert ("batch_id", "block_ref") in triples
    assert ("batch_id", "source_block_id_snapshot") in triples


def test_content_text_is_preserved_verbatim_by_validator():
    spaced = "  leading and trailing  \n"
    block = InferenceInputBlock(
        source_order=0,
        block_ref="B0001",
        source_block_id_snapshot=uuid.uuid4(),
        extraction_run_id_snapshot=uuid.uuid4(),
        block_type="paragraph",
        location_key="loc-1",
        anchor_hash=sha256("a"),
        content_text=spaced,
        content_hash=sha256(spaced),
    )
    assert block.content_text == spaced  # not stripped/normalized


def test_block_ref_validator_rejects_bad_format():
    with pytest.raises(ValueError):
        InferenceInputBlock(
            source_order=0,
            block_ref="X1",
            source_block_id_snapshot=uuid.uuid4(),
            extraction_run_id_snapshot=uuid.uuid4(),
            block_type="paragraph",
            location_key="loc-1",
            anchor_hash=sha256("a"),
            content_text="body",
            content_hash=sha256("body"),
        )


def test_safe_reads_hide_sensitive_fields():
    assert "content_text" not in InferenceInputBlockRead.model_fields
    assert "content_text" in InferenceInputBlockInternalRead.model_fields
    assert "response_json" not in InferenceRunRead.model_fields
    assert "failure_message" not in InferenceRunRead.model_fields
    assert "response_json" in InferenceRunInternalRead.model_fields
    assert "failure_message" in InferenceRunInternalRead.model_fields


def test_service_does_not_touch_facts_or_schema():
    source = (
        Path(__file__).resolve().parents[1] / "app" / "services" / "inference.py"
    ).read_text(encoding="utf-8")
    assert "from app.models.fact" not in source
    assert "from app.models.dynamic_schema" not in source
    assert "Fact(" not in source


def test_repository_uses_explicit_joins_no_lazy_load():
    source = (
        Path(__file__).resolve().parents[1] / "app" / "repositories" / "inference.py"
    ).read_text(encoding="utf-8")
    assert ".join(" in source


# --------------------------------------------------------------------------- #
# Input batch service
# --------------------------------------------------------------------------- #


def _patch_batch_repo(
    monkeypatch,
    *,
    rows,
    project_status="active",
    identity_results=None,
    create_raises=None,
):
    project = SimpleNamespace(status=project_status)
    monkeypatch.setattr(
        inference_repository, "get_project", _acoro(lambda *a, **k: project)
    )
    monkeypatch.setattr(
        inference_repository,
        "get_blocks_with_extraction_context",
        _acoro(lambda *a, **k: rows),
    )

    identity_iter = iter(identity_results or [None])

    async def fake_identity(*a, **k):
        try:
            return next(identity_iter)
        except StopIteration:
            return None

    monkeypatch.setattr(inference_repository, "get_batch_by_identity", fake_identity)

    created = {"count": 0}

    async def fake_create(_session, batch):
        created["count"] += 1
        if create_raises is not None:
            raise create_raises
        return batch

    monkeypatch.setattr(
        inference_repository, "create_inference_batch_with_blocks", fake_create
    )
    return created


def _acoro(fn):
    async def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


def test_create_batch_builds_ordered_verbatim_snapshot(monkeypatch):
    project_id = uuid.uuid4()
    b1 = build_source_block(raw_text="First block  ", location_key="loc-1")
    b2 = build_source_block(raw_text="第二个块", location_key="loc-2")
    rows = [fake_row(b2, project_id=project_id), fake_row(b1, project_id=project_id)]
    _patch_batch_repo(monkeypatch, rows=rows)
    session = FakeSession()

    batch = run_async(
        inference_service.create_inference_input_batch(
            session,
            project_id=project_id,
            task_type="fact_extraction",
            block_ids=[b1.id, b2.id],  # order should be preserved regardless of row order
            selection_strategy="manual",
        )
    )

    assert batch.block_count == 2
    assert batch.character_count == len("First block  ") + len("第二个块")
    assert len(batch.snapshot_hash) == 64
    refs = [blk.block_ref for blk in batch.blocks]
    orders = [blk.source_order for blk in batch.blocks]
    assert refs == ["B0001", "B0002"]
    assert orders == [0, 1]
    # verbatim content + hash
    assert batch.blocks[0].content_text == "First block  "
    assert batch.blocks[0].content_hash == sha256("First block  ")
    assert batch.blocks[0].source_block_id_snapshot == b1.id
    assert batch.blocks[0].document_block_id == b1.id
    assert session.committed == 1


def test_create_batch_snapshot_hash_is_deterministic(monkeypatch):
    project_id = uuid.uuid4()
    b1 = build_source_block(raw_text="A", location_key="loc-1")
    b2 = build_source_block(raw_text="B", location_key="loc-2")
    rows = [fake_row(b1, project_id=project_id), fake_row(b2, project_id=project_id)]

    _patch_batch_repo(monkeypatch, rows=rows)
    batch1 = run_async(
        inference_service.create_inference_input_batch(
            FakeSession(),
            project_id=project_id,
            task_type="fact_extraction",
            block_ids=[b1.id, b2.id],
            selection_strategy="manual",
        )
    )
    _patch_batch_repo(monkeypatch, rows=rows)
    batch2 = run_async(
        inference_service.create_inference_input_batch(
            FakeSession(),
            project_id=project_id,
            task_type="fact_extraction",
            block_ids=[b1.id, b2.id],
            selection_strategy="manual",
        )
    )
    assert batch1.snapshot_hash == batch2.snapshot_hash


def test_create_batch_is_idempotent_when_identity_exists(monkeypatch):
    project_id = uuid.uuid4()
    b1 = build_source_block(raw_text="A", location_key="loc-1")
    rows = [fake_row(b1, project_id=project_id)]
    existing = InferenceInputBatch(
        project_id=project_id,
        task_type="fact_extraction",
        selection_strategy="manual",
        block_count=1,
        character_count=1,
        snapshot_hash=sha256("x"),
    )
    created = _patch_batch_repo(monkeypatch, rows=rows, identity_results=[existing])
    result = run_async(
        inference_service.create_inference_input_batch(
            FakeSession(),
            project_id=project_id,
            task_type="fact_extraction",
            block_ids=[b1.id],
            selection_strategy="manual",
        )
    )
    assert result is existing
    assert created["count"] == 0


def test_create_batch_reraises_query_on_integrity_conflict(monkeypatch):
    from sqlalchemy.exc import IntegrityError

    project_id = uuid.uuid4()
    b1 = build_source_block(raw_text="A", location_key="loc-1")
    rows = [fake_row(b1, project_id=project_id)]
    existing = InferenceInputBatch(
        project_id=project_id,
        task_type="fact_extraction",
        selection_strategy="manual",
        block_count=1,
        character_count=1,
        snapshot_hash=sha256("x"),
    )
    # First identity check returns None, after conflict returns existing.
    created = _patch_batch_repo(
        monkeypatch,
        rows=rows,
        identity_results=[None, existing],
        create_raises=IntegrityError("dup", {}, Exception()),
    )
    session = FakeSession()
    result = run_async(
        inference_service.create_inference_input_batch(
            session,
            project_id=project_id,
            task_type="fact_extraction",
            block_ids=[b1.id],
            selection_strategy="manual",
        )
    )
    assert result is existing
    assert created["count"] == 1
    assert session.rolled_back == 1


def test_create_batch_rejects_duplicate_block_ids(monkeypatch):
    b1 = build_source_block(raw_text="A", location_key="loc-1")
    with pytest.raises(inference_service.InvalidInferenceInputError):
        run_async(
            inference_service.create_inference_input_batch(
                FakeSession(),
                project_id=uuid.uuid4(),
                task_type="fact_extraction",
                block_ids=[b1.id, b1.id],
                selection_strategy="manual",
            )
        )


def test_create_batch_rejects_cross_project_block(monkeypatch):
    project_id = uuid.uuid4()
    b1 = build_source_block(raw_text="A", location_key="loc-1")
    rows = [fake_row(b1, project_id=uuid.uuid4())]  # different project
    _patch_batch_repo(monkeypatch, rows=rows)
    with pytest.raises(inference_service.InvalidInferenceInputError):
        run_async(
            inference_service.create_inference_input_batch(
                FakeSession(),
                project_id=project_id,
                task_type="fact_extraction",
                block_ids=[b1.id],
                selection_strategy="manual",
            )
        )


@pytest.mark.parametrize(
    ("status", "outcome"),
    [("failed", "failed"), ("completed", "needs_ocr"), ("running", "success")],
)
def test_create_batch_rejects_non_admissible_extraction(monkeypatch, status, outcome):
    project_id = uuid.uuid4()
    b1 = build_source_block(raw_text="A", location_key="loc-1")
    rows = [fake_row(b1, project_id=project_id, status=status, outcome=outcome)]
    _patch_batch_repo(monkeypatch, rows=rows)
    with pytest.raises(inference_service.InferenceBlockNotReadyError):
        run_async(
            inference_service.create_inference_input_batch(
                FakeSession(),
                project_id=project_id,
                task_type="fact_extraction",
                block_ids=[b1.id],
                selection_strategy="manual",
            )
        )


def test_create_batch_rejects_missing_block(monkeypatch):
    project_id = uuid.uuid4()
    b1 = build_source_block(raw_text="A", location_key="loc-1")
    _patch_batch_repo(monkeypatch, rows=[])  # nothing returned
    with pytest.raises(inference_service.InferenceBlockNotFoundError):
        run_async(
            inference_service.create_inference_input_batch(
                FakeSession(),
                project_id=project_id,
                task_type="fact_extraction",
                block_ids=[b1.id],
                selection_strategy="manual",
            )
        )


def test_create_batch_rejects_inactive_project(monkeypatch):
    project_id = uuid.uuid4()
    b1 = build_source_block(raw_text="A", location_key="loc-1")
    _patch_batch_repo(monkeypatch, rows=[fake_row(b1, project_id=project_id)], project_status="archived")
    with pytest.raises(inference_service.InferenceProjectInactiveError):
        run_async(
            inference_service.create_inference_input_batch(
                FakeSession(),
                project_id=project_id,
                task_type="fact_extraction",
                block_ids=[b1.id],
                selection_strategy="manual",
            )
        )


def test_create_batch_rejects_unknown_task_type(monkeypatch):
    with pytest.raises(inference_service.InvalidInferenceInputError):
        run_async(
            inference_service.create_inference_input_batch(
                FakeSession(),
                project_id=uuid.uuid4(),
                task_type="translate",
                block_ids=[uuid.uuid4()],
                selection_strategy="manual",
            )
        )


# --------------------------------------------------------------------------- #
# Run lifecycle service
# --------------------------------------------------------------------------- #


def _fake_batch(project_id, *, task_type="fact_extraction", snapshot_hash=None):
    return SimpleNamespace(
        id=uuid.uuid4(),
        project_id=project_id,
        task_type=task_type,
        snapshot_hash=snapshot_hash or sha256("snap"),
    )


def _patch_create_run_repo(monkeypatch, *, batch, attempt_no=1):
    monkeypatch.setattr(
        inference_repository, "get_batch_for_update", _acoro(lambda *a, **k: batch)
    )
    monkeypatch.setattr(
        inference_repository,
        "get_next_run_attempt_no",
        _acoro(lambda *a, **k: attempt_no),
    )

    async def fake_create(_session, run):
        return run

    monkeypatch.setattr(inference_repository, "create_inference_run", fake_create)


def _create_run_kwargs(project_id, batch_id):
    return dict(
        project_id=project_id,
        input_batch_id=batch_id,
        task_type="fact_extraction",
        agent_name="extractor",
        agent_version="1.0.0",
        prompt_name="fact_v1",
        prompt_version="1",
        prompt_contract_hash=sha256("contract"),
        provider="deepseek",
        requested_model="deepseek-v4-flash",
        temperature=0.2,
        max_output_tokens=4096,
        request_metadata={"note": "x"},
    )


def test_create_run_initial_pending_state_and_request_hash(monkeypatch):
    project_id = uuid.uuid4()
    batch = _fake_batch(project_id)
    _patch_create_run_repo(monkeypatch, batch=batch)
    run = run_async(
        inference_service.create_inference_run(
            FakeSession(), **_create_run_kwargs(project_id, batch.id)
        )
    )
    assert run.status == InferenceRunStatus.PENDING.value
    assert run.attempt_no == 1
    assert run.attempt_count == 0
    assert run.started_at is None and run.completed_at is None
    assert run.response_json is None
    assert len(run.request_hash) == 64


def test_create_run_request_hash_deterministic(monkeypatch):
    project_id = uuid.uuid4()
    batch = _fake_batch(project_id)
    _patch_create_run_repo(monkeypatch, batch=batch)
    run1 = run_async(
        inference_service.create_inference_run(
            FakeSession(), **_create_run_kwargs(project_id, batch.id)
        )
    )
    run2 = run_async(
        inference_service.create_inference_run(
            FakeSession(), **_create_run_kwargs(project_id, batch.id)
        )
    )
    assert run1.request_hash == run2.request_hash


def test_create_run_rejects_project_or_task_mismatch(monkeypatch):
    project_id = uuid.uuid4()
    batch = _fake_batch(uuid.uuid4())  # different project
    _patch_create_run_repo(monkeypatch, batch=batch)
    with pytest.raises(inference_service.InferenceBatchMismatchError):
        run_async(
            inference_service.create_inference_run(
                FakeSession(), **_create_run_kwargs(project_id, batch.id)
            )
        )

    batch2 = _fake_batch(project_id, task_type="schema_inference")
    _patch_create_run_repo(monkeypatch, batch=batch2)
    with pytest.raises(inference_service.InferenceBatchMismatchError):
        run_async(
            inference_service.create_inference_run(
                FakeSession(), **_create_run_kwargs(project_id, batch2.id)
            )
        )


def _pending_run(project_id):
    return InferenceRun(
        id=uuid.uuid4(),
        project_id=project_id,
        input_batch_id=uuid.uuid4(),
        task_type="fact_extraction",
        attempt_no=1,
        status=InferenceRunStatus.PENDING.value,
        agent_name="extractor",
        agent_version="1.0.0",
        prompt_name="fact_v1",
        prompt_version="1",
        provider="deepseek",
        requested_model="deepseek-v4-flash",
        temperature=0.2,
        max_output_tokens=4096,
        attempt_count=0,
    )


def _patch_run_lookup(monkeypatch, run):
    monkeypatch.setattr(
        inference_repository, "get_run_for_update", _acoro(lambda *a, **k: run)
    )


def test_start_pending_to_running_and_idempotent(monkeypatch):
    run = _pending_run(uuid.uuid4())
    _patch_run_lookup(monkeypatch, run)
    started = run_async(inference_service.start_inference_run(FakeSession(), run_id=run.id))
    assert started.status == InferenceRunStatus.RUNNING.value
    assert started.started_at is not None
    first_started = started.started_at
    # idempotent: still running, started_at unchanged
    again = run_async(inference_service.start_inference_run(FakeSession(), run_id=run.id))
    assert again.started_at == first_started


def test_start_rejects_terminal(monkeypatch):
    run = _pending_run(uuid.uuid4())
    run.status = InferenceRunStatus.COMPLETED.value
    _patch_run_lookup(monkeypatch, run)
    with pytest.raises(inference_service.InferenceRunStateError):
        run_async(inference_service.start_inference_run(FakeSession(), run_id=run.id))


def _running_run(project_id, provider="deepseek"):
    run = _pending_run(project_id)
    run.provider = provider
    run.status = InferenceRunStatus.RUNNING.value
    run.started_at = inference_service.utc_now()
    return run


def test_complete_writes_identity_usage_and_hash(monkeypatch):
    run = _running_run(uuid.uuid4())
    _patch_run_lookup(monkeypatch, run)
    content = '{"facts": [{"k": "v"}]}'
    completion = make_stub_completion(
        content,
        provider="deepseek",
        model="deepseek-v4-flash",
        response_id="resp-9",
        system_fingerprint="fp_9",
        usage=LLMUsage(11, 22, 33, 5, 6, 7),
        attempt_count=2,
    )
    completed = run_async(
        inference_service.complete_inference_run(
            FakeSession(), run_id=run.id, completion=completion
        )
    )
    assert completed.status == InferenceRunStatus.COMPLETED.value
    assert completed.response_model == "deepseek-v4-flash"
    assert completed.response_id == "resp-9"
    assert completed.finish_reason == "stop"
    assert completed.attempt_count == 2
    assert completed.prompt_tokens == 11
    assert completed.reasoning_tokens == 7
    assert completed.response_json == {"facts": [{"k": "v"}]}
    assert completed.response_hash == sha256(content)
    assert completed.completed_at is not None


def test_complete_rejects_provider_mismatch(monkeypatch):
    run = _running_run(uuid.uuid4(), provider="deepseek")
    _patch_run_lookup(monkeypatch, run)
    completion = make_stub_completion('{"a": 1}', provider="openai")
    with pytest.raises(inference_service.InferenceProviderMismatchError):
        run_async(
            inference_service.complete_inference_run(
                FakeSession(), run_id=run.id, completion=completion
            )
        )


def test_complete_requires_finish_reason_stop(monkeypatch):
    run = _running_run(uuid.uuid4())
    _patch_run_lookup(monkeypatch, run)
    completion = make_stub_completion('{"a": 1}', provider="deepseek", finish_reason="length")
    with pytest.raises(inference_service.InvalidInferenceCompletionError):
        run_async(
            inference_service.complete_inference_run(
                FakeSession(), run_id=run.id, completion=completion
            )
        )


@pytest.mark.parametrize(
    "content",
    [
        "```json\n{\"a\": 1}\n```",
        'Here is the answer: {"a": 1}',
        "[{\"a\": 1}]",
        "42",
    ],
)
def test_complete_enforces_strict_json_object(monkeypatch, content):
    run = _running_run(uuid.uuid4())
    _patch_run_lookup(monkeypatch, run)
    completion = make_stub_completion(content, provider="deepseek")
    with pytest.raises(inference_service.InvalidInferenceCompletionError):
        run_async(
            inference_service.complete_inference_run(
                FakeSession(), run_id=run.id, completion=completion
            )
        )


def test_complete_idempotent_same_hash_conflict_on_different(monkeypatch):
    run = _running_run(uuid.uuid4())
    _patch_run_lookup(monkeypatch, run)
    content = '{"a": 1}'
    completion = make_stub_completion(content, provider="deepseek")
    run_async(
        inference_service.complete_inference_run(
            FakeSession(), run_id=run.id, completion=completion
        )
    )
    # same content again -> idempotent
    again = run_async(
        inference_service.complete_inference_run(
            FakeSession(), run_id=run.id, completion=completion
        )
    )
    assert again.status == InferenceRunStatus.COMPLETED.value
    # different content -> conflict
    other = make_stub_completion('{"a": 2}', provider="deepseek")
    with pytest.raises(inference_service.InferenceCompletionConflictError):
        run_async(
            inference_service.complete_inference_run(
                FakeSession(), run_id=run.id, completion=other
            )
        )


def test_complete_rejects_non_running(monkeypatch):
    run = _pending_run(uuid.uuid4())  # still pending
    _patch_run_lookup(monkeypatch, run)
    completion = make_stub_completion('{"a": 1}', provider="deepseek")
    with pytest.raises(inference_service.InferenceRunStateError):
        run_async(
            inference_service.complete_inference_run(
                FakeSession(), run_id=run.id, completion=completion
            )
        )


def test_fail_from_pending_stamps_started_and_completed(monkeypatch):
    run = _pending_run(uuid.uuid4())
    _patch_run_lookup(monkeypatch, run)
    failed = run_async(
        inference_service.fail_inference_run(
            FakeSession(), run_id=run.id, failure_code="upstream_unavailable"
        )
    )
    assert failed.status == InferenceRunStatus.FAILED.value
    assert failed.started_at is not None
    assert failed.completed_at is not None
    assert failed.failure_code == "upstream_unavailable"


def test_fail_from_running_keeps_started_at(monkeypatch):
    run = _running_run(uuid.uuid4())
    original_started = run.started_at
    _patch_run_lookup(monkeypatch, run)
    failed = run_async(
        inference_service.fail_inference_run(
            FakeSession(), run_id=run.id, failure_code="rate_limited", failure_message="429"
        )
    )
    assert failed.started_at == original_started
    assert failed.completed_at is not None


def test_fail_idempotent_and_completed_cannot_fail(monkeypatch):
    run = _running_run(uuid.uuid4())
    _patch_run_lookup(monkeypatch, run)
    run_async(
        inference_service.fail_inference_run(
            FakeSession(), run_id=run.id, failure_code="network_error"
        )
    )
    again = run_async(
        inference_service.fail_inference_run(
            FakeSession(), run_id=run.id, failure_code="network_error"
        )
    )
    assert again.status == InferenceRunStatus.FAILED.value
    with pytest.raises(inference_service.InferenceFailureConflictError):
        run_async(
            inference_service.fail_inference_run(
                FakeSession(), run_id=run.id, failure_code="different_code"
            )
        )


def test_completed_run_cannot_be_failed(monkeypatch):
    run = _running_run(uuid.uuid4())
    run.status = InferenceRunStatus.COMPLETED.value
    _patch_run_lookup(monkeypatch, run)
    with pytest.raises(inference_service.InferenceRunStateError):
        run_async(
            inference_service.fail_inference_run(
                FakeSession(), run_id=run.id, failure_code="network_error"
            )
        )


def test_fail_rejects_exception_objects(monkeypatch):
    run = _running_run(uuid.uuid4())
    _patch_run_lookup(monkeypatch, run)
    with pytest.raises(inference_service.InvalidInferenceFailureError):
        run_async(
            inference_service.fail_inference_run(
                FakeSession(), run_id=run.id, failure_code=ValueError("boom")  # type: ignore[arg-type]
            )
        )
