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


_BATCH_IDENTITY_CONSTRAINT = (
    "uq_inference_input_batches_project_id_task_type_snapshot_hash"
)


def make_integrity_error(constraint_name: str | None):
    """Build an IntegrityError whose orig.diag.constraint_name mimics psycopg."""
    from sqlalchemy.exc import IntegrityError

    orig = SimpleNamespace(diag=SimpleNamespace(constraint_name=constraint_name))
    return IntegrityError("stmt", {}, orig)


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


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class ContextSession:
    def __init__(self, rows):
        self._rows = rows

    async def execute(self, _statement):
        return FakeResult(self._rows)


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
    assert list(script.get_heads()) == ["202607311600"]


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
    assert "get_completed_fact_extraction_run_context" in source
    assert "content_text" not in source
    assert "extraction_run_id_snapshot" in source
    assert "source_block_id_snapshot" in source
    assert ".outerjoin(" in source


def test_completed_fact_extraction_run_context_returns_source_block_snapshots():
    run_id = uuid.uuid4()
    project_id = uuid.uuid4()
    batch_id = uuid.uuid4()
    extraction_run_id = uuid.uuid4()
    source_block_id = uuid.uuid4()
    other_block_id = uuid.uuid4()
    rows = [
        SimpleNamespace(
            run_id=run_id,
            project_id=project_id,
            task_type="fact_extraction",
            status="completed",
            batch_id=batch_id,
            extraction_run_id_snapshot=extraction_run_id,
            source_block_id_snapshot=source_block_id,
        ),
        SimpleNamespace(
            run_id=run_id,
            project_id=project_id,
            task_type="fact_extraction",
            status="completed",
            batch_id=batch_id,
            extraction_run_id_snapshot=extraction_run_id,
            source_block_id_snapshot=source_block_id,
        ),
        SimpleNamespace(
            run_id=run_id,
            project_id=project_id,
            task_type="fact_extraction",
            status="completed",
            batch_id=batch_id,
            extraction_run_id_snapshot=None,
            source_block_id_snapshot=other_block_id,
        ),
    ]

    context = run_async(
        inference_repository.get_completed_fact_extraction_run_context(
            ContextSession(rows),
            inference_run_id=run_id,
        )
    )

    assert context is not None
    assert context.extraction_run_id_snapshots == frozenset({extraction_run_id})
    assert context.source_block_id_snapshots == frozenset({source_block_id, other_block_id})


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
    assert created["count"] == 0
    assert session.committed == 1  # idempotent path still commits to release the txn


def test_create_batch_idempotent_only_on_target_constraint(monkeypatch):
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
    # First identity check returns None, after the target conflict returns existing.
    created = _patch_batch_repo(
        monkeypatch,
        rows=rows,
        identity_results=[None, existing],
        create_raises=make_integrity_error(_BATCH_IDENTITY_CONSTRAINT),
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
    assert session.committed == 1  # commit the re-query txn


def test_create_batch_does_not_swallow_unrelated_integrity_error(monkeypatch):
    from sqlalchemy.exc import IntegrityError

    project_id = uuid.uuid4()
    b1 = build_source_block(raw_text="A", location_key="loc-1")
    rows = [fake_row(b1, project_id=project_id)]
    session = FakeSession()
    _patch_batch_repo(
        monkeypatch,
        rows=rows,
        identity_results=[None],
        create_raises=make_integrity_error("some_other_constraint"),
    )
    with pytest.raises(IntegrityError):
        run_async(
            inference_service.create_inference_input_batch(
                session,
                project_id=project_id,
                task_type="fact_extraction",
                block_ids=[b1.id],
                selection_strategy="manual",
            )
        )
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


# --------------------------------------------------------------------------- #
# Round: transaction closure, identity hashing, metadata, DB invariants
# --------------------------------------------------------------------------- #


def test_lifecycle_idempotent_paths_commit_to_release_lock(monkeypatch):
    # start: already running -> idempotent commit
    run = _running_run(uuid.uuid4())
    _patch_run_lookup(monkeypatch, run)
    s = FakeSession()
    run_async(inference_service.start_inference_run(s, run_id=run.id))
    assert s.committed == 1 and s.rolled_back == 0

    # complete: same response hash -> idempotent commit
    content = '{"a": 1}'
    completed = _running_run(uuid.uuid4())
    completed.status = InferenceRunStatus.COMPLETED.value
    completed.response_hash = sha256(content)
    _patch_run_lookup(monkeypatch, completed)
    s2 = FakeSession()
    run_async(
        inference_service.complete_inference_run(
            s2, run_id=completed.id, completion=make_stub_completion(content, provider="deepseek")
        )
    )
    assert s2.committed == 1 and s2.rolled_back == 0

    # fail: same failure result -> idempotent commit
    failed = _running_run(uuid.uuid4())
    failed.status = InferenceRunStatus.FAILED.value
    failed.failure_code = "network_error"
    failed.failure_message = None
    _patch_run_lookup(monkeypatch, failed)
    s3 = FakeSession()
    run_async(inference_service.fail_inference_run(s3, run_id=failed.id, failure_code="network_error"))
    assert s3.committed == 1 and s3.rolled_back == 0


def test_lifecycle_error_paths_rollback(monkeypatch):
    # start on terminal
    run = _pending_run(uuid.uuid4())
    run.status = InferenceRunStatus.COMPLETED.value
    _patch_run_lookup(monkeypatch, run)
    s = FakeSession()
    with pytest.raises(inference_service.InferenceRunStateError):
        run_async(inference_service.start_inference_run(s, run_id=run.id))
    assert s.rolled_back == 1 and s.committed == 0

    # complete provider mismatch
    r2 = _running_run(uuid.uuid4(), provider="deepseek")
    _patch_run_lookup(monkeypatch, r2)
    s2 = FakeSession()
    with pytest.raises(inference_service.InferenceProviderMismatchError):
        run_async(
            inference_service.complete_inference_run(
                s2, run_id=r2.id, completion=make_stub_completion('{"a":1}', provider="openai")
            )
        )
    assert s2.rolled_back == 1 and s2.committed == 0

    # complete conflict (already completed, different hash)
    r3 = _running_run(uuid.uuid4())
    r3.status = InferenceRunStatus.COMPLETED.value
    r3.response_hash = sha256("different")
    _patch_run_lookup(monkeypatch, r3)
    s3 = FakeSession()
    with pytest.raises(inference_service.InferenceCompletionConflictError):
        run_async(
            inference_service.complete_inference_run(
                s3, run_id=r3.id, completion=make_stub_completion('{"a":1}', provider="deepseek")
            )
        )
    assert s3.rolled_back == 1 and s3.committed == 0

    # complete strict-json failure
    r4 = _running_run(uuid.uuid4())
    _patch_run_lookup(monkeypatch, r4)
    s4 = FakeSession()
    with pytest.raises(inference_service.InvalidInferenceCompletionError):
        run_async(
            inference_service.complete_inference_run(
                s4, run_id=r4.id, completion=make_stub_completion("not json", provider="deepseek")
            )
        )
    assert s4.rolled_back == 1 and s4.committed == 0

    # fail conflict
    r5 = _running_run(uuid.uuid4())
    r5.status = InferenceRunStatus.FAILED.value
    r5.failure_code = "code_a"
    _patch_run_lookup(monkeypatch, r5)
    s5 = FakeSession()
    with pytest.raises(inference_service.InferenceFailureConflictError):
        run_async(inference_service.fail_inference_run(s5, run_id=r5.id, failure_code="code_b"))
    assert s5.rolled_back == 1 and s5.committed == 0


def test_create_run_batch_mismatch_rolls_back(monkeypatch):
    project_id = uuid.uuid4()
    batch = _fake_batch(uuid.uuid4())  # different project
    _patch_create_run_repo(monkeypatch, batch=batch)
    s = FakeSession()
    with pytest.raises(inference_service.InferenceBatchMismatchError):
        run_async(inference_service.create_inference_run(s, **_create_run_kwargs(project_id, batch.id)))
    assert s.rolled_back == 1 and s.committed == 0


def test_identity_normalized_before_query_and_hash(monkeypatch):
    project_id = uuid.uuid4()
    batch = _fake_batch(project_id)
    captured = {}

    async def fake_attempt(session, input_batch_id, agent_name, prompt_version):
        captured["agent_name"] = agent_name
        captured["prompt_version"] = prompt_version
        return 1

    monkeypatch.setattr(inference_repository, "get_batch_for_update", _acoro(lambda *a, **k: batch))
    monkeypatch.setattr(inference_repository, "get_next_run_attempt_no", fake_attempt)

    async def fake_create(_s, run):
        return run

    monkeypatch.setattr(inference_repository, "create_inference_run", fake_create)

    kwargs = _create_run_kwargs(project_id, batch.id)
    kwargs.update(agent_name="  extractor  ", prompt_version="  1  ", requested_model=" deepseek-v4-flash ")
    run = run_async(inference_service.create_inference_run(FakeSession(), **kwargs))

    # attempt query received normalized values
    assert captured == {"agent_name": "extractor", "prompt_version": "1"}
    # persisted values are normalized
    assert run.agent_name == "extractor"
    assert run.prompt_version == "1"
    assert run.requested_model == "deepseek-v4-flash"


def test_request_hash_recomputable_from_persisted_fields(monkeypatch):
    project_id = uuid.uuid4()
    batch = _fake_batch(project_id)
    _patch_create_run_repo(monkeypatch, batch=batch)
    kwargs = _create_run_kwargs(project_id, batch.id)
    kwargs.update(agent_name="  extractor  ")  # whitespace should not break recompute
    run = run_async(inference_service.create_inference_run(FakeSession(), **kwargs))

    payload = {
        "snapshot_hash": batch.snapshot_hash,
        "task_type": run.task_type,
        "agent_name": run.agent_name,
        "agent_version": run.agent_version,
        "prompt_name": run.prompt_name,
        "prompt_version": run.prompt_version,
        "prompt_contract_hash": run.prompt_contract_hash,
        "provider": run.provider,
        "requested_model": run.requested_model,
        "temperature": run.temperature,
        "max_output_tokens": run.max_output_tokens,
        "request_metadata": run.request_metadata,
    }
    expected = inference_service._sha256(inference_service._canonical_json(payload))
    assert run.request_hash == expected


def test_negative_zero_temperature_hashes_same_as_zero(monkeypatch):
    project_id = uuid.uuid4()
    batch = _fake_batch(project_id)
    _patch_create_run_repo(monkeypatch, batch=batch)
    k1 = _create_run_kwargs(project_id, batch.id)
    k1["temperature"] = -0.0
    run1 = run_async(inference_service.create_inference_run(FakeSession(), **k1))
    _patch_create_run_repo(monkeypatch, batch=batch)
    k2 = _create_run_kwargs(project_id, batch.id)
    k2["temperature"] = 0.0
    run2 = run_async(inference_service.create_inference_run(FakeSession(), **k2))
    assert run1.request_hash == run2.request_hash


@pytest.mark.parametrize(
    "bad_metadata",
    [
        {"score": float("nan")},
        {"score": float("inf")},
        {1: "non-string-key"},
        {"id": uuid.uuid4()},
        {"created": __import__("datetime").datetime(2026, 1, 1)},
        {"tags": {1, 2, 3}},
    ],
)
def test_metadata_rejects_non_json(monkeypatch, bad_metadata):
    b1 = build_source_block(raw_text="A", location_key="loc-1")
    project_id = uuid.uuid4()
    _patch_batch_repo(monkeypatch, rows=[fake_row(b1, project_id=project_id)])
    with pytest.raises(inference_service.InvalidInferenceInputError):
        run_async(
            inference_service.create_inference_input_batch(
                FakeSession(),
                project_id=project_id,
                task_type="fact_extraction",
                block_ids=[b1.id],
                selection_strategy="manual",
                selection_metadata=bad_metadata,
            )
        )


def test_metadata_is_deep_copied(monkeypatch):
    project_id = uuid.uuid4()
    b1 = build_source_block(raw_text="A", location_key="loc-1")
    _patch_batch_repo(monkeypatch, rows=[fake_row(b1, project_id=project_id)])
    original = {"nested": {"k": [1, 2]}}
    batch = run_async(
        inference_service.create_inference_input_batch(
            FakeSession(),
            project_id=project_id,
            task_type="fact_extraction",
            block_ids=[b1.id],
            selection_strategy="manual",
            selection_metadata=original,
        )
    )
    original["nested"]["k"].append(999)
    assert batch.selection_metadata == {"nested": {"k": [1, 2]}}


def test_batch_by_identity_uses_selectinload_and_ordered_blocks():
    source = (
        Path(__file__).resolve().parents[1] / "app" / "repositories" / "inference.py"
    ).read_text(encoding="utf-8")
    assert "selectinload" in source
    rel = InferenceInputBatch.__mapper__.relationships["blocks"]
    assert rel.order_by  # ordering configured
    assert "source_order" in str(rel.order_by[0])


def test_pending_running_db_constraints_forbid_response_identity():
    from sqlalchemy import CheckConstraint

    runs = Base.metadata.tables["inference_runs"]
    checks = {
        c.name: str(c.sqltext)
        for c in runs.constraints
        if isinstance(c, CheckConstraint)
    }
    pending = next(v for k, v in checks.items() if "pending_shape" in k)
    running = next(v for k, v in checks.items() if "running_shape" in k)
    for sql in (pending, running):
        for field in (
            "response_model IS NULL",
            "response_id IS NULL",
            "system_fingerprint IS NULL",
            "finish_reason IS NULL",
            "prompt_tokens IS NULL",
            "reasoning_tokens IS NULL",
        ):
            assert field in sql


def test_jsonb_type_constraints_exist():
    from sqlalchemy import CheckConstraint

    def checks(table):
        return " ".join(
            str(c.sqltext)
            for c in Base.metadata.tables[table].constraints
            if isinstance(c, CheckConstraint)
        )

    assert "jsonb_typeof(selection_metadata) = 'object'" in checks("inference_input_batches")
    assert "jsonb_typeof(request_metadata) = 'object'" in checks("inference_runs")
    assert "jsonb_typeof(heading_path) = 'array'" in checks("inference_input_blocks")
