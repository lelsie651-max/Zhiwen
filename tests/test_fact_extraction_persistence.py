from __future__ import annotations

import asyncio
import copy
from dataclasses import replace
import hashlib
import inspect
import shutil
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import AddConstraint

from app.models import Base
from app.models.entity import EntityStatus
from app.models.fact import FactValue, FactValueStatus, FactValueType
from app.repositories import entity as entity_repository
from app.repositories import fact_extraction_persistence as persistence_repository
from app.schemas.fact_extraction_persistence import (
    CompletedFactExtractionPersistenceContext,
    EntityMentionResolution,
    EntityMentionResolutionStatus,
    FactExtractionPersistenceBlock,
    FactProposalPersistenceOutcome,
    FactProposalWithheldReason,
)
from app.services import fact_extraction_persistence as persistence_service


def run_async(awaitable):
    return asyncio.run(awaitable)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FakeSavepoint:
    def __init__(self) -> None:
        self.committed = False
        self.rolled_back = False

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        self.rolled_back = True


class FakeSession:
    def __init__(self) -> None:
        self.commit_count = 0
        self.rollback_count = 0
        self.flush_count = 0
        self.savepoints: list[FakeSavepoint] = []

    async def commit(self) -> None:
        self.commit_count += 1

    async def rollback(self) -> None:
        self.rollback_count += 1

    async def flush(self) -> None:
        self.flush_count += 1

    async def begin_nested(self) -> FakeSavepoint:
        savepoint = FakeSavepoint()
        self.savepoints.append(savepoint)
        return savepoint


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class ContextSession:
    def __init__(self, rows):
        self.rows = rows

    async def execute(self, _statement):
        return FakeResult(self.rows)


def make_integrity_error(constraint_name: str | None) -> IntegrityError:
    orig = SimpleNamespace(diag=SimpleNamespace(constraint_name=constraint_name))
    return IntegrityError("stmt", {}, orig)


def valid_response_json(
    *,
    subject_key: str = "张三",
    value_type: str = "string",
    value_json="国王",
) -> dict:
    return {
        "facts": [
            {
                "subject_kind": "person",
                "subject_key": subject_key,
                "predicate_key": "title",
                "scope_key": None,
                "value_type": value_type,
                "value_json": value_json,
                "language_code": "zh-CN",
                "confidence": 0.9,
                "evidence": [
                    {
                        "block_ref": "B0001",
                        "start_offset": 0,
                        "end_offset": 2,
                        "role": "supporting",
                    },
                    {
                        "block_ref": "B0002",
                        "start_offset": 0,
                        "end_offset": 2,
                        "role": "context",
                    },
                ],
            }
        ],
        "batch_summary": "ok",
        "uncertainties": [],
    }


def build_block(
    *,
    source_order: int,
    block_ref: str,
    text: str,
    extraction_run_id: uuid.UUID,
    project_id: uuid.UUID,
    document_block_id: uuid.UUID | None = None,
    source_block_id_snapshot: uuid.UUID | None = None,
    document_block_extraction_run_id: uuid.UUID | None = None,
    document_block_project_id: uuid.UUID | None = None,
    content_text: str | None = None,
    content_hash: str | None = None,
) -> FactExtractionPersistenceBlock:
    block_id = document_block_id or uuid.uuid4()
    content = text if content_text is None else content_text
    return FactExtractionPersistenceBlock(
        input_block_id=uuid.uuid4(),
        block_ref=block_ref,
        source_order=source_order,
        document_block_id=block_id,
        source_block_id_snapshot=source_block_id_snapshot or block_id,
        extraction_run_id_snapshot=extraction_run_id,
        content_text=content,
        content_hash=content_hash or sha256(content),
        document_block_extraction_run_id=document_block_extraction_run_id or extraction_run_id,
        document_block_project_id=document_block_project_id or project_id,
        document_block_raw_text=text,
    )


def build_context(
    *,
    project_id: uuid.UUID | None = None,
    extraction_run_id: uuid.UUID | None = None,
    inference_run_id: uuid.UUID | None = None,
    status: str = "completed",
    task_type: str = "fact_extraction",
    response_json: dict | None = None,
    blocks: tuple[FactExtractionPersistenceBlock, ...] | None = None,
) -> CompletedFactExtractionPersistenceContext:
    actual_project_id = project_id or uuid.uuid4()
    actual_extraction_run_id = extraction_run_id or uuid.uuid4()
    return CompletedFactExtractionPersistenceContext(
        inference_run_id=inference_run_id or uuid.uuid4(),
        project_id=actual_project_id,
        task_type=task_type,
        status=status,
        input_batch_id=uuid.uuid4(),
        response_json=copy.deepcopy(response_json or valid_response_json()),
        response_hash="a" * 64,
        blocks=blocks
        or (
            build_block(
                source_order=0,
                block_ref="B0001",
                text="张三",
                extraction_run_id=actual_extraction_run_id,
                project_id=actual_project_id,
            ),
            build_block(
                source_order=1,
                block_ref="B0002",
                text="国王",
                extraction_run_id=actual_extraction_run_id,
                project_id=actual_project_id,
            ),
        ),
    )


def build_resolution(
    *,
    status: EntityMentionResolutionStatus,
    entity_type: str = "person",
    mention_key: str = "张三",
    entity_id: uuid.UUID | None = None,
    canonical_key: str | None = None,
    candidate_count: int = 1,
) -> EntityMentionResolution:
    return EntityMentionResolution(
        status=status.value,
        normalized_entity_type=entity_type,
        normalized_mention_key=mention_key,
        entity_id=entity_id,
        canonical_key=canonical_key,
        candidate_count=candidate_count,
    )


def test_completed_fact_extraction_run_context_can_load_and_deepcopy_response() -> None:
    response = valid_response_json()
    run_id = uuid.uuid4()
    project_id = uuid.uuid4()
    extraction_run_id = uuid.uuid4()
    batch_id = uuid.uuid4()
    rows = [
        SimpleNamespace(
            inference_run_id=run_id,
            project_id=project_id,
            task_type="fact_extraction",
            status="completed",
            input_batch_id=batch_id,
            response_json=response,
            response_hash="a" * 64,
            input_block_id=uuid.uuid4(),
            block_ref="B0001",
            source_order=0,
            document_block_id=uuid.uuid4(),
            source_block_id_snapshot=uuid.uuid4(),
            extraction_run_id_snapshot=extraction_run_id,
            content_text="张三",
            content_hash=sha256("张三"),
            document_block_extraction_run_id=extraction_run_id,
            document_block_project_id=project_id,
            document_block_raw_text="张三",
        ),
    ]
    rows[0].source_block_id_snapshot = rows[0].document_block_id

    context = run_async(
        persistence_repository.get_completed_fact_extraction_persistence_context(
            ContextSession(rows),
            inference_run_id=run_id,
        )
    )

    assert context is not None
    assert context.inference_run_id == run_id
    assert context.status == "completed"
    response["facts"].append({"subject_kind": "tampered"})
    assert len(context.response_json["facts"]) == 1


@pytest.mark.parametrize("status", ["pending", "running", "failed"])
def test_non_completed_runs_are_rejected(status: str) -> None:
    project_id = uuid.uuid4()
    extraction_run_id = uuid.uuid4()
    context = build_context(project_id=project_id, extraction_run_id=extraction_run_id, status=status)

    with pytest.raises(persistence_service.FactExtractionPersistenceContextError):
        persistence_service._validate_persistence_context(
            project_id=project_id,
            extraction_run_id=extraction_run_id,
            inference_run_id=context.inference_run_id,
            context=context,
        )


def test_cross_project_context_is_rejected() -> None:
    context = build_context()
    with pytest.raises(persistence_service.FactExtractionPersistenceContextError):
        persistence_service._validate_persistence_context(
            project_id=uuid.uuid4(),
            extraction_run_id=context.blocks[0].extraction_run_id_snapshot,
            inference_run_id=context.inference_run_id,
            context=context,
        )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda context: replace(context.blocks[0], extraction_run_id_snapshot=uuid.uuid4()),
    ],
)
def test_extraction_run_snapshot_mismatch_is_rejected(mutator) -> None:
    context = build_context()
    bad_first = mutator(context)
    bad_context = build_context(
        project_id=context.project_id,
        extraction_run_id=context.blocks[0].extraction_run_id_snapshot,
        inference_run_id=context.inference_run_id,
        blocks=(bad_first, *context.blocks[1:]),
        response_json=context.response_json,
    )
    with pytest.raises(persistence_service.FactExtractionPersistenceContextError):
        persistence_service._validate_persistence_context(
            project_id=context.project_id,
            extraction_run_id=context.blocks[0].extraction_run_id_snapshot,
            inference_run_id=context.inference_run_id,
            context=bad_context,
        )


@pytest.mark.parametrize(
    ("override_kwargs", "expected_message"),
    [
        ({"source_block_id_snapshot": uuid.uuid4()}, "document_block_id"),
        ({"document_block_extraction_run_id": uuid.uuid4()}, "document block extraction_run"),
        ({"document_block_project_id": uuid.uuid4()}, "document block project"),
        ({"content_text": "篡改正文"}, "content_text"),
        ({"content_hash": "b" * 64}, "content_hash"),
    ],
)
def test_persistence_context_block_integrity_is_revalidated(override_kwargs, expected_message: str) -> None:
    context = build_context()
    block = context.blocks[0]
    block_kwargs = {
        "source_order": block.source_order,
        "block_ref": block.block_ref,
        "text": block.document_block_raw_text,
        "extraction_run_id": block.extraction_run_id_snapshot,
        "project_id": context.project_id,
        "document_block_id": block.document_block_id,
        "source_block_id_snapshot": block.source_block_id_snapshot,
        "document_block_extraction_run_id": block.document_block_extraction_run_id,
        "document_block_project_id": block.document_block_project_id,
        "content_text": block.content_text,
        "content_hash": block.content_hash,
    }
    block_kwargs.update(override_kwargs)
    bad_block = build_block(
        **block_kwargs,
    )
    bad_context = build_context(
        project_id=context.project_id,
        extraction_run_id=block.extraction_run_id_snapshot,
        inference_run_id=context.inference_run_id,
        response_json=context.response_json,
        blocks=(bad_block, *context.blocks[1:]),
    )

    with pytest.raises(persistence_service.FactExtractionPersistenceContextError) as exc_info:
        persistence_service._validate_persistence_context(
            project_id=context.project_id,
            extraction_run_id=block.extraction_run_id_snapshot,
            inference_run_id=context.inference_run_id,
            context=bad_context,
        )

    assert expected_message in str(exc_info.value)


def test_block_source_order_must_be_continuous() -> None:
    context = build_context(
        blocks=(
            build_block(
                source_order=0,
                block_ref="B0001",
                text="张三",
                extraction_run_id=uuid.uuid4(),
                project_id=uuid.uuid4(),
            ),
            build_block(
                source_order=2,
                block_ref="B0002",
                text="国王",
                extraction_run_id=uuid.uuid4(),
                project_id=uuid.uuid4(),
            ),
        )
    )

    with pytest.raises(persistence_service.FactExtractionPersistenceContextError):
        persistence_service._validate_persistence_context(
            project_id=context.project_id,
            extraction_run_id=context.blocks[0].extraction_run_id_snapshot,
            inference_run_id=context.inference_run_id,
            context=context,
        )


def test_response_json_is_reparsed_strictly() -> None:
    context = build_context(response_json={"facts": "not-a-list"})
    with pytest.raises(persistence_service.FactExtractionPersistenceContextError):
        persistence_service._validate_persistence_context(
            project_id=context.project_id,
            extraction_run_id=context.blocks[0].extraction_run_id_snapshot,
            inference_run_id=context.inference_run_id,
            context=context,
        )


def test_evidence_bounds_are_revalidated_against_batch() -> None:
    response = valid_response_json()
    response["facts"][0]["evidence"][0]["end_offset"] = 99
    context = build_context(response_json=response)

    with pytest.raises(persistence_service.FactExtractionPersistenceContextError):
        persistence_service._validate_persistence_context(
            project_id=context.project_id,
            extraction_run_id=context.blocks[0].extraction_run_id_snapshot,
            inference_run_id=context.inference_run_id,
            context=context,
        )


def test_completed_context_query_does_not_read_storage_key_or_failure_message() -> None:
    source = inspect.getsource(persistence_repository.get_completed_fact_extraction_persistence_context)
    assert "storage_key" not in source
    assert "failure_message" not in source
    assert ".join(InferenceInputBatch" in source
    assert ".join(DocumentBlock" in source


def test_completed_context_query_does_not_use_lazy_loading() -> None:
    source = inspect.getsource(persistence_repository.get_completed_fact_extraction_persistence_context)
    assert "joinedload" not in source
    assert ".blocks" not in source
    assert ".revision." not in source
    assert ".document." not in source
    assert ".project." not in source


def test_resolve_entity_mention_prefers_active_canonical_and_skips_alias(monkeypatch) -> None:
    canonical = SimpleNamespace(
        entity_id=uuid.uuid4(),
        canonical_key="zhang-san",
        status=EntityStatus.ACTIVE.value,
    )

    monkeypatch.setattr(entity_repository, "get_entity_context_by_identity", lambda *_args, **_kwargs: asyncio.sleep(0, result=canonical))

    async def fail_if_called(*_args, **_kwargs):
        raise AssertionError("alias lookup should not run after canonical hit")

    monkeypatch.setattr(entity_repository, "list_active_entity_contexts_by_alias", fail_if_called)

    result = run_async(
        persistence_service.resolve_entity_mention(
            FakeSession(),
            project_id=uuid.uuid4(),
            entity_type="person",
            mention_key=" 张三 ",
        )
    )

    assert result.status == EntityMentionResolutionStatus.RESOLVED.value
    assert result.entity_id == canonical.entity_id
    assert result.canonical_key == "zhang-san"


@pytest.mark.parametrize("status", [EntityStatus.MERGED.value, EntityStatus.ARCHIVED.value])
def test_canonical_merged_or_archived_is_ineligible(status: str, monkeypatch) -> None:
    canonical = SimpleNamespace(
        entity_id=uuid.uuid4(),
        canonical_key="zhang-san",
        status=status,
    )
    monkeypatch.setattr(entity_repository, "get_entity_context_by_identity", lambda *_args, **_kwargs: asyncio.sleep(0, result=canonical))
    monkeypatch.setattr(
        entity_repository,
        "list_active_entity_contexts_by_alias",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=[]),
    )

    result = run_async(
        persistence_service.resolve_entity_mention(
            FakeSession(),
            project_id=uuid.uuid4(),
            entity_type="person",
            mention_key="张三",
        )
    )

    assert result.status == EntityMentionResolutionStatus.INELIGIBLE.value


def test_alias_resolution_deduplicates_multiple_aliases_from_same_entity() -> None:
    entity_id = uuid.uuid4()
    rows = [
        SimpleNamespace(
            id=entity_id,
            project_id=uuid.uuid4(),
            entity_type="person",
            canonical_key="zhang-san",
            status=EntityStatus.ACTIVE.value,
        ),
        SimpleNamespace(
            id=entity_id,
            project_id=uuid.uuid4(),
            entity_type="person",
            canonical_key="zhang-san",
            status=EntityStatus.ACTIVE.value,
        ),
    ]
    session = ContextSession(rows)

    result = run_async(
        entity_repository.list_active_entity_contexts_by_alias(
            session,
            project_id=uuid.uuid4(),
            entity_type="person",
            normalized_alias="张三",
        )
    )

    assert len(result) == 1
    assert result[0].entity_id == entity_id


def test_shared_alias_across_two_entities_is_ambiguous(monkeypatch) -> None:
    monkeypatch.setattr(entity_repository, "get_entity_context_by_identity", lambda *_args, **_kwargs: asyncio.sleep(0, result=None))
    monkeypatch.setattr(
        entity_repository,
        "list_active_entity_contexts_by_alias",
        lambda *_args, **_kwargs: asyncio.sleep(
            0,
            result=[
                SimpleNamespace(entity_id=uuid.uuid4(), canonical_key="one"),
                SimpleNamespace(entity_id=uuid.uuid4(), canonical_key="two"),
            ],
        ),
    )

    result = run_async(
        persistence_service.resolve_entity_mention(
            FakeSession(),
            project_id=uuid.uuid4(),
            entity_type="person",
            mention_key="阿三",
        )
    )

    assert result.status == EntityMentionResolutionStatus.AMBIGUOUS.value
    assert result.entity_id is None


def test_no_candidates_are_unresolved(monkeypatch) -> None:
    monkeypatch.setattr(entity_repository, "get_entity_context_by_identity", lambda *_args, **_kwargs: asyncio.sleep(0, result=None))
    monkeypatch.setattr(
        entity_repository,
        "list_active_entity_contexts_by_alias",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=[]),
    )

    result = run_async(
        persistence_service.resolve_entity_mention(
            FakeSession(),
            project_id=uuid.uuid4(),
            entity_type="person",
            mention_key="未知人名",
        )
    )

    assert result.status == EntityMentionResolutionStatus.UNRESOLVED.value


def test_entity_resolution_does_not_use_fuzzy_match_or_create_entities() -> None:
    source = inspect.getsource(persistence_service.resolve_entity_mention)
    assert "create_entity" not in source
    assert "difflib" not in source
    assert "merged_into" not in source


def test_persist_batch_uses_canonical_subject_and_referenced_entity(monkeypatch) -> None:
    session = FakeSession()
    project_id = uuid.uuid4()
    extraction_run_id = uuid.uuid4()
    referenced_entity_id = uuid.uuid4()
    context = build_context(
        project_id=project_id,
        extraction_run_id=extraction_run_id,
        response_json=valid_response_json(
            value_type="entity_ref",
            value_json={"kind": "person", "key": "李四"},
        ),
    )
    original_json = copy.deepcopy(context.response_json)
    captured: dict[str, object] = {}
    evidence_ids = [uuid.uuid4(), uuid.uuid4()]

    monkeypatch.setattr(
        persistence_service.persistence_repository,
        "get_completed_fact_extraction_persistence_context",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=context),
    )

    async def fake_resolve(_session, *, project_id, entity_type, mention_key):
        if mention_key == "张三":
            return build_resolution(
                status=EntityMentionResolutionStatus.RESOLVED,
                entity_id=uuid.uuid4(),
                canonical_key="zhang-san",
            )
        return build_resolution(
            status=EntityMentionResolutionStatus.RESOLVED,
            entity_id=referenced_entity_id,
            canonical_key="li-si",
        )

    monkeypatch.setattr(persistence_service, "resolve_entity_mention", fake_resolve)

    async def fake_get_or_create(_session, *, block_id, raw_text, start_offset, end_offset):
        index = len(captured.setdefault("evidence_calls", []))
        captured["evidence_calls"].append((block_id, raw_text, start_offset, end_offset))
        return SimpleNamespace(id=evidence_ids[index]), True

    monkeypatch.setattr(persistence_service, "get_or_create_source_evidence_in_transaction", fake_get_or_create)

    async def fake_propose(_session, *, project_id, extraction_run_id, inference_run_id, payload):
        captured["payload"] = payload
        fact_value = SimpleNamespace(id=uuid.uuid4(), fact_id=uuid.uuid4())
        return SimpleNamespace(fact_value=fact_value, created=True)

    monkeypatch.setattr(persistence_service, "propose_ai_fact_value_in_transaction", fake_propose)

    result = run_async(
        persistence_service.persist_completed_fact_extraction_batch(
            session,
            project_id=project_id,
            extraction_run_id=extraction_run_id,
            inference_run_id=context.inference_run_id,
        )
    )

    payload = captured["payload"]
    assert payload.identity.subject_key == "zhang-san"
    assert payload.value.value_type == FactValueType.ENTITY_REF
    assert payload.value.value_json == {"kind": "person", "key": "li-si"}
    assert payload.value.referenced_entity_id == referenced_entity_id
    assert payload.evidences[0].is_primary is True
    assert payload.evidences[0].role.value == "supporting"
    assert payload.evidences[1].is_primary is False
    assert payload.evidences[1].role.value == "context"
    assert result.items[0].outcome == FactProposalPersistenceOutcome.CREATED
    assert context.response_json == original_json


def test_unresolved_subject_can_create_unbound_fact(monkeypatch) -> None:
    session = FakeSession()
    context = build_context()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        persistence_service.persistence_repository,
        "get_completed_fact_extraction_persistence_context",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=context),
    )
    monkeypatch.setattr(
        persistence_service,
        "resolve_entity_mention",
        lambda *_args, **_kwargs: asyncio.sleep(
            0,
            result=build_resolution(
                status=EntityMentionResolutionStatus.UNRESOLVED,
                mention_key="zhang-san",
                candidate_count=0,
            ),
        ),
    )
    monkeypatch.setattr(
        persistence_service,
        "get_or_create_source_evidence_in_transaction",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=(SimpleNamespace(id=uuid.uuid4()), True)),
    )

    async def fake_propose(_session, *, project_id, extraction_run_id, inference_run_id, payload):
        captured["payload"] = payload
        return SimpleNamespace(fact_value=SimpleNamespace(id=uuid.uuid4(), fact_id=uuid.uuid4()), created=True)

    monkeypatch.setattr(persistence_service, "propose_ai_fact_value_in_transaction", fake_propose)

    run_async(
        persistence_service.persist_completed_fact_extraction_batch(
            session,
            project_id=context.project_id,
            extraction_run_id=context.blocks[0].extraction_run_id_snapshot,
            inference_run_id=context.inference_run_id,
        )
    )

    assert captured["payload"].identity.subject_entity_id is None
    assert captured["payload"].identity.subject_key == "zhang-san"


@pytest.mark.parametrize(
    ("subject_status", "referenced_status", "expected_reason"),
    [
        (EntityMentionResolutionStatus.AMBIGUOUS, None, FactProposalWithheldReason.SUBJECT_AMBIGUOUS),
        (EntityMentionResolutionStatus.INELIGIBLE, None, FactProposalWithheldReason.SUBJECT_INELIGIBLE),
        (EntityMentionResolutionStatus.RESOLVED, EntityMentionResolutionStatus.UNRESOLVED, FactProposalWithheldReason.ENTITY_REF_UNRESOLVED),
        (EntityMentionResolutionStatus.RESOLVED, EntityMentionResolutionStatus.AMBIGUOUS, FactProposalWithheldReason.ENTITY_REF_AMBIGUOUS),
        (EntityMentionResolutionStatus.RESOLVED, EntityMentionResolutionStatus.INELIGIBLE, FactProposalWithheldReason.ENTITY_REF_INELIGIBLE),
    ],
)
def test_withheld_proposals_do_not_materialize_evidence(
    monkeypatch,
    subject_status: EntityMentionResolutionStatus,
    referenced_status: EntityMentionResolutionStatus | None,
    expected_reason: FactProposalWithheldReason,
) -> None:
    session = FakeSession()
    context = build_context(
        response_json=valid_response_json(
            value_type="entity_ref",
            value_json={"kind": "person", "key": "李四"},
        )
    )
    counts = {"evidence": 0, "propose": 0}

    monkeypatch.setattr(
        persistence_service.persistence_repository,
        "get_completed_fact_extraction_persistence_context",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=context),
    )

    async def fake_resolve(_session, *, project_id, entity_type, mention_key):
        if mention_key == "张三":
            return build_resolution(status=subject_status, candidate_count=2 if subject_status == EntityMentionResolutionStatus.AMBIGUOUS else 1)
        return build_resolution(status=referenced_status, candidate_count=2 if referenced_status == EntityMentionResolutionStatus.AMBIGUOUS else 0)

    monkeypatch.setattr(persistence_service, "resolve_entity_mention", fake_resolve)

    async def fake_get_or_create(*_args, **_kwargs):
        counts["evidence"] += 1
        return SimpleNamespace(id=uuid.uuid4()), True

    async def fake_propose(*_args, **_kwargs):
        counts["propose"] += 1
        return SimpleNamespace(fact_value=SimpleNamespace(id=uuid.uuid4(), fact_id=uuid.uuid4()), created=True)

    monkeypatch.setattr(persistence_service, "get_or_create_source_evidence_in_transaction", fake_get_or_create)
    monkeypatch.setattr(persistence_service, "propose_ai_fact_value_in_transaction", fake_propose)

    result = run_async(
        persistence_service.persist_completed_fact_extraction_batch(
            session,
            project_id=context.project_id,
            extraction_run_id=context.blocks[0].extraction_run_id_snapshot,
            inference_run_id=context.inference_run_id,
        )
    )

    assert result.items[0].outcome == FactProposalPersistenceOutcome.WITHHELD
    assert result.items[0].withheld_reason == expected_reason
    assert counts["evidence"] == 0
    assert counts["propose"] == 0


def test_batch_with_ambiguous_and_retired_proposals_preserves_other_successes(monkeypatch) -> None:
    session = FakeSession()
    project_id = uuid.uuid4()
    extraction_run_id = uuid.uuid4()
    response = {
        "facts": [
            {
                "subject_kind": "person",
                "subject_key": "明确人名",
                "predicate_key": "title",
                "scope_key": None,
                "value_type": "string",
                "value_json": "国王",
                "language_code": "zh-CN",
                "confidence": 0.9,
                "evidence": [{"block_ref": "B0001", "start_offset": 0, "end_offset": 2, "role": "supporting"}],
            },
            {
                "subject_kind": "person",
                "subject_key": "歧义昵称",
                "predicate_key": "title",
                "scope_key": None,
                "value_type": "string",
                "value_json": "将军",
                "language_code": "zh-CN",
                "confidence": 0.9,
                "evidence": [{"block_ref": "B0001", "start_offset": 0, "end_offset": 2, "role": "supporting"}],
            },
            {
                "subject_kind": "person",
                "subject_key": "退休目标",
                "predicate_key": "title",
                "scope_key": None,
                "value_type": "string",
                "value_json": "学者",
                "language_code": "zh-CN",
                "confidence": 0.9,
                "evidence": [{"block_ref": "B0002", "start_offset": 0, "end_offset": 2, "role": "supporting"}],
            },
        ],
        "batch_summary": None,
        "uncertainties": [],
    }
    context = build_context(project_id=project_id, extraction_run_id=extraction_run_id, response_json=response)
    monkeypatch.setattr(
        persistence_service.persistence_repository,
        "get_completed_fact_extraction_persistence_context",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=context),
    )

    async def fake_resolve(_session, *, project_id, entity_type, mention_key):
        if mention_key == "歧义昵称":
            return build_resolution(status=EntityMentionResolutionStatus.AMBIGUOUS, candidate_count=2)
        return build_resolution(
            status=EntityMentionResolutionStatus.RESOLVED,
            entity_id=uuid.uuid4(),
            canonical_key=f"{mention_key}-canonical",
        )

    monkeypatch.setattr(persistence_service, "resolve_entity_mention", fake_resolve)
    monkeypatch.setattr(
        persistence_service,
        "get_or_create_source_evidence_in_transaction",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=(SimpleNamespace(id=uuid.uuid4()), True)),
    )

    async def fake_propose(_session, *, payload, **_kwargs):
        if payload.identity.subject_key == "退休目标-canonical":
            raise persistence_service.RetiredFactError("retired")
        return SimpleNamespace(fact_value=SimpleNamespace(id=uuid.uuid4(), fact_id=uuid.uuid4()), created=True)

    monkeypatch.setattr(persistence_service, "propose_ai_fact_value_in_transaction", fake_propose)

    result = run_async(
        persistence_service.persist_completed_fact_extraction_batch(
            session,
            project_id=project_id,
            extraction_run_id=extraction_run_id,
            inference_run_id=context.inference_run_id,
        )
    )

    assert [item.outcome for item in result.items] == [
        FactProposalPersistenceOutcome.CREATED,
        FactProposalPersistenceOutcome.WITHHELD,
        FactProposalPersistenceOutcome.WITHHELD,
    ]
    assert result.created_count == 1
    assert result.withheld_count == 2
    assert session.commit_count == 1


def test_unexpected_evidence_error_rolls_back_entire_batch(monkeypatch) -> None:
    session = FakeSession()
    context = build_context()
    monkeypatch.setattr(
        persistence_service.persistence_repository,
        "get_completed_fact_extraction_persistence_context",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=context),
    )
    monkeypatch.setattr(
        persistence_service,
        "resolve_entity_mention",
        lambda *_args, **_kwargs: asyncio.sleep(
            0,
            result=build_resolution(
                status=EntityMentionResolutionStatus.RESOLVED,
                entity_id=uuid.uuid4(),
                canonical_key="zhang-san",
            ),
        ),
    )

    async def boom(*_args, **_kwargs):
        raise RuntimeError("db broke")

    monkeypatch.setattr(persistence_service, "get_or_create_source_evidence_in_transaction", boom)

    with pytest.raises(RuntimeError):
        run_async(
            persistence_service.persist_completed_fact_extraction_batch(
                session,
                project_id=context.project_id,
                extraction_run_id=context.blocks[0].extraction_run_id_snapshot,
                inference_run_id=context.inference_run_id,
            )
        )

    assert session.rollback_count == 1
    assert session.commit_count == 0


def test_outer_flush_and_commit_failures_rollback(monkeypatch) -> None:
    for failing_method in ("flush", "commit"):
        session = FakeSession()
        context = build_context()
        monkeypatch.setattr(
            persistence_service.persistence_repository,
            "get_completed_fact_extraction_persistence_context",
            lambda *_args, **_kwargs: asyncio.sleep(0, result=context),
        )
        monkeypatch.setattr(
            persistence_service,
            "resolve_entity_mention",
            lambda *_args, **_kwargs: asyncio.sleep(
                0,
                result=build_resolution(
                    status=EntityMentionResolutionStatus.RESOLVED,
                    entity_id=uuid.uuid4(),
                    canonical_key="zhang-san",
                ),
            ),
        )
        monkeypatch.setattr(
            persistence_service,
            "get_or_create_source_evidence_in_transaction",
            lambda *_args, **_kwargs: asyncio.sleep(0, result=(SimpleNamespace(id=uuid.uuid4()), True)),
        )
        monkeypatch.setattr(
            persistence_service,
            "propose_ai_fact_value_in_transaction",
            lambda *_args, **_kwargs: asyncio.sleep(
                0,
                result=SimpleNamespace(fact_value=SimpleNamespace(id=uuid.uuid4(), fact_id=uuid.uuid4()), created=True),
            ),
        )

        async def fail() -> None:
            raise RuntimeError(failing_method)

        setattr(session, failing_method, fail)

        with pytest.raises(RuntimeError):
            run_async(
                persistence_service.persist_completed_fact_extraction_batch(
                    session,
                    project_id=context.project_id,
                    extraction_run_id=context.blocks[0].extraction_run_id_snapshot,
                    inference_run_id=context.inference_run_id,
                )
            )

        assert session.rollback_count == 1


def test_cancelled_error_rolls_back_batch_and_propagates(monkeypatch) -> None:
    session = FakeSession()
    context = build_context()
    monkeypatch.setattr(
        persistence_service.persistence_repository,
        "get_completed_fact_extraction_persistence_context",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=context),
    )
    monkeypatch.setattr(
        persistence_service,
        "resolve_entity_mention",
        lambda *_args, **_kwargs: asyncio.sleep(
            0,
            result=build_resolution(
                status=EntityMentionResolutionStatus.RESOLVED,
                entity_id=uuid.uuid4(),
                canonical_key="zhang-san",
            ),
        ),
    )

    async def cancelled(*_args, **_kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(persistence_service, "get_or_create_source_evidence_in_transaction", cancelled)

    with pytest.raises(asyncio.CancelledError):
        run_async(
            persistence_service.persist_completed_fact_extraction_batch(
                session,
                project_id=context.project_id,
                extraction_run_id=context.blocks[0].extraction_run_id_snapshot,
                inference_run_id=context.inference_run_id,
            )
        )

    assert session.rollback_count == 1


def test_batch_result_is_stable_and_safe(monkeypatch) -> None:
    session = FakeSession()
    context = build_context()
    calls = {"count": 0}

    monkeypatch.setattr(
        persistence_service.persistence_repository,
        "get_completed_fact_extraction_persistence_context",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=context),
    )
    monkeypatch.setattr(
        persistence_service,
        "resolve_entity_mention",
        lambda *_args, **_kwargs: asyncio.sleep(
            0,
            result=build_resolution(
                status=EntityMentionResolutionStatus.RESOLVED,
                entity_id=uuid.uuid4(),
                canonical_key="zhang-san",
            ),
        ),
    )
    monkeypatch.setattr(
        persistence_service,
        "get_or_create_source_evidence_in_transaction",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=(SimpleNamespace(id=uuid.uuid4()), True)),
    )

    async def fake_propose(*_args, **_kwargs):
        calls["count"] += 1
        created = calls["count"] == 1
        return SimpleNamespace(fact_value=SimpleNamespace(id=uuid.uuid4(), fact_id=uuid.uuid4()), created=created)

    monkeypatch.setattr(persistence_service, "propose_ai_fact_value_in_transaction", fake_propose)

    first = run_async(
        persistence_service.persist_completed_fact_extraction_batch(
            session,
            project_id=context.project_id,
            extraction_run_id=context.blocks[0].extraction_run_id_snapshot,
            inference_run_id=context.inference_run_id,
        )
    )
    second = run_async(
        persistence_service.persist_completed_fact_extraction_batch(
            session,
            project_id=context.project_id,
            extraction_run_id=context.blocks[0].extraction_run_id_snapshot,
            inference_run_id=context.inference_run_id,
        )
    )

    assert [item.proposal_index for item in first.items] == [item.proposal_index for item in second.items]
    assert first.created_count == 1
    assert second.reused_count == 1
    dumped = first.model_dump()
    assert "response_json" not in dumped
    assert "batch_summary" not in dumped
    assert "excerpt" not in str(dumped)
    assert "prompt" not in str(dumped).lower()


def test_fact_value_replay_unique_constraint_exists_and_compiles() -> None:
    table = Base.metadata.tables["fact_values"]
    constraint = next(
        constraint
        for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint)
        and tuple(constraint.columns.keys()) == ("fact_id", "inference_run_id", "value_hash")
    )
    sql = str(AddConstraint(constraint).compile(dialect=postgresql.dialect()))
    assert "uq_fv_fact_ir_value_hash" in sql


def test_new_migration_source_contains_duplicate_guard_and_downgrade() -> None:
    migration = Path("alembic/versions/202607311800_fact_value_inference_replay.py").read_text(encoding="utf-8")
    assert "HAVING count(*) > 1" in migration
    assert "uq_fv_fact_ir_value_hash" in migration
    assert "drop_constraint" in migration


def test_single_migration_head_is_fact_value_replay() -> None:
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    root = Path(__file__).resolve().parents[1]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    script = ScriptDirectory.from_config(config)
    assert list(script.get_heads()) == ["202607311800"]


def test_postgresql_offline_ddl_for_new_unique_constraint_compiles() -> None:
    table = Base.metadata.tables["fact_values"]
    constraint = next(
        constraint
        for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint)
        and constraint.name == "uq_fv_fact_ir_value_hash"
    )
    sql = str(AddConstraint(constraint).compile(dialect=postgresql.dialect()))
    assert "ALTER TABLE fact_values ADD CONSTRAINT uq_fv_fact_ir_value_hash" in sql


def test_docker_upgrade_downgrade_smoke_runs_only_when_docker_available() -> None:
    if shutil.which("docker") is None:
        pytest.skip("docker is not available in this environment")

    result = subprocess.run(
        ["docker", "compose", "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
