from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import inspect
import uuid

import pytest

from app.repositories.document_revision_diff import (
    DocumentRevisionDiffBlockRecord,
    DocumentRevisionDiffDocumentRecord,
    DocumentRevisionDiffRevisionRecord,
    DocumentRevisionDiffRunRecord,
)
from app.services import document_revision_diff as diff_service


def run_async(awaitable):
    return asyncio.run(awaitable)


class FakeSession:
    def __init__(self) -> None:
        self.commit_count = 0
        self.rollback_count = 0

    async def commit(self) -> None:
        self.commit_count += 1

    async def rollback(self) -> None:
        self.rollback_count += 1


class SessionFactory:
    def __init__(self) -> None:
        self.sessions: list[FakeSession] = []

    def __call__(self):
        factory = self

        class _Context:
            async def __aenter__(self_inner):
                session = FakeSession()
                factory.sessions.append(session)
                return session

            async def __aexit__(self_inner, exc_type, exc, tb):
                return False

        return _Context()


def _uuid(seed: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, seed)


def _anchor_hash(*, detected_format: str, location_key: str, raw_text: str) -> str:
    return hashlib.sha256(
        f"{detected_format}|{location_key}|{raw_text}".encode("utf-8")
    ).hexdigest()


def _document(*, project_id: uuid.UUID, document_id: uuid.UUID) -> DocumentRevisionDiffDocumentRecord:
    return DocumentRevisionDiffDocumentRecord(
        id=document_id,
        project_id=project_id,
    )


def _revision(
    *,
    revision_id: uuid.UUID,
    document_id: uuid.UUID,
    revision_no: int,
    supersedes_revision_id: uuid.UUID | None,
) -> DocumentRevisionDiffRevisionRecord:
    return DocumentRevisionDiffRevisionRecord(
        id=revision_id,
        document_id=document_id,
        revision_no=revision_no,
        supersedes_revision_id=supersedes_revision_id,
    )


def _run(
    *,
    run_id: uuid.UUID,
    revision_id: uuid.UUID,
    outcome: str = "success",
    status: str = "completed",
    extractor_name: str = "zhiwen-deterministic-extractor",
    extractor_version: str = "1.0.0",
    detected_format: str = "md",
    character_count: int,
    block_count: int,
) -> DocumentRevisionDiffRunRecord:
    return DocumentRevisionDiffRunRecord(
        id=run_id,
        revision_id=revision_id,
        status=status,
        outcome=outcome,
        extractor_name=extractor_name,
        extractor_version=extractor_version,
        detected_format=detected_format,
        character_count=character_count,
        block_count=block_count,
    )


def _block(
    *,
    seed: str,
    extraction_run_id: uuid.UUID,
    source_order: int,
    raw_text: str,
    normalized_text: str | None = None,
    block_type: str = "paragraph",
    location_key: str | None = None,
    page_no: int | None = 1,
    start_line: int | None = None,
    end_line: int | None = None,
    table_index: int | None = None,
    row_index: int | None = None,
) -> DocumentRevisionDiffBlockRecord:
    actual_location_key = location_key or f"md:{seed}"
    actual_normalized_text = normalized_text or raw_text
    return DocumentRevisionDiffBlockRecord(
        id=_uuid(f"block-{seed}"),
        extraction_run_id=extraction_run_id,
        source_order=source_order,
        block_type=block_type,
        raw_text=raw_text,
        normalized_text=actual_normalized_text,
        location_key=actual_location_key,
        anchor_hash=_anchor_hash(
            detected_format="md",
            location_key=actual_location_key,
            raw_text=raw_text,
        ),
        page_no=page_no,
        start_line=start_line if start_line is not None else source_order + 1,
        end_line=end_line if end_line is not None else source_order + 1,
        table_index=table_index,
        row_index=row_index,
    )


def _run_counts(
    blocks: tuple[DocumentRevisionDiffBlockRecord, ...],
) -> tuple[int, int]:
    return (
        sum(len(block.normalized_text) for block in blocks),
        len(blocks),
    )


def _build_fixture():
    project_id = _uuid("project")
    document_id = _uuid("document")
    base_revision_id = _uuid("revision-base")
    target_revision_id = _uuid("revision-target")
    base_run_id = _uuid("run-base")
    target_run_id = _uuid("run-target")

    base_blocks = (
        _block(
            seed="unchanged-base",
            extraction_run_id=base_run_id,
            source_order=0,
            raw_text="same text",
            location_key="md:1",
        ),
        _block(
            seed="modified-base",
            extraction_run_id=base_run_id,
            source_order=1,
            raw_text="old body",
            normalized_text="old body",
            location_key="md:2",
        ),
        _block(
            seed="moved-base",
            extraction_run_id=base_run_id,
            source_order=2,
            raw_text="move me",
            normalized_text="move me",
            location_key="md:3",
        ),
        _block(
            seed="removed-base",
            extraction_run_id=base_run_id,
            source_order=3,
            raw_text="remove me",
            normalized_text="remove me",
            location_key="md:4",
        ),
    )
    target_blocks = (
        _block(
            seed="unchanged-target",
            extraction_run_id=target_run_id,
            source_order=0,
            raw_text="same text",
            location_key="md:1",
        ),
        _block(
            seed="modified-target",
            extraction_run_id=target_run_id,
            source_order=1,
            raw_text="new body",
            normalized_text="new body",
            location_key="md:2",
        ),
        _block(
            seed="moved-target",
            extraction_run_id=target_run_id,
            source_order=2,
            raw_text="move me",
            normalized_text="move me",
            location_key="md:9",
        ),
        _block(
            seed="added-target",
            extraction_run_id=target_run_id,
            source_order=3,
            raw_text="add me",
            normalized_text="add me",
            location_key="md:10",
        ),
    )
    base_character_count, base_block_count = _run_counts(base_blocks)
    target_character_count, target_block_count = _run_counts(target_blocks)
    return {
        "project_id": project_id,
        "document_id": document_id,
        "base_revision_id": base_revision_id,
        "target_revision_id": target_revision_id,
        "base_run_id": base_run_id,
        "target_run_id": target_run_id,
        "document": _document(project_id=project_id, document_id=document_id),
        "base_revision": _revision(
            revision_id=base_revision_id,
            document_id=document_id,
            revision_no=1,
            supersedes_revision_id=None,
        ),
        "target_revision": _revision(
            revision_id=target_revision_id,
            document_id=document_id,
            revision_no=2,
            supersedes_revision_id=base_revision_id,
        ),
        "base_run": _run(
            run_id=base_run_id,
            revision_id=base_revision_id,
            character_count=base_character_count,
            block_count=base_block_count,
        ),
        "target_run": _run(
            run_id=target_run_id,
            revision_id=target_revision_id,
            character_count=target_character_count,
            block_count=target_block_count,
        ),
        "base_blocks": base_blocks,
        "target_blocks": target_blocks,
    }


def _install_repository(
    monkeypatch: pytest.MonkeyPatch,
    *,
    document: DocumentRevisionDiffDocumentRecord | None,
    base_revision: DocumentRevisionDiffRevisionRecord | None,
    target_revision: DocumentRevisionDiffRevisionRecord | None,
    base_run: DocumentRevisionDiffRunRecord | None,
    target_run: DocumentRevisionDiffRunRecord | None,
    base_blocks: tuple[DocumentRevisionDiffBlockRecord, ...],
    target_blocks: tuple[DocumentRevisionDiffBlockRecord, ...],
    requested_ids: dict[str, list[uuid.UUID]] | None = None,
) -> None:
    async def fake_get_document_for_project(_session, *, project_id, document_id):
        if requested_ids is not None:
            requested_ids.setdefault("document", []).append(document_id)
        if document is None:
            return None
        if document.id == document_id and document.project_id == project_id:
            return document
        return None

    async def fake_get_document_revision_by_id(_session, *, revision_id):
        if requested_ids is not None:
            requested_ids.setdefault("revision", []).append(revision_id)
        if base_revision is not None and revision_id == base_revision.id:
            return base_revision
        if target_revision is not None and revision_id == target_revision.id:
            return target_revision
        return None

    async def fake_get_extraction_run_by_id(_session, *, extraction_run_id):
        if requested_ids is not None:
            requested_ids.setdefault("run", []).append(extraction_run_id)
        if base_run is not None and extraction_run_id == base_run.id:
            return base_run
        if target_run is not None and extraction_run_id == target_run.id:
            return target_run
        return None

    async def fake_list_document_blocks_for_extraction_run(_session, *, extraction_run_id):
        if requested_ids is not None:
            requested_ids.setdefault("blocks", []).append(extraction_run_id)
        if base_run is not None and extraction_run_id == base_run.id:
            return base_blocks
        if target_run is not None and extraction_run_id == target_run.id:
            return target_blocks
        return ()

    monkeypatch.setattr(
        diff_service.document_revision_diff_repository,
        "get_document_for_project",
        fake_get_document_for_project,
    )
    monkeypatch.setattr(
        diff_service.document_revision_diff_repository,
        "get_document_revision_by_id",
        fake_get_document_revision_by_id,
    )
    monkeypatch.setattr(
        diff_service.document_revision_diff_repository,
        "get_extraction_run_by_id",
        fake_get_extraction_run_by_id,
    )
    monkeypatch.setattr(
        diff_service.document_revision_diff_repository,
        "list_document_blocks_for_extraction_run",
        fake_list_document_blocks_for_extraction_run,
    )


def test_get_document_revision_block_diff_classifies_all_change_kinds_and_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture()
    session_factory = SessionFactory()
    requested_ids: dict[str, list[uuid.UUID]] = {}
    _install_repository(
        monkeypatch,
        document=fixture["document"],
        base_revision=fixture["base_revision"],
        target_revision=fixture["target_revision"],
        base_run=fixture["base_run"],
        target_run=fixture["target_run"],
        base_blocks=fixture["base_blocks"],
        target_blocks=fixture["target_blocks"],
        requested_ids=requested_ids,
    )

    first = run_async(
        diff_service.get_document_revision_block_diff(
            session_factory,
            project_id=fixture["project_id"],
            document_id=fixture["document_id"],
            base_revision_id=fixture["base_revision_id"],
            target_revision_id=fixture["target_revision_id"],
            base_extraction_run_id=fixture["base_run_id"],
            target_extraction_run_id=fixture["target_run_id"],
        )
    )
    second = run_async(
        diff_service.get_document_revision_block_diff(
            session_factory,
            project_id=fixture["project_id"],
            document_id=fixture["document_id"],
            base_revision_id=fixture["base_revision_id"],
            target_revision_id=fixture["target_revision_id"],
            base_extraction_run_id=fixture["base_run_id"],
            target_extraction_run_id=fixture["target_run_id"],
        )
    )

    assert first == second
    assert first.comparison_quality == "complete"
    assert (
        first.unchanged_count,
        first.modified_count,
        first.moved_count,
        first.added_count,
        first.removed_count,
    ) == (1, 1, 1, 1, 1)
    assert [item.change_kind for item in first.items] == [
        "unchanged",
        "modified",
        "moved",
        "added",
        "removed",
    ]
    modified = first.items[1]
    assert modified.base_block is not None
    assert modified.target_block is not None
    assert modified.raw_text_changed is True
    assert modified.normalized_text_changed is True
    assert modified.block_type_changed is False
    assert modified.locator_changed is False
    assert first.items[2].base_block is not None
    assert first.items[2].target_block is not None
    assert first.items[3].base_block is None
    assert first.items[4].target_block is None
    assert requested_ids == {
        "document": [fixture["document_id"], fixture["document_id"]],
        "revision": [
            fixture["base_revision_id"],
            fixture["target_revision_id"],
            fixture["base_revision_id"],
            fixture["target_revision_id"],
        ],
        "run": [
            fixture["base_run_id"],
            fixture["target_run_id"],
            fixture["base_run_id"],
            fixture["target_run_id"],
        ],
        "blocks": [
            fixture["base_run_id"],
            fixture["target_run_id"],
            fixture["base_run_id"],
            fixture["target_run_id"],
        ],
    }
    assert all(session.commit_count == 0 for session in session_factory.sessions)
    assert all(session.rollback_count == 1 for session in session_factory.sessions)


def test_get_document_revision_block_diff_does_not_force_ambiguous_duplicates_into_moved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture()
    base_blocks = (
        _block(
            seed="dup-base-a",
            extraction_run_id=fixture["base_run_id"],
            source_order=0,
            raw_text="duplicate",
            normalized_text="duplicate",
            location_key="md:a",
        ),
        _block(
            seed="dup-base-b",
            extraction_run_id=fixture["base_run_id"],
            source_order=1,
            raw_text="duplicate",
            normalized_text="duplicate",
            location_key="md:b",
        ),
    )
    target_blocks = (
        _block(
            seed="dup-target-a",
            extraction_run_id=fixture["target_run_id"],
            source_order=0,
            raw_text="duplicate",
            normalized_text="duplicate",
            location_key="md:x",
        ),
        _block(
            seed="dup-target-b",
            extraction_run_id=fixture["target_run_id"],
            source_order=1,
            raw_text="duplicate",
            normalized_text="duplicate",
            location_key="md:y",
        ),
    )
    base_character_count, base_block_count = _run_counts(base_blocks)
    target_character_count, target_block_count = _run_counts(target_blocks)
    _install_repository(
        monkeypatch,
        document=fixture["document"],
        base_revision=fixture["base_revision"],
        target_revision=fixture["target_revision"],
        base_run=replace(
            fixture["base_run"],
            character_count=base_character_count,
            block_count=base_block_count,
        ),
        target_run=replace(
            fixture["target_run"],
            character_count=target_character_count,
            block_count=target_block_count,
        ),
        base_blocks=base_blocks,
        target_blocks=target_blocks,
    )

    result = run_async(
        diff_service.get_document_revision_block_diff(
            SessionFactory(),
            project_id=fixture["project_id"],
            document_id=fixture["document_id"],
            base_revision_id=fixture["base_revision_id"],
            target_revision_id=fixture["target_revision_id"],
            base_extraction_run_id=fixture["base_run_id"],
            target_extraction_run_id=fixture["target_run_id"],
        )
    )

    assert [item.change_kind for item in result.items] == [
        "added",
        "added",
        "removed",
        "removed",
    ]
    assert result.moved_count == 0


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("cross_project", "document_revision_diff_document_not_found"),
        ("cross_document", "document_revision_diff_revision_document_mismatch"),
        ("not_adjacent", "document_revision_diff_revision_not_adjacent"),
    ],
)
def test_get_document_revision_block_diff_rejects_project_document_and_adjacency_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_code: str,
) -> None:
    fixture = _build_fixture()
    document = fixture["document"]
    target_revision = fixture["target_revision"]
    if mutation == "cross_project":
        document = None
    elif mutation == "cross_document":
        target_revision = replace(target_revision, document_id=_uuid("other-document"))
    else:
        target_revision = replace(
            target_revision,
            revision_no=3,
            supersedes_revision_id=_uuid("other-revision"),
        )
    _install_repository(
        monkeypatch,
        document=document,
        base_revision=fixture["base_revision"],
        target_revision=target_revision,
        base_run=fixture["base_run"],
        target_run=fixture["target_run"],
        base_blocks=fixture["base_blocks"],
        target_blocks=fixture["target_blocks"],
    )

    with pytest.raises(
        diff_service.DocumentRevisionBlockDiffStateError,
        match=expected_code,
    ):
        run_async(
            diff_service.get_document_revision_block_diff(
                SessionFactory(),
                project_id=fixture["project_id"],
                document_id=fixture["document_id"],
                base_revision_id=fixture["base_revision_id"],
                target_revision_id=fixture["target_revision_id"],
                base_extraction_run_id=fixture["base_run_id"],
                target_extraction_run_id=fixture["target_run_id"],
            )
        )


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("base_run_binding", "document_revision_diff_extraction_run_revision_mismatch"),
        ("run_status", "document_revision_diff_run_not_completed"),
        ("run_outcome", "document_revision_diff_run_outcome_not_comparable"),
        ("extractor_name", "document_revision_diff_runs_not_comparable"),
        ("extractor_version", "document_revision_diff_runs_not_comparable"),
        ("detected_format", "document_revision_diff_runs_not_comparable"),
    ],
)
def test_get_document_revision_block_diff_rejects_run_binding_state_and_comparability(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_code: str,
) -> None:
    fixture = _build_fixture()
    base_run = fixture["base_run"]
    target_run = fixture["target_run"]
    if mutation == "base_run_binding":
        base_run = replace(base_run, revision_id=_uuid("wrong-revision"))
    elif mutation == "run_status":
        target_run = replace(target_run, status="running")
    elif mutation == "run_outcome":
        target_run = replace(target_run, outcome="failed")
    elif mutation == "extractor_name":
        target_run = replace(target_run, extractor_name="other")
    elif mutation == "extractor_version":
        target_run = replace(target_run, extractor_version="2.0.0")
    else:
        target_run = replace(target_run, detected_format="pdf")
    _install_repository(
        monkeypatch,
        document=fixture["document"],
        base_revision=fixture["base_revision"],
        target_revision=fixture["target_revision"],
        base_run=base_run,
        target_run=target_run,
        base_blocks=fixture["base_blocks"],
        target_blocks=fixture["target_blocks"],
    )

    with pytest.raises(
        diff_service.DocumentRevisionBlockDiffStateError,
        match=expected_code,
    ):
        run_async(
            diff_service.get_document_revision_block_diff(
                SessionFactory(),
                project_id=fixture["project_id"],
                document_id=fixture["document_id"],
                base_revision_id=fixture["base_revision_id"],
                target_revision_id=fixture["target_revision_id"],
                base_extraction_run_id=fixture["base_run_id"],
                target_extraction_run_id=fixture["target_run_id"],
            )
        )


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("count", "document_revision_diff_block_source_mismatch"),
        ("character_count", "document_revision_diff_block_source_mismatch"),
        ("source_order", "document_revision_diff_block_source_mismatch"),
        ("duplicate_location", "document_revision_diff_block_source_mismatch"),
        ("duplicate_anchor", "document_revision_diff_block_source_mismatch"),
        ("anchor_drift", "document_revision_diff_block_anchor_drift"),
    ],
)
def test_get_document_revision_block_diff_fails_closed_on_block_drift(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_code: str,
) -> None:
    fixture = _build_fixture()
    base_run = fixture["base_run"]
    base_blocks = fixture["base_blocks"]
    if mutation == "count":
        base_run = replace(base_run, block_count=99)
    elif mutation == "character_count":
        base_run = replace(base_run, character_count=999)
    elif mutation == "source_order":
        base_blocks = (
            base_blocks[0],
            replace(base_blocks[1], source_order=3),
            base_blocks[2],
            base_blocks[3],
        )
    elif mutation == "duplicate_location":
        base_blocks = (
            base_blocks[0],
            replace(base_blocks[1], location_key=base_blocks[0].location_key),
            base_blocks[2],
            base_blocks[3],
        )
    elif mutation == "duplicate_anchor":
        base_blocks = (
            base_blocks[0],
            replace(base_blocks[1], anchor_hash=base_blocks[0].anchor_hash),
            base_blocks[2],
            base_blocks[3],
        )
    else:
        base_blocks = (
            base_blocks[0],
            replace(base_blocks[1], anchor_hash="0" * 64),
            base_blocks[2],
            base_blocks[3],
        )
    _install_repository(
        monkeypatch,
        document=fixture["document"],
        base_revision=fixture["base_revision"],
        target_revision=fixture["target_revision"],
        base_run=base_run,
        target_run=fixture["target_run"],
        base_blocks=base_blocks,
        target_blocks=fixture["target_blocks"],
    )

    with pytest.raises(
        diff_service.DocumentRevisionBlockDiffError,
        match=expected_code,
    ):
        run_async(
            diff_service.get_document_revision_block_diff(
                SessionFactory(),
                project_id=fixture["project_id"],
                document_id=fixture["document_id"],
                base_revision_id=fixture["base_revision_id"],
                target_revision_id=fixture["target_revision_id"],
                base_extraction_run_id=fixture["base_run_id"],
                target_extraction_run_id=fixture["target_run_id"],
            )
        )


def test_get_document_revision_block_diff_maps_partial_quality_and_zero_block_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture()
    target_blocks = (
        _block(
            seed="partial-added",
            extraction_run_id=fixture["target_run_id"],
            source_order=0,
            raw_text="one block",
            location_key="md:partial",
        ),
    )
    target_character_count, target_block_count = _run_counts(target_blocks)
    _install_repository(
        monkeypatch,
        document=fixture["document"],
        base_revision=fixture["base_revision"],
        target_revision=fixture["target_revision"],
        base_run=replace(
            fixture["base_run"],
            outcome="partial",
            character_count=0,
            block_count=0,
        ),
        target_run=replace(
            fixture["target_run"],
            character_count=target_character_count,
            block_count=target_block_count,
        ),
        base_blocks=(),
        target_blocks=target_blocks,
    )

    result = run_async(
        diff_service.get_document_revision_block_diff(
            SessionFactory(),
            project_id=fixture["project_id"],
            document_id=fixture["document_id"],
            base_revision_id=fixture["base_revision_id"],
            target_revision_id=fixture["target_revision_id"],
            base_extraction_run_id=fixture["base_run_id"],
            target_extraction_run_id=fixture["target_run_id"],
        )
    )

    assert result.comparison_quality == "partial"
    assert [item.change_kind for item in result.items] == ["added"]


def test_get_document_revision_block_diff_rejects_success_run_with_zero_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture()
    _install_repository(
        monkeypatch,
        document=fixture["document"],
        base_revision=fixture["base_revision"],
        target_revision=fixture["target_revision"],
        base_run=replace(
            fixture["base_run"],
            outcome="success",
            character_count=0,
            block_count=0,
        ),
        target_run=fixture["target_run"],
        base_blocks=(),
        target_blocks=fixture["target_blocks"],
    )

    with pytest.raises(
        diff_service.DocumentRevisionBlockDiffStateError,
        match="document_revision_diff_success_run_empty_blocks",
    ):
        run_async(
            diff_service.get_document_revision_block_diff(
                SessionFactory(),
                project_id=fixture["project_id"],
                document_id=fixture["document_id"],
                base_revision_id=fixture["base_revision_id"],
                target_revision_id=fixture["target_revision_id"],
                base_extraction_run_id=fixture["base_run_id"],
                target_extraction_run_id=fixture["target_run_id"],
            )
        )


@pytest.mark.parametrize(
    ("mutation", "expected_change_kinds"),
    [
        (
            "body",
            ["modified", "modified", "moved", "added", "removed"],
        ),
        (
            "locator",
            ["moved", "modified", "moved", "added", "removed"],
        ),
        (
            "identity",
            ["unchanged", "modified", "moved", "added", "removed"],
        ),
        (
            "change_kind",
            ["unchanged", "modified", "modified", "added", "removed"],
        ),
    ],
)
def test_get_document_revision_block_diff_manifest_changes_when_inputs_change(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_change_kinds: list[str],
) -> None:
    baseline = _build_fixture()
    _install_repository(
        monkeypatch,
        document=baseline["document"],
        base_revision=baseline["base_revision"],
        target_revision=baseline["target_revision"],
        base_run=baseline["base_run"],
        target_run=baseline["target_run"],
        base_blocks=baseline["base_blocks"],
        target_blocks=baseline["target_blocks"],
    )
    original = run_async(
        diff_service.get_document_revision_block_diff(
            SessionFactory(),
            project_id=baseline["project_id"],
            document_id=baseline["document_id"],
            base_revision_id=baseline["base_revision_id"],
            target_revision_id=baseline["target_revision_id"],
            base_extraction_run_id=baseline["base_run_id"],
            target_extraction_run_id=baseline["target_run_id"],
        )
    )

    changed = _build_fixture()
    target_blocks = changed["target_blocks"]
    target_revision = changed["target_revision"]
    target_run = changed["target_run"]
    if mutation == "body":
        target_blocks = (
            replace(
                target_blocks[0],
                raw_text="same text updated",
                anchor_hash=_anchor_hash(
                    detected_format="md",
                    location_key=target_blocks[0].location_key,
                    raw_text="same text updated",
                ),
            ),
            *target_blocks[1:],
        )
        character_count, block_count = _run_counts(target_blocks)
        target_run = replace(
            target_run,
            character_count=character_count,
            block_count=block_count,
        )
    elif mutation == "locator":
        target_blocks = (
            replace(
                target_blocks[0],
                location_key="md:1-updated",
                page_no=2,
                start_line=9,
                end_line=9,
                anchor_hash=_anchor_hash(
                    detected_format="md",
                    location_key="md:1-updated",
                    raw_text=target_blocks[0].raw_text,
                ),
            ),
            *target_blocks[1:],
        )
    elif mutation == "identity":
        target_revision = replace(target_revision, id=_uuid("revision-target-2"))
        target_run = replace(target_run, revision_id=target_revision.id)
    else:
        target_blocks = (
            target_blocks[0],
            target_blocks[1],
            replace(
                target_blocks[2],
                raw_text="move me updated",
                location_key=changed["base_blocks"][2].location_key,
                anchor_hash=_anchor_hash(
                    detected_format="md",
                    location_key=changed["base_blocks"][2].location_key,
                    raw_text="move me updated",
                ),
            ),
            target_blocks[3],
        )
        character_count, block_count = _run_counts(target_blocks)
        target_run = replace(
            target_run,
            character_count=character_count,
            block_count=block_count,
        )

    _install_repository(
        monkeypatch,
        document=changed["document"],
        base_revision=changed["base_revision"],
        target_revision=target_revision,
        base_run=changed["base_run"],
        target_run=target_run,
        base_blocks=changed["base_blocks"],
        target_blocks=target_blocks,
    )
    mutated = run_async(
        diff_service.get_document_revision_block_diff(
            SessionFactory(),
            project_id=changed["project_id"],
            document_id=changed["document_id"],
            base_revision_id=changed["base_revision_id"],
            target_revision_id=target_revision.id,
            base_extraction_run_id=changed["base_run_id"],
            target_extraction_run_id=target_run.id,
        )
    )

    assert mutated.diff_manifest_hash != original.diff_manifest_hash
    assert [item.change_kind for item in mutated.items] == expected_change_kinds


def test_get_document_revision_block_diff_does_not_write_and_does_not_leak_sensitive_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture()
    sentinel = "SENSITIVE_BLOCK_TEXT_SENTINEL"
    base_blocks = (
        replace(
            fixture["base_blocks"][0],
            raw_text=sentinel,
            anchor_hash="0" * 64,
        ),
        *fixture["base_blocks"][1:],
    )
    session_factory = SessionFactory()
    _install_repository(
        monkeypatch,
        document=fixture["document"],
        base_revision=fixture["base_revision"],
        target_revision=fixture["target_revision"],
        base_run=fixture["base_run"],
        target_run=fixture["target_run"],
        base_blocks=base_blocks,
        target_blocks=fixture["target_blocks"],
    )

    with pytest.raises(
        diff_service.DocumentRevisionBlockDiffInvariantError,
        match="document_revision_diff_block_anchor_drift",
    ) as exc_info:
        run_async(
            diff_service.get_document_revision_block_diff(
                session_factory,
                project_id=fixture["project_id"],
                document_id=fixture["document_id"],
                base_revision_id=fixture["base_revision_id"],
                target_revision_id=fixture["target_revision_id"],
                base_extraction_run_id=fixture["base_run_id"],
                target_extraction_run_id=fixture["target_run_id"],
            )
        )

    assert sentinel not in str(exc_info.value)
    assert all(session.commit_count == 0 for session in session_factory.sessions)
    assert all(session.rollback_count == 1 for session in session_factory.sessions)


def test_service_source_does_not_call_llm_or_latest_helpers() -> None:
    source = inspect.getsource(diff_service)
    assert "llm" not in source
    assert "latest" not in source
