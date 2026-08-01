from __future__ import annotations

import hashlib
import uuid
from collections import Counter
from collections.abc import Callable, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories import document_revision_diff as document_revision_diff_repository
from app.repositories.document_revision_diff import (
    DocumentRevisionDiffBlockRecord,
    DocumentRevisionDiffRunRecord,
)
from app.schemas.document_revision_diff import (
    DocumentRevisionBlockDiff,
    DocumentRevisionBlockDiffItem,
    DocumentRevisionDiffBlockSnapshot,
)
from app.services import fact_value_duplicate_grouping as duplicate_grouping_service

_COMPARABLE_OUTCOMES = frozenset({"success", "partial"})


class DocumentRevisionBlockDiffError(Exception):
    """Base class for adjacent revision diff failures."""


class DocumentRevisionBlockDiffStateError(DocumentRevisionBlockDiffError):
    """Raised when the requested revisions or runs are not comparable."""


class DocumentRevisionBlockDiffInvariantError(DocumentRevisionBlockDiffError):
    """Raised when persisted immutable extraction state drifts."""


def _require_uuid(value: object, *, field_name: str) -> uuid.UUID:
    if not isinstance(value, uuid.UUID):
        raise DocumentRevisionBlockDiffStateError(
            f"document_revision_diff_{field_name}_invalid"
        )
    return value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _build_anchor_hash(*, detected_format: str, location_key: str, raw_text: str) -> str:
    return hashlib.sha256(
        f"{detected_format}|{location_key}|{raw_text}".encode("utf-8")
    ).hexdigest()


def _block_fingerprint(block: DocumentRevisionDiffBlockRecord) -> tuple[str, str]:
    return (block.block_type, block.normalized_text)


def _build_block_snapshot(
    block: DocumentRevisionDiffBlockRecord,
) -> DocumentRevisionDiffBlockSnapshot:
    return DocumentRevisionDiffBlockSnapshot(
        block_id=block.id,
        source_order=block.source_order,
        block_type=block.block_type,
        location_key=block.location_key,
        page_no=block.page_no,
        start_line=block.start_line,
        end_line=block.end_line,
        table_index=block.table_index,
        row_index=block.row_index,
        anchor_hash=block.anchor_hash,
        raw_text_hash=_sha256_text(block.raw_text),
        normalized_text_hash=_sha256_text(block.normalized_text),
        raw_text=block.raw_text,
        normalized_text=block.normalized_text,
    )


def _build_modified_item(
    *,
    base_block: DocumentRevisionDiffBlockSnapshot,
    target_block: DocumentRevisionDiffBlockSnapshot,
) -> DocumentRevisionBlockDiffItem:
    return DocumentRevisionBlockDiffItem(
        change_kind="modified",
        base_block=base_block,
        target_block=target_block,
        raw_text_changed=base_block.raw_text != target_block.raw_text,
        normalized_text_changed=base_block.normalized_text != target_block.normalized_text,
        block_type_changed=base_block.block_type != target_block.block_type,
        locator_changed=(
            base_block.location_key != target_block.location_key
            or base_block.page_no != target_block.page_no
            or base_block.start_line != target_block.start_line
            or base_block.end_line != target_block.end_line
            or base_block.table_index != target_block.table_index
            or base_block.row_index != target_block.row_index
        ),
    )


def _validate_run_ready(
    run: DocumentRevisionDiffRunRecord,
) -> None:
    if run.status != "completed":
        raise DocumentRevisionBlockDiffStateError(
            "document_revision_diff_run_not_completed"
        )
    if run.outcome not in _COMPARABLE_OUTCOMES:
        raise DocumentRevisionBlockDiffStateError(
            "document_revision_diff_run_outcome_not_comparable"
        )


def _validate_runs_comparable(
    *,
    base_run: DocumentRevisionDiffRunRecord,
    target_run: DocumentRevisionDiffRunRecord,
) -> None:
    if (
        base_run.extractor_name != target_run.extractor_name
        or base_run.extractor_version != target_run.extractor_version
        or base_run.detected_format != target_run.detected_format
    ):
        raise DocumentRevisionBlockDiffStateError(
            "document_revision_diff_runs_not_comparable"
        )


def _validate_blocks_for_run(
    *,
    run: DocumentRevisionDiffRunRecord,
    blocks: Sequence[DocumentRevisionDiffBlockRecord],
) -> None:
    if any(block.extraction_run_id != run.id for block in blocks):
        raise DocumentRevisionBlockDiffInvariantError(
            "document_revision_diff_block_source_mismatch"
        )
    if run.block_count != len(blocks):
        raise DocumentRevisionBlockDiffInvariantError(
            "document_revision_diff_block_source_mismatch"
        )

    expected_source_order = list(range(len(blocks)))
    actual_source_order = [block.source_order for block in blocks]
    if actual_source_order != expected_source_order:
        raise DocumentRevisionBlockDiffInvariantError(
            "document_revision_diff_block_source_mismatch"
        )
    if len({block.location_key for block in blocks}) != len(blocks):
        raise DocumentRevisionBlockDiffInvariantError(
            "document_revision_diff_block_source_mismatch"
        )
    if len({block.anchor_hash for block in blocks}) != len(blocks):
        raise DocumentRevisionBlockDiffInvariantError(
            "document_revision_diff_block_source_mismatch"
        )

    actual_character_count = sum(len(block.normalized_text) for block in blocks)
    if actual_character_count != run.character_count:
        raise DocumentRevisionBlockDiffInvariantError(
            "document_revision_diff_block_source_mismatch"
        )

    for block in blocks:
        expected_anchor_hash = _build_anchor_hash(
            detected_format=run.detected_format,
            location_key=block.location_key,
            raw_text=block.raw_text,
        )
        if block.anchor_hash != expected_anchor_hash:
            raise DocumentRevisionBlockDiffInvariantError(
                "document_revision_diff_block_anchor_drift"
            )

    if run.outcome == "success" and len(blocks) == 0:
        raise DocumentRevisionBlockDiffStateError(
            "document_revision_diff_success_run_empty_blocks"
        )


def _serialize_block_snapshot(
    block: DocumentRevisionDiffBlockSnapshot | None,
) -> dict[str, object] | None:
    if block is None:
        return None
    return {
        "block_id": str(block.block_id),
        "source_order": block.source_order,
        "block_type": block.block_type,
        "location_key": block.location_key,
        "page_no": block.page_no,
        "start_line": block.start_line,
        "end_line": block.end_line,
        "table_index": block.table_index,
        "row_index": block.row_index,
        "anchor_hash": block.anchor_hash,
        "raw_text_hash": block.raw_text_hash,
        "normalized_text_hash": block.normalized_text_hash,
        "raw_text": block.raw_text,
        "normalized_text": block.normalized_text,
    }


def _serialize_item(item: DocumentRevisionBlockDiffItem) -> dict[str, object]:
    return {
        "change_kind": item.change_kind,
        "base_block": _serialize_block_snapshot(item.base_block),
        "target_block": _serialize_block_snapshot(item.target_block),
        "raw_text_changed": item.raw_text_changed,
        "normalized_text_changed": item.normalized_text_changed,
        "block_type_changed": item.block_type_changed,
        "locator_changed": item.locator_changed,
    }


def _build_manifest_hash(
    *,
    project_id: uuid.UUID,
    document_id: uuid.UUID,
    base_revision_id: uuid.UUID,
    target_revision_id: uuid.UUID,
    base_revision_no: int,
    target_revision_no: int,
    base_run: DocumentRevisionDiffRunRecord,
    target_run: DocumentRevisionDiffRunRecord,
    comparison_quality: str,
    unchanged_count: int,
    modified_count: int,
    moved_count: int,
    added_count: int,
    removed_count: int,
    items: Sequence[DocumentRevisionBlockDiffItem],
) -> str:
    return duplicate_grouping_service.hash_deterministic_payload(
        {
            "project_id": str(project_id),
            "document_id": str(document_id),
            "base_revision": {
                "revision_id": str(base_revision_id),
                "revision_no": base_revision_no,
                "extraction_run_id": str(base_run.id),
                "outcome": base_run.outcome,
                "extractor_name": base_run.extractor_name,
                "extractor_version": base_run.extractor_version,
                "detected_format": base_run.detected_format,
            },
            "target_revision": {
                "revision_id": str(target_revision_id),
                "revision_no": target_revision_no,
                "extraction_run_id": str(target_run.id),
                "outcome": target_run.outcome,
                "extractor_name": target_run.extractor_name,
                "extractor_version": target_run.extractor_version,
                "detected_format": target_run.detected_format,
            },
            "comparison_quality": comparison_quality,
            "counts": {
                "unchanged": unchanged_count,
                "modified": modified_count,
                "moved": moved_count,
                "added": added_count,
                "removed": removed_count,
            },
            "items": [_serialize_item(item) for item in items],
        }
    )


def _build_diff_items(
    *,
    base_blocks: Sequence[DocumentRevisionDiffBlockRecord],
    target_blocks: Sequence[DocumentRevisionDiffBlockRecord],
) -> tuple[DocumentRevisionBlockDiffItem, ...]:
    base_by_anchor = {block.anchor_hash: block for block in base_blocks}
    matched_base_ids: set[uuid.UUID] = set()
    matched_target_ids: set[uuid.UUID] = set()
    item_by_target_id: dict[uuid.UUID, DocumentRevisionBlockDiffItem] = {}

    for target_block in target_blocks:
        base_block = base_by_anchor.get(target_block.anchor_hash)
        if base_block is None:
            continue
        matched_base_ids.add(base_block.id)
        matched_target_ids.add(target_block.id)
        item_by_target_id[target_block.id] = (
            DocumentRevisionBlockDiffItem(
                change_kind="unchanged",
                base_block=_build_block_snapshot(base_block),
                target_block=_build_block_snapshot(target_block),
            )
        )

    remaining_base = [block for block in base_blocks if block.id not in matched_base_ids]
    remaining_target = [block for block in target_blocks if block.id not in matched_target_ids]
    base_by_location = {block.location_key: block for block in remaining_base}
    for target_block in remaining_target:
        base_block = base_by_location.get(target_block.location_key)
        if base_block is None:
            continue
        matched_base_ids.add(base_block.id)
        matched_target_ids.add(target_block.id)
        item_by_target_id[target_block.id] = (
            _build_modified_item(
                base_block=_build_block_snapshot(base_block),
                target_block=_build_block_snapshot(target_block),
            )
        )

    remaining_base = [block for block in base_blocks if block.id not in matched_base_ids]
    remaining_target = [block for block in target_blocks if block.id not in matched_target_ids]
    base_fingerprint_counts = Counter(_block_fingerprint(block) for block in remaining_base)
    target_fingerprint_counts = Counter(_block_fingerprint(block) for block in remaining_target)
    unique_base_by_fingerprint = {
        _block_fingerprint(block): block
        for block in remaining_base
        if base_fingerprint_counts[_block_fingerprint(block)] == 1
    }
    for target_block in remaining_target:
        fingerprint = _block_fingerprint(target_block)
        if target_fingerprint_counts[fingerprint] != 1:
            continue
        base_block = unique_base_by_fingerprint.get(fingerprint)
        if base_block is None:
            continue
        matched_base_ids.add(base_block.id)
        matched_target_ids.add(target_block.id)
        item_by_target_id[target_block.id] = (
            DocumentRevisionBlockDiffItem(
                change_kind="moved",
                base_block=_build_block_snapshot(base_block),
                target_block=_build_block_snapshot(target_block),
            )
        )

    items: list[DocumentRevisionBlockDiffItem] = []
    for target_block in target_blocks:
        items.append(
            item_by_target_id.get(
                target_block.id,
                DocumentRevisionBlockDiffItem(
                    change_kind="added",
                    base_block=None,
                    target_block=_build_block_snapshot(target_block),
                ),
            )
        )

    for base_block in base_blocks:
        if base_block.id in matched_base_ids:
            continue
        items.append(
            DocumentRevisionBlockDiffItem(
                change_kind="removed",
                base_block=_build_block_snapshot(base_block),
                target_block=None,
            )
        )

    return tuple(items)


async def get_document_revision_block_diff(
    session_factory: Callable[[], AsyncSession],
    *,
    project_id: uuid.UUID,
    document_id: uuid.UUID,
    base_revision_id: uuid.UUID,
    target_revision_id: uuid.UUID,
    base_extraction_run_id: uuid.UUID,
    target_extraction_run_id: uuid.UUID,
) -> DocumentRevisionBlockDiff:
    project_id = _require_uuid(project_id, field_name="project_id")
    document_id = _require_uuid(document_id, field_name="document_id")
    base_revision_id = _require_uuid(base_revision_id, field_name="base_revision_id")
    target_revision_id = _require_uuid(target_revision_id, field_name="target_revision_id")
    base_extraction_run_id = _require_uuid(
        base_extraction_run_id,
        field_name="base_extraction_run_id",
    )
    target_extraction_run_id = _require_uuid(
        target_extraction_run_id,
        field_name="target_extraction_run_id",
    )

    async with session_factory() as read_session:
        try:
            document = await document_revision_diff_repository.get_document_for_project(
                read_session,
                project_id=project_id,
                document_id=document_id,
            )
            if document is None:
                raise DocumentRevisionBlockDiffStateError(
                    "document_revision_diff_document_not_found"
                )

            base_revision = await document_revision_diff_repository.get_document_revision_by_id(
                read_session,
                revision_id=base_revision_id,
            )
            if base_revision is None:
                raise DocumentRevisionBlockDiffStateError(
                    "document_revision_diff_base_revision_not_found"
                )
            target_revision = await document_revision_diff_repository.get_document_revision_by_id(
                read_session,
                revision_id=target_revision_id,
            )
            if target_revision is None:
                raise DocumentRevisionBlockDiffStateError(
                    "document_revision_diff_target_revision_not_found"
                )
            if (
                base_revision.document_id != document.id
                or target_revision.document_id != document.id
            ):
                raise DocumentRevisionBlockDiffStateError(
                    "document_revision_diff_revision_document_mismatch"
                )
            if (
                target_revision.supersedes_revision_id != base_revision.id
                or target_revision.revision_no != base_revision.revision_no + 1
            ):
                raise DocumentRevisionBlockDiffStateError(
                    "document_revision_diff_revision_not_adjacent"
                )

            base_run = await document_revision_diff_repository.get_extraction_run_by_id(
                read_session,
                extraction_run_id=base_extraction_run_id,
            )
            if base_run is None:
                raise DocumentRevisionBlockDiffStateError(
                    "document_revision_diff_base_extraction_run_not_found"
                )
            target_run = await document_revision_diff_repository.get_extraction_run_by_id(
                read_session,
                extraction_run_id=target_extraction_run_id,
            )
            if target_run is None:
                raise DocumentRevisionBlockDiffStateError(
                    "document_revision_diff_target_extraction_run_not_found"
                )
            if (
                base_run.revision_id != base_revision.id
                or target_run.revision_id != target_revision.id
            ):
                raise DocumentRevisionBlockDiffStateError(
                    "document_revision_diff_extraction_run_revision_mismatch"
                )

            _validate_run_ready(base_run)
            _validate_run_ready(target_run)
            _validate_runs_comparable(base_run=base_run, target_run=target_run)

            base_blocks = (
                await document_revision_diff_repository.list_document_blocks_for_extraction_run(
                    read_session,
                    extraction_run_id=base_run.id,
                )
            )
            target_blocks = (
                await document_revision_diff_repository.list_document_blocks_for_extraction_run(
                    read_session,
                    extraction_run_id=target_run.id,
                )
            )
            _validate_blocks_for_run(run=base_run, blocks=base_blocks)
            _validate_blocks_for_run(run=target_run, blocks=target_blocks)
        except BaseException:
            await read_session.rollback()
            raise
        else:
            await read_session.rollback()

    items = _build_diff_items(base_blocks=base_blocks, target_blocks=target_blocks)
    unchanged_count = sum(1 for item in items if item.change_kind == "unchanged")
    modified_count = sum(1 for item in items if item.change_kind == "modified")
    moved_count = sum(1 for item in items if item.change_kind == "moved")
    added_count = sum(1 for item in items if item.change_kind == "added")
    removed_count = sum(1 for item in items if item.change_kind == "removed")
    comparison_quality = (
        "partial"
        if "partial" in {base_run.outcome, target_run.outcome}
        else "complete"
    )
    diff_manifest_hash = _build_manifest_hash(
        project_id=project_id,
        document_id=document_id,
        base_revision_id=base_revision.id,
        target_revision_id=target_revision.id,
        base_revision_no=base_revision.revision_no,
        target_revision_no=target_revision.revision_no,
        base_run=base_run,
        target_run=target_run,
        comparison_quality=comparison_quality,
        unchanged_count=unchanged_count,
        modified_count=modified_count,
        moved_count=moved_count,
        added_count=added_count,
        removed_count=removed_count,
        items=items,
    )
    return DocumentRevisionBlockDiff(
        project_id=project_id,
        document_id=document_id,
        base_revision_id=base_revision.id,
        target_revision_id=target_revision.id,
        base_extraction_run_id=base_run.id,
        target_extraction_run_id=target_run.id,
        base_revision_no=base_revision.revision_no,
        target_revision_no=target_revision.revision_no,
        comparison_quality=comparison_quality,
        unchanged_count=unchanged_count,
        modified_count=modified_count,
        moved_count=moved_count,
        added_count=added_count,
        removed_count=removed_count,
        items=items,
        diff_manifest_hash=diff_manifest_hash,
    )
