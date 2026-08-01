from __future__ import annotations

from dataclasses import dataclass
import uuid
from typing import Literal


DocumentRevisionDiffChangeKind = Literal[
    "unchanged",
    "modified",
    "moved",
    "added",
    "removed",
]

DocumentRevisionDiffQuality = Literal["complete", "partial"]


@dataclass(frozen=True, slots=True)
class DocumentRevisionDiffBlockSnapshot:
    block_id: uuid.UUID
    source_order: int
    block_type: str
    location_key: str
    page_no: int | None
    start_line: int | None
    end_line: int | None
    table_index: int | None
    row_index: int | None
    anchor_hash: str
    raw_text_hash: str
    normalized_text_hash: str
    raw_text: str
    normalized_text: str


@dataclass(frozen=True, slots=True)
class DocumentRevisionBlockDiffItem:
    change_kind: DocumentRevisionDiffChangeKind
    base_block: DocumentRevisionDiffBlockSnapshot | None
    target_block: DocumentRevisionDiffBlockSnapshot | None
    raw_text_changed: bool | None = None
    normalized_text_changed: bool | None = None
    block_type_changed: bool | None = None
    locator_changed: bool | None = None


@dataclass(frozen=True, slots=True)
class DocumentRevisionBlockDiff:
    project_id: uuid.UUID
    document_id: uuid.UUID
    base_revision_id: uuid.UUID
    target_revision_id: uuid.UUID
    base_extraction_run_id: uuid.UUID
    target_extraction_run_id: uuid.UUID
    base_revision_no: int
    target_revision_no: int
    algorithm_name: str
    algorithm_version: str
    extractor_name: str
    extractor_version: str
    detected_format: str
    comparison_quality: DocumentRevisionDiffQuality
    unchanged_count: int
    modified_count: int
    moved_count: int
    added_count: int
    removed_count: int
    items: tuple[DocumentRevisionBlockDiffItem, ...]
    diff_manifest_hash: str
