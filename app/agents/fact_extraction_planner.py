from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

from app.agents.fact_extraction import (
    render_fact_extraction_message_contents,
    validate_fact_extraction_prompt,
)
from app.agents.prompt_registry import PromptDefinition
from app.models.document_content import DocumentBlock, DocumentBlockType
from app.schemas.agent_fact_extraction import FactExtractionResponse
from app.schemas.fact_extraction_plan import (
    FactExtractionBatchPlan,
    FactExtractionPlan,
    FactExtractionPlannerConfig,
)


PLANNER_NAME = "deterministic_fact_block_planner"
PLANNER_VERSION = "1.0.1"
_SNAPSHOT_HASH_PLACEHOLDER = "0" * 64


class FactExtractionPlanningError(Exception):
    """Base class for deterministic batch-planning failures."""


class FactExtractionBlockTooLargeError(FactExtractionPlanningError):
    """Raised when one source block cannot fit into a single message budget."""

    def __init__(
        self,
        *,
        block_id: uuid.UUID,
        source_order: int,
        block_character_count: int,
        estimated_message_characters: int,
    ) -> None:
        self.block_id = block_id
        self.source_order = source_order
        self.block_character_count = block_character_count
        self.estimated_message_characters = estimated_message_characters
        super().__init__(
            "fact extraction block exceeds max_message_characters: "
            f"block_id={block_id} source_order={source_order} "
            f"block_character_count={block_character_count} "
            f"estimated_message_characters={estimated_message_characters}"
        )


@dataclass(frozen=True, slots=True)
class _SourceBlockSnapshot:
    id: uuid.UUID
    extraction_run_id: uuid.UUID
    source_order: int
    block_type: str
    raw_text: str
    location_key: str
    page_no: int | None
    heading_path: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _RenderablePlannedBlock:
    source_order: int
    block_ref: str
    block_type: str
    location_key: str
    page_no: int | None
    heading_path: tuple[Any, ...]
    content_text: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class _PlannedBatchCandidate:
    batch: FactExtractionBatchPlan
    final_indices: tuple[int, ...]
    primary_count: int


def plan_fact_extraction_batches(
    *,
    extraction_run_id: uuid.UUID,
    blocks: Sequence[DocumentBlock],
    prompt: PromptDefinition,
    config: FactExtractionPlannerConfig | None = None,
) -> FactExtractionPlan:
    validate_fact_extraction_prompt(prompt)
    if prompt.response_model is not FactExtractionResponse:
        raise FactExtractionPlanningError(
            "prompt response_model must be FactExtractionResponse"
        )

    resolved_config = config or FactExtractionPlannerConfig()
    source_blocks = _normalize_source_blocks(
        extraction_run_id=extraction_run_id,
        blocks=blocks,
    )
    latest_heading_before = _build_latest_heading_before(source_blocks)

    batches: list[FactExtractionBatchPlan] = []
    previous_tail_indices: tuple[int, ...] = ()
    cursor = 0
    while cursor < len(source_blocks):
        candidate = _plan_next_batch(
            batch_index=len(batches),
            cursor=cursor,
            previous_tail_indices=previous_tail_indices,
            source_blocks=source_blocks,
            latest_heading_before=latest_heading_before,
            prompt=prompt,
            config=resolved_config,
        )
        batches.append(candidate.batch)
        previous_tail_indices = (
            candidate.final_indices[-resolved_config.overlap_block_count :]
            if resolved_config.overlap_block_count > 0
            else ()
        )
        cursor += candidate.primary_count

    source_character_count = sum(len(block.raw_text) for block in source_blocks)
    plan_hash = _sha256_json(
        {
            "planner_name": PLANNER_NAME,
            "planner_version": PLANNER_VERSION,
            "prompt_contract_hash": prompt.contract_hash,
            "config": resolved_config.model_dump(mode="json"),
            "extraction_run_id": str(extraction_run_id),
            "batch_hashes": [batch.plan_hash for batch in batches],
            "source_block_count": len(source_blocks),
            "source_character_count": source_character_count,
        }
    )
    return FactExtractionPlan(
        extraction_run_id=extraction_run_id,
        prompt_contract_hash=prompt.contract_hash,
        planner_name=PLANNER_NAME,
        planner_version=PLANNER_VERSION,
        config=resolved_config,
        batches=tuple(batches),
        source_block_count=len(source_blocks),
        source_character_count=source_character_count,
        plan_hash=plan_hash,
    )


def _normalize_source_blocks(
    *,
    extraction_run_id: uuid.UUID,
    blocks: Sequence[DocumentBlock],
) -> tuple[_SourceBlockSnapshot, ...]:
    if not isinstance(extraction_run_id, uuid.UUID):
        raise FactExtractionPlanningError("extraction_run_id must be a UUID")

    input_blocks = list(blocks)
    if not input_blocks:
        raise FactExtractionPlanningError("blocks must contain at least one DocumentBlock")

    snapshots: list[_SourceBlockSnapshot] = []
    for block in input_blocks:
        try:
            block_id = _safe_getattr(block, "id")
            block_extraction_run_id = _safe_getattr(block, "extraction_run_id")
            source_order = _safe_getattr(block, "source_order")
            raw_text = _safe_getattr(block, "raw_text")
            block_type = _safe_getattr(block, "block_type")
            location_key = _safe_getattr(block, "location_key")
            page_no = _safe_getattr(block, "page_no")
            heading_path_value = _safe_getattr(block, "heading_path")
        except FactExtractionPlanningError:
            raise

        if not isinstance(block_id, uuid.UUID):
            raise FactExtractionPlanningError("each block must have a UUID id")
        if not isinstance(block_extraction_run_id, uuid.UUID):
            raise FactExtractionPlanningError("block extraction_run_id must be a UUID")
        if isinstance(source_order, bool) or not isinstance(source_order, int):
            raise FactExtractionPlanningError("block source_order must be an integer")
        if not isinstance(raw_text, str) or raw_text == "":
            raise FactExtractionPlanningError("block raw_text must be a non-empty string")
        if not isinstance(block_type, str) or not block_type:
            raise FactExtractionPlanningError("block block_type must be a non-empty string")
        if not isinstance(location_key, str) or not location_key:
            raise FactExtractionPlanningError("block location_key must be a non-empty string")
        if isinstance(page_no, bool) or (page_no is not None and not isinstance(page_no, int)):
            raise FactExtractionPlanningError("block page_no must be an integer or None")

        heading_path = _validate_heading_path(heading_path_value)
        snapshots.append(
            _SourceBlockSnapshot(
                id=block_id,
                extraction_run_id=block_extraction_run_id,
                source_order=source_order,
                block_type=block_type,
                raw_text=raw_text,
                location_key=location_key,
                page_no=page_no,
                heading_path=heading_path,
            )
        )

    ordered_snapshots = tuple(sorted(snapshots, key=lambda snapshot: snapshot.source_order))
    seen_ids: set[uuid.UUID] = set()
    seen_orders: set[int] = set()
    for expected_order, snapshot in enumerate(ordered_snapshots):
        if snapshot.source_order in seen_orders:
            raise FactExtractionPlanningError("block source_order values must be unique")
        seen_orders.add(snapshot.source_order)
        if snapshot.source_order != expected_order:
            raise FactExtractionPlanningError(
                "block source_order values must be contiguous and start at 0"
            )
        if snapshot.id in seen_ids:
            raise FactExtractionPlanningError("block ids must be unique")
        seen_ids.add(snapshot.id)
        if snapshot.extraction_run_id != extraction_run_id:
            raise FactExtractionPlanningError(
                "all blocks must belong to the given extraction_run_id"
            )

    return ordered_snapshots


def _build_latest_heading_before(
    source_blocks: tuple[_SourceBlockSnapshot, ...],
) -> tuple[int | None, ...]:
    latest_heading_index: int | None = None
    latest_heading_before: list[int | None] = []
    for block in source_blocks:
        latest_heading_before.append(latest_heading_index)
        if block.block_type == DocumentBlockType.HEADING.value:
            latest_heading_index = block.source_order
    return tuple(latest_heading_before)


def _plan_next_batch(
    *,
    batch_index: int,
    cursor: int,
    previous_tail_indices: tuple[int, ...],
    source_blocks: tuple[_SourceBlockSnapshot, ...],
    latest_heading_before: tuple[int | None, ...],
    prompt: PromptDefinition,
    config: FactExtractionPlannerConfig,
) -> _PlannedBatchCandidate:
    max_overlap = min(config.overlap_block_count, len(previous_tail_indices))
    heading_index = latest_heading_before[cursor] if config.include_preceding_heading else None

    for overlap_count in range(max_overlap, -1, -1):
        overlap_indices = (
            previous_tail_indices[-overlap_count:] if overlap_count > 0 else ()
        )
        if heading_index is not None and heading_index not in overlap_indices:
            with_context = _grow_batch_from_base(
                batch_index=batch_index,
                cursor=cursor,
                overlap_indices=overlap_indices,
                context_index=heading_index,
                source_blocks=source_blocks,
                prompt=prompt,
                config=config,
            )
            if with_context is not None:
                return with_context

        without_context = _grow_batch_from_base(
            batch_index=batch_index,
            cursor=cursor,
            overlap_indices=overlap_indices,
            context_index=None,
            source_blocks=source_blocks,
            prompt=prompt,
            config=config,
        )
        if without_context is not None:
            return without_context

    oversized = _build_batch_candidate(
        batch_index=batch_index,
        source_blocks=source_blocks,
        prompt=prompt,
        config=config,
        primary_indices=(cursor,),
        overlap_indices=(),
        context_indices=(),
    )
    block = source_blocks[cursor]
    raise FactExtractionBlockTooLargeError(
        block_id=block.id,
        source_order=block.source_order,
        block_character_count=len(block.raw_text),
        estimated_message_characters=oversized.batch.estimated_message_characters,
    )


def _grow_batch_from_base(
    *,
    batch_index: int,
    cursor: int,
    overlap_indices: tuple[int, ...],
    context_index: int | None,
    source_blocks: tuple[_SourceBlockSnapshot, ...],
    prompt: PromptDefinition,
    config: FactExtractionPlannerConfig,
) -> _PlannedBatchCandidate | None:
    valid_candidates: list[_PlannedBatchCandidate] = []
    context_indices = () if context_index is None else (context_index,)

    for end_index in range(cursor + 1, len(source_blocks) + 1):
        primary_indices = tuple(range(cursor, end_index))
        candidate = _build_batch_candidate(
            batch_index=batch_index,
            source_blocks=source_blocks,
            prompt=prompt,
            config=config,
            primary_indices=primary_indices,
            overlap_indices=overlap_indices,
            context_indices=context_indices,
        )
        if (
            len(candidate.batch.block_ids) > config.max_blocks_per_batch
            or candidate.batch.estimated_message_characters > config.max_message_characters
        ):
            break

        valid_candidates.append(candidate)
        if candidate.batch.estimated_message_characters >= config.target_message_characters:
            if len(valid_candidates) == 1:
                return candidate
            previous = valid_candidates[-2]
            previous_distance = abs(
                config.target_message_characters - previous.batch.estimated_message_characters
            )
            current_distance = abs(
                config.target_message_characters - candidate.batch.estimated_message_characters
            )
            return candidate if current_distance < previous_distance else previous

    if not valid_candidates:
        return None
    return valid_candidates[-1]


def _build_batch_candidate(
    *,
    batch_index: int,
    source_blocks: tuple[_SourceBlockSnapshot, ...],
    prompt: PromptDefinition,
    config: FactExtractionPlannerConfig,
    primary_indices: tuple[int, ...],
    overlap_indices: tuple[int, ...],
    context_indices: tuple[int, ...],
) -> _PlannedBatchCandidate:
    final_indices = tuple(sorted(set(overlap_indices) | set(context_indices) | set(primary_indices)))
    final_blocks = tuple(source_blocks[index] for index in final_indices)
    render_blocks = tuple(
        _RenderablePlannedBlock(
            source_order=local_order,
            block_ref=_block_ref(local_order),
            block_type=block.block_type,
            location_key=block.location_key,
            page_no=block.page_no,
            heading_path=block.heading_path,
            content_text=block.raw_text,
            content_hash=_sha256_text(block.raw_text),
        )
        for local_order, block in enumerate(final_blocks)
    )
    system_content, user_content = render_fact_extraction_message_contents(
        prompt=prompt,
        snapshot_hash=_SNAPSHOT_HASH_PLACEHOLDER,
        blocks=list(render_blocks),
    )
    estimated_message_characters = len(system_content) + len(user_content)
    message_template_hash = _sha256_json(
        {
            "system_content": system_content,
            "user_content": user_content,
        }
    )
    content_character_count = sum(len(block.raw_text) for block in final_blocks)
    block_ids = tuple(block.id for block in final_blocks)
    block_refs = tuple(block.block_ref for block in render_blocks)
    primary_block_ids = tuple(source_blocks[index].id for index in primary_indices)
    overlap_block_ids = tuple(source_blocks[index].id for index in sorted(set(overlap_indices)))
    context_block_ids = tuple(source_blocks[index].id for index in sorted(set(context_indices)))
    content_hashes = tuple(_sha256_text(block.raw_text) for block in final_blocks)
    batch_plan_hash = _sha256_json(
        {
            "planner_name": PLANNER_NAME,
            "planner_version": PLANNER_VERSION,
            "prompt_contract_hash": prompt.contract_hash,
            "config": config.model_dump(mode="json"),
            "batch_index": batch_index,
            "block_ids": [str(block_id) for block_id in block_ids],
            "block_refs": list(block_refs),
            "primary_block_ids": [str(block_id) for block_id in primary_block_ids],
            "overlap_block_ids": [str(block_id) for block_id in overlap_block_ids],
            "context_block_ids": [str(block_id) for block_id in context_block_ids],
            "content_hashes": list(content_hashes),
            "estimated_message_characters": estimated_message_characters,
            "message_template_hash": message_template_hash,
        }
    )
    return _PlannedBatchCandidate(
        batch=FactExtractionBatchPlan(
            batch_index=batch_index,
            block_ids=block_ids,
            block_refs=block_refs,
            primary_block_ids=primary_block_ids,
            overlap_block_ids=overlap_block_ids,
            context_block_ids=context_block_ids,
            estimated_message_characters=estimated_message_characters,
            content_character_count=content_character_count,
            message_template_hash=message_template_hash,
            plan_hash=batch_plan_hash,
        ),
        final_indices=final_indices,
        primary_count=len(primary_indices),
    )


def _block_ref(index: int) -> str:
    return f"B{index + 1:04d}"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _safe_getattr(obj: Any, field_name: str) -> Any:
    try:
        return getattr(obj, field_name)
    except (AttributeError, TypeError, KeyError):
        raise FactExtractionPlanningError(
            f"block {field_name} is missing or invalid"
        ) from None


def _validate_heading_path(value: Any) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise FactExtractionPlanningError("block heading_path must be a list or tuple")
    try:
        _validate_json_value(value, top_level=True)
    except FactExtractionPlanningError:
        raise
    return tuple(value)


def _validate_json_value(value: Any, *, top_level: bool = False) -> None:
    if top_level:
        if not isinstance(value, (list, tuple)):
            raise FactExtractionPlanningError("block heading_path must be a list or tuple")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise FactExtractionPlanningError("block heading_path must contain only finite JSON values")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_json_value(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise FactExtractionPlanningError("block heading_path object keys must be strings")
            _validate_json_value(item)
        return
    raise FactExtractionPlanningError("block heading_path must contain only standard JSON values")
