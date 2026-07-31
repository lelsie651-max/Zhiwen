from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.agents.fact_extraction import render_fact_extraction_message_contents
from app.agents.fact_extraction_planner import (
    FactExtractionBlockTooLargeError,
    FactExtractionPlanningError,
    plan_fact_extraction_batches,
)
from app.agents.prompt_registry import PromptDefinition, get_prompt
from app.models.document_content import DocumentBlock
from app.schemas.agent_fact_extraction import FactExtractionResponse
from app.schemas.fact_extraction_plan import FactExtractionPlannerConfig


PROMPT = get_prompt("agent1_fact_extraction", "1.0.0")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _make_prompt(**overrides) -> PromptDefinition:
    base = dict(
        task_type="fact_extraction",
        agent_name="agent1_fact_extractor",
        agent_version="1.0.0",
        prompt_name="agent1_fact_extraction",
        prompt_version="1.0.0",
        system_template=PROMPT.system_template,
        instruction_template=PROMPT.instruction_template,
        response_model=FactExtractionResponse,
        temperature=0.1,
        max_output_tokens=8192,
    )
    base.update(overrides)
    return PromptDefinition(**base)


def _block(
    source_order: int,
    raw_text: str,
    *,
    extraction_run_id: uuid.UUID,
    block_id: uuid.UUID | None = None,
    block_type: str = "paragraph",
    heading_path: list[str] | None = None,
    location_key: str | None = None,
    page_no: int | None = None,
) -> DocumentBlock:
    return DocumentBlock(
        id=block_id or uuid.uuid4(),
        extraction_run_id=extraction_run_id,
        source_order=source_order,
        block_type=block_type,
        raw_text=raw_text,
        normalized_text=raw_text.strip() or raw_text,
        location_key=location_key or f"loc-{source_order}",
        anchor_hash=_sha256(f"anchor-{source_order}-{raw_text}"),
        page_no=page_no,
        block_index=source_order,
        heading_path=heading_path or [],
    )


def _estimate_batch_total(batch, blocks_by_id, *, prompt=PROMPT) -> int:
    render_blocks = []
    for source_order, (block_id, block_ref) in enumerate(zip(batch.block_ids, batch.block_refs, strict=True)):
        source = blocks_by_id[block_id]
        render_blocks.append(
            SimpleNamespace(
                source_order=source_order,
                block_ref=block_ref,
                block_type=source.block_type,
                location_key=source.location_key,
                page_no=source.page_no,
                heading_path=tuple(source.heading_path),
                content_text=source.raw_text,
                content_hash=_sha256(source.raw_text),
            )
        )
    system_content, user_content = render_fact_extraction_message_contents(
        prompt=prompt,
        snapshot_hash="0" * 64,
        blocks=render_blocks,
    )
    return len(system_content) + len(user_content)


def _estimate_source_blocks(blocks, *, prompt=PROMPT) -> int:
    render_blocks = []
    for source_order, block in enumerate(blocks):
        render_blocks.append(
            SimpleNamespace(
                source_order=source_order,
                block_ref=f"B{source_order + 1:04d}",
                block_type=block.block_type,
                location_key=block.location_key,
                page_no=block.page_no,
                heading_path=tuple(block.heading_path),
                content_text=block.raw_text,
                content_hash=_sha256(block.raw_text),
            )
        )
    system_content, user_content = render_fact_extraction_message_contents(
        prompt=prompt,
        snapshot_hash="0" * 64,
        blocks=render_blocks,
    )
    return len(system_content) + len(user_content)


def _plan(blocks, *, extraction_run_id=None, prompt=PROMPT, config=None):
    run_id = extraction_run_id or blocks[0].extraction_run_id
    return plan_fact_extraction_batches(
        extraction_run_id=run_id,
        blocks=blocks,
        prompt=prompt,
        config=config,
    )


def test_config_validation_rejects_bool_and_invalid_ranges():
    with pytest.raises(ValidationError):
        FactExtractionPlannerConfig(target_message_characters=True)
    with pytest.raises(ValidationError):
        FactExtractionPlannerConfig(max_message_characters=False)
    with pytest.raises(ValidationError):
        FactExtractionPlannerConfig(max_blocks_per_batch=0)
    with pytest.raises(ValidationError):
        FactExtractionPlannerConfig(overlap_block_count=-1)
    with pytest.raises(ValidationError):
        FactExtractionPlannerConfig(target_message_characters=10, max_message_characters=9)
    with pytest.raises(ValidationError):
        FactExtractionPlannerConfig(max_blocks_per_batch=3, overlap_block_count=3)


@pytest.mark.parametrize("bad", [0, 1, "true", "false"])
def test_config_rejects_non_bool_include_preceding_heading(bad):
    with pytest.raises(ValidationError):
        FactExtractionPlannerConfig(include_preceding_heading=bad)


def test_single_batch_small_document():
    run_id = uuid.uuid4()
    blocks = [
        _block(0, "第一段。", extraction_run_id=run_id),
        _block(1, "第二段。", extraction_run_id=run_id),
    ]

    plan = _plan(blocks)

    assert len(plan.batches) == 1
    assert plan.source_block_count == 2
    assert plan.source_character_count == sum(len(block.raw_text) for block in blocks)
    assert plan.batches[0].primary_block_ids == tuple(block.id for block in blocks)
    assert plan.batches[0].overlap_block_ids == ()
    assert plan.batches[0].context_block_ids == ()


def test_over_budget_splits_into_multiple_batches():
    run_id = uuid.uuid4()
    blocks = [
        _block(0, "A" * 1500, extraction_run_id=run_id),
        _block(1, "B" * 1500, extraction_run_id=run_id),
        _block(2, "C" * 1500, extraction_run_id=run_id),
    ]
    one_block_total = _estimate_source_blocks(blocks[:1])
    two_block_total = _estimate_source_blocks(blocks[:2])
    config = FactExtractionPlannerConfig(
        target_message_characters=one_block_total,
        max_message_characters=two_block_total - 1,
        max_blocks_per_batch=10,
        overlap_block_count=1,
    )

    plan = _plan(blocks, config=config)

    assert len(plan.batches) >= 2
    assert all(batch.estimated_message_characters <= config.max_message_characters for batch in plan.batches)


def test_caller_order_does_not_change_result():
    run_id = uuid.uuid4()
    blocks = [
        _block(0, "alpha", extraction_run_id=run_id),
        _block(1, "beta", extraction_run_id=run_id),
        _block(2, "gamma", extraction_run_id=run_id),
    ]
    config = FactExtractionPlannerConfig(max_blocks_per_batch=2, overlap_block_count=1)

    ordered = _plan(blocks, config=config)
    shuffled = _plan([blocks[2], blocks[0], blocks[1]], config=config)

    assert ordered == shuffled
    assert ordered.plan_hash == shuffled.plan_hash


def test_each_source_block_appears_exactly_once_as_primary():
    run_id = uuid.uuid4()
    blocks = [_block(index, f"正文 {index}", extraction_run_id=run_id) for index in range(5)]
    config = FactExtractionPlannerConfig(max_blocks_per_batch=2, overlap_block_count=1)

    plan = _plan(blocks, config=config)
    primary_ids = [block_id for batch in plan.batches for block_id in batch.primary_block_ids]

    assert primary_ids == [block.id for block in blocks]
    assert len(primary_ids) == len(set(primary_ids))


def test_overlap_repeats_only_configured_tail_count():
    run_id = uuid.uuid4()
    blocks = [_block(index, f"正文 {index}", extraction_run_id=run_id) for index in range(4)]
    config = FactExtractionPlannerConfig(max_blocks_per_batch=2, overlap_block_count=1)

    plan = _plan(blocks, config=config)

    assert len(plan.batches) == 3
    assert plan.batches[1].overlap_block_ids == (plan.batches[0].block_ids[-1],)
    assert plan.batches[2].overlap_block_ids == (plan.batches[1].block_ids[-1],)


def test_overlap_zero():
    run_id = uuid.uuid4()
    blocks = [_block(index, f"正文 {index}", extraction_run_id=run_id) for index in range(4)]
    config = FactExtractionPlannerConfig(max_blocks_per_batch=2, overlap_block_count=0)

    plan = _plan(blocks, config=config)

    assert len(plan.batches) == 2
    assert all(batch.overlap_block_ids == () for batch in plan.batches)


def test_preceding_heading_added_as_context():
    run_id = uuid.uuid4()
    blocks = [
        _block(0, "第一章", extraction_run_id=run_id, block_type="heading", heading_path=["第一章"]),
        _block(1, "A" * 250, extraction_run_id=run_id, heading_path=["第一章"]),
        _block(2, "B" * 250, extraction_run_id=run_id, heading_path=["第一章"]),
    ]
    target = _estimate_source_blocks(blocks[:2])
    max_total = _estimate_source_blocks(blocks) + 100
    config = FactExtractionPlannerConfig(
        target_message_characters=target,
        max_message_characters=max_total,
        max_blocks_per_batch=3,
        overlap_block_count=1,
        include_preceding_heading=True,
    )

    plan = _plan(blocks, config=config)

    assert len(plan.batches) == 2
    assert plan.batches[1].context_block_ids == (blocks[0].id,)
    assert plan.batches[1].block_ids == (blocks[0].id, blocks[1].id, blocks[2].id)


def test_heading_not_duplicated_when_already_in_overlap():
    run_id = uuid.uuid4()
    blocks = [
        _block(0, "第一章", extraction_run_id=run_id, block_type="heading", heading_path=["第一章"]),
        _block(1, "正文一", extraction_run_id=run_id, heading_path=["第一章"]),
        _block(2, "正文二", extraction_run_id=run_id, heading_path=["第一章"]),
    ]
    config = FactExtractionPlannerConfig(
        target_message_characters=_estimate_source_blocks(blocks[:2]),
        max_message_characters=_estimate_source_blocks(blocks) + 100,
        max_blocks_per_batch=3,
        overlap_block_count=2,
    )

    plan = _plan(blocks, config=config)

    assert len(plan.batches) == 2
    assert plan.batches[1].overlap_block_ids == (blocks[0].id, blocks[1].id)
    assert plan.batches[1].context_block_ids == ()
    assert plan.batches[1].block_ids.count(blocks[0].id) == 1


def test_batch_block_ids_unique_ordered_and_block_refs_contiguous():
    run_id = uuid.uuid4()
    blocks = [_block(index, f"正文 {index}", extraction_run_id=run_id) for index in range(5)]
    config = FactExtractionPlannerConfig(max_blocks_per_batch=3, overlap_block_count=1)

    plan = _plan(blocks, config=config)
    order_map = {block.id: block.source_order for block in blocks}

    for batch in plan.batches:
        assert len(batch.block_ids) == len(set(batch.block_ids))
        assert list(batch.block_ids) == sorted(batch.block_ids, key=lambda block_id: order_map[block_id])
        assert batch.block_refs == tuple(f"B{index + 1:04d}" for index in range(len(batch.block_ids)))


def test_max_blocks_limit_applies():
    run_id = uuid.uuid4()
    blocks = [_block(index, f"正文 {index}", extraction_run_id=run_id) for index in range(5)]
    config = FactExtractionPlannerConfig(max_blocks_per_batch=2, overlap_block_count=0)

    plan = _plan(blocks, config=config)

    assert all(len(batch.block_ids) <= 2 for batch in plan.batches)


def test_each_batch_actual_renderer_length_matches_estimate_and_respects_max():
    run_id = uuid.uuid4()
    blocks = [_block(index, "X" * 500, extraction_run_id=run_id) for index in range(4)]
    one_block_total = _estimate_source_blocks(blocks[:1])
    two_block_total = _estimate_source_blocks(blocks[:2])
    config = FactExtractionPlannerConfig(
        target_message_characters=one_block_total,
        max_message_characters=two_block_total - 1,
        max_blocks_per_batch=3,
        overlap_block_count=0,
    )
    plan = _plan(blocks, config=config)
    blocks_by_id = {block.id: block for block in blocks}

    for batch in plan.batches:
        actual = _estimate_batch_total(batch, blocks_by_id)
        assert actual == batch.estimated_message_characters
        assert actual <= config.max_message_characters


def test_message_template_hash_is_sha256_and_deterministic():
    run_id = uuid.uuid4()
    blocks = [_block(0, "正文", extraction_run_id=run_id)]

    plan_a = _plan(blocks)
    plan_b = _plan(blocks)
    batch = plan_a.batches[0]

    render_blocks = [
        SimpleNamespace(
            source_order=0,
            block_ref="B0001",
            block_type=blocks[0].block_type,
            location_key=blocks[0].location_key,
            page_no=blocks[0].page_no,
            heading_path=tuple(blocks[0].heading_path),
            content_text=blocks[0].raw_text,
            content_hash=_sha256(blocks[0].raw_text),
        )
    ]
    system_content, user_content = render_fact_extraction_message_contents(
        prompt=PROMPT,
        snapshot_hash="0" * 64,
        blocks=render_blocks,
    )
    expected_hash = _sha256(
        _canonical_json(
            {
                "system_content": system_content,
                "user_content": user_content,
            }
        )
    )

    assert len(batch.message_template_hash) == 64
    assert batch.message_template_hash == expected_hash
    assert batch.message_template_hash == plan_b.batches[0].message_template_hash


def test_response_contract_is_counted_in_budget():
    run_id = uuid.uuid4()
    blocks = [_block(0, "短文本", extraction_run_id=run_id)]
    plan = _plan(blocks)
    batch = plan.batches[0]
    blocks_by_id = {block.id: block for block in blocks}
    actual_total = _estimate_batch_total(batch, blocks_by_id)

    system_content, user_content = render_fact_extraction_message_contents(
        prompt=PROMPT,
        snapshot_hash="0" * 64,
        blocks=[
            SimpleNamespace(
                source_order=0,
                block_ref="B0001",
                block_type=blocks[0].block_type,
                location_key=blocks[0].location_key,
                page_no=blocks[0].page_no,
                heading_path=tuple(blocks[0].heading_path),
                content_text=blocks[0].raw_text,
                content_hash=_sha256(blocks[0].raw_text),
            )
        ],
    )
    envelope = json.loads(user_content.split("\n\n", 1)[1])
    without_contract = {
        "input_batch": envelope["input_batch"],
    }
    without_contract_total = len(system_content) + len(
        f"{PROMPT.instruction_template}\n\n{_canonical_json(without_contract)}"
    )

    assert actual_total == batch.estimated_message_characters
    assert actual_total > without_contract_total


def test_oversized_single_block_fails_without_leaking_content():
    run_id = uuid.uuid4()
    raw_text = "SECRET_BLOCK_TEXT_" + ("X" * 8000)
    block = _block(0, raw_text, extraction_run_id=run_id)
    config = FactExtractionPlannerConfig(
        target_message_characters=1000,
        max_message_characters=1200,
        max_blocks_per_batch=5,
        overlap_block_count=0,
    )

    with pytest.raises(FactExtractionBlockTooLargeError) as exc_info:
        _plan([block], config=config)

    error = exc_info.value
    assert error.block_id == block.id
    assert error.source_order == 0
    assert error.block_character_count == len(raw_text)
    assert error.estimated_message_characters > config.max_message_characters
    assert raw_text not in str(error)


@pytest.mark.parametrize(
    "source_orders",
    [
        [1],
        [0, 0],
        [0, 2],
    ],
)
def test_rejects_missing_duplicate_or_non_contiguous_source_order(source_orders):
    run_id = uuid.uuid4()
    blocks = [
        _block(order, f"正文 {index}", extraction_run_id=run_id)
        for index, order in enumerate(source_orders)
    ]

    with pytest.raises(FactExtractionPlanningError):
        _plan(blocks)


def test_missing_source_order_raises_planning_error_without_attribute_error_leak():
    run_id = uuid.uuid4()
    bad = SimpleNamespace(
        id=uuid.uuid4(),
        extraction_run_id=run_id,
        raw_text="正文",
        block_type="paragraph",
        location_key="loc-0",
        page_no=None,
        heading_path=[],
    )

    with pytest.raises(FactExtractionPlanningError) as exc_info:
        _plan([bad], extraction_run_id=run_id)

    assert "AttributeError" not in str(exc_info.value)


def test_mixed_source_order_types_do_not_leak_type_error():
    run_id = uuid.uuid4()
    good = _block(0, "正文一", extraction_run_id=run_id)
    bad = SimpleNamespace(
        id=uuid.uuid4(),
        extraction_run_id=run_id,
        source_order="1",
        raw_text="正文二",
        block_type="paragraph",
        location_key="loc-1",
        page_no=None,
        heading_path=[],
    )

    with pytest.raises(FactExtractionPlanningError) as exc_info:
        _plan([good, bad], extraction_run_id=run_id)

    assert "TypeError" not in str(exc_info.value)


def test_bad_getattr_does_not_leak_attribute_error():
    run_id = uuid.uuid4()

    class BrokenBlock:
        id = uuid.uuid4()
        extraction_run_id = run_id

        def __getattr__(self, name):
            raise AttributeError(name)

    with pytest.raises(FactExtractionPlanningError) as exc_info:
        _plan([BrokenBlock()], extraction_run_id=run_id)

    assert "AttributeError" not in str(exc_info.value)


def test_non_uuid_extraction_run_id_rejected():
    run_id = uuid.uuid4()
    block = _block(0, "正文", extraction_run_id=run_id)
    with pytest.raises(FactExtractionPlanningError):
        _plan([block], extraction_run_id="not-a-uuid")


@pytest.mark.parametrize(
    "heading_path",
    [
        "chapter",
        {"a": 1},
        [float("nan")],
        [{"ok": {1: "bad"}}],
    ],
)
def test_heading_path_rejects_non_list_tuple_or_non_json_values(heading_path):
    run_id = uuid.uuid4()
    bad = SimpleNamespace(
        id=uuid.uuid4(),
        extraction_run_id=run_id,
        source_order=0,
        raw_text="正文",
        block_type="paragraph",
        location_key="loc-0",
        page_no=None,
        heading_path=heading_path,
    )
    with pytest.raises(FactExtractionPlanningError):
        _plan([bad], extraction_run_id=run_id)


def test_rejects_cross_extraction_run_and_empty_block_and_duplicate_id():
    run_id = uuid.uuid4()
    other_run_id = uuid.uuid4()
    duplicate_id = uuid.uuid4()
    first = _block(0, "正文一", extraction_run_id=run_id, block_id=duplicate_id)
    second = _block(1, "正文二", extraction_run_id=other_run_id)
    with pytest.raises(FactExtractionPlanningError):
        _plan([first, second], extraction_run_id=run_id)

    source = _block(0, "正文", extraction_run_id=run_id)
    bad_text = SimpleNamespace(
        id=source.id,
        extraction_run_id=source.extraction_run_id,
        source_order=source.source_order,
        block_type=source.block_type,
        raw_text="",
        location_key=source.location_key,
        page_no=source.page_no,
        heading_path=source.heading_path,
    )
    with pytest.raises(FactExtractionPlanningError):
        _plan([bad_text], extraction_run_id=run_id)

    dup_a = _block(0, "正文一", extraction_run_id=run_id, block_id=duplicate_id)
    dup_b = _block(1, "正文二", extraction_run_id=run_id, block_id=duplicate_id)
    with pytest.raises(FactExtractionPlanningError):
        _plan([dup_a, dup_b], extraction_run_id=run_id)


def test_plan_results_are_immutable():
    run_id = uuid.uuid4()
    plan = _plan([_block(0, "正文", extraction_run_id=run_id)])

    with pytest.raises((ValidationError, TypeError, AttributeError)):
        plan.plan_hash = "x" * 64
    with pytest.raises((ValidationError, TypeError, AttributeError)):
        plan.batches[0].block_ids = ()


def test_plan_hash_is_deterministic_and_changes_with_inputs():
    run_id = uuid.uuid4()
    blocks = [
        _block(0, "same-length-a", extraction_run_id=run_id),
        _block(1, "same-length-b", extraction_run_id=run_id),
    ]

    plan1 = _plan(blocks)
    plan2 = _plan([blocks[1], blocks[0]])
    assert plan1.plan_hash == plan2.plan_hash

    changed_config = _plan(
        blocks,
        config=FactExtractionPlannerConfig(max_blocks_per_batch=2, overlap_block_count=0),
    )
    assert changed_config.plan_hash != plan1.plan_hash

    changed_prompt = _plan(
        blocks,
        prompt=_make_prompt(instruction_template="return one strict json object only"),
    )
    assert changed_prompt.plan_hash != plan1.plan_hash

    changed_content_blocks = [
        _block(0, "same-length-c", extraction_run_id=run_id, block_id=blocks[0].id),
        _block(1, "same-length-b", extraction_run_id=run_id, block_id=blocks[1].id),
    ]
    changed_content = _plan(changed_content_blocks)
    assert changed_content.plan_hash != plan1.plan_hash


def test_same_length_location_key_change_updates_batch_and_plan_hash():
    run_id = uuid.uuid4()
    original = [_block(0, "正文", extraction_run_id=run_id, location_key="loc-a")]
    changed = [
        _block(
            0,
            "正文",
            extraction_run_id=run_id,
            block_id=original[0].id,
            location_key="loc-b",
        )
    ]

    plan_original = _plan(original)
    plan_changed = _plan(changed)

    assert (
        plan_original.batches[0].message_template_hash
        != plan_changed.batches[0].message_template_hash
    )
    assert plan_original.batches[0].plan_hash != plan_changed.batches[0].plan_hash
    assert plan_original.plan_hash != plan_changed.plan_hash


def test_page_no_heading_path_and_block_type_change_update_hashes():
    run_id = uuid.uuid4()
    original = [
        _block(
            0,
            "正文",
            extraction_run_id=run_id,
            page_no=1,
            heading_path=["A"],
            block_type="paragraph",
        )
    ]

    page_changed = [
        _block(
            0,
            "正文",
            extraction_run_id=run_id,
            block_id=original[0].id,
            page_no=2,
            heading_path=["A"],
            block_type="paragraph",
        )
    ]
    heading_changed = [
        _block(
            0,
            "正文",
            extraction_run_id=run_id,
            block_id=original[0].id,
            page_no=1,
            heading_path=["B"],
            block_type="paragraph",
        )
    ]
    type_changed = [
        _block(
            0,
            "正文",
            extraction_run_id=run_id,
            block_id=original[0].id,
            page_no=1,
            heading_path=["A"],
            block_type="code",
        )
    ]

    plan_original = _plan(original)
    assert (
        _plan(page_changed).batches[0].message_template_hash
        != plan_original.batches[0].message_template_hash
    )
    assert (
        _plan(heading_changed).batches[0].message_template_hash
        != plan_original.batches[0].message_template_hash
    )
    assert (
        _plan(type_changed).batches[0].message_template_hash
        != plan_original.batches[0].message_template_hash
    )


def test_planner_module_stays_pure_and_does_not_create_records():
    source = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "agents"
        / "fact_extraction_planner.py"
    ).read_text(encoding="utf-8")
    assert "AsyncSession" not in source
    assert "httpx" not in source
    assert "InferenceInputBatch(" not in source
    assert "InferenceRun(" not in source
    assert ".commit(" not in source
