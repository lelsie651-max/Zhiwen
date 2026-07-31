"""Agent 1 fact-extraction: safe context rendering and response parsing.

Pure functions only — no database access, no lazy loading, no LLM calls. The
renderer wraps every block's verbatim text as JSON string data so injected tags,
fake JSON, prompt-injection, or code fences inside block content can never break
out of the data boundary. The parser turns a raw completion into a validated
:class:`FactExtractionResponse` without ever leaking the raw model content into
exceptions.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, TYPE_CHECKING

from pydantic import ValidationError

from app.schemas.agent_fact_extraction import FactExtractionResponse
from app.services.llm import LLMCompletion, LLMMessage, LLMResponseError, parse_strict_json_object

if TYPE_CHECKING:
    from app.agents.prompt_registry import PromptDefinition
    from app.models.inference import InferenceInputBatch


_FACT_EXTRACTION_TASK_TYPE = "fact_extraction"

# Bounds on the structural error summary surfaced from a failed contract
# validation. They keep the message small and free of any model-supplied values.
_MAX_ERROR_ITEMS = 20
_MAX_ERROR_SUMMARY_CHARS = 1000


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


class AgentError(Exception):
    """Base class for agent errors."""


class AgentContextError(AgentError):
    """Raised when an input batch cannot be safely rendered into a prompt."""


class AgentResponseError(AgentError):
    """Raised when a completion cannot be parsed into the response contract.

    Never carries the raw model content — only safe, structural detail.
    """


def validate_fact_extraction_prompt(prompt: "PromptDefinition") -> None:
    if prompt.task_type != _FACT_EXTRACTION_TASK_TYPE:
        raise AgentContextError("prompt task_type must be fact_extraction")
    if prompt.response_model is not FactExtractionResponse:
        raise AgentContextError("prompt response_model must be FactExtractionResponse")


def render_fact_extraction_message_contents(
    *,
    prompt: "PromptDefinition",
    snapshot_hash: str,
    blocks: list[Any],
    task_type: str = _FACT_EXTRACTION_TASK_TYPE,
    expected_block_count: int | None = None,
    expected_character_count: int | None = None,
) -> tuple[str, str]:
    # Shared pure serializer used by both the real renderer and the planner's
    # budget estimation. Keeping one implementation prevents JSON envelope drift.
    validate_fact_extraction_prompt(prompt)
    if task_type != _FACT_EXTRACTION_TASK_TYPE:
        raise AgentContextError("input batch task_type must be fact_extraction")
    if not blocks:
        raise AgentContextError("input batch has no blocks")

    ordered = sorted(blocks, key=lambda b: b.source_order)
    if [b.source_order for b in ordered] != list(range(len(ordered))):
        raise AgentContextError("input batch blocks are not contiguous from 0")

    resolved_block_count = len(ordered) if expected_block_count is None else expected_block_count
    if len(ordered) != resolved_block_count:
        raise AgentContextError("input batch block_count does not match loaded blocks")

    expected_refs = [f"B{index + 1:04d}" for index in range(len(ordered))]
    if [b.block_ref for b in ordered] != expected_refs:
        raise AgentContextError("input batch block_refs must be sequential B0001, B0002, ...")

    total_characters = 0
    for block in ordered:
        content = block.content_text
        if not isinstance(content, str):
            raise AgentContextError("block content_text must be a string")
        total_characters += len(content)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if digest != block.content_hash:
            raise AgentContextError("block content_hash does not match content_text")

    resolved_character_count = (
        total_characters if expected_character_count is None else expected_character_count
    )
    if total_characters != resolved_character_count:
        raise AgentContextError("input batch character_count does not match block content")

    envelope = {
        "response_contract": prompt.response_json_schema,
        "input_batch": {
            "snapshot_hash": snapshot_hash,
            "blocks": [
                {
                    "block_ref": block.block_ref,
                    "block_type": block.block_type,
                    "location_key": block.location_key,
                    "page_no": block.page_no,
                    "heading_path": list(block.heading_path),
                    "content": block.content_text,
                }
                for block in ordered
            ],
        },
    }
    envelope_json = _canonical_json(envelope)
    return prompt.system_template, f"{prompt.instruction_template}\n\n{envelope_json}"


def render_fact_extraction_messages(
    *,
    prompt: "PromptDefinition",
    batch: "InferenceInputBatch",
) -> tuple[LLMMessage, LLMMessage]:
    # Require blocks to be explicitly loaded; reading batch.blocks on an unloaded
    # relationship would trigger an async lazy load. Checking __dict__ never does.
    if "blocks" not in batch.__dict__:
        raise AgentContextError("input batch blocks are not loaded")
    system_content, user_content = render_fact_extraction_message_contents(
        prompt=prompt,
        snapshot_hash=batch.snapshot_hash,
        blocks=list(batch.__dict__["blocks"]),
        task_type=batch.task_type,
        expected_block_count=batch.block_count,
        expected_character_count=batch.character_count,
    )
    system_message = LLMMessage(role="system", content=system_content)
    user_message = LLMMessage(role="user", content=user_content)
    return system_message, user_message


def parse_fact_extraction_completion(
    completion: LLMCompletion,
) -> FactExtractionResponse:
    if completion.finish_reason != "stop":
        raise AgentResponseError("fact extraction completion did not finish with 'stop'")

    # Raise outside the except blocks below so neither the LLMResponseError nor the
    # pydantic ValidationError (which carry the raw model content) can reach the
    # error's __cause__ or __context__.
    payload: dict | None = None
    parse_failed = False
    try:
        payload = parse_strict_json_object(completion.content)
    except LLMResponseError:
        parse_failed = True
    if parse_failed:
        raise AgentResponseError(
            "fact extraction response was not a single strict JSON object"
        )

    summary: str | None = None
    try:
        return FactExtractionResponse.model_validate(payload)
    except ValidationError as error:
        # Only field locations + error types — never the raw model content, the
        # offending input value, or validation context. Also bounded in count and
        # length so a pathological reply cannot bloat the message.
        parts = []
        for item in error.errors(include_url=False)[:_MAX_ERROR_ITEMS]:
            location = ".".join(str(part) for part in item["loc"])
            parts.append(f"{location}:{item['type']}")
        summary = "; ".join(parts)[:_MAX_ERROR_SUMMARY_CHARS]
    raise AgentResponseError(
        f"fact extraction response failed contract validation: {summary}"
    )
