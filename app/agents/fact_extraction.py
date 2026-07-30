"""Agent 1 fact-extraction: safe context rendering and response parsing.

Pure functions only — no database access, no lazy loading, no LLM calls. The
renderer wraps every block's verbatim text as JSON string data so injected tags,
fake JSON, prompt-injection, or code fences inside block content can never break
out of the data boundary. The parser turns a raw completion into a validated
:class:`FactExtractionResponse` without ever leaking the raw model content into
exceptions.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import ValidationError

from app.schemas.agent_fact_extraction import FactExtractionResponse
from app.services.llm import LLMCompletion, LLMMessage, LLMResponseError, parse_strict_json_object

if TYPE_CHECKING:
    from app.agents.prompt_registry import PromptDefinition
    from app.models.inference import InferenceInputBatch


_FACT_EXTRACTION_TASK_TYPE = "fact_extraction"


class AgentError(Exception):
    """Base class for agent errors."""


class AgentContextError(AgentError):
    """Raised when an input batch cannot be safely rendered into a prompt."""


class AgentResponseError(AgentError):
    """Raised when a completion cannot be parsed into the response contract.

    Never carries the raw model content — only safe, structural detail.
    """


def render_fact_extraction_messages(
    *,
    prompt: "PromptDefinition",
    batch: "InferenceInputBatch",
) -> tuple[LLMMessage, LLMMessage]:
    if batch.task_type != _FACT_EXTRACTION_TASK_TYPE:
        raise AgentContextError("input batch task_type must be fact_extraction")

    # Require blocks to be explicitly loaded; reading batch.blocks on an unloaded
    # relationship would trigger an async lazy load. Checking __dict__ never does.
    if "blocks" not in batch.__dict__:
        raise AgentContextError("input batch blocks are not loaded")
    blocks = list(batch.__dict__["blocks"])
    if not blocks:
        raise AgentContextError("input batch has no blocks")

    ordered = sorted(blocks, key=lambda b: b.source_order)
    if [b.source_order for b in ordered] != list(range(len(ordered))):
        raise AgentContextError("input batch blocks are not contiguous from 0")
    refs = [b.block_ref for b in ordered]
    if len(set(refs)) != len(refs):
        raise AgentContextError("input batch block_refs are not unique")

    envelope = {
        "input_batch": {
            "snapshot_hash": batch.snapshot_hash,
            "blocks": [
                {
                    "block_ref": block.block_ref,
                    "block_type": block.block_type,
                    "location_key": block.location_key,
                    "page_no": block.page_no,
                    "heading_path": list(block.heading_path),
                    # Verbatim block text, safely escaped as a JSON string value.
                    "content": block.content_text,
                }
                for block in ordered
            ],
        }
    }
    envelope_json = json.dumps(envelope, ensure_ascii=False, sort_keys=True)

    system_message = LLMMessage(role="system", content=prompt.system_template)
    user_message = LLMMessage(
        role="user",
        content=f"{prompt.instruction_template}\n\n{envelope_json}",
    )
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
        # Only field locations + error types — never the raw model content.
        summary = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'])}:{item['type']}"
            for item in error.errors()
        )
    raise AgentResponseError(
        f"fact extraction response failed contract validation: {summary}"
    )
