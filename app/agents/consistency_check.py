"""Agent 2 consistency-check prompt rendering and strict response parsing."""

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any, TYPE_CHECKING

from pydantic import ValidationError

from app.schemas.agent_consistency_check import (
    ConsistencyCheckAssessment,
    ConsistencyCheckResponse,
)
from app.schemas.consistency_check import (
    CONSISTENCY_CHECK_PLANNER_NAME,
    CONSISTENCY_CHECK_PLANNER_VERSION,
    ConsistencyCheckBatchPlan,
    ConsistencyCheckCandidateBundle,
    ConsistencyCheckPlan,
)
from app.services import fact_value_duplicate_grouping as duplicate_grouping_service
from app.services import consistency_check as consistency_check_service
from app.services.llm import LLMCompletion, LLMMessage, LLMResponseError, parse_strict_json_object

if TYPE_CHECKING:
    from app.agents.prompt_registry import PromptDefinition


_CONSISTENCY_CHECK_TASK_TYPE = "consistency_check"
_MAX_ERROR_ITEMS = 20
_MAX_ERROR_SUMMARY_CHARS = 1000
_SAFE_CONTRACT_LOC_SEGMENTS = frozenset(
    {
        "assessments",
        "candidate_id",
        "verdict",
        "severity",
        "confidence",
        "explanation",
        "cited_evidence_link_ids",
        "impact",
        "recommended_actions",
    }
)
_EXTRA_LOC_PLACEHOLDER = "<extra>"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


class AgentConsistencyCheckError(Exception):
    """Base class for Agent 2 prompt/render/parse failures."""


class AgentConsistencyCheckContextError(AgentConsistencyCheckError):
    """Raised when a plan or batch cannot be safely rendered."""


class AgentConsistencyCheckResponseError(AgentConsistencyCheckError):
    """Raised when a completion cannot be parsed or bound to the batch."""


def validate_consistency_check_prompt(prompt: "PromptDefinition") -> None:
    if prompt.task_type != _CONSISTENCY_CHECK_TASK_TYPE:
        raise AgentConsistencyCheckContextError("prompt task_type must be consistency_check")
    if prompt.response_model is not ConsistencyCheckResponse:
        raise AgentConsistencyCheckContextError(
            "prompt response_model must be ConsistencyCheckResponse"
        )


def _candidate_evidence_character_count(candidate: ConsistencyCheckCandidateBundle) -> int:
    return sum(len(evidence.excerpt) for member in candidate.members for evidence in member.evidences)


def _build_plan_manifest_hash(plan: ConsistencyCheckPlan) -> str:
    return duplicate_grouping_service.hash_deterministic_payload(
        {
            "consistency_application_id": str(plan.consistency_application_id),
            "source_result_manifest_hash": plan.source_result_manifest_hash,
            "planner_name": plan.planner_name,
            "planner_version": plan.planner_version,
            "config": asdict(plan.config),
            "batches": [
                {
                    "batch_index": batch.batch_index,
                    "candidate_ids": [str(candidate_id) for candidate_id in batch.candidate_ids],
                    "candidate_count": batch.candidate_count,
                    "evidence_character_count": batch.evidence_character_count,
                    "batch_manifest_hash": batch.batch_manifest_hash,
                }
                for batch in plan.batches
            ],
        }
    )


def _summarize_validation_errors(error: ValidationError) -> str:
    parts: list[str] = []
    for item in error.errors(include_url=False)[:_MAX_ERROR_ITEMS]:
        safe_segments: list[str] = []
        for segment in item["loc"]:
            if isinstance(segment, int) and not isinstance(segment, bool):
                safe_segments.append(str(segment))
            elif isinstance(segment, str) and segment in _SAFE_CONTRACT_LOC_SEGMENTS:
                safe_segments.append(segment)
            else:
                safe_segments.append(_EXTRA_LOC_PLACEHOLDER)
        parts.append(f"{'.'.join(safe_segments)}:{item['type']}")
    return "; ".join(parts)[:_MAX_ERROR_SUMMARY_CHARS]


def validate_consistency_check_batch_plan(
    *,
    plan: ConsistencyCheckPlan,
    batch: ConsistencyCheckBatchPlan,
) -> None:
    if not isinstance(plan, ConsistencyCheckPlan):
        raise AgentConsistencyCheckContextError("plan must be a ConsistencyCheckPlan")
    if not isinstance(batch, ConsistencyCheckBatchPlan):
        raise AgentConsistencyCheckContextError("batch must be a ConsistencyCheckBatchPlan")
    if not plan.batches:
        raise AgentConsistencyCheckContextError("plan must contain at least one batch")
    if plan.planner_name != CONSISTENCY_CHECK_PLANNER_NAME:
        raise AgentConsistencyCheckContextError("plan planner_name mismatch")
    if plan.planner_version != CONSISTENCY_CHECK_PLANNER_VERSION:
        raise AgentConsistencyCheckContextError("plan planner_version mismatch")
    if isinstance(batch.batch_index, bool) or not isinstance(batch.batch_index, int):
        raise AgentConsistencyCheckContextError("batch_index must be an integer")
    if batch.batch_index < 0 or batch.batch_index >= len(plan.batches):
        raise AgentConsistencyCheckContextError("batch_index is out of range for the plan")
    if plan.batches[batch.batch_index] != batch:
        raise AgentConsistencyCheckContextError("batch does not match the indexed plan batch")

    if batch.candidate_count != len(batch.candidate_ids) or batch.candidate_count != len(batch.candidates):
        raise AgentConsistencyCheckContextError("batch candidate_count does not match candidate boundaries")
    if tuple(candidate.candidate_id for candidate in batch.candidates) != batch.candidate_ids:
        raise AgentConsistencyCheckContextError("batch candidate_ids do not match batch candidates")
    if len(set(batch.candidate_ids)) != len(batch.candidate_ids):
        raise AgentConsistencyCheckContextError("batch candidate_ids must be unique")
    expected_evidence_character_count = sum(
        _candidate_evidence_character_count(candidate) for candidate in batch.candidates
    )
    if batch.evidence_character_count != expected_evidence_character_count:
        raise AgentConsistencyCheckContextError("batch evidence_character_count does not match evidence text")

    flattened_candidates = tuple(
        candidate
        for indexed_batch in plan.batches
        for candidate in indexed_batch.candidates
    )
    expected_batches = consistency_check_service._build_consistency_check_batches(
        consistency_application_id=plan.consistency_application_id,
        source_result_manifest_hash=plan.source_result_manifest_hash,
        config=plan.config,
        candidate_bundles=flattened_candidates,
    )
    if expected_batches != plan.batches:
        raise AgentConsistencyCheckContextError("plan batches failed manifest validation")
    if _build_plan_manifest_hash(plan) != plan.plan_manifest_hash:
        raise AgentConsistencyCheckContextError("plan manifest hash failed validation")


def render_consistency_check_message_contents(
    *,
    prompt: "PromptDefinition",
    plan: ConsistencyCheckPlan,
    batch: ConsistencyCheckBatchPlan,
    task_type: str = _CONSISTENCY_CHECK_TASK_TYPE,
) -> tuple[str, str]:
    validate_consistency_check_prompt(prompt)
    if task_type != _CONSISTENCY_CHECK_TASK_TYPE:
        raise AgentConsistencyCheckContextError("input batch task_type must be consistency_check")
    validate_consistency_check_batch_plan(plan=plan, batch=batch)

    envelope = {
        "response_contract": prompt.response_json_schema,
        "consistency_application_id": str(plan.consistency_application_id),
        "source_result_manifest_hash": plan.source_result_manifest_hash,
        "batch_index": batch.batch_index,
        "batch_manifest_hash": batch.batch_manifest_hash,
        "candidates": [
            {
                "candidate_id": str(candidate.candidate_id),
                "fact_id": str(candidate.fact_id),
                "candidate_kind": candidate.candidate_kind,
                "members": [
                    {
                        "fact_value_id": str(member.fact_value_id),
                        "source_batch_id": str(member.source_batch_id),
                        "semantic_key_hash": member.semantic_key_hash,
                        "value_type": member.value_type,
                        "value_json": member.value_json,
                        "referenced_entity_id": (
                            None
                            if member.referenced_entity_id is None
                            else str(member.referenced_entity_id)
                        ),
                        "evidences": [
                            {
                                "evidence_link_id": str(evidence.evidence_link_id),
                                "evidence_id": str(evidence.evidence_id),
                                "role": evidence.role,
                                "is_primary": evidence.is_primary,
                                "source_order": evidence.source_order,
                                "document_block_id": str(evidence.document_block_id),
                                "location_key": evidence.location_key,
                                "page_no": evidence.page_no,
                                "start_line": evidence.start_line,
                                "end_line": evidence.end_line,
                                "start_offset": evidence.start_offset,
                                "end_offset": evidence.end_offset,
                                "excerpt": evidence.excerpt,
                                "evidence_content_hash": evidence.evidence_content_hash,
                            }
                            for evidence in member.evidences
                        ],
                    }
                    for member in candidate.members
                ],
            }
            for candidate in batch.candidates
        ],
    }
    return prompt.system_template, f"{prompt.instruction_template}\n\n{_canonical_json(envelope)}"


def render_consistency_check_messages(
    *,
    prompt: "PromptDefinition",
    plan: ConsistencyCheckPlan,
    batch: ConsistencyCheckBatchPlan,
) -> tuple[LLMMessage, LLMMessage]:
    system_content, user_content = render_consistency_check_message_contents(
        prompt=prompt,
        plan=plan,
        batch=batch,
    )
    return (
        LLMMessage(role="system", content=system_content),
        LLMMessage(role="user", content=user_content),
    )


def parse_consistency_check_completion(
    completion: LLMCompletion,
    *,
    batch: ConsistencyCheckBatchPlan,
) -> ConsistencyCheckResponse:
    if completion.finish_reason != "stop":
        raise AgentConsistencyCheckResponseError(
            "consistency check completion did not finish with 'stop'"
        )
    payload: dict[str, Any] | None = None
    parse_failed = False
    try:
        payload = parse_strict_json_object(completion.content)
    except LLMResponseError:
        parse_failed = True
    if parse_failed:
        raise AgentConsistencyCheckResponseError(
            "consistency check response was not a single strict JSON object"
        )
    return parse_consistency_check_response_object(payload, batch=batch)


def _validate_response_against_batch(
    response: ConsistencyCheckResponse,
    *,
    batch: ConsistencyCheckBatchPlan,
) -> ConsistencyCheckResponse:
    expected_candidate_ids = set(batch.candidate_ids)
    actual_candidate_ids = {assessment.candidate_id for assessment in response.assessments}
    if actual_candidate_ids != expected_candidate_ids:
        raise AgentConsistencyCheckResponseError(
            "consistency check response assessments did not match batch candidate ids"
        )

    allowed_evidence_by_candidate_id = {
        candidate.candidate_id: {
            evidence.evidence_link_id
            for member in candidate.members
            for evidence in member.evidences
        }
        for candidate in batch.candidates
    }

    for assessment in response.assessments:
        allowed_ids = allowed_evidence_by_candidate_id.get(assessment.candidate_id)
        if allowed_ids is None:
            raise AgentConsistencyCheckResponseError(
                "consistency check response referenced an unknown candidate id"
            )
        cited_ids = list(assessment.cited_evidence_link_ids)
        if len(set(cited_ids)) != len(cited_ids):
            raise AgentConsistencyCheckResponseError(
                "consistency check response cited duplicate evidence link ids"
            )
        if any(cited_id not in allowed_ids for cited_id in cited_ids):
            raise AgentConsistencyCheckResponseError(
                "consistency check response cited evidence outside the candidate"
            )
    return response


def parse_consistency_check_response_object(
    response_json: dict[str, Any],
    *,
    batch: ConsistencyCheckBatchPlan,
) -> ConsistencyCheckResponse:
    summary: str | None = None
    try:
        response = ConsistencyCheckResponse.model_validate(response_json)
    except ValidationError as error:
        summary = _summarize_validation_errors(error)
    else:
        return _validate_response_against_batch(response, batch=batch)
    raise AgentConsistencyCheckResponseError(
        f"consistency check response failed contract validation: {summary}"
    )
