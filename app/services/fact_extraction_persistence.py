from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base import utc_now
from app.models.entity import EntityStatus, normalize_entity_alias
from app.models.fact import FactValueSourceKind, FactValueType
from app.models.fact_extraction_application import FactExtractionBatchApplication
from app.models.inference import InferenceRunStatus, InferenceTaskType
from app.repositories import entity as entity_repository
from app.repositories import fact as fact_repository
from app.repositories import fact_extraction_persistence as persistence_repository
from app.schemas.fact import FactIdentityInput, FactValueInput
from app.schemas.fact_commands import AIProposalInput, FactEvidenceInput
from app.schemas.fact_extraction_persistence import (
    AuthenticatedCompletedFactExtractionApplicationSnapshot,
    AuthenticatedPersistedFactProposalItem,
    CompletedFactExtractionPersistenceContext,
    EntityMentionResolution,
    EntityMentionResolutionStatus,
    FactExtractionBatchPersistenceResult,
    FactExtractionPersistenceBlock,
    FactProposalPersistenceItem,
    FactProposalPersistenceOutcome,
    FactProposalWithheldReason,
)
from app.services.document_content import get_or_create_source_evidence_in_transaction
from app.services.entity import normalize_entity_type
from app.services.fact import FactSubjectEntityConflictError, RetiredFactError, propose_ai_fact_value_in_transaction
from app.services.inference import (
    build_inference_input_batch_snapshot_hash,
    build_inference_response_json_hash,
)

if TYPE_CHECKING:
    from app.schemas.agent_fact_extraction import FactExtractionResponse, FactProposal


FACT_EXTRACTION_PERSISTENCE_NAME = "agent1_fact_persistence"
FACT_EXTRACTION_PERSISTENCE_VERSION = "1.0.0"

ENTITY_RESOLUTION_POLICY_NAME = "canonical_then_unique_active_alias"
ENTITY_RESOLUTION_POLICY_VERSION = "1.0.0"

_APPLICATION_STATUS_APPLYING = "applying"
_APPLICATION_STATUS_COMPLETED = "completed"
_APPLICATION_CONSTRAINT = "uq_feba_inference_run_id"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class FactExtractionPersistenceError(Exception):
    """Base class for completed batch persistence failures."""


class FactExtractionPersistenceContextError(FactExtractionPersistenceError):
    """Raised when the stored inference context is not safe to persist."""


class FactExtractionApplicationReplayConflictError(FactExtractionPersistenceError):
    """Raised when a stored application ledger cannot be replayed safely."""


@dataclass(frozen=True, slots=True)
class PreparedBatchApplication:
    application: FactExtractionBatchApplication
    replay_result: FactExtractionBatchPersistenceResult | None


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_uuid_instance(value: uuid.UUID, *, field_name: str) -> uuid.UUID:
    if not isinstance(value, uuid.UUID):
        raise FactExtractionPersistenceContextError(f"{field_name} must be a UUID")
    return value


def _require_sha256(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise FactExtractionPersistenceContextError(f"{field_name} must be a string")
    normalized = value.lower()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise FactExtractionPersistenceContextError(f"{field_name} must be a SHA-256 hex string")
    return normalized


def build_fact_extraction_application_result_hash(
    result_json: dict[str, Any],
) -> str:
    return build_inference_response_json_hash(result_json)


def _block_ref_for_order(source_order: int) -> str:
    return f"B{source_order + 1:04d}"


def _build_block_snapshots(
    *,
    input_batch_id: uuid.UUID,
    blocks: Sequence[FactExtractionPersistenceBlock],
) -> tuple[InferenceInputBlockSnapshot, ...]:
    from app.schemas.fact_extraction_execution import InferenceInputBlockSnapshot

    return tuple(
        InferenceInputBlockSnapshot(
            id=block.input_block_id,
            batch_id=input_batch_id,
            source_order=block.source_order,
            block_ref=block.block_ref,
            document_block_id=block.document_block_id,
            source_block_id_snapshot=block.source_block_id_snapshot,
            extraction_run_id_snapshot=block.extraction_run_id_snapshot,
            block_type=block.block_type,
            location_key=block.location_key,
            anchor_hash=block.anchor_hash,
            page_no=block.page_no,
            start_line=block.start_line,
            end_line=block.end_line,
            heading_path=list(block.heading_path),
            content_text=block.content_text,
            content_hash=block.content_hash,
        )
        for block in blocks
    )


def _build_snapshot_records(
    blocks: Sequence[FactExtractionPersistenceBlock],
) -> list[dict[str, Any]]:
    return [
        {
            "source_order": block.source_order,
            "block_ref": block.block_ref,
            "source_block_id": str(block.source_block_id_snapshot),
            "extraction_run_id": str(block.extraction_run_id_snapshot),
            "block_type": block.block_type,
            "location_key": block.location_key,
            "anchor_hash": block.anchor_hash,
            "page_no": block.page_no,
            "start_line": block.start_line,
            "end_line": block.end_line,
            "heading_path": list(block.heading_path),
            "content_hash": block.content_hash,
        }
        for block in blocks
    ]


def _validate_result_counts(result: FactExtractionBatchPersistenceResult) -> None:
    items = result.items
    created_count = sum(item.outcome == FactProposalPersistenceOutcome.CREATED for item in items)
    reused_count = sum(item.outcome == FactProposalPersistenceOutcome.REUSED for item in items)
    withheld_count = sum(item.outcome == FactProposalPersistenceOutcome.WITHHELD for item in items)
    if result.proposal_count != len(items):
        raise FactExtractionApplicationReplayConflictError("application result proposal_count is inconsistent")
    if result.created_count != created_count:
        raise FactExtractionApplicationReplayConflictError("application result created_count is inconsistent")
    if result.reused_count != reused_count:
        raise FactExtractionApplicationReplayConflictError("application result reused_count is inconsistent")
    if result.withheld_count != withheld_count:
        raise FactExtractionApplicationReplayConflictError("application result withheld_count is inconsistent")


def validate_fact_extraction_application_result_envelope(
    *,
    application: FactExtractionBatchApplication,
) -> FactExtractionBatchPersistenceResult:
    if application.status != _APPLICATION_STATUS_COMPLETED:
        raise FactExtractionApplicationReplayConflictError("application status must be completed")
    if application.result_json is None or application.result_hash is None:
        raise FactExtractionApplicationReplayConflictError("completed application is missing its result snapshot")
    if build_fact_extraction_application_result_hash(application.result_json) != application.result_hash:
        raise FactExtractionApplicationReplayConflictError("application result_hash does not match result_json")

    result = FactExtractionBatchPersistenceResult.model_validate(application.result_json)
    if result.replayed_application:
        raise FactExtractionApplicationReplayConflictError("application result snapshot must be canonical, not replayed")
    if result.application_id != application.id:
        raise FactExtractionApplicationReplayConflictError("application_id mismatch")
    if result.project_id != application.project_id:
        raise FactExtractionApplicationReplayConflictError("application project_id mismatch")
    if result.extraction_run_id != application.extraction_run_id:
        raise FactExtractionApplicationReplayConflictError("application extraction_run_id mismatch")
    if result.inference_run_id != application.inference_run_id:
        raise FactExtractionApplicationReplayConflictError("application inference_run_id mismatch")
    if result.input_batch_id != application.input_batch_id:
        raise FactExtractionApplicationReplayConflictError("application input_batch_id mismatch")
    if result.response_hash != application.response_hash:
        raise FactExtractionApplicationReplayConflictError("application response_hash mismatch")
    if result.persistence_name != application.persistence_name:
        raise FactExtractionApplicationReplayConflictError("application persistence_name mismatch")
    if result.persistence_version != application.persistence_version:
        raise FactExtractionApplicationReplayConflictError("application persistence_version mismatch")
    if result.entity_resolution_policy_name != application.entity_resolution_policy_name:
        raise FactExtractionApplicationReplayConflictError(
            "application entity_resolution_policy_name mismatch"
        )
    if result.entity_resolution_policy_version != application.entity_resolution_policy_version:
        raise FactExtractionApplicationReplayConflictError(
            "application entity_resolution_policy_version mismatch"
        )
    _validate_result_counts(result)
    return result


def _validate_replay_item_shape(
    item: FactProposalPersistenceItem,
    *,
    proposal: FactProposal,
) -> None:
    if item.outcome == FactProposalPersistenceOutcome.WITHHELD:
        if item.withheld_reason is None:
            raise FactExtractionApplicationReplayConflictError("withheld application item is missing withheld_reason")
        if (
            item.fact_id is not None
            or item.fact_value_id is not None
            or item.subject_entity_id is not None
            or item.referenced_entity_id is not None
            or item.evidence_ids
        ):
            raise FactExtractionApplicationReplayConflictError(
                "withheld application item carries persisted identifiers"
            )
        return
    if item.withheld_reason is not None:
        raise FactExtractionApplicationReplayConflictError(
            "persisted application item must not carry withheld_reason"
        )
    if item.fact_id is None or item.fact_value_id is None:
        raise FactExtractionApplicationReplayConflictError(
            "persisted application item is missing fact identifiers"
        )
    if len(item.evidence_ids) != len(proposal.evidence):
        raise FactExtractionApplicationReplayConflictError(
            "persisted application item evidence_ids do not match the original proposal"
        )


def _validate_resolution_consistency(
    item: FactProposalPersistenceItem,
    *,
    proposal: FactProposal,
) -> None:
    if item.subject_resolution_status == EntityMentionResolutionStatus.RESOLVED:
        if item.subject_entity_id is None:
            raise FactExtractionApplicationReplayConflictError("resolved subject must carry subject_entity_id")
    elif item.subject_resolution_status == EntityMentionResolutionStatus.UNRESOLVED:
        if item.subject_entity_id is not None:
            raise FactExtractionApplicationReplayConflictError("unresolved subject must not carry subject_entity_id")
    else:
        if item.outcome != FactProposalPersistenceOutcome.WITHHELD:
            raise FactExtractionApplicationReplayConflictError(
                "ambiguous or ineligible subject must be withheld"
            )
        if item.subject_entity_id is not None:
            raise FactExtractionApplicationReplayConflictError(
                "withheld subject must not carry subject_entity_id"
            )

    if proposal.value_type != FactValueType.ENTITY_REF:
        if item.referenced_resolution_status is not None or item.referenced_entity_id is not None:
            raise FactExtractionApplicationReplayConflictError(
                "non-entity_ref proposal must not carry referenced entity replay state"
            )
        return

    if item.referenced_resolution_status is None:
        raise FactExtractionApplicationReplayConflictError(
            "entity_ref replay item must carry referenced resolution status"
        )
    if item.referenced_resolution_status == EntityMentionResolutionStatus.RESOLVED:
        if item.referenced_entity_id is None:
            raise FactExtractionApplicationReplayConflictError(
                "resolved entity_ref must carry referenced_entity_id"
            )
    elif item.referenced_resolution_status == EntityMentionResolutionStatus.UNRESOLVED:
        if item.referenced_entity_id is not None:
            raise FactExtractionApplicationReplayConflictError(
                "unresolved entity_ref must not carry referenced_entity_id"
            )
    else:
        if item.outcome != FactProposalPersistenceOutcome.WITHHELD:
            raise FactExtractionApplicationReplayConflictError(
                "ambiguous or ineligible entity_ref must be withheld"
            )
        if item.referenced_entity_id is not None:
            raise FactExtractionApplicationReplayConflictError(
                "withheld entity_ref must not carry referenced_entity_id"
            )


def _validate_persistence_context(
    *,
    project_id: uuid.UUID,
    extraction_run_id: uuid.UUID,
    inference_run_id: uuid.UUID,
    context: CompletedFactExtractionPersistenceContext | None,
) -> tuple[CompletedFactExtractionPersistenceContext, FactExtractionResponse]:
    _require_uuid_instance(project_id, field_name="project_id")
    _require_uuid_instance(extraction_run_id, field_name="extraction_run_id")
    _require_uuid_instance(inference_run_id, field_name="inference_run_id")
    if context is None:
        raise FactExtractionPersistenceContextError("completed inference run context not found")
    if context.inference_run_id != inference_run_id:
        raise FactExtractionPersistenceContextError("persistence context run_id mismatch")
    if context.status != InferenceRunStatus.COMPLETED.value:
        raise FactExtractionPersistenceContextError("inference run must be completed")
    if context.task_type != InferenceTaskType.FACT_EXTRACTION.value:
        raise FactExtractionPersistenceContextError("inference run task_type must be fact_extraction")
    if context.project_id != project_id or context.batch_project_id != project_id:
        raise FactExtractionPersistenceContextError("inference run or batch project mismatch")
    if context.batch_task_type != context.task_type or context.batch_task_type != InferenceTaskType.FACT_EXTRACTION.value:
        raise FactExtractionPersistenceContextError("inference run or batch task_type mismatch")
    response_hash = _require_sha256(context.response_hash, field_name="response_hash")
    response_json_hash = _require_sha256(
        context.response_json_hash,
        field_name="response_json_hash",
    )
    batch_snapshot_hash = _require_sha256(
        context.batch_snapshot_hash,
        field_name="batch_snapshot_hash",
    )

    from app.agents.fact_extraction import parse_fact_extraction_response_object

    try:
        response = parse_fact_extraction_response_object(context.response_json)
    except Exception:
        raise FactExtractionPersistenceContextError(
            "stored inference response_json is not a valid fact extraction response"
        ) from None

    if context.batch_block_count <= 0:
        raise FactExtractionPersistenceContextError("inference input batch must contain at least one block")
    if len(context.blocks) != context.batch_block_count:
        raise FactExtractionPersistenceContextError("input block count does not match the batch header")
    actual_order = [block.source_order for block in context.blocks]
    if actual_order != list(range(context.batch_block_count)):
        raise FactExtractionPersistenceContextError("input blocks must be continuous from source_order 0")
    actual_character_count = sum(len(block.content_text) for block in context.blocks)
    if actual_character_count != context.batch_character_count:
        raise FactExtractionPersistenceContextError("input block character count does not match the batch header")

    for block in context.blocks:
        expected_block_ref = _block_ref_for_order(block.source_order)
        if block.block_ref != expected_block_ref:
            raise FactExtractionPersistenceContextError("input block_ref does not match source_order")
        if (
            block.document_block_id is None
            or block.document_block_extraction_run_id is None
            or block.document_block_project_id is None
            or block.document_block_raw_text is None
        ):
            raise FactExtractionPersistenceContextError("live document block context is missing")
        if block.document_block_id != block.source_block_id_snapshot:
            raise FactExtractionPersistenceContextError("document_block_id must match source_block_id_snapshot")
        if block.extraction_run_id_snapshot != extraction_run_id:
            raise FactExtractionPersistenceContextError("input block extraction_run snapshot mismatch")
        if block.document_block_extraction_run_id != extraction_run_id:
            raise FactExtractionPersistenceContextError("document block extraction_run mismatch")
        if block.document_block_project_id != project_id:
            raise FactExtractionPersistenceContextError("document block project mismatch")
        if block.content_text != block.document_block_raw_text:
            raise FactExtractionPersistenceContextError("input block content_text must match live document raw_text")
        if _sha256_text(block.content_text) != block.content_hash:
            raise FactExtractionPersistenceContextError("input block content_hash mismatch")

    recomputed_snapshot_hash = build_inference_input_batch_snapshot_hash(
        _build_snapshot_records(context.blocks)
    )
    if recomputed_snapshot_hash != batch_snapshot_hash:
        raise FactExtractionPersistenceContextError("input batch snapshot_hash mismatch")
    recomputed_response_json_hash = build_inference_response_json_hash(context.response_json)
    if recomputed_response_json_hash != response_json_hash:
        raise FactExtractionPersistenceContextError("stored response_json_hash mismatch")
    _require_sha256(response_hash, field_name="response_hash")

    from app.services.fact_extraction_execution import (
        validate_fact_extraction_response_against_batch,
    )

    try:
        validate_fact_extraction_response_against_batch(
            response=response,
            blocks=_build_block_snapshots(
                input_batch_id=context.input_batch_id,
                blocks=context.blocks,
            ),
        )
    except Exception:
        raise FactExtractionPersistenceContextError(
            "stored inference response is not valid for the persisted input batch"
        ) from None
    return context, response


async def resolve_entity_mention(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    entity_type: str,
    mention_key: str,
) -> EntityMentionResolution:
    normalized_entity_type = normalize_entity_type(entity_type)
    normalized_mention_key = normalize_entity_alias(mention_key)

    canonical = await entity_repository.get_entity_context_by_identity(
        session,
        project_id=project_id,
        entity_type=normalized_entity_type,
        canonical_key=normalized_mention_key,
    )
    if canonical is not None:
        if canonical.status == EntityStatus.ACTIVE.value:
            return EntityMentionResolution(
                status=EntityMentionResolutionStatus.RESOLVED.value,
                normalized_entity_type=normalized_entity_type,
                normalized_mention_key=normalized_mention_key,
                entity_id=canonical.entity_id,
                canonical_key=canonical.canonical_key,
                candidate_count=1,
            )
        return EntityMentionResolution(
            status=EntityMentionResolutionStatus.INELIGIBLE.value,
            normalized_entity_type=normalized_entity_type,
            normalized_mention_key=normalized_mention_key,
            entity_id=None,
            canonical_key=None,
            candidate_count=1,
        )

    candidates = await entity_repository.list_active_entity_contexts_by_alias(
        session,
        project_id=project_id,
        entity_type=normalized_entity_type,
        normalized_alias=normalized_mention_key,
    )
    if not candidates:
        return EntityMentionResolution(
            status=EntityMentionResolutionStatus.UNRESOLVED.value,
            normalized_entity_type=normalized_entity_type,
            normalized_mention_key=normalized_mention_key,
            entity_id=None,
            canonical_key=None,
            candidate_count=0,
        )
    if len(candidates) > 1:
        return EntityMentionResolution(
            status=EntityMentionResolutionStatus.AMBIGUOUS.value,
            normalized_entity_type=normalized_entity_type,
            normalized_mention_key=normalized_mention_key,
            entity_id=None,
            canonical_key=None,
            candidate_count=len(candidates),
        )
    candidate = candidates[0]
    return EntityMentionResolution(
        status=EntityMentionResolutionStatus.RESOLVED.value,
        normalized_entity_type=normalized_entity_type,
        normalized_mention_key=normalized_mention_key,
        entity_id=candidate.entity_id,
        canonical_key=candidate.canonical_key,
        candidate_count=1,
    )


def _proposal_hash(proposal: FactProposal) -> str:
    return _sha256_text(proposal.dedupe_signature)


def _withheld_item(
    *,
    proposal_index: int,
    proposal: FactProposal,
    withheld_reason: FactProposalWithheldReason,
    subject_resolution_status: EntityMentionResolutionStatus,
    referenced_resolution_status: EntityMentionResolutionStatus | None = None,
) -> FactProposalPersistenceItem:
    return FactProposalPersistenceItem(
        proposal_index=proposal_index,
        proposal_hash=_proposal_hash(proposal),
        outcome=FactProposalPersistenceOutcome.WITHHELD,
        withheld_reason=withheld_reason,
        subject_resolution_status=subject_resolution_status,
        referenced_resolution_status=referenced_resolution_status,
        fact_id=None,
        fact_value_id=None,
        subject_entity_id=None,
        referenced_entity_id=None,
        evidence_ids=(),
    )


def _build_identity_input(
    *,
    proposal: FactProposal,
    subject_resolution: EntityMentionResolution,
) -> FactIdentityInput:
    if subject_resolution.status == EntityMentionResolutionStatus.RESOLVED.value:
        return FactIdentityInput(
            subject_kind=subject_resolution.normalized_entity_type,
            subject_key=subject_resolution.canonical_key,
            subject_entity_id=subject_resolution.entity_id,
            predicate_key=proposal.predicate_key,
            scope_key=proposal.scope_key,
        )
    return FactIdentityInput(
        subject_kind=subject_resolution.normalized_entity_type,
        subject_key=subject_resolution.normalized_mention_key,
        subject_entity_id=None,
        predicate_key=proposal.predicate_key,
        scope_key=proposal.scope_key,
    )


def _build_value_input(
    *,
    proposal: FactProposal,
    referenced_resolution: EntityMentionResolution | None,
) -> FactValueInput:
    if proposal.value_type != FactValueType.ENTITY_REF:
        return FactValueInput(
            value_type=proposal.value_type,
            value_json=proposal.value_json,
            referenced_entity_id=None,
            language_code=proposal.language_code,
            confidence=proposal.confidence,
        )

    assert referenced_resolution is not None
    return FactValueInput(
        value_type=proposal.value_type,
        value_json={
            "kind": referenced_resolution.normalized_entity_type,
            "key": referenced_resolution.canonical_key,
        },
        referenced_entity_id=referenced_resolution.entity_id,
        language_code=proposal.language_code,
        confidence=proposal.confidence,
    )


def _build_evidence_inputs(
    *,
    proposal: FactProposal,
    block_by_ref: dict[str, FactExtractionPersistenceBlock],
    evidence_ids: tuple[uuid.UUID, ...],
) -> list[FactEvidenceInput]:
    if len(evidence_ids) != len(proposal.evidence):
        raise FactExtractionPersistenceContextError("evidence materialization count mismatch")
    supporting_seen = 0
    evidence_inputs: list[FactEvidenceInput] = []
    for index, evidence in enumerate(proposal.evidence):
        if evidence.block_ref not in block_by_ref:
            raise FactExtractionPersistenceContextError("proposal evidence block_ref is not present in the batch")
        is_primary = False
        if evidence.role == "supporting":
            supporting_seen += 1
            is_primary = supporting_seen == 1
        evidence_inputs.append(
            FactEvidenceInput(
                evidence_id=evidence_ids[index],
                role=evidence.role,
                is_primary=is_primary,
            )
        )
    if supporting_seen == 0:
        raise FactExtractionPersistenceContextError("proposal must contain at least one supporting evidence")
    return evidence_inputs


async def _validate_replayed_result_against_database(
    session: AsyncSession,
    *,
    result: FactExtractionBatchPersistenceResult,
    application: FactExtractionBatchApplication,
    response: FactExtractionResponse,
) -> None:
    _validate_result_counts(result)
    if len(result.items) != len(response.facts):
        raise FactExtractionApplicationReplayConflictError(
            "application result item count does not match the original response"
        )
    for proposal_index, (item, proposal) in enumerate(zip(result.items, response.facts, strict=True)):
        if item.proposal_index != proposal_index:
            raise FactExtractionApplicationReplayConflictError("application result proposal_index mismatch")
        if item.proposal_hash != _proposal_hash(proposal):
            raise FactExtractionApplicationReplayConflictError("application result proposal_hash mismatch")
        _validate_replay_item_shape(item, proposal=proposal)
        _validate_resolution_consistency(item, proposal=proposal)
        if item.outcome == FactProposalPersistenceOutcome.WITHHELD:
            continue

        fact_value = await fact_repository.get_fact_value_with_links(
            session,
            fact_value_id=item.fact_value_id,
        )
        if fact_value is None:
            raise FactExtractionApplicationReplayConflictError("application fact_value record is missing")
        if fact_value.source_kind != FactValueSourceKind.AI.value:
            raise FactExtractionApplicationReplayConflictError("application fact_value must remain an AI source")
        if fact_value.extraction_run_id != application.extraction_run_id:
            raise FactExtractionApplicationReplayConflictError("application fact_value extraction_run_id mismatch")
        if fact_value.inference_run_id != application.inference_run_id:
            raise FactExtractionApplicationReplayConflictError("application fact_value inference_run_id mismatch")
        if fact_value.fact_id != item.fact_id:
            raise FactExtractionApplicationReplayConflictError("application fact_value fact_id mismatch")
        if fact_value.referenced_entity_id != item.referenced_entity_id:
            raise FactExtractionApplicationReplayConflictError("application referenced_entity_id mismatch")
        if fact_value.fact is None:
            raise FactExtractionApplicationReplayConflictError("application fact record is missing")
        if fact_value.fact.project_id != application.project_id:
            raise FactExtractionApplicationReplayConflictError("application fact project_id mismatch")
        if fact_value.fact.subject_entity_id != item.subject_entity_id:
            raise FactExtractionApplicationReplayConflictError("application subject_entity_id mismatch")

        ordered_links = sorted(
            fact_value.evidence_links,
            key=lambda link: (link.source_order, str(link.id)),
        )
        if len(ordered_links) != len(proposal.evidence):
            raise FactExtractionApplicationReplayConflictError("application evidence link count mismatch")
        supporting_seen = 0
        for evidence_index, (link, expected) in enumerate(zip(ordered_links, proposal.evidence, strict=True)):
            if link.evidence_id != item.evidence_ids[evidence_index]:
                raise FactExtractionApplicationReplayConflictError("application evidence link ids do not match")
            if link.source_order != evidence_index:
                raise FactExtractionApplicationReplayConflictError("application evidence link source_order is inconsistent")
            if link.role != expected.role:
                raise FactExtractionApplicationReplayConflictError("application evidence link role mismatch")
            expected_is_primary = False
            if expected.role == "supporting":
                supporting_seen += 1
                expected_is_primary = supporting_seen == 1
            if link.is_primary != expected_is_primary:
                raise FactExtractionApplicationReplayConflictError("application evidence link is_primary mismatch")


async def _replay_completed_application(
    session: AsyncSession,
    *,
    application: FactExtractionBatchApplication,
    context: CompletedFactExtractionPersistenceContext,
    project_id: uuid.UUID,
    extraction_run_id: uuid.UUID,
    response: FactExtractionResponse,
) -> FactExtractionBatchPersistenceResult:
    if application.status != _APPLICATION_STATUS_COMPLETED:
        raise FactExtractionApplicationReplayConflictError("application is not completed")
    if (
        application.inference_run_id != context.inference_run_id
        or application.project_id != project_id
        or application.extraction_run_id != extraction_run_id
        or application.input_batch_id != context.input_batch_id
    ):
        raise FactExtractionApplicationReplayConflictError("application identity does not match the replay request")
    if application.response_hash != context.response_hash:
        raise FactExtractionApplicationReplayConflictError("application response_hash mismatch")
    if application.response_json_hash != context.response_json_hash:
        raise FactExtractionApplicationReplayConflictError("application response_json_hash mismatch")
    if (
        application.result_json is None
        or application.result_hash is None
        or application.completed_at is None
    ):
        raise FactExtractionApplicationReplayConflictError("application completed shape is invalid")
    recomputed_result_hash = build_fact_extraction_application_result_hash(application.result_json)
    if recomputed_result_hash != application.result_hash:
        raise FactExtractionApplicationReplayConflictError("application result_hash does not match stored result_json")

    try:
        stored_result = FactExtractionBatchPersistenceResult.model_validate(application.result_json)
    except Exception:
        raise FactExtractionApplicationReplayConflictError("application result_json is not a valid persistence result") from None

    if stored_result.application_id != application.id:
        raise FactExtractionApplicationReplayConflictError("application result_json has the wrong application_id")
    if stored_result.replayed_application is not False:
        raise FactExtractionApplicationReplayConflictError("stored application result must record replayed_application=false")
    if (
        stored_result.project_id != project_id
        or stored_result.extraction_run_id != extraction_run_id
        or stored_result.inference_run_id != context.inference_run_id
        or stored_result.input_batch_id != context.input_batch_id
        or stored_result.response_hash != context.response_hash
    ):
        raise FactExtractionApplicationReplayConflictError("stored application result header does not match the application")
    if stored_result.persistence_name != application.persistence_name:
        raise FactExtractionApplicationReplayConflictError("stored application persistence_name mismatch")
    if stored_result.persistence_version != application.persistence_version:
        raise FactExtractionApplicationReplayConflictError("stored application persistence_version mismatch")
    if stored_result.entity_resolution_policy_name != application.entity_resolution_policy_name:
        raise FactExtractionApplicationReplayConflictError("stored application entity resolution policy mismatch")
    if stored_result.entity_resolution_policy_version != application.entity_resolution_policy_version:
        raise FactExtractionApplicationReplayConflictError("stored application entity resolution policy version mismatch")

    await _validate_replayed_result_against_database(
        session,
        result=stored_result,
        application=application,
        response=response,
    )
    return stored_result.model_copy(update={"replayed_application": True})


async def _prepare_batch_application(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    extraction_run_id: uuid.UUID,
    context: CompletedFactExtractionPersistenceContext,
    response: FactExtractionResponse,
) -> PreparedBatchApplication:
    existing = await persistence_repository.get_batch_application_for_update(
        session,
        inference_run_id=context.inference_run_id,
    )
    if existing is not None:
        replay = await _replay_completed_application(
            session,
            application=existing,
            context=context,
            project_id=project_id,
            extraction_run_id=extraction_run_id,
            response=response,
        )
        return PreparedBatchApplication(application=existing, replay_result=replay)

    application = FactExtractionBatchApplication(
        inference_run_id=context.inference_run_id,
        project_id=project_id,
        extraction_run_id=extraction_run_id,
        input_batch_id=context.input_batch_id,
        response_hash=context.response_hash,
        response_json_hash=context.response_json_hash,
        status=_APPLICATION_STATUS_APPLYING,
        persistence_name=FACT_EXTRACTION_PERSISTENCE_NAME,
        persistence_version=FACT_EXTRACTION_PERSISTENCE_VERSION,
        entity_resolution_policy_name=ENTITY_RESOLUTION_POLICY_NAME,
        entity_resolution_policy_version=ENTITY_RESOLUTION_POLICY_VERSION,
    )

    savepoint = await session.begin_nested()
    try:
        await persistence_repository.create_batch_application(session, application)
        await savepoint.commit()
        return PreparedBatchApplication(application=application, replay_result=None)
    except IntegrityError as error:
        await savepoint.rollback()
        constraint_name = getattr(getattr(error.orig, "diag", None), "constraint_name", None)
        if constraint_name != _APPLICATION_CONSTRAINT:
            raise
        existing = await persistence_repository.get_batch_application_for_update(
            session,
            inference_run_id=context.inference_run_id,
        )
        if existing is None:
            raise error
        replay = await _replay_completed_application(
            session,
            application=existing,
            context=context,
            project_id=project_id,
            extraction_run_id=extraction_run_id,
            response=response,
        )
        return PreparedBatchApplication(application=existing, replay_result=replay)


def _build_result(
    *,
    application_id: uuid.UUID,
    project_id: uuid.UUID,
    extraction_run_id: uuid.UUID,
    inference_run_id: uuid.UUID,
    input_batch_id: uuid.UUID,
    response_hash: str,
    items: Sequence[FactProposalPersistenceItem],
) -> FactExtractionBatchPersistenceResult:
    items_tuple = tuple(items)
    created_count = sum(item.outcome == FactProposalPersistenceOutcome.CREATED for item in items_tuple)
    reused_count = sum(item.outcome == FactProposalPersistenceOutcome.REUSED for item in items_tuple)
    withheld_count = sum(item.outcome == FactProposalPersistenceOutcome.WITHHELD for item in items_tuple)
    return FactExtractionBatchPersistenceResult(
        application_id=application_id,
        replayed_application=False,
        project_id=project_id,
        extraction_run_id=extraction_run_id,
        inference_run_id=inference_run_id,
        input_batch_id=input_batch_id,
        response_hash=response_hash,
        persistence_name=FACT_EXTRACTION_PERSISTENCE_NAME,
        persistence_version=FACT_EXTRACTION_PERSISTENCE_VERSION,
        entity_resolution_policy_name=ENTITY_RESOLUTION_POLICY_NAME,
        entity_resolution_policy_version=ENTITY_RESOLUTION_POLICY_VERSION,
        proposal_count=len(items_tuple),
        created_count=created_count,
        reused_count=reused_count,
        withheld_count=withheld_count,
        items=items_tuple,
    )


def _build_authenticated_application_snapshot(
    *,
    application: FactExtractionBatchApplication,
    result: FactExtractionBatchPersistenceResult,
) -> AuthenticatedCompletedFactExtractionApplicationSnapshot:
    persisted_items = tuple(
        AuthenticatedPersistedFactProposalItem(
            proposal_index=item.proposal_index,
            fact_id=item.fact_id,
            fact_value_id=item.fact_value_id,
            subject_entity_id=item.subject_entity_id,
            referenced_entity_id=item.referenced_entity_id,
            evidence_ids=item.evidence_ids,
        )
        for item in sorted(result.items, key=lambda current: current.proposal_index)
        if item.outcome != FactProposalPersistenceOutcome.WITHHELD
    )
    return AuthenticatedCompletedFactExtractionApplicationSnapshot(
        application_id=application.id,
        project_id=application.project_id,
        extraction_run_id=application.extraction_run_id,
        inference_run_id=application.inference_run_id,
        input_batch_id=application.input_batch_id,
        persistence_name=application.persistence_name,
        persistence_version=application.persistence_version,
        entity_resolution_policy_name=application.entity_resolution_policy_name,
        entity_resolution_policy_version=application.entity_resolution_policy_version,
        items=persisted_items,
        result_hash=application.result_hash,
    )


async def authenticate_completed_fact_extraction_application(
    session: AsyncSession,
    *,
    application_id: uuid.UUID,
) -> AuthenticatedCompletedFactExtractionApplicationSnapshot:
    _require_uuid_instance(application_id, field_name="application_id")
    application = await persistence_repository.get_batch_application_by_id(
        session,
        application_id=application_id,
    )
    if application is None:
        raise FactExtractionApplicationReplayConflictError("application not found")
    validate_fact_extraction_application_result_envelope(application=application)
    context = await persistence_repository.get_completed_fact_extraction_persistence_context(
        session,
        inference_run_id=application.inference_run_id,
    )
    context, response = _validate_persistence_context(
        project_id=application.project_id,
        extraction_run_id=application.extraction_run_id,
        inference_run_id=application.inference_run_id,
        context=context,
    )
    replay_result = await _replay_completed_application(
        session,
        application=application,
        context=context,
        project_id=application.project_id,
        extraction_run_id=application.extraction_run_id,
        response=response,
    )
    return _build_authenticated_application_snapshot(
        application=application,
        result=replay_result,
    )


async def persist_completed_fact_extraction_batch(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    extraction_run_id: uuid.UUID,
    inference_run_id: uuid.UUID,
) -> FactExtractionBatchPersistenceResult:
    try:
        context = await persistence_repository.get_completed_fact_extraction_persistence_context(
            session,
            inference_run_id=inference_run_id,
        )
        context, response = _validate_persistence_context(
            project_id=project_id,
            extraction_run_id=extraction_run_id,
            inference_run_id=inference_run_id,
            context=context,
        )
        prepared_application = await _prepare_batch_application(
            session,
            project_id=project_id,
            extraction_run_id=extraction_run_id,
            context=context,
            response=response,
        )
        if prepared_application.replay_result is not None:
            await session.commit()
            return prepared_application.replay_result

        block_by_ref = {block.block_ref: block for block in context.blocks}
        items: list[FactProposalPersistenceItem] = []
        for proposal_index, proposal in enumerate(response.facts):
            subject_resolution = await resolve_entity_mention(
                session,
                project_id=project_id,
                entity_type=proposal.subject_kind,
                mention_key=proposal.subject_key,
            )
            subject_status = EntityMentionResolutionStatus(subject_resolution.status)
            if subject_status == EntityMentionResolutionStatus.AMBIGUOUS:
                items.append(
                    _withheld_item(
                        proposal_index=proposal_index,
                        proposal=proposal,
                        withheld_reason=FactProposalWithheldReason.SUBJECT_AMBIGUOUS,
                        subject_resolution_status=subject_status,
                    )
                )
                continue
            if subject_status == EntityMentionResolutionStatus.INELIGIBLE:
                items.append(
                    _withheld_item(
                        proposal_index=proposal_index,
                        proposal=proposal,
                        withheld_reason=FactProposalWithheldReason.SUBJECT_INELIGIBLE,
                        subject_resolution_status=subject_status,
                    )
                )
                continue

            referenced_resolution: EntityMentionResolution | None = None
            referenced_status: EntityMentionResolutionStatus | None = None
            if proposal.value_type == FactValueType.ENTITY_REF:
                if (
                    not isinstance(proposal.value_json, dict)
                    or set(proposal.value_json.keys()) != {"kind", "key"}
                    or not isinstance(proposal.value_json["kind"], str)
                    or not isinstance(proposal.value_json["key"], str)
                ):
                    raise FactExtractionPersistenceContextError("entity_ref proposal value must be an object")
                referenced_resolution = await resolve_entity_mention(
                    session,
                    project_id=project_id,
                    entity_type=proposal.value_json["kind"],
                    mention_key=proposal.value_json["key"],
                )
                referenced_status = EntityMentionResolutionStatus(referenced_resolution.status)
                if referenced_status != EntityMentionResolutionStatus.RESOLVED:
                    reason = {
                        EntityMentionResolutionStatus.UNRESOLVED: FactProposalWithheldReason.ENTITY_REF_UNRESOLVED,
                        EntityMentionResolutionStatus.AMBIGUOUS: FactProposalWithheldReason.ENTITY_REF_AMBIGUOUS,
                        EntityMentionResolutionStatus.INELIGIBLE: FactProposalWithheldReason.ENTITY_REF_INELIGIBLE,
                    }[referenced_status]
                    items.append(
                        _withheld_item(
                            proposal_index=proposal_index,
                            proposal=proposal,
                            withheld_reason=reason,
                            subject_resolution_status=subject_status,
                            referenced_resolution_status=referenced_status,
                        )
                    )
                    continue

            savepoint = await session.begin_nested()
            try:
                evidence_ids: list[uuid.UUID] = []
                for evidence in proposal.evidence:
                    block = block_by_ref[evidence.block_ref]
                    materialized, _created = await get_or_create_source_evidence_in_transaction(
                        session,
                        block_id=block.document_block_id,
                        raw_text=block.document_block_raw_text,
                        start_offset=evidence.start_offset,
                        end_offset=evidence.end_offset,
                    )
                    evidence_ids.append(materialized.id)

                identity = _build_identity_input(
                    proposal=proposal,
                    subject_resolution=subject_resolution,
                )
                value = _build_value_input(
                    proposal=proposal,
                    referenced_resolution=referenced_resolution,
                )
                payload = AIProposalInput(
                    identity=identity,
                    value=value,
                    evidences=_build_evidence_inputs(
                        proposal=proposal,
                        block_by_ref=block_by_ref,
                        evidence_ids=tuple(evidence_ids),
                    ),
                )
                persisted = await propose_ai_fact_value_in_transaction(
                    session,
                    project_id=project_id,
                    extraction_run_id=extraction_run_id,
                    inference_run_id=inference_run_id,
                    payload=payload,
                )
            except (RetiredFactError, FactSubjectEntityConflictError) as error:
                await savepoint.rollback()
                items.append(
                    _withheld_item(
                        proposal_index=proposal_index,
                        proposal=proposal,
                        withheld_reason=(
                            FactProposalWithheldReason.RETIRED_FACT
                            if isinstance(error, RetiredFactError)
                            else FactProposalWithheldReason.SUBJECT_ENTITY_CONFLICT
                        ),
                        subject_resolution_status=subject_status,
                        referenced_resolution_status=referenced_status,
                    )
                )
                continue
            except BaseException:
                await savepoint.rollback()
                raise
            else:
                await savepoint.commit()
                items.append(
                    FactProposalPersistenceItem(
                        proposal_index=proposal_index,
                        proposal_hash=_proposal_hash(proposal),
                        outcome=(
                            FactProposalPersistenceOutcome.CREATED
                            if persisted.created
                            else FactProposalPersistenceOutcome.REUSED
                        ),
                        withheld_reason=None,
                        subject_resolution_status=subject_status,
                        referenced_resolution_status=referenced_status,
                        fact_id=persisted.fact_value.fact_id,
                        fact_value_id=persisted.fact_value.id,
                        subject_entity_id=identity.subject_entity_id,
                        referenced_entity_id=value.referenced_entity_id,
                        evidence_ids=tuple(evidence_ids),
                    )
                )

        result = _build_result(
            application_id=prepared_application.application.id,
            project_id=project_id,
            extraction_run_id=extraction_run_id,
            inference_run_id=inference_run_id,
            input_batch_id=context.input_batch_id,
            response_hash=context.response_hash,
            items=items,
        )
        result_json = result.model_dump(mode="json")
        prepared_application.application.status = _APPLICATION_STATUS_COMPLETED
        prepared_application.application.result_json = result_json
        prepared_application.application.result_hash = build_fact_extraction_application_result_hash(result_json)
        prepared_application.application.completed_at = utc_now()

        await session.flush()
        await session.commit()
        return result
    except BaseException:
        await session.rollback()
        raise
