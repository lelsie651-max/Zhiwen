from __future__ import annotations

import hashlib
import re
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.fact_extraction import parse_fact_extraction_response_object
from app.models.entity import EntityStatus, normalize_entity_alias
from app.models.fact import FactValueType
from app.models.inference import InferenceRunStatus, InferenceTaskType
from app.repositories import entity as entity_repository
from app.repositories import fact_extraction_persistence as persistence_repository
from app.schemas.agent_fact_extraction import FactExtractionResponse, FactProposal
from app.schemas.fact import FactIdentityInput, FactValueInput
from app.schemas.fact_commands import AIProposalInput, FactEvidenceInput
from app.schemas.fact_extraction_execution import InferenceInputBlockSnapshot
from app.schemas.fact_extraction_persistence import (
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
from app.services.fact_extraction_execution import validate_fact_extraction_response_against_batch


FACT_EXTRACTION_PERSISTENCE_NAME = "agent1_fact_persistence"
FACT_EXTRACTION_PERSISTENCE_VERSION = "1.0.0"

ENTITY_RESOLUTION_POLICY_NAME = "canonical_then_unique_active_alias"
ENTITY_RESOLUTION_POLICY_VERSION = "1.0.0"

_BLOCK_REF_PATTERN = re.compile(r"^B[0-9]{4,}$")


class FactExtractionPersistenceError(Exception):
    """Base class for completed batch persistence failures."""


class FactExtractionPersistenceContextError(FactExtractionPersistenceError):
    """Raised when the stored inference context is not safe to persist."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_uuid_instance(value: uuid.UUID, *, field_name: str) -> uuid.UUID:
    if not isinstance(value, uuid.UUID):
        raise FactExtractionPersistenceContextError(f"{field_name} must be a UUID")
    return value


def _build_block_snapshots(
    *,
    input_batch_id: uuid.UUID,
    blocks: tuple[FactExtractionPersistenceBlock, ...],
) -> tuple[InferenceInputBlockSnapshot, ...]:
    return tuple(
        InferenceInputBlockSnapshot(
            id=block.input_block_id,
            batch_id=input_batch_id,
            source_order=block.source_order,
            block_ref=block.block_ref,
            document_block_id=block.document_block_id,
            source_block_id_snapshot=block.source_block_id_snapshot,
            extraction_run_id_snapshot=block.extraction_run_id_snapshot,
            block_type="paragraph",
            location_key="persisted",
            anchor_hash="0" * 64,
            content_text=block.content_text,
            content_hash=block.content_hash,
        )
        for block in blocks
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
    if context.project_id != project_id:
        raise FactExtractionPersistenceContextError("inference run project mismatch")
    if context.response_hash is None:
        raise FactExtractionPersistenceContextError("inference run response payload is missing")

    try:
        response = parse_fact_extraction_response_object(context.response_json)
    except Exception as error:
        raise FactExtractionPersistenceContextError(
            "stored inference response_json is not a valid fact extraction response"
        ) from None

    if not context.blocks:
        raise FactExtractionPersistenceContextError("inference input batch must contain blocks")
    expected_order = list(range(len(context.blocks)))
    actual_order = [block.source_order for block in context.blocks]
    if actual_order != expected_order:
        raise FactExtractionPersistenceContextError("input blocks must be continuous from source_order 0")

    block_refs: set[str] = set()
    for block in context.blocks:
        if block.document_block_id != block.source_block_id_snapshot:
            raise FactExtractionPersistenceContextError("document_block_id must match source_block_id_snapshot")
        if block.source_block_id_snapshot != block.document_block_id:
            raise FactExtractionPersistenceContextError("source block snapshot must match live document block")
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
        if not _BLOCK_REF_PATTERN.fullmatch(block.block_ref):
            raise FactExtractionPersistenceContextError("input block_ref format is invalid")
        if block.block_ref in block_refs:
            raise FactExtractionPersistenceContextError("input block_ref values must be unique")
        block_refs.add(block.block_ref)

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

        await session.flush()
        await session.commit()
    except BaseException:
        await session.rollback()
        raise

    items_tuple = tuple(items)
    created_count = sum(item.outcome == FactProposalPersistenceOutcome.CREATED for item in items_tuple)
    reused_count = sum(item.outcome == FactProposalPersistenceOutcome.REUSED for item in items_tuple)
    withheld_count = sum(item.outcome == FactProposalPersistenceOutcome.WITHHELD for item in items_tuple)
    return FactExtractionBatchPersistenceResult(
        project_id=project_id,
        extraction_run_id=extraction_run_id,
        inference_run_id=inference_run_id,
        input_batch_id=context.input_batch_id,
        response_hash=context.response_hash,
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
