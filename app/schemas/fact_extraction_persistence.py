from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, StrictBool


@dataclass(frozen=True, slots=True)
class FactExtractionPersistenceBlock:
    input_block_id: uuid.UUID
    block_ref: str
    source_order: int
    block_type: str
    location_key: str
    anchor_hash: str
    page_no: int | None
    start_line: int | None
    end_line: int | None
    heading_path: tuple[Any, ...]

    document_block_id: uuid.UUID | None
    source_block_id_snapshot: uuid.UUID
    extraction_run_id_snapshot: uuid.UUID

    content_text: str
    content_hash: str

    document_block_extraction_run_id: uuid.UUID | None
    document_block_project_id: uuid.UUID | None
    document_block_raw_text: str | None


@dataclass(frozen=True, slots=True)
class CompletedFactExtractionPersistenceContext:
    inference_run_id: uuid.UUID
    project_id: uuid.UUID
    task_type: str
    status: str

    input_batch_id: uuid.UUID
    batch_project_id: uuid.UUID
    batch_task_type: str
    batch_block_count: int
    batch_character_count: int
    batch_snapshot_hash: str
    response_json: dict[str, Any]
    response_hash: str
    response_json_hash: str

    blocks: tuple[FactExtractionPersistenceBlock, ...]


class EntityMentionResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    AMBIGUOUS = "ambiguous"
    INELIGIBLE = "ineligible"


@dataclass(frozen=True, slots=True)
class EntityMentionResolution:
    status: str

    normalized_entity_type: str
    normalized_mention_key: str

    entity_id: uuid.UUID | None
    canonical_key: str | None

    candidate_count: int


class FactProposalPersistenceOutcome(StrEnum):
    CREATED = "created"
    REUSED = "reused"
    WITHHELD = "withheld"


class FactProposalWithheldReason(StrEnum):
    SUBJECT_AMBIGUOUS = "subject_ambiguous"
    SUBJECT_INELIGIBLE = "subject_ineligible"

    ENTITY_REF_UNRESOLVED = "entity_ref_unresolved"
    ENTITY_REF_AMBIGUOUS = "entity_ref_ambiguous"
    ENTITY_REF_INELIGIBLE = "entity_ref_ineligible"

    RETIRED_FACT = "retired_fact"
    SUBJECT_ENTITY_CONFLICT = "subject_entity_conflict"


class FactProposalPersistenceItem(BaseModel):
    proposal_index: int
    proposal_hash: str

    outcome: FactProposalPersistenceOutcome
    withheld_reason: FactProposalWithheldReason | None

    subject_resolution_status: EntityMentionResolutionStatus
    referenced_resolution_status: EntityMentionResolutionStatus | None

    fact_id: uuid.UUID | None
    fact_value_id: uuid.UUID | None
    subject_entity_id: uuid.UUID | None
    referenced_entity_id: uuid.UUID | None

    evidence_ids: tuple[uuid.UUID, ...]

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )


class FactExtractionBatchPersistenceResult(BaseModel):
    application_id: uuid.UUID
    replayed_application: StrictBool
    project_id: uuid.UUID
    extraction_run_id: uuid.UUID
    inference_run_id: uuid.UUID
    input_batch_id: uuid.UUID
    response_hash: str

    persistence_name: str
    persistence_version: str
    entity_resolution_policy_name: str
    entity_resolution_policy_version: str

    proposal_count: int
    created_count: int
    reused_count: int
    withheld_count: int

    items: tuple[FactProposalPersistenceItem, ...]

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )
