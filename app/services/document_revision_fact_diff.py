from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories import document_revision_fact_diff as document_revision_fact_diff_repository
from app.repositories.document_revision_fact_diff import (
    DocumentRevisionFactDiffSourceRow,
)
from app.schemas.fact import FactIdentityInput, FactValueInput
from app.schemas.document_revision_fact_diff import (
    DocumentRevisionFactDiff,
    DocumentRevisionFactDiffChangeKind,
    DocumentRevisionFactDiffEvidenceLocator,
    DocumentRevisionFactDiffEvidenceSnapshot,
    DocumentRevisionFactDiffFactSnapshot,
    DocumentRevisionFactDiffItem,
    DocumentRevisionFactDiffValueGroup,
)
from app.schemas.fact_value_duplicate_grouping import (
    CROSS_BATCH_DUPLICATE_ALGORITHM_VERSION,
    DuplicateCandidate,
)
from app.services import document_revision_diff as document_revision_diff_service
from app.services import fact as fact_service
from app.services import fact_value_duplicate_grouping as duplicate_grouping_service

DOCUMENT_REVISION_FACT_DIFF_ALGORITHM_NAME = "document_revision_fact_diff"
DOCUMENT_REVISION_FACT_DIFF_ALGORITHM_VERSION = "1.0.0"
_SEMANTIC_FINGERPRINT_ORCHESTRATION_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
_SEMANTIC_FINGERPRINT_EXTRACTION_RUN_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")
_SEMANTIC_FINGERPRINT_SOURCE_BATCH_ID = uuid.UUID("00000000-0000-0000-0000-000000000003")


class DocumentRevisionFactDiffError(Exception):
    """Base class for adjacent revision fact diff failures."""


class DocumentRevisionFactDiffStateError(DocumentRevisionFactDiffError):
    """Raised when requested revisions or orchestrations are not comparable."""


class DocumentRevisionFactDiffInvariantError(DocumentRevisionFactDiffError):
    """Raised when immutable source facts or evidence drift."""


@dataclass(frozen=True, slots=True)
class _EvidenceSnapshotEnvelope:
    snapshot: DocumentRevisionFactDiffEvidenceSnapshot
    block_source_order: int
    evidence_link_source_order: int


@dataclass(frozen=True, slots=True)
class _ValueGroupEnvelope:
    snapshot: DocumentRevisionFactDiffValueGroup
    earliest_block_source_order: int


@dataclass(frozen=True, slots=True)
class _FactSideSnapshot:
    fact: DocumentRevisionFactDiffFactSnapshot
    value_groups: tuple[_ValueGroupEnvelope, ...]
    semantic_key_hashes: frozenset[str]
    earliest_block_source_order: int


@dataclass(frozen=True, slots=True)
class _CertifiedFactValue:
    value_type: str
    value_json: Any | None
    normalized_value_text: str
    value_hash: str
    referenced_entity_id: uuid.UUID | None


@dataclass(frozen=True, slots=True)
class AuthenticatedFactSourceEvidence:
    evidence_link_id: uuid.UUID
    evidence_id: uuid.UUID
    document_revision_id: uuid.UUID
    document_block_id: uuid.UUID
    locator: DocumentRevisionFactDiffEvidenceLocator
    excerpt: str
    excerpt_hash: str
    content_hash: str
    role: str
    is_primary: bool
    source_order: int
    block_source_order: int


@dataclass(frozen=True, slots=True)
class AuthenticatedFactSourceValue:
    fact_value_id: uuid.UUID
    source_batch_id: uuid.UUID
    source_application_id: uuid.UUID
    proposal_index: int
    normalized_value_text: str
    value_hash: str
    language_code: str | None
    confidence: float | None
    evidences: tuple[AuthenticatedFactSourceEvidence, ...]


@dataclass(frozen=True, slots=True)
class AuthenticatedFactSourceValueGroup:
    semantic_key_hash: str
    value_type: str
    value_json: Any | None
    referenced_entity_id: uuid.UUID | None
    fact_value_ids: tuple[uuid.UUID, ...]
    values: tuple[AuthenticatedFactSourceValue, ...]
    evidences: tuple[AuthenticatedFactSourceEvidence, ...]
    earliest_block_source_order: int


@dataclass(frozen=True, slots=True)
class AuthenticatedFactSourceFact:
    fact: DocumentRevisionFactDiffFactSnapshot
    value_groups: tuple[AuthenticatedFactSourceValueGroup, ...]
    semantic_key_hashes: frozenset[str]
    earliest_block_source_order: int


@dataclass(frozen=True, slots=True)
class _AuthenticatedEvidenceEnvelope:
    snapshot: AuthenticatedFactSourceEvidence
    block_source_order: int
    evidence_link_source_order: int


@dataclass(frozen=True, slots=True)
class _AuthenticatedValueEnvelope:
    snapshot: AuthenticatedFactSourceValue
    application_order: int
    proposal_index: int


def _require_uuid(value: object, *, field_name: str) -> uuid.UUID:
    if not isinstance(value, uuid.UUID):
        raise DocumentRevisionFactDiffStateError(
            f"document_revision_fact_diff_{field_name}_invalid"
        )
    return value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _build_semantic_key_hash(
    *,
    fact_id: uuid.UUID,
    value_type: str,
    value_json: Any | None,
    referenced_entity_id: uuid.UUID | None,
) -> str:
    return duplicate_grouping_service.build_duplicate_fingerprint(
        DuplicateCandidate(
            fact_value_id=uuid.UUID(int=0),
            fact_id=fact_id,
            orchestration_id=_SEMANTIC_FINGERPRINT_ORCHESTRATION_ID,
            extraction_run_id=_SEMANTIC_FINGERPRINT_EXTRACTION_RUN_ID,
            source_batch_id=_SEMANTIC_FINGERPRINT_SOURCE_BATCH_ID,
            value_type=value_type,
            value_json=value_json,
            referenced_entity_id=referenced_entity_id,
            evidence_link_ids=(),
        ),
        algorithm_version=CROSS_BATCH_DUPLICATE_ALGORITHM_VERSION,
    ).sha256_hex


def _build_authoritative_application_item_map(
    source_snapshot: duplicate_grouping_service.AuthenticatedDuplicateGroupingSourceSnapshot,
) -> dict[
    uuid.UUID,
    tuple[
        int,
        duplicate_grouping_service.fact_extraction_persistence_service.AuthenticatedCompletedFactExtractionApplicationSnapshot,
        duplicate_grouping_service.fact_extraction_persistence_service.AuthenticatedPersistedFactProposalItem,
    ],
]:
    item_by_fact_value_id: dict[
        uuid.UUID,
        tuple[
            int,
            duplicate_grouping_service.fact_extraction_persistence_service.AuthenticatedCompletedFactExtractionApplicationSnapshot,
            duplicate_grouping_service.fact_extraction_persistence_service.AuthenticatedPersistedFactProposalItem,
        ],
    ] = {}
    for application_order, application_snapshot in enumerate(
        source_snapshot.application_snapshots
    ):
        for item in application_snapshot.items:
            if item.fact_value_id in item_by_fact_value_id:
                raise DocumentRevisionFactDiffInvariantError(
                    "document_revision_fact_diff_fact_value_source_mismatch"
                )
            item_by_fact_value_id[item.fact_value_id] = (
                application_order,
                application_snapshot,
                item,
            )
    return item_by_fact_value_id


def _certify_fact_identity(
    row: DocumentRevisionFactDiffSourceRow,
) -> DocumentRevisionFactDiffFactSnapshot:
    try:
        identity_input = FactIdentityInput(
            subject_kind=row.subject_kind,
            subject_key=row.subject_key,
            subject_entity_id=row.subject_entity_id,
            predicate_key=row.predicate_key,
            scope_key=row.scope_key,
        )
    except ValidationError:
        raise DocumentRevisionFactDiffInvariantError(
            "document_revision_fact_diff_fact_identity_invalid"
        ) from None
    identity_hash = fact_service.build_fact_identity_hash(identity_input)
    if identity_hash != row.fact_identity_hash:
        raise DocumentRevisionFactDiffInvariantError(
            "document_revision_fact_diff_fact_identity_mismatch"
        ) from None
    return DocumentRevisionFactDiffFactSnapshot(
        fact_id=row.fact_id,
        identity_hash=row.fact_identity_hash,
        subject_kind=row.subject_kind,
        subject_key=row.subject_key,
        predicate_key=row.predicate_key,
        scope_key=row.scope_key,
        subject_entity_id=row.subject_entity_id,
    )


def _certify_fact_value(
    row: DocumentRevisionFactDiffSourceRow,
) -> _CertifiedFactValue:
    try:
        value_input = FactValueInput(
            value_type=row.value_type,
            value_json=row.value_json,
            referenced_entity_id=row.referenced_entity_id,
            language_code=None,
            confidence=None,
        )
        normalized = fact_service.normalize_fact_value_input(value_input)
    except (ValidationError, fact_service.InvalidFactProposalError, ValueError, TypeError):
        raise DocumentRevisionFactDiffInvariantError(
            "document_revision_fact_diff_fact_value_invalid"
        ) from None
    if normalized.value_type != row.value_type:
        raise DocumentRevisionFactDiffInvariantError(
            "document_revision_fact_diff_fact_value_mismatch"
        ) from None
    if normalized.value_json != row.value_json:
        raise DocumentRevisionFactDiffInvariantError(
            "document_revision_fact_diff_fact_value_mismatch"
        ) from None
    if normalized.referenced_entity_id != row.referenced_entity_id:
        raise DocumentRevisionFactDiffInvariantError(
            "document_revision_fact_diff_fact_value_mismatch"
        ) from None
    if normalized.normalized_value_text != row.normalized_value_text:
        raise DocumentRevisionFactDiffInvariantError(
            "document_revision_fact_diff_fact_value_mismatch"
        ) from None
    if normalized.value_hash != row.fact_value_hash:
        raise DocumentRevisionFactDiffInvariantError(
            "document_revision_fact_diff_fact_value_mismatch"
        ) from None
    return _CertifiedFactValue(
        value_type=normalized.value_type,
        value_json=normalized.value_json,
        normalized_value_text=normalized.normalized_value_text,
        value_hash=normalized.value_hash,
        referenced_entity_id=normalized.referenced_entity_id,
    )


def _serialize_fact_snapshot(
    fact: DocumentRevisionFactDiffFactSnapshot | None,
) -> dict[str, object] | None:
    if fact is None:
        return None
    return {
        "fact_id": str(fact.fact_id),
        "identity_hash": fact.identity_hash,
        "subject_kind": fact.subject_kind,
        "subject_key": fact.subject_key,
        "predicate_key": fact.predicate_key,
        "scope_key": fact.scope_key,
        "subject_entity_id": (
            str(fact.subject_entity_id) if fact.subject_entity_id is not None else None
        ),
    }


def _serialize_evidence_snapshot(
    evidence: DocumentRevisionFactDiffEvidenceSnapshot,
) -> dict[str, object]:
    return {
        "evidence_link_id": str(evidence.evidence_link_id),
        "evidence_id": str(evidence.evidence_id),
        "document_block_id": str(evidence.document_block_id),
        "locator": {
            "location_key": evidence.locator.location_key,
            "page_no": evidence.locator.page_no,
            "start_line": evidence.locator.start_line,
            "end_line": evidence.locator.end_line,
            "table_index": evidence.locator.table_index,
            "row_index": evidence.locator.row_index,
        },
        "excerpt": evidence.excerpt,
        "excerpt_hash": evidence.excerpt_hash,
        "block_change_kind": evidence.block_change_kind,
    }


def _serialize_value_group(
    group: DocumentRevisionFactDiffValueGroup,
) -> dict[str, object]:
    return {
        "semantic_key_hash": group.semantic_key_hash,
        "value_type": group.value_type,
        "value_json": group.value_json,
        "referenced_entity_id": (
            str(group.referenced_entity_id)
            if group.referenced_entity_id is not None
            else None
        ),
        "fact_value_ids": [str(fact_value_id) for fact_value_id in group.fact_value_ids],
        "evidences": [_serialize_evidence_snapshot(item) for item in group.evidences],
    }


def _serialize_item(item: DocumentRevisionFactDiffItem) -> dict[str, object]:
    return {
        "change_kind": item.change_kind,
        "base_fact": _serialize_fact_snapshot(item.base_fact),
        "target_fact": _serialize_fact_snapshot(item.target_fact),
        "base_value_groups": [_serialize_value_group(group) for group in item.base_value_groups],
        "target_value_groups": [
            _serialize_value_group(group) for group in item.target_value_groups
        ],
    }


def _build_manifest_hash(
    *,
    diff: DocumentRevisionFactDiff,
) -> str:
    return duplicate_grouping_service.hash_deterministic_payload(
        {
            "project_id": str(diff.project_id),
            "document_id": str(diff.document_id),
            "base_revision": {
                "revision_id": str(diff.base_revision_id),
                "revision_no": diff.base_revision_no,
                "extraction_run_id": str(diff.base_extraction_run_id),
                "orchestration_id": str(diff.base_orchestration_id),
                "orchestration_status": diff.base_orchestration_status,
            },
            "target_revision": {
                "revision_id": str(diff.target_revision_id),
                "revision_no": diff.target_revision_no,
                "extraction_run_id": str(diff.target_extraction_run_id),
                "orchestration_id": str(diff.target_orchestration_id),
                "orchestration_status": diff.target_orchestration_status,
            },
            "block_diff_manifest_hash": diff.block_diff_manifest_hash,
            "fact_diff_algorithm": {
                "name": diff.fact_diff_algorithm_name,
                "version": diff.fact_diff_algorithm_version,
            },
            "semantic_fingerprint_algorithm_version": diff.semantic_fingerprint_algorithm_version,
            "agent_identity": {
                "planner_name": diff.planner_name,
                "planner_version": diff.planner_version,
                "agent_name": diff.agent_name,
                "agent_version": diff.agent_version,
                "prompt_contract_hash": diff.prompt_contract_hash,
                "provider": diff.provider,
                "requested_model": diff.requested_model,
                "executor_name": diff.executor_name,
                "executor_version": diff.executor_version,
                "persistence_name": diff.persistence_name,
                "persistence_version": diff.persistence_version,
                "entity_resolution_policy_name": diff.entity_resolution_policy_name,
                "entity_resolution_policy_version": diff.entity_resolution_policy_version,
            },
            "comparison_quality": diff.comparison_quality,
            "counts": {
                "unchanged": diff.unchanged_count,
                "modified": diff.modified_count,
                "added": diff.added_count,
                "removed": diff.removed_count,
            },
            "items": [_serialize_item(item) for item in diff.items],
        }
    )


def _build_block_change_maps(
    block_diff: document_revision_diff_service.DocumentRevisionBlockDiff,
) -> tuple[dict[uuid.UUID, str], dict[uuid.UUID, str]]:
    base_change_by_block_id: dict[uuid.UUID, str] = {}
    target_change_by_block_id: dict[uuid.UUID, str] = {}
    for item in block_diff.items:
        if item.base_block is not None:
            if item.base_block.block_id in base_change_by_block_id:
                raise DocumentRevisionFactDiffInvariantError(
                    "document_revision_fact_diff_block_mapping_duplicate"
                )
            base_change_by_block_id[item.base_block.block_id] = item.change_kind
        if item.target_block is not None:
            if item.target_block.block_id in target_change_by_block_id:
                raise DocumentRevisionFactDiffInvariantError(
                    "document_revision_fact_diff_block_mapping_duplicate"
                )
            target_change_by_block_id[item.target_block.block_id] = item.change_kind
    return base_change_by_block_id, target_change_by_block_id


def _assert_agent_identity_matches(
    *,
    base_state: duplicate_grouping_service.duplicate_grouping_repository.DuplicateGroupingOrchestrationState,
    target_state: duplicate_grouping_service.duplicate_grouping_repository.DuplicateGroupingOrchestrationState,
) -> None:
    if (
        base_state.planner_name != target_state.planner_name
        or base_state.planner_version != target_state.planner_version
        or base_state.agent_name != target_state.agent_name
        or base_state.agent_version != target_state.agent_version
        or base_state.prompt_contract_hash != target_state.prompt_contract_hash
        or base_state.provider != target_state.provider
        or base_state.requested_model != target_state.requested_model
        or base_state.executor_name != target_state.executor_name
        or base_state.executor_version != target_state.executor_version
        or base_state.persistence_name != target_state.persistence_name
        or base_state.persistence_version != target_state.persistence_version
        or base_state.entity_resolution_policy_name
        != target_state.entity_resolution_policy_name
        or base_state.entity_resolution_policy_version
        != target_state.entity_resolution_policy_version
    ):
        raise DocumentRevisionFactDiffStateError(
            "document_revision_fact_diff_agent_identity_mismatch"
        )


def _rows_by_fact_value_id(
    rows: Sequence[DocumentRevisionFactDiffSourceRow],
) -> dict[uuid.UUID, list[DocumentRevisionFactDiffSourceRow]]:
    grouped: dict[uuid.UUID, list[DocumentRevisionFactDiffSourceRow]] = {}
    for row in rows:
        grouped.setdefault(row.fact_value_id, []).append(row)
    return grouped


def build_authenticated_fact_source_facts(
    *,
    rows: Sequence[DocumentRevisionFactDiffSourceRow],
    source_snapshot: duplicate_grouping_service.AuthenticatedDuplicateGroupingSourceSnapshot,
    expected_run_id: uuid.UUID,
) -> dict[uuid.UUID, AuthenticatedFactSourceFact]:
    candidate_by_fact_value_id = {
        candidate.fact_value_id: candidate for candidate in source_snapshot.candidates
    }
    authoritative_item_by_fact_value_id = _build_authoritative_application_item_map(
        source_snapshot
    )
    row_groups = _rows_by_fact_value_id(rows)
    if (
        source_snapshot.candidate_count != len(row_groups)
        or set(row_groups) != set(candidate_by_fact_value_id)
    ):
        raise DocumentRevisionFactDiffInvariantError(
            "document_revision_fact_diff_fact_value_source_mismatch"
        )

    fact_snapshot_by_id: dict[uuid.UUID, DocumentRevisionFactDiffFactSnapshot] = {}
    value_group_items_by_fact_id: dict[
        uuid.UUID, dict[str, dict[str, object]]
    ] = {}

    for fact_value_id, fact_value_rows in row_groups.items():
        candidate = candidate_by_fact_value_id.get(fact_value_id)
        if candidate is None:
            raise DocumentRevisionFactDiffInvariantError(
                "document_revision_fact_diff_fact_value_source_mismatch"
            )
        stable_row = fact_value_rows[0]
        for row in fact_value_rows[1:]:
            if (
                row.fact_project_id != stable_row.fact_project_id
                or row.fact_id != stable_row.fact_id
                or row.fact_identity_hash != stable_row.fact_identity_hash
                or row.subject_kind != stable_row.subject_kind
                or row.subject_key != stable_row.subject_key
                or row.predicate_key != stable_row.predicate_key
                or row.scope_key != stable_row.scope_key
                or row.subject_entity_id != stable_row.subject_entity_id
                or row.extraction_run_id != stable_row.extraction_run_id
                or row.inference_run_id != stable_row.inference_run_id
                or row.source_batch_id != stable_row.source_batch_id
                or row.application_id != stable_row.application_id
                or row.application_project_id != stable_row.application_project_id
                or row.application_extraction_run_id
                != stable_row.application_extraction_run_id
                or row.application_inference_run_id
                != stable_row.application_inference_run_id
                or row.orchestration_project_id != stable_row.orchestration_project_id
                or row.orchestration_extraction_run_id
                != stable_row.orchestration_extraction_run_id
                or row.batch_current_inference_run_id
                != stable_row.batch_current_inference_run_id
                or row.value_type != stable_row.value_type
                or row.value_json != stable_row.value_json
                or row.language_code != stable_row.language_code
                or row.confidence != stable_row.confidence
                or row.referenced_entity_id != stable_row.referenced_entity_id
            ):
                raise DocumentRevisionFactDiffInvariantError(
                    "document_revision_fact_diff_fact_value_row_mismatch"
                )

        if stable_row.fact_id != candidate.fact_id:
            raise DocumentRevisionFactDiffInvariantError(
                "document_revision_fact_diff_fact_value_source_mismatch"
            )
        authoritative_entry = authoritative_item_by_fact_value_id.get(fact_value_id)
        if authoritative_entry is None:
            raise DocumentRevisionFactDiffInvariantError(
                "document_revision_fact_diff_fact_value_source_mismatch"
            )
        application_order, application_snapshot, authoritative_item = authoritative_entry
        if stable_row.fact_project_id != source_snapshot.state.project_id:
            raise DocumentRevisionFactDiffInvariantError(
                "document_revision_fact_diff_fact_project_mismatch"
            )
        if stable_row.application_id != application_snapshot.application_id:
            raise DocumentRevisionFactDiffInvariantError(
                "document_revision_fact_diff_fact_value_source_mismatch"
            )
        if stable_row.application_project_id != source_snapshot.state.project_id:
            raise DocumentRevisionFactDiffInvariantError(
                "document_revision_fact_diff_fact_value_source_mismatch"
            )
        if stable_row.orchestration_project_id != source_snapshot.state.project_id:
            raise DocumentRevisionFactDiffInvariantError(
                "document_revision_fact_diff_fact_value_source_mismatch"
            )
        if stable_row.extraction_run_id != expected_run_id:
            raise DocumentRevisionFactDiffInvariantError(
                "document_revision_fact_diff_fact_value_source_mismatch"
            )
        if stable_row.extraction_run_id != source_snapshot.state.extraction_run_id:
            raise DocumentRevisionFactDiffInvariantError(
                "document_revision_fact_diff_fact_value_source_mismatch"
            )
        if stable_row.application_extraction_run_id != expected_run_id:
            raise DocumentRevisionFactDiffInvariantError(
                "document_revision_fact_diff_fact_value_source_mismatch"
            )
        if stable_row.orchestration_extraction_run_id != expected_run_id:
            raise DocumentRevisionFactDiffInvariantError(
                "document_revision_fact_diff_fact_value_source_mismatch"
            )
        if stable_row.application_inference_run_id != stable_row.inference_run_id:
            raise DocumentRevisionFactDiffInvariantError(
                "document_revision_fact_diff_fact_value_source_mismatch"
            )
        if stable_row.batch_current_inference_run_id != stable_row.inference_run_id:
            raise DocumentRevisionFactDiffInvariantError(
                "document_revision_fact_diff_fact_value_source_mismatch"
            )
        if stable_row.source_batch_id != candidate.source_batch_id:
            raise DocumentRevisionFactDiffInvariantError(
                "document_revision_fact_diff_fact_value_source_mismatch"
            )
        if stable_row.fact_id != authoritative_item.fact_id:
            raise DocumentRevisionFactDiffInvariantError(
                "document_revision_fact_diff_fact_value_source_mismatch"
            )
        if stable_row.subject_entity_id != authoritative_item.subject_entity_id:
            raise DocumentRevisionFactDiffInvariantError(
                "document_revision_fact_diff_fact_value_source_mismatch"
            )
        if stable_row.referenced_entity_id != authoritative_item.referenced_entity_id:
            raise DocumentRevisionFactDiffInvariantError(
                "document_revision_fact_diff_fact_value_source_mismatch"
            )
        certified_value = _certify_fact_value(stable_row)
        if certified_value.value_type != candidate.value_type:
            raise DocumentRevisionFactDiffInvariantError(
                "document_revision_fact_diff_fact_value_source_mismatch"
            )
        if certified_value.value_json != candidate.value_json:
            raise DocumentRevisionFactDiffInvariantError(
                "document_revision_fact_diff_fact_value_source_mismatch"
            )
        if certified_value.referenced_entity_id != candidate.referenced_entity_id:
            raise DocumentRevisionFactDiffInvariantError(
                "document_revision_fact_diff_fact_value_source_mismatch"
            )
        fact_snapshot = _certify_fact_identity(stable_row)
        existing_fact_snapshot = fact_snapshot_by_id.get(stable_row.fact_id)
        if existing_fact_snapshot is None:
            fact_snapshot_by_id[stable_row.fact_id] = fact_snapshot
        elif existing_fact_snapshot != fact_snapshot:
            raise DocumentRevisionFactDiffInvariantError(
                "document_revision_fact_diff_fact_identity_mismatch"
            )

        evidence_link_ids_seen: set[uuid.UUID] = set()
        evidence_link_ids_in_order: list[uuid.UUID] = []
        evidence_envelopes: list[_AuthenticatedEvidenceEnvelope] = []
        for row in fact_value_rows:
            if (
                row.evidence_link_id is None
                or row.evidence_id is None
                or row.evidence_role is None
                or row.evidence_is_primary is None
                or row.document_block_id is None
                or row.document_revision_id is None
                or row.evidence_link_source_order is None
                or row.evidence_start_offset is None
                or row.evidence_end_offset is None
                or row.evidence_excerpt is None
                or row.evidence_excerpt_hash is None
                or row.block_extraction_run_id is None
                or row.block_source_order is None
                or row.block_location_key is None
                or row.block_raw_text is None
            ):
                raise DocumentRevisionFactDiffInvariantError(
                    "document_revision_fact_diff_evidence_missing"
                )
            if not isinstance(row.evidence_is_primary, bool):
                raise DocumentRevisionFactDiffInvariantError(
                    "document_revision_fact_diff_evidence_missing"
                )
            if row.block_extraction_run_id != expected_run_id:
                raise DocumentRevisionFactDiffInvariantError(
                    "document_revision_fact_diff_evidence_run_mismatch"
                )
            excerpt = row.block_raw_text[row.evidence_start_offset : row.evidence_end_offset]
            if excerpt != row.evidence_excerpt:
                raise DocumentRevisionFactDiffInvariantError(
                    "document_revision_fact_diff_evidence_excerpt_mismatch"
                )
            if row.evidence_excerpt_hash != _sha256_text(row.evidence_excerpt):
                raise DocumentRevisionFactDiffInvariantError(
                    "document_revision_fact_diff_evidence_excerpt_hash_mismatch"
                )
            if row.evidence_link_id in evidence_link_ids_seen:
                raise DocumentRevisionFactDiffInvariantError(
                    "document_revision_fact_diff_evidence_duplicate"
                )
            evidence_link_ids_seen.add(row.evidence_link_id)
            evidence_link_ids_in_order.append(row.evidence_link_id)
            evidence_envelopes.append(
                _AuthenticatedEvidenceEnvelope(
                    snapshot=AuthenticatedFactSourceEvidence(
                        evidence_link_id=row.evidence_link_id,
                        evidence_id=row.evidence_id,
                        document_revision_id=row.document_revision_id,
                        document_block_id=row.document_block_id,
                        locator=DocumentRevisionFactDiffEvidenceLocator(
                            location_key=row.block_location_key,
                            page_no=row.block_page_no,
                            start_line=row.block_start_line,
                            end_line=row.block_end_line,
                            table_index=row.block_table_index,
                            row_index=row.block_row_index,
                        ),
                        excerpt=row.evidence_excerpt,
                        excerpt_hash=row.evidence_excerpt_hash,
                        content_hash=_sha256_text(row.block_raw_text),
                        role=row.evidence_role,
                        is_primary=row.evidence_is_primary,
                        source_order=row.evidence_link_source_order,
                        block_source_order=row.block_source_order,
                    ),
                    block_source_order=row.block_source_order,
                    evidence_link_source_order=row.evidence_link_source_order,
                )
            )
        if tuple(evidence_link_ids_in_order) != candidate.evidence_link_ids:
            raise DocumentRevisionFactDiffInvariantError(
                "document_revision_fact_diff_evidence_link_set_mismatch"
            )
        if tuple(item.snapshot.evidence_id for item in evidence_envelopes) != authoritative_item.evidence_ids:
            raise DocumentRevisionFactDiffInvariantError(
                "document_revision_fact_diff_evidence_link_set_mismatch"
            )

        semantic_key_hash = _build_semantic_key_hash(
            fact_id=stable_row.fact_id,
            value_type=certified_value.value_type,
            value_json=certified_value.value_json,
            referenced_entity_id=certified_value.referenced_entity_id,
        )
        groups_for_fact = value_group_items_by_fact_id.setdefault(stable_row.fact_id, {})
        group_entry = groups_for_fact.get(semantic_key_hash)
        if group_entry is None:
            group_entry = {
                "value_type": certified_value.value_type,
                "value_json": certified_value.value_json,
                "referenced_entity_id": certified_value.referenced_entity_id,
                "values": [],
                "evidences": [],
            }
            groups_for_fact[semantic_key_hash] = group_entry
        elif (
            group_entry["value_type"] != certified_value.value_type
            or group_entry["value_json"] != certified_value.value_json
            or group_entry["referenced_entity_id"] != certified_value.referenced_entity_id
        ):
            raise DocumentRevisionFactDiffInvariantError(
                "document_revision_fact_diff_semantic_group_mismatch"
            )

        ordered_value_evidences = tuple(
            item.snapshot
            for item in sorted(
                evidence_envelopes,
                key=lambda item: (
                    item.block_source_order,
                    item.evidence_link_source_order,
                    str(item.snapshot.evidence_link_id),
                ),
            )
        )
        group_entry["values"].append(
            _AuthenticatedValueEnvelope(
                snapshot=AuthenticatedFactSourceValue(
                    fact_value_id=fact_value_id,
                    source_batch_id=stable_row.source_batch_id,
                    source_application_id=application_snapshot.application_id,
                    proposal_index=authoritative_item.proposal_index,
                    normalized_value_text=certified_value.normalized_value_text,
                    value_hash=certified_value.value_hash,
                    language_code=stable_row.language_code,
                    confidence=stable_row.confidence,
                    evidences=ordered_value_evidences,
                ),
                application_order=application_order,
                proposal_index=authoritative_item.proposal_index,
            )
        )
        group_entry["evidences"].extend(evidence_envelopes)

    fact_side_snapshots: dict[uuid.UUID, AuthenticatedFactSourceFact] = {}
    for fact_id, groups in value_group_items_by_fact_id.items():
        value_groups: list[AuthenticatedFactSourceValueGroup] = []
        for semantic_key_hash in sorted(groups):
            group_entry = groups[semantic_key_hash]
            value_items = sorted(
                group_entry["values"],
                key=lambda item: (
                    item.application_order,
                    item.proposal_index,
                    str(item.snapshot.fact_value_id),
                ),
            )
            evidence_items = sorted(
                group_entry["evidences"],
                key=lambda item: (
                    item.block_source_order,
                    item.evidence_link_source_order,
                    str(item.snapshot.evidence_link_id),
                ),
            )
            if not evidence_items:
                raise DocumentRevisionFactDiffInvariantError(
                    "document_revision_fact_diff_evidence_missing"
                )
            seen_link_ids: set[uuid.UUID] = set()
            for evidence_item in evidence_items:
                if evidence_item.snapshot.evidence_link_id in seen_link_ids:
                    raise DocumentRevisionFactDiffInvariantError(
                        "document_revision_fact_diff_evidence_duplicate"
                    )
                seen_link_ids.add(evidence_item.snapshot.evidence_link_id)
            value_groups.append(
                AuthenticatedFactSourceValueGroup(
                    semantic_key_hash=semantic_key_hash,
                    value_type=group_entry["value_type"],
                    value_json=group_entry["value_json"],
                    referenced_entity_id=group_entry["referenced_entity_id"],
                    fact_value_ids=tuple(item.snapshot.fact_value_id for item in value_items),
                    values=tuple(item.snapshot for item in value_items),
                    evidences=tuple(item.snapshot for item in evidence_items),
                    earliest_block_source_order=evidence_items[0].block_source_order,
                )
            )
        earliest_block_source_order = min(
            group.earliest_block_source_order for group in value_groups
        )
        fact_side_snapshots[fact_id] = AuthenticatedFactSourceFact(
            fact=fact_snapshot_by_id[fact_id],
            value_groups=tuple(value_groups),
            semantic_key_hashes=frozenset(groups),
            earliest_block_source_order=earliest_block_source_order,
        )
    return fact_side_snapshots


def _build_fact_side_snapshots(
    *,
    rows: Sequence[DocumentRevisionFactDiffSourceRow],
    source_snapshot: duplicate_grouping_service.AuthenticatedDuplicateGroupingSourceSnapshot,
    block_change_by_block_id: dict[uuid.UUID, str],
    expected_run_id: uuid.UUID,
) -> dict[uuid.UUID, _FactSideSnapshot]:
    authenticated_facts = build_authenticated_fact_source_facts(
        rows=rows,
        source_snapshot=source_snapshot,
        expected_run_id=expected_run_id,
    )
    fact_side_snapshots: dict[uuid.UUID, _FactSideSnapshot] = {}
    for fact_id, authenticated_fact in authenticated_facts.items():
        value_group_envelopes: list[_ValueGroupEnvelope] = []
        for value_group in authenticated_fact.value_groups:
            diff_evidences: list[_EvidenceSnapshotEnvelope] = []
            for evidence in value_group.evidences:
                block_change_kind = block_change_by_block_id.get(evidence.document_block_id)
                if block_change_kind is None:
                    raise DocumentRevisionFactDiffInvariantError(
                        "document_revision_fact_diff_block_mapping_missing"
                    )
                diff_evidences.append(
                    _EvidenceSnapshotEnvelope(
                        snapshot=DocumentRevisionFactDiffEvidenceSnapshot(
                            evidence_link_id=evidence.evidence_link_id,
                            evidence_id=evidence.evidence_id,
                            document_block_id=evidence.document_block_id,
                            locator=evidence.locator,
                            excerpt=evidence.excerpt,
                            excerpt_hash=evidence.excerpt_hash,
                            block_change_kind=block_change_kind,
                        ),
                        block_source_order=evidence.block_source_order,
                        evidence_link_source_order=evidence.source_order,
                    )
                )
            value_group_envelopes.append(
                _ValueGroupEnvelope(
                    snapshot=DocumentRevisionFactDiffValueGroup(
                        semantic_key_hash=value_group.semantic_key_hash,
                        value_type=value_group.value_type,
                        value_json=value_group.value_json,
                        referenced_entity_id=value_group.referenced_entity_id,
                        fact_value_ids=tuple(sorted(value_group.fact_value_ids, key=str)),
                        evidences=tuple(item.snapshot for item in diff_evidences),
                    ),
                    earliest_block_source_order=value_group.earliest_block_source_order,
                )
            )
        fact_side_snapshots[fact_id] = _FactSideSnapshot(
            fact=authenticated_fact.fact,
            value_groups=tuple(value_group_envelopes),
            semantic_key_hashes=authenticated_fact.semantic_key_hashes,
            earliest_block_source_order=authenticated_fact.earliest_block_source_order,
        )
    return fact_side_snapshots


def _item_sort_key(
    *,
    item: DocumentRevisionFactDiffItem,
    earliest_block_source_order: int,
    use_target: bool,
) -> tuple[int, str, str]:
    fact = item.target_fact if use_target else item.base_fact
    if fact is None:
        raise DocumentRevisionFactDiffInvariantError(
            "document_revision_fact_diff_sort_key_missing"
        )
    return (
        earliest_block_source_order,
        fact.identity_hash,
        str(fact.fact_id),
    )


async def get_document_revision_fact_diff(
    session_factory: Callable[[], AsyncSession],
    *,
    project_id: uuid.UUID,
    document_id: uuid.UUID,
    base_revision_id: uuid.UUID,
    target_revision_id: uuid.UUID,
    base_extraction_run_id: uuid.UUID,
    target_extraction_run_id: uuid.UUID,
    base_orchestration_id: uuid.UUID,
    target_orchestration_id: uuid.UUID,
) -> DocumentRevisionFactDiff:
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
    base_orchestration_id = _require_uuid(
        base_orchestration_id,
        field_name="base_orchestration_id",
    )
    target_orchestration_id = _require_uuid(
        target_orchestration_id,
        field_name="target_orchestration_id",
    )

    block_diff = await document_revision_diff_service.get_document_revision_block_diff(
        session_factory,
        project_id=project_id,
        document_id=document_id,
        base_revision_id=base_revision_id,
        target_revision_id=target_revision_id,
        base_extraction_run_id=base_extraction_run_id,
        target_extraction_run_id=target_extraction_run_id,
    )
    base_source_snapshot = (
        await duplicate_grouping_service.authenticate_duplicate_grouping_source_snapshot(
            session_factory,
            orchestration_id=base_orchestration_id,
        )
    )
    target_source_snapshot = (
        await duplicate_grouping_service.authenticate_duplicate_grouping_source_snapshot(
            session_factory,
            orchestration_id=target_orchestration_id,
        )
    )
    if (
        base_source_snapshot.state.project_id != project_id
        or target_source_snapshot.state.project_id != project_id
    ):
        raise DocumentRevisionFactDiffStateError(
            "document_revision_fact_diff_orchestration_project_mismatch"
        )
    if base_source_snapshot.state.extraction_run_id != base_extraction_run_id:
        raise DocumentRevisionFactDiffStateError(
            "document_revision_fact_diff_base_orchestration_run_mismatch"
        )
    if target_source_snapshot.state.extraction_run_id != target_extraction_run_id:
        raise DocumentRevisionFactDiffStateError(
            "document_revision_fact_diff_target_orchestration_run_mismatch"
        )
    if (
        block_diff.base_extraction_run_id != base_extraction_run_id
        or block_diff.target_extraction_run_id != target_extraction_run_id
    ):
        raise DocumentRevisionFactDiffInvariantError(
            "document_revision_fact_diff_block_diff_run_mismatch"
        )
    _assert_agent_identity_matches(
        base_state=base_source_snapshot.state,
        target_state=target_source_snapshot.state,
    )
    base_block_change_by_id, target_block_change_by_id = _build_block_change_maps(block_diff)

    async with session_factory() as read_session:
        try:
            base_rows = (
                await document_revision_fact_diff_repository.list_document_revision_fact_diff_source_rows(
                    read_session,
                    orchestration_id=base_orchestration_id,
                )
            )
            target_rows = (
                await document_revision_fact_diff_repository.list_document_revision_fact_diff_source_rows(
                    read_session,
                    orchestration_id=target_orchestration_id,
                )
            )
        except BaseException:
            await read_session.rollback()
            raise
        else:
            await read_session.rollback()

    base_facts = _build_fact_side_snapshots(
        rows=base_rows,
        source_snapshot=base_source_snapshot,
        block_change_by_block_id=base_block_change_by_id,
        expected_run_id=base_extraction_run_id,
    )
    target_facts = _build_fact_side_snapshots(
        rows=target_rows,
        source_snapshot=target_source_snapshot,
        block_change_by_block_id=target_block_change_by_id,
        expected_run_id=target_extraction_run_id,
    )

    target_items_with_keys: list[tuple[tuple[int, str, str], DocumentRevisionFactDiffItem]] = []
    removed_items_with_keys: list[tuple[tuple[int, str, str], DocumentRevisionFactDiffItem]] = []
    for fact_id in sorted(set(base_facts) | set(target_facts), key=str):
        base_fact = base_facts.get(fact_id)
        target_fact = target_facts.get(fact_id)
        if base_fact is None:
            item = DocumentRevisionFactDiffItem(
                change_kind="added",
                base_fact=None,
                target_fact=target_fact.fact,
                base_value_groups=(),
                target_value_groups=tuple(group.snapshot for group in target_fact.value_groups),
            )
            target_items_with_keys.append(
                (
                    _item_sort_key(
                        item=item,
                        earliest_block_source_order=target_fact.earliest_block_source_order,
                        use_target=True,
                    ),
                    item,
                )
            )
            continue
        if target_fact is None:
            item = DocumentRevisionFactDiffItem(
                change_kind="removed",
                base_fact=base_fact.fact,
                target_fact=None,
                base_value_groups=tuple(group.snapshot for group in base_fact.value_groups),
                target_value_groups=(),
            )
            removed_items_with_keys.append(
                (
                    _item_sort_key(
                        item=item,
                        earliest_block_source_order=base_fact.earliest_block_source_order,
                        use_target=False,
                    ),
                    item,
                )
            )
            continue
        change_kind: DocumentRevisionFactDiffChangeKind = (
            "unchanged"
            if base_fact.semantic_key_hashes == target_fact.semantic_key_hashes
            else "modified"
        )
        item = DocumentRevisionFactDiffItem(
            change_kind=change_kind,
            base_fact=base_fact.fact,
            target_fact=target_fact.fact,
            base_value_groups=tuple(group.snapshot for group in base_fact.value_groups),
            target_value_groups=tuple(group.snapshot for group in target_fact.value_groups),
        )
        target_items_with_keys.append(
            (
                _item_sort_key(
                    item=item,
                    earliest_block_source_order=target_fact.earliest_block_source_order,
                    use_target=True,
                ),
                item,
            )
        )

    items = tuple(
        [item for _key, item in sorted(target_items_with_keys, key=lambda item: item[0])]
        + [item for _key, item in sorted(removed_items_with_keys, key=lambda item: item[0])]
    )
    unchanged_count = sum(1 for item in items if item.change_kind == "unchanged")
    modified_count = sum(1 for item in items if item.change_kind == "modified")
    added_count = sum(1 for item in items if item.change_kind == "added")
    removed_count = sum(1 for item in items if item.change_kind == "removed")
    comparison_quality = (
        "partial"
        if block_diff.comparison_quality == "partial"
        or base_source_snapshot.state.orchestration_status == "partial"
        or target_source_snapshot.state.orchestration_status == "partial"
        else "complete"
    )
    diff = DocumentRevisionFactDiff(
        project_id=project_id,
        document_id=document_id,
        base_revision_id=base_revision_id,
        target_revision_id=target_revision_id,
        base_extraction_run_id=base_extraction_run_id,
        target_extraction_run_id=target_extraction_run_id,
        base_orchestration_id=base_orchestration_id,
        target_orchestration_id=target_orchestration_id,
        base_revision_no=block_diff.base_revision_no,
        target_revision_no=block_diff.target_revision_no,
        base_orchestration_status=base_source_snapshot.state.orchestration_status,
        target_orchestration_status=target_source_snapshot.state.orchestration_status,
        block_diff_manifest_hash=block_diff.diff_manifest_hash,
        fact_diff_algorithm_name=DOCUMENT_REVISION_FACT_DIFF_ALGORITHM_NAME,
        fact_diff_algorithm_version=DOCUMENT_REVISION_FACT_DIFF_ALGORITHM_VERSION,
        semantic_fingerprint_algorithm_version=CROSS_BATCH_DUPLICATE_ALGORITHM_VERSION,
        planner_name=base_source_snapshot.state.planner_name,
        planner_version=base_source_snapshot.state.planner_version,
        agent_name=base_source_snapshot.state.agent_name,
        agent_version=base_source_snapshot.state.agent_version,
        prompt_contract_hash=base_source_snapshot.state.prompt_contract_hash,
        provider=base_source_snapshot.state.provider,
        requested_model=base_source_snapshot.state.requested_model,
        executor_name=base_source_snapshot.state.executor_name,
        executor_version=base_source_snapshot.state.executor_version,
        persistence_name=base_source_snapshot.state.persistence_name,
        persistence_version=base_source_snapshot.state.persistence_version,
        entity_resolution_policy_name=base_source_snapshot.state.entity_resolution_policy_name,
        entity_resolution_policy_version=base_source_snapshot.state.entity_resolution_policy_version,
        comparison_quality=comparison_quality,
        unchanged_count=unchanged_count,
        modified_count=modified_count,
        added_count=added_count,
        removed_count=removed_count,
        items=items,
        fact_diff_manifest_hash="",
    )
    fact_diff_manifest_hash = _build_manifest_hash(diff=diff)
    return DocumentRevisionFactDiff(
        project_id=diff.project_id,
        document_id=diff.document_id,
        base_revision_id=diff.base_revision_id,
        target_revision_id=diff.target_revision_id,
        base_extraction_run_id=diff.base_extraction_run_id,
        target_extraction_run_id=diff.target_extraction_run_id,
        base_orchestration_id=diff.base_orchestration_id,
        target_orchestration_id=diff.target_orchestration_id,
        base_revision_no=diff.base_revision_no,
        target_revision_no=diff.target_revision_no,
        base_orchestration_status=diff.base_orchestration_status,
        target_orchestration_status=diff.target_orchestration_status,
        block_diff_manifest_hash=diff.block_diff_manifest_hash,
        fact_diff_algorithm_name=diff.fact_diff_algorithm_name,
        fact_diff_algorithm_version=diff.fact_diff_algorithm_version,
        semantic_fingerprint_algorithm_version=diff.semantic_fingerprint_algorithm_version,
        planner_name=diff.planner_name,
        planner_version=diff.planner_version,
        agent_name=diff.agent_name,
        agent_version=diff.agent_version,
        prompt_contract_hash=diff.prompt_contract_hash,
        provider=diff.provider,
        requested_model=diff.requested_model,
        executor_name=diff.executor_name,
        executor_version=diff.executor_version,
        persistence_name=diff.persistence_name,
        persistence_version=diff.persistence_version,
        entity_resolution_policy_name=diff.entity_resolution_policy_name,
        entity_resolution_policy_version=diff.entity_resolution_policy_version,
        comparison_quality=diff.comparison_quality,
        unchanged_count=diff.unchanged_count,
        modified_count=diff.modified_count,
        added_count=diff.added_count,
        removed_count=diff.removed_count,
        items=diff.items,
        fact_diff_manifest_hash=fact_diff_manifest_hash,
    )
