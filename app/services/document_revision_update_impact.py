from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.schemas.document_revision_fact_diff import (
    DocumentRevisionFactDiff,
    DocumentRevisionFactDiffFactSnapshot,
    DocumentRevisionFactDiffItem,
    DocumentRevisionFactDiffValueGroup,
)
from app.schemas.document_revision_update_impact import (
    DocumentRevisionUpdateImpact,
    DocumentRevisionUpdateImpactItem,
    DocumentRevisionUpdateImpactKind,
)
from app.services import consistency_check_persistence as consistency_check_persistence_service
from app.services import document_revision_fact_diff as fact_diff_service
from app.services import effective_fact_value as effective_fact_value_service
from app.services import fact_value_duplicate_grouping as duplicate_grouping_service


DOCUMENT_REVISION_UPDATE_IMPACT_ALGORITHM_NAME = "document_revision_update_impact"
DOCUMENT_REVISION_UPDATE_IMPACT_ALGORITHM_VERSION = "1.0.0"


class DocumentRevisionUpdateImpactError(Exception):
    """Base class for revision update impact failures."""


class DocumentRevisionUpdateImpactStateError(DocumentRevisionUpdateImpactError):
    """Raised when requested source identifiers are not admissible."""


class DocumentRevisionUpdateImpactInvariantError(DocumentRevisionUpdateImpactError):
    """Raised when authenticated read-only projections diverge."""


@dataclass(frozen=True, slots=True)
class _BaseEffectiveContext:
    assessment_id: uuid.UUID
    review_status: str
    resolution_status: str
    resolution_basis: str
    current_decision_id: uuid.UUID | None
    current_decision_kind: str | None
    effective_fact_value_ids: tuple[uuid.UUID, ...]


def _require_uuid(value: object, *, field_name: str) -> uuid.UUID:
    if not isinstance(value, uuid.UUID):
        raise DocumentRevisionUpdateImpactStateError(
            f"document_revision_update_impact_{field_name}_invalid"
        )
    return value


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


def _serialize_value_group(
    value_group: DocumentRevisionFactDiffValueGroup,
) -> dict[str, object]:
    return {
        "semantic_key_hash": value_group.semantic_key_hash,
        "value_type": value_group.value_type,
        "value_json": value_group.value_json,
        "referenced_entity_id": (
            str(value_group.referenced_entity_id)
            if value_group.referenced_entity_id is not None
            else None
        ),
        "fact_value_ids": [str(item) for item in value_group.fact_value_ids],
        "evidences": [
            {
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
            for evidence in value_group.evidences
        ],
    }


def _serialize_item(item: DocumentRevisionUpdateImpactItem) -> dict[str, object]:
    return {
        "fact_id": str(item.fact_id),
        "fact_change_kind": item.fact_change_kind,
        "impact_kind": item.impact_kind,
        "requires_review": item.requires_review,
        "base_assessment_id": (
            str(item.base_assessment_id) if item.base_assessment_id is not None else None
        ),
        "base_review_status": item.base_review_status,
        "base_resolution_status": item.base_resolution_status,
        "base_resolution_basis": item.base_resolution_basis,
        "base_current_decision_id": (
            str(item.base_current_decision_id)
            if item.base_current_decision_id is not None
            else None
        ),
        "base_current_decision_kind": item.base_current_decision_kind,
        "base_effective_fact_value_ids": [
            str(fact_value_id) for fact_value_id in item.base_effective_fact_value_ids
        ],
        "base_fact": _serialize_fact_snapshot(item.base_fact),
        "base_value_groups": [
            _serialize_value_group(group) for group in item.base_value_groups
        ],
        "target_fact": _serialize_fact_snapshot(item.target_fact),
        "target_value_groups": [
            _serialize_value_group(group) for group in item.target_value_groups
        ],
    }


def _build_manifest_hash(
    *,
    impact: DocumentRevisionUpdateImpact,
) -> str:
    return duplicate_grouping_service.hash_deterministic_payload(
        {
            "project_id": str(impact.project_id),
            "document_id": str(impact.document_id),
            "base_revision_id": str(impact.base_revision_id),
            "target_revision_id": str(impact.target_revision_id),
            "base_extraction_run_id": str(impact.base_extraction_run_id),
            "target_extraction_run_id": str(impact.target_extraction_run_id),
            "base_orchestration_id": str(impact.base_orchestration_id),
            "target_orchestration_id": str(impact.target_orchestration_id),
            "base_consistency_check_application_id": str(
                impact.base_consistency_check_application_id
            ),
            "source_manifests": {
                "block_diff_manifest_hash": impact.block_diff_manifest_hash,
                "fact_diff_manifest_hash": impact.fact_diff_manifest_hash,
                "base_consistency_result_manifest_hash": (
                    impact.base_consistency_result_manifest_hash
                ),
            },
            "impact_algorithm": {
                "name": impact.impact_algorithm_name,
                "version": impact.impact_algorithm_version,
            },
            "counts": {
                "fact_count": impact.fact_count,
                "review_required_count": impact.review_required_count,
                "unchanged_resolved_count": impact.unchanged_resolved_count,
                "unchanged_no_review_context_count": (
                    impact.unchanged_no_review_context_count
                ),
                "unchanged_unresolved_count": impact.unchanged_unresolved_count,
                "modified_count": impact.modified_count,
                "added_count": impact.added_count,
                "removed_count": impact.removed_count,
            },
            "items": [_serialize_item(item) for item in impact.items],
        }
    )


def _build_effective_context_map(
    *,
    fact_diff: DocumentRevisionFactDiff,
    effective_projection: effective_fact_value_service.EffectiveFactValueProjection,
    expected_project_id: uuid.UUID,
    expected_application_id: uuid.UUID,
    expected_result_manifest_hash: str,
) -> dict[uuid.UUID, _BaseEffectiveContext]:
    if effective_projection.project_id != expected_project_id:
        raise DocumentRevisionUpdateImpactInvariantError(
            "document_revision_update_impact_effective_projection_source_mismatch"
        )
    if effective_projection.consistency_check_application_id != expected_application_id:
        raise DocumentRevisionUpdateImpactInvariantError(
            "document_revision_update_impact_effective_projection_source_mismatch"
        )
    if effective_projection.result_manifest_hash != expected_result_manifest_hash:
        raise DocumentRevisionUpdateImpactInvariantError(
            "document_revision_update_impact_effective_projection_source_mismatch"
        )

    base_fact_ids = {
        item.base_fact.fact_id for item in fact_diff.items if item.base_fact is not None
    }
    context_by_fact_id: dict[uuid.UUID, _BaseEffectiveContext] = {}
    for item in effective_projection.items:
        if item.fact_id in context_by_fact_id:
            raise DocumentRevisionUpdateImpactInvariantError(
                "document_revision_update_impact_effective_context_duplicate_fact"
            )
        if item.fact_id not in base_fact_ids:
            raise DocumentRevisionUpdateImpactInvariantError(
                "document_revision_update_impact_effective_context_unknown_fact"
            )
        context_by_fact_id[item.fact_id] = _BaseEffectiveContext(
            assessment_id=item.assessment_id,
            review_status=item.review_status,
            resolution_status=item.resolution_status,
            resolution_basis=item.resolution_basis,
            current_decision_id=item.current_decision_id,
            current_decision_kind=item.current_decision_kind,
            effective_fact_value_ids=item.effective_fact_value_ids,
        )
    return context_by_fact_id


def _classify_impact(
    diff_item: DocumentRevisionFactDiffItem,
    *,
    base_effective_context: _BaseEffectiveContext | None,
) -> tuple[DocumentRevisionUpdateImpactKind, bool]:
    if diff_item.change_kind == "unchanged":
        if base_effective_context is None:
            return "unchanged_no_review_context", False
        if base_effective_context.resolution_status == "resolved":
            return "unchanged_resolved", False
        return "unchanged_unresolved", True
    if diff_item.change_kind == "modified":
        return "modified", True
    if diff_item.change_kind == "added":
        return "added", True
    if diff_item.change_kind == "removed":
        return "removed", True
    raise DocumentRevisionUpdateImpactInvariantError(
        "document_revision_update_impact_fact_change_kind_invalid"
    )


def _build_impact_item(
    diff_item: DocumentRevisionFactDiffItem,
    *,
    base_effective_context: _BaseEffectiveContext | None,
) -> DocumentRevisionUpdateImpactItem:
    impact_kind, requires_review = _classify_impact(
        diff_item,
        base_effective_context=base_effective_context,
    )
    fact = diff_item.target_fact or diff_item.base_fact
    if fact is None:
        raise DocumentRevisionUpdateImpactInvariantError(
            "document_revision_update_impact_fact_missing"
        )
    return DocumentRevisionUpdateImpactItem(
        fact_id=fact.fact_id,
        fact_change_kind=diff_item.change_kind,
        impact_kind=impact_kind,
        requires_review=requires_review,
        base_assessment_id=(
            None if base_effective_context is None else base_effective_context.assessment_id
        ),
        base_review_status=(
            None if base_effective_context is None else base_effective_context.review_status
        ),
        base_resolution_status=(
            None
            if base_effective_context is None
            else base_effective_context.resolution_status
        ),
        base_resolution_basis=(
            None
            if base_effective_context is None
            else base_effective_context.resolution_basis
        ),
        base_current_decision_id=(
            None
            if base_effective_context is None
            else base_effective_context.current_decision_id
        ),
        base_current_decision_kind=(
            None
            if base_effective_context is None
            else base_effective_context.current_decision_kind
        ),
        base_effective_fact_value_ids=(
            ()
            if base_effective_context is None
            else base_effective_context.effective_fact_value_ids
        ),
        base_fact=diff_item.base_fact,
        base_value_groups=diff_item.base_value_groups,
        target_fact=diff_item.target_fact,
        target_value_groups=diff_item.target_value_groups,
    )


async def get_document_revision_update_impact(
    session_factory,
    *,
    project_id: uuid.UUID,
    document_id: uuid.UUID,
    base_revision_id: uuid.UUID,
    target_revision_id: uuid.UUID,
    base_extraction_run_id: uuid.UUID,
    target_extraction_run_id: uuid.UUID,
    base_orchestration_id: uuid.UUID,
    target_orchestration_id: uuid.UUID,
    base_consistency_check_application_id: uuid.UUID,
) -> DocumentRevisionUpdateImpact:
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
    base_consistency_check_application_id = _require_uuid(
        base_consistency_check_application_id,
        field_name="base_consistency_check_application_id",
    )

    fact_diff = await fact_diff_service.get_document_revision_fact_diff(
        session_factory,
        project_id=project_id,
        document_id=document_id,
        base_revision_id=base_revision_id,
        target_revision_id=target_revision_id,
        base_extraction_run_id=base_extraction_run_id,
        target_extraction_run_id=target_extraction_run_id,
        base_orchestration_id=base_orchestration_id,
        target_orchestration_id=target_orchestration_id,
    )
    if (
        fact_diff.project_id != project_id
        or fact_diff.document_id != document_id
        or fact_diff.base_revision_id != base_revision_id
        or fact_diff.target_revision_id != target_revision_id
        or fact_diff.base_extraction_run_id != base_extraction_run_id
        or fact_diff.target_extraction_run_id != target_extraction_run_id
        or fact_diff.base_orchestration_id != base_orchestration_id
        or fact_diff.target_orchestration_id != target_orchestration_id
    ):
        raise DocumentRevisionUpdateImpactInvariantError(
            "document_revision_update_impact_fact_diff_source_mismatch"
        )
    authenticated_application = (
        await consistency_check_persistence_service.authenticate_persisted_consistency_check_application(
            session_factory,
            project_id=project_id,
            consistency_check_application_id=base_consistency_check_application_id,
        )
    )
    if authenticated_application.application.project_id != project_id:
        raise DocumentRevisionUpdateImpactStateError(
            "document_revision_update_impact_base_application_project_mismatch"
        )
    if authenticated_application.application.id != base_consistency_check_application_id:
        raise DocumentRevisionUpdateImpactInvariantError(
            "document_revision_update_impact_base_application_source_mismatch"
        )
    source_application = authenticated_application.authenticated_source.application
    source_duplicate_grouping_application = (
        authenticated_application.authenticated_source.source_duplicate_grouping_application
    )
    if source_application.orchestration_id != base_orchestration_id:
        raise DocumentRevisionUpdateImpactStateError(
            "document_revision_update_impact_base_application_orchestration_mismatch"
        )
    if source_application.extraction_run_id != base_extraction_run_id:
        raise DocumentRevisionUpdateImpactStateError(
            "document_revision_update_impact_base_application_extraction_run_mismatch"
        )
    if source_duplicate_grouping_application.orchestration_id != base_orchestration_id:
        raise DocumentRevisionUpdateImpactStateError(
            "document_revision_update_impact_base_application_orchestration_mismatch"
        )
    if source_duplicate_grouping_application.extraction_run_id != base_extraction_run_id:
        raise DocumentRevisionUpdateImpactStateError(
            "document_revision_update_impact_base_application_extraction_run_mismatch"
        )

    effective_projection = await effective_fact_value_service.get_effective_fact_value_projection(
        session_factory,
        project_id=project_id,
        consistency_check_application_id=base_consistency_check_application_id,
    )
    effective_context_by_fact_id = _build_effective_context_map(
        fact_diff=fact_diff,
        effective_projection=effective_projection,
        expected_project_id=project_id,
        expected_application_id=base_consistency_check_application_id,
        expected_result_manifest_hash=authenticated_application.application.result_manifest_hash,
    )

    items = tuple(
        _build_impact_item(
            diff_item,
            base_effective_context=(
                None
                if diff_item.base_fact is None
                else effective_context_by_fact_id.get(diff_item.base_fact.fact_id)
            ),
        )
        for diff_item in fact_diff.items
    )

    impact = DocumentRevisionUpdateImpact(
        project_id=project_id,
        document_id=document_id,
        base_revision_id=base_revision_id,
        target_revision_id=target_revision_id,
        base_extraction_run_id=base_extraction_run_id,
        target_extraction_run_id=target_extraction_run_id,
        base_orchestration_id=base_orchestration_id,
        target_orchestration_id=target_orchestration_id,
        base_consistency_check_application_id=base_consistency_check_application_id,
        block_diff_manifest_hash=fact_diff.block_diff_manifest_hash,
        fact_diff_manifest_hash=fact_diff.fact_diff_manifest_hash,
        base_consistency_result_manifest_hash=(
            authenticated_application.application.result_manifest_hash
        ),
        impact_algorithm_name=DOCUMENT_REVISION_UPDATE_IMPACT_ALGORITHM_NAME,
        impact_algorithm_version=DOCUMENT_REVISION_UPDATE_IMPACT_ALGORITHM_VERSION,
        fact_count=len(items),
        review_required_count=sum(1 for item in items if item.requires_review),
        unchanged_resolved_count=sum(
            1 for item in items if item.impact_kind == "unchanged_resolved"
        ),
        unchanged_no_review_context_count=sum(
            1
            for item in items
            if item.impact_kind == "unchanged_no_review_context"
        ),
        unchanged_unresolved_count=sum(
            1 for item in items if item.impact_kind == "unchanged_unresolved"
        ),
        modified_count=sum(1 for item in items if item.impact_kind == "modified"),
        added_count=sum(1 for item in items if item.impact_kind == "added"),
        removed_count=sum(1 for item in items if item.impact_kind == "removed"),
        items=items,
        impact_manifest_hash="",
    )
    impact_manifest_hash = _build_manifest_hash(impact=impact)
    return DocumentRevisionUpdateImpact(
        project_id=impact.project_id,
        document_id=impact.document_id,
        base_revision_id=impact.base_revision_id,
        target_revision_id=impact.target_revision_id,
        base_extraction_run_id=impact.base_extraction_run_id,
        target_extraction_run_id=impact.target_extraction_run_id,
        base_orchestration_id=impact.base_orchestration_id,
        target_orchestration_id=impact.target_orchestration_id,
        base_consistency_check_application_id=impact.base_consistency_check_application_id,
        block_diff_manifest_hash=impact.block_diff_manifest_hash,
        fact_diff_manifest_hash=impact.fact_diff_manifest_hash,
        base_consistency_result_manifest_hash=impact.base_consistency_result_manifest_hash,
        impact_algorithm_name=impact.impact_algorithm_name,
        impact_algorithm_version=impact.impact_algorithm_version,
        fact_count=impact.fact_count,
        review_required_count=impact.review_required_count,
        unchanged_resolved_count=impact.unchanged_resolved_count,
        unchanged_no_review_context_count=impact.unchanged_no_review_context_count,
        unchanged_unresolved_count=impact.unchanged_unresolved_count,
        modified_count=impact.modified_count,
        added_count=impact.added_count,
        removed_count=impact.removed_count,
        items=impact.items,
        impact_manifest_hash=impact_manifest_hash,
    )
