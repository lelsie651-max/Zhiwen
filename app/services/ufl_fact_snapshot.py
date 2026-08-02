from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories import ufl_fact_snapshot as ufl_fact_snapshot_repository
from app.schemas.ufl_fact_snapshot import (
    OrchestrationUFLFactSnapshot,
    UFLFactEvidenceLocator,
    UFLFactEvidenceSnapshot,
    UFLFactSnapshot,
    UFLFactValueGroupSnapshot,
    UFLFactValueSnapshot,
)
from app.services import document_revision_fact_diff as fact_diff_service
from app.services import fact_value_duplicate_grouping as duplicate_grouping_service
from app.utils.deterministic_json import freeze_deterministic_json_value
from app.utils.fact_value_metadata import (
    validate_fact_value_confidence,
    validate_fact_value_language_code,
)


ORCHESTRATION_UFL_FACT_SNAPSHOT_ALGORITHM_NAME = "orchestration_ufl_fact_snapshot"
ORCHESTRATION_UFL_FACT_SNAPSHOT_ALGORITHM_VERSION = "1.0.0"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class OrchestrationUFLFactSnapshotError(Exception):
    """Base class for orchestration UFL fact snapshot failures."""


class OrchestrationUFLFactSnapshotStateError(OrchestrationUFLFactSnapshotError):
    """Raised when the requested orchestration is not ready for snapshotting."""


class OrchestrationUFLFactSnapshotInvariantError(OrchestrationUFLFactSnapshotError):
    """Raised when immutable fact snapshot source invariants drift."""


def _require_uuid(value: object, *, field_name: str) -> uuid.UUID:
    if not isinstance(value, uuid.UUID):
        raise OrchestrationUFLFactSnapshotStateError(
            f"orchestration_ufl_fact_snapshot_{field_name}_invalid"
        )
    return value


def _require_invariant_uuid(value: object, *, error_code: str) -> uuid.UUID:
    if not isinstance(value, uuid.UUID):
        raise OrchestrationUFLFactSnapshotInvariantError(error_code)
    return value


def _require_sha256(value: object, *, error_code: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise OrchestrationUFLFactSnapshotInvariantError(error_code)
    return value


def _require_nonnegative_int(value: object, *, error_code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OrchestrationUFLFactSnapshotInvariantError(error_code)
    return value


def _freeze_value_json(value: Any) -> object:
    try:
        frozen = freeze_deterministic_json_value(value)
        duplicate_grouping_service.canonicalize_deterministic_payload(frozen)
    except (
        ValueError,
        TypeError,
        duplicate_grouping_service.CrossBatchDuplicateGroupingError,
        duplicate_grouping_service.CrossBatchDuplicateGroupingInvariantError,
    ):
        raise OrchestrationUFLFactSnapshotInvariantError(
            "orchestration_ufl_fact_snapshot_value_json_invalid"
        ) from None
    return frozen


def _validate_language_code(value: object | None) -> str | None:
    try:
        return validate_fact_value_language_code(value)
    except ValueError:
        raise OrchestrationUFLFactSnapshotInvariantError(
            "orchestration_ufl_fact_snapshot_value_metadata_invalid"
        ) from None


def _validate_confidence(value: object | None) -> float | None:
    try:
        return validate_fact_value_confidence(value)
    except ValueError:
        raise OrchestrationUFLFactSnapshotInvariantError(
            "orchestration_ufl_fact_snapshot_value_metadata_invalid"
        ) from None


def _serialize_locator(locator: UFLFactEvidenceLocator) -> dict[str, object]:
    return {
        "location_key": locator.location_key,
        "page_no": locator.page_no,
        "start_line": locator.start_line,
        "end_line": locator.end_line,
        "table_index": locator.table_index,
        "row_index": locator.row_index,
    }


def _serialize_evidence(evidence: UFLFactEvidenceSnapshot) -> dict[str, object]:
    return {
        "evidence_link_id": str(evidence.evidence_link_id),
        "evidence_id": str(evidence.evidence_id),
        "document_revision_id": str(evidence.document_revision_id),
        "document_block_id": str(evidence.document_block_id),
        "locator": _serialize_locator(evidence.locator),
        "excerpt": evidence.excerpt,
        "excerpt_hash": evidence.excerpt_hash,
        "content_hash": evidence.content_hash,
        "role": evidence.role,
        "is_primary": evidence.is_primary,
        "source_order": evidence.source_order,
    }


def _serialize_value(value: UFLFactValueSnapshot) -> dict[str, object]:
    return {
        "fact_value_id": str(value.fact_value_id),
        "source_batch_id": str(value.source_batch_id),
        "source_application_id": str(value.source_application_id),
        "proposal_index": value.proposal_index,
        "normalized_value_text": value.normalized_value_text,
        "value_hash": value.value_hash,
        "language_code": value.language_code,
        "confidence": value.confidence,
    }


def _serialize_value_group(group: UFLFactValueGroupSnapshot) -> dict[str, object]:
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
        "values": [_serialize_value(value) for value in group.values],
        "evidences": [_serialize_evidence(evidence) for evidence in group.evidences],
    }


def _serialize_fact(fact: UFLFactSnapshot) -> dict[str, object]:
    return {
        "fact_id": str(fact.fact_id),
        "identity_hash": fact.identity_hash,
        "subject_kind": fact.subject_kind,
        "subject_key": fact.subject_key,
        "subject_entity_id": (
            str(fact.subject_entity_id) if fact.subject_entity_id is not None else None
        ),
        "predicate_key": fact.predicate_key,
        "scope_key": fact.scope_key,
        "semantic_group_count": fact.semantic_group_count,
        "fact_value_count": fact.fact_value_count,
        "value_groups": [_serialize_value_group(group) for group in fact.value_groups],
    }


def _build_manifest_hash(
    *,
    snapshot: OrchestrationUFLFactSnapshot,
    source_snapshot: duplicate_grouping_service.AuthenticatedDuplicateGroupingSourceSnapshot,
    application_result_hashes: tuple[str, ...],
) -> str:
    return duplicate_grouping_service.hash_deterministic_payload(
        {
            "project_id": str(snapshot.project_id),
            "orchestration_id": str(snapshot.orchestration_id),
            "extraction_run_id": str(snapshot.extraction_run_id),
            "orchestration_status": snapshot.orchestration_status,
            "comparison_quality": snapshot.comparison_quality,
            "applications": [
                {
                    "application_id": str(application.application_id),
                    "result_hash": result_hash,
                }
                for application, result_hash in zip(
                    source_snapshot.application_snapshots,
                    application_result_hashes,
                    strict=True,
                )
            ],
            "source_application_count": snapshot.source_application_count,
            "counts": {
                "fact_count": snapshot.fact_count,
                "fact_value_count": snapshot.fact_value_count,
                "evidence_count": snapshot.evidence_count,
            },
            "algorithm": {
                "name": snapshot.algorithm_name,
                "version": snapshot.algorithm_version,
            },
            "facts": [_serialize_fact(fact) for fact in snapshot.facts],
        }
    )


def _fact_sort_key(
    fact: fact_diff_service.AuthenticatedFactSourceFact,
) -> tuple[str, str, str, int, str, str]:
    scope_key = fact.fact.scope_key
    return (
        fact.fact.subject_kind,
        fact.fact.subject_key,
        fact.fact.predicate_key,
        0 if scope_key is None else 1,
        "" if scope_key is None else scope_key,
        f"{fact.fact.identity_hash}:{fact.fact.fact_id}",
    )


def _build_snapshot(
    *,
    project_id: uuid.UUID,
    source_snapshot: duplicate_grouping_service.AuthenticatedDuplicateGroupingSourceSnapshot,
    authenticated_facts: dict[uuid.UUID, fact_diff_service.AuthenticatedFactSourceFact],
) -> OrchestrationUFLFactSnapshot:
    facts: list[UFLFactSnapshot] = []
    fact_value_count = 0
    evidence_count = 0
    for authenticated_fact in sorted(authenticated_facts.values(), key=_fact_sort_key):
        value_groups: list[UFLFactValueGroupSnapshot] = []
        current_fact_value_count = 0
        for group in authenticated_fact.value_groups:
            semantic_key_hash = _require_sha256(
                group.semantic_key_hash,
                error_code="orchestration_ufl_fact_snapshot_value_metadata_invalid",
            )
            frozen_value_json = _freeze_value_json(group.value_json)
            values = tuple(
                UFLFactValueSnapshot(
                    fact_value_id=_require_invariant_uuid(
                        value.fact_value_id,
                        error_code="orchestration_ufl_fact_snapshot_value_metadata_invalid",
                    ),
                    source_batch_id=_require_invariant_uuid(
                        value.source_batch_id,
                        error_code="orchestration_ufl_fact_snapshot_value_metadata_invalid",
                    ),
                    source_application_id=_require_invariant_uuid(
                        value.source_application_id,
                        error_code="orchestration_ufl_fact_snapshot_value_metadata_invalid",
                    ),
                    proposal_index=_require_nonnegative_int(
                        value.proposal_index,
                        error_code="orchestration_ufl_fact_snapshot_value_metadata_invalid",
                    ),
                    normalized_value_text=value.normalized_value_text,
                    value_hash=_require_sha256(
                        value.value_hash,
                        error_code="orchestration_ufl_fact_snapshot_value_metadata_invalid",
                    ),
                    language_code=_validate_language_code(value.language_code),
                    confidence=_validate_confidence(value.confidence),
                )
                for value in group.values
            )
            evidences = tuple(
                UFLFactEvidenceSnapshot(
                    evidence_link_id=evidence.evidence_link_id,
                    evidence_id=evidence.evidence_id,
                    document_revision_id=evidence.document_revision_id,
                    document_block_id=evidence.document_block_id,
                    locator=UFLFactEvidenceLocator(
                        location_key=evidence.locator.location_key,
                        page_no=evidence.locator.page_no,
                        start_line=evidence.locator.start_line,
                        end_line=evidence.locator.end_line,
                        table_index=evidence.locator.table_index,
                        row_index=evidence.locator.row_index,
                    ),
                    excerpt=evidence.excerpt,
                    excerpt_hash=evidence.excerpt_hash,
                    content_hash=evidence.content_hash,
                    role=evidence.role,
                    is_primary=evidence.is_primary,
                    source_order=evidence.source_order,
                )
                for evidence in group.evidences
            )
            value_groups.append(
                UFLFactValueGroupSnapshot(
                    semantic_key_hash=semantic_key_hash,
                    value_type=group.value_type,
                    value_json=frozen_value_json,
                    referenced_entity_id=group.referenced_entity_id,
                    fact_value_ids=group.fact_value_ids,
                    values=values,
                    evidences=evidences,
                )
            )
            current_fact_value_count += len(group.values)
            evidence_count += len(group.evidences)
        facts.append(
            UFLFactSnapshot(
                fact_id=authenticated_fact.fact.fact_id,
                identity_hash=authenticated_fact.fact.identity_hash,
                subject_kind=authenticated_fact.fact.subject_kind,
                subject_key=authenticated_fact.fact.subject_key,
                subject_entity_id=authenticated_fact.fact.subject_entity_id,
                predicate_key=authenticated_fact.fact.predicate_key,
                scope_key=authenticated_fact.fact.scope_key,
                semantic_group_count=len(value_groups),
                fact_value_count=current_fact_value_count,
                value_groups=tuple(value_groups),
            )
        )
        fact_value_count += current_fact_value_count

    comparison_quality = (
        "partial" if source_snapshot.state.orchestration_status == "partial" else "complete"
    )
    application_result_hashes = tuple(
        _require_sha256(
            application.result_hash,
            error_code="orchestration_ufl_fact_snapshot_application_result_hash_invalid",
        )
        for application in source_snapshot.application_snapshots
    )
    snapshot = OrchestrationUFLFactSnapshot(
        project_id=project_id,
        orchestration_id=source_snapshot.state.orchestration_id,
        extraction_run_id=source_snapshot.state.extraction_run_id,
        orchestration_status=source_snapshot.state.orchestration_status,
        comparison_quality=comparison_quality,
        source_application_count=len(source_snapshot.application_snapshots),
        fact_count=len(facts),
        fact_value_count=fact_value_count,
        evidence_count=evidence_count,
        algorithm_name=ORCHESTRATION_UFL_FACT_SNAPSHOT_ALGORITHM_NAME,
        algorithm_version=ORCHESTRATION_UFL_FACT_SNAPSHOT_ALGORITHM_VERSION,
        facts=tuple(facts),
        source_manifest_hash="",
    )
    return OrchestrationUFLFactSnapshot(
        project_id=snapshot.project_id,
        orchestration_id=snapshot.orchestration_id,
        extraction_run_id=snapshot.extraction_run_id,
        orchestration_status=snapshot.orchestration_status,
        comparison_quality=snapshot.comparison_quality,
        source_application_count=snapshot.source_application_count,
        fact_count=snapshot.fact_count,
        fact_value_count=snapshot.fact_value_count,
        evidence_count=snapshot.evidence_count,
        algorithm_name=snapshot.algorithm_name,
        algorithm_version=snapshot.algorithm_version,
        facts=snapshot.facts,
        source_manifest_hash=_build_manifest_hash(
            snapshot=snapshot,
            source_snapshot=source_snapshot,
            application_result_hashes=application_result_hashes,
        ),
    )


async def get_orchestration_ufl_fact_snapshot(
    session_factory: Callable[[], AsyncSession],
    *,
    project_id: uuid.UUID,
    orchestration_id: uuid.UUID,
) -> OrchestrationUFLFactSnapshot:
    project_id = _require_uuid(project_id, field_name="project_id")
    orchestration_id = _require_uuid(orchestration_id, field_name="orchestration_id")

    try:
        source_snapshot = await duplicate_grouping_service.authenticate_duplicate_grouping_source_snapshot(
            session_factory,
            orchestration_id=orchestration_id,
        )
    except duplicate_grouping_service.CrossBatchDuplicateGroupingStateError:
        raise OrchestrationUFLFactSnapshotStateError(
            "orchestration_ufl_fact_snapshot_orchestration_not_ready"
        ) from None
    except duplicate_grouping_service.CrossBatchDuplicateGroupingInvariantError:
        raise OrchestrationUFLFactSnapshotInvariantError(
            "orchestration_ufl_fact_snapshot_source_mismatch"
        ) from None

    if source_snapshot.state.project_id != project_id:
        raise OrchestrationUFLFactSnapshotStateError(
            "orchestration_ufl_fact_snapshot_project_mismatch"
        )

    async with session_factory() as session:
        try:
            rows = await ufl_fact_snapshot_repository.list_orchestration_ufl_fact_source_rows(
                session,
                orchestration_id=orchestration_id,
            )
            authenticated_facts = fact_diff_service.build_authenticated_fact_source_facts(
                rows=rows,
                source_snapshot=source_snapshot,
                expected_run_id=source_snapshot.state.extraction_run_id,
            )
            return _build_snapshot(
                project_id=project_id,
                source_snapshot=source_snapshot,
                authenticated_facts=authenticated_facts,
            )
        except fact_diff_service.DocumentRevisionFactDiffInvariantError:
            raise OrchestrationUFLFactSnapshotInvariantError(
                "orchestration_ufl_fact_snapshot_source_mismatch"
            ) from None
        finally:
            await session.rollback()
