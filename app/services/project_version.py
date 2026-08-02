from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import re
import uuid
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.project import ProjectStatus
from app.models.project_version import ProjectVersion, ProjectVersionCreationKind
import app.repositories.project_version as project_version_repository
from app.schemas.dynamic_schema_knowledge_view import (
    DynamicSchemaKnowledgeField,
    DynamicSchemaKnowledgeRecord,
    DynamicSchemaKnowledgeSection,
    DynamicSchemaKnowledgeView,
)
from app.schemas.dynamic_schema_review_projection import (
    DynamicSchemaReviewedFact,
    DynamicSchemaReviewedField,
)
from app.schemas.dynamic_schema_ufl_projection import DynamicSchemaUFLProjectedField
from app.schemas.project_version import (
    ProjectVersionCreateResult,
    ProjectVersionSnapshot,
)
from app.schemas.ufl_fact_snapshot import (
    UFLFactEvidenceLocator,
    UFLFactEvidenceSnapshot,
    UFLFactSnapshot,
    UFLFactValueGroupSnapshot,
    UFLFactValueSnapshot,
)
import app.services.dynamic_schema_knowledge_view as knowledge_view_service
import app.services.fact_value_duplicate_grouping as duplicate_grouping_service
from app.utils.deterministic_json import freeze_deterministic_json_value
from app.utils.validation import normalize_text


PROJECT_VERSION_MANIFEST_ALGORITHM_NAME = "project_version_manifest"
PROJECT_VERSION_MANIFEST_ALGORITHM_VERSION = "1.0.0"
SNAPSHOT_FORMAT_VERSION = "1.0.0"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_KNOWN_WRITE_CONSTRAINTS = {
    "pk_project_versions",
    "uq_projver_id_project",
    "uq_projver_project_verno",
    "uq_projver_manifest_hash",
}
_IDENTITY_CONSTRAINTS = {
    "pk_project_versions",
    "uq_projver_id_project",
}
_SUPPORTED_CREATE_KINDS = {
    ProjectVersionCreationKind.MANUAL.value,
    ProjectVersionCreationKind.AUTOMATIC.value,
    ProjectVersionCreationKind.PRE_PUBLISH.value,
}


class ProjectVersionError(Exception):
    """Base error for ProjectVersion services."""


class ProjectVersionStateError(ProjectVersionError):
    """Raised when ProjectVersion inputs or state are invalid."""


class ProjectVersionInvariantError(ProjectVersionError):
    """Raised when immutable ProjectVersion invariants are violated."""


@dataclass(frozen=True, slots=True)
class _PreparedProjectVersionSource:
    project_version_id: uuid.UUID
    project_id: uuid.UUID
    schema_id: uuid.UUID
    schema_version_id: uuid.UUID
    orchestration_id: uuid.UUID
    extraction_run_id: uuid.UUID
    consistency_check_application_id: uuid.UUID
    source_consistency_application_id: uuid.UUID
    created_by_id: uuid.UUID
    creation_kind: str
    reason: str | None
    schema_definition_manifest_hash: str
    ufl_source_manifest_hash: str
    consistency_result_manifest_hash: str
    raw_projection_manifest_hash: str
    reviewed_projection_manifest_hash: str
    knowledge_view_manifest_hash: str
    knowledge_view_algorithm_name: str
    knowledge_view_algorithm_version: str
    snapshot_json: dict[str, object]
    snapshot_json_hash: str
    record_count: int
    section_count: int
    field_count: int
    missing_field_count: int
    review_required_field_count: int
    resolved_field_count: int
    observation_only_field_count: int
    mixed_field_count: int


def _require_uuid(value: object, *, field_name: str) -> uuid.UUID:
    if not isinstance(value, uuid.UUID):
        raise ProjectVersionStateError(f"project_version_{field_name}_invalid")
    return value


def _require_snapshot_uuid(value: object) -> uuid.UUID:
    if not isinstance(value, uuid.UUID):
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    return value


def _require_snapshot_sha256(value: object) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    return value


def _require_snapshot_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    return value


def _require_snapshot_bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    return value


def _require_aware_datetime(value: object, *, error_code: str) -> datetime:
    if not isinstance(value, datetime):
        raise ProjectVersionInvariantError(error_code)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ProjectVersionInvariantError(error_code)
    return value.astimezone(timezone.utc)


def _normalize_creation_kind(value: object) -> str:
    if not isinstance(value, str):
        raise ProjectVersionStateError("project_version_creation_kind_invalid")
    normalized = normalize_text(value)
    if normalized not in {kind.value for kind in ProjectVersionCreationKind}:
        raise ProjectVersionStateError("project_version_creation_kind_invalid")
    if normalized == ProjectVersionCreationKind.ROLLBACK.value:
        raise ProjectVersionStateError("project_version_creation_kind_unsupported")
    return normalized


def _normalize_reason(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ProjectVersionStateError("project_version_reason_invalid")
    normalized = normalize_text(value)
    if not normalized or len(normalized) > 2000:
        raise ProjectVersionStateError("project_version_reason_invalid")
    return normalized


def _get_integrity_constraint_name(error: IntegrityError) -> str | None:
    diag = getattr(error.orig, "diag", None)
    constraint_name = getattr(diag, "constraint_name", None)
    if constraint_name is None or not isinstance(constraint_name, str):
        return None
    return constraint_name


def _freeze_snapshot_json(snapshot_json: object) -> Mapping[str, object]:
    try:
        frozen = freeze_deterministic_json_value(snapshot_json)
    except ValueError:
        raise ProjectVersionInvariantError("project_version_snapshot_invalid") from None
    if not isinstance(frozen, Mapping):
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    return frozen


def _canonical_bytes(payload: object) -> bytes:
    try:
        return duplicate_grouping_service.canonicalize_deterministic_payload(payload)
    except Exception:
        raise ProjectVersionInvariantError("project_version_snapshot_invalid") from None


def _parse_json_object(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    return value


def _parse_json_sequence(value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    return value


def _parse_json_str(value: object) -> str:
    if not isinstance(value, str):
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    return value


def _parse_json_uuid(value: object) -> uuid.UUID:
    if not isinstance(value, str):
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    try:
        return uuid.UUID(value)
    except ValueError:
        raise ProjectVersionInvariantError("project_version_snapshot_invalid") from None


def _parse_optional_json_uuid(value: object) -> uuid.UUID | None:
    if value is None:
        return None
    return _parse_json_uuid(value)


def _parse_json_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    return value


def _parse_optional_json_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    return float(value)


def _parse_json_bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    return value


def _parse_json_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ProjectVersionInvariantError("project_version_snapshot_invalid") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    return parsed


def _parse_json_value(value: object) -> object:
    try:
        return freeze_deterministic_json_value(value)
    except ValueError:
        raise ProjectVersionInvariantError("project_version_snapshot_invalid") from None


def _deserialize_locator(payload: object) -> UFLFactEvidenceLocator:
    obj = _parse_json_object(payload)
    return UFLFactEvidenceLocator(
        location_key=_parse_json_str(obj["location_key"]),
        page_no=None if obj["page_no"] is None else _parse_json_int(obj["page_no"]),
        start_line=(
            None if obj["start_line"] is None else _parse_json_int(obj["start_line"])
        ),
        end_line=None if obj["end_line"] is None else _parse_json_int(obj["end_line"]),
        table_index=(
            None if obj["table_index"] is None else _parse_json_int(obj["table_index"])
        ),
        row_index=None if obj["row_index"] is None else _parse_json_int(obj["row_index"]),
    )


def _deserialize_evidence(payload: object) -> UFLFactEvidenceSnapshot:
    obj = _parse_json_object(payload)
    return UFLFactEvidenceSnapshot(
        evidence_link_id=_parse_json_uuid(obj["evidence_link_id"]),
        evidence_id=_parse_json_uuid(obj["evidence_id"]),
        document_revision_id=_parse_json_uuid(obj["document_revision_id"]),
        document_block_id=_parse_json_uuid(obj["document_block_id"]),
        locator=_deserialize_locator(obj["locator"]),
        excerpt=_parse_json_str(obj["excerpt"]),
        excerpt_hash=_parse_json_str(obj["excerpt_hash"]),
        content_hash=_parse_json_str(obj["content_hash"]),
        role=_parse_json_str(obj["role"]),
        is_primary=_parse_json_bool(obj["is_primary"]),
        source_order=_parse_json_int(obj["source_order"]),
    )


def _deserialize_value(payload: object) -> UFLFactValueSnapshot:
    obj = _parse_json_object(payload)
    return UFLFactValueSnapshot(
        fact_value_id=_parse_json_uuid(obj["fact_value_id"]),
        source_batch_id=_parse_json_uuid(obj["source_batch_id"]),
        source_application_id=_parse_json_uuid(obj["source_application_id"]),
        proposal_index=_parse_json_int(obj["proposal_index"]),
        normalized_value_text=_parse_json_str(obj["normalized_value_text"]),
        value_hash=_parse_json_str(obj["value_hash"]),
        language_code=(
            None if obj["language_code"] is None else _parse_json_str(obj["language_code"])
        ),
        confidence=_parse_optional_json_float(obj["confidence"]),
    )


def _deserialize_value_group(payload: object) -> UFLFactValueGroupSnapshot:
    obj = _parse_json_object(payload)
    return UFLFactValueGroupSnapshot(
        semantic_key_hash=_parse_json_str(obj["semantic_key_hash"]),
        value_type=_parse_json_str(obj["value_type"]),
        value_json=_parse_json_value(obj["value_json"]),
        referenced_entity_id=_parse_optional_json_uuid(obj["referenced_entity_id"]),
        fact_value_ids=tuple(
            _parse_json_uuid(item) for item in _parse_json_sequence(obj["fact_value_ids"])
        ),
        values=tuple(_deserialize_value(item) for item in _parse_json_sequence(obj["values"])),
        evidences=tuple(
            _deserialize_evidence(item) for item in _parse_json_sequence(obj["evidences"])
        ),
    )


def _deserialize_fact(payload: object) -> UFLFactSnapshot:
    obj = _parse_json_object(payload)
    return UFLFactSnapshot(
        fact_id=_parse_json_uuid(obj["fact_id"]),
        identity_hash=_parse_json_str(obj["identity_hash"]),
        subject_kind=_parse_json_str(obj["subject_kind"]),
        subject_key=_parse_json_str(obj["subject_key"]),
        subject_entity_id=_parse_optional_json_uuid(obj["subject_entity_id"]),
        predicate_key=_parse_json_str(obj["predicate_key"]),
        scope_key=None if obj["scope_key"] is None else _parse_json_str(obj["scope_key"]),
        semantic_group_count=_parse_json_int(obj["semantic_group_count"]),
        fact_value_count=_parse_json_int(obj["fact_value_count"]),
        value_groups=tuple(
            _deserialize_value_group(item)
            for item in _parse_json_sequence(obj["value_groups"])
        ),
    )


def _deserialize_source_field(payload: object) -> DynamicSchemaUFLProjectedField:
    obj = _parse_json_object(payload)
    return DynamicSchemaUFLProjectedField(
        field_id=_parse_json_uuid(obj["field_id"]),
        schema_version_id=_parse_json_uuid(obj["schema_version_id"]),
        field_key=_parse_json_str(obj["field_key"]),
        label=_parse_json_str(obj["label"]),
        description=(
            None if obj["description"] is None else _parse_json_str(obj["description"])
        ),
        predicate_key=_parse_json_str(obj["predicate_key"]),
        scope_key=None if obj["scope_key"] is None else _parse_json_str(obj["scope_key"]),
        expected_value_type=_parse_json_str(obj["expected_value_type"]),
        cardinality=_parse_json_str(obj["cardinality"]),
        is_required=_parse_json_bool(obj["is_required"]),
        is_title=_parse_json_bool(obj["is_title"]),
        is_summary=_parse_json_bool(obj["is_summary"]),
        is_hidden=_parse_json_bool(obj["is_hidden"]),
        group_key=None if obj["group_key"] is None else _parse_json_str(obj["group_key"]),
        display_order=_parse_json_int(obj["display_order"]),
        display_config=_parse_json_value(obj["display_config"]),
        validation_rules=_parse_json_value(obj["validation_rules"]),
        created_at=_parse_json_datetime(obj["created_at"]),
        matched_facts=tuple(
            _deserialize_fact(item) for item in _parse_json_sequence(obj["matched_facts"])
        ),
        matched_fact_count=_parse_json_int(obj["matched_fact_count"]),
        semantic_value_count=_parse_json_int(obj["semantic_value_count"]),
        is_missing=_parse_json_bool(obj["is_missing"]),
        type_compatible=_parse_json_bool(obj["type_compatible"]),
        issues=tuple(
            _parse_json_str(item) for item in _parse_json_sequence(obj["issues"])
        ),
    )


def _deserialize_reviewed_fact(payload: object) -> DynamicSchemaReviewedFact:
    obj = _parse_json_object(payload)
    return DynamicSchemaReviewedFact(
        fact=_deserialize_fact(obj["fact"]),
        review_state=_parse_json_str(obj["review_state"]),
        candidate_id=_parse_optional_json_uuid(obj["candidate_id"]),
        assessment_id=_parse_optional_json_uuid(obj["assessment_id"]),
        resolution_basis=_parse_json_str(obj["resolution_basis"]),
        current_decision_id=_parse_optional_json_uuid(obj["current_decision_id"]),
        current_decision_kind=(
            None
            if obj["current_decision_kind"] is None
            else _parse_json_str(obj["current_decision_kind"])
        ),
        effective_fact_value_ids=tuple(
            _parse_json_uuid(item)
            for item in _parse_json_sequence(obj["effective_fact_value_ids"])
        ),
        requires_review=_parse_json_bool(obj["requires_review"]),
    )


def _deserialize_reviewed_field(payload: object) -> DynamicSchemaReviewedField:
    obj = _parse_json_object(payload)
    return DynamicSchemaReviewedField(
        source_field=_deserialize_source_field(obj["source_field"]),
        reviewed_facts=tuple(
            _deserialize_reviewed_fact(item)
            for item in _parse_json_sequence(obj["reviewed_facts"])
        ),
        review_required=_parse_json_bool(obj["review_required"]),
        resolved_fact_count=_parse_json_int(obj["resolved_fact_count"]),
        review_required_fact_count=_parse_json_int(obj["review_required_fact_count"]),
        effective_fact_value_ids=tuple(
            _parse_json_uuid(item)
            for item in _parse_json_sequence(obj["effective_fact_value_ids"])
        ),
    )


def _deserialize_knowledge_field(payload: object) -> DynamicSchemaKnowledgeField:
    obj = _parse_json_object(payload)
    return DynamicSchemaKnowledgeField(
        source_field=_deserialize_source_field(obj["source_field"]),
        reviewed_facts=tuple(
            _deserialize_reviewed_fact(item)
            for item in _parse_json_sequence(obj["reviewed_facts"])
        ),
        knowledge_state=_parse_json_str(obj["knowledge_state"]),
        effective_fact_value_ids=tuple(
            _parse_json_uuid(item)
            for item in _parse_json_sequence(obj["effective_fact_value_ids"])
        ),
        observed_fact_value_count=_parse_json_int(obj["observed_fact_value_count"]),
        semantic_value_count=_parse_json_int(obj["semantic_value_count"]),
        has_schema_issues=_parse_json_bool(obj["has_schema_issues"]),
    )


def _deserialize_knowledge_section(payload: object) -> DynamicSchemaKnowledgeSection:
    obj = _parse_json_object(payload)
    return DynamicSchemaKnowledgeSection(
        group_key=None if obj["group_key"] is None else _parse_json_str(obj["group_key"]),
        display_order=_parse_json_int(obj["display_order"]),
        fields=tuple(
            _deserialize_knowledge_field(item)
            for item in _parse_json_sequence(obj["fields"])
        ),
    )


def _deserialize_knowledge_record(payload: object) -> DynamicSchemaKnowledgeRecord:
    obj = _parse_json_object(payload)
    return DynamicSchemaKnowledgeRecord(
        subject_key=_parse_json_str(obj["subject_key"]),
        title_field_key=(
            None if obj["title_field_key"] is None else _parse_json_str(obj["title_field_key"])
        ),
        has_review_required=_parse_json_bool(obj["has_review_required"]),
        issue_count=_parse_json_int(obj["issue_count"]),
        sections=tuple(
            _deserialize_knowledge_section(item)
            for item in _parse_json_sequence(obj["sections"])
        ),
    )


def _deserialize_dynamic_schema_knowledge_view(
    payload: Mapping[str, object],
) -> DynamicSchemaKnowledgeView:
    return DynamicSchemaKnowledgeView(
        project_id=_parse_json_uuid(payload["project_id"]),
        schema_id=_parse_json_uuid(payload["schema_id"]),
        schema_version_id=_parse_json_uuid(payload["schema_version_id"]),
        orchestration_id=_parse_json_uuid(payload["orchestration_id"]),
        extraction_run_id=_parse_json_uuid(payload["extraction_run_id"]),
        consistency_check_application_id=_parse_json_uuid(
            payload["consistency_check_application_id"]
        ),
        source_consistency_application_id=_parse_json_uuid(
            payload["source_consistency_application_id"]
        ),
        schema_definition_manifest_hash=_parse_json_str(
            payload["schema_definition_manifest_hash"]
        ),
        ufl_source_manifest_hash=_parse_json_str(payload["ufl_source_manifest_hash"]),
        consistency_result_manifest_hash=_parse_json_str(
            payload["consistency_result_manifest_hash"]
        ),
        raw_projection_manifest_hash=_parse_json_str(payload["raw_projection_manifest_hash"]),
        reviewed_projection_manifest_hash=_parse_json_str(
            payload["reviewed_projection_manifest_hash"]
        ),
        comparison_quality=_parse_json_str(payload["comparison_quality"]),
        algorithm_name=_parse_json_str(payload["algorithm_name"]),
        algorithm_version=_parse_json_str(payload["algorithm_version"]),
        record_count=_parse_json_int(payload["record_count"]),
        section_count=_parse_json_int(payload["section_count"]),
        field_count=_parse_json_int(payload["field_count"]),
        missing_field_count=_parse_json_int(payload["missing_field_count"]),
        review_required_field_count=_parse_json_int(
            payload["review_required_field_count"]
        ),
        resolved_field_count=_parse_json_int(payload["resolved_field_count"]),
        observation_only_field_count=_parse_json_int(
            payload["observation_only_field_count"]
        ),
        mixed_field_count=_parse_json_int(payload["mixed_field_count"]),
        records=tuple(
            _deserialize_knowledge_record(item)
            for item in _parse_json_sequence(payload["records"])
        ),
        knowledge_view_manifest_hash=_parse_json_str(payload["knowledge_view_manifest_hash"]),
    )


def _build_version_manifest_hash(
    *,
    project_version_id: uuid.UUID,
    project_id: uuid.UUID,
    version_no: int,
    created_by_id: uuid.UUID,
    creation_kind: str,
    reason: str | None,
    copied_from_version_id: uuid.UUID | None,
    schema_id: uuid.UUID,
    schema_version_id: uuid.UUID,
    orchestration_id: uuid.UUID,
    extraction_run_id: uuid.UUID,
    consistency_check_application_id: uuid.UUID,
    source_consistency_application_id: uuid.UUID,
    schema_definition_manifest_hash: str,
    ufl_source_manifest_hash: str,
    consistency_result_manifest_hash: str,
    raw_projection_manifest_hash: str,
    reviewed_projection_manifest_hash: str,
    knowledge_view_manifest_hash: str,
    knowledge_view_algorithm_name: str,
    knowledge_view_algorithm_version: str,
    snapshot_json_hash: str,
    record_count: int,
    section_count: int,
    field_count: int,
    missing_field_count: int,
    review_required_field_count: int,
    resolved_field_count: int,
    observation_only_field_count: int,
    mixed_field_count: int,
    created_at: datetime,
) -> str:
    return duplicate_grouping_service.hash_deterministic_payload(
        {
            "project_version_id": str(project_version_id),
            "project_id": str(project_id),
            "version_no": version_no,
            "created_by_id": str(created_by_id),
            "creation_kind": creation_kind,
            "reason": reason,
            "copied_from_version_id": (
                None if copied_from_version_id is None else str(copied_from_version_id)
            ),
            "schema_id": str(schema_id),
            "schema_version_id": str(schema_version_id),
            "orchestration_id": str(orchestration_id),
            "extraction_run_id": str(extraction_run_id),
            "consistency_check_application_id": str(consistency_check_application_id),
            "source_consistency_application_id": str(source_consistency_application_id),
            "source_manifests": {
                "schema_definition_manifest_hash": schema_definition_manifest_hash,
                "ufl_source_manifest_hash": ufl_source_manifest_hash,
                "consistency_result_manifest_hash": consistency_result_manifest_hash,
                "raw_projection_manifest_hash": raw_projection_manifest_hash,
                "reviewed_projection_manifest_hash": reviewed_projection_manifest_hash,
                "knowledge_view_manifest_hash": knowledge_view_manifest_hash,
            },
            "knowledge_view_algorithm": {
                "name": knowledge_view_algorithm_name,
                "version": knowledge_view_algorithm_version,
            },
            "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
            "snapshot_json_hash": snapshot_json_hash,
            "counts": {
                "record_count": record_count,
                "section_count": section_count,
                "field_count": field_count,
                "missing_field_count": missing_field_count,
                "review_required_field_count": review_required_field_count,
                "resolved_field_count": resolved_field_count,
                "observation_only_field_count": observation_only_field_count,
                "mixed_field_count": mixed_field_count,
            },
            "created_at": created_at.astimezone(timezone.utc).isoformat(),
            "manifest_algorithm": {
                "name": PROJECT_VERSION_MANIFEST_ALGORITHM_NAME,
                "version": PROJECT_VERSION_MANIFEST_ALGORITHM_VERSION,
            },
        }
    )


def _prepared_manifest_hash(
    prepared: _PreparedProjectVersionSource,
    *,
    version_no: int,
    created_at: datetime,
) -> str:
    return _build_version_manifest_hash(
        project_version_id=prepared.project_version_id,
        project_id=prepared.project_id,
        version_no=version_no,
        created_by_id=prepared.created_by_id,
        creation_kind=prepared.creation_kind,
        reason=prepared.reason,
        copied_from_version_id=None,
        schema_id=prepared.schema_id,
        schema_version_id=prepared.schema_version_id,
        orchestration_id=prepared.orchestration_id,
        extraction_run_id=prepared.extraction_run_id,
        consistency_check_application_id=prepared.consistency_check_application_id,
        source_consistency_application_id=prepared.source_consistency_application_id,
        schema_definition_manifest_hash=prepared.schema_definition_manifest_hash,
        ufl_source_manifest_hash=prepared.ufl_source_manifest_hash,
        consistency_result_manifest_hash=prepared.consistency_result_manifest_hash,
        raw_projection_manifest_hash=prepared.raw_projection_manifest_hash,
        reviewed_projection_manifest_hash=prepared.reviewed_projection_manifest_hash,
        knowledge_view_manifest_hash=prepared.knowledge_view_manifest_hash,
        knowledge_view_algorithm_name=prepared.knowledge_view_algorithm_name,
        knowledge_view_algorithm_version=prepared.knowledge_view_algorithm_version,
        snapshot_json_hash=prepared.snapshot_json_hash,
        record_count=prepared.record_count,
        section_count=prepared.section_count,
        field_count=prepared.field_count,
        missing_field_count=prepared.missing_field_count,
        review_required_field_count=prepared.review_required_field_count,
        resolved_field_count=prepared.resolved_field_count,
        observation_only_field_count=prepared.observation_only_field_count,
        mixed_field_count=prepared.mixed_field_count,
        created_at=created_at,
    )


def _snapshot_kwargs(snapshot: ProjectVersionSnapshot) -> dict[str, object]:
    return {
        "id": snapshot.id,
        "project_id": snapshot.project_id,
        "version_no": snapshot.version_no,
        "created_by_id": snapshot.created_by_id,
        "creation_kind": snapshot.creation_kind,
        "copied_from_version_id": snapshot.copied_from_version_id,
        "reason": snapshot.reason,
        "schema_id": snapshot.schema_id,
        "schema_version_id": snapshot.schema_version_id,
        "orchestration_id": snapshot.orchestration_id,
        "extraction_run_id": snapshot.extraction_run_id,
        "consistency_check_application_id": snapshot.consistency_check_application_id,
        "source_consistency_application_id": snapshot.source_consistency_application_id,
        "schema_definition_manifest_hash": snapshot.schema_definition_manifest_hash,
        "ufl_source_manifest_hash": snapshot.ufl_source_manifest_hash,
        "consistency_result_manifest_hash": snapshot.consistency_result_manifest_hash,
        "raw_projection_manifest_hash": snapshot.raw_projection_manifest_hash,
        "reviewed_projection_manifest_hash": snapshot.reviewed_projection_manifest_hash,
        "knowledge_view_manifest_hash": snapshot.knowledge_view_manifest_hash,
        "knowledge_view_algorithm_name": snapshot.knowledge_view_algorithm_name,
        "knowledge_view_algorithm_version": snapshot.knowledge_view_algorithm_version,
        "snapshot_format_version": snapshot.snapshot_format_version,
        "snapshot_json": snapshot.snapshot_json,
        "snapshot_json_hash": snapshot.snapshot_json_hash,
        "version_manifest_hash": snapshot.version_manifest_hash,
        "record_count": snapshot.record_count,
        "section_count": snapshot.section_count,
        "field_count": snapshot.field_count,
        "missing_field_count": snapshot.missing_field_count,
        "review_required_field_count": snapshot.review_required_field_count,
        "resolved_field_count": snapshot.resolved_field_count,
        "observation_only_field_count": snapshot.observation_only_field_count,
        "mixed_field_count": snapshot.mixed_field_count,
        "created_at": snapshot.created_at,
        "is_current": snapshot.is_current,
    }


def _to_create_result(
    snapshot: ProjectVersionSnapshot,
    *,
    created_new: bool,
) -> ProjectVersionCreateResult:
    return ProjectVersionCreateResult(
        **_snapshot_kwargs(snapshot),
        created_new=created_new,
    )


def _build_snapshot_from_row(
    row: object,
    *,
    is_current: bool,
) -> ProjectVersionSnapshot:
    return ProjectVersionSnapshot(
        id=row.id,
        project_id=row.project_id,
        version_no=row.version_no,
        created_by_id=row.created_by_id,
        creation_kind=row.creation_kind,
        copied_from_version_id=row.copied_from_version_id,
        reason=row.reason,
        schema_id=row.schema_id,
        schema_version_id=row.schema_version_id,
        orchestration_id=row.orchestration_id,
        extraction_run_id=row.extraction_run_id,
        consistency_check_application_id=row.consistency_check_application_id,
        source_consistency_application_id=row.source_consistency_application_id,
        schema_definition_manifest_hash=row.schema_definition_manifest_hash,
        ufl_source_manifest_hash=row.ufl_source_manifest_hash,
        consistency_result_manifest_hash=row.consistency_result_manifest_hash,
        raw_projection_manifest_hash=row.raw_projection_manifest_hash,
        reviewed_projection_manifest_hash=row.reviewed_projection_manifest_hash,
        knowledge_view_manifest_hash=row.knowledge_view_manifest_hash,
        knowledge_view_algorithm_name=row.knowledge_view_algorithm_name,
        knowledge_view_algorithm_version=row.knowledge_view_algorithm_version,
        snapshot_format_version=row.snapshot_format_version,
        snapshot_json=_freeze_snapshot_json(row.snapshot_json),
        snapshot_json_hash=row.snapshot_json_hash,
        version_manifest_hash=row.version_manifest_hash,
        record_count=row.record_count,
        section_count=row.section_count,
        field_count=row.field_count,
        missing_field_count=row.missing_field_count,
        review_required_field_count=row.review_required_field_count,
        resolved_field_count=row.resolved_field_count,
        observation_only_field_count=row.observation_only_field_count,
        mixed_field_count=row.mixed_field_count,
        created_at=row.created_at,
        is_current=is_current,
    )


def _assert_snapshot_matches_knowledge_view(
    snapshot: ProjectVersionSnapshot,
    *,
    view: DynamicSchemaKnowledgeView,
) -> None:
    if snapshot.project_id != view.project_id:
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    if snapshot.schema_id != view.schema_id:
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    if snapshot.schema_version_id != view.schema_version_id:
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    if snapshot.orchestration_id != view.orchestration_id:
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    if snapshot.extraction_run_id != view.extraction_run_id:
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    if (
        snapshot.consistency_check_application_id
        != view.consistency_check_application_id
    ):
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    if (
        snapshot.source_consistency_application_id
        != view.source_consistency_application_id
    ):
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    if (
        snapshot.schema_definition_manifest_hash
        != view.schema_definition_manifest_hash
    ):
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    if snapshot.ufl_source_manifest_hash != view.ufl_source_manifest_hash:
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    if (
        snapshot.consistency_result_manifest_hash
        != view.consistency_result_manifest_hash
    ):
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    if (
        snapshot.raw_projection_manifest_hash != view.raw_projection_manifest_hash
    ):
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    if (
        snapshot.reviewed_projection_manifest_hash
        != view.reviewed_projection_manifest_hash
    ):
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    if snapshot.knowledge_view_manifest_hash != view.knowledge_view_manifest_hash:
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    if snapshot.knowledge_view_algorithm_name != view.algorithm_name:
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    if snapshot.knowledge_view_algorithm_version != view.algorithm_version:
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    if snapshot.record_count != view.record_count:
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    if snapshot.section_count != view.section_count:
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    if snapshot.field_count != view.field_count:
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    if snapshot.missing_field_count != view.missing_field_count:
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    if (
        snapshot.review_required_field_count
        != view.review_required_field_count
    ):
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    if snapshot.resolved_field_count != view.resolved_field_count:
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    if (
        snapshot.observation_only_field_count
        != view.observation_only_field_count
    ):
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    if snapshot.mixed_field_count != view.mixed_field_count:
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")


def authenticate_project_version_snapshot(
    snapshot: ProjectVersionSnapshot,
) -> ProjectVersionSnapshot:
    if not isinstance(snapshot, ProjectVersionSnapshot):
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    _require_snapshot_uuid(snapshot.id)
    _require_snapshot_uuid(snapshot.project_id)
    _require_snapshot_uuid(snapshot.created_by_id)
    _require_snapshot_uuid(snapshot.schema_id)
    _require_snapshot_uuid(snapshot.schema_version_id)
    _require_snapshot_uuid(snapshot.orchestration_id)
    _require_snapshot_uuid(snapshot.extraction_run_id)
    _require_snapshot_uuid(snapshot.consistency_check_application_id)
    _require_snapshot_uuid(snapshot.source_consistency_application_id)
    if snapshot.copied_from_version_id is not None:
        _require_snapshot_uuid(snapshot.copied_from_version_id)
    if snapshot.creation_kind not in {kind.value for kind in ProjectVersionCreationKind}:
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    if snapshot.creation_kind == ProjectVersionCreationKind.ROLLBACK.value:
        if snapshot.copied_from_version_id is None:
            raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    elif snapshot.copied_from_version_id is not None:
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    if snapshot.reason is not None and (
        not isinstance(snapshot.reason, str)
        or not snapshot.reason
        or len(snapshot.reason) > 2000
    ):
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    if isinstance(snapshot.version_no, bool) or not isinstance(snapshot.version_no, int):
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    if snapshot.version_no <= 0:
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    for value in (
        snapshot.schema_definition_manifest_hash,
        snapshot.ufl_source_manifest_hash,
        snapshot.consistency_result_manifest_hash,
        snapshot.raw_projection_manifest_hash,
        snapshot.reviewed_projection_manifest_hash,
        snapshot.knowledge_view_manifest_hash,
        snapshot.snapshot_json_hash,
        snapshot.version_manifest_hash,
    ):
        _require_snapshot_sha256(value)
    _require_snapshot_count(snapshot.record_count)
    _require_snapshot_count(snapshot.section_count)
    _require_snapshot_count(snapshot.field_count)
    _require_snapshot_count(snapshot.missing_field_count)
    _require_snapshot_count(snapshot.review_required_field_count)
    _require_snapshot_count(snapshot.resolved_field_count)
    _require_snapshot_count(snapshot.observation_only_field_count)
    _require_snapshot_count(snapshot.mixed_field_count)
    if (
        snapshot.missing_field_count
        + snapshot.review_required_field_count
        + snapshot.resolved_field_count
        + snapshot.observation_only_field_count
        + snapshot.mixed_field_count
        != snapshot.field_count
    ):
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    if snapshot.knowledge_view_algorithm_name != knowledge_view_service.DYNAMIC_SCHEMA_KNOWLEDGE_VIEW_ALGORITHM_NAME:
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    if snapshot.knowledge_view_algorithm_version != knowledge_view_service.DYNAMIC_SCHEMA_KNOWLEDGE_VIEW_ALGORITHM_VERSION:
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    if snapshot.snapshot_format_version != SNAPSHOT_FORMAT_VERSION:
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    _require_aware_datetime(snapshot.created_at, error_code="project_version_snapshot_invalid")
    _require_snapshot_bool(snapshot.is_current)
    frozen_snapshot_json = _freeze_snapshot_json(snapshot.snapshot_json)
    knowledge_view = _deserialize_dynamic_schema_knowledge_view(frozen_snapshot_json)
    try:
        authenticated_view = knowledge_view_service.authenticate_dynamic_schema_knowledge_view(
            knowledge_view,
            subject_keys=None,
        )
    except knowledge_view_service.DynamicSchemaKnowledgeViewError:
        raise ProjectVersionInvariantError("project_version_snapshot_invalid") from None
    _assert_snapshot_matches_knowledge_view(snapshot, view=authenticated_view)
    expected_snapshot_json = knowledge_view_service.serialize_dynamic_schema_knowledge_view(
        authenticated_view,
        subject_keys=None,
    )
    if _canonical_bytes(frozen_snapshot_json) != _canonical_bytes(expected_snapshot_json):
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    if (
        duplicate_grouping_service.hash_deterministic_payload(expected_snapshot_json)
        != snapshot.snapshot_json_hash
    ):
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    expected_version_manifest_hash = _build_version_manifest_hash(
        project_version_id=snapshot.id,
        project_id=snapshot.project_id,
        version_no=snapshot.version_no,
        created_by_id=snapshot.created_by_id,
        creation_kind=snapshot.creation_kind,
        reason=snapshot.reason,
        copied_from_version_id=snapshot.copied_from_version_id,
        schema_id=snapshot.schema_id,
        schema_version_id=snapshot.schema_version_id,
        orchestration_id=snapshot.orchestration_id,
        extraction_run_id=snapshot.extraction_run_id,
        consistency_check_application_id=snapshot.consistency_check_application_id,
        source_consistency_application_id=snapshot.source_consistency_application_id,
        schema_definition_manifest_hash=snapshot.schema_definition_manifest_hash,
        ufl_source_manifest_hash=snapshot.ufl_source_manifest_hash,
        consistency_result_manifest_hash=snapshot.consistency_result_manifest_hash,
        raw_projection_manifest_hash=snapshot.raw_projection_manifest_hash,
        reviewed_projection_manifest_hash=snapshot.reviewed_projection_manifest_hash,
        knowledge_view_manifest_hash=snapshot.knowledge_view_manifest_hash,
        knowledge_view_algorithm_name=snapshot.knowledge_view_algorithm_name,
        knowledge_view_algorithm_version=snapshot.knowledge_view_algorithm_version,
        snapshot_json_hash=snapshot.snapshot_json_hash,
        record_count=snapshot.record_count,
        section_count=snapshot.section_count,
        field_count=snapshot.field_count,
        missing_field_count=snapshot.missing_field_count,
        review_required_field_count=snapshot.review_required_field_count,
        resolved_field_count=snapshot.resolved_field_count,
        observation_only_field_count=snapshot.observation_only_field_count,
        mixed_field_count=snapshot.mixed_field_count,
        created_at=snapshot.created_at,
    )
    if expected_version_manifest_hash != snapshot.version_manifest_hash:
        raise ProjectVersionInvariantError("project_version_snapshot_invalid")
    return snapshot


def _prepared_matches_snapshot(
    prepared: _PreparedProjectVersionSource,
    *,
    snapshot: ProjectVersionSnapshot,
) -> bool:
    if snapshot.project_id != prepared.project_id:
        return False
    if snapshot.id != prepared.project_version_id:
        return False
    if snapshot.created_by_id != prepared.created_by_id:
        return False
    if snapshot.creation_kind != prepared.creation_kind:
        return False
    if snapshot.reason != prepared.reason:
        return False
    if snapshot.copied_from_version_id is not None:
        return False
    if snapshot.schema_id != prepared.schema_id:
        return False
    if snapshot.schema_version_id != prepared.schema_version_id:
        return False
    if snapshot.orchestration_id != prepared.orchestration_id:
        return False
    if snapshot.extraction_run_id != prepared.extraction_run_id:
        return False
    if (
        snapshot.consistency_check_application_id
        != prepared.consistency_check_application_id
    ):
        return False
    if (
        snapshot.source_consistency_application_id
        != prepared.source_consistency_application_id
    ):
        return False
    if (
        snapshot.schema_definition_manifest_hash
        != prepared.schema_definition_manifest_hash
    ):
        return False
    if snapshot.ufl_source_manifest_hash != prepared.ufl_source_manifest_hash:
        return False
    if (
        snapshot.consistency_result_manifest_hash
        != prepared.consistency_result_manifest_hash
    ):
        return False
    if snapshot.raw_projection_manifest_hash != prepared.raw_projection_manifest_hash:
        return False
    if (
        snapshot.reviewed_projection_manifest_hash
        != prepared.reviewed_projection_manifest_hash
    ):
        return False
    if (
        snapshot.knowledge_view_manifest_hash
        != prepared.knowledge_view_manifest_hash
    ):
        return False
    if (
        snapshot.knowledge_view_algorithm_name
        != prepared.knowledge_view_algorithm_name
    ):
        return False
    if (
        snapshot.knowledge_view_algorithm_version
        != prepared.knowledge_view_algorithm_version
    ):
        return False
    if snapshot.snapshot_format_version != SNAPSHOT_FORMAT_VERSION:
        return False
    if snapshot.snapshot_json_hash != prepared.snapshot_json_hash:
        return False
    if snapshot.record_count != prepared.record_count:
        return False
    if snapshot.section_count != prepared.section_count:
        return False
    if snapshot.field_count != prepared.field_count:
        return False
    if snapshot.missing_field_count != prepared.missing_field_count:
        return False
    if (
        snapshot.review_required_field_count
        != prepared.review_required_field_count
    ):
        return False
    if snapshot.resolved_field_count != prepared.resolved_field_count:
        return False
    if (
        snapshot.observation_only_field_count
        != prepared.observation_only_field_count
    ):
        return False
    if snapshot.mixed_field_count != prepared.mixed_field_count:
        return False
    if _canonical_bytes(snapshot.snapshot_json) != _canonical_bytes(prepared.snapshot_json):
        return False
    return (
        snapshot.version_manifest_hash
        == _prepared_manifest_hash(
            prepared,
            version_no=snapshot.version_no,
            created_at=snapshot.created_at,
        )
    )


def _request_identity_matches_existing_snapshot(
    snapshot: ProjectVersionSnapshot,
    *,
    project_id: uuid.UUID,
    schema_id: uuid.UUID,
    schema_version_id: uuid.UUID,
    orchestration_id: uuid.UUID,
    consistency_check_application_id: uuid.UUID,
    created_by_id: uuid.UUID,
    creation_kind: str,
    reason: str | None,
) -> bool:
    return (
        snapshot.project_id == project_id
        and snapshot.schema_id == schema_id
        and snapshot.schema_version_id == schema_version_id
        and snapshot.orchestration_id == orchestration_id
        and snapshot.consistency_check_application_id
        == consistency_check_application_id
        and snapshot.created_by_id == created_by_id
        and snapshot.creation_kind == creation_kind
        and snapshot.reason == reason
        and snapshot.copied_from_version_id is None
    )


async def _prepare_project_version_source(
    session_factory: Callable[[], AsyncSession],
    *,
    project_version_id: uuid.UUID,
    project_id: uuid.UUID,
    schema_id: uuid.UUID,
    schema_version_id: uuid.UUID,
    orchestration_id: uuid.UUID,
    consistency_check_application_id: uuid.UUID,
    created_by_id: uuid.UUID,
    creation_kind: str,
    reason: str | None,
) -> _PreparedProjectVersionSource:
    try:
        knowledge_view = await knowledge_view_service.build_dynamic_schema_knowledge_view(
            session_factory,
            project_id=project_id,
            schema_id=schema_id,
            schema_version_id=schema_version_id,
            orchestration_id=orchestration_id,
            consistency_check_application_id=consistency_check_application_id,
            subject_keys=None,
        )
        authenticated_view = knowledge_view_service.authenticate_dynamic_schema_knowledge_view(
            knowledge_view,
            subject_keys=None,
        )
    except knowledge_view_service.DynamicSchemaKnowledgeViewStateError:
        raise ProjectVersionStateError("project_version_knowledge_view_invalid") from None
    except knowledge_view_service.DynamicSchemaKnowledgeViewInvariantError:
        raise ProjectVersionInvariantError("project_version_knowledge_view_mismatch") from None

    if authenticated_view.project_id != project_id:
        raise ProjectVersionInvariantError("project_version_knowledge_view_mismatch")
    if authenticated_view.schema_id != schema_id:
        raise ProjectVersionInvariantError("project_version_knowledge_view_mismatch")
    if authenticated_view.schema_version_id != schema_version_id:
        raise ProjectVersionInvariantError("project_version_knowledge_view_mismatch")
    if authenticated_view.orchestration_id != orchestration_id:
        raise ProjectVersionInvariantError("project_version_knowledge_view_mismatch")
    if (
        authenticated_view.consistency_check_application_id
        != consistency_check_application_id
    ):
        raise ProjectVersionInvariantError("project_version_knowledge_view_mismatch")

    snapshot_json = knowledge_view_service.serialize_dynamic_schema_knowledge_view(
        authenticated_view,
        subject_keys=None,
    )
    return _PreparedProjectVersionSource(
        project_version_id=project_version_id,
        project_id=project_id,
        schema_id=schema_id,
        schema_version_id=schema_version_id,
        orchestration_id=orchestration_id,
        extraction_run_id=authenticated_view.extraction_run_id,
        consistency_check_application_id=consistency_check_application_id,
        source_consistency_application_id=authenticated_view.source_consistency_application_id,
        created_by_id=created_by_id,
        creation_kind=creation_kind,
        reason=reason,
        schema_definition_manifest_hash=authenticated_view.schema_definition_manifest_hash,
        ufl_source_manifest_hash=authenticated_view.ufl_source_manifest_hash,
        consistency_result_manifest_hash=authenticated_view.consistency_result_manifest_hash,
        raw_projection_manifest_hash=authenticated_view.raw_projection_manifest_hash,
        reviewed_projection_manifest_hash=authenticated_view.reviewed_projection_manifest_hash,
        knowledge_view_manifest_hash=authenticated_view.knowledge_view_manifest_hash,
        knowledge_view_algorithm_name=authenticated_view.algorithm_name,
        knowledge_view_algorithm_version=authenticated_view.algorithm_version,
        snapshot_json=snapshot_json,
        snapshot_json_hash=duplicate_grouping_service.hash_deterministic_payload(
            snapshot_json
        ),
        record_count=authenticated_view.record_count,
        section_count=authenticated_view.section_count,
        field_count=authenticated_view.field_count,
        missing_field_count=authenticated_view.missing_field_count,
        review_required_field_count=authenticated_view.review_required_field_count,
        resolved_field_count=authenticated_view.resolved_field_count,
        observation_only_field_count=authenticated_view.observation_only_field_count,
        mixed_field_count=authenticated_view.mixed_field_count,
    )


async def _recover_existing_snapshot_after_integrity_error(
    session_factory: Callable[[], AsyncSession],
    *,
    prepared: _PreparedProjectVersionSource,
) -> ProjectVersionCreateResult:
    async with session_factory() as read_session:
        try:
            existing = await project_version_repository.get_project_version_by_id(
                read_session,
                project_version_id=prepared.project_version_id,
            )
            if existing is None:
                raise ProjectVersionInvariantError(
                    "project_version_concurrent_ledger_missing"
                )
            project = await project_version_repository.get_project_by_id(
                read_session,
                project_id=existing.project_id,
            )
            snapshot = authenticate_project_version_snapshot(
                _build_snapshot_from_row(
                    existing,
                    is_current=(
                        project is not None and project.current_version_id == existing.id
                    ),
                )
            )
            await read_session.rollback()
        except BaseException:
            await read_session.rollback()
            raise
    if snapshot.project_id != prepared.project_id:
        raise ProjectVersionInvariantError("project_version_idempotency_mismatch")
    if not _prepared_matches_snapshot(prepared, snapshot=snapshot):
        raise ProjectVersionInvariantError("project_version_idempotency_mismatch")
    return _to_create_result(snapshot, created_new=False)


async def create_project_version(
    session_factory: Callable[[], AsyncSession],
    *,
    project_version_id: uuid.UUID,
    project_id: uuid.UUID,
    schema_id: uuid.UUID,
    schema_version_id: uuid.UUID,
    orchestration_id: uuid.UUID,
    consistency_check_application_id: uuid.UUID,
    created_by_id: uuid.UUID,
    creation_kind: str,
    reason: str | None = None,
) -> ProjectVersionCreateResult:
    project_version_id = _require_uuid(
        project_version_id,
        field_name="project_version_id",
    )
    project_id = _require_uuid(project_id, field_name="project_id")
    schema_id = _require_uuid(schema_id, field_name="schema_id")
    schema_version_id = _require_uuid(
        schema_version_id,
        field_name="schema_version_id",
    )
    orchestration_id = _require_uuid(
        orchestration_id,
        field_name="orchestration_id",
    )
    consistency_check_application_id = _require_uuid(
        consistency_check_application_id,
        field_name="consistency_check_application_id",
    )
    created_by_id = _require_uuid(created_by_id, field_name="created_by_id")
    normalized_creation_kind = _normalize_creation_kind(creation_kind)
    normalized_reason = _normalize_reason(reason)
    prepared: _PreparedProjectVersionSource | None = None

    async with session_factory() as write_session:
        try:
            project = await project_version_repository.get_project_for_update(
                write_session,
                project_id=project_id,
            )
            if project is None:
                raise ProjectVersionStateError("project_version_project_not_found")
            if project.status == ProjectStatus.ARCHIVED.value:
                raise ProjectVersionStateError("project_version_project_archived")
            created_by = await project_version_repository.get_user_by_id(
                write_session,
                user_id=created_by_id,
            )
            if created_by is None:
                raise ProjectVersionStateError("project_version_created_by_not_found")

            existing = await project_version_repository.get_project_version_by_id(
                write_session,
                project_version_id=project_version_id,
            )
            if existing is not None:
                snapshot = authenticate_project_version_snapshot(
                    _build_snapshot_from_row(
                        existing,
                        is_current=project.current_version_id == existing.id,
                    )
                )
                if not _request_identity_matches_existing_snapshot(
                    snapshot,
                    project_id=project_id,
                    schema_id=schema_id,
                    schema_version_id=schema_version_id,
                    orchestration_id=orchestration_id,
                    consistency_check_application_id=consistency_check_application_id,
                    created_by_id=created_by_id,
                    creation_kind=normalized_creation_kind,
                    reason=normalized_reason,
                ):
                    raise ProjectVersionInvariantError(
                        "project_version_idempotency_mismatch"
                    )
                prepared = await _prepare_project_version_source(
                    session_factory,
                    project_version_id=project_version_id,
                    project_id=project_id,
                    schema_id=schema_id,
                    schema_version_id=schema_version_id,
                    orchestration_id=orchestration_id,
                    consistency_check_application_id=consistency_check_application_id,
                    created_by_id=created_by_id,
                    creation_kind=normalized_creation_kind,
                    reason=normalized_reason,
                )
                if not _prepared_matches_snapshot(prepared, snapshot=snapshot):
                    raise ProjectVersionInvariantError(
                        "project_version_idempotency_mismatch"
                    )
                await write_session.commit()
                return _to_create_result(snapshot, created_new=False)

            prepared = await _prepare_project_version_source(
                session_factory,
                project_version_id=project_version_id,
                project_id=project_id,
                schema_id=schema_id,
                schema_version_id=schema_version_id,
                orchestration_id=orchestration_id,
                consistency_check_application_id=consistency_check_application_id,
                created_by_id=created_by_id,
                creation_kind=normalized_creation_kind,
                reason=normalized_reason,
            )
            next_version_no = (
                await project_version_repository.get_max_project_version_no(
                    write_session,
                    project_id=project_id,
                )
            ) + 1
            created_at = datetime.now(timezone.utc)
            project_version = ProjectVersion(
                id=project_version_id,
                project_id=project_id,
                version_no=next_version_no,
                created_by_id=created_by_id,
                creation_kind=normalized_creation_kind,
                copied_from_version_id=None,
                reason=normalized_reason,
                schema_id=schema_id,
                schema_version_id=schema_version_id,
                orchestration_id=orchestration_id,
                extraction_run_id=prepared.extraction_run_id,
                consistency_check_application_id=consistency_check_application_id,
                source_consistency_application_id=prepared.source_consistency_application_id,
                schema_definition_manifest_hash=prepared.schema_definition_manifest_hash,
                ufl_source_manifest_hash=prepared.ufl_source_manifest_hash,
                consistency_result_manifest_hash=prepared.consistency_result_manifest_hash,
                raw_projection_manifest_hash=prepared.raw_projection_manifest_hash,
                reviewed_projection_manifest_hash=prepared.reviewed_projection_manifest_hash,
                knowledge_view_manifest_hash=prepared.knowledge_view_manifest_hash,
                knowledge_view_algorithm_name=prepared.knowledge_view_algorithm_name,
                knowledge_view_algorithm_version=prepared.knowledge_view_algorithm_version,
                snapshot_format_version=SNAPSHOT_FORMAT_VERSION,
                snapshot_json=prepared.snapshot_json,
                snapshot_json_hash=prepared.snapshot_json_hash,
                version_manifest_hash=_prepared_manifest_hash(
                    prepared,
                    version_no=next_version_no,
                    created_at=created_at,
                ),
                record_count=prepared.record_count,
                section_count=prepared.section_count,
                field_count=prepared.field_count,
                missing_field_count=prepared.missing_field_count,
                review_required_field_count=prepared.review_required_field_count,
                resolved_field_count=prepared.resolved_field_count,
                observation_only_field_count=prepared.observation_only_field_count,
                mixed_field_count=prepared.mixed_field_count,
                created_at=created_at,
            )
            await project_version_repository.create_project_version(
                write_session,
                project_version,
            )
            project.current_version_id = project_version.id
            await write_session.commit()
            snapshot = authenticate_project_version_snapshot(
                _build_snapshot_from_row(project_version, is_current=True)
            )
            return _to_create_result(snapshot, created_new=True)
        except IntegrityError as error:
            constraint_name = _get_integrity_constraint_name(error)
            await write_session.rollback()
            if constraint_name not in _KNOWN_WRITE_CONSTRAINTS:
                raise ProjectVersionInvariantError(
                    "project_version_write_integrity_error"
                ) from None
        except BaseException:
            await write_session.rollback()
            raise

    return await _recover_existing_snapshot_after_integrity_error(
        session_factory,
        prepared=prepared,
    )


async def get_project_version_snapshot(
    session_factory: Callable[[], AsyncSession],
    *,
    project_id: uuid.UUID,
    project_version_id: uuid.UUID,
) -> ProjectVersionSnapshot:
    project_id = _require_uuid(project_id, field_name="project_id")
    project_version_id = _require_uuid(
        project_version_id,
        field_name="project_version_id",
    )
    async with session_factory() as read_session:
        try:
            project = await project_version_repository.get_project_by_id(
                read_session,
                project_id=project_id,
            )
            version = await project_version_repository.get_project_version_for_project(
                read_session,
                project_id=project_id,
                project_version_id=project_version_id,
            )
            if project is None or version is None:
                raise ProjectVersionStateError("project_version_not_found")
            snapshot = authenticate_project_version_snapshot(
                _build_snapshot_from_row(
                    version,
                    is_current=project.current_version_id == version.id,
                )
            )
            await read_session.rollback()
            return snapshot
        except BaseException:
            await read_session.rollback()
            raise
