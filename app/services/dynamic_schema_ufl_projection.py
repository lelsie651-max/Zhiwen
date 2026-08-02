from __future__ import annotations

import re
import unicodedata
import uuid
from collections.abc import Callable, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dynamic_schema import (
    DynamicSchemaFieldCardinality,
    DynamicSchemaFieldValueType,
)
from app.schemas.dynamic_schema_ufl_projection import (
    DynamicSchemaUFLProjectedField,
    DynamicSchemaUFLProjectedRecord,
    DynamicSchemaUFLProjection,
)
from app.schemas.fact import _normalize_required_text
from app.schemas.ufl_fact_snapshot import (
    OrchestrationUFLFactSnapshot,
    UFLFactEvidenceLocator,
    UFLFactEvidenceSnapshot,
    UFLFactSnapshot,
    UFLFactValueGroupSnapshot,
    UFLFactValueSnapshot,
)
from app.services import dynamic_schema_projection as dynamic_schema_projection_service
from app.services import fact_value_duplicate_grouping as duplicate_grouping_service
from app.services import ufl_fact_snapshot as ufl_fact_snapshot_service


DYNAMIC_SCHEMA_UFL_PROJECTION_ALGORITHM_NAME = "dynamic_schema_ufl_projection"
DYNAMIC_SCHEMA_UFL_PROJECTION_ALGORITHM_VERSION = "1.0.0"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DYNAMIC_SCHEMA_UFL_PROJECTION_ISSUE_ORDER = (
    "required_missing",
    "cardinality_one_multiple_facts",
    "cardinality_one_multiple_semantic_values",
    "value_type_mismatch",
)


class DynamicSchemaUFLProjectionError(Exception):
    """Base error for dynamic schema UFL projection failures."""


class DynamicSchemaUFLProjectionStateError(DynamicSchemaUFLProjectionError):
    """Raised when projection inputs are invalid."""


class DynamicSchemaUFLProjectionInvariantError(DynamicSchemaUFLProjectionError):
    """Raised when authenticated source snapshots drift or conflict."""


def _require_uuid(value: object, *, field_name: str) -> uuid.UUID:
    if not isinstance(value, uuid.UUID):
        raise DynamicSchemaUFLProjectionStateError(
            f"dynamic_schema_ufl_projection_{field_name}_invalid"
        )
    return value


def _require_projection_uuid(value: object) -> uuid.UUID:
    if not isinstance(value, uuid.UUID):
        raise DynamicSchemaUFLProjectionInvariantError(
            "dynamic_schema_ufl_projection_projection_invalid"
        )
    return value


def _require_sha256(value: object, *, error_code: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise DynamicSchemaUFLProjectionInvariantError(error_code)
    return value


def _require_projection_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DynamicSchemaUFLProjectionInvariantError(
            "dynamic_schema_ufl_projection_projection_invalid"
        )
    return value


def _normalize_subject_key(value: object) -> str:
    if not isinstance(value, str):
        raise DynamicSchemaUFLProjectionStateError(
            "dynamic_schema_ufl_projection_subject_keys_invalid"
        )
    normalized = unicodedata.normalize("NFC", value)
    try:
        return _normalize_required_text(
            normalized,
            field_name="subject_key",
            max_length=255,
        )
    except ValueError:
        raise DynamicSchemaUFLProjectionStateError(
            "dynamic_schema_ufl_projection_subject_keys_invalid"
        ) from None


def _validate_subject_keys(subject_keys: object) -> list[str] | None:
    if subject_keys is None:
        return None
    if not isinstance(subject_keys, list):
        raise DynamicSchemaUFLProjectionStateError(
            "dynamic_schema_ufl_projection_subject_keys_invalid"
        )
    if len(subject_keys) > 500:
        raise DynamicSchemaUFLProjectionStateError(
            "dynamic_schema_ufl_projection_subject_keys_invalid"
        )
    normalized_subject_keys = [_normalize_subject_key(value) for value in subject_keys]
    if len(set(normalized_subject_keys)) != len(normalized_subject_keys):
        raise DynamicSchemaUFLProjectionStateError(
            "dynamic_schema_ufl_projection_subject_keys_invalid"
        )
    return normalized_subject_keys


def normalize_dynamic_schema_ufl_subject_keys(
    subject_keys: object,
) -> list[str] | None:
    return _validate_subject_keys(subject_keys)


def _validate_schema_snapshot(
    *,
    project_id: uuid.UUID,
    schema_id: uuid.UUID,
    schema_version_id: uuid.UUID,
    snapshot: dynamic_schema_projection_service.DynamicSchemaDefinitionSnapshot,
) -> None:
    if snapshot.project_id != project_id:
        raise DynamicSchemaUFLProjectionInvariantError(
            "dynamic_schema_ufl_projection_source_project_mismatch"
        )
    if snapshot.schema_id != schema_id or snapshot.schema_version_id != schema_version_id:
        raise DynamicSchemaUFLProjectionInvariantError(
            "dynamic_schema_ufl_projection_schema_identity_mismatch"
        )
    if (
        snapshot.algorithm_name
        != dynamic_schema_projection_service.DYNAMIC_SCHEMA_DEFINITION_SNAPSHOT_ALGORITHM_NAME
        or snapshot.algorithm_version
        != dynamic_schema_projection_service.DYNAMIC_SCHEMA_DEFINITION_SNAPSHOT_ALGORITHM_VERSION
    ):
        raise DynamicSchemaUFLProjectionInvariantError(
            "dynamic_schema_ufl_projection_schema_identity_mismatch"
        )
    _require_sha256(
        snapshot.definition_manifest_hash,
        error_code="dynamic_schema_ufl_projection_schema_identity_mismatch",
    )


def _validate_ufl_snapshot(
    *,
    project_id: uuid.UUID,
    orchestration_id: uuid.UUID,
    snapshot: OrchestrationUFLFactSnapshot,
) -> None:
    if snapshot.project_id != project_id:
        raise DynamicSchemaUFLProjectionInvariantError(
            "dynamic_schema_ufl_projection_source_project_mismatch"
        )
    if snapshot.orchestration_id != orchestration_id:
        raise DynamicSchemaUFLProjectionInvariantError(
            "dynamic_schema_ufl_projection_ufl_identity_mismatch"
        )
    if (
        snapshot.algorithm_name
        != ufl_fact_snapshot_service.ORCHESTRATION_UFL_FACT_SNAPSHOT_ALGORITHM_NAME
        or snapshot.algorithm_version
        != ufl_fact_snapshot_service.ORCHESTRATION_UFL_FACT_SNAPSHOT_ALGORITHM_VERSION
    ):
        raise DynamicSchemaUFLProjectionInvariantError(
            "dynamic_schema_ufl_projection_ufl_identity_mismatch"
        )
    _require_sha256(
        snapshot.source_manifest_hash,
        error_code="dynamic_schema_ufl_projection_ufl_identity_mismatch",
    )


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


def serialize_dynamic_schema_ufl_fact(
    fact: UFLFactSnapshot,
) -> dict[str, object]:
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


def serialize_dynamic_schema_ufl_projected_field(
    field: DynamicSchemaUFLProjectedField,
) -> dict[str, object]:
    return {
        "field_id": str(field.field_id),
        "schema_version_id": str(field.schema_version_id),
        "field_key": field.field_key,
        "label": field.label,
        "description": field.description,
        "predicate_key": field.predicate_key,
        "scope_key": field.scope_key,
        "expected_value_type": field.expected_value_type,
        "cardinality": field.cardinality,
        "is_required": field.is_required,
        "is_title": field.is_title,
        "is_summary": field.is_summary,
        "is_hidden": field.is_hidden,
        "group_key": field.group_key,
        "display_order": field.display_order,
        "display_config": field.display_config,
        "validation_rules": field.validation_rules,
        "created_at": field.created_at.isoformat(),
        "matched_fact_count": field.matched_fact_count,
        "semantic_value_count": field.semantic_value_count,
        "is_missing": field.is_missing,
        "type_compatible": field.type_compatible,
        "issues": list(field.issues),
        "matched_facts": [
            serialize_dynamic_schema_ufl_fact(fact) for fact in field.matched_facts
        ],
    }


def serialize_dynamic_schema_ufl_projected_record(
    record: DynamicSchemaUFLProjectedRecord,
) -> dict[str, object]:
    return {
        "subject_key": record.subject_key,
        "required_missing_field_keys": list(record.required_missing_field_keys),
        "issue_count": record.issue_count,
        "fields": [
            serialize_dynamic_schema_ufl_projected_field(field) for field in record.fields
        ],
    }


def _serialize_ufl_fact(fact: UFLFactSnapshot) -> dict[str, object]:
    return serialize_dynamic_schema_ufl_fact(fact)


def _serialize_projected_field(field: DynamicSchemaUFLProjectedField) -> dict[str, object]:
    return serialize_dynamic_schema_ufl_projected_field(field)


def _serialize_record(record: DynamicSchemaUFLProjectedRecord) -> dict[str, object]:
    return serialize_dynamic_schema_ufl_projected_record(record)


def _match_field_facts(
    *,
    field: dynamic_schema_projection_service.DynamicSchemaFieldDefinitionSnapshot,
    facts: Sequence[UFLFactSnapshot],
) -> tuple[UFLFactSnapshot, ...]:
    predicate_facts = tuple(
        fact for fact in facts if fact.predicate_key == field.predicate_key
    )
    if field.scope_key is not None:
        return tuple(fact for fact in predicate_facts if fact.scope_key == field.scope_key)
    if field.cardinality == DynamicSchemaFieldCardinality.ONE.value:
        return tuple(fact for fact in predicate_facts if fact.scope_key is None)
    return predicate_facts


def _is_type_compatible(
    *,
    expected_value_type: str,
    facts: Sequence[UFLFactSnapshot],
) -> bool:
    if expected_value_type == DynamicSchemaFieldValueType.ANY.value:
        return True
    return all(
        value_group.value_type == expected_value_type
        for fact in facts
        for value_group in fact.value_groups
    )


def _build_projected_field(
    *,
    field: dynamic_schema_projection_service.DynamicSchemaFieldDefinitionSnapshot,
    subject_facts: Sequence[UFLFactSnapshot],
) -> DynamicSchemaUFLProjectedField:
    matched_facts = _match_field_facts(field=field, facts=subject_facts)
    semantic_value_count = sum(
        len(fact.value_groups) for fact in matched_facts
    )
    is_missing = len(matched_facts) == 0
    type_compatible = _is_type_compatible(
        expected_value_type=field.expected_value_type,
        facts=matched_facts,
    )

    issues: list[str] = []
    if is_missing and field.is_required:
        issues.append("required_missing")
    if (
        field.cardinality == DynamicSchemaFieldCardinality.ONE.value
        and len(matched_facts) > 1
    ):
        issues.append("cardinality_one_multiple_facts")
    if (
        field.cardinality == DynamicSchemaFieldCardinality.ONE.value
        and semantic_value_count > 1
    ):
        issues.append("cardinality_one_multiple_semantic_values")
    if not type_compatible:
        issues.append("value_type_mismatch")
    ordered_issues = tuple(
        issue
        for issue in _DYNAMIC_SCHEMA_UFL_PROJECTION_ISSUE_ORDER
        if issue in issues
    )
    return DynamicSchemaUFLProjectedField(
        field_id=field.field_id,
        schema_version_id=field.schema_version_id,
        field_key=field.field_key,
        label=field.label,
        description=field.description,
        predicate_key=field.predicate_key,
        scope_key=field.scope_key,
        expected_value_type=field.expected_value_type,
        cardinality=field.cardinality,
        is_required=field.is_required,
        is_title=field.is_title,
        is_summary=field.is_summary,
        is_hidden=field.is_hidden,
        group_key=field.group_key,
        display_order=field.display_order,
        display_config=field.display_config,
        validation_rules=field.validation_rules,
        created_at=field.created_at,
        matched_facts=matched_facts,
        matched_fact_count=len(matched_facts),
        semantic_value_count=semantic_value_count,
        is_missing=is_missing,
        type_compatible=type_compatible,
        issues=ordered_issues,
    )


def authenticate_dynamic_schema_ufl_projected_field(
    field: DynamicSchemaUFLProjectedField,
    *,
    record_subject_key: str,
    subject_kind: str | None,
) -> DynamicSchemaUFLProjectedField:
    _require_projection_uuid(field.field_id)
    _require_projection_uuid(field.schema_version_id)
    if isinstance(field.display_order, bool) or not isinstance(field.display_order, int):
        raise DynamicSchemaUFLProjectionInvariantError(
            "dynamic_schema_ufl_projection_projection_invalid"
        )
    if field.cardinality not in {
        DynamicSchemaFieldCardinality.ONE.value,
        DynamicSchemaFieldCardinality.MANY.value,
    }:
        raise DynamicSchemaUFLProjectionInvariantError(
            "dynamic_schema_ufl_projection_projection_invalid"
        )
    if field.expected_value_type not in {
        value_type.value for value_type in DynamicSchemaFieldValueType
    }:
        raise DynamicSchemaUFLProjectionInvariantError(
            "dynamic_schema_ufl_projection_projection_invalid"
        )
    matched_fact_count = _require_projection_count(field.matched_fact_count)
    semantic_value_count = _require_projection_count(field.semantic_value_count)
    field_definition = dynamic_schema_projection_service.DynamicSchemaFieldDefinitionSnapshot(
        field_id=field.field_id,
        schema_version_id=field.schema_version_id,
        field_key=field.field_key,
        label=field.label,
        description=field.description,
        predicate_key=field.predicate_key,
        scope_key=field.scope_key,
        expected_value_type=field.expected_value_type,
        cardinality=field.cardinality,
        is_required=field.is_required,
        is_title=field.is_title,
        is_summary=field.is_summary,
        is_hidden=field.is_hidden,
        group_key=field.group_key,
        display_order=field.display_order,
        display_config=field.display_config,
        validation_rules=field.validation_rules,
        created_at=field.created_at,
    )
    seen_fact_ids: set[uuid.UUID] = set()
    for fact in field.matched_facts:
        _require_projection_uuid(fact.fact_id)
        if fact.fact_id in seen_fact_ids:
            raise DynamicSchemaUFLProjectionInvariantError(
                "dynamic_schema_ufl_projection_projection_invalid"
            )
        seen_fact_ids.add(fact.fact_id)
        if fact.subject_key != record_subject_key:
            raise DynamicSchemaUFLProjectionInvariantError(
                "dynamic_schema_ufl_projection_projection_invalid"
            )
        if subject_kind is not None and fact.subject_kind != subject_kind:
            raise DynamicSchemaUFLProjectionInvariantError(
                "dynamic_schema_ufl_projection_projection_invalid"
            )
    expected_field = _build_projected_field(
        field=field_definition,
        subject_facts=field.matched_facts,
    )
    if expected_field.matched_facts != field.matched_facts:
        raise DynamicSchemaUFLProjectionInvariantError(
            "dynamic_schema_ufl_projection_projection_invalid"
        )
    if expected_field.matched_fact_count != matched_fact_count:
        raise DynamicSchemaUFLProjectionInvariantError(
            "dynamic_schema_ufl_projection_projection_invalid"
        )
    if expected_field.semantic_value_count != semantic_value_count:
        raise DynamicSchemaUFLProjectionInvariantError(
            "dynamic_schema_ufl_projection_projection_invalid"
        )
    if expected_field.is_missing != field.is_missing:
        raise DynamicSchemaUFLProjectionInvariantError(
            "dynamic_schema_ufl_projection_projection_invalid"
        )
    if expected_field.type_compatible != field.type_compatible:
        raise DynamicSchemaUFLProjectionInvariantError(
            "dynamic_schema_ufl_projection_projection_invalid"
        )
    if expected_field.issues != field.issues:
        raise DynamicSchemaUFLProjectionInvariantError(
            "dynamic_schema_ufl_projection_projection_invalid"
        )
    return field


def _build_record(
    *,
    subject_key: str,
    fields: Sequence[dynamic_schema_projection_service.DynamicSchemaFieldDefinitionSnapshot],
    subject_facts: Sequence[UFLFactSnapshot],
) -> DynamicSchemaUFLProjectedRecord:
    projected_fields = tuple(
        _build_projected_field(field=field, subject_facts=subject_facts)
        for field in fields
    )
    required_missing_field_keys = tuple(
        field.field_key
        for field in projected_fields
        if "required_missing" in field.issues
    )
    issue_count = sum(len(field.issues) for field in projected_fields)
    return DynamicSchemaUFLProjectedRecord(
        subject_key=subject_key,
        fields=projected_fields,
        required_missing_field_keys=required_missing_field_keys,
        issue_count=issue_count,
    )


def _build_manifest_hash(
    *,
    projection: DynamicSchemaUFLProjection,
    subject_keys_filter: list[str] | None,
) -> str:
    return duplicate_grouping_service.hash_deterministic_payload(
        {
            "project_id": str(projection.project_id),
            "schema_id": str(projection.schema_id),
            "schema_version_id": str(projection.schema_version_id),
            "orchestration_id": str(projection.orchestration_id),
            "extraction_run_id": str(projection.extraction_run_id),
            "schema_definition_manifest_hash": projection.schema_definition_manifest_hash,
            "ufl_source_manifest_hash": projection.ufl_source_manifest_hash,
            "comparison_quality": projection.comparison_quality,
            "subject_kind": projection.subject_kind,
            "subject_keys_filter": subject_keys_filter,
            "algorithm": {
                "name": projection.algorithm_name,
                "version": projection.algorithm_version,
            },
            "counts": {
                "record_count": projection.record_count,
                "projected_field_count": projection.projected_field_count,
                "required_missing_count": projection.required_missing_count,
                "issue_count": projection.issue_count,
            },
            "records": [
                serialize_dynamic_schema_ufl_projected_record(record)
                for record in projection.records
            ],
        }
    )


def authenticate_dynamic_schema_ufl_projection(
    projection: DynamicSchemaUFLProjection,
    *,
    subject_keys: object,
) -> DynamicSchemaUFLProjection:
    if not isinstance(projection, DynamicSchemaUFLProjection):
        raise DynamicSchemaUFLProjectionInvariantError(
            "dynamic_schema_ufl_projection_projection_invalid"
        )
    normalized_subject_keys = normalize_dynamic_schema_ufl_subject_keys(subject_keys)
    _require_projection_uuid(projection.project_id)
    _require_projection_uuid(projection.schema_id)
    _require_projection_uuid(projection.schema_version_id)
    _require_projection_uuid(projection.orchestration_id)
    _require_projection_uuid(projection.extraction_run_id)
    _require_sha256(
        projection.schema_definition_manifest_hash,
        error_code="dynamic_schema_ufl_projection_projection_invalid",
    )
    _require_sha256(
        projection.ufl_source_manifest_hash,
        error_code="dynamic_schema_ufl_projection_projection_invalid",
    )
    _require_sha256(
        projection.projection_manifest_hash,
        error_code="dynamic_schema_ufl_projection_projection_invalid",
    )
    if projection.algorithm_name != DYNAMIC_SCHEMA_UFL_PROJECTION_ALGORITHM_NAME:
        raise DynamicSchemaUFLProjectionInvariantError(
            "dynamic_schema_ufl_projection_projection_invalid"
        )
    if projection.algorithm_version != DYNAMIC_SCHEMA_UFL_PROJECTION_ALGORITHM_VERSION:
        raise DynamicSchemaUFLProjectionInvariantError(
            "dynamic_schema_ufl_projection_projection_invalid"
        )
    if projection.comparison_quality not in {"complete", "partial"}:
        raise DynamicSchemaUFLProjectionInvariantError(
            "dynamic_schema_ufl_projection_projection_invalid"
        )
    record_count = _require_projection_count(projection.record_count)
    projected_field_count = _require_projection_count(projection.projected_field_count)
    required_missing_count = _require_projection_count(projection.required_missing_count)
    issue_count = _require_projection_count(projection.issue_count)

    record_subject_keys: list[str] = []
    seen_record_subject_keys: set[str] = set()
    recomputed_projected_field_count = 0
    recomputed_required_missing_count = 0
    recomputed_issue_count = 0
    for record in projection.records:
        record_subject_key = _normalize_subject_key(record.subject_key)
        if record.subject_key != record_subject_key:
            raise DynamicSchemaUFLProjectionInvariantError(
                "dynamic_schema_ufl_projection_projection_invalid"
            )
        if record_subject_key in seen_record_subject_keys:
            raise DynamicSchemaUFLProjectionInvariantError(
                "dynamic_schema_ufl_projection_projection_invalid"
            )
        seen_record_subject_keys.add(record_subject_key)
        record_subject_keys.append(record_subject_key)

        sorted_fields = tuple(
            sorted(
                record.fields,
                key=lambda field: (
                    field.display_order,
                    field.field_key,
                    field.field_id,
                ),
            )
        )
        if record.fields != sorted_fields:
            raise DynamicSchemaUFLProjectionInvariantError(
                "dynamic_schema_ufl_projection_projection_invalid"
            )
        record_issue_count = _require_projection_count(record.issue_count)
        seen_field_keys: set[str] = set()
        recomputed_record_required_missing_field_keys: list[str] = []
        recomputed_record_issue_count = 0
        for field in record.fields:
            if field.field_key in seen_field_keys:
                raise DynamicSchemaUFLProjectionInvariantError(
                    "dynamic_schema_ufl_projection_projection_invalid"
                )
            seen_field_keys.add(field.field_key)
            authenticate_dynamic_schema_ufl_projected_field(
                field,
                record_subject_key=record.subject_key,
                subject_kind=projection.subject_kind,
            )
            recomputed_projected_field_count += 1
            recomputed_record_issue_count += len(field.issues)
            if "required_missing" in field.issues:
                recomputed_record_required_missing_field_keys.append(field.field_key)

        if record.required_missing_field_keys != tuple(
            recomputed_record_required_missing_field_keys
        ):
            raise DynamicSchemaUFLProjectionInvariantError(
                "dynamic_schema_ufl_projection_projection_invalid"
            )
        if record_issue_count != recomputed_record_issue_count:
            raise DynamicSchemaUFLProjectionInvariantError(
                "dynamic_schema_ufl_projection_projection_invalid"
            )
        recomputed_required_missing_count += len(record.required_missing_field_keys)
        recomputed_issue_count += record_issue_count

    if normalized_subject_keys is None:
        if tuple(record_subject_keys) != tuple(sorted(record_subject_keys)):
            raise DynamicSchemaUFLProjectionInvariantError(
                "dynamic_schema_ufl_projection_projection_invalid"
            )
    elif tuple(record_subject_keys) != tuple(normalized_subject_keys):
        raise DynamicSchemaUFLProjectionInvariantError(
            "dynamic_schema_ufl_projection_projection_invalid"
        )

    if record_count != len(projection.records):
        raise DynamicSchemaUFLProjectionInvariantError(
            "dynamic_schema_ufl_projection_projection_invalid"
        )
    if projected_field_count != recomputed_projected_field_count:
        raise DynamicSchemaUFLProjectionInvariantError(
            "dynamic_schema_ufl_projection_projection_invalid"
        )
    if required_missing_count != recomputed_required_missing_count:
        raise DynamicSchemaUFLProjectionInvariantError(
            "dynamic_schema_ufl_projection_projection_invalid"
        )
    if issue_count != recomputed_issue_count:
        raise DynamicSchemaUFLProjectionInvariantError(
            "dynamic_schema_ufl_projection_projection_invalid"
        )
    if projection.projection_manifest_hash != _build_manifest_hash(
        projection=projection,
        subject_keys_filter=normalized_subject_keys,
    ):
        raise DynamicSchemaUFLProjectionInvariantError(
            "dynamic_schema_ufl_projection_projection_invalid"
        )
    return projection


async def project_orchestration_ufl_to_dynamic_schema(
    session_factory: Callable[[], AsyncSession],
    *,
    project_id: uuid.UUID,
    schema_id: uuid.UUID,
    schema_version_id: uuid.UUID,
    orchestration_id: uuid.UUID,
    subject_keys: list[str] | None = None,
) -> DynamicSchemaUFLProjection:
    project_id = _require_uuid(project_id, field_name="project_id")
    schema_id = _require_uuid(schema_id, field_name="schema_id")
    schema_version_id = _require_uuid(schema_version_id, field_name="schema_version_id")
    orchestration_id = _require_uuid(orchestration_id, field_name="orchestration_id")
    normalized_subject_keys = normalize_dynamic_schema_ufl_subject_keys(subject_keys)

    schema_snapshot = await dynamic_schema_projection_service.get_dynamic_schema_definition_snapshot(
        session_factory,
        project_id=project_id,
        schema_id=schema_id,
        schema_version_id=schema_version_id,
    )
    ufl_snapshot = await ufl_fact_snapshot_service.get_orchestration_ufl_fact_snapshot(
        session_factory,
        project_id=project_id,
        orchestration_id=orchestration_id,
    )
    _validate_schema_snapshot(
        project_id=project_id,
        schema_id=schema_id,
        schema_version_id=schema_version_id,
        snapshot=schema_snapshot,
    )
    _validate_ufl_snapshot(
        project_id=project_id,
        orchestration_id=orchestration_id,
        snapshot=ufl_snapshot,
    )

    subject_facts_by_key: dict[str, list[UFLFactSnapshot]] = {}
    for fact in ufl_snapshot.facts:
        if fact.subject_kind != schema_snapshot.subject_kind:
            continue
        subject_facts_by_key.setdefault(fact.subject_key, []).append(fact)

    ordered_subject_keys = (
        sorted(subject_facts_by_key.keys())
        if normalized_subject_keys is None
        else list(normalized_subject_keys)
    )
    records = tuple(
        _build_record(
            subject_key=subject_key,
            fields=schema_snapshot.fields,
            subject_facts=tuple(subject_facts_by_key.get(subject_key, ())),
        )
        for subject_key in ordered_subject_keys
    )
    required_missing_count = sum(
        len(record.required_missing_field_keys) for record in records
    )
    issue_count = sum(record.issue_count for record in records)
    projection = DynamicSchemaUFLProjection(
        project_id=project_id,
        schema_id=schema_snapshot.schema_id,
        schema_version_id=schema_snapshot.schema_version_id,
        orchestration_id=ufl_snapshot.orchestration_id,
        extraction_run_id=ufl_snapshot.extraction_run_id,
        schema_definition_manifest_hash=schema_snapshot.definition_manifest_hash,
        ufl_source_manifest_hash=ufl_snapshot.source_manifest_hash,
        comparison_quality=ufl_snapshot.comparison_quality,
        subject_kind=schema_snapshot.subject_kind,
        algorithm_name=DYNAMIC_SCHEMA_UFL_PROJECTION_ALGORITHM_NAME,
        algorithm_version=DYNAMIC_SCHEMA_UFL_PROJECTION_ALGORITHM_VERSION,
        record_count=len(records),
        projected_field_count=len(records) * len(schema_snapshot.fields),
        required_missing_count=required_missing_count,
        issue_count=issue_count,
        records=records,
        projection_manifest_hash="",
    )
    return authenticate_dynamic_schema_ufl_projection(
        DynamicSchemaUFLProjection(
            project_id=projection.project_id,
            schema_id=projection.schema_id,
            schema_version_id=projection.schema_version_id,
            orchestration_id=projection.orchestration_id,
            extraction_run_id=projection.extraction_run_id,
            schema_definition_manifest_hash=projection.schema_definition_manifest_hash,
            ufl_source_manifest_hash=projection.ufl_source_manifest_hash,
            comparison_quality=projection.comparison_quality,
            subject_kind=projection.subject_kind,
            algorithm_name=projection.algorithm_name,
            algorithm_version=projection.algorithm_version,
            record_count=projection.record_count,
            projected_field_count=projection.projected_field_count,
            required_missing_count=projection.required_missing_count,
            issue_count=projection.issue_count,
            records=projection.records,
            projection_manifest_hash=_build_manifest_hash(
                projection=projection,
                subject_keys_filter=normalized_subject_keys,
            ),
        ),
        subject_keys=subject_keys,
    )
