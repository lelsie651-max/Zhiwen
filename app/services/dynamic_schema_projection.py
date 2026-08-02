from __future__ import annotations

from collections import defaultdict
from dataclasses import is_dataclass
from datetime import date, datetime, time
import math
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence
import uuid

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dynamic_schema import (
    DynamicSchema,
    DynamicSchemaField,
    DynamicSchemaFieldCardinality,
    DynamicSchemaFieldValueType,
    DynamicSchemaStatus,
    DynamicSchemaVersion,
    DynamicSchemaVersionSourceKind,
    DynamicSchemaVersionStatus,
)
from app.models.fact import Fact, FactValueStatus
from app.repositories import dynamic_schema_projection as projection_repository
from app.schemas.dynamic_schema import (
    DynamicSchemaFieldInput,
    DynamicSchemaIdentityInput,
    DynamicSchemaVersionInput,
)
from app.schemas.dynamic_schema_projection import (
    DynamicSchemaProjection,
    DynamicSchemaDefinitionSnapshot,
    DynamicSchemaFieldDefinitionSnapshot,
    ProjectedEvidence,
    ProjectedField,
    ProjectedRecord,
    ProjectedValue,
)
from app.services import fact_value_duplicate_grouping as duplicate_grouping_service


DYNAMIC_SCHEMA_DEFINITION_SNAPSHOT_ALGORITHM_NAME = "dynamic_schema_definition_snapshot"
DYNAMIC_SCHEMA_DEFINITION_SNAPSHOT_ALGORITHM_VERSION = "1.0.0"


class DynamicSchemaProjectionError(Exception):
    """Raised when a dynamic schema projection cannot be built."""


class DynamicSchemaProjectionNotFoundError(DynamicSchemaProjectionError):
    """Raised when the target schema or version is not found."""


class ProjectionStateCorruptionError(DynamicSchemaProjectionError):
    """Raised when fact or schema state violates projection invariants."""


class DynamicSchemaDefinitionSnapshotError(Exception):
    """Raised when dynamic schema definition snapshotting fails."""


class DynamicSchemaDefinitionSnapshotNotFoundError(DynamicSchemaDefinitionSnapshotError):
    """Raised when the target project, schema, or version cannot be resolved."""


class DynamicSchemaDefinitionSnapshotInvariantError(DynamicSchemaDefinitionSnapshotError):
    """Raised when stored schema definition state is internally inconsistent."""


async def project_current_dynamic_schema(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    schema_id: uuid.UUID,
    subject_keys: list[str] | None = None,
    include_hidden: bool = False,
) -> DynamicSchemaProjection:
    schema = await projection_repository.get_dynamic_schema_by_id(
        session,
        project_id=project_id,
        schema_id=schema_id,
    )
    if schema is None:
        raise DynamicSchemaProjectionNotFoundError("Schema must belong to the target project.")
    if schema.status != DynamicSchemaStatus.ACTIVE.value:
        raise DynamicSchemaProjectionError("Schema must be active for current-version projection.")
    if schema.current_version_id is None:
        raise ProjectionStateCorruptionError("Schema current_version_id is missing.")

    version = await projection_repository.get_dynamic_schema_version_with_fields_for_projection(
        session,
        schema_id=schema.id,
        version_id=schema.current_version_id,
    )
    if version is None:
        raise ProjectionStateCorruptionError("Schema current_version_id points to a missing version.")
    if version.status != DynamicSchemaVersionStatus.ACTIVE.value:
        raise ProjectionStateCorruptionError("Schema current version must be active.")

    return await _project_dynamic_schema(
        session,
        schema=schema,
        version=version,
        subject_keys=subject_keys,
        include_hidden=include_hidden,
    )


async def project_dynamic_schema_version(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    schema_id: uuid.UUID,
    version_id: uuid.UUID,
    subject_keys: list[str] | None = None,
    include_hidden: bool = False,
) -> DynamicSchemaProjection:
    schema = await projection_repository.get_dynamic_schema_by_id(
        session,
        project_id=project_id,
        schema_id=schema_id,
    )
    if schema is None:
        raise DynamicSchemaProjectionNotFoundError("Schema must belong to the target project.")
    if schema.status != DynamicSchemaStatus.ACTIVE.value:
        raise DynamicSchemaProjectionError("Schema must be active for projection.")

    version = await projection_repository.get_dynamic_schema_version_with_fields_for_projection(
        session,
        schema_id=schema.id,
        version_id=version_id,
    )
    if version is None:
        raise DynamicSchemaProjectionNotFoundError("Version must belong to the target schema.")

    return await _project_dynamic_schema(
        session,
        schema=schema,
        version=version,
        subject_keys=subject_keys,
        include_hidden=include_hidden,
    )


def _require_uuid(value: object, *, field_name: str) -> uuid.UUID:
    if not isinstance(value, uuid.UUID):
        raise DynamicSchemaDefinitionSnapshotInvariantError(
            f"dynamic_schema_definition_snapshot_{field_name}_invalid"
        )
    return value


def _require_record_uuid(value: object, *, error_code: str) -> uuid.UUID:
    if not isinstance(value, uuid.UUID):
        raise DynamicSchemaDefinitionSnapshotInvariantError(error_code)
    return value


def _require_optional_record_uuid(
    value: object | None,
    *,
    error_code: str,
) -> uuid.UUID | None:
    if value is None:
        return None
    return _require_record_uuid(value, error_code=error_code)


def _require_strict_bool(value: object, *, error_code: str) -> bool:
    if type(value) is not bool:
        raise DynamicSchemaDefinitionSnapshotInvariantError(error_code)
    return value


def _require_strict_int(value: object, *, error_code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DynamicSchemaDefinitionSnapshotInvariantError(error_code)
    return value


def _require_aware_datetime(
    value: object,
    *,
    error_code: str,
) -> datetime:
    if not isinstance(value, datetime):
        raise DynamicSchemaDefinitionSnapshotInvariantError(error_code)
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise DynamicSchemaDefinitionSnapshotInvariantError(error_code)
    return value


def _require_optional_aware_datetime(
    value: object | None,
    *,
    error_code: str,
) -> datetime | None:
    if value is None:
        return None
    return _require_aware_datetime(value, error_code=error_code)


def _freeze_json_value(value: Any) -> object:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DynamicSchemaDefinitionSnapshotInvariantError(
                "dynamic_schema_definition_snapshot_json_config_invalid"
            )
        return value
    if isinstance(value, Mapping):
        normalized_items: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise DynamicSchemaDefinitionSnapshotInvariantError(
                    "dynamic_schema_definition_snapshot_json_config_invalid"
                )
            normalized_items[key] = _freeze_json_value(item)
        return MappingProxyType(normalized_items)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json_value(item) for item in value)
    if isinstance(value, (bytes, bytearray, datetime, date, time)):
        raise DynamicSchemaDefinitionSnapshotInvariantError(
            "dynamic_schema_definition_snapshot_json_config_invalid"
        )
    if is_dataclass(value):
        raise DynamicSchemaDefinitionSnapshotInvariantError(
            "dynamic_schema_definition_snapshot_json_config_invalid"
        )
    raise DynamicSchemaDefinitionSnapshotInvariantError(
        "dynamic_schema_definition_snapshot_json_config_invalid"
    )


def _validate_json_config(value: object) -> object:
    frozen = _freeze_json_value(value)
    try:
        duplicate_grouping_service.canonicalize_deterministic_payload(frozen)
    except (
        duplicate_grouping_service.CrossBatchDuplicateGroupingError,
        duplicate_grouping_service.CrossBatchDuplicateGroupingInvariantError,
        TypeError,
        ValueError,
    ):
        raise DynamicSchemaDefinitionSnapshotInvariantError(
            "dynamic_schema_definition_snapshot_json_config_invalid"
        ) from None
    return frozen


def _validate_mapping_json_config(value: object) -> object:
    if not isinstance(value, Mapping):
        raise DynamicSchemaDefinitionSnapshotInvariantError(
            "dynamic_schema_definition_snapshot_json_config_invalid"
        )
    if any(not isinstance(key, str) for key in value.keys()):
        raise DynamicSchemaDefinitionSnapshotInvariantError(
            "dynamic_schema_definition_snapshot_json_config_invalid"
        )
    frozen = _validate_json_config(value)
    if not isinstance(frozen, Mapping):
        raise DynamicSchemaDefinitionSnapshotInvariantError(
            "dynamic_schema_definition_snapshot_json_config_invalid"
        )
    return frozen


def _build_manifest_hash(
    *,
    snapshot: DynamicSchemaDefinitionSnapshot,
) -> str:
    return duplicate_grouping_service.hash_deterministic_payload(
        {
            "project_id": snapshot.project_id,
            "schema_id": snapshot.schema_id,
            "schema_key": snapshot.schema_key,
            "name": snapshot.name,
            "subject_kind": snapshot.subject_kind,
            "description": snapshot.description,
            "schema_status": snapshot.schema_status,
            "schema_version_id": snapshot.schema_version_id,
            "version_no": snapshot.version_no,
            "version_status": snapshot.version_status,
            "source_kind": snapshot.source_kind,
            "summary": snapshot.summary,
            "layout_config": snapshot.layout_config,
            "created_by_id": snapshot.created_by_id,
            "activated_by_id": snapshot.activated_by_id,
            "activated_at": snapshot.activated_at,
            "is_current": snapshot.is_current,
            "algorithm_name": snapshot.algorithm_name,
            "algorithm_version": snapshot.algorithm_version,
            "field_count": snapshot.field_count,
            "fields": [
                {
                    "field_id": field.field_id,
                    "schema_version_id": field.schema_version_id,
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
                    "created_at": field.created_at,
                }
                for field in snapshot.fields
            ],
        }
    )


async def _project_dynamic_schema(
    session: AsyncSession,
    *,
    schema: DynamicSchema,
    version: DynamicSchemaVersion,
    subject_keys: list[str] | None,
    include_hidden: bool,
) -> DynamicSchemaProjection:
    ordered_fields = sorted(version.fields, key=lambda field: (field.display_order, field.field_key))
    if not ordered_fields:
        raise DynamicSchemaProjectionError("Schema version must contain at least one field.")

    visible_fields = ordered_fields if include_hidden else [field for field in ordered_fields if not field.is_hidden]
    predicate_keys = sorted({field.predicate_key for field in ordered_fields})
    facts = await projection_repository.list_facts_for_projection(
        session,
        project_id=schema.project_id,
        subject_kind=schema.subject_kind,
        predicate_keys=predicate_keys,
        subject_keys=subject_keys,
    )
    _validate_facts_for_projection(facts)

    facts_by_subject: dict[str, list[Fact]] = defaultdict(list)
    for fact in facts:
        facts_by_subject[fact.subject_key].append(fact)

    if subject_keys is None:
        ordered_subject_keys = sorted(facts_by_subject.keys())
    else:
        ordered_subject_keys = list(subject_keys)

    records = [
        _build_projected_record(
            subject_key=subject_key,
            fields=visible_fields,
            facts=facts_by_subject.get(subject_key, []),
        )
        for subject_key in ordered_subject_keys
    ]

    return DynamicSchemaProjection(
        schema_id=schema.id,
        schema_key=schema.schema_key,
        schema_name=schema.name,
        schema_status=schema.status,
        version_id=version.id,
        version_no=version.version_no,
        version_status=version.status,
        version_source_kind=version.source_kind,
        subject_kind=schema.subject_kind,
        records=records,
        warnings=[],
    )


def _validate_facts_for_projection(facts: list[Fact]) -> None:
    for fact in facts:
        if fact.current_value_id is None:
            raise ProjectionStateCorruptionError("Fact current_value_id must be present for projection.")
        current_value = fact.current_value
        if current_value is None:
            raise ProjectionStateCorruptionError("Fact current_value relationship must be explicitly loaded.")
        if current_value.fact_id != fact.id:
            raise ProjectionStateCorruptionError("Fact current value must belong to the same fact.")
        if current_value.status != FactValueStatus.ACCEPTED.value:
            raise ProjectionStateCorruptionError("Fact current value must be accepted.")


def _build_projected_record(
    *,
    subject_key: str,
    fields: list[DynamicSchemaField],
    facts: list[Fact],
) -> ProjectedRecord:
    projected_fields: list[ProjectedField] = []
    required_missing_field_keys: list[str] = []

    for field in fields:
        matched_facts = _match_field_facts(field=field, facts=facts)
        issues: list[str] = []
        values = [_build_projected_value(fact) for fact in matched_facts]
        incompatible_count = 0
        for value in values:
            if field.expected_value_type == DynamicSchemaFieldValueType.ANY.value:
                value.type_compatible = True
            else:
                value.type_compatible = value.value_type == field.expected_value_type
            if not value.type_compatible:
                incompatible_count += 1
        if incompatible_count > 0:
            issues.append(f"{incompatible_count} value(s) have incompatible value_type")

        is_missing = len(values) == 0
        if is_missing and field.is_required:
            required_missing_field_keys.append(field.field_key)

        projected_fields.append(
            ProjectedField(
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
                values=values,
                is_missing=is_missing,
                issues=issues,
            )
        )

    return ProjectedRecord(
        subject_key=subject_key,
        fields=projected_fields,
        required_missing_field_keys=required_missing_field_keys,
    )


def _match_field_facts(
    *,
    field: DynamicSchemaField,
    facts: list[Fact],
) -> list[Fact]:
    predicate_facts = [fact for fact in facts if fact.predicate_key == field.predicate_key]

    if field.scope_key is not None:
        matched = [fact for fact in predicate_facts if fact.scope_key == field.scope_key]
    elif field.cardinality == DynamicSchemaFieldCardinality.ONE.value:
        matched = [fact for fact in predicate_facts if fact.scope_key is None]
    else:
        matched = list(predicate_facts)

    if field.cardinality == DynamicSchemaFieldCardinality.ONE.value:
        if len(matched) > 1:
            raise ProjectionStateCorruptionError(
                f"Field '{field.field_key}' expects one fact but matched multiple active facts."
            )
        return matched

    return sorted(
        matched,
        key=lambda fact: (
            fact.scope_key is not None,
            fact.scope_key or "",
            fact.identity_hash,
        ),
    )


def _build_projected_value(fact: Fact) -> ProjectedValue:
    current_value = fact.current_value
    assert current_value is not None
    evidence_links = sorted(
        current_value.evidence_links,
        key=lambda link: (link.source_order, link.id),
    )
    return ProjectedValue(
        fact_id=fact.id,
        fact_value_id=current_value.id,
        scope_key=fact.scope_key,
        value_type=current_value.value_type,
        value_json=current_value.value_json,
        normalized_value_text=current_value.normalized_value_text,
        language_code=current_value.language_code,
        source_kind=current_value.source_kind,
        confidence=current_value.confidence,
        type_compatible=True,
        evidences=[
            ProjectedEvidence(
                evidence_id=link.evidence.id,
                role=link.role,
                is_primary=link.is_primary,
                source_order=link.source_order,
                excerpt=link.evidence.excerpt,
                block_id=link.evidence.block_id,
                location_key=link.evidence.block.location_key,
                page_no=link.evidence.block.page_no,
                start_line=link.evidence.block.start_line,
                end_line=link.evidence.block.end_line,
                start_offset=link.evidence.start_offset,
                end_offset=link.evidence.end_offset,
            )
            for link in evidence_links
        ],
    )


def _validate_schema_record(schema: DynamicSchema) -> DynamicSchemaIdentityInput:
    _require_record_uuid(
        schema.id,
        error_code="dynamic_schema_definition_snapshot_schema_invalid",
    )
    _require_record_uuid(
        schema.project_id,
        error_code="dynamic_schema_definition_snapshot_schema_invalid",
    )
    _require_optional_record_uuid(
        schema.current_version_id,
        error_code="dynamic_schema_definition_snapshot_schema_invalid",
    )
    try:
        identity = DynamicSchemaIdentityInput(
            schema_key=schema.schema_key,
            name=schema.name,
            subject_kind=schema.subject_kind,
            description=schema.description,
        )
    except ValidationError:
        raise DynamicSchemaDefinitionSnapshotInvariantError(
            "dynamic_schema_definition_snapshot_schema_invalid"
        ) from None
    if schema.status not in {
        DynamicSchemaStatus.ACTIVE.value,
        DynamicSchemaStatus.ARCHIVED.value,
    }:
        raise DynamicSchemaDefinitionSnapshotInvariantError(
            "dynamic_schema_definition_snapshot_schema_invalid"
        )
    if (
        identity.schema_key != schema.schema_key
        or identity.name != schema.name
        or identity.subject_kind != schema.subject_kind
        or identity.description != schema.description
    ):
        raise DynamicSchemaDefinitionSnapshotInvariantError(
            "dynamic_schema_definition_snapshot_schema_invalid"
        )
    return identity


def _validate_version_record(
    version: DynamicSchemaVersion,
    *,
    field_inputs: list[DynamicSchemaFieldInput],
    layout_config: object,
) -> DynamicSchemaVersionInput:
    _require_record_uuid(
        version.id,
        error_code="dynamic_schema_definition_snapshot_version_invalid",
    )
    _require_record_uuid(
        version.schema_id,
        error_code="dynamic_schema_definition_snapshot_version_invalid",
    )
    _require_optional_record_uuid(
        version.created_by_id,
        error_code="dynamic_schema_definition_snapshot_version_invalid",
    )
    _require_optional_record_uuid(
        version.activated_by_id,
        error_code="dynamic_schema_definition_snapshot_version_invalid",
    )
    _require_optional_aware_datetime(
        version.activated_at,
        error_code="dynamic_schema_definition_snapshot_version_invalid",
    )
    if _require_strict_int(
        version.version_no,
        error_code="dynamic_schema_definition_snapshot_version_invalid",
    ) <= 0:
        raise DynamicSchemaDefinitionSnapshotInvariantError(
            "dynamic_schema_definition_snapshot_version_invalid"
        )
    if version.status not in {
        DynamicSchemaVersionStatus.DRAFT.value,
        DynamicSchemaVersionStatus.PROPOSED.value,
        DynamicSchemaVersionStatus.ACTIVE.value,
        DynamicSchemaVersionStatus.RETIRED.value,
    }:
        raise DynamicSchemaDefinitionSnapshotInvariantError(
            "dynamic_schema_definition_snapshot_version_invalid"
        )
    if version.source_kind not in {
        DynamicSchemaVersionSourceKind.AI.value,
        DynamicSchemaVersionSourceKind.HUMAN.value,
        DynamicSchemaVersionSourceKind.IMPORT.value,
        DynamicSchemaVersionSourceKind.SYSTEM.value,
    }:
        raise DynamicSchemaDefinitionSnapshotInvariantError(
            "dynamic_schema_definition_snapshot_version_invalid"
        )
    if version.status in {
        DynamicSchemaVersionStatus.DRAFT.value,
        DynamicSchemaVersionStatus.PROPOSED.value,
    }:
        if version.activated_by_id is not None or version.activated_at is not None:
            raise DynamicSchemaDefinitionSnapshotInvariantError(
                "dynamic_schema_definition_snapshot_version_invalid"
            )
    elif version.activated_by_id is None or version.activated_at is None:
        raise DynamicSchemaDefinitionSnapshotInvariantError(
            "dynamic_schema_definition_snapshot_version_invalid"
        )
    if (
        version.source_kind == DynamicSchemaVersionSourceKind.HUMAN.value
        and version.created_by_id is None
    ):
        raise DynamicSchemaDefinitionSnapshotInvariantError(
            "dynamic_schema_definition_snapshot_version_invalid"
        )
    try:
        version_input = DynamicSchemaVersionInput(
            summary=version.summary,
            layout_config=layout_config,
            fields=field_inputs,
        )
    except ValidationError:
        raise DynamicSchemaDefinitionSnapshotInvariantError(
            "dynamic_schema_definition_snapshot_version_invalid"
        ) from None
    if version_input.summary != version.summary:
        raise DynamicSchemaDefinitionSnapshotInvariantError(
            "dynamic_schema_definition_snapshot_version_invalid"
        )
    return version_input


def _validate_field_records(
    fields: Sequence[DynamicSchemaField],
    *,
    expected_schema_version_id: uuid.UUID,
) -> tuple[list[DynamicSchemaFieldInput], tuple[DynamicSchemaFieldDefinitionSnapshot, ...]]:
    validated_field_inputs: list[DynamicSchemaFieldInput] = []
    field_snapshots: list[DynamicSchemaFieldDefinitionSnapshot] = []
    for field in fields:
        field_id = _require_record_uuid(
            field.id,
            error_code="dynamic_schema_definition_snapshot_field_invalid",
        )
        field_schema_version_id = _require_record_uuid(
            field.schema_version_id,
            error_code="dynamic_schema_definition_snapshot_field_invalid",
        )
        if field_schema_version_id != expected_schema_version_id:
            raise DynamicSchemaDefinitionSnapshotInvariantError(
                "dynamic_schema_definition_snapshot_field_invalid"
            )
        display_config = _validate_mapping_json_config(field.display_config)
        validation_rules = _validate_mapping_json_config(field.validation_rules)
        expected_value_type = field.expected_value_type
        cardinality = field.cardinality
        if not isinstance(expected_value_type, str) or expected_value_type not in {
            member.value for member in DynamicSchemaFieldValueType
        }:
            raise DynamicSchemaDefinitionSnapshotInvariantError(
                "dynamic_schema_definition_snapshot_field_invalid"
            )
        if not isinstance(cardinality, str) or cardinality not in {
            member.value for member in DynamicSchemaFieldCardinality
        }:
            raise DynamicSchemaDefinitionSnapshotInvariantError(
                "dynamic_schema_definition_snapshot_field_invalid"
            )
        is_required = _require_strict_bool(
            field.is_required,
            error_code="dynamic_schema_definition_snapshot_field_invalid",
        )
        is_title = _require_strict_bool(
            field.is_title,
            error_code="dynamic_schema_definition_snapshot_field_invalid",
        )
        is_summary = _require_strict_bool(
            field.is_summary,
            error_code="dynamic_schema_definition_snapshot_field_invalid",
        )
        is_hidden = _require_strict_bool(
            field.is_hidden,
            error_code="dynamic_schema_definition_snapshot_field_invalid",
        )
        display_order = _require_strict_int(
            field.display_order,
            error_code="dynamic_schema_definition_snapshot_field_invalid",
        )
        created_at = _require_aware_datetime(
            field.created_at,
            error_code="dynamic_schema_definition_snapshot_field_invalid",
        )
        try:
            field_input = DynamicSchemaFieldInput(
                field_key=field.field_key,
                label=field.label,
                description=field.description,
                predicate_key=field.predicate_key,
                scope_key=field.scope_key,
                expected_value_type=expected_value_type,
                cardinality=cardinality,
                is_required=is_required,
                is_title=is_title,
                is_summary=is_summary,
                is_hidden=is_hidden,
                group_key=field.group_key,
                display_order=display_order,
                display_config=display_config,
                validation_rules=validation_rules,
            )
        except ValidationError:
            raise DynamicSchemaDefinitionSnapshotInvariantError(
                "dynamic_schema_definition_snapshot_field_invalid"
            ) from None
        if (
            field_input.field_key != field.field_key
            or field_input.label != field.label
            or field_input.description != field.description
            or field_input.predicate_key != field.predicate_key
            or field_input.scope_key != field.scope_key
            or field_input.expected_value_type.value != expected_value_type
            or field_input.cardinality.value != cardinality
            or field_input.is_required is not is_required
            or field_input.is_title is not is_title
            or field_input.is_summary is not is_summary
            or field_input.is_hidden is not is_hidden
            or field_input.group_key != field.group_key
            or field_input.display_order != display_order
            or field_input.display_config != display_config
            or field_input.validation_rules != validation_rules
        ):
            raise DynamicSchemaDefinitionSnapshotInvariantError(
                "dynamic_schema_definition_snapshot_field_invalid"
            )
        validated_field_inputs.append(field_input)
        field_snapshots.append(
            DynamicSchemaFieldDefinitionSnapshot(
                field_id=field_id,
                schema_version_id=field_schema_version_id,
                field_key=field.field_key,
                label=field.label,
                description=field.description,
                predicate_key=field.predicate_key,
                scope_key=field.scope_key,
                expected_value_type=expected_value_type,
                cardinality=cardinality,
                is_required=is_required,
                is_title=is_title,
                is_summary=is_summary,
                is_hidden=is_hidden,
                group_key=field.group_key,
                display_order=display_order,
                display_config=display_config,
                validation_rules=validation_rules,
                created_at=created_at,
            )
        )
    ordered_field_snapshots = tuple(
        sorted(
            field_snapshots,
            key=lambda item: (item.display_order, item.field_key, item.field_id),
        )
    )
    return validated_field_inputs, ordered_field_snapshots


def _validate_current_state(
    *,
    schema: DynamicSchema,
    version: DynamicSchemaVersion,
    active_versions: Sequence[DynamicSchemaVersion],
) -> bool:
    for active_version in active_versions:
        _require_record_uuid(
            active_version.id,
            error_code="dynamic_schema_definition_snapshot_current_state_invalid",
        )
        _require_record_uuid(
            active_version.schema_id,
            error_code="dynamic_schema_definition_snapshot_current_state_invalid",
        )
        _require_optional_record_uuid(
            active_version.activated_by_id,
            error_code="dynamic_schema_definition_snapshot_current_state_invalid",
        )
        _require_optional_aware_datetime(
            active_version.activated_at,
            error_code="dynamic_schema_definition_snapshot_current_state_invalid",
        )
        if (
            active_version.schema_id != schema.id
            or active_version.status != DynamicSchemaVersionStatus.ACTIVE.value
        ):
            raise DynamicSchemaDefinitionSnapshotInvariantError(
                "dynamic_schema_definition_snapshot_current_state_invalid"
            )
    if len(active_versions) > 1:
        raise DynamicSchemaDefinitionSnapshotInvariantError(
            "dynamic_schema_definition_snapshot_current_state_invalid"
        )
    active_version = active_versions[0] if active_versions else None
    if schema.current_version_id is None:
        if active_version is not None:
            raise DynamicSchemaDefinitionSnapshotInvariantError(
                "dynamic_schema_definition_snapshot_current_state_invalid"
            )
        if version.status == DynamicSchemaVersionStatus.ACTIVE.value:
            raise DynamicSchemaDefinitionSnapshotInvariantError(
                "dynamic_schema_definition_snapshot_current_state_invalid"
            )
        return False
    if active_version is None:
        raise DynamicSchemaDefinitionSnapshotInvariantError(
            "dynamic_schema_definition_snapshot_current_state_invalid"
        )
    if active_version.id != schema.current_version_id:
        raise DynamicSchemaDefinitionSnapshotInvariantError(
            "dynamic_schema_definition_snapshot_current_state_invalid"
        )
    if active_version.status != DynamicSchemaVersionStatus.ACTIVE.value:
        raise DynamicSchemaDefinitionSnapshotInvariantError(
            "dynamic_schema_definition_snapshot_current_state_invalid"
        )
    if version.id == schema.current_version_id:
        if version.status != DynamicSchemaVersionStatus.ACTIVE.value:
            raise DynamicSchemaDefinitionSnapshotInvariantError(
                "dynamic_schema_definition_snapshot_current_state_invalid"
            )
        return True
    if version.status == DynamicSchemaVersionStatus.ACTIVE.value:
        raise DynamicSchemaDefinitionSnapshotInvariantError(
            "dynamic_schema_definition_snapshot_current_state_invalid"
        )
    return False


def _build_snapshot(
    *,
    project_id: uuid.UUID,
    schema: DynamicSchema,
    version: DynamicSchemaVersion,
    fields: Sequence[DynamicSchemaField],
    active_versions: Sequence[DynamicSchemaVersion],
) -> DynamicSchemaDefinitionSnapshot:
    _validate_schema_record(schema)
    schema_id = _require_record_uuid(
        schema.id,
        error_code="dynamic_schema_definition_snapshot_schema_invalid",
    )
    schema_project_id = _require_record_uuid(
        schema.project_id,
        error_code="dynamic_schema_definition_snapshot_schema_invalid",
    )
    if schema_project_id != project_id:
        raise DynamicSchemaDefinitionSnapshotInvariantError(
            "dynamic_schema_definition_snapshot_schema_invalid"
        )
    layout_config = _validate_json_config(version.layout_config)
    validated_field_inputs, field_snapshots = _validate_field_records(
        fields,
        expected_schema_version_id=version.id,
    )
    _validate_version_record(
        version,
        field_inputs=validated_field_inputs,
        layout_config=layout_config,
    )
    version_id = _require_record_uuid(
        version.id,
        error_code="dynamic_schema_definition_snapshot_version_invalid",
    )
    version_schema_id = _require_record_uuid(
        version.schema_id,
        error_code="dynamic_schema_definition_snapshot_version_invalid",
    )
    if version_schema_id != schema_id:
        raise DynamicSchemaDefinitionSnapshotInvariantError(
            "dynamic_schema_definition_snapshot_version_invalid"
        )
    is_current = _validate_current_state(
        schema=schema,
        version=version,
        active_versions=active_versions,
    )
    snapshot = DynamicSchemaDefinitionSnapshot(
        project_id=project_id,
        schema_id=schema_id,
        schema_key=schema.schema_key,
        name=schema.name,
        subject_kind=schema.subject_kind,
        description=schema.description,
        schema_status=schema.status,
        schema_version_id=version_id,
        version_no=version.version_no,
        version_status=version.status,
        source_kind=version.source_kind,
        summary=version.summary,
        layout_config=layout_config,
        created_by_id=version.created_by_id,
        activated_by_id=version.activated_by_id,
        activated_at=version.activated_at,
        is_current=is_current,
        algorithm_name=DYNAMIC_SCHEMA_DEFINITION_SNAPSHOT_ALGORITHM_NAME,
        algorithm_version=DYNAMIC_SCHEMA_DEFINITION_SNAPSHOT_ALGORITHM_VERSION,
        field_count=len(field_snapshots),
        fields=field_snapshots,
        definition_manifest_hash="",
    )
    return DynamicSchemaDefinitionSnapshot(
        project_id=snapshot.project_id,
        schema_id=snapshot.schema_id,
        schema_key=snapshot.schema_key,
        name=snapshot.name,
        subject_kind=snapshot.subject_kind,
        description=snapshot.description,
        schema_status=snapshot.schema_status,
        schema_version_id=snapshot.schema_version_id,
        version_no=snapshot.version_no,
        version_status=snapshot.version_status,
        source_kind=snapshot.source_kind,
        summary=snapshot.summary,
        layout_config=layout_config,
        created_by_id=snapshot.created_by_id,
        activated_by_id=snapshot.activated_by_id,
        activated_at=snapshot.activated_at,
        is_current=snapshot.is_current,
        algorithm_name=snapshot.algorithm_name,
        algorithm_version=snapshot.algorithm_version,
        field_count=snapshot.field_count,
        fields=field_snapshots,
        definition_manifest_hash=_build_manifest_hash(snapshot=snapshot),
    )


async def _get_dynamic_schema_definition_snapshot_in_session(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    schema_id: uuid.UUID,
    schema_version_id: uuid.UUID | None = None,
    current_only: bool,
) -> DynamicSchemaDefinitionSnapshot:
    project = await projection_repository.get_project_by_id(
        session,
        project_id=project_id,
    )
    if project is None:
        raise DynamicSchemaDefinitionSnapshotNotFoundError(
            "dynamic_schema_definition_snapshot_project_not_found"
        )

    schema = await projection_repository.get_dynamic_schema_by_id(
        session,
        schema_id=schema_id,
    )
    if schema is None:
        raise DynamicSchemaDefinitionSnapshotNotFoundError(
            "dynamic_schema_definition_snapshot_schema_not_found"
        )
    schema_record_id = _require_record_uuid(
        schema.id,
        error_code="dynamic_schema_definition_snapshot_schema_invalid",
    )
    schema_project_id = _require_record_uuid(
        schema.project_id,
        error_code="dynamic_schema_definition_snapshot_schema_invalid",
    )
    if schema_project_id != project_id:
        raise DynamicSchemaDefinitionSnapshotNotFoundError(
            "dynamic_schema_definition_snapshot_schema_not_found"
        )

    if current_only:
        current_version_id = _require_optional_record_uuid(
            schema.current_version_id,
            error_code="dynamic_schema_definition_snapshot_schema_invalid",
        )
        if current_version_id is None:
            raise DynamicSchemaDefinitionSnapshotInvariantError(
                "dynamic_schema_definition_snapshot_current_state_invalid"
            )
        version = await projection_repository.get_dynamic_schema_version_by_id(
            session,
            schema_version_id=current_version_id,
        )
        if version is None:
            raise DynamicSchemaDefinitionSnapshotInvariantError(
                "dynamic_schema_definition_snapshot_current_state_invalid"
            )
        _require_record_uuid(
            version.id,
            error_code="dynamic_schema_definition_snapshot_version_invalid",
        )
        version_schema_id = _require_record_uuid(
            version.schema_id,
            error_code="dynamic_schema_definition_snapshot_version_invalid",
        )
        if version_schema_id != schema_record_id:
            raise DynamicSchemaDefinitionSnapshotInvariantError(
                "dynamic_schema_definition_snapshot_current_state_invalid"
            )
    else:
        assert schema_version_id is not None
        version = await projection_repository.get_dynamic_schema_version_by_id(
            session,
            schema_version_id=schema_version_id,
        )
        if version is None:
            raise DynamicSchemaDefinitionSnapshotNotFoundError(
                "dynamic_schema_definition_snapshot_version_not_found"
            )
        _require_record_uuid(
            version.id,
            error_code="dynamic_schema_definition_snapshot_version_invalid",
        )
        version_schema_id = _require_record_uuid(
            version.schema_id,
            error_code="dynamic_schema_definition_snapshot_version_invalid",
        )
        if version_schema_id != schema_record_id:
            raise DynamicSchemaDefinitionSnapshotNotFoundError(
                "dynamic_schema_definition_snapshot_version_not_found"
            )

    fields = await projection_repository.list_dynamic_schema_fields_by_version_id(
        session,
        schema_version_id=version.id,
    )
    active_versions = await projection_repository.list_active_dynamic_schema_versions(
        session,
        schema_id=schema_record_id,
    )
    snapshot = _build_snapshot(
        project_id=project_id,
        schema=schema,
        version=version,
        fields=fields,
        active_versions=active_versions,
    )
    if current_only and snapshot.is_current is not True:
        raise DynamicSchemaDefinitionSnapshotInvariantError(
            "dynamic_schema_definition_snapshot_current_state_invalid"
        )
    return snapshot


async def get_dynamic_schema_definition_snapshot(
    session_factory: Callable[[], AsyncSession],
    *,
    project_id: uuid.UUID,
    schema_id: uuid.UUID,
    schema_version_id: uuid.UUID,
) -> DynamicSchemaDefinitionSnapshot:
    project_id = _require_uuid(project_id, field_name="project_id")
    schema_id = _require_uuid(schema_id, field_name="schema_id")
    schema_version_id = _require_uuid(schema_version_id, field_name="schema_version_id")
    async with session_factory() as session:
        try:
            return await _get_dynamic_schema_definition_snapshot_in_session(
                session,
                project_id=project_id,
                schema_id=schema_id,
                schema_version_id=schema_version_id,
                current_only=False,
            )
        finally:
            await session.rollback()


async def get_current_dynamic_schema_definition_snapshot(
    session_factory: Callable[[], AsyncSession],
    *,
    project_id: uuid.UUID,
    schema_id: uuid.UUID,
) -> DynamicSchemaDefinitionSnapshot:
    project_id = _require_uuid(project_id, field_name="project_id")
    schema_id = _require_uuid(schema_id, field_name="schema_id")
    async with session_factory() as session:
        try:
            return await _get_dynamic_schema_definition_snapshot_in_session(
                session,
                project_id=project_id,
                schema_id=schema_id,
                current_only=True,
            )
        finally:
            await session.rollback()
