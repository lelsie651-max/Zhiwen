from __future__ import annotations

from collections import defaultdict
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dynamic_schema import (
    DynamicSchema,
    DynamicSchemaField,
    DynamicSchemaFieldCardinality,
    DynamicSchemaFieldValueType,
    DynamicSchemaStatus,
    DynamicSchemaVersion,
    DynamicSchemaVersionStatus,
)
from app.models.fact import Fact, FactValueStatus
from app.repositories import dynamic_schema_projection as projection_repository
from app.schemas.dynamic_schema_projection import (
    DynamicSchemaProjection,
    ProjectedEvidence,
    ProjectedField,
    ProjectedRecord,
    ProjectedValue,
)


class DynamicSchemaProjectionError(Exception):
    """Raised when a dynamic schema projection cannot be built."""


class DynamicSchemaProjectionNotFoundError(DynamicSchemaProjectionError):
    """Raised when the target schema or version is not found."""


class ProjectionStateCorruptionError(DynamicSchemaProjectionError):
    """Raised when fact or schema state violates projection invariants."""


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
