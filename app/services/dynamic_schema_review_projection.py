from __future__ import annotations

import re
import unicodedata
import uuid
from collections.abc import Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.dynamic_schema_review_projection import (
    DynamicSchemaReviewProjection,
    DynamicSchemaReviewedFact,
    DynamicSchemaReviewedField,
    DynamicSchemaReviewedRecord,
)
from app.schemas.dynamic_schema_ufl_projection import DynamicSchemaUFLProjection
from app.schemas.effective_fact_value import (
    EffectiveFactValueProjection,
    EffectiveFactValueProjectionItem,
)
from app.schemas.fact import _normalize_required_text
from app.schemas.ufl_fact_snapshot import UFLFactSnapshot
import app.services.consistency_check_persistence as consistency_persistence_service
import app.services.dynamic_schema_ufl_projection as raw_projection_service
import app.services.effective_fact_value as effective_fact_value_service
import app.services.fact_value_duplicate_grouping as duplicate_grouping_service


DYNAMIC_SCHEMA_REVIEW_PROJECTION_ALGORITHM_NAME = "dynamic_schema_review_projection"
DYNAMIC_SCHEMA_REVIEW_PROJECTION_ALGORITHM_VERSION = "1.0.0"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class DynamicSchemaReviewProjectionError(Exception):
    """Base error for dynamic schema review projection failures."""


class DynamicSchemaReviewProjectionStateError(DynamicSchemaReviewProjectionError):
    """Raised when review projection inputs are invalid."""


class DynamicSchemaReviewProjectionInvariantError(DynamicSchemaReviewProjectionError):
    """Raised when authenticated source snapshots drift or conflict."""


def _require_uuid(value: object, *, field_name: str) -> uuid.UUID:
    if not isinstance(value, uuid.UUID):
        raise DynamicSchemaReviewProjectionStateError(
            f"dynamic_schema_review_projection_{field_name}_invalid"
        )
    return value


def _require_sha256(value: object, *, error_code: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise DynamicSchemaReviewProjectionInvariantError(error_code)
    return value


def _normalize_subject_key(value: object) -> str:
    if not isinstance(value, str):
        raise DynamicSchemaReviewProjectionStateError(
            "dynamic_schema_review_projection_subject_keys_invalid"
        )
    normalized = unicodedata.normalize("NFC", value)
    try:
        return _normalize_required_text(
            normalized,
            field_name="subject_key",
            max_length=255,
        )
    except ValueError:
        raise DynamicSchemaReviewProjectionStateError(
            "dynamic_schema_review_projection_subject_keys_invalid"
        ) from None


def _validate_subject_keys(subject_keys: object) -> list[str] | None:
    if subject_keys is None:
        return None
    if not isinstance(subject_keys, list):
        raise DynamicSchemaReviewProjectionStateError(
            "dynamic_schema_review_projection_subject_keys_invalid"
        )
    if len(subject_keys) > 500:
        raise DynamicSchemaReviewProjectionStateError(
            "dynamic_schema_review_projection_subject_keys_invalid"
        )
    normalized = [_normalize_subject_key(value) for value in subject_keys]
    if len(set(normalized)) != len(normalized):
        raise DynamicSchemaReviewProjectionStateError(
            "dynamic_schema_review_projection_subject_keys_invalid"
        )
    return normalized


def _validate_raw_projection(
    *,
    project_id: uuid.UUID,
    schema_id: uuid.UUID,
    schema_version_id: uuid.UUID,
    orchestration_id: uuid.UUID,
    projection: DynamicSchemaUFLProjection,
) -> None:
    if projection.project_id != project_id:
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_raw_projection_mismatch"
        )
    if projection.schema_id != schema_id or projection.schema_version_id != schema_version_id:
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_raw_projection_mismatch"
        )
    if projection.orchestration_id != orchestration_id:
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_raw_projection_mismatch"
        )
    if (
        projection.algorithm_name
        != raw_projection_service.DYNAMIC_SCHEMA_UFL_PROJECTION_ALGORITHM_NAME
        or projection.algorithm_version
        != raw_projection_service.DYNAMIC_SCHEMA_UFL_PROJECTION_ALGORITHM_VERSION
    ):
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_raw_projection_mismatch"
        )
    _require_sha256(
        projection.schema_definition_manifest_hash,
        error_code="dynamic_schema_review_projection_raw_projection_mismatch",
    )
    _require_sha256(
        projection.ufl_source_manifest_hash,
        error_code="dynamic_schema_review_projection_raw_projection_mismatch",
    )
    _require_sha256(
        projection.projection_manifest_hash,
        error_code="dynamic_schema_review_projection_raw_projection_mismatch",
    )


def _validate_authenticated_application(
    *,
    project_id: uuid.UUID,
    orchestration_id: uuid.UUID,
    context: consistency_persistence_service.AuthenticatedConsistencyCheckLedgerProjectionContext,
) -> None:
    application = context.application
    if application.project_id != project_id:
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_source_application_mismatch"
        )
    if application.orchestration_id != orchestration_id:
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_source_application_mismatch"
        )
    _require_sha256(
        application.result_manifest_hash,
        error_code="dynamic_schema_review_projection_source_application_mismatch",
    )


def _validate_effective_projection(
    *,
    project_id: uuid.UUID,
    consistency_check_application_id: uuid.UUID,
    authenticated_context: consistency_persistence_service.AuthenticatedConsistencyCheckLedgerProjectionContext,
    effective_projection: EffectiveFactValueProjection,
) -> None:
    if effective_projection.project_id != project_id:
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_effective_projection_mismatch"
        )
    if effective_projection.consistency_check_application_id != consistency_check_application_id:
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_effective_projection_mismatch"
        )
    if (
        effective_projection.consistency_check_application_id
        != authenticated_context.application.id
    ):
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_effective_projection_mismatch"
        )
    if (
        effective_projection.source_consistency_application_id
        != authenticated_context.application.consistency_application_id
    ):
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_effective_projection_mismatch"
        )
    if (
        effective_projection.result_manifest_hash
        != authenticated_context.application.result_manifest_hash
    ):
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_effective_projection_mismatch"
        )
    _require_sha256(
        effective_projection.result_manifest_hash,
        error_code="dynamic_schema_review_projection_effective_projection_mismatch",
    )

    resolved_count = sum(
        1
        for item in effective_projection.items
        if item.resolution_status == "resolved"
    )
    pending_count = sum(
        1
        for item in effective_projection.items
        if item.resolution_status in {"pending_review", "unreviewed_compatible"}
    )
    deferred_count = sum(
        1
        for item in effective_projection.items
        if item.resolution_status == "deferred"
    )
    if effective_projection.fact_count != len(effective_projection.items):
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_effective_count_mismatch"
        )
    if effective_projection.resolved_count != resolved_count:
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_effective_count_mismatch"
        )
    if effective_projection.pending_count != pending_count:
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_effective_count_mismatch"
        )
    if effective_projection.deferred_count != deferred_count:
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_effective_count_mismatch"
        )


def _index_effective_items(
    effective_projection: EffectiveFactValueProjection,
) -> dict[uuid.UUID, EffectiveFactValueProjectionItem]:
    indexed: dict[uuid.UUID, EffectiveFactValueProjectionItem] = {}
    for item in effective_projection.items:
        if item.fact_id in indexed:
            raise DynamicSchemaReviewProjectionInvariantError(
                "dynamic_schema_review_projection_effective_projection_mismatch"
            )
        indexed[item.fact_id] = item
    return indexed


def _collect_fact_value_ids(fact: UFLFactSnapshot) -> frozenset[uuid.UUID]:
    fact_value_ids: set[uuid.UUID] = set()
    for group in fact.value_groups:
        for fact_value_id in group.fact_value_ids:
            if fact_value_id in fact_value_ids:
                raise DynamicSchemaReviewProjectionInvariantError(
                    "dynamic_schema_review_projection_fact_binding_mismatch"
                )
            fact_value_ids.add(fact_value_id)
    return frozenset(fact_value_ids)


def _index_matched_facts(
    raw_projection: DynamicSchemaUFLProjection,
) -> dict[uuid.UUID, tuple[UFLFactSnapshot, frozenset[uuid.UUID]]]:
    indexed: dict[uuid.UUID, tuple[UFLFactSnapshot, frozenset[uuid.UUID]]] = {}
    for record in raw_projection.records:
        for field in record.fields:
            for fact in field.matched_facts:
                fact_value_ids = _collect_fact_value_ids(fact)
                prior = indexed.get(fact.fact_id)
                current = (fact, fact_value_ids)
                if prior is None:
                    indexed[fact.fact_id] = current
                    continue
                if prior != current:
                    raise DynamicSchemaReviewProjectionInvariantError(
                        "dynamic_schema_review_projection_fact_binding_mismatch"
                    )
    return indexed


def _validate_fact_binding(
    *,
    fact_value_ids: frozenset[uuid.UUID],
    effective_item: EffectiveFactValueProjectionItem,
) -> None:
    candidate_member_ids = tuple(
        member.fact_value_id for member in effective_item.candidate_members
    )
    if len(candidate_member_ids) != len(set(candidate_member_ids)):
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_fact_binding_mismatch"
        )
    if any(fact_value_id not in fact_value_ids for fact_value_id in candidate_member_ids):
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_fact_binding_mismatch"
        )
    if len(effective_item.effective_fact_value_ids) != len(
        set(effective_item.effective_fact_value_ids)
    ):
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_fact_binding_mismatch"
        )
    if any(
        fact_value_id not in fact_value_ids
        for fact_value_id in effective_item.effective_fact_value_ids
    ):
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_fact_binding_mismatch"
        )
    if any(
        fact_value_id not in set(candidate_member_ids)
        for fact_value_id in effective_item.effective_fact_value_ids
    ):
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_fact_binding_mismatch"
        )


def _build_reviewed_fact(
    *,
    fact: UFLFactSnapshot,
    effective_item: EffectiveFactValueProjectionItem | None,
) -> DynamicSchemaReviewedFact:
    if effective_item is None:
        return DynamicSchemaReviewedFact(
            fact=fact,
            review_state="no_consistency_candidate",
            candidate_id=None,
            assessment_id=None,
            resolution_basis="none",
            current_decision_id=None,
            current_decision_kind=None,
            effective_fact_value_ids=(),
            requires_review=False,
        )

    if effective_item.resolution_status == "resolved":
        review_state = "resolved"
        requires_review = False
    elif effective_item.resolution_status == "pending_review":
        review_state = "pending_review"
        requires_review = True
    elif effective_item.resolution_status == "deferred":
        review_state = "deferred"
        requires_review = True
    elif effective_item.resolution_status == "unreviewed_compatible":
        review_state = "unreviewed_compatible"
        requires_review = True
    else:
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_effective_projection_mismatch"
        )

    if requires_review and effective_item.effective_fact_value_ids:
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_fact_binding_mismatch"
        )

    return DynamicSchemaReviewedFact(
        fact=fact,
        review_state=review_state,
        candidate_id=effective_item.candidate_id,
        assessment_id=effective_item.assessment_id,
        resolution_basis=effective_item.resolution_basis,
        current_decision_id=effective_item.current_decision_id,
        current_decision_kind=effective_item.current_decision_kind,
        effective_fact_value_ids=effective_item.effective_fact_value_ids,
        requires_review=requires_review,
    )


def _serialize_reviewed_fact(reviewed_fact: DynamicSchemaReviewedFact) -> dict[str, object]:
    return {
        "fact": raw_projection_service._serialize_ufl_fact(reviewed_fact.fact),
        "review_state": reviewed_fact.review_state,
        "candidate_id": (
            str(reviewed_fact.candidate_id)
            if reviewed_fact.candidate_id is not None
            else None
        ),
        "assessment_id": (
            str(reviewed_fact.assessment_id)
            if reviewed_fact.assessment_id is not None
            else None
        ),
        "resolution_basis": reviewed_fact.resolution_basis,
        "current_decision_id": (
            str(reviewed_fact.current_decision_id)
            if reviewed_fact.current_decision_id is not None
            else None
        ),
        "current_decision_kind": reviewed_fact.current_decision_kind,
        "effective_fact_value_ids": [
            str(fact_value_id)
            for fact_value_id in reviewed_fact.effective_fact_value_ids
        ],
        "requires_review": reviewed_fact.requires_review,
    }


def _serialize_reviewed_field(field: DynamicSchemaReviewedField) -> dict[str, object]:
    return {
        "source_field": raw_projection_service._serialize_projected_field(
            field.source_field
        ),
        "review_required": field.review_required,
        "resolved_fact_count": field.resolved_fact_count,
        "review_required_fact_count": field.review_required_fact_count,
        "effective_fact_value_ids": [
            str(fact_value_id) for fact_value_id in field.effective_fact_value_ids
        ],
        "reviewed_facts": [
            _serialize_reviewed_fact(reviewed_fact)
            for reviewed_fact in field.reviewed_facts
        ],
    }


def _serialize_record(record: DynamicSchemaReviewedRecord) -> dict[str, object]:
    return {
        "subject_key": record.subject_key,
        "required_missing_field_keys": list(record.required_missing_field_keys),
        "issue_count": record.issue_count,
        "fields": [_serialize_reviewed_field(field) for field in record.fields],
    }


def _build_manifest_hash(
    *,
    projection: DynamicSchemaReviewProjection,
    subject_keys_filter: list[str] | None,
) -> str:
    return duplicate_grouping_service.hash_deterministic_payload(
        {
            "project_id": str(projection.project_id),
            "schema_id": str(projection.schema_id),
            "schema_version_id": str(projection.schema_version_id),
            "orchestration_id": str(projection.orchestration_id),
            "extraction_run_id": str(projection.extraction_run_id),
            "consistency_check_application_id": str(
                projection.consistency_check_application_id
            ),
            "source_consistency_application_id": str(
                projection.source_consistency_application_id
            ),
            "schema_definition_manifest_hash": projection.schema_definition_manifest_hash,
            "ufl_source_manifest_hash": projection.ufl_source_manifest_hash,
            "consistency_result_manifest_hash": projection.consistency_result_manifest_hash,
            "raw_projection_manifest_hash": projection.raw_projection_manifest_hash,
            "comparison_quality": projection.comparison_quality,
            "subject_keys_filter": subject_keys_filter,
            "algorithm": {
                "name": projection.algorithm_name,
                "version": projection.algorithm_version,
            },
            "counts": {
                "record_count": projection.record_count,
                "unique_matched_fact_count": projection.unique_matched_fact_count,
                "resolved_fact_count": projection.resolved_fact_count,
                "review_required_fact_count": projection.review_required_fact_count,
                "no_candidate_fact_count": projection.no_candidate_fact_count,
                "field_review_required_count": projection.field_review_required_count,
            },
            "records": [_serialize_record(record) for record in projection.records],
        }
    )


async def project_reviewed_orchestration_ufl_to_dynamic_schema(
    session_factory: Callable[[], AsyncSession],
    *,
    project_id: uuid.UUID,
    schema_id: uuid.UUID,
    schema_version_id: uuid.UUID,
    orchestration_id: uuid.UUID,
    consistency_check_application_id: uuid.UUID,
    subject_keys: list[str] | None = None,
) -> DynamicSchemaReviewProjection:
    project_id = _require_uuid(project_id, field_name="project_id")
    schema_id = _require_uuid(schema_id, field_name="schema_id")
    schema_version_id = _require_uuid(schema_version_id, field_name="schema_version_id")
    orchestration_id = _require_uuid(orchestration_id, field_name="orchestration_id")
    consistency_check_application_id = _require_uuid(
        consistency_check_application_id,
        field_name="consistency_check_application_id",
    )
    normalized_subject_keys = _validate_subject_keys(subject_keys)

    raw_projection = await raw_projection_service.project_orchestration_ufl_to_dynamic_schema(
        session_factory,
        project_id=project_id,
        schema_id=schema_id,
        schema_version_id=schema_version_id,
        orchestration_id=orchestration_id,
        subject_keys=subject_keys,
    )
    authenticated_context = await (
        consistency_persistence_service.authenticate_persisted_consistency_check_application(
            session_factory,
            project_id=project_id,
            consistency_check_application_id=consistency_check_application_id,
        )
    )
    effective_projection = await effective_fact_value_service.get_effective_fact_value_projection(
        session_factory,
        project_id=project_id,
        consistency_check_application_id=consistency_check_application_id,
    )

    _validate_raw_projection(
        project_id=project_id,
        schema_id=schema_id,
        schema_version_id=schema_version_id,
        orchestration_id=orchestration_id,
        projection=raw_projection,
    )
    _validate_authenticated_application(
        project_id=project_id,
        orchestration_id=orchestration_id,
        context=authenticated_context,
    )
    _validate_effective_projection(
        project_id=project_id,
        consistency_check_application_id=consistency_check_application_id,
        authenticated_context=authenticated_context,
        effective_projection=effective_projection,
    )

    matched_facts_by_id = _index_matched_facts(raw_projection)
    effective_items_by_fact_id = _index_effective_items(effective_projection)
    unique_fact_summaries: dict[uuid.UUID, DynamicSchemaReviewedFact] = {}
    reviewed_records: list[DynamicSchemaReviewedRecord] = []
    field_review_required_count = 0

    for record in raw_projection.records:
        reviewed_fields: list[DynamicSchemaReviewedField] = []
        for field in record.fields:
            reviewed_facts: list[DynamicSchemaReviewedFact] = []
            field_effective_fact_value_ids: list[uuid.UUID] = []
            resolved_fact_count = 0
            review_required_fact_count = 0
            for fact in field.matched_facts:
                effective_item = effective_items_by_fact_id.get(fact.fact_id)
                if effective_item is not None:
                    _validate_fact_binding(
                        fact_value_ids=matched_facts_by_id[fact.fact_id][1],
                        effective_item=effective_item,
                    )
                reviewed_fact = _build_reviewed_fact(
                    fact=fact,
                    effective_item=effective_item,
                )
                existing_fact_summary = unique_fact_summaries.get(fact.fact_id)
                if existing_fact_summary is None:
                    unique_fact_summaries[fact.fact_id] = reviewed_fact
                elif existing_fact_summary != reviewed_fact:
                    raise DynamicSchemaReviewProjectionInvariantError(
                        "dynamic_schema_review_projection_fact_binding_mismatch"
                    )
                reviewed_facts.append(reviewed_fact)
                if reviewed_fact.review_state == "resolved":
                    resolved_fact_count += 1
                    field_effective_fact_value_ids.extend(
                        reviewed_fact.effective_fact_value_ids
                    )
                if reviewed_fact.requires_review:
                    review_required_fact_count += 1
            review_required = review_required_fact_count > 0
            if review_required:
                field_review_required_count += 1
            reviewed_fields.append(
                DynamicSchemaReviewedField(
                    source_field=field,
                    reviewed_facts=tuple(reviewed_facts),
                    review_required=review_required,
                    resolved_fact_count=resolved_fact_count,
                    review_required_fact_count=review_required_fact_count,
                    effective_fact_value_ids=tuple(field_effective_fact_value_ids),
                )
            )
        reviewed_records.append(
            DynamicSchemaReviewedRecord(
                subject_key=record.subject_key,
                required_missing_field_keys=record.required_missing_field_keys,
                issue_count=record.issue_count,
                fields=tuple(reviewed_fields),
            )
        )

    unique_matched_fact_count = len(unique_fact_summaries)
    resolved_fact_count = sum(
        1
        for reviewed_fact in unique_fact_summaries.values()
        if reviewed_fact.review_state == "resolved"
    )
    review_required_fact_count = sum(
        1
        for reviewed_fact in unique_fact_summaries.values()
        if reviewed_fact.requires_review
    )
    no_candidate_fact_count = sum(
        1
        for reviewed_fact in unique_fact_summaries.values()
        if reviewed_fact.review_state == "no_consistency_candidate"
    )

    projection = DynamicSchemaReviewProjection(
        project_id=raw_projection.project_id,
        schema_id=raw_projection.schema_id,
        schema_version_id=raw_projection.schema_version_id,
        orchestration_id=raw_projection.orchestration_id,
        extraction_run_id=raw_projection.extraction_run_id,
        consistency_check_application_id=consistency_check_application_id,
        source_consistency_application_id=authenticated_context.application.consistency_application_id,
        schema_definition_manifest_hash=raw_projection.schema_definition_manifest_hash,
        ufl_source_manifest_hash=raw_projection.ufl_source_manifest_hash,
        consistency_result_manifest_hash=authenticated_context.application.result_manifest_hash,
        raw_projection_manifest_hash=raw_projection.projection_manifest_hash,
        comparison_quality=raw_projection.comparison_quality,
        algorithm_name=DYNAMIC_SCHEMA_REVIEW_PROJECTION_ALGORITHM_NAME,
        algorithm_version=DYNAMIC_SCHEMA_REVIEW_PROJECTION_ALGORITHM_VERSION,
        record_count=len(reviewed_records),
        unique_matched_fact_count=unique_matched_fact_count,
        resolved_fact_count=resolved_fact_count,
        review_required_fact_count=review_required_fact_count,
        no_candidate_fact_count=no_candidate_fact_count,
        field_review_required_count=field_review_required_count,
        records=tuple(reviewed_records),
        reviewed_projection_manifest_hash="",
    )
    return DynamicSchemaReviewProjection(
        project_id=projection.project_id,
        schema_id=projection.schema_id,
        schema_version_id=projection.schema_version_id,
        orchestration_id=projection.orchestration_id,
        extraction_run_id=projection.extraction_run_id,
        consistency_check_application_id=projection.consistency_check_application_id,
        source_consistency_application_id=projection.source_consistency_application_id,
        schema_definition_manifest_hash=projection.schema_definition_manifest_hash,
        ufl_source_manifest_hash=projection.ufl_source_manifest_hash,
        consistency_result_manifest_hash=projection.consistency_result_manifest_hash,
        raw_projection_manifest_hash=projection.raw_projection_manifest_hash,
        comparison_quality=projection.comparison_quality,
        algorithm_name=projection.algorithm_name,
        algorithm_version=projection.algorithm_version,
        record_count=projection.record_count,
        unique_matched_fact_count=projection.unique_matched_fact_count,
        resolved_fact_count=projection.resolved_fact_count,
        review_required_fact_count=projection.review_required_fact_count,
        no_candidate_fact_count=projection.no_candidate_fact_count,
        field_review_required_count=projection.field_review_required_count,
        records=projection.records,
        reviewed_projection_manifest_hash=_build_manifest_hash(
            projection=projection,
            subject_keys_filter=normalized_subject_keys,
        ),
    )
