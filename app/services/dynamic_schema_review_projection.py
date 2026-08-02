from __future__ import annotations

import re
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


def _require_projection_uuid(value: object) -> uuid.UUID:
    if not isinstance(value, uuid.UUID):
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_projection_invalid"
        )
    return value


def _require_projection_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_projection_invalid"
        )
    return value


def _require_projection_bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_projection_invalid"
        )
    return value


def _normalize_subject_keys(subject_keys: object) -> list[str] | None:
    try:
        return raw_projection_service.normalize_dynamic_schema_ufl_subject_keys(
            subject_keys
        )
    except raw_projection_service.DynamicSchemaUFLProjectionStateError:
        raise DynamicSchemaReviewProjectionStateError(
            "dynamic_schema_review_projection_subject_keys_invalid"
        ) from None


def _authenticate_raw_projection(
    *,
    projection: DynamicSchemaUFLProjection,
    subject_keys: object,
) -> DynamicSchemaUFLProjection:
    try:
        return raw_projection_service.authenticate_dynamic_schema_ufl_projection(
            projection,
            subject_keys=subject_keys,
        )
    except raw_projection_service.DynamicSchemaUFLProjectionStateError:
        raise DynamicSchemaReviewProjectionStateError(
            "dynamic_schema_review_projection_subject_keys_invalid"
        ) from None
    except raw_projection_service.DynamicSchemaUFLProjectionInvariantError:
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_raw_projection_mismatch"
        ) from None


def _validate_raw_projection_source_binding(
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


def _authenticate_effective_projection(
    projection: EffectiveFactValueProjection,
) -> EffectiveFactValueProjection:
    try:
        return effective_fact_value_service.authenticate_effective_fact_value_projection(
            projection
        )
    except effective_fact_value_service.EffectiveFactValueProjectionInvariantError:
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_effective_projection_mismatch"
        ) from None


def _validate_effective_projection_source_binding(
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


def _validate_reviewed_fact_shape(
    reviewed_fact: DynamicSchemaReviewedFact,
) -> frozenset[uuid.UUID]:
    fact_value_ids = _collect_fact_value_ids(reviewed_fact.fact)
    requires_review = _require_projection_bool(reviewed_fact.requires_review)
    if reviewed_fact.review_state == "no_consistency_candidate":
        if (
            reviewed_fact.candidate_id is not None
            or reviewed_fact.assessment_id is not None
            or reviewed_fact.current_decision_id is not None
            or reviewed_fact.current_decision_kind is not None
        ):
            raise DynamicSchemaReviewProjectionInvariantError(
                "dynamic_schema_review_projection_projection_invalid"
            )
        if reviewed_fact.resolution_basis != "none":
            raise DynamicSchemaReviewProjectionInvariantError(
                "dynamic_schema_review_projection_projection_invalid"
            )
        if reviewed_fact.effective_fact_value_ids:
            raise DynamicSchemaReviewProjectionInvariantError(
                "dynamic_schema_review_projection_projection_invalid"
            )
        if requires_review:
            raise DynamicSchemaReviewProjectionInvariantError(
                "dynamic_schema_review_projection_projection_invalid"
            )
        return fact_value_ids

    _require_projection_uuid(reviewed_fact.candidate_id)
    _require_projection_uuid(reviewed_fact.assessment_id)
    if reviewed_fact.review_state == "resolved":
        _require_projection_uuid(reviewed_fact.current_decision_id)
        if requires_review:
            raise DynamicSchemaReviewProjectionInvariantError(
                "dynamic_schema_review_projection_projection_invalid"
            )
        if reviewed_fact.resolution_basis == "human_selection":
            if reviewed_fact.current_decision_kind not in {"select_one", "keep_multiple"}:
                raise DynamicSchemaReviewProjectionInvariantError(
                    "dynamic_schema_review_projection_projection_invalid"
                )
        elif reviewed_fact.resolution_basis == "human_confirmed_compatibility":
            if reviewed_fact.current_decision_kind != "confirm_compatible":
                raise DynamicSchemaReviewProjectionInvariantError(
                    "dynamic_schema_review_projection_projection_invalid"
                )
        else:
            raise DynamicSchemaReviewProjectionInvariantError(
                "dynamic_schema_review_projection_projection_invalid"
            )
        if not reviewed_fact.effective_fact_value_ids:
            raise DynamicSchemaReviewProjectionInvariantError(
                "dynamic_schema_review_projection_projection_invalid"
            )
        if len(set(reviewed_fact.effective_fact_value_ids)) != len(
            reviewed_fact.effective_fact_value_ids
        ):
            raise DynamicSchemaReviewProjectionInvariantError(
                "dynamic_schema_review_projection_projection_invalid"
            )
        if any(
            fact_value_id not in fact_value_ids
            for fact_value_id in reviewed_fact.effective_fact_value_ids
        ):
            raise DynamicSchemaReviewProjectionInvariantError(
                "dynamic_schema_review_projection_projection_invalid"
            )
        return fact_value_ids

    if reviewed_fact.review_state in {"pending_review", "unreviewed_compatible"}:
        if reviewed_fact.current_decision_id is not None or reviewed_fact.current_decision_kind is not None:
            raise DynamicSchemaReviewProjectionInvariantError(
                "dynamic_schema_review_projection_projection_invalid"
            )
        if reviewed_fact.resolution_basis != "none":
            raise DynamicSchemaReviewProjectionInvariantError(
                "dynamic_schema_review_projection_projection_invalid"
            )
        if reviewed_fact.effective_fact_value_ids:
            raise DynamicSchemaReviewProjectionInvariantError(
                "dynamic_schema_review_projection_projection_invalid"
            )
        if not requires_review:
            raise DynamicSchemaReviewProjectionInvariantError(
                "dynamic_schema_review_projection_projection_invalid"
            )
        return fact_value_ids

    if reviewed_fact.review_state == "deferred":
        _require_projection_uuid(reviewed_fact.current_decision_id)
        if reviewed_fact.current_decision_kind != "defer":
            raise DynamicSchemaReviewProjectionInvariantError(
                "dynamic_schema_review_projection_projection_invalid"
            )
        if reviewed_fact.resolution_basis != "none":
            raise DynamicSchemaReviewProjectionInvariantError(
                "dynamic_schema_review_projection_projection_invalid"
            )
        if reviewed_fact.effective_fact_value_ids:
            raise DynamicSchemaReviewProjectionInvariantError(
                "dynamic_schema_review_projection_projection_invalid"
            )
        if not requires_review:
            raise DynamicSchemaReviewProjectionInvariantError(
                "dynamic_schema_review_projection_projection_invalid"
            )
        return fact_value_ids

    raise DynamicSchemaReviewProjectionInvariantError(
        "dynamic_schema_review_projection_projection_invalid"
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
        "fact": raw_projection_service.serialize_dynamic_schema_ufl_fact(
            reviewed_fact.fact
        ),
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
        "source_field": raw_projection_service.serialize_dynamic_schema_ufl_projected_field(
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


def authenticate_dynamic_schema_reviewed_field(
    field: DynamicSchemaReviewedField,
    *,
    record_subject_key: str,
    subject_kind: str | None,
) -> DynamicSchemaReviewedField:
    try:
        raw_projection_service.authenticate_dynamic_schema_ufl_projected_field(
            field.source_field,
            record_subject_key=record_subject_key,
            subject_kind=subject_kind,
        )
    except raw_projection_service.DynamicSchemaUFLProjectionInvariantError:
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_projection_invalid"
        ) from None

    review_required = _require_projection_bool(field.review_required)
    resolved_fact_count = _require_projection_count(field.resolved_fact_count)
    review_required_fact_count = _require_projection_count(
        field.review_required_fact_count
    )
    if len(field.reviewed_facts) != len(field.source_field.matched_facts):
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_projection_invalid"
        )
    if tuple(reviewed_fact.fact for reviewed_fact in field.reviewed_facts) != field.source_field.matched_facts:
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_projection_invalid"
        )

    seen_field_fact_ids: set[uuid.UUID] = set()
    recomputed_effective_fact_value_ids: list[uuid.UUID] = []
    recomputed_resolved_fact_count = 0
    recomputed_review_required_fact_count = 0
    for reviewed_fact in field.reviewed_facts:
        _validate_reviewed_fact_shape(reviewed_fact)
        if reviewed_fact.fact.fact_id in seen_field_fact_ids:
            raise DynamicSchemaReviewProjectionInvariantError(
                "dynamic_schema_review_projection_projection_invalid"
            )
        seen_field_fact_ids.add(reviewed_fact.fact.fact_id)
        if reviewed_fact.review_state == "resolved":
            recomputed_resolved_fact_count += 1
            recomputed_effective_fact_value_ids.extend(
                reviewed_fact.effective_fact_value_ids
            )
        if reviewed_fact.requires_review:
            recomputed_review_required_fact_count += 1

    if resolved_fact_count != recomputed_resolved_fact_count:
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_projection_invalid"
        )
    if review_required_fact_count != recomputed_review_required_fact_count:
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_projection_invalid"
        )
    if review_required != (recomputed_review_required_fact_count > 0):
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_projection_invalid"
        )
    if field.effective_fact_value_ids != tuple(recomputed_effective_fact_value_ids):
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_projection_invalid"
        )
    return field


def authenticate_dynamic_schema_review_projection(
    projection: DynamicSchemaReviewProjection,
    *,
    subject_keys: object,
) -> DynamicSchemaReviewProjection:
    if not isinstance(projection, DynamicSchemaReviewProjection):
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_projection_invalid"
        )
    normalized_subject_keys = _normalize_subject_keys(subject_keys)
    for value in (
        projection.project_id,
        projection.schema_id,
        projection.schema_version_id,
        projection.orchestration_id,
        projection.extraction_run_id,
        projection.consistency_check_application_id,
        projection.source_consistency_application_id,
    ):
        _require_projection_uuid(value)
    for value in (
        projection.schema_definition_manifest_hash,
        projection.ufl_source_manifest_hash,
        projection.consistency_result_manifest_hash,
        projection.raw_projection_manifest_hash,
        projection.reviewed_projection_manifest_hash,
    ):
        _require_sha256(
            value,
            error_code="dynamic_schema_review_projection_projection_invalid",
        )
    if projection.algorithm_name != DYNAMIC_SCHEMA_REVIEW_PROJECTION_ALGORITHM_NAME:
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_projection_invalid"
        )
    if projection.algorithm_version != DYNAMIC_SCHEMA_REVIEW_PROJECTION_ALGORITHM_VERSION:
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_projection_invalid"
        )
    if projection.comparison_quality not in {"complete", "partial"}:
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_projection_invalid"
        )
    record_count = _require_projection_count(projection.record_count)
    unique_matched_fact_count = _require_projection_count(
        projection.unique_matched_fact_count
    )
    resolved_fact_count = _require_projection_count(projection.resolved_fact_count)
    review_required_fact_count = _require_projection_count(
        projection.review_required_fact_count
    )
    no_candidate_fact_count = _require_projection_count(
        projection.no_candidate_fact_count
    )
    field_review_required_count = _require_projection_count(
        projection.field_review_required_count
    )

    seen_subject_keys: set[str] = set()
    record_subject_keys: list[str] = []
    unique_reviewed_facts: dict[uuid.UUID, DynamicSchemaReviewedFact] = {}
    recomputed_field_review_required_count = 0
    for record in projection.records:
        subject_key = raw_projection_service.normalize_dynamic_schema_ufl_subject_keys(
            [record.subject_key]
        )[0]
        if subject_key != record.subject_key:
            raise DynamicSchemaReviewProjectionInvariantError(
                "dynamic_schema_review_projection_projection_invalid"
            )
        if subject_key in seen_subject_keys:
            raise DynamicSchemaReviewProjectionInvariantError(
                "dynamic_schema_review_projection_projection_invalid"
            )
        seen_subject_keys.add(subject_key)
        record_subject_keys.append(subject_key)
        record_issue_count = _require_projection_count(record.issue_count)
        sorted_fields = tuple(
            sorted(
                record.fields,
                key=lambda field: (
                    field.source_field.display_order,
                    field.source_field.field_key,
                    field.source_field.field_id,
                ),
            )
        )
        if record.fields != sorted_fields:
            raise DynamicSchemaReviewProjectionInvariantError(
                "dynamic_schema_review_projection_projection_invalid"
            )
        seen_field_keys: set[str] = set()
        recomputed_required_missing_field_keys: list[str] = []
        recomputed_record_issue_count = 0
        for field in record.fields:
            if field.source_field.field_key in seen_field_keys:
                raise DynamicSchemaReviewProjectionInvariantError(
                    "dynamic_schema_review_projection_projection_invalid"
                )
            seen_field_keys.add(field.source_field.field_key)
            subject_kinds = {
                fact.subject_kind for fact in field.source_field.matched_facts
            }
            if len(subject_kinds) > 1:
                raise DynamicSchemaReviewProjectionInvariantError(
                    "dynamic_schema_review_projection_projection_invalid"
                )
            authenticated_field = authenticate_dynamic_schema_reviewed_field(
                field,
                record_subject_key=record.subject_key,
                subject_kind=next(iter(subject_kinds), None),
            )
            for reviewed_fact in authenticated_field.reviewed_facts:
                prior_reviewed_fact = unique_reviewed_facts.get(reviewed_fact.fact.fact_id)
                if prior_reviewed_fact is None:
                    unique_reviewed_facts[reviewed_fact.fact.fact_id] = reviewed_fact
                elif prior_reviewed_fact != reviewed_fact:
                    raise DynamicSchemaReviewProjectionInvariantError(
                        "dynamic_schema_review_projection_projection_invalid"
                    )
            recomputed_record_issue_count += len(authenticated_field.source_field.issues)
            if "required_missing" in authenticated_field.source_field.issues:
                recomputed_required_missing_field_keys.append(
                    authenticated_field.source_field.field_key
                )
            if authenticated_field.review_required:
                recomputed_field_review_required_count += 1
        if record.required_missing_field_keys != tuple(
            recomputed_required_missing_field_keys
        ):
            raise DynamicSchemaReviewProjectionInvariantError(
                "dynamic_schema_review_projection_projection_invalid"
            )
        if record_issue_count != recomputed_record_issue_count:
            raise DynamicSchemaReviewProjectionInvariantError(
                "dynamic_schema_review_projection_projection_invalid"
            )

    if normalized_subject_keys is None:
        if tuple(record_subject_keys) != tuple(sorted(record_subject_keys)):
            raise DynamicSchemaReviewProjectionInvariantError(
                "dynamic_schema_review_projection_projection_invalid"
            )
    elif tuple(record_subject_keys) != tuple(normalized_subject_keys):
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_projection_invalid"
        )
    if record_count != len(projection.records):
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_projection_invalid"
        )
    recomputed_unique_matched_fact_count = len(unique_reviewed_facts)
    recomputed_resolved_fact_count = sum(
        1
        for reviewed_fact in unique_reviewed_facts.values()
        if reviewed_fact.review_state == "resolved"
    )
    recomputed_review_required_fact_count = sum(
        1 for reviewed_fact in unique_reviewed_facts.values() if reviewed_fact.requires_review
    )
    recomputed_no_candidate_fact_count = sum(
        1
        for reviewed_fact in unique_reviewed_facts.values()
        if reviewed_fact.review_state == "no_consistency_candidate"
    )
    if unique_matched_fact_count != recomputed_unique_matched_fact_count:
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_projection_invalid"
        )
    if resolved_fact_count != recomputed_resolved_fact_count:
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_projection_invalid"
        )
    if review_required_fact_count != recomputed_review_required_fact_count:
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_projection_invalid"
        )
    if no_candidate_fact_count != recomputed_no_candidate_fact_count:
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_projection_invalid"
        )
    if field_review_required_count != recomputed_field_review_required_count:
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_projection_invalid"
        )
    if projection.reviewed_projection_manifest_hash != _build_manifest_hash(
        projection=projection,
        subject_keys_filter=normalized_subject_keys,
    ):
        raise DynamicSchemaReviewProjectionInvariantError(
            "dynamic_schema_review_projection_projection_invalid"
        )
    return projection


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
    normalized_subject_keys = _normalize_subject_keys(subject_keys)

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

    authenticated_raw_projection = _authenticate_raw_projection(
        projection=raw_projection,
        subject_keys=subject_keys,
    )
    authenticated_effective_projection = _authenticate_effective_projection(
        effective_projection
    )

    _validate_raw_projection_source_binding(
        project_id=project_id,
        schema_id=schema_id,
        schema_version_id=schema_version_id,
        orchestration_id=orchestration_id,
        projection=authenticated_raw_projection,
    )
    _validate_authenticated_application(
        project_id=project_id,
        orchestration_id=orchestration_id,
        context=authenticated_context,
    )
    _validate_effective_projection_source_binding(
        project_id=project_id,
        consistency_check_application_id=consistency_check_application_id,
        authenticated_context=authenticated_context,
        effective_projection=authenticated_effective_projection,
    )

    matched_facts_by_id = _index_matched_facts(authenticated_raw_projection)
    effective_items_by_fact_id = _index_effective_items(authenticated_effective_projection)
    unique_fact_summaries: dict[uuid.UUID, DynamicSchemaReviewedFact] = {}
    reviewed_records: list[DynamicSchemaReviewedRecord] = []
    field_review_required_count = 0

    for record in authenticated_raw_projection.records:
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
        project_id=authenticated_raw_projection.project_id,
        schema_id=authenticated_raw_projection.schema_id,
        schema_version_id=authenticated_raw_projection.schema_version_id,
        orchestration_id=authenticated_raw_projection.orchestration_id,
        extraction_run_id=authenticated_raw_projection.extraction_run_id,
        consistency_check_application_id=consistency_check_application_id,
        source_consistency_application_id=authenticated_context.application.consistency_application_id,
        schema_definition_manifest_hash=authenticated_raw_projection.schema_definition_manifest_hash,
        ufl_source_manifest_hash=authenticated_raw_projection.ufl_source_manifest_hash,
        consistency_result_manifest_hash=authenticated_context.application.result_manifest_hash,
        raw_projection_manifest_hash=authenticated_raw_projection.projection_manifest_hash,
        comparison_quality=authenticated_raw_projection.comparison_quality,
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
    return authenticate_dynamic_schema_review_projection(
        DynamicSchemaReviewProjection(
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
        ),
        subject_keys=subject_keys,
    )
