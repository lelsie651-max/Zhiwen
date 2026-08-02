from __future__ import annotations

import re
import uuid
from collections.abc import Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.dynamic_schema_knowledge_view import (
    DynamicSchemaKnowledgeField,
    DynamicSchemaKnowledgeRecord,
    DynamicSchemaKnowledgeSection,
    DynamicSchemaKnowledgeView,
)
from app.schemas.dynamic_schema_review_projection import (
    DynamicSchemaReviewProjection,
    DynamicSchemaReviewedFact,
    DynamicSchemaReviewedField,
    DynamicSchemaReviewedRecord,
)
import app.services.dynamic_schema_review_projection as review_projection_service
import app.services.fact_value_duplicate_grouping as duplicate_grouping_service


DYNAMIC_SCHEMA_KNOWLEDGE_VIEW_ALGORITHM_NAME = "dynamic_schema_knowledge_view"
DYNAMIC_SCHEMA_KNOWLEDGE_VIEW_ALGORITHM_VERSION = "1.0.0"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class DynamicSchemaKnowledgeViewError(Exception):
    """Base error for dynamic schema knowledge view failures."""


class DynamicSchemaKnowledgeViewStateError(DynamicSchemaKnowledgeViewError):
    """Raised when knowledge view inputs are invalid."""


class DynamicSchemaKnowledgeViewInvariantError(DynamicSchemaKnowledgeViewError):
    """Raised when knowledge view sources drift or conflict."""


def _require_uuid(value: object, *, field_name: str) -> uuid.UUID:
    if not isinstance(value, uuid.UUID):
        raise DynamicSchemaKnowledgeViewStateError(
            f"dynamic_schema_knowledge_view_{field_name}_invalid"
        )
    return value


def _require_projection_uuid(value: object) -> uuid.UUID:
    if not isinstance(value, uuid.UUID):
        raise DynamicSchemaKnowledgeViewInvariantError(
            "dynamic_schema_knowledge_view_projection_invalid"
        )
    return value


def _require_projection_sha256(value: object) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise DynamicSchemaKnowledgeViewInvariantError(
            "dynamic_schema_knowledge_view_projection_invalid"
        )
    return value


def _require_projection_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DynamicSchemaKnowledgeViewInvariantError(
            "dynamic_schema_knowledge_view_projection_invalid"
        )
    return value


def _require_projection_bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise DynamicSchemaKnowledgeViewInvariantError(
            "dynamic_schema_knowledge_view_projection_invalid"
        )
    return value


def _normalize_subject_keys(subject_keys: object) -> list[str] | None:
    try:
        return review_projection_service.raw_projection_service.normalize_dynamic_schema_ufl_subject_keys(
            subject_keys
        )
    except review_projection_service.raw_projection_service.DynamicSchemaUFLProjectionStateError:
        raise DynamicSchemaKnowledgeViewStateError(
            "dynamic_schema_knowledge_view_subject_keys_invalid"
        ) from None


def _build_knowledge_state(field: DynamicSchemaReviewedField) -> str:
    if field.source_field.is_missing:
        return "missing"
    if any(reviewed_fact.requires_review for reviewed_fact in field.reviewed_facts):
        return "review_required"
    reviewed_states = {reviewed_fact.review_state for reviewed_fact in field.reviewed_facts}
    if reviewed_states == {"resolved"}:
        return "resolved"
    if reviewed_states == {"no_consistency_candidate"}:
        return "observation_only"
    if reviewed_states <= {"resolved", "no_consistency_candidate"} and {
        "resolved",
        "no_consistency_candidate",
    } <= reviewed_states:
        return "mixed_reviewed_observation"
    raise DynamicSchemaKnowledgeViewInvariantError(
        "dynamic_schema_knowledge_view_projection_invalid"
    )


def _build_knowledge_field(
    field: DynamicSchemaReviewedField,
) -> DynamicSchemaKnowledgeField:
    observed_fact_value_count = sum(
        len(value_group.fact_value_ids)
        for reviewed_fact in field.reviewed_facts
        for value_group in reviewed_fact.fact.value_groups
    )
    semantic_value_count = sum(
        len(reviewed_fact.fact.value_groups) for reviewed_fact in field.reviewed_facts
    )
    return DynamicSchemaKnowledgeField(
        source_field=field.source_field,
        reviewed_facts=field.reviewed_facts,
        knowledge_state=_build_knowledge_state(field),
        effective_fact_value_ids=field.effective_fact_value_ids,
        observed_fact_value_count=observed_fact_value_count,
        semantic_value_count=semantic_value_count,
        has_schema_issues=bool(field.source_field.issues),
    )


def _build_sections(
    record: DynamicSchemaReviewedRecord,
) -> tuple[DynamicSchemaKnowledgeSection, ...]:
    grouped_fields: dict[str | None, list[DynamicSchemaKnowledgeField]] = {}
    group_order: list[str | None] = []
    group_display_order: dict[str | None, int] = {}
    for field in record.fields:
        knowledge_field = _build_knowledge_field(field)
        group_key = knowledge_field.source_field.group_key
        if group_key not in grouped_fields:
            grouped_fields[group_key] = []
            group_order.append(group_key)
            group_display_order[group_key] = knowledge_field.source_field.display_order
        grouped_fields[group_key].append(knowledge_field)
    return tuple(
        DynamicSchemaKnowledgeSection(
            group_key=group_key,
            display_order=group_display_order[group_key],
            fields=tuple(grouped_fields[group_key]),
        )
        for group_key in group_order
    )


def _serialize_reviewed_fact(reviewed_fact: DynamicSchemaReviewedFact) -> dict[str, object]:
    return {
        "fact": review_projection_service.raw_projection_service.serialize_dynamic_schema_ufl_fact(
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


def _serialize_field(field: DynamicSchemaKnowledgeField) -> dict[str, object]:
    return {
        "source_field": review_projection_service.raw_projection_service.serialize_dynamic_schema_ufl_projected_field(
            field.source_field
        ),
        "knowledge_state": field.knowledge_state,
        "effective_fact_value_ids": [
            str(fact_value_id) for fact_value_id in field.effective_fact_value_ids
        ],
        "observed_fact_value_count": field.observed_fact_value_count,
        "semantic_value_count": field.semantic_value_count,
        "has_schema_issues": field.has_schema_issues,
        "reviewed_facts": [
            _serialize_reviewed_fact(reviewed_fact) for reviewed_fact in field.reviewed_facts
        ],
    }


def _serialize_section(section: DynamicSchemaKnowledgeSection) -> dict[str, object]:
    return {
        "group_key": section.group_key,
        "display_order": section.display_order,
        "fields": [_serialize_field(field) for field in section.fields],
    }


def _serialize_record(record: DynamicSchemaKnowledgeRecord) -> dict[str, object]:
    return {
        "subject_key": record.subject_key,
        "title_field_key": record.title_field_key,
        "has_review_required": record.has_review_required,
        "issue_count": record.issue_count,
        "sections": [_serialize_section(section) for section in record.sections],
    }


def _serialize_view(
    view: DynamicSchemaKnowledgeView,
) -> dict[str, object]:
    return {
        "project_id": str(view.project_id),
        "schema_id": str(view.schema_id),
        "schema_version_id": str(view.schema_version_id),
        "orchestration_id": str(view.orchestration_id),
        "extraction_run_id": str(view.extraction_run_id),
        "consistency_check_application_id": str(
            view.consistency_check_application_id
        ),
        "source_consistency_application_id": str(
            view.source_consistency_application_id
        ),
        "schema_definition_manifest_hash": view.schema_definition_manifest_hash,
        "ufl_source_manifest_hash": view.ufl_source_manifest_hash,
        "consistency_result_manifest_hash": view.consistency_result_manifest_hash,
        "raw_projection_manifest_hash": view.raw_projection_manifest_hash,
        "reviewed_projection_manifest_hash": view.reviewed_projection_manifest_hash,
        "comparison_quality": view.comparison_quality,
        "algorithm_name": view.algorithm_name,
        "algorithm_version": view.algorithm_version,
        "record_count": view.record_count,
        "section_count": view.section_count,
        "field_count": view.field_count,
        "missing_field_count": view.missing_field_count,
        "review_required_field_count": view.review_required_field_count,
        "resolved_field_count": view.resolved_field_count,
        "observation_only_field_count": view.observation_only_field_count,
        "mixed_field_count": view.mixed_field_count,
        "records": [_serialize_record(record) for record in view.records],
        "knowledge_view_manifest_hash": view.knowledge_view_manifest_hash,
    }


def _build_manifest_hash(
    *,
    view: DynamicSchemaKnowledgeView,
    subject_keys_filter: list[str] | None,
) -> str:
    return duplicate_grouping_service.hash_deterministic_payload(
        {
            "project_id": str(view.project_id),
            "schema_id": str(view.schema_id),
            "schema_version_id": str(view.schema_version_id),
            "orchestration_id": str(view.orchestration_id),
            "extraction_run_id": str(view.extraction_run_id),
            "consistency_check_application_id": str(
                view.consistency_check_application_id
            ),
            "source_consistency_application_id": str(
                view.source_consistency_application_id
            ),
            "schema_definition_manifest_hash": view.schema_definition_manifest_hash,
            "ufl_source_manifest_hash": view.ufl_source_manifest_hash,
            "consistency_result_manifest_hash": view.consistency_result_manifest_hash,
            "raw_projection_manifest_hash": view.raw_projection_manifest_hash,
            "reviewed_projection_manifest_hash": view.reviewed_projection_manifest_hash,
            "comparison_quality": view.comparison_quality,
            "subject_keys_filter": subject_keys_filter,
            "algorithm": {
                "name": view.algorithm_name,
                "version": view.algorithm_version,
            },
            "counts": {
                "record_count": view.record_count,
                "section_count": view.section_count,
                "field_count": view.field_count,
                "missing_field_count": view.missing_field_count,
                "review_required_field_count": view.review_required_field_count,
                "resolved_field_count": view.resolved_field_count,
                "observation_only_field_count": view.observation_only_field_count,
                "mixed_field_count": view.mixed_field_count,
            },
            "records": [_serialize_record(record) for record in view.records],
        }
    )


def serialize_dynamic_schema_knowledge_view(
    view: DynamicSchemaKnowledgeView,
    *,
    subject_keys: object,
) -> dict[str, object]:
    authenticated_view = authenticate_dynamic_schema_knowledge_view(
        view,
        subject_keys=subject_keys,
    )
    return _serialize_view(authenticated_view)


def authenticate_dynamic_schema_knowledge_view(
    view: DynamicSchemaKnowledgeView,
    *,
    subject_keys: object,
) -> DynamicSchemaKnowledgeView:
    normalized_subject_keys = _normalize_subject_keys(subject_keys)
    for value in (
        view.project_id,
        view.schema_id,
        view.schema_version_id,
        view.orchestration_id,
        view.extraction_run_id,
        view.consistency_check_application_id,
        view.source_consistency_application_id,
    ):
        _require_projection_uuid(value)
    for value in (
        view.schema_definition_manifest_hash,
        view.ufl_source_manifest_hash,
        view.consistency_result_manifest_hash,
        view.raw_projection_manifest_hash,
        view.reviewed_projection_manifest_hash,
        view.knowledge_view_manifest_hash,
    ):
        _require_projection_sha256(value)
    if view.algorithm_name != DYNAMIC_SCHEMA_KNOWLEDGE_VIEW_ALGORITHM_NAME:
        raise DynamicSchemaKnowledgeViewInvariantError(
            "dynamic_schema_knowledge_view_projection_invalid"
        )
    if view.algorithm_version != DYNAMIC_SCHEMA_KNOWLEDGE_VIEW_ALGORITHM_VERSION:
        raise DynamicSchemaKnowledgeViewInvariantError(
            "dynamic_schema_knowledge_view_projection_invalid"
        )
    if view.comparison_quality not in {"complete", "partial"}:
        raise DynamicSchemaKnowledgeViewInvariantError(
            "dynamic_schema_knowledge_view_projection_invalid"
        )
    record_count = _require_projection_count(view.record_count)
    section_count = _require_projection_count(view.section_count)
    field_count = _require_projection_count(view.field_count)
    missing_field_count = _require_projection_count(view.missing_field_count)
    review_required_field_count = _require_projection_count(
        view.review_required_field_count
    )
    resolved_field_count = _require_projection_count(view.resolved_field_count)
    observation_only_field_count = _require_projection_count(
        view.observation_only_field_count
    )
    mixed_field_count = _require_projection_count(view.mixed_field_count)

    seen_subject_keys: set[str] = set()
    record_subject_keys: list[str] = []
    recomputed_section_count = 0
    recomputed_field_count = 0
    recomputed_missing_field_count = 0
    recomputed_review_required_field_count = 0
    recomputed_resolved_field_count = 0
    recomputed_observation_only_field_count = 0
    recomputed_mixed_field_count = 0
    unique_reviewed_facts: dict[uuid.UUID, DynamicSchemaReviewedFact] = {}
    for record in view.records:
        normalized_subject_key = review_projection_service.raw_projection_service.normalize_dynamic_schema_ufl_subject_keys(
            [record.subject_key]
        )[0]
        if normalized_subject_key != record.subject_key:
            raise DynamicSchemaKnowledgeViewInvariantError(
                "dynamic_schema_knowledge_view_projection_invalid"
            )
        if normalized_subject_key in seen_subject_keys:
            raise DynamicSchemaKnowledgeViewInvariantError(
                "dynamic_schema_knowledge_view_projection_invalid"
            )
        seen_subject_keys.add(normalized_subject_key)
        record_subject_keys.append(normalized_subject_key)
        record_issue_count = _require_projection_count(record.issue_count)
        has_review_required = _require_projection_bool(record.has_review_required)
        title_field_keys = [
            field.source_field.field_key
            for section in record.sections
            for field in section.fields
            if field.source_field.is_title
        ]
        if len(title_field_keys) > 1:
            raise DynamicSchemaKnowledgeViewInvariantError(
                "dynamic_schema_knowledge_view_projection_invalid"
            )
        if record.title_field_key != (title_field_keys[0] if title_field_keys else None):
            raise DynamicSchemaKnowledgeViewInvariantError(
                "dynamic_schema_knowledge_view_projection_invalid"
            )
        section_group_keys_seen: set[str | None] = set()
        record_has_review_required = False
        recomputed_record_issue_count = 0
        flat_fields: list[DynamicSchemaKnowledgeField] = []
        for section in record.sections:
            recomputed_section_count += 1
            if section.group_key in section_group_keys_seen:
                raise DynamicSchemaKnowledgeViewInvariantError(
                    "dynamic_schema_knowledge_view_projection_invalid"
                )
            section_group_keys_seen.add(section.group_key)
            section_display_order = _require_projection_count(section.display_order)
            if section.fields:
                if section_display_order != section.fields[0].source_field.display_order:
                    raise DynamicSchemaKnowledgeViewInvariantError(
                        "dynamic_schema_knowledge_view_projection_invalid"
                    )
            for field in section.fields:
                flat_fields.append(field)
                recomputed_field_count += 1
                has_schema_issues = _require_projection_bool(field.has_schema_issues)
                observed_fact_value_count = _require_projection_count(
                    field.observed_fact_value_count
                )
                semantic_value_count = _require_projection_count(field.semantic_value_count)
                authenticated_reviewed_field = (
                    review_projection_service.authenticate_dynamic_schema_reviewed_field(
                        DynamicSchemaReviewedField(
                            source_field=field.source_field,
                            reviewed_facts=field.reviewed_facts,
                            review_required=any(
                                reviewed_fact.requires_review
                                for reviewed_fact in field.reviewed_facts
                            ),
                            resolved_fact_count=sum(
                                1
                                for reviewed_fact in field.reviewed_facts
                                if reviewed_fact.review_state == "resolved"
                            ),
                            review_required_fact_count=sum(
                                1
                                for reviewed_fact in field.reviewed_facts
                                if reviewed_fact.requires_review
                            ),
                            effective_fact_value_ids=field.effective_fact_value_ids,
                        ),
                        record_subject_key=record.subject_key,
                        subject_kind=next(
                            iter(
                                {
                                    reviewed_fact.fact.subject_kind
                                    for reviewed_fact in field.reviewed_facts
                                }
                            ),
                            None,
                        ),
                    )
                )
                for reviewed_fact in authenticated_reviewed_field.reviewed_facts:
                    prior_reviewed_fact = unique_reviewed_facts.get(reviewed_fact.fact.fact_id)
                    if prior_reviewed_fact is None:
                        unique_reviewed_facts[reviewed_fact.fact.fact_id] = reviewed_fact
                    elif prior_reviewed_fact != reviewed_fact:
                        raise DynamicSchemaKnowledgeViewInvariantError(
                            "dynamic_schema_knowledge_view_projection_invalid"
                        )
                recomputed_observed_fact_value_count = sum(
                    len(value_group.fact_value_ids)
                    for reviewed_fact in authenticated_reviewed_field.reviewed_facts
                    for value_group in reviewed_fact.fact.value_groups
                )
                recomputed_semantic_value_count = sum(
                    len(reviewed_fact.fact.value_groups)
                    for reviewed_fact in authenticated_reviewed_field.reviewed_facts
                )
                if observed_fact_value_count != recomputed_observed_fact_value_count:
                    raise DynamicSchemaKnowledgeViewInvariantError(
                        "dynamic_schema_knowledge_view_projection_invalid"
                    )
                if semantic_value_count != recomputed_semantic_value_count:
                    raise DynamicSchemaKnowledgeViewInvariantError(
                        "dynamic_schema_knowledge_view_projection_invalid"
                    )
                if semantic_value_count != authenticated_reviewed_field.source_field.semantic_value_count:
                    raise DynamicSchemaKnowledgeViewInvariantError(
                        "dynamic_schema_knowledge_view_projection_invalid"
                    )
                if has_schema_issues != bool(authenticated_reviewed_field.source_field.issues):
                    raise DynamicSchemaKnowledgeViewInvariantError(
                        "dynamic_schema_knowledge_view_projection_invalid"
                    )
                recomputed_state = _build_knowledge_state(authenticated_reviewed_field)
                if field.knowledge_state != recomputed_state:
                    raise DynamicSchemaKnowledgeViewInvariantError(
                        "dynamic_schema_knowledge_view_projection_invalid"
                    )
                if field.effective_fact_value_ids != tuple(
                    fact_value_id
                    for reviewed_fact in authenticated_reviewed_field.reviewed_facts
                    if reviewed_fact.review_state == "resolved"
                    for fact_value_id in reviewed_fact.effective_fact_value_ids
                ):
                    raise DynamicSchemaKnowledgeViewInvariantError(
                        "dynamic_schema_knowledge_view_projection_invalid"
                    )
                recomputed_record_issue_count += len(authenticated_reviewed_field.source_field.issues)
                if field.knowledge_state == "missing":
                    recomputed_missing_field_count += 1
                elif field.knowledge_state == "review_required":
                    recomputed_review_required_field_count += 1
                    record_has_review_required = True
                elif field.knowledge_state == "resolved":
                    recomputed_resolved_field_count += 1
                elif field.knowledge_state == "observation_only":
                    recomputed_observation_only_field_count += 1
                elif field.knowledge_state == "mixed_reviewed_observation":
                    recomputed_mixed_field_count += 1
                else:
                    raise DynamicSchemaKnowledgeViewInvariantError(
                        "dynamic_schema_knowledge_view_projection_invalid"
                    )
        expected_sections = _build_sections(
            DynamicSchemaReviewedRecord(
                subject_key=record.subject_key,
                required_missing_field_keys=(),
                issue_count=record.issue_count,
                fields=tuple(
                    DynamicSchemaReviewedField(
                        source_field=field.source_field,
                        reviewed_facts=field.reviewed_facts,
                        review_required=field.knowledge_state == "review_required",
                        resolved_fact_count=sum(
                            1
                            for reviewed_fact in field.reviewed_facts
                            if reviewed_fact.review_state == "resolved"
                        ),
                        review_required_fact_count=sum(
                            1
                            for reviewed_fact in field.reviewed_facts
                            if reviewed_fact.requires_review
                        ),
                        effective_fact_value_ids=field.effective_fact_value_ids,
                    )
                    for field in flat_fields
                ),
            )
        )
        if tuple(
            (section.group_key, section.display_order, section.fields)
            for section in expected_sections
        ) != tuple(
            (section.group_key, section.display_order, section.fields)
            for section in record.sections
        ):
            raise DynamicSchemaKnowledgeViewInvariantError(
                "dynamic_schema_knowledge_view_projection_invalid"
            )
        if has_review_required != record_has_review_required:
            raise DynamicSchemaKnowledgeViewInvariantError(
                "dynamic_schema_knowledge_view_projection_invalid"
            )
        if record_issue_count != recomputed_record_issue_count:
            raise DynamicSchemaKnowledgeViewInvariantError(
                "dynamic_schema_knowledge_view_projection_invalid"
            )

    if normalized_subject_keys is None:
        if tuple(record_subject_keys) != tuple(sorted(record_subject_keys)):
            raise DynamicSchemaKnowledgeViewInvariantError(
                "dynamic_schema_knowledge_view_projection_invalid"
            )
    elif tuple(record_subject_keys) != tuple(normalized_subject_keys):
        raise DynamicSchemaKnowledgeViewInvariantError(
            "dynamic_schema_knowledge_view_projection_invalid"
        )
    if record_count != len(view.records):
        raise DynamicSchemaKnowledgeViewInvariantError(
            "dynamic_schema_knowledge_view_projection_invalid"
        )
    if section_count != recomputed_section_count:
        raise DynamicSchemaKnowledgeViewInvariantError(
            "dynamic_schema_knowledge_view_projection_invalid"
        )
    if field_count != recomputed_field_count:
        raise DynamicSchemaKnowledgeViewInvariantError(
            "dynamic_schema_knowledge_view_projection_invalid"
        )
    if missing_field_count != recomputed_missing_field_count:
        raise DynamicSchemaKnowledgeViewInvariantError(
            "dynamic_schema_knowledge_view_projection_invalid"
        )
    if review_required_field_count != recomputed_review_required_field_count:
        raise DynamicSchemaKnowledgeViewInvariantError(
            "dynamic_schema_knowledge_view_projection_invalid"
        )
    if resolved_field_count != recomputed_resolved_field_count:
        raise DynamicSchemaKnowledgeViewInvariantError(
            "dynamic_schema_knowledge_view_projection_invalid"
        )
    if observation_only_field_count != recomputed_observation_only_field_count:
        raise DynamicSchemaKnowledgeViewInvariantError(
            "dynamic_schema_knowledge_view_projection_invalid"
        )
    if mixed_field_count != recomputed_mixed_field_count:
        raise DynamicSchemaKnowledgeViewInvariantError(
            "dynamic_schema_knowledge_view_projection_invalid"
        )
    if view.knowledge_view_manifest_hash != _build_manifest_hash(
        view=view,
        subject_keys_filter=normalized_subject_keys,
    ):
        raise DynamicSchemaKnowledgeViewInvariantError(
            "dynamic_schema_knowledge_view_projection_invalid"
        )
    return view


async def build_dynamic_schema_knowledge_view(
    session_factory: Callable[[], AsyncSession],
    *,
    project_id: uuid.UUID,
    schema_id: uuid.UUID,
    schema_version_id: uuid.UUID,
    orchestration_id: uuid.UUID,
    consistency_check_application_id: uuid.UUID,
    subject_keys: list[str] | None = None,
) -> DynamicSchemaKnowledgeView:
    project_id = _require_uuid(project_id, field_name="project_id")
    schema_id = _require_uuid(schema_id, field_name="schema_id")
    schema_version_id = _require_uuid(schema_version_id, field_name="schema_version_id")
    orchestration_id = _require_uuid(orchestration_id, field_name="orchestration_id")
    consistency_check_application_id = _require_uuid(
        consistency_check_application_id,
        field_name="consistency_check_application_id",
    )
    normalized_subject_keys = _normalize_subject_keys(subject_keys)
    reviewed_projection = await review_projection_service.project_reviewed_orchestration_ufl_to_dynamic_schema(
        session_factory,
        project_id=project_id,
        schema_id=schema_id,
        schema_version_id=schema_version_id,
        orchestration_id=orchestration_id,
        consistency_check_application_id=consistency_check_application_id,
        subject_keys=subject_keys,
    )
    try:
        authenticated_projection = (
            review_projection_service.authenticate_dynamic_schema_review_projection(
                reviewed_projection,
                subject_keys=subject_keys,
            )
        )
    except review_projection_service.DynamicSchemaReviewProjectionStateError:
        raise DynamicSchemaKnowledgeViewStateError(
            "dynamic_schema_knowledge_view_subject_keys_invalid"
        ) from None
    except review_projection_service.DynamicSchemaReviewProjectionInvariantError:
        raise DynamicSchemaKnowledgeViewInvariantError(
            "dynamic_schema_knowledge_view_review_projection_mismatch"
        ) from None

    records = tuple(
        DynamicSchemaKnowledgeRecord(
            subject_key=record.subject_key,
            title_field_key=next(
                (
                    field.source_field.field_key
                    for field in record.fields
                    if field.source_field.is_title
                ),
                None,
            ),
            has_review_required=any(field.review_required for field in record.fields),
            issue_count=record.issue_count,
            sections=_build_sections(record),
        )
        for record in authenticated_projection.records
    )
    all_fields = [
        field
        for record in records
        for section in record.sections
        for field in section.fields
    ]
    view = DynamicSchemaKnowledgeView(
        project_id=authenticated_projection.project_id,
        schema_id=authenticated_projection.schema_id,
        schema_version_id=authenticated_projection.schema_version_id,
        orchestration_id=authenticated_projection.orchestration_id,
        extraction_run_id=authenticated_projection.extraction_run_id,
        consistency_check_application_id=authenticated_projection.consistency_check_application_id,
        source_consistency_application_id=authenticated_projection.source_consistency_application_id,
        schema_definition_manifest_hash=authenticated_projection.schema_definition_manifest_hash,
        ufl_source_manifest_hash=authenticated_projection.ufl_source_manifest_hash,
        consistency_result_manifest_hash=authenticated_projection.consistency_result_manifest_hash,
        raw_projection_manifest_hash=authenticated_projection.raw_projection_manifest_hash,
        reviewed_projection_manifest_hash=authenticated_projection.reviewed_projection_manifest_hash,
        comparison_quality=authenticated_projection.comparison_quality,
        algorithm_name=DYNAMIC_SCHEMA_KNOWLEDGE_VIEW_ALGORITHM_NAME,
        algorithm_version=DYNAMIC_SCHEMA_KNOWLEDGE_VIEW_ALGORITHM_VERSION,
        record_count=len(records),
        section_count=sum(len(record.sections) for record in records),
        field_count=len(all_fields),
        missing_field_count=sum(
            1 for field in all_fields if field.knowledge_state == "missing"
        ),
        review_required_field_count=sum(
            1 for field in all_fields if field.knowledge_state == "review_required"
        ),
        resolved_field_count=sum(
            1 for field in all_fields if field.knowledge_state == "resolved"
        ),
        observation_only_field_count=sum(
            1 for field in all_fields if field.knowledge_state == "observation_only"
        ),
        mixed_field_count=sum(
            1
            for field in all_fields
            if field.knowledge_state == "mixed_reviewed_observation"
        ),
        records=records,
        knowledge_view_manifest_hash="",
    )
    return authenticate_dynamic_schema_knowledge_view(
        DynamicSchemaKnowledgeView(
            project_id=view.project_id,
            schema_id=view.schema_id,
            schema_version_id=view.schema_version_id,
            orchestration_id=view.orchestration_id,
            extraction_run_id=view.extraction_run_id,
            consistency_check_application_id=view.consistency_check_application_id,
            source_consistency_application_id=view.source_consistency_application_id,
            schema_definition_manifest_hash=view.schema_definition_manifest_hash,
            ufl_source_manifest_hash=view.ufl_source_manifest_hash,
            consistency_result_manifest_hash=view.consistency_result_manifest_hash,
            raw_projection_manifest_hash=view.raw_projection_manifest_hash,
            reviewed_projection_manifest_hash=view.reviewed_projection_manifest_hash,
            comparison_quality=view.comparison_quality,
            algorithm_name=view.algorithm_name,
            algorithm_version=view.algorithm_version,
            record_count=view.record_count,
            section_count=view.section_count,
            field_count=view.field_count,
            missing_field_count=view.missing_field_count,
            review_required_field_count=view.review_required_field_count,
            resolved_field_count=view.resolved_field_count,
            observation_only_field_count=view.observation_only_field_count,
            mixed_field_count=view.mixed_field_count,
            records=view.records,
            knowledge_view_manifest_hash=_build_manifest_hash(
                view=view,
                subject_keys_filter=normalized_subject_keys,
            ),
        ),
        subject_keys=subject_keys,
    )
