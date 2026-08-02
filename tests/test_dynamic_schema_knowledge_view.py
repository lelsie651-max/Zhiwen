from __future__ import annotations

import asyncio
import hashlib
import inspect
from dataclasses import replace
from datetime import datetime, timezone
import uuid

import pytest

from app.schemas.dynamic_schema_knowledge_view import DynamicSchemaKnowledgeView
from app.schemas.dynamic_schema_review_projection import (
    DynamicSchemaReviewProjection,
    DynamicSchemaReviewedFact,
    DynamicSchemaReviewedField,
    DynamicSchemaReviewedRecord,
)
from app.schemas.dynamic_schema_ufl_projection import DynamicSchemaUFLProjectedField
from app.schemas.ufl_fact_snapshot import (
    UFLFactEvidenceLocator,
    UFLFactEvidenceSnapshot,
    UFLFactSnapshot,
    UFLFactValueGroupSnapshot,
    UFLFactValueSnapshot,
)
from app.services import dynamic_schema_knowledge_view as knowledge_view_service
from app.services import dynamic_schema_review_projection as review_projection_service
from app.utils.deterministic_json import freeze_deterministic_json_value


def run_async(awaitable):
    return asyncio.run(awaitable)


class SessionFactory:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self):
        self.calls += 1
        raise AssertionError("knowledge view service must not open sessions directly")


def _uuid(seed: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"dynamic-schema-knowledge:{seed}")


def _hash(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _value(fact_value_id: uuid.UUID, *, seed: str, proposal_index: int) -> UFLFactValueSnapshot:
    return UFLFactValueSnapshot(
        fact_value_id=fact_value_id,
        source_batch_id=_uuid(f"batch:{seed}:{proposal_index}"),
        source_application_id=_uuid(f"application:{seed}:{proposal_index}"),
        proposal_index=proposal_index,
        normalized_value_text=f"text:{seed}:{proposal_index}",
        value_hash=_hash(f"value:{seed}:{proposal_index}"),
        language_code=None,
        confidence=0.5,
    )


def _evidence(seed: str) -> UFLFactEvidenceSnapshot:
    return UFLFactEvidenceSnapshot(
        evidence_link_id=_uuid(f"evidence-link:{seed}"),
        evidence_id=_uuid(f"evidence:{seed}"),
        document_revision_id=_uuid("document-revision"),
        document_block_id=_uuid(f"block:{seed}"),
        locator=UFLFactEvidenceLocator(
            location_key=f"loc:{seed}",
            page_no=1,
            start_line=1,
            end_line=1,
            table_index=None,
            row_index=None,
        ),
        excerpt=f"excerpt:{seed}",
        excerpt_hash=_hash(f"excerpt:{seed}"),
        content_hash=_hash(f"content:{seed}"),
        role="supporting",
        is_primary=True,
        source_order=0,
    )


def _fact(
    *,
    seed: str,
    predicate_key: str,
    fact_value_ids: tuple[uuid.UUID, ...],
    subject_key: str = "alpha",
) -> UFLFactSnapshot:
    group = UFLFactValueGroupSnapshot(
        semantic_key_hash=_hash(f"semantic:{seed}"),
        value_type="string",
        value_json=freeze_deterministic_json_value({"seed": seed}),
        referenced_entity_id=None,
        fact_value_ids=fact_value_ids,
        values=tuple(
            _value(fact_value_id, seed=seed, proposal_index=index)
            for index, fact_value_id in enumerate(fact_value_ids)
        ),
        evidences=(_evidence(seed),),
    )
    return UFLFactSnapshot(
        fact_id=_uuid(f"fact:{seed}"),
        identity_hash=_hash(f"fact-identity:{seed}"),
        subject_kind="person",
        subject_key=subject_key,
        subject_entity_id=None,
        predicate_key=predicate_key,
        scope_key=None,
        semantic_group_count=1,
        fact_value_count=len(fact_value_ids),
        value_groups=(group,),
    )


def _source_field(
    *,
    field_key: str,
    display_order: int,
    matched_facts: tuple[UFLFactSnapshot, ...],
    predicate_key: str | None = None,
    cardinality: str = "one",
    group_key: str | None = None,
    is_required: bool = False,
    is_hidden: bool = False,
    is_title: bool = False,
    issues: tuple[str, ...] = (),
) -> DynamicSchemaUFLProjectedField:
    return DynamicSchemaUFLProjectedField(
        field_id=_uuid(f"field:{field_key}"),
        schema_version_id=_uuid("schema-version"),
        field_key=field_key,
        label=field_key.title(),
        description=None,
        predicate_key=predicate_key or field_key,
        scope_key=None,
        expected_value_type="string",
        cardinality=cardinality,
        is_required=is_required,
        is_title=is_title,
        is_summary=False,
        is_hidden=is_hidden,
        group_key=group_key,
        display_order=display_order,
        display_config=freeze_deterministic_json_value({"field": field_key}),
        validation_rules=freeze_deterministic_json_value({"field": field_key}),
        created_at=datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc),
        matched_facts=matched_facts,
        matched_fact_count=len(matched_facts),
        semantic_value_count=sum(len(fact.value_groups) for fact in matched_facts),
        is_missing=len(matched_facts) == 0,
        type_compatible=True,
        issues=issues,
    )


def _reviewed_fact_no_candidate(fact: UFLFactSnapshot) -> DynamicSchemaReviewedFact:
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


def _reviewed_fact_pending(fact: UFLFactSnapshot) -> DynamicSchemaReviewedFact:
    return DynamicSchemaReviewedFact(
        fact=fact,
        review_state="pending_review",
        candidate_id=_uuid(f"candidate:{fact.fact_id}"),
        assessment_id=_uuid(f"assessment:{fact.fact_id}"),
        resolution_basis="none",
        current_decision_id=None,
        current_decision_kind=None,
        effective_fact_value_ids=(),
        requires_review=True,
    )


def _reviewed_fact_resolved(
    fact: UFLFactSnapshot,
    *,
    fact_value_ids: tuple[uuid.UUID, ...],
    decision_kind: str = "select_one",
    resolution_basis: str = "human_selection",
) -> DynamicSchemaReviewedFact:
    return DynamicSchemaReviewedFact(
        fact=fact,
        review_state="resolved",
        candidate_id=_uuid(f"candidate:{fact.fact_id}"),
        assessment_id=_uuid(f"assessment:{fact.fact_id}"),
        resolution_basis=resolution_basis,  # type: ignore[arg-type]
        current_decision_id=_uuid(f"decision:{fact.fact_id}"),
        current_decision_kind=decision_kind,
        effective_fact_value_ids=fact_value_ids,
        requires_review=False,
    )


def _reviewed_field(
    source_field: DynamicSchemaUFLProjectedField,
    reviewed_facts: tuple[DynamicSchemaReviewedFact, ...],
) -> DynamicSchemaReviewedField:
    return DynamicSchemaReviewedField(
        source_field=source_field,
        reviewed_facts=reviewed_facts,
        review_required=any(reviewed_fact.requires_review for reviewed_fact in reviewed_facts),
        resolved_fact_count=sum(
            1 for reviewed_fact in reviewed_facts if reviewed_fact.review_state == "resolved"
        ),
        review_required_fact_count=sum(
            1 for reviewed_fact in reviewed_facts if reviewed_fact.requires_review
        ),
        effective_fact_value_ids=tuple(
            fact_value_id
            for reviewed_fact in reviewed_facts
            if reviewed_fact.review_state == "resolved"
            for fact_value_id in reviewed_fact.effective_fact_value_ids
        ),
    )


def _review_projection(
    *,
    records: tuple[DynamicSchemaReviewedRecord, ...],
    comparison_quality: str = "complete",
) -> DynamicSchemaReviewProjection:
    unique_reviewed_facts: dict[uuid.UUID, DynamicSchemaReviewedFact] = {}
    field_review_required_count = 0
    for record in records:
        for field in record.fields:
            if field.review_required:
                field_review_required_count += 1
            for reviewed_fact in field.reviewed_facts:
                unique_reviewed_facts.setdefault(reviewed_fact.fact.fact_id, reviewed_fact)
    projection = DynamicSchemaReviewProjection(
        project_id=_uuid("project"),
        schema_id=_uuid("schema"),
        schema_version_id=_uuid("schema-version"),
        orchestration_id=_uuid("orchestration"),
        extraction_run_id=_uuid("extraction-run"),
        consistency_check_application_id=_uuid("consistency-check-application"),
        source_consistency_application_id=_uuid("source-consistency-application"),
        schema_definition_manifest_hash=_hash("schema-manifest"),
        ufl_source_manifest_hash=_hash("ufl-manifest"),
        consistency_result_manifest_hash=_hash("consistency-result"),
        raw_projection_manifest_hash=_hash("raw-manifest"),
        comparison_quality=comparison_quality,  # type: ignore[arg-type]
        algorithm_name="dynamic_schema_review_projection",
        algorithm_version="1.0.0",
        record_count=len(records),
        unique_matched_fact_count=len(unique_reviewed_facts),
        resolved_fact_count=sum(
            1
            for reviewed_fact in unique_reviewed_facts.values()
            if reviewed_fact.review_state == "resolved"
        ),
        review_required_fact_count=sum(
            1 for reviewed_fact in unique_reviewed_facts.values() if reviewed_fact.requires_review
        ),
        no_candidate_fact_count=sum(
            1
            for reviewed_fact in unique_reviewed_facts.values()
            if reviewed_fact.review_state == "no_consistency_candidate"
        ),
        field_review_required_count=field_review_required_count,
        records=records,
        reviewed_projection_manifest_hash="",
    )
    return replace(
        projection,
        reviewed_projection_manifest_hash=review_projection_service._build_manifest_hash(
            projection=projection,
            subject_keys_filter=None,
        ),
    )


def _valid_review_projection() -> DynamicSchemaReviewProjection:
    title_fact = _fact(
        seed="title",
        predicate_key="title",
        fact_value_ids=(_uuid("fv-title"),),
    )
    review_fact = _fact(
        seed="review",
        predicate_key="review",
        fact_value_ids=(_uuid("fv-review"),),
    )
    resolved_fact = _fact(
        seed="resolved",
        predicate_key="resolved",
        fact_value_ids=(_uuid("fv-resolved"),),
    )
    mixed_resolved_fact = _fact(
        seed="mixed-resolved",
        predicate_key="mixed",
        fact_value_ids=(_uuid("fv-mixed-resolved"),),
    )
    mixed_observed_fact = _fact(
        seed="mixed-observed",
        predicate_key="mixed",
        fact_value_ids=(_uuid("fv-mixed-observed"),),
    )
    alpha_record = DynamicSchemaReviewedRecord(
        subject_key="alpha",
        required_missing_field_keys=("missing",),
        issue_count=1,
        fields=(
            _reviewed_field(
                _source_field(
                    field_key="title",
                    display_order=0,
                    matched_facts=(title_fact,),
                    group_key=None,
                    is_title=True,
                ),
                (_reviewed_fact_no_candidate(title_fact),),
            ),
            _reviewed_field(
                _source_field(
                    field_key="review",
                    display_order=1,
                    matched_facts=(review_fact,),
                    group_key="group-a",
                ),
                (_reviewed_fact_pending(review_fact),),
            ),
            _reviewed_field(
                _source_field(
                    field_key="missing",
                    display_order=2,
                    matched_facts=(),
                    group_key=None,
                    is_required=True,
                    is_hidden=True,
                    issues=("required_missing",),
                ),
                (),
            ),
            _reviewed_field(
                _source_field(
                    field_key="resolved",
                    display_order=3,
                    matched_facts=(resolved_fact,),
                    group_key="group-a",
                ),
                (
                    _reviewed_fact_resolved(
                        resolved_fact,
                        fact_value_ids=(_uuid("fv-resolved"),),
                    ),
                ),
            ),
            _reviewed_field(
                _source_field(
                    field_key="mixed",
                    display_order=4,
                    matched_facts=(mixed_resolved_fact, mixed_observed_fact),
                    cardinality="many",
                    group_key="group-b",
                    predicate_key="mixed",
                ),
                (
                    _reviewed_fact_resolved(
                        mixed_resolved_fact,
                        fact_value_ids=(_uuid("fv-mixed-resolved"),),
                    ),
                    _reviewed_fact_no_candidate(mixed_observed_fact),
                ),
            ),
        ),
    )
    beta_record = DynamicSchemaReviewedRecord(
        subject_key="beta",
        required_missing_field_keys=(),
        issue_count=0,
        fields=(),
    )
    return _review_projection(records=(alpha_record, beta_record))


def _install_review_source(
    monkeypatch: pytest.MonkeyPatch,
    *,
    review_projection_factory,
) -> None:
    async def fake_review_projection(
        _session_factory,
        *,
        project_id,
        schema_id,
        schema_version_id,
        orchestration_id,
        consistency_check_application_id,
        subject_keys=None,
    ):
        del project_id, schema_id, schema_version_id, orchestration_id
        del consistency_check_application_id
        return review_projection_factory(subject_keys)

    monkeypatch.setattr(
        knowledge_view_service.review_projection_service,
        "project_reviewed_orchestration_ufl_to_dynamic_schema",
        fake_review_projection,
    )


def test_build_dynamic_schema_knowledge_view_maps_five_states_and_preserves_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    review_projection = _valid_review_projection()
    _install_review_source(
        monkeypatch,
        review_projection_factory=lambda _subject_keys: review_projection,
    )
    view = run_async(
        knowledge_view_service.build_dynamic_schema_knowledge_view(
            SessionFactory(),
            project_id=review_projection.project_id,
            schema_id=review_projection.schema_id,
            schema_version_id=review_projection.schema_version_id,
            orchestration_id=review_projection.orchestration_id,
            consistency_check_application_id=review_projection.consistency_check_application_id,
        )
    )

    assert view.record_count == 2
    assert view.section_count == 3
    assert view.field_count == 5
    assert view.missing_field_count == 1
    assert view.review_required_field_count == 1
    assert view.resolved_field_count == 1
    assert view.observation_only_field_count == 1
    assert view.mixed_field_count == 1

    alpha_record = view.records[0]
    assert alpha_record.subject_key == "alpha"
    assert alpha_record.title_field_key == "title"
    assert alpha_record.has_review_required is True
    assert alpha_record.issue_count == 1
    assert [section.group_key for section in alpha_record.sections] == [None, "group-a", "group-b"]

    none_section = alpha_record.sections[0]
    assert [field.source_field.field_key for field in none_section.fields] == ["title", "missing"]
    assert none_section.fields[0].knowledge_state == "observation_only"
    assert none_section.fields[0].observed_fact_value_count == 1
    assert none_section.fields[0].effective_fact_value_ids == ()
    assert none_section.fields[0].reviewed_facts[0].fact.value_groups[0].evidences[0].excerpt == "excerpt:title"
    assert none_section.fields[1].knowledge_state == "missing"
    assert none_section.fields[1].source_field.is_hidden is True
    assert none_section.fields[1].has_schema_issues is True

    group_a_section = alpha_record.sections[1]
    assert [field.source_field.field_key for field in group_a_section.fields] == ["review", "resolved"]
    assert group_a_section.fields[0].knowledge_state == "review_required"
    assert group_a_section.fields[1].knowledge_state == "resolved"
    assert group_a_section.fields[1].effective_fact_value_ids == (_uuid("fv-resolved"),)

    group_b_section = alpha_record.sections[2]
    assert group_b_section.fields[0].knowledge_state == "mixed_reviewed_observation"
    assert group_b_section.fields[0].effective_fact_value_ids == (_uuid("fv-mixed-resolved"),)
    assert group_b_section.fields[0].observed_fact_value_count == 2


def test_build_dynamic_schema_knowledge_view_is_deterministic_and_filters_subjects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    review_projection = _valid_review_projection()
    record_by_subject = {record.subject_key: record for record in review_projection.records}

    def build_projection(subject_keys):
        records = (
            tuple(record_by_subject[key] for key in subject_keys)
            if subject_keys is not None
            else tuple(record_by_subject[key] for key in sorted(record_by_subject))
        )
        projection = replace(
            review_projection,
            record_count=len(records),
            records=records,
            reviewed_projection_manifest_hash="",
        )
        return replace(
            projection,
            reviewed_projection_manifest_hash=review_projection_service._build_manifest_hash(
                projection=projection,
                subject_keys_filter=subject_keys,
            ),
        )

    _install_review_source(monkeypatch, review_projection_factory=build_projection)
    first = run_async(
        knowledge_view_service.build_dynamic_schema_knowledge_view(
            SessionFactory(),
            project_id=review_projection.project_id,
            schema_id=review_projection.schema_id,
            schema_version_id=review_projection.schema_version_id,
            orchestration_id=review_projection.orchestration_id,
            consistency_check_application_id=review_projection.consistency_check_application_id,
            subject_keys=["beta", "alpha"],
        )
    )
    second = run_async(
        knowledge_view_service.build_dynamic_schema_knowledge_view(
            SessionFactory(),
            project_id=review_projection.project_id,
            schema_id=review_projection.schema_id,
            schema_version_id=review_projection.schema_version_id,
            orchestration_id=review_projection.orchestration_id,
            consistency_check_application_id=review_projection.consistency_check_application_id,
            subject_keys=["beta", "alpha"],
        )
    )
    partial_empty_projection = _review_projection(records=(), comparison_quality="partial")
    partial_empty_projection = replace(
        partial_empty_projection,
        reviewed_projection_manifest_hash=review_projection_service._build_manifest_hash(
            projection=replace(
                partial_empty_projection,
                reviewed_projection_manifest_hash="",
            ),
            subject_keys_filter=[],
        ),
    )
    _install_review_source(
        monkeypatch,
        review_projection_factory=lambda _subject_keys: partial_empty_projection,
    )
    empty = run_async(
        knowledge_view_service.build_dynamic_schema_knowledge_view(
            SessionFactory(),
            project_id=review_projection.project_id,
            schema_id=review_projection.schema_id,
            schema_version_id=review_projection.schema_version_id,
            orchestration_id=review_projection.orchestration_id,
            consistency_check_application_id=review_projection.consistency_check_application_id,
            subject_keys=[],
        )
    )

    assert first == second
    assert [record.subject_key for record in first.records] == ["beta", "alpha"]
    assert empty.records == ()
    assert empty.comparison_quality == "partial"
    assert empty.record_count == 0


def test_build_dynamic_schema_knowledge_view_rejects_replaced_review_child_even_if_manifest_is_resigned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    review_projection = _valid_review_projection()
    mutated_projection = replace(
        review_projection,
        records=(
            replace(
                review_projection.records[0],
                fields=(
                    replace(
                        review_projection.records[0].fields[4],
                        reviewed_facts=(
                            replace(
                                review_projection.records[0].fields[4].reviewed_facts[0],
                                effective_fact_value_ids=(),
                            ),
                            review_projection.records[0].fields[4].reviewed_facts[1],
                        ),
                    ),
                ),
            ),
            review_projection.records[1],
        ),
        reviewed_projection_manifest_hash="",
    )
    resigned_projection = replace(
        mutated_projection,
        reviewed_projection_manifest_hash=review_projection_service._build_manifest_hash(
            projection=mutated_projection,
            subject_keys_filter=None,
        ),
    )
    _install_review_source(
        monkeypatch,
        review_projection_factory=lambda _subject_keys: resigned_projection,
    )

    with pytest.raises(
        knowledge_view_service.DynamicSchemaKnowledgeViewInvariantError,
        match="dynamic_schema_knowledge_view_review_projection_mismatch",
    ):
        run_async(
            knowledge_view_service.build_dynamic_schema_knowledge_view(
                SessionFactory(),
                project_id=review_projection.project_id,
                schema_id=review_projection.schema_id,
                schema_version_id=review_projection.schema_version_id,
                orchestration_id=review_projection.orchestration_id,
                consistency_check_application_id=review_projection.consistency_check_application_id,
            )
        )


def test_build_dynamic_schema_knowledge_view_calls_public_review_authentication_and_stays_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    review_projection = _valid_review_projection()
    factory = SessionFactory()
    _install_review_source(
        monkeypatch,
        review_projection_factory=lambda _subject_keys: review_projection,
    )
    original_authenticate = (
        knowledge_view_service.review_projection_service.authenticate_dynamic_schema_review_projection
    )
    calls: list[int] = []

    def tracking_authenticate(projection, *, subject_keys):
        calls.append(1)
        return original_authenticate(projection, subject_keys=subject_keys)

    monkeypatch.setattr(
        knowledge_view_service.review_projection_service,
        "authenticate_dynamic_schema_review_projection",
        tracking_authenticate,
    )

    view = run_async(
        knowledge_view_service.build_dynamic_schema_knowledge_view(
            factory,
            project_id=review_projection.project_id,
            schema_id=review_projection.schema_id,
            schema_version_id=review_projection.schema_version_id,
            orchestration_id=review_projection.orchestration_id,
            consistency_check_application_id=review_projection.consistency_check_application_id,
        )
    )

    source = inspect.getsource(knowledge_view_service)
    assert calls == [1]
    assert factory.calls == 0
    assert "current_value_id" not in source
    assert isinstance(view, DynamicSchemaKnowledgeView)
