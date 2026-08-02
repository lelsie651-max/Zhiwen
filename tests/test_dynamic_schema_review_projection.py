from __future__ import annotations

import asyncio
import hashlib
import inspect
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
import uuid

import pytest

from app.schemas.consistency_check_persistence import (
    ConsistencyCheckApplicationLedgerRecord,
)
from app.schemas.consistency_projection import ConsistencyReviewProjectionMember
from app.schemas.dynamic_schema_review_projection import (
    DynamicSchemaReviewProjection,
)
from app.schemas.dynamic_schema_ufl_projection import (
    DynamicSchemaUFLProjectedField,
    DynamicSchemaUFLProjectedRecord,
    DynamicSchemaUFLProjection,
)
from app.schemas.effective_fact_value import (
    EffectiveFactValueProjection,
    EffectiveFactValueProjectionItem,
)
from app.schemas.ufl_fact_snapshot import (
    UFLFactEvidenceLocator,
    UFLFactEvidenceSnapshot,
    UFLFactSnapshot,
    UFLFactValueGroupSnapshot,
    UFLFactValueSnapshot,
)
from app.services import dynamic_schema_review_projection as review_projection_service
from app.services import dynamic_schema_ufl_projection as raw_projection_service
from app.utils.deterministic_json import freeze_deterministic_json_value


def run_async(awaitable):
    return asyncio.run(awaitable)


class SessionFactory:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self):
        self.calls += 1
        raise AssertionError("review projection service must not open sessions directly")


def _uuid(seed: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"dynamic-schema-review:{seed}")


def _hash(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _value(
    *,
    seed: str,
    fact_value_id: uuid.UUID,
    proposal_index: int,
) -> UFLFactValueSnapshot:
    return UFLFactValueSnapshot(
        fact_value_id=fact_value_id,
        source_batch_id=_uuid(f"batch:{seed}:{proposal_index}"),
        source_application_id=_uuid(f"application:{seed}:{proposal_index}"),
        proposal_index=proposal_index,
        normalized_value_text=f"{seed}:{proposal_index}",
        value_hash=_hash(f"value:{seed}:{proposal_index}"),
        language_code=None,
        confidence=0.5,
    )


def _evidence(*, seed: str, source_order: int) -> UFLFactEvidenceSnapshot:
    return UFLFactEvidenceSnapshot(
        evidence_link_id=_uuid(f"evidence-link:{seed}"),
        evidence_id=_uuid(f"evidence:{seed}"),
        document_revision_id=_uuid("document-revision"),
        document_block_id=_uuid(f"block:{seed}"),
        locator=UFLFactEvidenceLocator(
            location_key=f"loc:{seed}",
            page_no=1,
            start_line=source_order + 1,
            end_line=source_order + 1,
            table_index=None,
            row_index=None,
        ),
        excerpt=f"excerpt:{seed}",
        excerpt_hash=_hash(f"excerpt:{seed}"),
        content_hash=_hash(f"content:{seed}"),
        role="supporting",
        is_primary=True,
        source_order=source_order,
    )


def _group(
    *,
    seed: str,
    fact_value_ids: tuple[uuid.UUID, ...],
    value_json: object,
) -> UFLFactValueGroupSnapshot:
    return UFLFactValueGroupSnapshot(
        semantic_key_hash=_hash(f"semantic:{seed}"),
        value_type="string",
        value_json=freeze_deterministic_json_value(value_json),
        referenced_entity_id=None,
        fact_value_ids=fact_value_ids,
        values=tuple(
            _value(
                seed=seed,
                fact_value_id=fact_value_id,
                proposal_index=index,
            )
            for index, fact_value_id in enumerate(fact_value_ids)
        ),
        evidences=tuple(
            _evidence(seed=f"{seed}:{index}", source_order=index)
            for index, _fact_value_id in enumerate(fact_value_ids)
        ),
    )


def _fact(
    *,
    seed: str,
    subject_key: str = "alpha",
    predicate_key: str,
    scope_key: str | None = None,
    group_specs: tuple[tuple[str, tuple[uuid.UUID, ...], object], ...],
) -> UFLFactSnapshot:
    groups = tuple(
        _group(seed=group_seed, fact_value_ids=fact_value_ids, value_json=value_json)
        for group_seed, fact_value_ids, value_json in group_specs
    )
    return UFLFactSnapshot(
        fact_id=_uuid(f"fact:{seed}"),
        identity_hash=_hash(f"fact-identity:{seed}"),
        subject_kind="person",
        subject_key=subject_key,
        subject_entity_id=None,
        predicate_key=predicate_key,
        scope_key=scope_key,
        semantic_group_count=len(groups),
        fact_value_count=sum(len(group.values) for group in groups),
        value_groups=groups,
    )


def _field(
    *,
    field_key: str,
    display_order: int,
    matched_facts: tuple[UFLFactSnapshot, ...],
    predicate_key: str | None = None,
    is_required: bool = False,
    issues: tuple[str, ...] = (),
) -> DynamicSchemaUFLProjectedField:
    return DynamicSchemaUFLProjectedField(
        field_id=_uuid(f"field:{field_key}"),
        schema_version_id=_uuid("schema-version"),
        field_key=field_key,
        label=field_key.title(),
        description=None,
        predicate_key=field_key if predicate_key is None else predicate_key,
        scope_key=None,
        expected_value_type="string",
        cardinality="one",
        is_required=is_required,
        is_title=False,
        is_summary=False,
        is_hidden=False,
        group_key=None,
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


def _record(
    *,
    subject_key: str,
    fields: tuple[DynamicSchemaUFLProjectedField, ...],
    required_missing_field_keys: tuple[str, ...] = (),
    issue_count: int = 0,
) -> DynamicSchemaUFLProjectedRecord:
    return DynamicSchemaUFLProjectedRecord(
        subject_key=subject_key,
        fields=fields,
        required_missing_field_keys=required_missing_field_keys,
        issue_count=issue_count,
    )


def _raw_projection(
    *,
    records: tuple[DynamicSchemaUFLProjectedRecord, ...],
    comparison_quality: str = "complete",
    projection_manifest_hash: str = "",
) -> DynamicSchemaUFLProjection:
    projection = DynamicSchemaUFLProjection(
        project_id=_uuid("project"),
        schema_id=_uuid("schema"),
        schema_version_id=_uuid("schema-version"),
        orchestration_id=_uuid("orchestration"),
        extraction_run_id=_uuid("extraction-run"),
        schema_definition_manifest_hash=_hash("schema-manifest"),
        ufl_source_manifest_hash=_hash("ufl-manifest"),
        comparison_quality=comparison_quality,  # type: ignore[arg-type]
        subject_kind="person",
        algorithm_name="dynamic_schema_ufl_projection",
        algorithm_version="1.0.0",
        record_count=len(records),
        projected_field_count=sum(len(record.fields) for record in records),
        required_missing_count=sum(
            len(record.required_missing_field_keys) for record in records
        ),
        issue_count=sum(record.issue_count for record in records),
        records=records,
        projection_manifest_hash="",
    )
    manifest_hash = projection_manifest_hash or raw_projection_service._build_manifest_hash(
        projection=projection,
        subject_keys_filter=None,
    )
    return replace(projection, projection_manifest_hash=manifest_hash)


def _member(fact_value_id: uuid.UUID) -> ConsistencyReviewProjectionMember:
    return ConsistencyReviewProjectionMember(
        fact_value_id=fact_value_id,
        value_type="string",
        value_json=f"value:{fact_value_id}",
        normalized_value_text=f"text:{fact_value_id}",
        referenced_entity_id=None,
        selected_by_current_decision=False,
        current_selection_order=None,
        evidences=(),
    )


def _effective_item(
    *,
    fact: UFLFactSnapshot,
    resolution_status: str,
    resolution_basis: str,
    effective_fact_value_ids: tuple[uuid.UUID, ...],
    candidate_member_ids: tuple[uuid.UUID, ...] | None = None,
    decision_kind: str | None = None,
    agent_verdict: str | None = None,
    review_status: str | None = None,
) -> EffectiveFactValueProjectionItem:
    candidate_member_ids = (
        candidate_member_ids
        if candidate_member_ids is not None
        else tuple(
            fact_value_id
            for group in fact.value_groups
            for fact_value_id in group.fact_value_ids
        )
    )
    if agent_verdict is None:
        if resolution_status == "unreviewed_compatible":
            agent_verdict = "compatible"
        elif resolution_status == "pending_review":
            agent_verdict = "conflict"
        else:
            agent_verdict = "conflict"
    if review_status is None:
        if resolution_status == "pending_review":
            review_status = "pending_review"
        elif resolution_status == "deferred":
            review_status = "deferred"
        elif resolution_status == "unreviewed_compatible":
            review_status = "not_required"
        else:
            review_status = "reviewed"
    return EffectiveFactValueProjectionItem(
        fact_id=fact.fact_id,
        candidate_id=_uuid(f"candidate:{fact.fact_id}"),
        assessment_id=_uuid(f"assessment:{fact.fact_id}"),
        agent_verdict=agent_verdict,
        review_status=review_status,
        resolution_status=resolution_status,  # type: ignore[arg-type]
        resolution_basis=resolution_basis,  # type: ignore[arg-type]
        current_decision_id=(
            None if decision_kind is None else _uuid(f"decision:{fact.fact_id}")
        ),
        current_decision_kind=decision_kind,
        effective_fact_value_ids=effective_fact_value_ids,
        candidate_members=tuple(_member(fact_value_id) for fact_value_id in candidate_member_ids),
    )


def _effective_projection(
    *,
    items: tuple[EffectiveFactValueProjectionItem, ...],
    source_consistency_application_id: uuid.UUID | None = None,
    result_manifest_hash: str | None = None,
) -> EffectiveFactValueProjection:
    return EffectiveFactValueProjection(
        project_id=_uuid("project"),
        consistency_check_application_id=_uuid("consistency-check-application"),
        source_consistency_application_id=source_consistency_application_id
        or _uuid("source-consistency-application"),
        result_manifest_hash=result_manifest_hash or _hash("consistency-result"),
        fact_count=len(items),
        resolved_count=sum(1 for item in items if item.resolution_status == "resolved"),
        pending_count=sum(
            1
            for item in items
            if item.resolution_status in {"pending_review", "unreviewed_compatible"}
        ),
        deferred_count=sum(
            1 for item in items if item.resolution_status == "deferred"
        ),
        items=items,
    )


def _authenticated_context(
    *,
    project_id: uuid.UUID | None = None,
    orchestration_id: uuid.UUID | None = None,
    result_manifest_hash: str | None = None,
    source_consistency_application_id: uuid.UUID | None = None,
) -> review_projection_service.consistency_persistence_service.AuthenticatedConsistencyCheckLedgerProjectionContext:
    application = ConsistencyCheckApplicationLedgerRecord(
        id=_uuid("consistency-check-application"),
        project_id=project_id or _uuid("project"),
        consistency_application_id=source_consistency_application_id
        or _uuid("source-consistency-application"),
        orchestration_id=orchestration_id or _uuid("orchestration"),
        source_result_manifest_hash=_hash("source-result"),
        plan_manifest_hash=_hash("plan"),
        execution_identity_hash=_hash("execution"),
        result_manifest_hash=result_manifest_hash or _hash("consistency-result"),
        prompt_contract_hash=_hash("prompt"),
        provider="provider",
        requested_model="model",
        executor_name="executor",
        executor_version="1.0.0",
        batch_count=1,
        executed_batch_count=1,
        skipped_empty_batch_count=0,
        inference_run_count=1,
        assessment_count=1,
        created_at=datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc),
    )
    return review_projection_service.consistency_persistence_service.AuthenticatedConsistencyCheckLedgerProjectionContext(
        application=application,
        authenticated_source=SimpleNamespace(),
        candidate_bundles=(),
        source_rows=(),
        batches=(),
        assessments=(),
        citations=(),
    )


def _install_sources(
    monkeypatch: pytest.MonkeyPatch,
    *,
    raw_projection_factory,
    authenticated_context,
    effective_projection,
) -> None:
    async def fake_raw_projection(
        _session_factory,
        *,
        project_id,
        schema_id,
        schema_version_id,
        orchestration_id,
        subject_keys=None,
    ):
        del project_id, schema_id, schema_version_id, orchestration_id
        return raw_projection_factory(subject_keys)

    async def fake_authenticated_context(
        _session_factory,
        *,
        project_id,
        consistency_check_application_id,
    ):
        del project_id, consistency_check_application_id
        return authenticated_context

    async def fake_effective_projection(
        _session_factory,
        *,
        project_id,
        consistency_check_application_id,
    ):
        del project_id, consistency_check_application_id
        return effective_projection

    monkeypatch.setattr(
        review_projection_service.raw_projection_service,
        "project_orchestration_ufl_to_dynamic_schema",
        fake_raw_projection,
    )
    monkeypatch.setattr(
        review_projection_service.consistency_persistence_service,
        "authenticate_persisted_consistency_check_application",
        fake_authenticated_context,
    )
    monkeypatch.setattr(
        review_projection_service.effective_fact_value_service,
        "get_effective_fact_value_projection",
        fake_effective_projection,
    )


def _base_projection() -> tuple[DynamicSchemaUFLProjection, dict[str, UFLFactSnapshot]]:
    no_candidate = _fact(
        seed="no-candidate",
        predicate_key="no_candidate",
        group_specs=(("no-candidate-group", (_uuid("fv-no-candidate"),), "raw"),),
    )
    select_one = _fact(
        seed="select-one",
        predicate_key="select_one",
        group_specs=((
            "select-one-group",
            (_uuid("fv-select-one-a"), _uuid("fv-select-one-b")),
            "value",
        ),),
    )
    keep_multiple = _fact(
        seed="keep-multiple",
        predicate_key="keep_multiple",
        group_specs=((
            "keep-multiple-group",
            (_uuid("fv-keep-multiple-a"), _uuid("fv-keep-multiple-b")),
            "value",
        ),),
    )
    confirm_compatible = _fact(
        seed="confirm-compatible",
        predicate_key="confirm_compatible",
        group_specs=((
            "confirm-compatible-group",
            (_uuid("fv-confirm-compatible-a"), _uuid("fv-confirm-compatible-b")),
            "value",
        ),),
    )
    pending = _fact(
        seed="pending",
        predicate_key="pending",
        group_specs=(("pending-group", (_uuid("fv-pending"),), "value"),),
    )
    deferred = _fact(
        seed="deferred",
        predicate_key="deferred",
        group_specs=(("deferred-group", (_uuid("fv-deferred"),), "value"),),
    )
    unreviewed = _fact(
        seed="unreviewed",
        predicate_key="unreviewed",
        group_specs=(("unreviewed-group", (_uuid("fv-unreviewed"),), "value"),),
    )
    alpha_record = _record(
        subject_key="alpha",
        required_missing_field_keys=("missing_required",),
        issue_count=1,
        fields=(
            _field(field_key="no_candidate", display_order=0, matched_facts=(no_candidate,)),
            _field(field_key="select_one", display_order=1, matched_facts=(select_one,)),
            _field(
                field_key="keep_multiple",
                display_order=2,
                matched_facts=(keep_multiple,),
            ),
            _field(
                field_key="confirm_compatible",
                display_order=3,
                matched_facts=(confirm_compatible,),
            ),
            _field(
                field_key="confirm_compatible_duplicate",
                display_order=4,
                matched_facts=(confirm_compatible,),
                predicate_key="confirm_compatible",
            ),
            _field(field_key="pending", display_order=5, matched_facts=(pending,)),
            _field(field_key="deferred", display_order=6, matched_facts=(deferred,)),
            _field(field_key="unreviewed", display_order=7, matched_facts=(unreviewed,)),
            _field(
                field_key="missing_required",
                display_order=8,
                matched_facts=(),
                is_required=True,
                issues=("required_missing",),
            ),
        ),
    )
    beta_record = _record(subject_key="beta", fields=())
    projection = _raw_projection(records=(alpha_record, beta_record))
    return projection, {
        "no_candidate": no_candidate,
        "select_one": select_one,
        "keep_multiple": keep_multiple,
        "confirm_compatible": confirm_compatible,
        "pending": pending,
        "deferred": deferred,
        "unreviewed": unreviewed,
    }


def _build_valid_review_projection(monkeypatch: pytest.MonkeyPatch) -> DynamicSchemaReviewProjection:
    raw_projection, facts = _base_projection()
    effective_projection = _effective_projection(
        items=(
            _effective_item(
                fact=facts["select_one"],
                resolution_status="resolved",
                resolution_basis="human_selection",
                effective_fact_value_ids=(_uuid("fv-select-one-b"),),
                decision_kind="select_one",
            ),
            _effective_item(
                fact=facts["keep_multiple"],
                resolution_status="resolved",
                resolution_basis="human_selection",
                effective_fact_value_ids=(
                    _uuid("fv-keep-multiple-b"),
                    _uuid("fv-keep-multiple-a"),
                ),
                decision_kind="keep_multiple",
            ),
            _effective_item(
                fact=facts["confirm_compatible"],
                resolution_status="resolved",
                resolution_basis="human_confirmed_compatibility",
                effective_fact_value_ids=(
                    _uuid("fv-confirm-compatible-a"),
                    _uuid("fv-confirm-compatible-b"),
                ),
                decision_kind="confirm_compatible",
            ),
            _effective_item(
                fact=facts["pending"],
                resolution_status="pending_review",
                resolution_basis="none",
                effective_fact_value_ids=(),
            ),
            _effective_item(
                fact=facts["deferred"],
                resolution_status="deferred",
                resolution_basis="none",
                effective_fact_value_ids=(),
                decision_kind="defer",
            ),
            _effective_item(
                fact=facts["unreviewed"],
                resolution_status="unreviewed_compatible",
                resolution_basis="none",
                effective_fact_value_ids=(),
            ),
        )
    )
    _install_sources(
        monkeypatch,
        raw_projection_factory=lambda _subject_keys: raw_projection,
        authenticated_context=_authenticated_context(),
        effective_projection=effective_projection,
    )
    return run_async(
        review_projection_service.project_reviewed_orchestration_ufl_to_dynamic_schema(
            SessionFactory(),
            project_id=raw_projection.project_id,
            schema_id=raw_projection.schema_id,
            schema_version_id=raw_projection.schema_version_id,
            orchestration_id=raw_projection.orchestration_id,
            consistency_check_application_id=_uuid("consistency-check-application"),
        )
    )


def test_project_reviewed_orchestration_ufl_to_dynamic_schema_maps_review_states_and_preserves_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_projection, facts = _base_projection()
    effective_projection = _effective_projection(
        items=(
            _effective_item(
                fact=facts["select_one"],
                resolution_status="resolved",
                resolution_basis="human_selection",
                effective_fact_value_ids=(_uuid("fv-select-one-b"),),
                decision_kind="select_one",
            ),
            _effective_item(
                fact=facts["keep_multiple"],
                resolution_status="resolved",
                resolution_basis="human_selection",
                effective_fact_value_ids=(
                    _uuid("fv-keep-multiple-b"),
                    _uuid("fv-keep-multiple-a"),
                ),
                decision_kind="keep_multiple",
            ),
            _effective_item(
                fact=facts["confirm_compatible"],
                resolution_status="resolved",
                resolution_basis="human_confirmed_compatibility",
                effective_fact_value_ids=(
                    _uuid("fv-confirm-compatible-a"),
                    _uuid("fv-confirm-compatible-b"),
                ),
                decision_kind="confirm_compatible",
            ),
            _effective_item(
                fact=facts["pending"],
                resolution_status="pending_review",
                resolution_basis="none",
                effective_fact_value_ids=(),
            ),
            _effective_item(
                fact=facts["deferred"],
                resolution_status="deferred",
                resolution_basis="none",
                effective_fact_value_ids=(),
                decision_kind="defer",
            ),
            _effective_item(
                fact=facts["unreviewed"],
                resolution_status="unreviewed_compatible",
                resolution_basis="none",
                effective_fact_value_ids=(),
            ),
        )
    )
    authenticated_context = _authenticated_context()
    factory = SessionFactory()
    _install_sources(
        monkeypatch,
        raw_projection_factory=lambda _subject_keys: raw_projection,
        authenticated_context=authenticated_context,
        effective_projection=effective_projection,
    )

    projection = run_async(
        review_projection_service.project_reviewed_orchestration_ufl_to_dynamic_schema(
            factory,
            project_id=raw_projection.project_id,
            schema_id=raw_projection.schema_id,
            schema_version_id=raw_projection.schema_version_id,
            orchestration_id=raw_projection.orchestration_id,
            consistency_check_application_id=_uuid("consistency-check-application"),
        )
    )

    assert factory.calls == 0
    assert projection.record_count == 2
    assert projection.unique_matched_fact_count == 7
    assert projection.resolved_fact_count == 3
    assert projection.review_required_fact_count == 3
    assert projection.no_candidate_fact_count == 1
    assert projection.field_review_required_count == 3

    alpha_record = projection.records[0]
    assert alpha_record.subject_key == "alpha"
    assert alpha_record.required_missing_field_keys == ("missing_required",)
    assert alpha_record.issue_count == 1
    assert alpha_record.fields[8].source_field.issues == ("required_missing",)
    assert alpha_record.fields[8].reviewed_facts == ()

    no_candidate_fact = alpha_record.fields[0].reviewed_facts[0]
    assert no_candidate_fact.review_state == "no_consistency_candidate"
    assert no_candidate_fact.requires_review is False
    assert no_candidate_fact.effective_fact_value_ids == ()

    select_one_fact = alpha_record.fields[1].reviewed_facts[0]
    assert select_one_fact.review_state == "resolved"
    assert select_one_fact.current_decision_kind == "select_one"
    assert select_one_fact.effective_fact_value_ids == (_uuid("fv-select-one-b"),)

    keep_multiple_fact = alpha_record.fields[2].reviewed_facts[0]
    assert keep_multiple_fact.review_state == "resolved"
    assert keep_multiple_fact.effective_fact_value_ids == (
        _uuid("fv-keep-multiple-b"),
        _uuid("fv-keep-multiple-a"),
    )

    confirm_fact = alpha_record.fields[3].reviewed_facts[0]
    assert confirm_fact.review_state == "resolved"
    assert confirm_fact.current_decision_kind == "confirm_compatible"
    assert confirm_fact.effective_fact_value_ids == (
        _uuid("fv-confirm-compatible-a"),
        _uuid("fv-confirm-compatible-b"),
    )
    assert alpha_record.fields[4].reviewed_facts[0] == confirm_fact

    pending_fact = alpha_record.fields[5].reviewed_facts[0]
    assert pending_fact.review_state == "pending_review"
    assert pending_fact.requires_review is True
    assert pending_fact.effective_fact_value_ids == ()

    deferred_fact = alpha_record.fields[6].reviewed_facts[0]
    assert deferred_fact.review_state == "deferred"
    assert deferred_fact.requires_review is True
    assert deferred_fact.current_decision_kind == "defer"

    compatible_fact = alpha_record.fields[7].reviewed_facts[0]
    assert compatible_fact.review_state == "unreviewed_compatible"
    assert compatible_fact.requires_review is True

    confirm_field = alpha_record.fields[3]
    assert confirm_field.review_required is False
    assert confirm_field.resolved_fact_count == 1
    assert confirm_field.review_required_fact_count == 0
    assert confirm_field.effective_fact_value_ids == (
        _uuid("fv-confirm-compatible-a"),
        _uuid("fv-confirm-compatible-b"),
    )
    assert (
        confirm_field.reviewed_facts[0].fact.value_groups[0].values[0].fact_value_id
        == _uuid("fv-confirm-compatible-a")
    )
    assert (
        confirm_field.reviewed_facts[0].fact.value_groups[0].evidences[0].excerpt
        == "excerpt:confirm-compatible-group:0"
    )


def test_project_reviewed_orchestration_ufl_to_dynamic_schema_subject_filter_order_and_manifest_are_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_projection, _facts = _base_projection()
    record_by_subject = {record.subject_key: record for record in raw_projection.records}

    def build_raw_projection(subject_keys: list[str] | None):
        if subject_keys is None:
            records = tuple(record_by_subject[key] for key in sorted(record_by_subject))
        else:
            records = tuple(record_by_subject[key] for key in subject_keys if key in record_by_subject)
        projection = replace(
            raw_projection,
            record_count=len(records),
            projected_field_count=sum(len(record.fields) for record in records),
            required_missing_count=sum(
                len(record.required_missing_field_keys) for record in records
            ),
            issue_count=sum(record.issue_count for record in records),
            records=records,
            projection_manifest_hash="",
        )
        return replace(
            projection,
            projection_manifest_hash=raw_projection_service._build_manifest_hash(
                projection=projection,
                subject_keys_filter=subject_keys,
            ),
        )

    _install_sources(
        monkeypatch,
        raw_projection_factory=build_raw_projection,
        authenticated_context=_authenticated_context(),
        effective_projection=_effective_projection(items=()),
    )

    first = run_async(
        review_projection_service.project_reviewed_orchestration_ufl_to_dynamic_schema(
            SessionFactory(),
            project_id=raw_projection.project_id,
            schema_id=raw_projection.schema_id,
            schema_version_id=raw_projection.schema_version_id,
            orchestration_id=raw_projection.orchestration_id,
            consistency_check_application_id=_uuid("consistency-check-application"),
            subject_keys=["beta", "alpha"],
        )
    )
    second = run_async(
        review_projection_service.project_reviewed_orchestration_ufl_to_dynamic_schema(
            SessionFactory(),
            project_id=raw_projection.project_id,
            schema_id=raw_projection.schema_id,
            schema_version_id=raw_projection.schema_version_id,
            orchestration_id=raw_projection.orchestration_id,
            consistency_check_application_id=_uuid("consistency-check-application"),
            subject_keys=["beta", "alpha"],
        )
    )
    unfiltered = run_async(
        review_projection_service.project_reviewed_orchestration_ufl_to_dynamic_schema(
            SessionFactory(),
            project_id=raw_projection.project_id,
            schema_id=raw_projection.schema_id,
            schema_version_id=raw_projection.schema_version_id,
            orchestration_id=raw_projection.orchestration_id,
            consistency_check_application_id=_uuid("consistency-check-application"),
        )
    )

    assert first == second
    assert [record.subject_key for record in first.records] == ["beta", "alpha"]
    assert first.reviewed_projection_manifest_hash != unfiltered.reviewed_projection_manifest_hash


@pytest.mark.parametrize(
    "mutate_effective_projection",
    [
        lambda projection, facts: replace(
            projection,
            items=(
                replace(
                    projection.items[0],
                    effective_fact_value_ids=(_uuid("unknown-effective-id"),),
                ),
            )
            + projection.items[1:],
        ),
        lambda projection, facts: replace(
            projection,
            items=(
                replace(
                    projection.items[0],
                    candidate_members=(
                        _member(_uuid("fv-select-one-a")),
                        _member(_uuid("fv-select-one-a")),
                    ),
                ),
            )
            + projection.items[1:],
        ),
        lambda projection, facts: replace(
            projection,
            items=(
                replace(
                    projection.items[0],
                    effective_fact_value_ids=(_uuid("fv-keep-multiple-a"),),
                ),
            )
            + projection.items[1:],
        ),
    ],
)
def test_project_reviewed_orchestration_ufl_to_dynamic_schema_rejects_invalid_fact_value_bindings(
    monkeypatch: pytest.MonkeyPatch,
    mutate_effective_projection,
) -> None:
    raw_projection, facts = _base_projection()
    effective_projection = _effective_projection(
        items=(
            _effective_item(
                fact=facts["select_one"],
                resolution_status="resolved",
                resolution_basis="human_selection",
                effective_fact_value_ids=(_uuid("fv-select-one-a"),),
                decision_kind="select_one",
            ),
            _effective_item(
                fact=facts["keep_multiple"],
                resolution_status="resolved",
                resolution_basis="human_selection",
                effective_fact_value_ids=(
                    _uuid("fv-keep-multiple-a"),
                    _uuid("fv-keep-multiple-b"),
                ),
                decision_kind="keep_multiple",
            ),
        )
    )
    _install_sources(
        monkeypatch,
        raw_projection_factory=lambda _subject_keys: raw_projection,
        authenticated_context=_authenticated_context(),
        effective_projection=mutate_effective_projection(effective_projection, facts),
    )

    with pytest.raises(
        review_projection_service.DynamicSchemaReviewProjectionInvariantError,
        match="dynamic_schema_review_projection_effective_projection_mismatch",
    ):
        run_async(
            review_projection_service.project_reviewed_orchestration_ufl_to_dynamic_schema(
                SessionFactory(),
                project_id=raw_projection.project_id,
                schema_id=raw_projection.schema_id,
                schema_version_id=raw_projection.schema_version_id,
                orchestration_id=raw_projection.orchestration_id,
                consistency_check_application_id=_uuid("consistency-check-application"),
            )
        )


@pytest.mark.parametrize(
    ("authenticated_context", "effective_projection", "expected_code"),
    [
        (
            _authenticated_context(project_id=_uuid("other-project")),
            _effective_projection(items=()),
            "dynamic_schema_review_projection_source_application_mismatch",
        ),
        (
            _authenticated_context(orchestration_id=_uuid("other-orchestration")),
            _effective_projection(items=()),
            "dynamic_schema_review_projection_source_application_mismatch",
        ),
        (
            _authenticated_context(),
            _effective_projection(
                items=(),
                source_consistency_application_id=_uuid("other-source-consistency-application"),
            ),
            "dynamic_schema_review_projection_effective_projection_mismatch",
        ),
        (
            _authenticated_context(),
            _effective_projection(
                items=(),
                result_manifest_hash=_hash("other-consistency-result"),
            ),
            "dynamic_schema_review_projection_effective_projection_mismatch",
        ),
    ],
)
def test_project_reviewed_orchestration_ufl_to_dynamic_schema_rejects_source_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    authenticated_context,
    effective_projection,
    expected_code: str,
) -> None:
    raw_projection, _facts = _base_projection()
    _install_sources(
        monkeypatch,
        raw_projection_factory=lambda _subject_keys: raw_projection,
        authenticated_context=authenticated_context,
        effective_projection=effective_projection,
    )

    with pytest.raises(
        review_projection_service.DynamicSchemaReviewProjectionInvariantError,
        match=expected_code,
    ):
        run_async(
            review_projection_service.project_reviewed_orchestration_ufl_to_dynamic_schema(
                SessionFactory(),
                project_id=raw_projection.project_id,
                schema_id=raw_projection.schema_id,
                schema_version_id=raw_projection.schema_version_id,
                orchestration_id=raw_projection.orchestration_id,
                consistency_check_application_id=_uuid("consistency-check-application"),
            )
        )


@pytest.mark.parametrize(
    "mutate_projection",
    [
        lambda projection: replace(projection, fact_count=projection.fact_count + 1),
        lambda projection: replace(
            projection,
            resolved_count=projection.resolved_count + 1,
        ),
        lambda projection: replace(
            projection,
            pending_count=projection.pending_count + 1,
        ),
        lambda projection: replace(
            projection,
            deferred_count=projection.deferred_count + 1,
        ),
    ],
)
def test_project_reviewed_orchestration_ufl_to_dynamic_schema_rejects_effective_count_drift(
    monkeypatch: pytest.MonkeyPatch,
    mutate_projection,
) -> None:
    raw_projection, facts = _base_projection()
    effective_projection = _effective_projection(
        items=(
            _effective_item(
                fact=facts["select_one"],
                resolution_status="resolved",
                resolution_basis="human_selection",
                effective_fact_value_ids=(_uuid("fv-select-one-a"),),
                decision_kind="select_one",
            ),
        )
    )
    _install_sources(
        monkeypatch,
        raw_projection_factory=lambda _subject_keys: raw_projection,
        authenticated_context=_authenticated_context(),
        effective_projection=mutate_projection(effective_projection),
    )

    with pytest.raises(
        review_projection_service.DynamicSchemaReviewProjectionInvariantError,
        match="dynamic_schema_review_projection_effective_projection_mismatch",
    ):
        run_async(
            review_projection_service.project_reviewed_orchestration_ufl_to_dynamic_schema(
                SessionFactory(),
                project_id=raw_projection.project_id,
                schema_id=raw_projection.schema_id,
                schema_version_id=raw_projection.schema_version_id,
                orchestration_id=raw_projection.orchestration_id,
                consistency_check_application_id=_uuid("consistency-check-application"),
            )
        )


def test_project_reviewed_orchestration_ufl_to_dynamic_schema_allows_unused_effective_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_projection, facts = _base_projection()
    effective_projection = _effective_projection(
        items=(
            _effective_item(
                fact=facts["select_one"],
                resolution_status="resolved",
                resolution_basis="human_selection",
                effective_fact_value_ids=(_uuid("fv-select-one-a"),),
                decision_kind="select_one",
            ),
            EffectiveFactValueProjectionItem(
                fact_id=_uuid("unused-fact"),
                candidate_id=_uuid("unused-candidate"),
                assessment_id=_uuid("unused-assessment"),
                agent_verdict="conflict",
                review_status="reviewed",
                resolution_status="resolved",
                resolution_basis="human_selection",
                current_decision_id=_uuid("unused-decision"),
                current_decision_kind="select_one",
                effective_fact_value_ids=(_uuid("unused-fv"),),
                candidate_members=(_member(_uuid("unused-fv")),),
            ),
        )
    )
    _install_sources(
        monkeypatch,
        raw_projection_factory=lambda _subject_keys: raw_projection,
        authenticated_context=_authenticated_context(),
        effective_projection=effective_projection,
    )

    projection = run_async(
        review_projection_service.project_reviewed_orchestration_ufl_to_dynamic_schema(
            SessionFactory(),
            project_id=raw_projection.project_id,
            schema_id=raw_projection.schema_id,
            schema_version_id=raw_projection.schema_version_id,
            orchestration_id=raw_projection.orchestration_id,
            consistency_check_application_id=_uuid("consistency-check-application"),
        )
    )

    assert projection.unique_matched_fact_count == 7
    assert all(
        reviewed_fact.fact.fact_id != _uuid("unused-fact")
        for record in projection.records
        for field in record.fields
        for reviewed_fact in field.reviewed_facts
    )


def test_project_reviewed_orchestration_ufl_to_dynamic_schema_handles_zero_fact_and_zero_assessment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_projection = _raw_projection(records=())
    _install_sources(
        monkeypatch,
        raw_projection_factory=lambda _subject_keys: raw_projection,
        authenticated_context=_authenticated_context(),
        effective_projection=_effective_projection(items=()),
    )

    projection = run_async(
        review_projection_service.project_reviewed_orchestration_ufl_to_dynamic_schema(
            SessionFactory(),
            project_id=raw_projection.project_id,
            schema_id=raw_projection.schema_id,
            schema_version_id=raw_projection.schema_version_id,
            orchestration_id=raw_projection.orchestration_id,
            consistency_check_application_id=_uuid("consistency-check-application"),
        )
    )

    assert projection.records == ()
    assert projection.record_count == 0
    assert projection.unique_matched_fact_count == 0
    assert projection.resolved_fact_count == 0
    assert projection.review_required_fact_count == 0
    assert projection.no_candidate_fact_count == 0
    assert projection.field_review_required_count == 0


def test_project_reviewed_orchestration_ufl_to_dynamic_schema_rejects_invalid_subject_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_projection, _facts = _base_projection()
    _install_sources(
        monkeypatch,
        raw_projection_factory=lambda _subject_keys: raw_projection,
        authenticated_context=_authenticated_context(),
        effective_projection=_effective_projection(items=()),
    )

    with pytest.raises(
        review_projection_service.DynamicSchemaReviewProjectionStateError,
        match="dynamic_schema_review_projection_subject_keys_invalid",
    ):
        run_async(
            review_projection_service.project_reviewed_orchestration_ufl_to_dynamic_schema(
                SessionFactory(),
                project_id=raw_projection.project_id,
                schema_id=raw_projection.schema_id,
                schema_version_id=raw_projection.schema_version_id,
                orchestration_id=raw_projection.orchestration_id,
                consistency_check_application_id=_uuid("consistency-check-application"),
                subject_keys=["alpha", "alpha"],
            )
        )


def test_project_reviewed_orchestration_ufl_to_dynamic_schema_does_not_open_sessions_or_read_current_value_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_projection, _facts = _base_projection()
    factory = SessionFactory()
    _install_sources(
        monkeypatch,
        raw_projection_factory=lambda _subject_keys: raw_projection,
        authenticated_context=_authenticated_context(),
        effective_projection=_effective_projection(items=()),
    )

    projection = run_async(
        review_projection_service.project_reviewed_orchestration_ufl_to_dynamic_schema(
            factory,
            project_id=raw_projection.project_id,
            schema_id=raw_projection.schema_id,
            schema_version_id=raw_projection.schema_version_id,
            orchestration_id=raw_projection.orchestration_id,
            consistency_check_application_id=_uuid("consistency-check-application"),
        )
    )

    source = inspect.getsource(review_projection_service)
    assert factory.calls == 0
    assert "current_value_id" not in source
    assert "project_orchestration_ufl_to_dynamic_schema" in source
    assert "authenticate_persisted_consistency_check_application" in source
    assert "get_effective_fact_value_projection" in source
    assert isinstance(projection, DynamicSchemaReviewProjection)


def test_project_reviewed_orchestration_ufl_to_dynamic_schema_rejects_replaced_raw_child_projection_even_if_manifest_is_resigned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_projection, _facts = _base_projection()
    mutated_projection = replace(
        raw_projection,
        records=(
            replace(
                raw_projection.records[0],
                fields=(
                    replace(
                        raw_projection.records[0].fields[0],
                        matched_facts=(
                            replace(
                                raw_projection.records[0].fields[0].matched_facts[0],
                                subject_key="other",
                            ),
                        ),
                    ),
                )
                + raw_projection.records[0].fields[1:],
            ),
            raw_projection.records[1],
        ),
        projection_manifest_hash="",
    )
    resigned_projection = replace(
        mutated_projection,
        projection_manifest_hash=raw_projection_service._build_manifest_hash(
            projection=mutated_projection,
            subject_keys_filter=None,
        ),
    )
    _install_sources(
        monkeypatch,
        raw_projection_factory=lambda _subject_keys: resigned_projection,
        authenticated_context=_authenticated_context(),
        effective_projection=_effective_projection(items=()),
    )

    with pytest.raises(
        review_projection_service.DynamicSchemaReviewProjectionInvariantError,
        match="dynamic_schema_review_projection_raw_projection_mismatch",
    ):
        run_async(
            review_projection_service.project_reviewed_orchestration_ufl_to_dynamic_schema(
                SessionFactory(),
                project_id=raw_projection.project_id,
                schema_id=raw_projection.schema_id,
                schema_version_id=raw_projection.schema_version_id,
                orchestration_id=raw_projection.orchestration_id,
                consistency_check_application_id=_uuid("consistency-check-application"),
            )
        )


def test_project_reviewed_orchestration_ufl_to_dynamic_schema_calls_public_child_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_projection, _facts = _base_projection()
    authenticated_context = _authenticated_context()
    effective_projection = _effective_projection(items=())
    _install_sources(
        monkeypatch,
        raw_projection_factory=lambda _subject_keys: raw_projection,
        authenticated_context=authenticated_context,
        effective_projection=effective_projection,
    )
    original_raw_authenticate = (
        review_projection_service.raw_projection_service.authenticate_dynamic_schema_ufl_projection
    )
    original_effective_authenticate = (
        review_projection_service.effective_fact_value_service.authenticate_effective_fact_value_projection
    )
    calls: list[str] = []

    def tracking_raw_authenticate(projection, *, subject_keys):
        calls.append("raw")
        return original_raw_authenticate(projection, subject_keys=subject_keys)

    def tracking_effective_authenticate(projection):
        calls.append("effective")
        return original_effective_authenticate(projection)

    monkeypatch.setattr(
        review_projection_service.raw_projection_service,
        "authenticate_dynamic_schema_ufl_projection",
        tracking_raw_authenticate,
    )
    monkeypatch.setattr(
        review_projection_service.effective_fact_value_service,
        "authenticate_effective_fact_value_projection",
        tracking_effective_authenticate,
    )

    run_async(
        review_projection_service.project_reviewed_orchestration_ufl_to_dynamic_schema(
            SessionFactory(),
            project_id=raw_projection.project_id,
            schema_id=raw_projection.schema_id,
            schema_version_id=raw_projection.schema_version_id,
            orchestration_id=raw_projection.orchestration_id,
            consistency_check_application_id=_uuid("consistency-check-application"),
        )
    )

    assert calls == ["raw", "effective"]


def test_authenticate_dynamic_schema_review_projection_accepts_valid_projection_and_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = _build_valid_review_projection(monkeypatch)

    authenticated = review_projection_service.authenticate_dynamic_schema_review_projection(
        projection,
        subject_keys=None,
    )

    assert authenticated == projection


@pytest.mark.parametrize("field_name", ["requires_review", "review_required"])
@pytest.mark.parametrize("bad_value", [0, 1])
def test_authenticate_dynamic_schema_reviewed_field_rejects_non_bool_flags(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    bad_value: int,
) -> None:
    projection = _build_valid_review_projection(monkeypatch)
    valid_field = projection.records[0].fields[1]
    mutated_field = (
        replace(
            valid_field,
            reviewed_facts=(
                replace(valid_field.reviewed_facts[0], requires_review=bad_value),
            ),
        )
        if field_name == "requires_review"
        else replace(valid_field, review_required=bad_value)
    )

    with pytest.raises(
        review_projection_service.DynamicSchemaReviewProjectionInvariantError,
        match="dynamic_schema_review_projection_projection_invalid",
    ):
        review_projection_service.authenticate_dynamic_schema_reviewed_field(
            mutated_field,
            record_subject_key=projection.records[0].subject_key,
            subject_kind="person",
        )


def test_authenticate_dynamic_schema_review_projection_rejects_resigned_non_bool_field_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = _build_valid_review_projection(monkeypatch)
    mutated_projection = replace(
        projection,
        records=(
            replace(
                projection.records[0],
                fields=projection.records[0].fields[:1]
                + (
                    replace(projection.records[0].fields[1], review_required=1),
                )
                + projection.records[0].fields[2:],
            ),
            projection.records[1],
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

    with pytest.raises(
        review_projection_service.DynamicSchemaReviewProjectionInvariantError,
        match="dynamic_schema_review_projection_projection_invalid",
    ):
        review_projection_service.authenticate_dynamic_schema_review_projection(
            resigned_projection,
            subject_keys=None,
        )


def test_authenticate_dynamic_schema_review_projection_calls_public_reviewed_field_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = _build_valid_review_projection(monkeypatch)
    original_authenticate = (
        review_projection_service.authenticate_dynamic_schema_reviewed_field
    )
    calls: list[str] = []

    def tracking_authenticate(field, *, record_subject_key, subject_kind):
        calls.append(field.source_field.field_key)
        return original_authenticate(
            field,
            record_subject_key=record_subject_key,
            subject_kind=subject_kind,
        )

    monkeypatch.setattr(
        review_projection_service,
        "authenticate_dynamic_schema_reviewed_field",
        tracking_authenticate,
    )

    authenticated = review_projection_service.authenticate_dynamic_schema_review_projection(
        projection,
        subject_keys=None,
    )

    assert authenticated == projection
    assert calls == [field.source_field.field_key for field in projection.records[0].fields]


@pytest.mark.parametrize(
    "mutate_projection",
    [
        lambda projection: replace(
            projection,
            records=(
                replace(
                    projection.records[0],
                    fields=(
                        replace(
                            projection.records[0].fields[0],
                            reviewed_facts=(
                                replace(
                                    projection.records[0].fields[0].reviewed_facts[0],
                                    candidate_id=_uuid("bad-candidate"),
                                ),
                            ),
                        ),
                    )
                    + projection.records[0].fields[1:],
                ),
                projection.records[1],
            ),
        ),
        lambda projection: replace(
            projection,
            records=(
                replace(
                    projection.records[0],
                    fields=projection.records[0].fields[:1]
                    + (
                        replace(
                            projection.records[0].fields[1],
                            reviewed_facts=(
                                replace(
                                    projection.records[0].fields[1].reviewed_facts[0],
                                    current_decision_id=None,
                                ),
                            ),
                        ),
                    )
                    + projection.records[0].fields[2:],
                ),
                projection.records[1],
            ),
        ),
        lambda projection: replace(
            projection,
            records=(
                replace(
                    projection.records[0],
                    fields=projection.records[0].fields[:5]
                    + (
                        replace(
                            projection.records[0].fields[5],
                            reviewed_facts=(
                                replace(
                                    projection.records[0].fields[5].reviewed_facts[0],
                                    current_decision_id=_uuid("bad-decision"),
                                ),
                            ),
                        ),
                    )
                    + projection.records[0].fields[6:],
                ),
                projection.records[1],
            ),
        ),
        lambda projection: replace(
            projection,
            records=(
                replace(
                    projection.records[0],
                    fields=projection.records[0].fields[:6]
                    + (
                        replace(
                            projection.records[0].fields[6],
                            reviewed_facts=(
                                replace(
                                    projection.records[0].fields[6].reviewed_facts[0],
                                    current_decision_kind="select_one",
                                ),
                            ),
                        ),
                    )
                    + projection.records[0].fields[7:],
                ),
                projection.records[1],
            ),
        ),
    ],
)
def test_authenticate_dynamic_schema_review_projection_rejects_reviewed_fact_shape_drift(
    monkeypatch: pytest.MonkeyPatch,
    mutate_projection,
) -> None:
    projection = _build_valid_review_projection(monkeypatch)
    mutated_projection = mutate_projection(projection)
    resigned_projection = replace(
        mutated_projection,
        reviewed_projection_manifest_hash=review_projection_service._build_manifest_hash(
            projection=mutated_projection,
            subject_keys_filter=None,
        ),
    )

    with pytest.raises(
        review_projection_service.DynamicSchemaReviewProjectionInvariantError,
        match="dynamic_schema_review_projection_projection_invalid",
    ):
        review_projection_service.authenticate_dynamic_schema_review_projection(
            resigned_projection,
            subject_keys=None,
        )


def test_authenticate_dynamic_schema_review_projection_rejects_cross_field_fact_drift_even_if_manifest_is_resigned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = _build_valid_review_projection(monkeypatch)
    mutated_projection = replace(
        projection,
        records=(
            replace(
                projection.records[0],
                fields=projection.records[0].fields[:4]
                + (
                    replace(
                        projection.records[0].fields[4],
                        reviewed_facts=(
                            replace(
                                projection.records[0].fields[4].reviewed_facts[0],
                                effective_fact_value_ids=(
                                    _uuid("fv-confirm-compatible-a"),
                                ),
                            ),
                        ),
                    ),
                )
                + projection.records[0].fields[5:],
            ),
            projection.records[1],
        ),
    )
    resigned_projection = replace(
        mutated_projection,
        reviewed_projection_manifest_hash=review_projection_service._build_manifest_hash(
            projection=mutated_projection,
            subject_keys_filter=None,
        ),
    )

    with pytest.raises(
        review_projection_service.DynamicSchemaReviewProjectionInvariantError,
        match="dynamic_schema_review_projection_projection_invalid",
    ):
        review_projection_service.authenticate_dynamic_schema_review_projection(
            resigned_projection,
            subject_keys=None,
        )


def test_project_reviewed_orchestration_ufl_to_dynamic_schema_calls_public_review_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_projection, _facts = _base_projection()
    _install_sources(
        monkeypatch,
        raw_projection_factory=lambda _subject_keys: raw_projection,
        authenticated_context=_authenticated_context(),
        effective_projection=_effective_projection(items=()),
    )
    original_authenticate = (
        review_projection_service.authenticate_dynamic_schema_review_projection
    )
    calls: list[int] = []

    def tracking_authenticate(projection, *, subject_keys):
        calls.append(1)
        return original_authenticate(projection, subject_keys=subject_keys)

    monkeypatch.setattr(
        review_projection_service,
        "authenticate_dynamic_schema_review_projection",
        tracking_authenticate,
    )

    run_async(
        review_projection_service.project_reviewed_orchestration_ufl_to_dynamic_schema(
            SessionFactory(),
            project_id=raw_projection.project_id,
            schema_id=raw_projection.schema_id,
            schema_version_id=raw_projection.schema_version_id,
            orchestration_id=raw_projection.orchestration_id,
            consistency_check_application_id=_uuid("consistency-check-application"),
        )
    )

    assert calls == [1]
