from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import uuid

import pytest

from app.schemas.dynamic_schema_review_projection import (
    DynamicSchemaReviewProjection,
    DynamicSchemaReviewedFact,
    DynamicSchemaReviewedField,
    DynamicSchemaReviewedRecord,
)
from app.schemas.dynamic_schema_ufl_projection import DynamicSchemaUFLProjectedField
from app.schemas.project_version import ProjectVersionSnapshot
from app.schemas.ufl_fact_snapshot import (
    UFLFactEvidenceLocator,
    UFLFactEvidenceSnapshot,
    UFLFactSnapshot,
    UFLFactValueGroupSnapshot,
    UFLFactValueSnapshot,
)
from app.services import bailian_review_tools as bailian_tools_service


def run_async(awaitable):
    return asyncio.run(awaitable)


def _uuid(seed: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"bailian-review-tools:{seed}")


def _hash(seed: str) -> str:
    return bailian_tools_service.duplicate_grouping_service.hash_deterministic_payload(
        {"seed": seed}
    )


def _fact(
    seed: str,
    *,
    subject_key: str,
    predicate_key: str,
    scope_key: str | None,
    value_texts: tuple[str, ...],
    evidence_excerpt: str,
) -> UFLFactSnapshot:
    values = tuple(
        UFLFactValueSnapshot(
            fact_value_id=_uuid(f"{seed}-value-{index}"),
            source_batch_id=_uuid(f"{seed}-batch"),
            source_application_id=_uuid(f"{seed}-app"),
            proposal_index=index,
            normalized_value_text=value_text,
            value_hash=_hash(f"{seed}-value-hash-{index}"),
            language_code="zh",
            confidence=0.9,
        )
        for index, value_text in enumerate(value_texts)
    )
    evidences = (
        UFLFactEvidenceSnapshot(
            evidence_link_id=_uuid(f"{seed}-evidence-link"),
            evidence_id=_uuid(f"{seed}-evidence"),
            document_revision_id=_uuid(f"{seed}-doc-rev"),
            document_block_id=_uuid(f"{seed}-doc-block"),
            locator=UFLFactEvidenceLocator(
                location_key=f"loc:{seed}",
                page_no=1,
                start_line=1,
                end_line=1,
                table_index=None,
                row_index=None,
            ),
            excerpt=evidence_excerpt,
            excerpt_hash=_hash(f"{seed}-excerpt"),
            content_hash=_hash(f"{seed}-content"),
            role="supporting",
            is_primary=True,
            source_order=0,
        ),
    )
    return UFLFactSnapshot(
        fact_id=_uuid(f"{seed}-fact"),
        identity_hash=_hash(f"{seed}-identity"),
        subject_kind="person",
        subject_key=subject_key,
        subject_entity_id=None,
        predicate_key=predicate_key,
        scope_key=scope_key,
        semantic_group_count=1,
        fact_value_count=len(values),
        value_groups=(
            UFLFactValueGroupSnapshot(
                semantic_key_hash=_hash(f"{seed}-semantic"),
                value_type="string",
                value_json={"texts": list(value_texts)},
                referenced_entity_id=None,
                fact_value_ids=tuple(value.fact_value_id for value in values),
                values=values,
                evidences=evidences,
            ),
        ),
    )


def _field(
    seed: str,
    *,
    field_key: str,
    predicate_key: str,
    matched_facts: tuple[UFLFactSnapshot, ...],
    display_order: int,
) -> DynamicSchemaUFLProjectedField:
    return DynamicSchemaUFLProjectedField(
        field_id=_uuid(f"{seed}-field"),
        schema_version_id=_uuid("schema-version"),
        field_key=field_key,
        label=field_key.title(),
        description=None,
        predicate_key=predicate_key,
        scope_key=None,
        expected_value_type="string",
        cardinality="many",
        is_required=False,
        is_title=False,
        is_summary=False,
        is_hidden=False,
        group_key=None,
        display_order=display_order,
        display_config={"kind": "text"},
        validation_rules={},
        created_at=datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc),
        matched_facts=matched_facts,
        matched_fact_count=len(matched_facts),
        semantic_value_count=sum(fact.semantic_group_count for fact in matched_facts),
        is_missing=False,
        type_compatible=True,
        issues=(),
    )


def _reviewed_fact(
    fact: UFLFactSnapshot,
    *,
    review_state: str,
    resolution_basis: str,
    requires_review: bool,
    effective_fact_value_ids: tuple[uuid.UUID, ...] = (),
) -> DynamicSchemaReviewedFact:
    return DynamicSchemaReviewedFact(
        fact=fact,
        review_state=review_state,
        candidate_id=None if review_state == "no_consistency_candidate" else _uuid(f"{fact.fact_id}-candidate"),
        assessment_id=None if review_state == "no_consistency_candidate" else _uuid(f"{fact.fact_id}-assessment"),
        resolution_basis=resolution_basis,
        current_decision_id=(
            _uuid(f"{fact.fact_id}-decision")
            if review_state == "resolved"
            else None
        ),
        current_decision_kind="select_one" if review_state == "resolved" else None,
        effective_fact_value_ids=effective_fact_value_ids,
        requires_review=requires_review,
    )


def _build_review_projection() -> DynamicSchemaReviewProjection:
    review_fact = _fact(
        "review",
        subject_key="alpha",
        predicate_key="title",
        scope_key=None,
        value_texts=("Alice", "Alicia"),
        evidence_excerpt="alpha evidence",
    )
    resolved_fact = _fact(
        "resolved",
        subject_key="alpha",
        predicate_key="status",
        scope_key="profile",
        value_texts=("active",),
        evidence_excerpt="resolved evidence",
    )
    observation_fact = _fact(
        "observation",
        subject_key="beta",
        predicate_key="alias",
        scope_key=None,
        value_texts=("Bee",),
        evidence_excerpt="beta evidence",
    )
    review_reviewed_fact = _reviewed_fact(
        review_fact,
        review_state="pending_review",
        resolution_basis="none",
        requires_review=True,
    )
    resolved_reviewed_fact = _reviewed_fact(
        resolved_fact,
        review_state="resolved",
        resolution_basis="human_selection",
        requires_review=False,
        effective_fact_value_ids=(resolved_fact.value_groups[0].fact_value_ids[0],),
    )
    observation_reviewed_fact = _reviewed_fact(
        observation_fact,
        review_state="no_consistency_candidate",
        resolution_basis="none",
        requires_review=False,
    )
    title_field = DynamicSchemaReviewedField(
        source_field=_field(
            "title",
            field_key="title",
            predicate_key="title",
            matched_facts=(review_fact,),
            display_order=0,
        ),
        reviewed_facts=(review_reviewed_fact,),
        review_required=True,
        resolved_fact_count=0,
        review_required_fact_count=1,
        effective_fact_value_ids=(),
    )
    alias_field = DynamicSchemaReviewedField(
        source_field=_field(
            "alias",
            field_key="alias",
            predicate_key="title",
            matched_facts=(review_fact,),
            display_order=1,
        ),
        reviewed_facts=(review_reviewed_fact,),
        review_required=True,
        resolved_fact_count=0,
        review_required_fact_count=1,
        effective_fact_value_ids=(),
    )
    status_field = DynamicSchemaReviewedField(
        source_field=_field(
            "status",
            field_key="status",
            predicate_key="status",
            matched_facts=(resolved_fact,),
            display_order=2,
        ),
        reviewed_facts=(resolved_reviewed_fact,),
        review_required=False,
        resolved_fact_count=1,
        review_required_fact_count=0,
        effective_fact_value_ids=resolved_reviewed_fact.effective_fact_value_ids,
    )
    beta_field = DynamicSchemaReviewedField(
        source_field=_field(
            "beta-alias",
            field_key="beta_alias",
            predicate_key="alias",
            matched_facts=(observation_fact,),
            display_order=0,
        ),
        reviewed_facts=(observation_reviewed_fact,),
        review_required=False,
        resolved_fact_count=0,
        review_required_fact_count=0,
        effective_fact_value_ids=(),
    )
    return DynamicSchemaReviewProjection(
        project_id=_uuid("project"),
        schema_id=_uuid("schema"),
        schema_version_id=_uuid("schema-version"),
        orchestration_id=_uuid("orchestration"),
        extraction_run_id=_uuid("extraction-run"),
        consistency_check_application_id=_uuid("consistency-app"),
        source_consistency_application_id=_uuid("source-consistency-app"),
        schema_definition_manifest_hash=_hash("schema-manifest"),
        ufl_source_manifest_hash=_hash("ufl-manifest"),
        consistency_result_manifest_hash=_hash("consistency-manifest"),
        raw_projection_manifest_hash=_hash("raw-manifest"),
        comparison_quality="complete",
        algorithm_name="dynamic_schema_review_projection",
        algorithm_version="1.0.0",
        record_count=2,
        unique_matched_fact_count=3,
        resolved_fact_count=1,
        review_required_fact_count=1,
        no_candidate_fact_count=1,
        field_review_required_count=2,
        records=(
            DynamicSchemaReviewedRecord(
                subject_key="alpha",
                required_missing_field_keys=(),
                issue_count=0,
                fields=(title_field, alias_field, status_field),
            ),
            DynamicSchemaReviewedRecord(
                subject_key="beta",
                required_missing_field_keys=(),
                issue_count=0,
                fields=(beta_field,),
            ),
        ),
        reviewed_projection_manifest_hash=_hash("reviewed-manifest"),
    )


def _build_project_version_snapshot() -> ProjectVersionSnapshot:
    record_json = {
        "subject_key": "alpha",
        "title_field_key": "title",
        "has_review_required": True,
        "issue_count": 0,
        "sections": [
            {
                "group_key": None,
                "display_order": 0,
                "fields": [
                    {
                        "source_field": {"field_key": "title"},
                        "knowledge_state": "review_required",
                        "effective_fact_value_ids": [],
                        "observed_fact_value_count": 2,
                        "semantic_value_count": 1,
                        "has_schema_issues": False,
                        "reviewed_facts": [],
                    }
                ],
            }
        ],
    }
    return ProjectVersionSnapshot(
        id=_uuid("project-version"),
        project_id=_uuid("project"),
        version_no=3,
        created_by_id=_uuid("creator"),
        creation_kind="manual",
        copied_from_version_id=None,
        reason="snapshot",
        schema_id=_uuid("schema"),
        schema_version_id=_uuid("schema-version"),
        orchestration_id=_uuid("orchestration"),
        extraction_run_id=_uuid("extraction-run"),
        consistency_check_application_id=_uuid("consistency-app"),
        source_consistency_application_id=_uuid("source-consistency-app"),
        schema_definition_manifest_hash=_hash("schema-manifest"),
        ufl_source_manifest_hash=_hash("ufl-manifest"),
        consistency_result_manifest_hash=_hash("consistency-manifest"),
        raw_projection_manifest_hash=_hash("raw-manifest"),
        reviewed_projection_manifest_hash=_hash("reviewed-manifest"),
        knowledge_view_manifest_hash=_hash("knowledge-manifest"),
        knowledge_view_algorithm_name="dynamic_schema_knowledge_view",
        knowledge_view_algorithm_version="1.0.0",
        snapshot_format_version="1.0.0",
        snapshot_json={"records": [record_json]},
        snapshot_json_hash=_hash("snapshot-json"),
        version_manifest_hash=_hash("version-manifest"),
        record_count=1,
        section_count=1,
        field_count=1,
        missing_field_count=0,
        review_required_field_count=1,
        resolved_field_count=0,
        observation_only_field_count=0,
        mixed_field_count=0,
        created_at=datetime(2026, 8, 3, 13, 0, tzinfo=timezone.utc),
        is_current=True,
    )


def test_list_review_items_deduplicates_filters_sorts_and_hashes(monkeypatch) -> None:
    projection = _build_review_projection()
    build_calls: list[dict[str, object]] = []
    auth_calls: list[dict[str, object]] = []

    async def fake_build(*args, **kwargs):
        build_calls.append(kwargs)
        return projection

    def fake_auth(value, *, subject_keys):
        auth_calls.append({"subject_keys": subject_keys, "projection": value})
        return value

    monkeypatch.setattr(
        bailian_tools_service.review_projection_service,
        "project_reviewed_orchestration_ufl_to_dynamic_schema",
        fake_build,
    )
    monkeypatch.setattr(
        bailian_tools_service.review_projection_service,
        "authenticate_dynamic_schema_review_projection",
        fake_auth,
    )

    response = run_async(
        bailian_tools_service.list_review_items(
            object(),
            project_id=str(projection.project_id),
            schema_id=str(projection.schema_id),
            schema_version_id=str(projection.schema_version_id),
            orchestration_id=str(projection.orchestration_id),
            consistency_check_application_id=str(
                projection.consistency_check_application_id
            ),
            state="all",
            limit="10",
        )
    )
    review_required_only = run_async(
        bailian_tools_service.list_review_items(
            object(),
            project_id=str(projection.project_id),
            schema_id=str(projection.schema_id),
            schema_version_id=str(projection.schema_version_id),
            orchestration_id=str(projection.orchestration_id),
            consistency_check_application_id=str(
                projection.consistency_check_application_id
            ),
            state="review_required",
            limit="10",
        )
    )
    deterministic = run_async(
        bailian_tools_service.list_review_items(
            object(),
            project_id=str(projection.project_id),
            schema_id=str(projection.schema_id),
            schema_version_id=str(projection.schema_version_id),
            orchestration_id=str(projection.orchestration_id),
            consistency_check_application_id=str(
                projection.consistency_check_application_id
            ),
            state="all",
            limit="10",
        )
    )

    assert len(build_calls) == 3
    assert all(call["subject_keys"] is None for call in build_calls)
    assert [call["subject_keys"] for call in auth_calls] == [None, None, None]
    assert [item.subject_key for item in response.items] == ["alpha", "alpha", "beta"]
    assert response.items[0].fact_id == _uuid("review-fact")
    assert response.items[0].matched_field_keys == ("title", "alias")
    assert response.items[0].fact_value_count == 2
    assert response.items[0].evidence_count == 1
    assert review_required_only.item_count == 1
    assert review_required_only.items[0].fact_id == _uuid("review-fact")
    assert response.payload_hash == deterministic.payload_hash

    changed = run_async(
        bailian_tools_service.list_review_items(
            object(),
            project_id=str(projection.project_id),
            schema_id=str(projection.schema_id),
            schema_version_id=str(projection.schema_version_id),
            orchestration_id=str(projection.orchestration_id),
            consistency_check_application_id=str(
                projection.consistency_check_application_id
            ),
            state="resolved",
            limit="10",
        )
    )
    assert changed.payload_hash != response.payload_hash


def test_get_review_item_detail_preserves_values_and_evidence(monkeypatch) -> None:
    projection = _build_review_projection()

    async def fake_build(*args, **kwargs):
        return projection

    monkeypatch.setattr(
        bailian_tools_service.review_projection_service,
        "project_reviewed_orchestration_ufl_to_dynamic_schema",
        fake_build,
    )
    monkeypatch.setattr(
        bailian_tools_service.review_projection_service,
        "authenticate_dynamic_schema_review_projection",
        lambda value, *, subject_keys: value,
    )

    detail = run_async(
        bailian_tools_service.get_review_item_detail(
            object(),
            project_id=str(projection.project_id),
            fact_id=str(_uuid("review-fact")),
            schema_id=str(projection.schema_id),
            schema_version_id=str(projection.schema_version_id),
            orchestration_id=str(projection.orchestration_id),
            consistency_check_application_id=str(
                projection.consistency_check_application_id
            ),
        )
    )

    assert detail.fact_id == _uuid("review-fact")
    assert detail.matched_field_keys == ("title", "alias")
    assert detail.requires_review is True
    assert [value["normalized_value_text"] for value in detail.value_groups[0]["values"]] == [
        "Alice",
        "Alicia",
    ]
    assert detail.value_groups[0]["evidences"][0]["excerpt"] == "alpha evidence"


def test_get_review_item_detail_rejects_unknown_fact(monkeypatch) -> None:
    projection = _build_review_projection()

    async def fake_build(*args, **kwargs):
        return projection

    monkeypatch.setattr(
        bailian_tools_service.review_projection_service,
        "project_reviewed_orchestration_ufl_to_dynamic_schema",
        fake_build,
    )
    monkeypatch.setattr(
        bailian_tools_service.review_projection_service,
        "authenticate_dynamic_schema_review_projection",
        lambda value, *, subject_keys: value,
    )

    with pytest.raises(
        bailian_tools_service.BailianReviewToolNotFoundError,
        match="bailian_review_item_not_found",
    ):
        run_async(
            bailian_tools_service.get_review_item_detail(
                object(),
                project_id=str(projection.project_id),
                fact_id=str(_uuid("missing-fact")),
                schema_id=str(projection.schema_id),
                schema_version_id=str(projection.schema_version_id),
                orchestration_id=str(projection.orchestration_id),
                consistency_check_application_id=str(
                    projection.consistency_check_application_id
                ),
            )
        )


def test_get_version_record_reads_exact_subject_and_hash_changes(monkeypatch) -> None:
    snapshot = _build_project_version_snapshot()
    get_calls: list[dict[str, object]] = []
    auth_calls: list[ProjectVersionSnapshot] = []

    async def fake_get_snapshot(*args, **kwargs):
        get_calls.append(kwargs)
        return snapshot

    def fake_authenticate(value):
        auth_calls.append(value)
        return value

    monkeypatch.setattr(
        bailian_tools_service.project_version_service,
        "get_project_version_snapshot",
        fake_get_snapshot,
    )
    monkeypatch.setattr(
        bailian_tools_service.project_version_service,
        "authenticate_project_version_snapshot",
        fake_authenticate,
    )

    response = run_async(
        bailian_tools_service.get_version_record(
            object(),
            project_id=str(snapshot.project_id),
            project_version_id=str(snapshot.id),
            subject_key="alpha",
        )
    )
    deterministic = run_async(
        bailian_tools_service.get_version_record(
            object(),
            project_id=str(snapshot.project_id),
            project_version_id=str(snapshot.id),
            subject_key="alpha",
        )
    )
    changed = run_async(
        bailian_tools_service.get_version_record(
            object(),
            project_id=str(snapshot.project_id),
            project_version_id=str(_uuid("project-version-2")),
            subject_key="alpha",
        )
    )

    assert get_calls == [
        {"project_id": snapshot.project_id, "project_version_id": snapshot.id},
        {"project_id": snapshot.project_id, "project_version_id": snapshot.id},
        {
            "project_id": snapshot.project_id,
            "project_version_id": _uuid("project-version-2"),
        },
    ]
    assert auth_calls == [snapshot, snapshot, snapshot]
    assert response.subject_key == "alpha"
    assert response.record_json["subject_key"] == "alpha"
    assert response.payload_hash == deterministic.payload_hash
    assert changed.payload_hash != response.payload_hash


def test_get_version_record_rejects_missing_subject(monkeypatch) -> None:
    snapshot = _build_project_version_snapshot()

    async def fake_get_snapshot(*args, **kwargs):
        return snapshot

    monkeypatch.setattr(
        bailian_tools_service.project_version_service,
        "get_project_version_snapshot",
        fake_get_snapshot,
    )
    monkeypatch.setattr(
        bailian_tools_service.project_version_service,
        "authenticate_project_version_snapshot",
        lambda value: value,
    )

    with pytest.raises(
        bailian_tools_service.BailianReviewToolNotFoundError,
        match="bailian_version_record_not_found",
    ):
        run_async(
            bailian_tools_service.get_version_record(
                object(),
                project_id=str(snapshot.project_id),
                project_version_id=str(snapshot.id),
                subject_key="beta",
            )
        )


def test_get_version_record_maps_project_version_errors(monkeypatch) -> None:
    snapshot = _build_project_version_snapshot()

    async def fake_missing(*args, **kwargs):
        raise bailian_tools_service.project_version_service.ProjectVersionStateError(
            "project_version_not_found"
        )

    async def fake_invalid(*args, **kwargs):
        raise bailian_tools_service.project_version_service.ProjectVersionInvariantError(
            "project_version_snapshot_invalid"
        )

    monkeypatch.setattr(
        bailian_tools_service.project_version_service,
        "authenticate_project_version_snapshot",
        lambda value: value,
    )
    monkeypatch.setattr(
        bailian_tools_service.project_version_service,
        "get_project_version_snapshot",
        fake_missing,
    )
    with pytest.raises(
        bailian_tools_service.BailianReviewToolNotFoundError,
        match="bailian_version_record_not_found",
    ):
        run_async(
            bailian_tools_service.get_version_record(
                object(),
                project_id=str(snapshot.project_id),
                project_version_id=str(snapshot.id),
                subject_key="alpha",
            )
        )

    monkeypatch.setattr(
        bailian_tools_service.project_version_service,
        "get_project_version_snapshot",
        fake_invalid,
    )
    with pytest.raises(
        bailian_tools_service.BailianReviewToolInvariantError,
        match="bailian_review_tool_source_mismatch",
    ):
        run_async(
            bailian_tools_service.get_version_record(
                object(),
                project_id=str(snapshot.project_id),
                project_version_id=str(snapshot.id),
                subject_key="alpha",
            )
        )


def test_list_review_items_uses_only_review_projection_public_services(monkeypatch) -> None:
    projection = _build_review_projection()
    calls = {"build": 0, "auth": 0}

    async def fake_build(*args, **kwargs):
        calls["build"] += 1
        return projection

    def fake_auth(value, *, subject_keys):
        calls["auth"] += 1
        return value

    monkeypatch.setattr(
        bailian_tools_service.review_projection_service,
        "project_reviewed_orchestration_ufl_to_dynamic_schema",
        fake_build,
    )
    monkeypatch.setattr(
        bailian_tools_service.review_projection_service,
        "authenticate_dynamic_schema_review_projection",
        fake_auth,
    )
    monkeypatch.setattr(
        bailian_tools_service.project_version_service,
        "get_project_version_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected")),
    )

    run_async(
        bailian_tools_service.list_review_items(
            object(),
            project_id=str(projection.project_id),
            schema_id=str(projection.schema_id),
            schema_version_id=str(projection.schema_version_id),
            orchestration_id=str(projection.orchestration_id),
            consistency_check_application_id=str(
                projection.consistency_check_application_id
            ),
        )
    )

    assert calls == {"build": 1, "auth": 1}


def test_get_version_record_uses_only_project_version_public_services(monkeypatch) -> None:
    snapshot = _build_project_version_snapshot()
    calls = {"get": 0, "auth": 0}

    async def fake_get(*args, **kwargs):
        calls["get"] += 1
        return snapshot

    def fake_auth(value):
        calls["auth"] += 1
        return value

    monkeypatch.setattr(
        bailian_tools_service.project_version_service,
        "get_project_version_snapshot",
        fake_get,
    )
    monkeypatch.setattr(
        bailian_tools_service.project_version_service,
        "authenticate_project_version_snapshot",
        fake_auth,
    )
    monkeypatch.setattr(
        bailian_tools_service.review_projection_service,
        "project_reviewed_orchestration_ufl_to_dynamic_schema",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected")),
    )

    run_async(
        bailian_tools_service.get_version_record(
            object(),
            project_id=str(snapshot.project_id),
            project_version_id=str(snapshot.id),
            subject_key="alpha",
        )
    )

    assert calls == {"get": 1, "auth": 1}
