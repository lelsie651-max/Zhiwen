from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import inspect
from types import MappingProxyType
import uuid

import pytest

from app.schemas.dynamic_schema_projection import (
    DynamicSchemaDefinitionSnapshot,
    DynamicSchemaFieldDefinitionSnapshot,
)
from app.schemas.ufl_fact_snapshot import (
    OrchestrationUFLFactSnapshot,
    UFLFactEvidenceLocator,
    UFLFactEvidenceSnapshot,
    UFLFactSnapshot,
    UFLFactValueGroupSnapshot,
    UFLFactValueSnapshot,
)
from app.services import dynamic_schema_ufl_projection as projection_service
from app.utils.deterministic_json import freeze_deterministic_json_value


def run_async(awaitable):
    return asyncio.run(awaitable)


class SessionFactory:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self):
        self.calls += 1
        raise AssertionError("projection service must not open sessions directly")


def _uuid(seed: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"dynamic-schema-ufl:{seed}")


def _field(
    *,
    schema_version_id: uuid.UUID,
    field_key: str,
    predicate_key: str,
    display_order: int,
    scope_key: str | None = None,
    expected_value_type: str = "string",
    cardinality: str = "one",
    is_required: bool = False,
    is_title: bool = False,
    is_summary: bool = False,
    is_hidden: bool = False,
) -> DynamicSchemaFieldDefinitionSnapshot:
    return DynamicSchemaFieldDefinitionSnapshot(
        field_id=_uuid(f"field:{field_key}"),
        schema_version_id=schema_version_id,
        field_key=field_key,
        label=field_key.title(),
        description=None,
        predicate_key=predicate_key,
        scope_key=scope_key,
        expected_value_type=expected_value_type,
        cardinality=cardinality,
        is_required=is_required,
        is_title=is_title,
        is_summary=is_summary,
        is_hidden=is_hidden,
        group_key=None,
        display_order=display_order,
        display_config=freeze_deterministic_json_value({"field": field_key}),
        validation_rules=freeze_deterministic_json_value({"required": is_required}),
        created_at=datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc),
    )


def _value(
    *,
    fact_value_id: uuid.UUID,
    source_batch_id: uuid.UUID,
    source_application_id: uuid.UUID,
    proposal_index: int,
    normalized_value_text: str,
    value_hash: str,
) -> UFLFactValueSnapshot:
    return UFLFactValueSnapshot(
        fact_value_id=fact_value_id,
        source_batch_id=source_batch_id,
        source_application_id=source_application_id,
        proposal_index=proposal_index,
        normalized_value_text=normalized_value_text,
        value_hash=value_hash,
        language_code=None,
        confidence=0.5,
    )


def _evidence(
    *,
    seed: str,
    source_order: int,
) -> UFLFactEvidenceSnapshot:
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
        excerpt_hash="a" * 64,
        content_hash="b" * 64,
        role="supporting",
        is_primary=True,
        source_order=source_order,
    )


def _group(
    *,
    seed: str,
    value_type: str,
    value_json: object,
    fact_value_ids: tuple[uuid.UUID, ...],
    proposal_indexes: tuple[int, ...],
) -> UFLFactValueGroupSnapshot:
    values = tuple(
        _value(
            fact_value_id=fact_value_id,
            source_batch_id=_uuid(f"batch:{seed}:{index}"),
            source_application_id=_uuid(f"application:{seed}:{index}"),
            proposal_index=proposal_index,
            normalized_value_text=f"{seed}:{index}",
            value_hash=f"{index:x}".rjust(64, "c"),
        )
        for index, (fact_value_id, proposal_index) in enumerate(
            zip(fact_value_ids, proposal_indexes, strict=True)
        )
    )
    evidences = tuple(
        _evidence(seed=f"{seed}:{index}", source_order=index)
        for index, _fact_value_id in enumerate(fact_value_ids)
    )
    return UFLFactValueGroupSnapshot(
        semantic_key_hash=seed[:64].ljust(64, "d"),
        value_type=value_type,
        value_json=freeze_deterministic_json_value(value_json),
        referenced_entity_id=None,
        fact_value_ids=fact_value_ids,
        values=values,
        evidences=evidences,
    )


def _fact(
    *,
    seed: str,
    subject_kind: str,
    subject_key: str,
    predicate_key: str,
    scope_key: str | None,
    value_groups: tuple[UFLFactValueGroupSnapshot, ...],
) -> UFLFactSnapshot:
    return UFLFactSnapshot(
        fact_id=_uuid(f"fact:{seed}"),
        identity_hash=seed[:64].ljust(64, "e"),
        subject_kind=subject_kind,
        subject_key=subject_key,
        subject_entity_id=None,
        predicate_key=predicate_key,
        scope_key=scope_key,
        semantic_group_count=len(value_groups),
        fact_value_count=sum(len(group.values) for group in value_groups),
        value_groups=value_groups,
    )


def _schema_snapshot() -> DynamicSchemaDefinitionSnapshot:
    schema_version_id = _uuid("schema-version")
    fields = (
        _field(
            schema_version_id=schema_version_id,
            field_key="title",
            predicate_key="title",
            display_order=0,
            is_required=True,
            is_title=True,
        ),
        _field(
            schema_version_id=schema_version_id,
            field_key="aliases",
            predicate_key="alias",
            display_order=1,
            cardinality="many",
            is_hidden=True,
        ),
        _field(
            schema_version_id=schema_version_id,
            field_key="status_current",
            predicate_key="status",
            scope_key="current",
            display_order=2,
        ),
        _field(
            schema_version_id=schema_version_id,
            field_key="age",
            predicate_key="age",
            display_order=3,
            expected_value_type="number",
        ),
        _field(
            schema_version_id=schema_version_id,
            field_key="nickname",
            predicate_key="nickname",
            display_order=4,
            is_required=True,
        ),
        _field(
            schema_version_id=schema_version_id,
            field_key="preference",
            predicate_key="preference",
            display_order=5,
        ),
    )
    return DynamicSchemaDefinitionSnapshot(
        project_id=_uuid("project"),
        schema_id=_uuid("schema"),
        schema_key="profile.main",
        name="Profile Main",
        subject_kind="person",
        description="Main profile",
        schema_status="active",
        schema_version_id=schema_version_id,
        version_no=2,
        version_status="active",
        source_kind="human",
        summary="Profile schema",
        layout_config=freeze_deterministic_json_value({"sections": ["main"]}),
        created_by_id=_uuid("created-by"),
        activated_by_id=_uuid("activated-by"),
        activated_at=datetime(2026, 8, 2, 11, 0, tzinfo=timezone.utc),
        is_current=True,
        algorithm_name="dynamic_schema_definition_snapshot",
        algorithm_version="1.0.0",
        field_count=len(fields),
        fields=fields,
        definition_manifest_hash="1" * 64,
    )


def _ufl_snapshot() -> OrchestrationUFLFactSnapshot:
    facts = (
        _fact(
            seed="alpha-title",
            subject_kind="person",
            subject_key="alpha",
            predicate_key="title",
            scope_key=None,
            value_groups=(
                _group(
                    seed="alpha-title-group",
                    value_type="string",
                    value_json="Engineer",
                    fact_value_ids=(_uuid("alpha-title-fv"),),
                    proposal_indexes=(0,),
                ),
            ),
        ),
        _fact(
            seed="alpha-alias-none",
            subject_kind="person",
            subject_key="alpha",
            predicate_key="alias",
            scope_key=None,
            value_groups=(
                _group(
                    seed="alpha-alias-none-group",
                    value_type="string",
                    value_json="Al",
                    fact_value_ids=(
                        _uuid("alpha-alias-none-fv-a"),
                        _uuid("alpha-alias-none-fv-b"),
                    ),
                    proposal_indexes=(0, 1),
                ),
            ),
        ),
        _fact(
            seed="alpha-alias-scope",
            subject_kind="person",
            subject_key="alpha",
            predicate_key="alias",
            scope_key="preferred",
            value_groups=(
                _group(
                    seed="alpha-alias-scope-group",
                    value_type="string",
                    value_json="A",
                    fact_value_ids=(_uuid("alpha-alias-scope-fv"),),
                    proposal_indexes=(0,),
                ),
            ),
        ),
        _fact(
            seed="alpha-status-current",
            subject_kind="person",
            subject_key="alpha",
            predicate_key="status",
            scope_key="current",
            value_groups=(
                _group(
                    seed="alpha-status-current-group",
                    value_type="string",
                    value_json="active",
                    fact_value_ids=(_uuid("alpha-status-current-fv"),),
                    proposal_indexes=(0,),
                ),
            ),
        ),
        _fact(
            seed="alpha-status-old",
            subject_kind="person",
            subject_key="alpha",
            predicate_key="status",
            scope_key="archived",
            value_groups=(
                _group(
                    seed="alpha-status-old-group",
                    value_type="string",
                    value_json="inactive",
                    fact_value_ids=(_uuid("alpha-status-old-fv"),),
                    proposal_indexes=(0,),
                ),
            ),
        ),
        _fact(
            seed="alpha-age",
            subject_kind="person",
            subject_key="alpha",
            predicate_key="age",
            scope_key=None,
            value_groups=(
                _group(
                    seed="alpha-age-group",
                    value_type="string",
                    value_json="twenty",
                    fact_value_ids=(_uuid("alpha-age-fv"),),
                    proposal_indexes=(0,),
                ),
            ),
        ),
        _fact(
            seed="alpha-preference",
            subject_kind="person",
            subject_key="alpha",
            predicate_key="preference",
            scope_key=None,
            value_groups=(
                _group(
                    seed="alpha-preference-group-a",
                    value_type="string",
                    value_json="red",
                    fact_value_ids=(_uuid("alpha-preference-fv-a"),),
                    proposal_indexes=(0,),
                ),
                _group(
                    seed="alpha-preference-group-b",
                    value_type="string",
                    value_json="blue",
                    fact_value_ids=(_uuid("alpha-preference-fv-b"),),
                    proposal_indexes=(1,),
                ),
            ),
        ),
        _fact(
            seed="beta-title-a",
            subject_kind="person",
            subject_key="beta",
            predicate_key="title",
            scope_key=None,
            value_groups=(
                _group(
                    seed="beta-title-group-a",
                    value_type="string",
                    value_json="Manager",
                    fact_value_ids=(_uuid("beta-title-fv-a"),),
                    proposal_indexes=(0,),
                ),
            ),
        ),
        _fact(
            seed="beta-title-b",
            subject_kind="person",
            subject_key="beta",
            predicate_key="title",
            scope_key=None,
            value_groups=(
                _group(
                    seed="beta-title-group-b",
                    value_type="string",
                    value_json="Director",
                    fact_value_ids=(_uuid("beta-title-fv-b"),),
                    proposal_indexes=(0,),
                ),
            ),
        ),
        _fact(
            seed="org-title",
            subject_kind="organization",
            subject_key="org-1",
            predicate_key="title",
            scope_key=None,
            value_groups=(
                _group(
                    seed="org-title-group",
                    value_type="string",
                    value_json="Org Title",
                    fact_value_ids=(_uuid("org-title-fv"),),
                    proposal_indexes=(0,),
                ),
            ),
        ),
    )
    return OrchestrationUFLFactSnapshot(
        project_id=_uuid("project"),
        orchestration_id=_uuid("orchestration"),
        extraction_run_id=_uuid("extraction-run"),
        orchestration_status="completed",
        comparison_quality="complete",
        source_application_count=2,
        fact_count=len(facts),
        fact_value_count=sum(fact.fact_value_count for fact in facts),
        evidence_count=sum(
            len(group.evidences) for fact in facts for group in fact.value_groups
        ),
        algorithm_name="orchestration_ufl_fact_snapshot",
        algorithm_version="1.0.0",
        facts=facts,
        source_manifest_hash="2" * 64,
    )


def _install_sources(
    monkeypatch: pytest.MonkeyPatch,
    *,
    schema_snapshot: DynamicSchemaDefinitionSnapshot,
    ufl_snapshot: OrchestrationUFLFactSnapshot,
) -> None:
    async def fake_schema_snapshot(*_args, **_kwargs):
        return schema_snapshot

    async def fake_ufl_snapshot(*_args, **_kwargs):
        return ufl_snapshot

    monkeypatch.setattr(
        projection_service.dynamic_schema_projection_service,
        "get_dynamic_schema_definition_snapshot",
        fake_schema_snapshot,
    )
    monkeypatch.setattr(
        projection_service.ufl_fact_snapshot_service,
        "get_orchestration_ufl_fact_snapshot",
        fake_ufl_snapshot,
    )


def test_project_orchestration_ufl_to_dynamic_schema_matches_subject_field_scope_and_preserves_hidden_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema_snapshot = _schema_snapshot()
    ufl_snapshot = _ufl_snapshot()
    factory = SessionFactory()
    _install_sources(
        monkeypatch,
        schema_snapshot=schema_snapshot,
        ufl_snapshot=ufl_snapshot,
    )

    projection = run_async(
        projection_service.project_orchestration_ufl_to_dynamic_schema(
            factory,
            project_id=schema_snapshot.project_id,
            schema_id=schema_snapshot.schema_id,
            schema_version_id=schema_snapshot.schema_version_id,
            orchestration_id=ufl_snapshot.orchestration_id,
        )
    )

    assert factory.calls == 0
    assert projection.record_count == 2
    assert [record.subject_key for record in projection.records] == ["alpha", "beta"]
    alpha_record = projection.records[0]
    assert [field.field_key for field in alpha_record.fields] == [
        field.field_key for field in schema_snapshot.fields
    ]

    aliases_field = alpha_record.fields[1]
    assert aliases_field.is_hidden is True
    assert aliases_field.matched_fact_count == 2
    assert aliases_field.semantic_value_count == 2
    assert [fact.scope_key for fact in aliases_field.matched_facts] == [None, "preferred"]
    assert aliases_field.matched_facts[0] is ufl_snapshot.facts[1]
    assert aliases_field.matched_facts[1] is ufl_snapshot.facts[2]

    status_field = alpha_record.fields[2]
    assert status_field.matched_fact_count == 1
    assert status_field.matched_facts[0].scope_key == "current"

    age_field = alpha_record.fields[3]
    assert age_field.type_compatible is False
    assert age_field.issues == ("value_type_mismatch",)

    preference_field = alpha_record.fields[5]
    assert preference_field.semantic_value_count == 2
    assert preference_field.issues == ("cardinality_one_multiple_semantic_values",)

    beta_record = projection.records[1]
    beta_title_field = beta_record.fields[0]
    assert beta_title_field.matched_fact_count == 2
    assert beta_title_field.semantic_value_count == 2
    assert beta_title_field.issues == (
        "cardinality_one_multiple_facts",
        "cardinality_one_multiple_semantic_values",
    )


def test_project_orchestration_ufl_to_dynamic_schema_subject_keys_preserve_order_and_generate_empty_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema_snapshot = _schema_snapshot()
    ufl_snapshot = _ufl_snapshot()
    _install_sources(
        monkeypatch,
        schema_snapshot=schema_snapshot,
        ufl_snapshot=ufl_snapshot,
    )

    projection = run_async(
        projection_service.project_orchestration_ufl_to_dynamic_schema(
            SessionFactory(),
            project_id=schema_snapshot.project_id,
            schema_id=schema_snapshot.schema_id,
            schema_version_id=schema_snapshot.schema_version_id,
            orchestration_id=ufl_snapshot.orchestration_id,
            subject_keys=["beta", "gamma", " alpha "],
        )
    )

    assert [record.subject_key for record in projection.records] == ["beta", "gamma", "alpha"]
    gamma_record = projection.records[1]
    assert gamma_record.issue_count == 2
    assert gamma_record.required_missing_field_keys == ("title", "nickname")
    assert all(field.is_missing for field in gamma_record.fields)


def test_project_orchestration_ufl_to_dynamic_schema_required_missing_type_mismatch_and_no_auto_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema_snapshot = _schema_snapshot()
    ufl_snapshot = _ufl_snapshot()
    _install_sources(
        monkeypatch,
        schema_snapshot=schema_snapshot,
        ufl_snapshot=ufl_snapshot,
    )

    projection = run_async(
        projection_service.project_orchestration_ufl_to_dynamic_schema(
            SessionFactory(),
            project_id=schema_snapshot.project_id,
            schema_id=schema_snapshot.schema_id,
            schema_version_id=schema_snapshot.schema_version_id,
            orchestration_id=ufl_snapshot.orchestration_id,
        )
    )

    alpha_record = projection.records[0]
    assert alpha_record.required_missing_field_keys == ("nickname",)
    assert alpha_record.issue_count == 3
    aliases_field = alpha_record.fields[1]
    assert aliases_field.semantic_value_count == 2
    assert len(aliases_field.matched_facts[0].value_groups[0].values) == 2

    beta_record = projection.records[1]
    assert beta_record.required_missing_field_keys == ("nickname",)
    assert beta_record.issue_count == 3
    title_field = beta_record.fields[0]
    assert title_field.matched_facts[0].value_groups[0].value_json == "Manager"
    assert title_field.matched_facts[1].value_groups[0].value_json == "Director"


def test_project_orchestration_ufl_to_dynamic_schema_excludes_non_matching_subject_predicate_and_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema_snapshot = _schema_snapshot()
    ufl_snapshot = _ufl_snapshot()
    _install_sources(
        monkeypatch,
        schema_snapshot=schema_snapshot,
        ufl_snapshot=ufl_snapshot,
    )

    projection = run_async(
        projection_service.project_orchestration_ufl_to_dynamic_schema(
            SessionFactory(),
            project_id=schema_snapshot.project_id,
            schema_id=schema_snapshot.schema_id,
            schema_version_id=schema_snapshot.schema_version_id,
            orchestration_id=ufl_snapshot.orchestration_id,
        )
    )

    alpha_record = projection.records[0]
    title_field = alpha_record.fields[0]
    assert all(fact.predicate_key == "title" for fact in title_field.matched_facts)
    assert all(fact.scope_key is None for fact in title_field.matched_facts)
    status_field = alpha_record.fields[2]
    assert [fact.scope_key for fact in status_field.matched_facts] == ["current"]
    assert all(fact.subject_kind == "person" for record in projection.records for field in record.fields for fact in field.matched_facts)


def test_project_orchestration_ufl_to_dynamic_schema_preserves_all_values_and_evidence_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema_snapshot = _schema_snapshot()
    ufl_snapshot = _ufl_snapshot()
    _install_sources(
        monkeypatch,
        schema_snapshot=schema_snapshot,
        ufl_snapshot=ufl_snapshot,
    )

    projection = run_async(
        projection_service.project_orchestration_ufl_to_dynamic_schema(
            SessionFactory(),
            project_id=schema_snapshot.project_id,
            schema_id=schema_snapshot.schema_id,
            schema_version_id=schema_snapshot.schema_version_id,
            orchestration_id=ufl_snapshot.orchestration_id,
        )
    )

    aliases_fact = projection.records[0].fields[1].matched_facts[0]
    assert [group.semantic_key_hash for group in aliases_fact.value_groups] == [
        ufl_snapshot.facts[1].value_groups[0].semantic_key_hash
    ]
    assert [value.proposal_index for value in aliases_fact.value_groups[0].values] == [0, 1]
    assert [evidence.source_order for evidence in aliases_fact.value_groups[0].evidences] == [0, 1]


@pytest.mark.parametrize(
    ("schema_snapshot", "ufl_snapshot", "expected_code"),
    [
        (
            replace(_schema_snapshot(), project_id=_uuid("other-project")),
            _ufl_snapshot(),
            "dynamic_schema_ufl_projection_source_project_mismatch",
        ),
        (
            _schema_snapshot(),
            replace(_ufl_snapshot(), project_id=_uuid("other-project")),
            "dynamic_schema_ufl_projection_source_project_mismatch",
        ),
        (
            replace(_schema_snapshot(), schema_id=_uuid("other-schema")),
            _ufl_snapshot(),
            "dynamic_schema_ufl_projection_schema_identity_mismatch",
        ),
        (
            _schema_snapshot(),
            replace(_ufl_snapshot(), orchestration_id=_uuid("other-orchestration")),
            "dynamic_schema_ufl_projection_ufl_identity_mismatch",
        ),
    ],
)
def test_project_orchestration_ufl_to_dynamic_schema_rejects_source_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    schema_snapshot: DynamicSchemaDefinitionSnapshot,
    ufl_snapshot: OrchestrationUFLFactSnapshot,
    expected_code: str,
) -> None:
    expected_schema_id = _schema_snapshot().schema_id
    expected_schema_version_id = _schema_snapshot().schema_version_id
    expected_orchestration_id = _ufl_snapshot().orchestration_id
    _install_sources(
        monkeypatch,
        schema_snapshot=schema_snapshot,
        ufl_snapshot=ufl_snapshot,
    )

    with pytest.raises(
        projection_service.DynamicSchemaUFLProjectionInvariantError,
        match=expected_code,
    ):
        run_async(
            projection_service.project_orchestration_ufl_to_dynamic_schema(
                SessionFactory(),
                project_id=_uuid("project"),
                schema_id=expected_schema_id,
                schema_version_id=expected_schema_version_id,
                orchestration_id=expected_orchestration_id,
            )
        )


@pytest.mark.parametrize(
    "subject_keys",
    [
        ("alpha",),
        ["alpha", "alpha"],
        ["   "],
        [str(index) for index in range(501)],
    ],
)
def test_project_orchestration_ufl_to_dynamic_schema_rejects_invalid_subject_keys(
    monkeypatch: pytest.MonkeyPatch,
    subject_keys,
) -> None:
    _install_sources(
        monkeypatch,
        schema_snapshot=_schema_snapshot(),
        ufl_snapshot=_ufl_snapshot(),
    )

    with pytest.raises(
        projection_service.DynamicSchemaUFLProjectionStateError,
        match="dynamic_schema_ufl_projection_subject_keys_invalid",
    ):
        run_async(
            projection_service.project_orchestration_ufl_to_dynamic_schema(
                SessionFactory(),
                project_id=_uuid("project"),
                schema_id=_uuid("schema"),
                schema_version_id=_uuid("schema-version"),
                orchestration_id=_uuid("orchestration"),
                subject_keys=subject_keys,  # type: ignore[arg-type]
            )
        )


def test_project_orchestration_ufl_to_dynamic_schema_counts_sorting_and_manifest_are_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema_snapshot = _schema_snapshot()
    ufl_snapshot = _ufl_snapshot()
    _install_sources(
        monkeypatch,
        schema_snapshot=schema_snapshot,
        ufl_snapshot=ufl_snapshot,
    )

    first = run_async(
        projection_service.project_orchestration_ufl_to_dynamic_schema(
            SessionFactory(),
            project_id=schema_snapshot.project_id,
            schema_id=schema_snapshot.schema_id,
            schema_version_id=schema_snapshot.schema_version_id,
            orchestration_id=ufl_snapshot.orchestration_id,
        )
    )
    second = run_async(
        projection_service.project_orchestration_ufl_to_dynamic_schema(
            SessionFactory(),
            project_id=schema_snapshot.project_id,
            schema_id=schema_snapshot.schema_id,
            schema_version_id=schema_snapshot.schema_version_id,
            orchestration_id=ufl_snapshot.orchestration_id,
        )
    )
    reordered_filter = run_async(
        projection_service.project_orchestration_ufl_to_dynamic_schema(
            SessionFactory(),
            project_id=schema_snapshot.project_id,
            schema_id=schema_snapshot.schema_id,
            schema_version_id=schema_snapshot.schema_version_id,
            orchestration_id=ufl_snapshot.orchestration_id,
            subject_keys=["beta", "alpha"],
        )
    )

    assert first == second
    assert first.record_count == 2
    assert first.projected_field_count == 12
    assert first.required_missing_count == 2
    assert first.issue_count == 6
    assert first.projection_manifest_hash != reordered_filter.projection_manifest_hash


def test_project_orchestration_ufl_to_dynamic_schema_manifest_changes_with_source_manifest_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema_snapshot = _schema_snapshot()
    ufl_snapshot = _ufl_snapshot()
    _install_sources(
        monkeypatch,
        schema_snapshot=schema_snapshot,
        ufl_snapshot=ufl_snapshot,
    )
    baseline = run_async(
        projection_service.project_orchestration_ufl_to_dynamic_schema(
            SessionFactory(),
            project_id=schema_snapshot.project_id,
            schema_id=schema_snapshot.schema_id,
            schema_version_id=schema_snapshot.schema_version_id,
            orchestration_id=ufl_snapshot.orchestration_id,
        )
    )

    _install_sources(
        monkeypatch,
        schema_snapshot=schema_snapshot,
        ufl_snapshot=replace(ufl_snapshot, source_manifest_hash="3" * 64),
    )
    changed = run_async(
        projection_service.project_orchestration_ufl_to_dynamic_schema(
            SessionFactory(),
            project_id=schema_snapshot.project_id,
            schema_id=schema_snapshot.schema_id,
            schema_version_id=schema_snapshot.schema_version_id,
            orchestration_id=ufl_snapshot.orchestration_id,
        )
    )

    assert baseline.projection_manifest_hash != changed.projection_manifest_hash


def test_project_orchestration_ufl_to_dynamic_schema_handles_zero_fact_and_all_withheld_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema_snapshot = _schema_snapshot()
    empty_ufl_snapshot = replace(
        _ufl_snapshot(),
        fact_count=0,
        fact_value_count=0,
        evidence_count=0,
        facts=(),
    )
    _install_sources(
        monkeypatch,
        schema_snapshot=schema_snapshot,
        ufl_snapshot=empty_ufl_snapshot,
    )

    projection = run_async(
        projection_service.project_orchestration_ufl_to_dynamic_schema(
            SessionFactory(),
            project_id=schema_snapshot.project_id,
            schema_id=schema_snapshot.schema_id,
            schema_version_id=schema_snapshot.schema_version_id,
            orchestration_id=empty_ufl_snapshot.orchestration_id,
        )
    )
    filtered_projection = run_async(
        projection_service.project_orchestration_ufl_to_dynamic_schema(
            SessionFactory(),
            project_id=schema_snapshot.project_id,
            schema_id=schema_snapshot.schema_id,
            schema_version_id=schema_snapshot.schema_version_id,
            orchestration_id=empty_ufl_snapshot.orchestration_id,
            subject_keys=["alpha"],
        )
    )

    assert projection.records == ()
    assert projection.record_count == 0
    assert filtered_projection.record_count == 1
    assert filtered_projection.records[0].required_missing_field_keys == ("title", "nickname")


def test_project_orchestration_ufl_to_dynamic_schema_uses_only_public_snapshots_and_no_current_value_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_sources(
        monkeypatch,
        schema_snapshot=_schema_snapshot(),
        ufl_snapshot=_ufl_snapshot(),
    )

    projection = run_async(
        projection_service.project_orchestration_ufl_to_dynamic_schema(
            SessionFactory(),
            project_id=_uuid("project"),
            schema_id=_uuid("schema"),
            schema_version_id=_uuid("schema-version"),
            orchestration_id=_uuid("orchestration"),
        )
    )

    source = inspect.getsource(projection_service)
    assert "current_value_id" not in source
    assert "get_dynamic_schema_definition_snapshot" in source
    assert "get_orchestration_ufl_fact_snapshot" in source
    assert projection.algorithm_name == "dynamic_schema_ufl_projection"
    assert projection.algorithm_version == "1.0.0"
    assert isinstance(projection.records[0].fields[0].display_config, MappingProxyType)
