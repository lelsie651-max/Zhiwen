import asyncio
import inspect
from datetime import datetime, timezone
import uuid

import pytest
from sqlalchemy.orm import joinedload as sa_joinedload
from sqlalchemy.orm import selectinload as sa_selectinload

from app.models.document_content import DocumentBlock, ExtractionRun, SourceEvidence
from app.models.dynamic_schema import DynamicSchema, DynamicSchemaField, DynamicSchemaVersion
from app.models.fact import Fact, FactEvidenceLink, FactValue
from app.models.project import Project
from app.schemas.dynamic_schema_projection import DynamicSchemaProjection
from app.services import dynamic_schema_projection as projection_service


def run_async(awaitable):
    return asyncio.run(awaitable)


class ReadOnlySession:
    async def execute(self, *_args, **_kwargs):
        raise AssertionError("execute should be mocked in read-only tests")

    async def commit(self):
        raise AssertionError("projection service must not commit")

    async def rollback(self):
        raise AssertionError("projection service must not rollback")

    async def flush(self):
        raise AssertionError("projection service must not flush")

    def add(self, *_args, **_kwargs):
        raise AssertionError("projection service must not add objects")

    def add_all(self, *_args, **_kwargs):
        raise AssertionError("projection service must not add objects")


class StubSchemaWithoutCurrentVersion:
    def __init__(
        self,
        *,
        project_id: uuid.UUID,
        schema_id: uuid.UUID | None = None,
        current_version_id: uuid.UUID | None = None,
        status: str = "active",
    ) -> None:
        self.id = schema_id or uuid.uuid4()
        self.project_id = project_id
        self.schema_key = "profile.main"
        self.name = "Profile Main"
        self.subject_kind = "person"
        self.description = "Profile"
        self.status = status
        self.current_version_id = current_version_id

    @property
    def current_version(self):
        raise AssertionError("schema.current_version must not be accessed")


def build_project(*, project_id: uuid.UUID | None = None) -> Project:
    actual_id = project_id or uuid.uuid4()
    return Project(
        id=actual_id,
        name="Projection Project",
        slug=f"projection-{str(actual_id)[:8]}",
        created_by_id=uuid.uuid4(),
        status="active",
    )


def build_schema(
    *,
    project_id: uuid.UUID,
    schema_id: uuid.UUID | None = None,
    current_version_id: uuid.UUID | None = None,
    status: str = "active",
    subject_kind: str = "person",
) -> DynamicSchema:
    return DynamicSchema(
        id=schema_id or uuid.uuid4(),
        project_id=project_id,
        schema_key="profile.main",
        name="Profile Main",
        subject_kind=subject_kind,
        description="Projection schema",
        status=status,
        current_version_id=current_version_id,
        created_by_id=None,
    )


def build_version(
    *,
    schema_id: uuid.UUID,
    version_id: uuid.UUID | None = None,
    version_no: int = 1,
    status: str = "active",
) -> DynamicSchemaVersion:
    return DynamicSchemaVersion(
        id=version_id or uuid.uuid4(),
        schema_id=schema_id,
        version_no=version_no,
        status=status,
        source_kind="human",
        summary="Projection version",
        layout_config={"layout": "default"},
        created_by_id=None,
        activated_by_id=None if status in {"draft", "proposed"} else uuid.uuid4(),
        activated_at=None if status in {"draft", "proposed"} else datetime.now(timezone.utc),
    )


def build_field(
    *,
    version_id: uuid.UUID,
    field_key: str,
    predicate_key: str,
    display_order: int,
    scope_key: str | None = None,
    cardinality: str = "one",
    expected_value_type: str = "any",
    is_required: bool = False,
    is_hidden: bool = False,
) -> DynamicSchemaField:
    return DynamicSchemaField(
        id=uuid.uuid4(),
        schema_version_id=version_id,
        field_key=field_key,
        label=field_key.title(),
        description=None,
        predicate_key=predicate_key,
        scope_key=scope_key,
        expected_value_type=expected_value_type,
        cardinality=cardinality,
        is_required=is_required,
        is_title=False,
        is_summary=False,
        is_hidden=is_hidden,
        group_key=None,
        display_order=display_order,
        display_config={},
        validation_rules={},
    )


def build_fact(
    *,
    project_id: uuid.UUID,
    subject_kind: str = "person",
    subject_key: str,
    predicate_key: str,
    scope_key: str | None = None,
    identity_hash: str | None = None,
    current_value: FactValue | None = None,
    status: str = "active",
) -> Fact:
    fact = Fact(
        id=uuid.uuid4(),
        project_id=project_id,
        subject_kind=subject_kind,
        subject_key=subject_key,
        predicate_key=predicate_key,
        scope_key=scope_key,
        identity_hash=identity_hash or uuid.uuid4().hex + uuid.uuid4().hex,
        status=status,
        current_value_id=current_value.id if current_value is not None else None,
        created_by_id=None,
    )
    if current_value is not None:
        fact.current_value = current_value
        current_value.fact = fact
        current_value.fact_id = fact.id
    return fact


def build_fact_value(
    *,
    fact_id: uuid.UUID | None = None,
    value_type: str = "string",
    value_json="Alice",
    normalized_value_text: str = "Alice",
    status: str = "accepted",
    source_kind: str = "human",
    evidence_links: list[FactEvidenceLink] | None = None,
) -> FactValue:
    value = FactValue(
        id=uuid.uuid4(),
        fact_id=fact_id or uuid.uuid4(),
        version_no=1,
        value_type=value_type,
        value_json=value_json,
        normalized_value_text=normalized_value_text,
        value_hash="a" * 64,
        language_code="zh-CN",
        status=status,
        source_kind=source_kind,
        extraction_run_id=None,
        confidence=0.9,
        created_by_id=None,
        decided_by_id=uuid.uuid4() if status == "accepted" else None,
        decided_at=datetime.now(timezone.utc) if status == "accepted" else None,
    )
    value.evidence_links = evidence_links or []
    for link in value.evidence_links:
        link.fact_value = value
        link.fact_value_id = value.id
    return value


def build_evidence_link(
    *,
    source_order: int,
    role: str = "supporting",
    is_primary: bool = False,
    evidence: SourceEvidence | None = None,
) -> FactEvidenceLink:
    actual_evidence = evidence or build_source_evidence()
    link = FactEvidenceLink(
        id=uuid.uuid4(),
        fact_value_id=uuid.uuid4(),
        evidence_id=actual_evidence.id,
        role=role,
        is_primary=is_primary,
        source_order=source_order,
    )
    link.evidence = actual_evidence
    return link


def build_source_evidence(
    *,
    excerpt: str = "Alice",
    location_key: str = "md:p1",
    page_no: int | None = None,
    start_line: int | None = 1,
    end_line: int | None = 1,
    start_offset: int = 0,
    end_offset: int = 5,
) -> SourceEvidence:
    run = ExtractionRun(
        id=uuid.uuid4(),
        revision_id=uuid.uuid4(),
        attempt_no=1,
        status="completed",
        outcome="success",
        extractor_name="extractor",
        extractor_version="1.0.0",
        detected_format="md",
        detected_encoding="utf-8",
        page_count=None,
        character_count=10,
        block_count=1,
        warnings=[],
        content_metadata={},
    )
    block = DocumentBlock(
        id=uuid.uuid4(),
        extraction_run_id=run.id,
        source_order=0,
        block_type="paragraph",
        raw_text="Alice text",
        normalized_text="Alice text",
        location_key=location_key,
        anchor_hash="b" * 64,
        page_no=page_no,
        block_index=0,
        heading_path=[],
        start_line=start_line,
        end_line=end_line,
        block_metadata={},
    )
    block.extraction_run = run
    evidence = SourceEvidence(
        id=uuid.uuid4(),
        block_id=block.id,
        start_offset=start_offset,
        end_offset=end_offset,
        excerpt=excerpt,
        excerpt_hash="c" * 64,
    )
    evidence.block = block
    return evidence


def attach_version_fields(version: DynamicSchemaVersion, fields: list[DynamicSchemaField]) -> None:
    version.fields = fields
    for field in fields:
        field.schema_version = version


def patch_projection_dependencies(
    monkeypatch,
    *,
    schema,
    version,
    facts: list[Fact],
) -> None:
    async def fake_get_dynamic_schema_by_id(_session, *, project_id, schema_id):
        if project_id == schema.project_id and schema_id == schema.id:
            return schema
        return None

    async def fake_get_dynamic_schema_version_with_fields_for_projection(_session, *, schema_id, version_id):
        if schema_id == version.schema_id and version_id == version.id:
            return version
        return None

    async def fake_list_facts_for_projection(
        _session,
        *,
        project_id,
        subject_kind,
        predicate_keys,
        subject_keys=None,
    ):
        selected = [
            fact
            for fact in facts
            if fact.project_id == project_id
            and fact.subject_kind == subject_kind
            and fact.predicate_key in predicate_keys
            and (subject_keys is None or fact.subject_key in subject_keys)
        ]
        return selected

    monkeypatch.setattr(
        projection_service.projection_repository,
        "get_dynamic_schema_by_id",
        fake_get_dynamic_schema_by_id,
    )
    monkeypatch.setattr(
        projection_service.projection_repository,
        "get_dynamic_schema_version_with_fields_for_projection",
        fake_get_dynamic_schema_version_with_fields_for_projection,
    )
    monkeypatch.setattr(
        projection_service.projection_repository,
        "list_facts_for_projection",
        fake_list_facts_for_projection,
    )


def test_project_current_dynamic_schema_projects_current_schema(monkeypatch) -> None:
    session = ReadOnlySession()
    project = build_project()
    schema = build_schema(project_id=project.id, current_version_id=uuid.uuid4())
    version = build_version(schema_id=schema.id, version_id=schema.current_version_id, status="active")
    attach_version_fields(
        version,
        [build_field(version_id=version.id, field_key="name", predicate_key="person.name", display_order=0)],
    )
    value = build_fact_value(evidence_links=[build_evidence_link(source_order=0, is_primary=True)])
    fact = build_fact(project_id=project.id, subject_key="alice", predicate_key="person.name", current_value=value)

    patch_projection_dependencies(monkeypatch, schema=schema, version=version, facts=[fact])

    result = run_async(
        projection_service.project_current_dynamic_schema(
            session,
            project_id=project.id,
            schema_id=schema.id,
        )
    )

    assert isinstance(result, DynamicSchemaProjection)
    assert result.version_id == version.id
    assert result.records[0].subject_key == "alice"
    assert result.records[0].fields[0].values[0].fact_id == fact.id


@pytest.mark.parametrize("status", ["draft", "proposed", "retired", "active"])
def test_explicit_version_projection_allows_preview_statuses(monkeypatch, status: str) -> None:
    session = ReadOnlySession()
    project = build_project()
    schema = build_schema(project_id=project.id)
    version = build_version(schema_id=schema.id, status=status)
    attach_version_fields(
        version,
        [build_field(version_id=version.id, field_key="name", predicate_key="person.name", display_order=0)],
    )
    value = build_fact_value()
    fact = build_fact(project_id=project.id, subject_key="alice", predicate_key="person.name", current_value=value)
    patch_projection_dependencies(monkeypatch, schema=schema, version=version, facts=[fact])

    result = run_async(
        projection_service.project_dynamic_schema_version(
            session,
            project_id=project.id,
            schema_id=schema.id,
            version_id=version.id,
        )
    )

    assert result.version_status == status


def test_cross_project_schema_or_version_is_rejected(monkeypatch) -> None:
    session = ReadOnlySession()
    project = build_project()
    other_project = build_project()
    schema = build_schema(project_id=other_project.id)
    version = build_version(schema_id=schema.id, status="active")
    attach_version_fields(version, [build_field(version_id=version.id, field_key="name", predicate_key="person.name", display_order=0)])
    patch_projection_dependencies(monkeypatch, schema=schema, version=version, facts=[])

    with pytest.raises(projection_service.DynamicSchemaProjectionNotFoundError):
        run_async(
            projection_service.project_dynamic_schema_version(
                session,
                project_id=project.id,
                schema_id=schema.id,
                version_id=version.id,
            )
        )


def test_hidden_fields_are_excluded_by_default_and_can_be_included(monkeypatch) -> None:
    session = ReadOnlySession()
    project = build_project()
    schema = build_schema(project_id=project.id)
    version = build_version(schema_id=schema.id, status="active")
    fields = [
        build_field(version_id=version.id, field_key="visible", predicate_key="person.name", display_order=0),
        build_field(version_id=version.id, field_key="secret", predicate_key="person.secret", display_order=1, is_hidden=True),
    ]
    attach_version_fields(version, fields)
    value = build_fact_value()
    secret_value = build_fact_value(value_json="secret", normalized_value_text="secret")
    facts = [
        build_fact(project_id=project.id, subject_key="alice", predicate_key="person.name", current_value=value),
        build_fact(project_id=project.id, subject_key="alice", predicate_key="person.secret", current_value=secret_value),
    ]
    patch_projection_dependencies(monkeypatch, schema=schema, version=version, facts=facts)

    hidden_excluded = run_async(
        projection_service.project_dynamic_schema_version(
            session,
            project_id=project.id,
            schema_id=schema.id,
            version_id=version.id,
        )
    )
    hidden_included = run_async(
        projection_service.project_dynamic_schema_version(
            session,
            project_id=project.id,
            schema_id=schema.id,
            version_id=version.id,
            include_hidden=True,
        )
    )

    assert [field.field_key for field in hidden_excluded.records[0].fields] == ["visible"]
    assert [field.field_key for field in hidden_included.records[0].fields] == ["visible", "secret"]


def test_subject_kind_isolated_from_other_facts(monkeypatch) -> None:
    session = ReadOnlySession()
    project = build_project()
    schema = build_schema(project_id=project.id, subject_kind="person")
    version = build_version(schema_id=schema.id, status="active")
    attach_version_fields(version, [build_field(version_id=version.id, field_key="name", predicate_key="person.name", display_order=0)])
    person_value = build_fact_value()
    org_value = build_fact_value(value_json="Acme", normalized_value_text="Acme")
    facts = [
        build_fact(project_id=project.id, subject_kind="person", subject_key="alice", predicate_key="person.name", current_value=person_value),
        build_fact(project_id=project.id, subject_kind="organization", subject_key="acme", predicate_key="person.name", current_value=org_value),
    ]
    patch_projection_dependencies(monkeypatch, schema=schema, version=version, facts=facts)

    result = run_async(
        projection_service.project_dynamic_schema_version(
            session,
            project_id=project.id,
            schema_id=schema.id,
            version_id=version.id,
        )
    )

    assert [record.subject_key for record in result.records] == ["alice"]


def test_one_field_with_null_scope_matches_exact_null_scope(monkeypatch) -> None:
    session = ReadOnlySession()
    project = build_project()
    schema = build_schema(project_id=project.id)
    version = build_version(schema_id=schema.id, status="active")
    attach_version_fields(version, [build_field(version_id=version.id, field_key="name", predicate_key="person.name", display_order=0)])
    null_scope_value = build_fact_value(value_json="Alice")
    scoped_value = build_fact_value(value_json="Alias")
    facts = [
        build_fact(project_id=project.id, subject_key="alice", predicate_key="person.name", scope_key=None, current_value=null_scope_value),
        build_fact(project_id=project.id, subject_key="alice", predicate_key="person.name", scope_key="alias", current_value=scoped_value),
    ]
    patch_projection_dependencies(monkeypatch, schema=schema, version=version, facts=facts)

    result = run_async(
        projection_service.project_dynamic_schema_version(
            session,
            project_id=project.id,
            schema_id=schema.id,
            version_id=version.id,
        )
    )

    assert [value.value_json for value in result.records[0].fields[0].values] == ["Alice"]


def test_many_field_aggregates_scopes_with_stable_sort(monkeypatch) -> None:
    session = ReadOnlySession()
    project = build_project()
    schema = build_schema(project_id=project.id)
    version = build_version(schema_id=schema.id, status="active")
    attach_version_fields(
        version,
        [build_field(version_id=version.id, field_key="aliases", predicate_key="person.alias", display_order=0, cardinality="many")],
    )
    facts = [
        build_fact(
            project_id=project.id,
            subject_key="alice",
            predicate_key="person.alias",
            scope_key="z_scope",
            identity_hash="f" * 64,
            current_value=build_fact_value(value_json="Zulu"),
        ),
        build_fact(
            project_id=project.id,
            subject_key="alice",
            predicate_key="person.alias",
            scope_key=None,
            identity_hash="a" * 64,
            current_value=build_fact_value(value_json="Primary"),
        ),
        build_fact(
            project_id=project.id,
            subject_key="alice",
            predicate_key="person.alias",
            scope_key="a_scope",
            identity_hash="b" * 64,
            current_value=build_fact_value(value_json="Alpha"),
        ),
    ]
    patch_projection_dependencies(monkeypatch, schema=schema, version=version, facts=facts)

    result = run_async(
        projection_service.project_dynamic_schema_version(
            session,
            project_id=project.id,
            schema_id=schema.id,
            version_id=version.id,
        )
    )

    projected_values = result.records[0].fields[0].values
    assert [value.value_json for value in projected_values] == ["Primary", "Alpha", "Zulu"]
    assert [value.scope_key for value in projected_values] == [None, "a_scope", "z_scope"]


def test_required_missing_field_is_marked(monkeypatch) -> None:
    session = ReadOnlySession()
    project = build_project()
    schema = build_schema(project_id=project.id)
    version = build_version(schema_id=schema.id, status="active")
    attach_version_fields(
        version,
        [build_field(version_id=version.id, field_key="birth_date", predicate_key="person.birth_date", display_order=0, is_required=True)],
    )
    patch_projection_dependencies(monkeypatch, schema=schema, version=version, facts=[])

    result = run_async(
        projection_service.project_dynamic_schema_version(
            session,
            project_id=project.id,
            schema_id=schema.id,
            version_id=version.id,
            subject_keys=["alice"],
        )
    )

    assert result.records[0].fields[0].is_missing is True
    assert result.records[0].required_missing_field_keys == ["birth_date"]


def test_type_incompatible_value_is_returned_with_issue(monkeypatch) -> None:
    session = ReadOnlySession()
    project = build_project()
    schema = build_schema(project_id=project.id)
    version = build_version(schema_id=schema.id, status="active")
    attach_version_fields(
        version,
        [build_field(version_id=version.id, field_key="birth_date", predicate_key="person.birth_date", display_order=0, expected_value_type="date")],
    )
    value = build_fact_value(value_type="string", value_json="not-a-date", normalized_value_text="not-a-date")
    fact = build_fact(project_id=project.id, subject_key="alice", predicate_key="person.birth_date", current_value=value)
    patch_projection_dependencies(monkeypatch, schema=schema, version=version, facts=[fact])

    result = run_async(
        projection_service.project_dynamic_schema_version(
            session,
            project_id=project.id,
            schema_id=schema.id,
            version_id=version.id,
        )
    )

    projected_field = result.records[0].fields[0]
    assert projected_field.values[0].value_json == "not-a-date"
    assert projected_field.values[0].type_compatible is False
    assert projected_field.issues


def test_non_accepted_or_missing_current_value_is_rejected(monkeypatch) -> None:
    session = ReadOnlySession()
    project = build_project()
    schema = build_schema(project_id=project.id)
    version = build_version(schema_id=schema.id, status="active")
    attach_version_fields(version, [build_field(version_id=version.id, field_key="name", predicate_key="person.name", display_order=0)])
    rejected_value = build_fact_value(status="rejected", value_json="Alice")
    fact = build_fact(project_id=project.id, subject_key="alice", predicate_key="person.name", current_value=rejected_value)
    patch_projection_dependencies(monkeypatch, schema=schema, version=version, facts=[fact])

    with pytest.raises(projection_service.ProjectionStateCorruptionError):
        run_async(
            projection_service.project_dynamic_schema_version(
                session,
                project_id=project.id,
                schema_id=schema.id,
                version_id=version.id,
            )
        )


def test_current_value_belonging_to_other_fact_is_rejected(monkeypatch) -> None:
    session = ReadOnlySession()
    project = build_project()
    schema = build_schema(project_id=project.id)
    version = build_version(schema_id=schema.id, status="active")
    attach_version_fields(version, [build_field(version_id=version.id, field_key="name", predicate_key="person.name", display_order=0)])
    value = build_fact_value(value_json="Alice")
    fact = build_fact(project_id=project.id, subject_key="alice", predicate_key="person.name", current_value=value)
    value.fact_id = uuid.uuid4()
    patch_projection_dependencies(monkeypatch, schema=schema, version=version, facts=[fact])

    with pytest.raises(projection_service.ProjectionStateCorruptionError):
        run_async(
            projection_service.project_dynamic_schema_version(
                session,
                project_id=project.id,
                schema_id=schema.id,
                version_id=version.id,
            )
        )


def test_evidence_order_and_location_fields_are_projected(monkeypatch) -> None:
    session = ReadOnlySession()
    project = build_project()
    schema = build_schema(project_id=project.id)
    version = build_version(schema_id=schema.id, status="active")
    attach_version_fields(version, [build_field(version_id=version.id, field_key="name", predicate_key="person.name", display_order=0)])
    evidence_a = build_source_evidence(excerpt="A", location_key="md:a", start_line=2, end_line=2, start_offset=3, end_offset=4)
    evidence_b = build_source_evidence(excerpt="B", location_key="md:b", start_line=1, end_line=1, start_offset=0, end_offset=1)
    links = [
        build_evidence_link(source_order=2, role="context", evidence=evidence_a),
        build_evidence_link(source_order=1, role="supporting", is_primary=True, evidence=evidence_b),
    ]
    value = build_fact_value(evidence_links=links)
    fact = build_fact(project_id=project.id, subject_key="alice", predicate_key="person.name", current_value=value)
    patch_projection_dependencies(monkeypatch, schema=schema, version=version, facts=[fact])

    result = run_async(
        projection_service.project_dynamic_schema_version(
            session,
            project_id=project.id,
            schema_id=schema.id,
            version_id=version.id,
        )
    )

    evidences = result.records[0].fields[0].values[0].evidences
    assert [evidence.excerpt for evidence in evidences] == ["B", "A"]
    assert evidences[0].location_key == "md:b"
    assert evidences[0].start_line == 1
    assert evidences[0].start_offset == 0


def test_subject_keys_preserve_request_order_and_allow_empty_records(monkeypatch) -> None:
    session = ReadOnlySession()
    project = build_project()
    schema = build_schema(project_id=project.id)
    version = build_version(schema_id=schema.id, status="active")
    attach_version_fields(version, [build_field(version_id=version.id, field_key="name", predicate_key="person.name", display_order=0)])
    value = build_fact_value(value_json="Alice")
    fact = build_fact(project_id=project.id, subject_key="alice", predicate_key="person.name", current_value=value)
    patch_projection_dependencies(monkeypatch, schema=schema, version=version, facts=[fact])

    result = run_async(
        projection_service.project_dynamic_schema_version(
            session,
            project_id=project.id,
            schema_id=schema.id,
            version_id=version.id,
            subject_keys=["missing", "alice"],
        )
    )

    assert [record.subject_key for record in result.records] == ["missing", "alice"]
    assert result.records[0].fields[0].values == []
    assert result.records[1].fields[0].values[0].value_json == "Alice"


def test_service_does_not_trigger_write_operations(monkeypatch) -> None:
    session = ReadOnlySession()
    project = build_project()
    schema = build_schema(project_id=project.id)
    version = build_version(schema_id=schema.id, status="active")
    attach_version_fields(version, [build_field(version_id=version.id, field_key="name", predicate_key="person.name", display_order=0)])
    value = build_fact_value()
    fact = build_fact(project_id=project.id, subject_key="alice", predicate_key="person.name", current_value=value)
    patch_projection_dependencies(monkeypatch, schema=schema, version=version, facts=[fact])

    result = run_async(
        projection_service.project_dynamic_schema_version(
            session,
            project_id=project.id,
            schema_id=schema.id,
            version_id=version.id,
        )
    )

    assert result.records[0].subject_key == "alice"


def test_current_wrapper_does_not_access_schema_current_version(monkeypatch) -> None:
    session = ReadOnlySession()
    project = build_project()
    version_id = uuid.uuid4()
    schema = StubSchemaWithoutCurrentVersion(project_id=project.id, current_version_id=version_id)
    version = build_version(schema_id=schema.id, version_id=version_id, status="active")
    attach_version_fields(version, [build_field(version_id=version.id, field_key="name", predicate_key="person.name", display_order=0)])
    value = build_fact_value()
    fact = build_fact(project_id=project.id, subject_key="alice", predicate_key="person.name", current_value=value)
    patch_projection_dependencies(monkeypatch, schema=schema, version=version, facts=[fact])

    result = run_async(
        projection_service.project_current_dynamic_schema(
            session,
            project_id=project.id,
            schema_id=schema.id,
        )
    )

    assert result.records[0].subject_key == "alice"


def test_one_field_matching_multiple_null_scope_facts_is_corruption(monkeypatch) -> None:
    session = ReadOnlySession()
    project = build_project()
    schema = build_schema(project_id=project.id)
    version = build_version(schema_id=schema.id, status="active")
    attach_version_fields(version, [build_field(version_id=version.id, field_key="name", predicate_key="person.name", display_order=0)])
    facts = [
        build_fact(project_id=project.id, subject_key="alice", predicate_key="person.name", current_value=build_fact_value(value_json="A")),
        build_fact(project_id=project.id, subject_key="alice", predicate_key="person.name", current_value=build_fact_value(value_json="B")),
    ]
    patch_projection_dependencies(monkeypatch, schema=schema, version=version, facts=facts)

    with pytest.raises(projection_service.ProjectionStateCorruptionError):
        run_async(
            projection_service.project_dynamic_schema_version(
                session,
                project_id=project.id,
                schema_id=schema.id,
                version_id=version.id,
            )
        )


def test_projection_repository_uses_explicit_loader_strategy(monkeypatch) -> None:
    from app.repositories import dynamic_schema_projection as projection_repository

    recorded: dict[str, list[object]] = {"joined": [], "selectin": []}

    class FakeResult:
        def scalar_one_or_none(self):
            return None

        def scalars(self):
            class FakeScalars:
                def unique(self_inner):
                    return self_inner

                def all(self_inner):
                    return []

            return FakeScalars()

    class FakeSession:
        async def execute(self, _statement):
            return FakeResult()

    def tracked_joinedload(attribute):
        recorded["joined"].append(attribute)
        return sa_joinedload(attribute)

    def tracked_selectinload(attribute):
        recorded["selectin"].append(attribute)
        return sa_selectinload(attribute)

    monkeypatch.setattr(projection_repository, "joinedload", tracked_joinedload)
    monkeypatch.setattr(projection_repository, "selectinload", tracked_selectinload)

    run_async(
        projection_repository.get_dynamic_schema_version_with_fields_for_projection(
            FakeSession(),
            schema_id=uuid.uuid4(),
            version_id=uuid.uuid4(),
        )
    )
    run_async(
        projection_repository.list_facts_for_projection(
            FakeSession(),
            project_id=uuid.uuid4(),
            subject_kind="person",
            predicate_keys=["person.name"],
        )
    )

    source = inspect.getsource(projection_repository.list_facts_for_projection)
    assert any(
        attribute is DynamicSchemaVersion.schema or attribute is Fact.current_value
        for attribute in recorded["joined"]
    )
    assert any(attribute is FactValue.evidence_links for attribute in recorded["selectin"])
    assert "selectinload(FactValue.evidence_links)" in source
    assert "joinedload(FactEvidenceLink.evidence)" in source
    assert "joinedload(SourceEvidence.block)" in source
    assert "'" not in source.split("options(", 1)[1]
