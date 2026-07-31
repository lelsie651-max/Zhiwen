import asyncio
from datetime import datetime, timezone
import uuid
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError

from app.models.document import Document
from app.models.document_content import DocumentBlock, ExtractionRun, SourceEvidence
from app.models.document_revision import DocumentRevision
from app.models.fact import Fact, FactStatus, FactValue
from app.models.project import Project
from app.models.project_member import ProjectMember
from app.models.user import User
from app.schemas.fact import FactIdentityInput, FactValueInput
from app.schemas.fact_commands import (
    AIProposalInput,
    FactEvidenceInput,
    FactValueDecisionInput,
    HumanFactValueInput,
)
from app.services import fact as fact_service


def run_async(awaitable):
    return asyncio.run(awaitable)


class FakeSavepoint:
    def __init__(self) -> None:
        self.commit_called = False
        self.rollback_called = False

    async def commit(self) -> None:
        self.commit_called = True

    async def rollback(self) -> None:
        self.rollback_called = True


class FakeSession:
    def __init__(self) -> None:
        self.commit_called = False
        self.rollback_called = False
        self.commit_count = 0
        self.rollback_count = 0
        self.savepoints: list[FakeSavepoint] = []

    async def commit(self) -> None:
        self.commit_called = True
        self.commit_count += 1

    async def rollback(self) -> None:
        self.rollback_called = True
        self.rollback_count += 1

    async def begin_nested(self) -> FakeSavepoint:
        savepoint = FakeSavepoint()
        self.savepoints.append(savepoint)
        return savepoint


def build_identity(
    *,
    scope_key: str | None = None,
    subject_kind: str = "company",
    subject_key: str = "acme",
    subject_entity_id: uuid.UUID | None = None,
) -> FactIdentityInput:
    return FactIdentityInput(
        subject_kind=subject_kind,
        subject_key=subject_key,
        subject_entity_id=subject_entity_id,
        predicate_key="legal_name",
        scope_key=scope_key,
    )


def build_value_input(
    *,
    value_type: str = "string",
    value_json="Acme Corp",
    referenced_entity_id: uuid.UUID | None = None,
    language_code: str | None = "zh-CN",
    confidence: float | None = 0.9,
) -> FactValueInput:
    return FactValueInput(
        value_type=value_type,
        value_json=value_json,
        referenced_entity_id=referenced_entity_id,
        language_code=language_code,
        confidence=confidence,
    )


def build_ai_payload(
    evidence_ids: list[uuid.UUID],
    *,
    identity: FactIdentityInput | None = None,
    value: FactValueInput | None = None,
) -> AIProposalInput:
    return AIProposalInput(
        identity=identity or build_identity(),
        value=value or build_value_input(),
        evidences=[
            FactEvidenceInput(
                evidence_id=evidence_id,
                role="supporting" if index == 0 else "context",
                is_primary=index == 0,
            )
            for index, evidence_id in enumerate(evidence_ids)
        ],
    )


def build_project(*, project_id: uuid.UUID | None = None) -> Project:
    actual_project_id = project_id or uuid.uuid4()
    return Project(
        id=actual_project_id,
        name="Zhiwen Project",
        slug=f"proj-{str(actual_project_id)[:8]}",
        created_by_id=uuid.uuid4(),
    )


def build_run(
    *,
    project: Project | None = None,
    run_id: uuid.UUID | None = None,
    status: str = "completed",
    outcome: str = "success",
) -> ExtractionRun:
    actual_project = project or build_project()
    document = Document(
        id=uuid.uuid4(),
        project_id=actual_project.id,
        title="Document",
        created_by_id=uuid.uuid4(),
    )
    revision = DocumentRevision(
        id=uuid.uuid4(),
        document_id=document.id,
        revision_no=1,
        upload_intent="new_document",
        supersedes_revision_id=None,
        original_filename="input.md",
        storage_key="files/aa/input.bin",
        mime_type="text/markdown",
        file_size_bytes=128,
        sha256="a" * 64,
        source_authority="pending",
        status="uploaded",
        uploaded_by_id=uuid.uuid4(),
        uploaded_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    extraction_run = ExtractionRun(
        id=run_id or uuid.uuid4(),
        revision_id=revision.id,
        attempt_no=1,
        status=status,
        outcome=outcome,
        extractor_name="deterministic-extractor",
        extractor_version="1.0.0",
        detected_format="md",
        detected_encoding="utf-8",
        page_count=None,
        character_count=10,
        block_count=1,
        warnings=[],
        content_metadata={},
    )

    document.project = actual_project
    revision.document = document
    extraction_run.revision = revision
    return extraction_run


def build_evidence(
    *,
    extraction_run: ExtractionRun,
    evidence_id: uuid.UUID | None = None,
) -> SourceEvidence:
    block = DocumentBlock(
        id=uuid.uuid4(),
        extraction_run_id=extraction_run.id,
        source_order=0,
        block_type="paragraph",
        raw_text="alpha beta gamma",
        normalized_text="alpha beta gamma",
        location_key=f"loc-{uuid.uuid4()}",
        anchor_hash="a" * 64,
        block_index=0,
        heading_path=[],
        block_metadata={},
    )
    evidence = SourceEvidence(
        id=evidence_id or uuid.uuid4(),
        block_id=block.id,
        start_offset=0,
        end_offset=5,
        excerpt="alpha",
        excerpt_hash="b" * 64,
    )
    block.extraction_run = extraction_run
    evidence.block = block
    return evidence


def build_inference_run_context(
    *,
    project_id: uuid.UUID,
    extraction_run_ids: list[uuid.UUID],
    source_block_ids: list[uuid.UUID] | None = None,
    run_id: uuid.UUID | None = None,
    status: str = "completed",
    task_type: str = "fact_extraction",
):
    return SimpleNamespace(
        run_id=run_id or uuid.uuid4(),
        project_id=project_id,
        task_type=task_type,
        status=status,
        batch_id=uuid.uuid4(),
        extraction_run_id_snapshots=frozenset(extraction_run_ids),
        source_block_id_snapshots=frozenset(source_block_ids or []),
    )


def build_entity_context(
    *,
    entity_id: uuid.UUID | None = None,
    project_id: uuid.UUID,
    entity_type: str = "company",
    canonical_key: str = "acme",
    status: str = "active",
):
    return SimpleNamespace(
        entity_id=entity_id or uuid.uuid4(),
        project_id=project_id,
        entity_type=entity_type,
        canonical_key=canonical_key,
        status=status,
    )


async def fake_completed_context(_inference_run_id, context):
    assert _inference_run_id == context.run_id
    return context


def build_fact(
    *,
    project_id: uuid.UUID,
    identity_hash: str,
    subject_kind: str = "company",
    subject_key: str = "acme",
    subject_entity_id: uuid.UUID | None = None,
    status: str = FactStatus.ACTIVE.value,
    current_value_id: uuid.UUID | None = None,
) -> Fact:
    return Fact(
        id=uuid.uuid4(),
        project_id=project_id,
        subject_kind=subject_kind,
        subject_key=subject_key,
        subject_entity_id=subject_entity_id,
        predicate_key="legal_name",
        scope_key=None,
        identity_hash=identity_hash,
        status=status,
        current_value_id=current_value_id,
        created_by_id=None,
    )


def build_user(*, user_id: uuid.UUID | None = None, status: str = "active") -> User:
    actual_user_id = user_id or uuid.uuid4()
    return User(
        id=actual_user_id,
        handle=f"user-{str(actual_user_id)[:8]}",
        display_name="Fact Actor",
        status=status,
    )


def build_project_member(*, project_id: uuid.UUID, user_id: uuid.UUID, role: str) -> ProjectMember:
    return ProjectMember(
        id=uuid.uuid4(),
        project_id=project_id,
        user_id=user_id,
        role=role,
    )


def build_human_payload(
    *,
    identity: FactIdentityInput | None = None,
    value: FactValueInput | None = None,
    evidence_ids: list[uuid.UUID] | None = None,
) -> HumanFactValueInput:
    actual_evidence_ids = evidence_ids or []
    evidences = [
        FactEvidenceInput(
            evidence_id=evidence_id,
            role="supporting" if index == 0 else "context",
            is_primary=index == 0,
        )
        for index, evidence_id in enumerate(actual_evidence_ids)
    ]
    return HumanFactValueInput(
        identity=identity or build_identity(),
        value=value or build_value_input(),
        evidences=evidences,
    )


def patch_actor_permission(monkeypatch, *, project_id: uuid.UUID, actor: User, role: str) -> None:
    project_member = build_project_member(project_id=project_id, user_id=actor.id, role=role)

    async def fake_get_user_by_id(_session, user_id):
        if user_id == actor.id:
            return actor
        return None

    async def fake_get_project_member_for_project(_session, *, project_id, user_id):
        if project_id == project_member.project_id and user_id == project_member.user_id:
            return project_member
        return None

    monkeypatch.setattr(fact_service.user_repository, "get_user_by_id", fake_get_user_by_id)
    monkeypatch.setattr(
        fact_service.fact_repository,
        "get_project_member_for_project",
        fake_get_project_member_for_project,
    )


def patch_create_human_dependencies(
    monkeypatch,
    *,
    evidences: list[SourceEvidence] | None = None,
    fact: Fact | None = None,
    current_value: FactValue | None = None,
    next_version: int = 1,
    create_links_error: Exception | None = None,
):
    captured: dict[str, object] = {"updated_values": []}

    async def fake_list_source_evidences_with_context(_session, _evidence_ids):
        return list(evidences or [])

    async def fake_get_fact_by_identity_for_update(_session, **_kwargs):
        return fact

    async def fake_create_fact(_session, created_fact):
        captured["fact"] = created_fact
        return created_fact

    async def fake_get_next_fact_version_no(_session, _fact_id):
        return next_version

    async def fake_create_fact_value(_session, fact_value):
        captured["fact_value"] = fact_value
        return fact_value

    async def fake_create_fact_evidence_links(_session, links):
        if create_links_error is not None:
            raise create_links_error
        captured["links"] = links
        return links

    async def fake_update_fact(_session, updated_fact):
        captured["updated_fact"] = updated_fact
        return updated_fact

    async def fake_update_fact_value(_session, updated_value):
        captured["updated_values"].append(updated_value)
        return updated_value

    async def fake_get_fact_value_for_update(_session, *, project_id, fact_value_id):
        if current_value is not None and current_value.id == fact_value_id:
            return current_value
        return None

    monkeypatch.setattr(
        fact_service.fact_repository,
        "list_source_evidences_with_context",
        fake_list_source_evidences_with_context,
    )
    monkeypatch.setattr(
        fact_service.fact_repository,
        "get_fact_by_identity_for_update",
        fake_get_fact_by_identity_for_update,
    )
    monkeypatch.setattr(fact_service.fact_repository, "create_fact", fake_create_fact)
    monkeypatch.setattr(
        fact_service.fact_repository,
        "get_next_fact_version_no",
        fake_get_next_fact_version_no,
    )
    monkeypatch.setattr(
        fact_service.fact_repository,
        "create_fact_value",
        fake_create_fact_value,
    )
    monkeypatch.setattr(
        fact_service.fact_repository,
        "create_fact_evidence_links",
        fake_create_fact_evidence_links,
    )
    monkeypatch.setattr(fact_service.fact_repository, "update_fact", fake_update_fact)
    monkeypatch.setattr(
        fact_service.fact_repository,
        "update_fact_value",
        fake_update_fact_value,
    )
    monkeypatch.setattr(
        fact_service.fact_repository,
        "get_fact_value_for_update",
        fake_get_fact_value_for_update,
    )
    return captured


def patch_decision_dependencies(
    monkeypatch,
    *,
    fact: Fact,
    target_value: FactValue,
    current_value: FactValue | None = None,
):
    captured: dict[str, object] = {"updated_values": []}

    async def fake_get_fact_value_for_update(_session, *, project_id, fact_value_id):
        if fact_value_id == target_value.id:
            return target_value
        if current_value is not None and fact_value_id == current_value.id:
            return current_value
        return None

    async def fake_get_fact_by_id_for_update(_session, *, project_id, fact_id):
        if fact_id == fact.id and project_id == fact.project_id:
            return fact
        return None

    async def fake_update_fact_value(_session, fact_value):
        captured["updated_values"].append(fact_value)
        return fact_value

    async def fake_update_fact(_session, updated_fact):
        captured["updated_fact"] = updated_fact
        return updated_fact

    monkeypatch.setattr(
        fact_service.fact_repository,
        "get_fact_value_for_update",
        fake_get_fact_value_for_update,
    )
    monkeypatch.setattr(
        fact_service.fact_repository,
        "get_fact_by_id_for_update",
        fake_get_fact_by_id_for_update,
    )
    monkeypatch.setattr(
        fact_service.fact_repository,
        "update_fact_value",
        fake_update_fact_value,
    )
    monkeypatch.setattr(
        fact_service.fact_repository,
        "update_fact",
        fake_update_fact,
    )
    return captured


def test_build_fact_identity_hash_is_deterministic() -> None:
    identity = build_identity()

    hash_one = fact_service.build_fact_identity_hash(identity)
    hash_two = fact_service.build_fact_identity_hash(
        FactIdentityInput(
            subject_kind="company",
            subject_key="acme",
            predicate_key="legal_name",
            scope_key=None,
        )
    )

    assert hash_one == hash_two


def test_build_fact_identity_hash_changes_when_scope_changes() -> None:
    assert fact_service.build_fact_identity_hash(build_identity(scope_key=None)) != fact_service.build_fact_identity_hash(
        build_identity(scope_key="2026")
    )


def test_value_hash_is_stable_for_equivalent_object_key_order() -> None:
    result_one = fact_service.normalize_fact_value_input(
        build_value_input(
            value_type="object",
            value_json={"b": 2, "a": 1},
            language_code=None,
            confidence=None,
        )
    )
    result_two = fact_service.normalize_fact_value_input(
        build_value_input(
            value_type="object",
            value_json={"a": 1, "b": 2},
            language_code=None,
            confidence=None,
        )
    )

    assert result_one.value_hash == result_two.value_hash
    assert result_one.normalized_value_text == result_two.normalized_value_text


def test_entity_ref_value_hash_binds_referenced_entity_id() -> None:
    entity_one = uuid.uuid4()
    entity_two = uuid.uuid4()

    result_one = fact_service.normalize_fact_value_input(
        build_value_input(
            value_type="entity_ref",
            value_json={"kind": "company", "key": " Acme "},
            referenced_entity_id=entity_one,
            language_code=None,
            confidence=None,
        )
    )
    result_two = fact_service.normalize_fact_value_input(
        build_value_input(
            value_type="entity_ref",
            value_json={"kind": "company", "key": " Acme "},
            referenced_entity_id=entity_two,
            language_code=None,
            confidence=None,
        )
    )

    assert result_one.value_json == {"key": "Acme", "kind": "company"}
    assert result_one.value_hash != result_two.value_hash


def test_entity_ref_value_hash_normalizes_equivalent_entity_keys() -> None:
    entity_id = uuid.uuid4()

    result_one = fact_service.normalize_fact_value_input(
        build_value_input(
            value_type="entity_ref",
            value_json={"kind": "company", "key": " ACME "},
            referenced_entity_id=entity_id,
            language_code=None,
            confidence=None,
        )
    )
    result_two = fact_service.normalize_fact_value_input(
        build_value_input(
            value_type="entity_ref",
            value_json={"kind": "company", "key": "ａｃｍｅ"},
            referenced_entity_id=entity_id,
            language_code=None,
            confidence=None,
        )
    )

    assert result_one.value_hash == result_two.value_hash
    assert result_one.normalized_value_text != result_two.normalized_value_text


def test_non_entity_ref_value_hash_keeps_existing_algorithm() -> None:
    result = fact_service.normalize_fact_value_input(
        build_value_input(
            value_type="string",
            value_json="  Acme Corp  ",
            language_code=None,
            confidence=None,
        )
    )

    expected = fact_service.hashlib.sha256(
        fact_service.json.dumps(
            {
                "value": "Acme Corp",
                "value_type": "string",
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert result.value_hash == expected


@pytest.mark.parametrize(
    ("value_type", "value_json", "expected_message"),
    [
        ("string", 1, "string"),
        ("number", True, "number"),
        ("boolean", "true", "boolean"),
        ("date", "2026-13-01", "YYYY-MM-DD"),
        ("datetime", "2026-07-30T10:00:00", "timezone"),
        ("entity_ref", {"kind": "org"}, "kind and key"),
        ("list", {"not": "a list"}, "lists"),
        ("object", ["not", "an", "object"], "objects"),
        ("null", "null", "None"),
    ],
)
def test_value_types_are_strictly_validated(value_type: str, value_json, expected_message: str) -> None:
    referenced_entity_id = uuid.uuid4() if value_type == "entity_ref" else None

    with pytest.raises((fact_service.InvalidFactProposalError, ValidationError)) as exc_info:
        fact_service.normalize_fact_value_input(
            build_value_input(
                value_type=value_type,
                value_json=value_json,
                referenced_entity_id=referenced_entity_id,
                language_code=None,
                confidence=None,
            )
        )

    assert expected_message in str(exc_info.value)


@pytest.mark.parametrize(
    "invalid_number",
    [float("nan"), float("inf"), float("-inf")],
)
def test_nan_and_infinity_are_rejected(invalid_number: float) -> None:
    with pytest.raises(fact_service.InvalidFactProposalError):
        fact_service.normalize_fact_value_input(
            build_value_input(
                value_type="number",
                value_json=invalid_number,
                language_code=None,
                confidence=None,
            )
        )


def test_invalid_datetime_is_rejected() -> None:
    with pytest.raises(fact_service.InvalidFactProposalError):
        fact_service.normalize_fact_value_input(
            build_value_input(
                value_type="datetime",
                value_json="not-a-datetime",
                language_code=None,
                confidence=None,
            )
        )


def test_ai_proposal_schema_enforces_primary_and_supporting_rules() -> None:
    evidence_id = uuid.uuid4()
    with pytest.raises(ValueError):
        AIProposalInput(
            identity=build_identity(),
            value=build_value_input(),
            evidences=[
                FactEvidenceInput(evidence_id=evidence_id, role="contradicting", is_primary=True),
            ],
        )


def test_propose_ai_fact_value_is_always_proposed(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    run = build_run(project=project)
    evidence = build_evidence(extraction_run=run)
    inference_context = build_inference_run_context(
        project_id=project.id,
        extraction_run_ids=[run.id],
        source_block_ids=[evidence.block.id],
    )
    payload = build_ai_payload([evidence.id])
    captured: dict[str, object] = {}

    async def fake_get_extraction_run_with_project(_session, _extraction_run_id):
        return run

    async def fake_get_completed_fact_extraction_run_context(_session, *, inference_run_id):
        assert inference_run_id == inference_context.run_id
        return inference_context

    async def fake_list_source_evidences_with_context(_session, _evidence_ids):
        return [evidence]

    async def fake_get_fact_by_identity_for_update(_session, **_kwargs):
        return None

    async def fake_create_fact(_session, fact):
        captured["fact"] = fact
        return fact

    async def fake_get_next_fact_version_no(_session, _fact_id):
        return 1

    async def fake_create_fact_value(_session, fact_value):
        captured["fact_value"] = fact_value
        return fact_value

    async def fake_create_fact_evidence_links(_session, links):
        captured["links"] = links
        return links

    monkeypatch.setattr(fact_service.fact_repository, "get_extraction_run_with_project", fake_get_extraction_run_with_project)
    monkeypatch.setattr(
        fact_service.inference_repository,
        "get_completed_fact_extraction_run_context",
        fake_get_completed_fact_extraction_run_context,
    )
    monkeypatch.setattr(fact_service.fact_repository, "list_source_evidences_with_context", fake_list_source_evidences_with_context)
    monkeypatch.setattr(fact_service.fact_repository, "get_fact_by_identity_for_update", fake_get_fact_by_identity_for_update)
    monkeypatch.setattr(fact_service.fact_repository, "create_fact", fake_create_fact)
    monkeypatch.setattr(fact_service.fact_repository, "get_next_fact_version_no", fake_get_next_fact_version_no)
    monkeypatch.setattr(fact_service.fact_repository, "create_fact_value", fake_create_fact_value)
    monkeypatch.setattr(fact_service.fact_repository, "create_fact_evidence_links", fake_create_fact_evidence_links)

    result = run_async(
        fact_service.propose_ai_fact_value(
            session,
            project_id=project.id,
            extraction_run_id=run.id,
            inference_run_id=inference_context.run_id,
            payload=payload,
        )
    )

    assert result.status == "proposed"
    assert result.source_kind == "ai"
    assert result.extraction_run_id == run.id
    assert result.inference_run_id == inference_context.run_id
    assert result.created_by_id is None
    assert result.decided_by_id is None
    assert result.decided_at is None
    assert session.commit_called is True
    assert session.commit_count == 1
    assert session.rollback_called is False
    assert captured["links"][0].source_order == 0


def test_current_value_id_is_not_changed_by_ai_proposal(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    run = build_run(project=project)
    evidence = build_evidence(extraction_run=run)
    inference_context = build_inference_run_context(
        project_id=project.id,
        extraction_run_ids=[run.id],
        source_block_ids=[evidence.block.id],
    )
    human_current_value = FactValue(
        id=uuid.uuid4(),
        fact_id=uuid.uuid4(),
        version_no=1,
        value_type="string",
        value_json="Human Value",
        normalized_value_text="Human Value",
        value_hash="a" * 64,
        language_code=None,
        status="accepted",
        source_kind="human",
        extraction_run_id=None,
        confidence=None,
        created_by_id=uuid.uuid4(),
        decided_by_id=uuid.uuid4(),
        decided_at=datetime.now(timezone.utc),
    )
    fact = build_fact(
        project_id=project.id,
        identity_hash=fact_service.build_fact_identity_hash(build_identity()),
        current_value_id=human_current_value.id,
    )
    payload = build_ai_payload([evidence.id])

    async def fake_get_extraction_run_with_project(_session, _extraction_run_id):
        return run

    async def fake_get_completed_fact_extraction_run_context(_session, *, inference_run_id):
        assert inference_run_id == inference_context.run_id
        return inference_context

    async def fake_list_source_evidences_with_context(_session, _evidence_ids):
        return [evidence]

    async def fake_get_fact_by_identity_for_update(_session, **_kwargs):
        return fact

    async def fake_get_next_fact_version_no(_session, _fact_id):
        return 2

    async def fake_create_fact_value(_session, fact_value):
        return fact_value

    async def fake_create_fact_evidence_links(_session, links):
        return links

    monkeypatch.setattr(fact_service.fact_repository, "get_extraction_run_with_project", fake_get_extraction_run_with_project)
    monkeypatch.setattr(
        fact_service.inference_repository,
        "get_completed_fact_extraction_run_context",
        fake_get_completed_fact_extraction_run_context,
    )
    monkeypatch.setattr(fact_service.fact_repository, "list_source_evidences_with_context", fake_list_source_evidences_with_context)
    monkeypatch.setattr(fact_service.fact_repository, "get_fact_by_identity_for_update", fake_get_fact_by_identity_for_update)
    monkeypatch.setattr(fact_service.fact_repository, "get_next_fact_version_no", fake_get_next_fact_version_no)
    monkeypatch.setattr(fact_service.fact_repository, "create_fact_value", fake_create_fact_value)
    monkeypatch.setattr(fact_service.fact_repository, "create_fact_evidence_links", fake_create_fact_evidence_links)

    result = run_async(
        fact_service.propose_ai_fact_value(
            session,
            project_id=project.id,
            extraction_run_id=run.id,
            inference_run_id=inference_context.run_id,
            payload=payload,
        )
    )

    assert result.id != human_current_value.id
    assert fact.current_value_id == human_current_value.id
    assert result.status == "proposed"


def test_existing_human_current_value_is_not_overwritten(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    run = build_run(project=project)
    evidence = build_evidence(extraction_run=run)
    inference_context = build_inference_run_context(
        project_id=project.id,
        extraction_run_ids=[run.id],
        source_block_ids=[evidence.block.id],
    )
    current_value_id = uuid.uuid4()
    fact = build_fact(
        project_id=project.id,
        identity_hash=fact_service.build_fact_identity_hash(build_identity()),
        current_value_id=current_value_id,
    )
    payload = build_ai_payload([evidence.id])

    async def fake_get_extraction_run_with_project(_session, _extraction_run_id):
        return run

    async def fake_get_completed_fact_extraction_run_context(_session, *, inference_run_id):
        assert inference_run_id == inference_context.run_id
        return inference_context

    async def fake_list_source_evidences_with_context(_session, _evidence_ids):
        return [evidence]

    async def fake_get_fact_by_identity_for_update(_session, **_kwargs):
        return fact

    async def fake_get_next_fact_version_no(_session, _fact_id):
        return 3

    async def fake_create_fact_value(_session, fact_value):
        return fact_value

    async def fake_create_fact_evidence_links(_session, links):
        return links

    monkeypatch.setattr(fact_service.fact_repository, "get_extraction_run_with_project", fake_get_extraction_run_with_project)
    monkeypatch.setattr(
        fact_service.inference_repository,
        "get_completed_fact_extraction_run_context",
        fake_get_completed_fact_extraction_run_context,
    )
    monkeypatch.setattr(fact_service.fact_repository, "list_source_evidences_with_context", fake_list_source_evidences_with_context)
    monkeypatch.setattr(fact_service.fact_repository, "get_fact_by_identity_for_update", fake_get_fact_by_identity_for_update)
    monkeypatch.setattr(fact_service.fact_repository, "get_next_fact_version_no", fake_get_next_fact_version_no)
    monkeypatch.setattr(fact_service.fact_repository, "create_fact_value", fake_create_fact_value)
    monkeypatch.setattr(fact_service.fact_repository, "create_fact_evidence_links", fake_create_fact_evidence_links)

    result = run_async(
        fact_service.propose_ai_fact_value(
            session,
            project_id=project.id,
            extraction_run_id=run.id,
            inference_run_id=inference_context.run_id,
            payload=payload,
        )
    )

    assert fact.current_value_id == current_value_id
    assert result.version_no == 3


def test_evidence_from_other_project_is_rejected(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    other_project = build_project()
    run = build_run(project=project)
    inference_context = build_inference_run_context(
        project_id=project.id,
        extraction_run_ids=[run.id],
    )
    evidence = build_evidence(extraction_run=build_run(project=other_project))
    payload = build_ai_payload([evidence.id])

    async def fake_get_extraction_run_with_project(_session, _extraction_run_id):
        return run

    async def fake_get_completed_fact_extraction_run_context(_session, *, inference_run_id):
        assert inference_run_id == inference_context.run_id
        return inference_context

    async def fake_list_source_evidences_with_context(_session, _evidence_ids):
        return [evidence]

    monkeypatch.setattr(fact_service.fact_repository, "get_extraction_run_with_project", fake_get_extraction_run_with_project)
    monkeypatch.setattr(
        fact_service.inference_repository,
        "get_completed_fact_extraction_run_context",
        fake_get_completed_fact_extraction_run_context,
    )
    monkeypatch.setattr(fact_service.fact_repository, "list_source_evidences_with_context", fake_list_source_evidences_with_context)

    with pytest.raises(fact_service.InvalidFactProposalError):
        run_async(
            fact_service.propose_ai_fact_value(
                session,
                project_id=project.id,
                extraction_run_id=run.id,
                inference_run_id=inference_context.run_id,
                payload=payload,
            )
        )


def test_evidence_from_other_run_is_rejected(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    run = build_run(project=project)
    inference_context = build_inference_run_context(
        project_id=project.id,
        extraction_run_ids=[run.id],
    )
    other_run = build_run(project=project)
    evidence = build_evidence(extraction_run=other_run)
    payload = build_ai_payload([evidence.id])

    async def fake_get_extraction_run_with_project(_session, _extraction_run_id):
        return run

    async def fake_get_completed_fact_extraction_run_context(_session, *, inference_run_id):
        assert inference_run_id == inference_context.run_id
        return inference_context

    async def fake_list_source_evidences_with_context(_session, _evidence_ids):
        return [evidence]

    monkeypatch.setattr(fact_service.fact_repository, "get_extraction_run_with_project", fake_get_extraction_run_with_project)
    monkeypatch.setattr(
        fact_service.inference_repository,
        "get_completed_fact_extraction_run_context",
        fake_get_completed_fact_extraction_run_context,
    )
    monkeypatch.setattr(fact_service.fact_repository, "list_source_evidences_with_context", fake_list_source_evidences_with_context)

    with pytest.raises(fact_service.InvalidFactProposalError):
        run_async(
            fact_service.propose_ai_fact_value(
                session,
                project_id=project.id,
                extraction_run_id=run.id,
                inference_run_id=inference_context.run_id,
                payload=payload,
            )
        )


@pytest.mark.parametrize("outcome", ["failed", "needs_ocr"])
def test_failed_or_needs_ocr_runs_are_rejected(monkeypatch, outcome: str) -> None:
    session = FakeSession()
    project = build_project()
    run = build_run(project=project, outcome=outcome)
    inference_context = build_inference_run_context(
        project_id=project.id,
        extraction_run_ids=[run.id],
    )
    evidence = build_evidence(extraction_run=run)
    payload = build_ai_payload([evidence.id])

    async def fake_get_extraction_run_with_project(_session, _extraction_run_id):
        return run

    monkeypatch.setattr(fact_service.fact_repository, "get_extraction_run_with_project", fake_get_extraction_run_with_project)
    monkeypatch.setattr(
        fact_service.inference_repository,
        "get_completed_fact_extraction_run_context",
        lambda _session, *, inference_run_id: fake_completed_context(inference_run_id, inference_context),
    )

    with pytest.raises(fact_service.InvalidFactProposalError):
        run_async(
            fact_service.propose_ai_fact_value(
                session,
                project_id=project.id,
                extraction_run_id=run.id,
                inference_run_id=inference_context.run_id,
                payload=payload,
            )
        )


@pytest.mark.parametrize("status", ["pending", "running", "failed"])
def test_non_completed_inference_runs_are_rejected(monkeypatch, status: str) -> None:
    session = FakeSession()
    project = build_project()
    run = build_run(project=project)
    inference_context = build_inference_run_context(
        project_id=project.id,
        extraction_run_ids=[run.id],
        status=status,
    )
    payload = build_ai_payload([build_evidence(extraction_run=run).id])

    async def fake_get_extraction_run_with_project(_session, _extraction_run_id):
        return run

    async def fake_get_completed_fact_extraction_run_context(_session, *, inference_run_id):
        assert inference_run_id == inference_context.run_id
        return inference_context

    monkeypatch.setattr(fact_service.fact_repository, "get_extraction_run_with_project", fake_get_extraction_run_with_project)
    monkeypatch.setattr(
        fact_service.inference_repository,
        "get_completed_fact_extraction_run_context",
        fake_get_completed_fact_extraction_run_context,
    )

    with pytest.raises(fact_service.InferenceRunNotEligibleForFactError):
        run_async(
            fact_service.propose_ai_fact_value(
                session,
                project_id=project.id,
                extraction_run_id=run.id,
                inference_run_id=inference_context.run_id,
                payload=payload,
            )
        )
    assert session.rollback_called is True
    assert session.commit_called is False


@pytest.mark.parametrize("task_type", ["schema_inference", "consistency_check"])
def test_non_fact_extraction_inference_runs_are_rejected(monkeypatch, task_type: str) -> None:
    session = FakeSession()
    project = build_project()
    run = build_run(project=project)
    inference_context = build_inference_run_context(
        project_id=project.id,
        extraction_run_ids=[run.id],
        task_type=task_type,
    )
    payload = build_ai_payload([build_evidence(extraction_run=run).id])

    async def fake_get_extraction_run_with_project(_session, _extraction_run_id):
        return run

    async def fake_get_completed_fact_extraction_run_context(_session, *, inference_run_id):
        assert inference_run_id == inference_context.run_id
        return inference_context

    monkeypatch.setattr(fact_service.fact_repository, "get_extraction_run_with_project", fake_get_extraction_run_with_project)
    monkeypatch.setattr(
        fact_service.inference_repository,
        "get_completed_fact_extraction_run_context",
        fake_get_completed_fact_extraction_run_context,
    )

    with pytest.raises(fact_service.InferenceRunNotEligibleForFactError):
        run_async(
            fact_service.propose_ai_fact_value(
                session,
                project_id=project.id,
                extraction_run_id=run.id,
                inference_run_id=inference_context.run_id,
                payload=payload,
            )
        )


def test_cross_project_inference_run_is_rejected(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    run = build_run(project=project)
    inference_context = build_inference_run_context(
        project_id=uuid.uuid4(),
        extraction_run_ids=[run.id],
    )
    payload = build_ai_payload([build_evidence(extraction_run=run).id])

    async def fake_get_extraction_run_with_project(_session, _extraction_run_id):
        return run

    async def fake_get_completed_fact_extraction_run_context(_session, *, inference_run_id):
        assert inference_run_id == inference_context.run_id
        return inference_context

    monkeypatch.setattr(fact_service.fact_repository, "get_extraction_run_with_project", fake_get_extraction_run_with_project)
    monkeypatch.setattr(
        fact_service.inference_repository,
        "get_completed_fact_extraction_run_context",
        fake_get_completed_fact_extraction_run_context,
    )

    with pytest.raises(fact_service.FactInferenceProjectMismatchError):
        run_async(
            fact_service.propose_ai_fact_value(
                session,
                project_id=project.id,
                extraction_run_id=run.id,
                inference_run_id=inference_context.run_id,
                payload=payload,
            )
        )
    assert session.rollback_called is True
    assert session.commit_called is False


def test_inference_run_without_target_extraction_snapshot_is_rejected(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    run = build_run(project=project)
    evidence = build_evidence(extraction_run=run)
    inference_context = build_inference_run_context(
        project_id=project.id,
        extraction_run_ids=[uuid.uuid4()],
        source_block_ids=[evidence.block.id],
    )
    payload = build_ai_payload([evidence.id])

    async def fake_get_extraction_run_with_project(_session, _extraction_run_id):
        return run

    async def fake_get_completed_fact_extraction_run_context(_session, *, inference_run_id):
        assert inference_run_id == inference_context.run_id
        return inference_context

    monkeypatch.setattr(fact_service.fact_repository, "get_extraction_run_with_project", fake_get_extraction_run_with_project)
    monkeypatch.setattr(
        fact_service.inference_repository,
        "get_completed_fact_extraction_run_context",
        fake_get_completed_fact_extraction_run_context,
    )

    with pytest.raises(fact_service.FactInferenceSourceMismatchError):
        run_async(
            fact_service.propose_ai_fact_value(
                session,
                project_id=project.id,
                extraction_run_id=run.id,
                inference_run_id=inference_context.run_id,
                payload=payload,
            )
        )
    assert session.rollback_called is True
    assert session.commit_called is False


def test_evidence_block_not_in_inference_batch_is_rejected(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    run = build_run(project=project)
    evidence = build_evidence(extraction_run=run)
    inference_context = build_inference_run_context(
        project_id=project.id,
        extraction_run_ids=[run.id],
        source_block_ids=[uuid.uuid4()],
    )
    payload = build_ai_payload([evidence.id])

    async def fake_get_extraction_run_with_project(_session, _extraction_run_id):
        return run

    async def fake_get_completed_fact_extraction_run_context(_session, *, inference_run_id):
        assert inference_run_id == inference_context.run_id
        return inference_context

    async def fake_list_source_evidences_with_context(_session, _evidence_ids):
        return [evidence]

    monkeypatch.setattr(fact_service.fact_repository, "get_extraction_run_with_project", fake_get_extraction_run_with_project)
    monkeypatch.setattr(
        fact_service.inference_repository,
        "get_completed_fact_extraction_run_context",
        fake_get_completed_fact_extraction_run_context,
    )
    monkeypatch.setattr(fact_service.fact_repository, "list_source_evidences_with_context", fake_list_source_evidences_with_context)

    with pytest.raises(fact_service.FactInferenceEvidenceMismatchError) as exc_info:
        run_async(
            fact_service.propose_ai_fact_value(
                session,
                project_id=project.id,
                extraction_run_id=run.id,
                inference_run_id=inference_context.run_id,
                payload=payload,
            )
        )

    assert "alpha beta gamma" not in str(exc_info.value)
    assert "alpha" not in str(exc_info.value)
    assert session.rollback_called is True
    assert session.commit_called is False


def test_any_mismatched_evidence_block_rejects_entire_batch(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    run = build_run(project=project)
    evidence_one = build_evidence(extraction_run=run)
    evidence_two = build_evidence(extraction_run=run)
    inference_context = build_inference_run_context(
        project_id=project.id,
        extraction_run_ids=[run.id],
        source_block_ids=[evidence_one.block.id],
    )
    payload = build_ai_payload([evidence_one.id, evidence_two.id])

    async def fake_get_extraction_run_with_project(_session, _extraction_run_id):
        return run

    async def fake_get_completed_fact_extraction_run_context(_session, *, inference_run_id):
        assert inference_run_id == inference_context.run_id
        return inference_context

    async def fake_list_source_evidences_with_context(_session, _evidence_ids):
        return [evidence_one, evidence_two]

    monkeypatch.setattr(fact_service.fact_repository, "get_extraction_run_with_project", fake_get_extraction_run_with_project)
    monkeypatch.setattr(
        fact_service.inference_repository,
        "get_completed_fact_extraction_run_context",
        fake_get_completed_fact_extraction_run_context,
    )
    monkeypatch.setattr(fact_service.fact_repository, "list_source_evidences_with_context", fake_list_source_evidences_with_context)

    with pytest.raises(fact_service.FactInferenceEvidenceMismatchError):
        run_async(
            fact_service.propose_ai_fact_value(
                session,
                project_id=project.id,
                extraction_run_id=run.id,
                inference_run_id=inference_context.run_id,
                payload=payload,
            )
        )

    assert session.rollback_called is True
    assert session.commit_called is False


def test_all_evidence_blocks_in_batch_succeeds(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    run = build_run(project=project)
    evidence_one = build_evidence(extraction_run=run)
    evidence_two = build_evidence(extraction_run=run)
    inference_context = build_inference_run_context(
        project_id=project.id,
        extraction_run_ids=[run.id],
        source_block_ids=[evidence_one.block.id, evidence_two.block.id],
    )
    payload = build_ai_payload([evidence_one.id, evidence_two.id])
    captured: dict[str, object] = {}

    async def fake_get_extraction_run_with_project(_session, _extraction_run_id):
        return run

    async def fake_get_completed_fact_extraction_run_context(_session, *, inference_run_id):
        assert inference_run_id == inference_context.run_id
        return inference_context

    async def fake_list_source_evidences_with_context(_session, _evidence_ids):
        return [evidence_one, evidence_two]

    async def fake_get_fact_by_identity_for_update(_session, **_kwargs):
        return None

    async def fake_create_fact(_session, fact):
        captured["fact"] = fact
        return fact

    async def fake_get_next_fact_version_no(_session, _fact_id):
        return 1

    async def fake_create_fact_value(_session, fact_value):
        captured["fact_value"] = fact_value
        return fact_value

    async def fake_create_fact_evidence_links(_session, links):
        captured["links"] = links
        return links

    monkeypatch.setattr(fact_service.fact_repository, "get_extraction_run_with_project", fake_get_extraction_run_with_project)
    monkeypatch.setattr(
        fact_service.inference_repository,
        "get_completed_fact_extraction_run_context",
        fake_get_completed_fact_extraction_run_context,
    )
    monkeypatch.setattr(fact_service.fact_repository, "list_source_evidences_with_context", fake_list_source_evidences_with_context)
    monkeypatch.setattr(fact_service.fact_repository, "get_fact_by_identity_for_update", fake_get_fact_by_identity_for_update)
    monkeypatch.setattr(fact_service.fact_repository, "create_fact", fake_create_fact)
    monkeypatch.setattr(fact_service.fact_repository, "get_next_fact_version_no", fake_get_next_fact_version_no)
    monkeypatch.setattr(fact_service.fact_repository, "create_fact_value", fake_create_fact_value)
    monkeypatch.setattr(fact_service.fact_repository, "create_fact_evidence_links", fake_create_fact_evidence_links)

    result = run_async(
        fact_service.propose_ai_fact_value(
            session,
            project_id=project.id,
            extraction_run_id=run.id,
            inference_run_id=inference_context.run_id,
            payload=payload,
        )
    )

    assert result.status == "proposed"
    assert session.commit_count == 1
    assert session.rollback_count == 0


def test_source_order_follows_input_list_order(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    run = build_run(project=project)
    evidence_one = build_evidence(extraction_run=run)
    evidence_two = build_evidence(extraction_run=run)
    inference_context = build_inference_run_context(
        project_id=project.id,
        extraction_run_ids=[run.id],
        source_block_ids=[evidence_one.block.id, evidence_two.block.id],
    )
    payload = AIProposalInput(
        identity=build_identity(),
        value=build_value_input(),
        evidences=[
            FactEvidenceInput(evidence_id=evidence_two.id, role="supporting", is_primary=True),
            FactEvidenceInput(evidence_id=evidence_one.id, role="context", is_primary=False),
        ],
    )
    captured: dict[str, object] = {}

    async def fake_get_extraction_run_with_project(_session, _extraction_run_id):
        return run

    async def fake_get_completed_fact_extraction_run_context(_session, *, inference_run_id):
        assert inference_run_id == inference_context.run_id
        return inference_context

    async def fake_list_source_evidences_with_context(_session, _evidence_ids):
        return [evidence_one, evidence_two]

    async def fake_get_fact_by_identity_for_update(_session, **_kwargs):
        return None

    async def fake_create_fact(_session, fact):
        return fact

    async def fake_get_next_fact_version_no(_session, _fact_id):
        return 1

    async def fake_create_fact_value(_session, fact_value):
        return fact_value

    async def fake_create_fact_evidence_links(_session, links):
        captured["links"] = links
        return links

    monkeypatch.setattr(fact_service.fact_repository, "get_extraction_run_with_project", fake_get_extraction_run_with_project)
    monkeypatch.setattr(
        fact_service.inference_repository,
        "get_completed_fact_extraction_run_context",
        fake_get_completed_fact_extraction_run_context,
    )
    monkeypatch.setattr(fact_service.fact_repository, "list_source_evidences_with_context", fake_list_source_evidences_with_context)
    monkeypatch.setattr(fact_service.fact_repository, "get_fact_by_identity_for_update", fake_get_fact_by_identity_for_update)
    monkeypatch.setattr(fact_service.fact_repository, "create_fact", fake_create_fact)
    monkeypatch.setattr(fact_service.fact_repository, "get_next_fact_version_no", fake_get_next_fact_version_no)
    monkeypatch.setattr(fact_service.fact_repository, "create_fact_value", fake_create_fact_value)
    monkeypatch.setattr(fact_service.fact_repository, "create_fact_evidence_links", fake_create_fact_evidence_links)

    run_async(
        fact_service.propose_ai_fact_value(
            session,
            project_id=project.id,
            extraction_run_id=run.id,
            inference_run_id=inference_context.run_id,
            payload=payload,
        )
    )

    assert [link.evidence_id for link in captured["links"]] == [evidence_two.id, evidence_one.id]
    assert [link.source_order for link in captured["links"]] == [0, 1]


def test_concurrent_fact_creation_returns_existing_fact(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    run = build_run(project=project)
    evidence = build_evidence(extraction_run=run)
    inference_context = build_inference_run_context(
        project_id=project.id,
        extraction_run_ids=[run.id],
        source_block_ids=[evidence.block.id],
    )
    payload = build_ai_payload([evidence.id])
    existing_fact = build_fact(
        project_id=project.id,
        identity_hash=fact_service.build_fact_identity_hash(payload.identity),
    )
    lookup_count = {"count": 0}

    async def fake_get_extraction_run_with_project(_session, _extraction_run_id):
        return run

    async def fake_get_completed_fact_extraction_run_context(_session, *, inference_run_id):
        assert inference_run_id == inference_context.run_id
        return inference_context

    async def fake_list_source_evidences_with_context(_session, _evidence_ids):
        return [evidence]

    async def fake_get_fact_by_identity_for_update(_session, **_kwargs):
        lookup_count["count"] += 1
        if lookup_count["count"] == 1:
            return None
        return existing_fact

    class FakeDiag:
        def __init__(self, constraint_name: str | None) -> None:
            self.constraint_name = constraint_name

    class FakeOrigError(Exception):
        def __init__(self, *, constraint_name: str | None) -> None:
            self.diag = FakeDiag(constraint_name)

    async def fake_create_fact(_session, _fact):
        raise IntegrityError(
            "insert",
            {},
            FakeOrigError(constraint_name="uq_facts_project_id_identity_hash"),
        )

    async def fake_get_next_fact_version_no(_session, _fact_id):
        return 4

    async def fake_create_fact_value(_session, fact_value):
        return fact_value

    async def fake_create_fact_evidence_links(_session, links):
        return links

    monkeypatch.setattr(fact_service.fact_repository, "get_extraction_run_with_project", fake_get_extraction_run_with_project)
    monkeypatch.setattr(
        fact_service.inference_repository,
        "get_completed_fact_extraction_run_context",
        fake_get_completed_fact_extraction_run_context,
    )
    monkeypatch.setattr(fact_service.fact_repository, "list_source_evidences_with_context", fake_list_source_evidences_with_context)
    monkeypatch.setattr(fact_service.fact_repository, "get_fact_by_identity_for_update", fake_get_fact_by_identity_for_update)
    monkeypatch.setattr(fact_service.fact_repository, "create_fact", fake_create_fact)
    monkeypatch.setattr(fact_service.fact_repository, "get_next_fact_version_no", fake_get_next_fact_version_no)
    monkeypatch.setattr(fact_service.fact_repository, "create_fact_value", fake_create_fact_value)
    monkeypatch.setattr(fact_service.fact_repository, "create_fact_evidence_links", fake_create_fact_evidence_links)

    result = run_async(
        fact_service.propose_ai_fact_value(
            session,
            project_id=project.id,
            extraction_run_id=run.id,
            inference_run_id=inference_context.run_id,
            payload=payload,
        )
    )

    assert result.fact_id == existing_fact.id
    assert session.rollback_called is False
    assert session.savepoints[0].rollback_called is True


def test_version_number_is_computed_after_fact_lock(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    run = build_run(project=project)
    evidence = build_evidence(extraction_run=run)
    inference_context = build_inference_run_context(
        project_id=project.id,
        extraction_run_ids=[run.id],
        source_block_ids=[evidence.block.id],
    )
    payload = build_ai_payload([evidence.id])
    fact = build_fact(
        project_id=project.id,
        identity_hash=fact_service.build_fact_identity_hash(payload.identity),
    )
    call_order: list[str] = []

    async def fake_get_extraction_run_with_project(_session, _extraction_run_id):
        return run

    async def fake_get_completed_fact_extraction_run_context(_session, *, inference_run_id):
        assert inference_run_id == inference_context.run_id
        return inference_context

    async def fake_list_source_evidences_with_context(_session, _evidence_ids):
        return [evidence]

    async def fake_get_fact_by_identity_for_update(_session, **_kwargs):
        call_order.append("lock")
        return fact

    async def fake_get_next_fact_version_no(_session, _fact_id):
        call_order.append("next_version")
        return 7

    async def fake_create_fact_value(_session, fact_value):
        return fact_value

    async def fake_create_fact_evidence_links(_session, links):
        return links

    monkeypatch.setattr(fact_service.fact_repository, "get_extraction_run_with_project", fake_get_extraction_run_with_project)
    monkeypatch.setattr(
        fact_service.inference_repository,
        "get_completed_fact_extraction_run_context",
        fake_get_completed_fact_extraction_run_context,
    )
    monkeypatch.setattr(fact_service.fact_repository, "list_source_evidences_with_context", fake_list_source_evidences_with_context)
    monkeypatch.setattr(fact_service.fact_repository, "get_fact_by_identity_for_update", fake_get_fact_by_identity_for_update)
    monkeypatch.setattr(fact_service.fact_repository, "get_next_fact_version_no", fake_get_next_fact_version_no)
    monkeypatch.setattr(fact_service.fact_repository, "create_fact_value", fake_create_fact_value)
    monkeypatch.setattr(fact_service.fact_repository, "create_fact_evidence_links", fake_create_fact_evidence_links)

    result = run_async(
        fact_service.propose_ai_fact_value(
            session,
            project_id=project.id,
            extraction_run_id=run.id,
            inference_run_id=inference_context.run_id,
            payload=payload,
        )
    )

    assert result.version_no == 7
    assert call_order == ["lock", "next_version"]


def test_failure_rolls_back_entire_transaction(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    run = build_run(project=project)
    evidence = build_evidence(extraction_run=run)
    inference_context = build_inference_run_context(
        project_id=project.id,
        extraction_run_ids=[run.id],
        source_block_ids=[evidence.block.id],
    )
    payload = build_ai_payload([evidence.id])

    async def fake_get_extraction_run_with_project(_session, _extraction_run_id):
        return run

    async def fake_get_completed_fact_extraction_run_context(_session, *, inference_run_id):
        assert inference_run_id == inference_context.run_id
        return inference_context

    async def fake_list_source_evidences_with_context(_session, _evidence_ids):
        return [evidence]

    async def fake_get_fact_by_identity_for_update(_session, **_kwargs):
        return None

    async def fake_create_fact(_session, fact):
        return fact

    async def fake_get_next_fact_version_no(_session, _fact_id):
        return 1

    async def fake_create_fact_value(_session, fact_value):
        return fact_value

    async def fake_create_fact_evidence_links(_session, _links):
        raise RuntimeError("link insert failed")

    monkeypatch.setattr(fact_service.fact_repository, "get_extraction_run_with_project", fake_get_extraction_run_with_project)
    monkeypatch.setattr(
        fact_service.inference_repository,
        "get_completed_fact_extraction_run_context",
        fake_get_completed_fact_extraction_run_context,
    )
    monkeypatch.setattr(fact_service.fact_repository, "list_source_evidences_with_context", fake_list_source_evidences_with_context)
    monkeypatch.setattr(fact_service.fact_repository, "get_fact_by_identity_for_update", fake_get_fact_by_identity_for_update)
    monkeypatch.setattr(fact_service.fact_repository, "create_fact", fake_create_fact)
    monkeypatch.setattr(fact_service.fact_repository, "get_next_fact_version_no", fake_get_next_fact_version_no)
    monkeypatch.setattr(fact_service.fact_repository, "create_fact_value", fake_create_fact_value)
    monkeypatch.setattr(fact_service.fact_repository, "create_fact_evidence_links", fake_create_fact_evidence_links)

    with pytest.raises(RuntimeError):
        run_async(
            fact_service.propose_ai_fact_value(
                session,
                project_id=project.id,
                extraction_run_id=run.id,
                inference_run_id=inference_context.run_id,
                payload=payload,
            )
        )

    assert session.rollback_called is True
    assert session.commit_called is False


def test_cancelled_error_rolls_back_and_propagates(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    run = build_run(project=project)
    evidence = build_evidence(extraction_run=run)
    inference_context = build_inference_run_context(
        project_id=project.id,
        extraction_run_ids=[run.id],
        source_block_ids=[evidence.block.id],
    )
    payload = build_ai_payload([evidence.id])

    async def fake_get_extraction_run_with_project(_session, _extraction_run_id):
        return run

    async def fake_get_completed_fact_extraction_run_context(_session, *, inference_run_id):
        assert inference_run_id == inference_context.run_id
        return inference_context

    async def fake_list_source_evidences_with_context(_session, _evidence_ids):
        return [evidence]

    async def fake_get_fact_by_identity_for_update(_session, **_kwargs):
        return None

    async def fake_create_fact(_session, fact):
        return fact

    async def fake_get_next_fact_version_no(_session, _fact_id):
        raise asyncio.CancelledError()

    monkeypatch.setattr(fact_service.fact_repository, "get_extraction_run_with_project", fake_get_extraction_run_with_project)
    monkeypatch.setattr(
        fact_service.inference_repository,
        "get_completed_fact_extraction_run_context",
        fake_get_completed_fact_extraction_run_context,
    )
    monkeypatch.setattr(fact_service.fact_repository, "list_source_evidences_with_context", fake_list_source_evidences_with_context)
    monkeypatch.setattr(fact_service.fact_repository, "get_fact_by_identity_for_update", fake_get_fact_by_identity_for_update)
    monkeypatch.setattr(fact_service.fact_repository, "create_fact", fake_create_fact)
    monkeypatch.setattr(fact_service.fact_repository, "get_next_fact_version_no", fake_get_next_fact_version_no)

    with pytest.raises(asyncio.CancelledError):
        run_async(
            fact_service.propose_ai_fact_value(
                session,
                project_id=project.id,
                extraction_run_id=run.id,
                inference_run_id=inference_context.run_id,
                payload=payload,
            )
        )

    assert session.rollback_count == 1
    assert session.commit_count == 0


@pytest.mark.parametrize("role", ["owner", "editor"])
def test_owner_and_editor_can_create_human_fact_values(monkeypatch, role: str) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    payload = build_human_payload()

    patch_actor_permission(monkeypatch, project_id=project.id, actor=actor, role=role)
    captured = patch_create_human_dependencies(monkeypatch)

    result = run_async(
        fact_service.create_human_fact_value(
            session,
            project_id=project.id,
            actor_id=actor.id,
            payload=payload,
        )
    )

    assert result.status == "accepted"
    assert result.source_kind == "human"
    assert result.created_by_id == actor.id
    assert result.decided_by_id == actor.id
    assert result.extraction_run_id is None
    assert captured["updated_fact"].current_value_id == result.id


def test_viewer_cannot_create_human_fact_values(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()

    patch_actor_permission(monkeypatch, project_id=project.id, actor=actor, role="viewer")

    with pytest.raises(fact_service.FactPermissionError):
        run_async(
            fact_service.create_human_fact_value(
                session,
                project_id=project.id,
                actor_id=actor.id,
                payload=build_human_payload(),
            )
        )


def test_human_value_can_be_current_without_evidence(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()

    patch_actor_permission(monkeypatch, project_id=project.id, actor=actor, role="owner")
    captured = patch_create_human_dependencies(monkeypatch)

    result = run_async(
        fact_service.create_human_fact_value(
            session,
            project_id=project.id,
            actor_id=actor.id,
            payload=build_human_payload(evidence_ids=[]),
        )
    )

    assert result.status == "accepted"
    assert result.source_kind == "human"
    assert result.evidence_links == []
    assert captured.get("links") is None


def test_create_human_fact_value_rejects_cross_project_evidence(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    other_project = build_project()
    actor = build_user()
    evidence = build_evidence(extraction_run=build_run(project=other_project))

    patch_actor_permission(monkeypatch, project_id=project.id, actor=actor, role="owner")
    patch_create_human_dependencies(monkeypatch, evidences=[evidence])

    with pytest.raises(fact_service.InvalidFactProposalError):
        run_async(
            fact_service.create_human_fact_value(
                session,
                project_id=project.id,
                actor_id=actor.id,
                payload=build_human_payload(evidence_ids=[evidence.id]),
            )
        )


def test_create_human_fact_value_supersedes_old_current_without_overwriting_decision_info(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    old_decider = uuid.uuid4()
    old_decided_at = datetime.now(timezone.utc)
    current_value = FactValue(
        id=uuid.uuid4(),
        fact_id=uuid.uuid4(),
        version_no=1,
        value_type="string",
        value_json="Old",
        normalized_value_text="Old",
        value_hash="a" * 64,
        language_code=None,
        status="accepted",
        source_kind="human",
        extraction_run_id=None,
        confidence=None,
        created_by_id=actor.id,
        decided_by_id=old_decider,
        decided_at=old_decided_at,
    )
    fact = build_fact(
        project_id=project.id,
        identity_hash=fact_service.build_fact_identity_hash(build_identity()),
        current_value_id=current_value.id,
    )
    current_value.fact_id = fact.id

    patch_actor_permission(monkeypatch, project_id=project.id, actor=actor, role="owner")
    captured = patch_create_human_dependencies(
        monkeypatch,
        fact=fact,
        current_value=current_value,
        next_version=2,
    )

    result = run_async(
        fact_service.create_human_fact_value(
            session,
            project_id=project.id,
            actor_id=actor.id,
            payload=build_human_payload(),
        )
    )

    assert current_value.status == "superseded"
    assert current_value.decided_by_id == old_decider
    assert current_value.decided_at == old_decided_at
    assert fact.current_value_id == result.id
    assert captured["updated_values"][0] is current_value


def test_accept_fact_value_without_current_sets_current(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    fact = build_fact(
        project_id=project.id,
        identity_hash="a" * 64,
        current_value_id=None,
    )
    target_value = FactValue(
        id=uuid.uuid4(),
        fact_id=fact.id,
        version_no=2,
        value_type="string",
        value_json="AI",
        normalized_value_text="AI",
        value_hash="b" * 64,
        language_code=None,
        status="proposed",
        source_kind="ai",
        extraction_run_id=uuid.uuid4(),
        confidence=0.7,
        created_by_id=None,
        decided_by_id=None,
        decided_at=None,
    )

    patch_actor_permission(monkeypatch, project_id=project.id, actor=actor, role="editor")
    patch_decision_dependencies(monkeypatch, fact=fact, target_value=target_value)

    result = run_async(
        fact_service.accept_fact_value(
            session,
            project_id=project.id,
            actor_id=actor.id,
            fact_value_id=target_value.id,
            command=FactValueDecisionInput(replace_current=False),
        )
    )

    assert result.status == "accepted"
    assert result.decided_by_id == actor.id
    assert fact.current_value_id == target_value.id


def test_accept_fact_value_requires_explicit_replace_when_current_exists(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    current_value = FactValue(
        id=uuid.uuid4(),
        fact_id=uuid.uuid4(),
        version_no=1,
        value_type="string",
        value_json="Current",
        normalized_value_text="Current",
        value_hash="c" * 64,
        language_code=None,
        status="accepted",
        source_kind="human",
        extraction_run_id=None,
        confidence=None,
        created_by_id=actor.id,
        decided_by_id=actor.id,
        decided_at=datetime.now(timezone.utc),
    )
    fact = build_fact(project_id=project.id, identity_hash="d" * 64, current_value_id=current_value.id)
    current_value.fact_id = fact.id
    target_value = FactValue(
        id=uuid.uuid4(),
        fact_id=fact.id,
        version_no=2,
        value_type="string",
        value_json="Proposal",
        normalized_value_text="Proposal",
        value_hash="e" * 64,
        language_code=None,
        status="proposed",
        source_kind="ai",
        extraction_run_id=uuid.uuid4(),
        confidence=0.7,
        created_by_id=None,
        decided_by_id=None,
        decided_at=None,
    )

    patch_actor_permission(monkeypatch, project_id=project.id, actor=actor, role="owner")
    patch_decision_dependencies(
        monkeypatch,
        fact=fact,
        target_value=target_value,
        current_value=current_value,
    )

    with pytest.raises(fact_service.InvalidFactProposalError):
        run_async(
            fact_service.accept_fact_value(
                session,
                project_id=project.id,
                actor_id=actor.id,
                fact_value_id=target_value.id,
                command=FactValueDecisionInput(replace_current=False),
            )
        )


def test_accept_fact_value_replace_true_switches_current_and_supersedes_old(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    old_decider = uuid.uuid4()
    old_decided_at = datetime.now(timezone.utc)
    current_value = FactValue(
        id=uuid.uuid4(),
        fact_id=uuid.uuid4(),
        version_no=1,
        value_type="string",
        value_json="Human Current",
        normalized_value_text="Human Current",
        value_hash="f" * 64,
        language_code=None,
        status="accepted",
        source_kind="human",
        extraction_run_id=None,
        confidence=None,
        created_by_id=actor.id,
        decided_by_id=old_decider,
        decided_at=old_decided_at,
    )
    fact = build_fact(project_id=project.id, identity_hash="1" * 64, current_value_id=current_value.id)
    current_value.fact_id = fact.id
    target_value = FactValue(
        id=uuid.uuid4(),
        fact_id=fact.id,
        version_no=2,
        value_type="string",
        value_json="AI Proposal",
        normalized_value_text="AI Proposal",
        value_hash="2" * 64,
        language_code=None,
        status="proposed",
        source_kind="ai",
        extraction_run_id=uuid.uuid4(),
        confidence=0.8,
        created_by_id=None,
        decided_by_id=None,
        decided_at=None,
    )

    patch_actor_permission(monkeypatch, project_id=project.id, actor=actor, role="owner")
    captured = patch_decision_dependencies(
        monkeypatch,
        fact=fact,
        target_value=target_value,
        current_value=current_value,
    )

    result = run_async(
        fact_service.accept_fact_value(
            session,
            project_id=project.id,
            actor_id=actor.id,
            fact_value_id=target_value.id,
            command=FactValueDecisionInput(replace_current=True),
        )
    )

    assert result.status == "accepted"
    assert fact.current_value_id == target_value.id
    assert current_value.status == "superseded"
    assert current_value.decided_by_id == old_decider
    assert current_value.decided_at == old_decided_at
    assert captured["updated_values"][0] is current_value


def test_reject_fact_value_does_not_change_current(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    current_id = uuid.uuid4()
    fact = build_fact(project_id=project.id, identity_hash="3" * 64, current_value_id=current_id)
    target_value = FactValue(
        id=uuid.uuid4(),
        fact_id=fact.id,
        version_no=2,
        value_type="string",
        value_json="AI Proposal",
        normalized_value_text="AI Proposal",
        value_hash="4" * 64,
        language_code=None,
        status="proposed",
        source_kind="ai",
        extraction_run_id=uuid.uuid4(),
        confidence=0.6,
        created_by_id=None,
        decided_by_id=None,
        decided_at=None,
    )

    patch_actor_permission(monkeypatch, project_id=project.id, actor=actor, role="editor")
    patch_decision_dependencies(monkeypatch, fact=fact, target_value=target_value)

    result = run_async(
        fact_service.reject_fact_value(
            session,
            project_id=project.id,
            actor_id=actor.id,
            fact_value_id=target_value.id,
        )
    )

    assert result.status == "rejected"
    assert fact.current_value_id == current_id


def test_illegal_state_transitions_are_rejected(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    fact = build_fact(project_id=project.id, identity_hash="5" * 64)
    accepted_value = FactValue(
        id=uuid.uuid4(),
        fact_id=fact.id,
        version_no=1,
        value_type="string",
        value_json="Accepted",
        normalized_value_text="Accepted",
        value_hash="6" * 64,
        language_code=None,
        status="accepted",
        source_kind="human",
        extraction_run_id=None,
        confidence=None,
        created_by_id=actor.id,
        decided_by_id=actor.id,
        decided_at=datetime.now(timezone.utc),
    )

    patch_actor_permission(monkeypatch, project_id=project.id, actor=actor, role="owner")
    patch_decision_dependencies(monkeypatch, fact=fact, target_value=accepted_value)

    with pytest.raises(fact_service.InvalidFactProposalError):
        run_async(
            fact_service.reject_fact_value(
                session,
                project_id=project.id,
                actor_id=actor.id,
                fact_value_id=accepted_value.id,
            )
        )


def test_repeated_accept_and_reject_are_idempotent(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    accepted_current = FactValue(
        id=uuid.uuid4(),
        fact_id=uuid.uuid4(),
        version_no=1,
        value_type="string",
        value_json="Current",
        normalized_value_text="Current",
        value_hash="7" * 64,
        language_code=None,
        status="accepted",
        source_kind="human",
        extraction_run_id=None,
        confidence=None,
        created_by_id=actor.id,
        decided_by_id=actor.id,
        decided_at=datetime.now(timezone.utc),
    )
    fact = build_fact(project_id=project.id, identity_hash="8" * 64, current_value_id=accepted_current.id)
    accepted_current.fact_id = fact.id

    rejected_value = FactValue(
        id=uuid.uuid4(),
        fact_id=fact.id,
        version_no=2,
        value_type="string",
        value_json="Rejected",
        normalized_value_text="Rejected",
        value_hash="9" * 64,
        language_code=None,
        status="rejected",
        source_kind="ai",
        extraction_run_id=uuid.uuid4(),
        confidence=0.3,
        created_by_id=None,
        decided_by_id=actor.id,
        decided_at=datetime.now(timezone.utc),
    )

    patch_actor_permission(monkeypatch, project_id=project.id, actor=actor, role="owner")
    patch_decision_dependencies(
        monkeypatch,
        fact=fact,
        target_value=accepted_current,
        current_value=accepted_current,
    )

    accepted_result = run_async(
        fact_service.accept_fact_value(
            session,
            project_id=project.id,
            actor_id=actor.id,
            fact_value_id=accepted_current.id,
            command=FactValueDecisionInput(replace_current=False),
        )
    )
    assert accepted_result is accepted_current

    patch_decision_dependencies(monkeypatch, fact=fact, target_value=rejected_value)
    rejected_result = run_async(
        fact_service.reject_fact_value(
            session,
            project_id=project.id,
            actor_id=actor.id,
            fact_value_id=rejected_value.id,
        )
    )
    assert rejected_result is rejected_value


def test_retired_fact_rejects_human_write(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    fact = build_fact(
        project_id=project.id,
        identity_hash=fact_service.build_fact_identity_hash(build_identity()),
        status="retired",
    )

    patch_actor_permission(monkeypatch, project_id=project.id, actor=actor, role="owner")
    patch_create_human_dependencies(monkeypatch, fact=fact)

    with pytest.raises(fact_service.RetiredFactError):
        run_async(
            fact_service.create_human_fact_value(
                session,
                project_id=project.id,
                actor_id=actor.id,
                payload=build_human_payload(),
            )
        )


def test_current_value_must_belong_to_same_fact_and_be_accepted(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    current_value = FactValue(
        id=uuid.uuid4(),
        fact_id=uuid.uuid4(),
        version_no=1,
        value_type="string",
        value_json="Broken Current",
        normalized_value_text="Broken Current",
        value_hash="a1" * 32,
        language_code=None,
        status="rejected",
        source_kind="human",
        extraction_run_id=None,
        confidence=None,
        created_by_id=actor.id,
        decided_by_id=actor.id,
        decided_at=datetime.now(timezone.utc),
    )
    fact = build_fact(project_id=project.id, identity_hash="b1" * 32, current_value_id=current_value.id)
    target_value = FactValue(
        id=uuid.uuid4(),
        fact_id=fact.id,
        version_no=2,
        value_type="string",
        value_json="Proposal",
        normalized_value_text="Proposal",
        value_hash="c1" * 32,
        language_code=None,
        status="proposed",
        source_kind="ai",
        extraction_run_id=uuid.uuid4(),
        confidence=0.4,
        created_by_id=None,
        decided_by_id=None,
        decided_at=None,
    )

    patch_actor_permission(monkeypatch, project_id=project.id, actor=actor, role="owner")
    patch_decision_dependencies(
        monkeypatch,
        fact=fact,
        target_value=target_value,
        current_value=current_value,
    )

    with pytest.raises(fact_service.InvalidFactProposalError):
        run_async(
            fact_service.accept_fact_value(
                session,
                project_id=project.id,
                actor_id=actor.id,
                fact_value_id=target_value.id,
                command=FactValueDecisionInput(replace_current=True),
            )
        )


def test_create_human_fact_value_rolls_back_on_failure(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    evidence = build_evidence(extraction_run=build_run(project=project))

    patch_actor_permission(monkeypatch, project_id=project.id, actor=actor, role="owner")
    patch_create_human_dependencies(
        monkeypatch,
        evidences=[evidence],
        create_links_error=RuntimeError("evidence link insert failed"),
    )

    with pytest.raises(RuntimeError):
        run_async(
            fact_service.create_human_fact_value(
                session,
                project_id=project.id,
                actor_id=actor.id,
                payload=build_human_payload(evidence_ids=[evidence.id]),
            )
        )

    assert session.rollback_called is True
    assert session.commit_called is False


def test_new_fact_writes_subject_entity_and_referenced_entity(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    run = build_run(project=project)
    evidence = build_evidence(extraction_run=run)
    subject_entity = build_entity_context(project_id=project.id, entity_type="company", canonical_key="acme")
    referenced_entity = build_entity_context(
        project_id=project.id,
        entity_type="company",
        canonical_key="beta llc",
    )
    inference_context = build_inference_run_context(
        project_id=project.id,
        extraction_run_ids=[run.id],
        source_block_ids=[evidence.block.id],
    )
    payload = build_ai_payload(
        [evidence.id],
        identity=build_identity(
            subject_kind="company",
            subject_key=" ACME ",
            subject_entity_id=subject_entity.entity_id,
        ),
        value=build_value_input(
            value_type="entity_ref",
            value_json={"kind": "company", "key": " Beta LLC "},
            referenced_entity_id=referenced_entity.entity_id,
            language_code=None,
            confidence=None,
        ),
    )
    captured: dict[str, object] = {}

    async def fake_get_extraction_run_with_project(_session, _extraction_run_id):
        return run

    async def fake_get_completed_fact_extraction_run_context(_session, *, inference_run_id):
        assert inference_run_id == inference_context.run_id
        return inference_context

    async def fake_list_source_evidences_with_context(_session, _evidence_ids):
        return [evidence]

    async def fake_get_fact_entity_context(_session, *, entity_id):
        if entity_id == subject_entity.entity_id:
            return subject_entity
        if entity_id == referenced_entity.entity_id:
            return referenced_entity
        return None

    async def fake_get_fact_by_identity_for_update(_session, **_kwargs):
        return None

    async def fake_create_fact(_session, fact):
        captured["fact"] = fact
        return fact

    async def fake_get_next_fact_version_no(_session, _fact_id):
        return 1

    async def fake_create_fact_value(_session, fact_value):
        captured["fact_value"] = fact_value
        return fact_value

    async def fake_create_fact_evidence_links(_session, links):
        return links

    monkeypatch.setattr(fact_service.fact_repository, "get_extraction_run_with_project", fake_get_extraction_run_with_project)
    monkeypatch.setattr(
        fact_service.inference_repository,
        "get_completed_fact_extraction_run_context",
        fake_get_completed_fact_extraction_run_context,
    )
    monkeypatch.setattr(fact_service.fact_repository, "list_source_evidences_with_context", fake_list_source_evidences_with_context)
    monkeypatch.setattr(fact_service.entity_repository, "get_fact_entity_context", fake_get_fact_entity_context)
    monkeypatch.setattr(fact_service.fact_repository, "get_fact_by_identity_for_update", fake_get_fact_by_identity_for_update)
    monkeypatch.setattr(fact_service.fact_repository, "create_fact", fake_create_fact)
    monkeypatch.setattr(fact_service.fact_repository, "get_next_fact_version_no", fake_get_next_fact_version_no)
    monkeypatch.setattr(fact_service.fact_repository, "create_fact_value", fake_create_fact_value)
    monkeypatch.setattr(fact_service.fact_repository, "create_fact_evidence_links", fake_create_fact_evidence_links)

    result = run_async(
        fact_service.propose_ai_fact_value(
            session,
            project_id=project.id,
            extraction_run_id=run.id,
            inference_run_id=inference_context.run_id,
            payload=payload,
        )
    )

    created_fact = captured["fact"]
    created_value = captured["fact_value"]
    assert created_fact.subject_kind == "company"
    assert created_fact.subject_key == "acme"
    assert created_fact.subject_entity_id == subject_entity.entity_id
    assert created_value.referenced_entity_id == referenced_entity.entity_id
    assert result.referenced_entity_id == referenced_entity.entity_id


def test_cross_project_subject_entity_is_rejected(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    other_project = build_project()
    actor = build_user()

    patch_actor_permission(monkeypatch, project_id=project.id, actor=actor, role="owner")

    async def fake_get_fact_entity_context(_session, *, entity_id):
        return build_entity_context(
            entity_id=entity_id,
            project_id=other_project.id,
            entity_type="company",
            canonical_key="acme",
        )

    monkeypatch.setattr(fact_service.entity_repository, "get_fact_entity_context", fake_get_fact_entity_context)

    with pytest.raises(fact_service.FactSubjectEntityNotEligibleError):
        run_async(
            fact_service.create_human_fact_value(
                session,
                project_id=project.id,
                actor_id=actor.id,
                payload=build_human_payload(
                    identity=build_identity(
                        subject_kind="company",
                        subject_key="acme",
                        subject_entity_id=uuid.uuid4(),
                    )
                ),
            )
        )

    assert session.rollback_count == 1


@pytest.mark.parametrize("status", ["merged", "archived"])
def test_inactive_subject_entity_is_rejected(monkeypatch, status: str) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    entity_id = uuid.uuid4()

    patch_actor_permission(monkeypatch, project_id=project.id, actor=actor, role="owner")

    async def fake_get_fact_entity_context(_session, *, entity_id):
        return build_entity_context(
            entity_id=entity_id,
            project_id=project.id,
            entity_type="company",
            canonical_key="acme",
            status=status,
        )

    monkeypatch.setattr(fact_service.entity_repository, "get_fact_entity_context", fake_get_fact_entity_context)

    with pytest.raises(fact_service.FactSubjectEntityNotEligibleError):
        run_async(
            fact_service.create_human_fact_value(
                session,
                project_id=project.id,
                actor_id=actor.id,
                payload=build_human_payload(
                    identity=build_identity(
                        subject_kind="company",
                        subject_key="acme",
                        subject_entity_id=entity_id,
                    )
                ),
            )
        )


def test_subject_entity_kind_and_key_mismatch_is_rejected(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    entity = build_entity_context(project_id=project.id, entity_type="company", canonical_key="acme")

    patch_actor_permission(monkeypatch, project_id=project.id, actor=actor, role="owner")
    monkeypatch.setattr(
        fact_service.entity_repository,
        "get_fact_entity_context",
        lambda _session, *, entity_id: asyncio.sleep(0, result=entity),
    )

    with pytest.raises(fact_service.FactSubjectEntityMismatchError):
        run_async(
            fact_service.create_human_fact_value(
                session,
                project_id=project.id,
                actor_id=actor.id,
                payload=build_human_payload(
                    identity=build_identity(
                        subject_kind="person",
                        subject_key="acme",
                        subject_entity_id=entity.entity_id,
                    )
                ),
            )
        )

    with pytest.raises(fact_service.FactSubjectEntityMismatchError):
        run_async(
            fact_service.create_human_fact_value(
                session,
                project_id=project.id,
                actor_id=actor.id,
                payload=build_human_payload(
                    identity=build_identity(
                        subject_kind="company",
                        subject_key="other",
                        subject_entity_id=entity.entity_id,
                    )
                ),
            )
        )


def test_existing_fact_without_subject_entity_is_upgraded_in_place(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    entity = build_entity_context(project_id=project.id, entity_type="company", canonical_key="acme")
    identity = build_identity(subject_kind="company", subject_key="ACME", subject_entity_id=entity.entity_id)
    identity_hash = fact_service.build_fact_identity_hash(
        FactIdentityInput(
            subject_kind="company",
            subject_key="acme",
            subject_entity_id=entity.entity_id,
            predicate_key="legal_name",
            scope_key=None,
        )
    )
    existing_fact = build_fact(
        project_id=project.id,
        identity_hash=identity_hash,
        subject_kind="company",
        subject_key="acme",
        subject_entity_id=None,
    )
    create_calls = {"count": 0}
    updated: dict[str, object] = {}

    patch_actor_permission(monkeypatch, project_id=project.id, actor=actor, role="owner")
    monkeypatch.setattr(
        fact_service.entity_repository,
        "get_fact_entity_context",
        lambda _session, *, entity_id: asyncio.sleep(0, result=entity),
    )

    async def fake_get_fact_by_identity_for_update(_session, **_kwargs):
        return existing_fact

    async def fake_create_fact(_session, fact):
        create_calls["count"] += 1
        return fact

    async def fake_update_fact(_session, fact):
        updated["fact"] = fact
        return fact

    monkeypatch.setattr(fact_service.fact_repository, "get_fact_by_identity_for_update", fake_get_fact_by_identity_for_update)
    monkeypatch.setattr(fact_service.fact_repository, "create_fact", fake_create_fact)
    monkeypatch.setattr(fact_service.fact_repository, "update_fact", fake_update_fact)
    monkeypatch.setattr(fact_service.fact_repository, "get_next_fact_version_no", lambda _s, _id: asyncio.sleep(0, result=1))
    monkeypatch.setattr(fact_service.fact_repository, "create_fact_value", lambda _s, fact_value: asyncio.sleep(0, result=fact_value))

    result = run_async(
        fact_service.create_human_fact_value(
            session,
            project_id=project.id,
            actor_id=actor.id,
            payload=build_human_payload(identity=identity),
        )
    )

    assert result.fact_id == existing_fact.id
    assert existing_fact.subject_entity_id == entity.entity_id
    assert updated["fact"] is existing_fact
    assert create_calls["count"] == 0
    assert existing_fact.identity_hash == identity_hash


def test_existing_fact_with_different_subject_entity_is_rejected(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    requested_entity = build_entity_context(project_id=project.id, entity_type="company", canonical_key="acme")
    existing_fact = build_fact(
        project_id=project.id,
        identity_hash=fact_service.build_fact_identity_hash(
            FactIdentityInput(
                subject_kind="company",
                subject_key="acme",
                subject_entity_id=requested_entity.entity_id,
                predicate_key="legal_name",
                scope_key=None,
            )
        ),
        subject_entity_id=uuid.uuid4(),
    )

    patch_actor_permission(monkeypatch, project_id=project.id, actor=actor, role="owner")
    monkeypatch.setattr(
        fact_service.entity_repository,
        "get_fact_entity_context",
        lambda _session, *, entity_id: asyncio.sleep(0, result=requested_entity),
    )
    monkeypatch.setattr(
        fact_service.fact_repository,
        "get_fact_by_identity_for_update",
        lambda _session, **_kwargs: asyncio.sleep(0, result=existing_fact),
    )

    with pytest.raises(fact_service.FactSubjectEntityConflictError):
        run_async(
            fact_service.create_human_fact_value(
                session,
                project_id=project.id,
                actor_id=actor.id,
                payload=build_human_payload(
                    identity=build_identity(
                        subject_kind="company",
                        subject_key="acme",
                        subject_entity_id=requested_entity.entity_id,
                    )
                ),
            )
        )


def test_missing_subject_entity_does_not_clear_existing_binding(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    existing_fact = build_fact(
        project_id=project.id,
        identity_hash=fact_service.build_fact_identity_hash(build_identity()),
        subject_entity_id=uuid.uuid4(),
    )

    patch_actor_permission(monkeypatch, project_id=project.id, actor=actor, role="owner")
    monkeypatch.setattr(
        fact_service.fact_repository,
        "get_fact_by_identity_for_update",
        lambda _session, **_kwargs: asyncio.sleep(0, result=existing_fact),
    )
    monkeypatch.setattr(
        fact_service.fact_repository,
        "update_fact",
        lambda _session, fact: asyncio.sleep(0, result=fact),
    )
    monkeypatch.setattr(fact_service.fact_repository, "get_next_fact_version_no", lambda _s, _id: asyncio.sleep(0, result=1))
    monkeypatch.setattr(fact_service.fact_repository, "create_fact_value", lambda _s, fact_value: asyncio.sleep(0, result=fact_value))

    result = run_async(
        fact_service.create_human_fact_value(
            session,
            project_id=project.id,
            actor_id=actor.id,
            payload=build_human_payload(),
        )
    )

    assert result.fact_id == existing_fact.id
    assert existing_fact.subject_entity_id is not None


def test_cross_project_and_inactive_referenced_entity_are_rejected(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    referenced_entity_id = uuid.uuid4()

    patch_actor_permission(monkeypatch, project_id=project.id, actor=actor, role="owner")

    async def fake_cross_project(_session, *, entity_id):
        return build_entity_context(
            entity_id=entity_id,
            project_id=uuid.uuid4(),
            entity_type="company",
            canonical_key="beta llc",
        )

    monkeypatch.setattr(fact_service.entity_repository, "get_fact_entity_context", fake_cross_project)

    with pytest.raises(fact_service.FactReferencedEntityNotEligibleError):
        run_async(
            fact_service.create_human_fact_value(
                session,
                project_id=project.id,
                actor_id=actor.id,
                payload=build_human_payload(
                    value=build_value_input(
                        value_type="entity_ref",
                        value_json={"kind": "company", "key": "Beta LLC"},
                        referenced_entity_id=referenced_entity_id,
                        language_code=None,
                        confidence=None,
                    )
                ),
            )
        )

    async def fake_archived(_session, *, entity_id):
        return build_entity_context(
            entity_id=entity_id,
            project_id=project.id,
            entity_type="company",
            canonical_key="beta llc",
            status="archived",
        )

    monkeypatch.setattr(fact_service.entity_repository, "get_fact_entity_context", fake_archived)

    with pytest.raises(fact_service.FactReferencedEntityNotEligibleError):
        run_async(
            fact_service.create_human_fact_value(
                session,
                project_id=project.id,
                actor_id=actor.id,
                payload=build_human_payload(
                    value=build_value_input(
                        value_type="entity_ref",
                        value_json={"kind": "company", "key": "Beta LLC"},
                        referenced_entity_id=referenced_entity_id,
                        language_code=None,
                        confidence=None,
                    )
                ),
            )
        )


def test_referenced_entity_kind_and_key_mismatch_are_rejected(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    referenced_entity = build_entity_context(
        project_id=project.id,
        entity_type="company",
        canonical_key="beta llc",
    )

    patch_actor_permission(monkeypatch, project_id=project.id, actor=actor, role="owner")
    monkeypatch.setattr(
        fact_service.entity_repository,
        "get_fact_entity_context",
        lambda _session, *, entity_id: asyncio.sleep(0, result=referenced_entity),
    )

    with pytest.raises(fact_service.FactReferencedEntityMismatchError):
        run_async(
            fact_service.create_human_fact_value(
                session,
                project_id=project.id,
                actor_id=actor.id,
                payload=build_human_payload(
                    value=build_value_input(
                        value_type="entity_ref",
                        value_json={"kind": "person", "key": "Beta LLC"},
                        referenced_entity_id=referenced_entity.entity_id,
                        language_code=None,
                        confidence=None,
                    )
                ),
            )
        )

    with pytest.raises(fact_service.FactReferencedEntityMismatchError):
        run_async(
            fact_service.create_human_fact_value(
                session,
                project_id=project.id,
                actor_id=actor.id,
                payload=build_human_payload(
                    value=build_value_input(
                        value_type="entity_ref",
                        value_json={"kind": "company", "key": "Other"},
                        referenced_entity_id=referenced_entity.entity_id,
                        language_code=None,
                        confidence=None,
                    )
                ),
            )
        )


def test_non_target_integrity_error_with_matching_text_is_not_swallowed(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()

    patch_actor_permission(monkeypatch, project_id=project.id, actor=actor, role="owner")
    monkeypatch.setattr(
        fact_service.fact_repository,
        "get_fact_by_identity_for_update",
        lambda _session, **_kwargs: asyncio.sleep(0, result=None),
    )

    class FakeDiag:
        def __init__(self, constraint_name: str | None) -> None:
            self.constraint_name = constraint_name

    class FakeOrigError(Exception):
        def __init__(self, message: str, *, constraint_name: str | None = None) -> None:
            super().__init__(message)
            if constraint_name is not None:
                self.diag = FakeDiag(constraint_name)

    async def fake_create_fact(_session, _fact):
        raise IntegrityError(
            "insert",
            {},
            FakeOrigError(
                "uq_facts_project_id_identity_hash appears in the text",
                constraint_name="uq_some_other_constraint",
            ),
        )

    monkeypatch.setattr(fact_service.fact_repository, "create_fact", fake_create_fact)

    with pytest.raises(IntegrityError):
        run_async(
            fact_service.create_human_fact_value(
                session,
                project_id=project.id,
                actor_id=actor.id,
                payload=build_human_payload(),
            )
        )


def test_integrity_error_without_diag_is_reraised(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()

    patch_actor_permission(monkeypatch, project_id=project.id, actor=actor, role="owner")
    monkeypatch.setattr(
        fact_service.fact_repository,
        "get_fact_by_identity_for_update",
        lambda _session, **_kwargs: asyncio.sleep(0, result=None),
    )
    monkeypatch.setattr(
        fact_service.fact_repository,
        "create_fact",
        lambda _session, _fact: asyncio.sleep(
            0,
            result=(_ for _ in ()).throw(IntegrityError("insert", {}, Exception("duplicate"))),
        ),
    )

    with pytest.raises(IntegrityError):
        run_async(
            fact_service.create_human_fact_value(
                session,
                project_id=project.id,
                actor_id=actor.id,
                payload=build_human_payload(),
            )
        )


def test_concurrent_requery_with_mismatched_identity_is_rejected(monkeypatch) -> None:
    session = FakeSession()
    project = build_project()
    actor = build_user()
    mismatched_fact = build_fact(
        project_id=project.id,
        identity_hash=fact_service.build_fact_identity_hash(build_identity()),
        subject_kind="person",
        subject_key="acme",
    )
    lookup_count = {"count": 0}

    patch_actor_permission(monkeypatch, project_id=project.id, actor=actor, role="owner")

    async def fake_get_fact_by_identity_for_update(_session, **_kwargs):
        lookup_count["count"] += 1
        if lookup_count["count"] == 1:
            return None
        return mismatched_fact

    class FakeDiag:
        def __init__(self, constraint_name: str | None) -> None:
            self.constraint_name = constraint_name

    class FakeOrigError(Exception):
        def __init__(self, *, constraint_name: str | None) -> None:
            self.diag = FakeDiag(constraint_name)

    async def fake_create_fact(_session, _fact):
        raise IntegrityError("insert", {}, FakeOrigError(constraint_name="uq_facts_project_id_identity_hash"))

    monkeypatch.setattr(fact_service.fact_repository, "get_fact_by_identity_for_update", fake_get_fact_by_identity_for_update)
    monkeypatch.setattr(fact_service.fact_repository, "create_fact", fake_create_fact)

    with pytest.raises(fact_service.FactIdentityConflictError):
        run_async(
            fact_service.create_human_fact_value(
                session,
                project_id=project.id,
                actor_id=actor.id,
                payload=build_human_payload(),
            )
        )
