from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import inspect
from types import MappingProxyType
import uuid

import pytest

from app.repositories.fact_value_duplicate_grouping import DuplicateGroupingOrchestrationState
from app.schemas.fact import FactIdentityInput, FactValueInput
from app.schemas.fact_extraction_persistence import (
    AuthenticatedCompletedFactExtractionApplicationSnapshot,
    AuthenticatedPersistedFactProposalItem,
)
from app.schemas.fact_value_duplicate_grouping import DuplicateCandidate
from app.services import ufl_fact_snapshot as ufl_fact_snapshot_service


class CustomJSONValue:
    pass


def run_async(awaitable):
    return asyncio.run(awaitable)


class FakeSession:
    def __init__(self) -> None:
        self.commit_count = 0
        self.rollback_count = 0

    async def commit(self) -> None:
        self.commit_count += 1

    async def rollback(self) -> None:
        self.rollback_count += 1


class SessionFactory:
    def __init__(self) -> None:
        self.sessions: list[FakeSession] = []

    def __call__(self):
        factory = self

        class _Context:
            async def __aenter__(self_inner):
                session = FakeSession()
                factory.sessions.append(session)
                return session

            async def __aexit__(self_inner, exc_type, exc, tb):
                return False

        return _Context()


def _uuid(seed: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, seed)


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fact_identity_hash(
    *,
    subject_kind: str,
    subject_key: str,
    predicate_key: str,
    scope_key: str | None = None,
    subject_entity_id: uuid.UUID | None = None,
) -> str:
    return ufl_fact_snapshot_service.fact_diff_service.fact_service.build_fact_identity_hash(
        FactIdentityInput(
            subject_kind=subject_kind,
            subject_key=subject_key,
            subject_entity_id=subject_entity_id,
            predicate_key=predicate_key,
            scope_key=scope_key,
        )
    )


def _normalized_fact_value(
    *,
    value_type: str,
    value_json: object | None,
    referenced_entity_id: uuid.UUID | None = None,
):
    return ufl_fact_snapshot_service.fact_diff_service.fact_service.normalize_fact_value_input(
        FactValueInput(
            value_type=value_type,
            value_json=value_json,
            referenced_entity_id=referenced_entity_id,
            language_code=None,
            confidence=None,
        )
    )


def _state(
    *,
    orchestration_id: uuid.UUID,
    extraction_run_id: uuid.UUID,
    project_id: uuid.UUID,
    orchestration_status: str = "completed",
) -> DuplicateGroupingOrchestrationState:
    return DuplicateGroupingOrchestrationState(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        project_id=project_id,
        extraction_run_status="completed",
        extraction_run_outcome="partial" if orchestration_status == "partial" else "success",
        orchestration_status=orchestration_status,
        batch_count=2 if orchestration_status == "completed" else 3,
        completed_batch_count=2,
        failed_batch_count=0 if orchestration_status == "completed" else 1,
        planner_name="planner",
        planner_version="1.0.0",
        agent_name="agent",
        agent_version="1.0.0",
        prompt_contract_hash="a" * 64,
        provider="provider",
        requested_model="model",
        executor_name="executor",
        executor_version="1.0.0",
        persistence_name="persistence",
        persistence_version="1.0.0",
        entity_resolution_policy_name="entity-policy",
        entity_resolution_policy_version="1.0.0",
    )


def _candidate(
    *,
    fact_value_id: uuid.UUID,
    fact_id: uuid.UUID,
    orchestration_id: uuid.UUID,
    extraction_run_id: uuid.UUID,
    source_batch_id: uuid.UUID,
    value_type: str,
    value_json: object | None,
    referenced_entity_id: uuid.UUID | None = None,
    evidence_link_ids: tuple[uuid.UUID, ...],
    evidence_ids: tuple[uuid.UUID, ...],
) -> DuplicateCandidate:
    return DuplicateCandidate(
        fact_value_id=fact_value_id,
        fact_id=fact_id,
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        source_batch_id=source_batch_id,
        value_type=value_type,
        value_json=value_json,
        referenced_entity_id=referenced_entity_id,
        evidence_link_ids=evidence_link_ids,
        evidence_ids=evidence_ids,
    )


def _application_snapshot(
    *,
    application_id: uuid.UUID,
    project_id: uuid.UUID,
    extraction_run_id: uuid.UUID,
    inference_run_id: uuid.UUID,
    input_batch_id: uuid.UUID,
    items: tuple[AuthenticatedPersistedFactProposalItem, ...],
    result_hash: str = "d" * 64,
) -> AuthenticatedCompletedFactExtractionApplicationSnapshot:
    return AuthenticatedCompletedFactExtractionApplicationSnapshot(
        application_id=application_id,
        project_id=project_id,
        extraction_run_id=extraction_run_id,
        inference_run_id=inference_run_id,
        input_batch_id=input_batch_id,
        persistence_name="persistence",
        persistence_version="1.0.0",
        entity_resolution_policy_name="entity-policy",
        entity_resolution_policy_version="1.0.0",
        items=items,
        result_hash=result_hash,
    )


def _row(
    *,
    project_id: uuid.UUID,
    fact_id: uuid.UUID,
    fact_value_id: uuid.UUID,
    orchestration_id: uuid.UUID,
    extraction_run_id: uuid.UUID,
    source_batch_id: uuid.UUID,
    application_id: uuid.UUID,
    inference_run_id: uuid.UUID,
    document_revision_id: uuid.UUID,
    subject_kind: str,
    subject_key: str,
    predicate_key: str,
    scope_key: str | None,
    value_type: str,
    value_json: object | None,
    evidence_link_id: uuid.UUID,
    evidence_id: uuid.UUID,
    document_block_id: uuid.UUID,
    block_source_order: int,
    proposal_index: int,
    referenced_entity_id: uuid.UUID | None = None,
) -> ufl_fact_snapshot_service.ufl_fact_snapshot_repository.DocumentRevisionFactDiffSourceRow:
    normalized_value = _normalized_fact_value(
        value_type=value_type,
        value_json=value_json,
        referenced_entity_id=referenced_entity_id,
    )
    excerpt = f"excerpt-{proposal_index}-{fact_value_id}"
    return (
        ufl_fact_snapshot_service.ufl_fact_snapshot_repository.DocumentRevisionFactDiffSourceRow(
            fact_project_id=project_id,
            fact_id=fact_id,
            fact_identity_hash=_fact_identity_hash(
                subject_kind=subject_kind,
                subject_key=subject_key,
                predicate_key=predicate_key,
                scope_key=scope_key,
            ),
            subject_kind=subject_kind,
            subject_key=subject_key,
            predicate_key=predicate_key,
            scope_key=scope_key,
            subject_entity_id=None,
            fact_value_id=fact_value_id,
            extraction_run_id=extraction_run_id,
            inference_run_id=inference_run_id,
            source_batch_id=source_batch_id,
            application_project_id=project_id,
            application_extraction_run_id=extraction_run_id,
            application_inference_run_id=inference_run_id,
            orchestration_project_id=project_id,
            orchestration_extraction_run_id=extraction_run_id,
            batch_current_inference_run_id=inference_run_id,
            value_type=normalized_value.value_type,
            value_json=normalized_value.value_json,
            normalized_value_text=normalized_value.normalized_value_text,
            fact_value_hash=normalized_value.value_hash,
            referenced_entity_id=normalized_value.referenced_entity_id,
            evidence_link_id=evidence_link_id,
            evidence_link_source_order=proposal_index,
            evidence_id=evidence_id,
            document_block_id=document_block_id,
            evidence_start_offset=0,
            evidence_end_offset=len(excerpt),
            evidence_excerpt=excerpt,
            evidence_excerpt_hash=_hash_text(excerpt),
            block_extraction_run_id=extraction_run_id,
            block_source_order=block_source_order,
            block_location_key=f"loc-{document_block_id}",
            block_page_no=1,
            block_start_line=block_source_order + 1,
            block_end_line=block_source_order + 1,
            block_table_index=None,
            block_row_index=None,
            block_raw_text=excerpt,
            application_id=application_id,
            language_code=None,
            confidence=0.5,
            evidence_role="supporting",
            evidence_is_primary=True,
            document_revision_id=document_revision_id,
        )
    )


def _install_sources(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_snapshot,
    rows,
) -> None:
    async def fake_authenticate(*_args, **_kwargs):
        return source_snapshot

    async def fake_list_rows(_session, *, orchestration_id):
        assert orchestration_id == source_snapshot.state.orchestration_id
        return rows

    monkeypatch.setattr(
        ufl_fact_snapshot_service.duplicate_grouping_service,
        "authenticate_duplicate_grouping_source_snapshot",
        fake_authenticate,
    )
    monkeypatch.setattr(
        ufl_fact_snapshot_service.ufl_fact_snapshot_repository,
        "list_orchestration_ufl_fact_source_rows",
        fake_list_rows,
    )


def _rich_fixture(*, orchestration_status: str = "completed") -> dict[str, object]:
    project_id = _uuid("project")
    orchestration_id = _uuid("orchestration")
    extraction_run_id = _uuid("extraction-run")
    document_revision_id = _uuid("document-revision")
    batch_a = _uuid("batch-a")
    batch_b = _uuid("batch-b")
    app_a = _uuid("app-a")
    app_b = _uuid("app-b")
    inference_a = _uuid("inference-a")
    inference_b = _uuid("inference-b")
    fact_a = _uuid("fact-a")
    fact_b = _uuid("fact-b")
    fv_a1 = _uuid("fv-a1")
    fv_a2 = _uuid("fv-a2")
    fv_b1 = _uuid("fv-b1")
    fv_b2 = _uuid("fv-b2")

    candidate_a1 = _candidate(
        fact_value_id=fv_a1,
        fact_id=fact_a,
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        source_batch_id=batch_a,
        value_type="string",
        value_json="same",
        evidence_link_ids=(_uuid("link-a1"),),
        evidence_ids=(_uuid("evidence-a1"),),
    )
    candidate_a2 = _candidate(
        fact_value_id=fv_a2,
        fact_id=fact_a,
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        source_batch_id=batch_b,
        value_type="string",
        value_json="same",
        evidence_link_ids=(_uuid("link-a2"),),
        evidence_ids=(_uuid("evidence-a2"),),
    )
    candidate_b1 = _candidate(
        fact_value_id=fv_b1,
        fact_id=fact_b,
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        source_batch_id=batch_a,
        value_type="number",
        value_json=10,
        evidence_link_ids=(_uuid("link-b1"),),
        evidence_ids=(_uuid("evidence-b1"),),
    )
    candidate_b2 = _candidate(
        fact_value_id=fv_b2,
        fact_id=fact_b,
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        source_batch_id=batch_b,
        value_type="number",
        value_json=20,
        evidence_link_ids=(_uuid("link-b2"),),
        evidence_ids=(_uuid("evidence-b2"),),
    )
    application_snapshot_a = _application_snapshot(
        application_id=app_a,
        project_id=project_id,
        extraction_run_id=extraction_run_id,
        inference_run_id=inference_a,
        input_batch_id=_uuid("input-a"),
        items=(
            AuthenticatedPersistedFactProposalItem(
                proposal_index=0,
                fact_id=fact_a,
                fact_value_id=fv_a1,
                subject_entity_id=None,
                referenced_entity_id=None,
                evidence_ids=candidate_a1.evidence_ids,
            ),
            AuthenticatedPersistedFactProposalItem(
                proposal_index=1,
                fact_id=fact_b,
                fact_value_id=fv_b1,
                subject_entity_id=None,
                referenced_entity_id=None,
                evidence_ids=candidate_b1.evidence_ids,
            ),
        ),
        result_hash="1" * 64,
    )
    application_snapshot_b = _application_snapshot(
        application_id=app_b,
        project_id=project_id,
        extraction_run_id=extraction_run_id,
        inference_run_id=inference_b,
        input_batch_id=_uuid("input-b"),
        items=(
            AuthenticatedPersistedFactProposalItem(
                proposal_index=0,
                fact_id=fact_b,
                fact_value_id=fv_b2,
                subject_entity_id=None,
                referenced_entity_id=None,
                evidence_ids=candidate_b2.evidence_ids,
            ),
            AuthenticatedPersistedFactProposalItem(
                proposal_index=1,
                fact_id=fact_a,
                fact_value_id=fv_a2,
                subject_entity_id=None,
                referenced_entity_id=None,
                evidence_ids=candidate_a2.evidence_ids,
            ),
        ),
        result_hash="2" * 64,
    )
    source_snapshot = ufl_fact_snapshot_service.duplicate_grouping_service.AuthenticatedDuplicateGroupingSourceSnapshot(
        state=_state(
            orchestration_id=orchestration_id,
            extraction_run_id=extraction_run_id,
            project_id=project_id,
            orchestration_status=orchestration_status,
        ),
        candidate_count=4,
        candidates=(candidate_b2, candidate_a2, candidate_b1, candidate_a1),
        application_snapshots=(application_snapshot_a, application_snapshot_b),
    )
    rows = (
        _row(
            project_id=project_id,
            fact_id=fact_b,
            fact_value_id=fv_b2,
            orchestration_id=orchestration_id,
            extraction_run_id=extraction_run_id,
            source_batch_id=batch_b,
            application_id=app_b,
            inference_run_id=inference_b,
            document_revision_id=document_revision_id,
            subject_kind="person",
            subject_key="zeta",
            predicate_key="score",
            scope_key="2025",
            value_type="number",
            value_json=20,
            evidence_link_id=candidate_b2.evidence_link_ids[0],
            evidence_id=candidate_b2.evidence_ids[0],
            document_block_id=_uuid("block-b2"),
            block_source_order=3,
            proposal_index=0,
        ),
        _row(
            project_id=project_id,
            fact_id=fact_a,
            fact_value_id=fv_a2,
            orchestration_id=orchestration_id,
            extraction_run_id=extraction_run_id,
            source_batch_id=batch_b,
            application_id=app_b,
            inference_run_id=inference_b,
            document_revision_id=document_revision_id,
            subject_kind="person",
            subject_key="alpha",
            predicate_key="title",
            scope_key=None,
            value_type="string",
            value_json="same",
            evidence_link_id=candidate_a2.evidence_link_ids[0],
            evidence_id=candidate_a2.evidence_ids[0],
            document_block_id=_uuid("block-a2"),
            block_source_order=1,
            proposal_index=1,
        ),
        _row(
            project_id=project_id,
            fact_id=fact_b,
            fact_value_id=fv_b1,
            orchestration_id=orchestration_id,
            extraction_run_id=extraction_run_id,
            source_batch_id=batch_a,
            application_id=app_a,
            inference_run_id=inference_a,
            document_revision_id=document_revision_id,
            subject_kind="person",
            subject_key="zeta",
            predicate_key="score",
            scope_key="2025",
            value_type="number",
            value_json=10,
            evidence_link_id=candidate_b1.evidence_link_ids[0],
            evidence_id=candidate_b1.evidence_ids[0],
            document_block_id=_uuid("block-b1"),
            block_source_order=2,
            proposal_index=1,
        ),
        _row(
            project_id=project_id,
            fact_id=fact_a,
            fact_value_id=fv_a1,
            orchestration_id=orchestration_id,
            extraction_run_id=extraction_run_id,
            source_batch_id=batch_a,
            application_id=app_a,
            inference_run_id=inference_a,
            document_revision_id=document_revision_id,
            subject_kind="person",
            subject_key="alpha",
            predicate_key="title",
            scope_key=None,
            value_type="string",
            value_json="same",
            evidence_link_id=candidate_a1.evidence_link_ids[0],
            evidence_id=candidate_a1.evidence_ids[0],
            document_block_id=_uuid("block-a1"),
            block_source_order=0,
            proposal_index=0,
        ),
    )
    return {
        "project_id": project_id,
        "orchestration_id": orchestration_id,
        "source_snapshot": source_snapshot,
        "rows": rows,
    }


def test_get_orchestration_ufl_fact_snapshot_returns_completed_snapshot_with_grouping_and_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _rich_fixture()
    factory = SessionFactory()
    _install_sources(
        monkeypatch,
        source_snapshot=fixture["source_snapshot"],
        rows=fixture["rows"],
    )

    snapshot = run_async(
        ufl_fact_snapshot_service.get_orchestration_ufl_fact_snapshot(
            factory,
            project_id=fixture["project_id"],
            orchestration_id=fixture["orchestration_id"],
        )
    )

    assert snapshot.orchestration_status == "completed"
    assert snapshot.comparison_quality == "complete"
    assert snapshot.source_application_count == 2
    assert snapshot.fact_count == 2
    assert snapshot.fact_value_count == 4
    assert snapshot.evidence_count == 4
    assert snapshot.facts[0].subject_key == "alpha"
    assert snapshot.facts[0].semantic_group_count == 1
    assert snapshot.facts[0].fact_value_count == 2
    assert [value.source_application_id for value in snapshot.facts[0].value_groups[0].values] == [
        _uuid("app-a"),
        _uuid("app-b"),
    ]
    assert snapshot.facts[1].semantic_group_count == 2
    assert [group.semantic_key_hash for group in snapshot.facts[1].value_groups] == sorted(
        group.semantic_key_hash for group in snapshot.facts[1].value_groups
    )
    assert factory.sessions[0].commit_count == 0
    assert factory.sessions[0].rollback_count == 1


def test_get_orchestration_ufl_fact_snapshot_returns_partial_quality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _rich_fixture(orchestration_status="partial")
    factory = SessionFactory()
    _install_sources(
        monkeypatch,
        source_snapshot=fixture["source_snapshot"],
        rows=fixture["rows"],
    )

    snapshot = run_async(
        ufl_fact_snapshot_service.get_orchestration_ufl_fact_snapshot(
            factory,
            project_id=fixture["project_id"],
            orchestration_id=fixture["orchestration_id"],
        )
    )

    assert snapshot.orchestration_status == "partial"
    assert snapshot.comparison_quality == "partial"


def test_get_orchestration_ufl_fact_snapshot_returns_zero_fact_snapshot_for_all_withheld(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id = _uuid("withheld-project")
    orchestration_id = _uuid("withheld-orchestration")
    source_snapshot = ufl_fact_snapshot_service.duplicate_grouping_service.AuthenticatedDuplicateGroupingSourceSnapshot(
        state=_state(
            orchestration_id=orchestration_id,
            extraction_run_id=_uuid("withheld-run"),
            project_id=project_id,
        ),
        candidate_count=0,
        candidates=(),
        application_snapshots=(
            _application_snapshot(
                application_id=_uuid("withheld-app"),
                project_id=project_id,
                extraction_run_id=_uuid("withheld-run"),
                inference_run_id=_uuid("withheld-inference"),
                input_batch_id=_uuid("withheld-input"),
                items=(),
                result_hash="3" * 64,
            ),
        ),
    )
    factory = SessionFactory()
    _install_sources(monkeypatch, source_snapshot=source_snapshot, rows=())

    snapshot = run_async(
        ufl_fact_snapshot_service.get_orchestration_ufl_fact_snapshot(
            factory,
            project_id=project_id,
            orchestration_id=orchestration_id,
        )
    )

    assert snapshot.fact_count == 0
    assert snapshot.fact_value_count == 0
    assert snapshot.evidence_count == 0
    assert snapshot.facts == ()


@pytest.mark.parametrize(
    "mutator",
    [
        lambda fixture: fixture.update(rows=fixture["rows"][:-1]),
        lambda fixture: fixture.update(
            rows=fixture["rows"]
            + (
                replace(
                    fixture["rows"][0],
                    fact_value_id=_uuid("extra-fact-value"),
                    fact_id=_uuid("extra-fact"),
                ),
            )
        ),
        lambda fixture: fixture.update(
            rows=(replace(fixture["rows"][0], source_batch_id=_uuid("wrong-batch")),)
            + fixture["rows"][1:]
        ),
    ],
)
def test_get_orchestration_ufl_fact_snapshot_rejects_membership_drift(
    monkeypatch: pytest.MonkeyPatch,
    mutator,
) -> None:
    fixture = _rich_fixture()
    mutator(fixture)
    factory = SessionFactory()
    _install_sources(monkeypatch, source_snapshot=fixture["source_snapshot"], rows=fixture["rows"])

    with pytest.raises(
        ufl_fact_snapshot_service.OrchestrationUFLFactSnapshotInvariantError,
        match="orchestration_ufl_fact_snapshot_source_mismatch",
    ):
        run_async(
            ufl_fact_snapshot_service.get_orchestration_ufl_fact_snapshot(
                factory,
                project_id=fixture["project_id"],
                orchestration_id=fixture["orchestration_id"],
            )
        )


@pytest.mark.parametrize(
    ("index", "field_name", "value"),
    [
        (0, "fact_identity_hash", "0" * 64),
        (1, "fact_value_hash", "1" * 64),
    ],
)
def test_get_orchestration_ufl_fact_snapshot_rejects_fact_or_value_hash_drift(
    monkeypatch: pytest.MonkeyPatch,
    index: int,
    field_name: str,
    value: object,
) -> None:
    fixture = _rich_fixture()
    mutated_rows = list(fixture["rows"])
    mutated_rows[index] = replace(mutated_rows[index], **{field_name: value})
    factory = SessionFactory()
    _install_sources(
        monkeypatch,
        source_snapshot=fixture["source_snapshot"],
        rows=tuple(mutated_rows),
    )

    with pytest.raises(
        ufl_fact_snapshot_service.OrchestrationUFLFactSnapshotInvariantError,
        match="orchestration_ufl_fact_snapshot_source_mismatch",
    ):
        run_async(
            ufl_fact_snapshot_service.get_orchestration_ufl_fact_snapshot(
                factory,
                project_id=fixture["project_id"],
                orchestration_id=fixture["orchestration_id"],
            )
        )


def test_get_orchestration_ufl_fact_snapshot_rejects_evidence_binding_or_order_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _rich_fixture()
    candidates = list(fixture["source_snapshot"].candidates)
    candidates[0] = replace(
        candidates[0],
        evidence_link_ids=(_uuid("wrong-evidence-link"),),
    )
    fixture["source_snapshot"] = replace(
        fixture["source_snapshot"],
        candidates=tuple(candidates),
    )
    factory = SessionFactory()
    _install_sources(monkeypatch, source_snapshot=fixture["source_snapshot"], rows=fixture["rows"])

    with pytest.raises(
        ufl_fact_snapshot_service.OrchestrationUFLFactSnapshotInvariantError,
        match="orchestration_ufl_fact_snapshot_source_mismatch",
    ):
        run_async(
            ufl_fact_snapshot_service.get_orchestration_ufl_fact_snapshot(
                factory,
                project_id=fixture["project_id"],
                orchestration_id=fixture["orchestration_id"],
            )
        )


def test_get_orchestration_ufl_fact_snapshot_rejects_candidate_value_drift_without_leaking_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _rich_fixture()
    candidates = list(fixture["source_snapshot"].candidates)
    candidates[0] = replace(candidates[0], value_json="SENSITIVE_UFL_SENTINEL")
    fixture["source_snapshot"] = replace(
        fixture["source_snapshot"],
        candidates=tuple(candidates),
    )
    factory = SessionFactory()
    _install_sources(monkeypatch, source_snapshot=fixture["source_snapshot"], rows=fixture["rows"])

    with pytest.raises(
        ufl_fact_snapshot_service.OrchestrationUFLFactSnapshotInvariantError,
        match="orchestration_ufl_fact_snapshot_source_mismatch",
    ) as exc_info:
        run_async(
            ufl_fact_snapshot_service.get_orchestration_ufl_fact_snapshot(
                factory,
                project_id=fixture["project_id"],
                orchestration_id=fixture["orchestration_id"],
            )
        )

    assert "SENSITIVE_UFL_SENTINEL" not in str(exc_info.value)


def test_get_orchestration_ufl_fact_snapshot_is_deterministic_and_manifest_changes_with_identity_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _rich_fixture()
    factory = SessionFactory()
    _install_sources(
        monkeypatch,
        source_snapshot=fixture["source_snapshot"],
        rows=(
            fixture["rows"][2],
            fixture["rows"][0],
            fixture["rows"][3],
            fixture["rows"][1],
        ),
    )

    first = run_async(
        ufl_fact_snapshot_service.get_orchestration_ufl_fact_snapshot(
            factory,
            project_id=fixture["project_id"],
            orchestration_id=fixture["orchestration_id"],
        )
    )
    second = run_async(
        ufl_fact_snapshot_service.get_orchestration_ufl_fact_snapshot(
            factory,
            project_id=fixture["project_id"],
            orchestration_id=fixture["orchestration_id"],
        )
    )

    assert first == second
    changed_source_snapshot = replace(
        fixture["source_snapshot"],
        application_snapshots=(
            replace(fixture["source_snapshot"].application_snapshots[0], result_hash="4" * 64),
            fixture["source_snapshot"].application_snapshots[1],
        ),
    )
    _install_sources(
        monkeypatch,
        source_snapshot=changed_source_snapshot,
        rows=fixture["rows"],
    )
    changed = run_async(
        ufl_fact_snapshot_service.get_orchestration_ufl_fact_snapshot(
            factory,
            project_id=fixture["project_id"],
            orchestration_id=fixture["orchestration_id"],
        )
    )

    assert first.source_manifest_hash != changed.source_manifest_hash


def test_get_orchestration_ufl_fact_snapshot_deep_freezes_value_json_and_detaches_from_fixture_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _rich_fixture()
    nested_value_json = {
        "profile": {
            "aliases": ["alpha", "beta"],
            "metrics": [1, {"score": 2.0}],
        }
    }
    candidates = list(fixture["source_snapshot"].candidates)
    candidates[0] = replace(
        candidates[0],
        value_type="object",
        value_json=nested_value_json,
    )
    normalized_object_value = _normalized_fact_value(
        value_type="object",
        value_json=nested_value_json,
    )
    rows = list(fixture["rows"])
    rows[0] = replace(
        rows[0],
        value_type="object",
        value_json=nested_value_json,
        normalized_value_text=normalized_object_value.normalized_value_text,
        fact_value_hash=normalized_object_value.value_hash,
    )
    fixture["source_snapshot"] = replace(
        fixture["source_snapshot"],
        candidates=tuple(candidates),
    )
    factory = SessionFactory()
    _install_sources(
        monkeypatch,
        source_snapshot=fixture["source_snapshot"],
        rows=tuple(rows),
    )

    snapshot = run_async(
        ufl_fact_snapshot_service.get_orchestration_ufl_fact_snapshot(
            factory,
            project_id=fixture["project_id"],
            orchestration_id=fixture["orchestration_id"],
        )
    )

    nested_value_json["profile"]["aliases"].append("gamma")
    nested_value_json["profile"]["metrics"][1]["score"] = 3.0

    frozen_value_json = next(
        group.value_json
        for group in snapshot.facts[1].value_groups
        if group.value_type == "object"
    )
    assert isinstance(frozen_value_json, MappingProxyType)
    assert isinstance(frozen_value_json["profile"], MappingProxyType)
    assert isinstance(frozen_value_json["profile"]["aliases"], tuple)
    assert frozen_value_json["profile"]["aliases"] == ("alpha", "beta")
    assert frozen_value_json["profile"]["metrics"][1]["score"] == 2.0
    with pytest.raises(TypeError):
        frozen_value_json["profile"] = {}
    with pytest.raises(TypeError):
        frozen_value_json["profile"]["aliases"] += ("delta",)
    with pytest.raises(AttributeError):
        frozen_value_json["profile"]["aliases"].append("delta")


@pytest.mark.parametrize(
    "invalid_value_json",
    [
        {"bad": {"nan": float("nan")}},
        {"bad": {"payload": b"bytes"}},
        {"bad": {1: "value"}},
        {"bad": {"nested": CustomJSONValue()}},
    ],
)
def test_get_orchestration_ufl_fact_snapshot_rejects_invalid_nested_value_json(
    monkeypatch: pytest.MonkeyPatch,
    invalid_value_json: object,
) -> None:
    fixture = _rich_fixture()
    authenticated_facts = ufl_fact_snapshot_service.fact_diff_service.build_authenticated_fact_source_facts(
        rows=fixture["rows"],
        source_snapshot=fixture["source_snapshot"],
        expected_run_id=fixture["source_snapshot"].state.extraction_run_id,
    )

    mutated_fact = next(iter(authenticated_facts.values()))
    mutated_group = replace(mutated_fact.value_groups[0], value_json=invalid_value_json)
    malformed_facts = {
        mutated_fact.fact.fact_id: replace(
            mutated_fact,
            value_groups=(mutated_group,) + mutated_fact.value_groups[1:],
        )
    }

    monkeypatch.setattr(
        ufl_fact_snapshot_service.fact_diff_service,
        "build_authenticated_fact_source_facts",
        lambda **_kwargs: malformed_facts,
    )
    factory = SessionFactory()
    _install_sources(
        monkeypatch,
        source_snapshot=fixture["source_snapshot"],
        rows=fixture["rows"],
    )

    with pytest.raises(
        ufl_fact_snapshot_service.OrchestrationUFLFactSnapshotInvariantError,
        match="orchestration_ufl_fact_snapshot_value_json_invalid",
    ):
        run_async(
            ufl_fact_snapshot_service.get_orchestration_ufl_fact_snapshot(
                factory,
                project_id=fixture["project_id"],
                orchestration_id=fixture["orchestration_id"],
            )
        )


@pytest.mark.parametrize(
    ("row_overrides", "application_overrides", "expected_code"),
    [
        ({"language_code": " zh-CN "}, None, "orchestration_ufl_fact_snapshot_value_metadata_invalid"),
        ({"confidence": True}, None, "orchestration_ufl_fact_snapshot_value_metadata_invalid"),
        ({"confidence": float("inf")}, None, "orchestration_ufl_fact_snapshot_value_metadata_invalid"),
        (None, {"proposal_index": True}, "orchestration_ufl_fact_snapshot_value_metadata_invalid"),
        (None, {"proposal_index": -1}, "orchestration_ufl_fact_snapshot_value_metadata_invalid"),
    ],
)
def test_get_orchestration_ufl_fact_snapshot_rejects_invalid_value_metadata_shapes(
    monkeypatch: pytest.MonkeyPatch,
    row_overrides: dict | None,
    application_overrides: dict | None,
    expected_code: str,
) -> None:
    fixture = _rich_fixture()
    rows = list(fixture["rows"])
    if row_overrides is not None:
        rows[0] = replace(rows[0], **row_overrides)
    application_snapshots = list(fixture["source_snapshot"].application_snapshots)
    if application_overrides is not None:
        items = list(application_snapshots[1].items)
        items[0] = replace(items[0], **{
            key: value
            for key, value in application_overrides.items()
            if key == "proposal_index"
        })
        application_snapshots[1] = replace(
            application_snapshots[1],
            application_id=application_overrides.get(
                "application_id",
                application_snapshots[1].application_id,
            ),
            items=tuple(items),
        )
    fixture["source_snapshot"] = replace(
        fixture["source_snapshot"],
        application_snapshots=tuple(application_snapshots),
    )
    factory = SessionFactory()
    _install_sources(
        monkeypatch,
        source_snapshot=fixture["source_snapshot"],
        rows=tuple(rows),
    )

    with pytest.raises(
        ufl_fact_snapshot_service.OrchestrationUFLFactSnapshotInvariantError,
        match=expected_code,
    ):
        run_async(
            ufl_fact_snapshot_service.get_orchestration_ufl_fact_snapshot(
                factory,
                project_id=fixture["project_id"],
                orchestration_id=fixture["orchestration_id"],
            )
        )


def test_get_orchestration_ufl_fact_snapshot_preserves_canonical_json_semantics_for_frozen_value_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _rich_fixture()
    raw_value_json = {
        "items": ["alpha", {"nested": [1, 2, 3]}],
        "enabled": True,
    }
    candidates = list(fixture["source_snapshot"].candidates)
    candidates[0] = replace(
        candidates[0],
        value_type="object",
        value_json=raw_value_json,
    )
    normalized_object_value = _normalized_fact_value(
        value_type="object",
        value_json=raw_value_json,
    )
    rows = list(fixture["rows"])
    rows[0] = replace(
        rows[0],
        value_type="object",
        value_json=raw_value_json,
        normalized_value_text=normalized_object_value.normalized_value_text,
        fact_value_hash=normalized_object_value.value_hash,
    )
    fixture["source_snapshot"] = replace(
        fixture["source_snapshot"],
        candidates=tuple(candidates),
    )
    factory = SessionFactory()
    _install_sources(
        monkeypatch,
        source_snapshot=fixture["source_snapshot"],
        rows=tuple(rows),
    )

    snapshot = run_async(
        ufl_fact_snapshot_service.get_orchestration_ufl_fact_snapshot(
            factory,
            project_id=fixture["project_id"],
            orchestration_id=fixture["orchestration_id"],
        )
    )

    frozen_value_json = next(
        group.value_json
        for group in snapshot.facts[1].value_groups
        if group.value_type == "object"
    )
    assert (
        ufl_fact_snapshot_service.duplicate_grouping_service.hash_deterministic_payload(
            {"value_json": raw_value_json}
        )
        == ufl_fact_snapshot_service.duplicate_grouping_service.hash_deterministic_payload(
            {"value_json": frozen_value_json}
        )
    )


@pytest.mark.parametrize(
    ("value_type", "value_json"),
    [
        ("string", "same"),
        ("list", ["alpha", {"nested": [1, 2]}]),
        ("object", {"items": ["alpha", {"nested": [1, 2]}]}),
    ],
)
def test_get_orchestration_ufl_fact_snapshot_keeps_manifest_deterministic_for_valid_value_shapes(
    monkeypatch: pytest.MonkeyPatch,
    value_type: str,
    value_json: object,
) -> None:
    fixture = _rich_fixture()
    candidates = list(fixture["source_snapshot"].candidates)
    candidates[0] = replace(
        candidates[0],
        value_type=value_type,
        value_json=value_json,
    )
    rows = list(fixture["rows"])
    normalized_value = _normalized_fact_value(
        value_type=value_type,
        value_json=value_json,
    )
    rows[0] = replace(
        rows[0],
        value_type=normalized_value.value_type,
        value_json=normalized_value.value_json,
        normalized_value_text=normalized_value.normalized_value_text,
        fact_value_hash=normalized_value.value_hash,
    )
    fixture["source_snapshot"] = replace(
        fixture["source_snapshot"],
        candidates=tuple(candidates),
    )
    factory = SessionFactory()
    _install_sources(
        monkeypatch,
        source_snapshot=fixture["source_snapshot"],
        rows=tuple(rows),
    )

    first = run_async(
        ufl_fact_snapshot_service.get_orchestration_ufl_fact_snapshot(
            factory,
            project_id=fixture["project_id"],
            orchestration_id=fixture["orchestration_id"],
        )
    )
    second = run_async(
        ufl_fact_snapshot_service.get_orchestration_ufl_fact_snapshot(
            factory,
            project_id=fixture["project_id"],
            orchestration_id=fixture["orchestration_id"],
        )
    )

    assert first.source_manifest_hash == second.source_manifest_hash


@pytest.mark.parametrize(
    ("group_overrides", "value_overrides"),
    [
        ({"semantic_key_hash": "BAD"}, None),
        (None, {"value_hash": "BAD"}),
        (None, {"source_application_id": "not-a-uuid"}),
    ],
)
def test_get_orchestration_ufl_fact_snapshot_rejects_invalid_hash_shapes_in_authenticated_facts(
    monkeypatch: pytest.MonkeyPatch,
    group_overrides: dict | None,
    value_overrides: dict | None,
) -> None:
    fixture = _rich_fixture()
    authenticated_facts = ufl_fact_snapshot_service.fact_diff_service.build_authenticated_fact_source_facts(
        rows=fixture["rows"],
        source_snapshot=fixture["source_snapshot"],
        expected_run_id=fixture["source_snapshot"].state.extraction_run_id,
    )
    mutated_fact = next(iter(authenticated_facts.values()))
    mutated_group = mutated_fact.value_groups[0]
    if value_overrides is not None:
        mutated_values = list(mutated_group.values)
        mutated_values[0] = replace(mutated_values[0], **value_overrides)
        mutated_group = replace(mutated_group, values=tuple(mutated_values))
    if group_overrides is not None:
        mutated_group = replace(mutated_group, **group_overrides)
    malformed_facts = {
        mutated_fact.fact.fact_id: replace(
            mutated_fact,
            value_groups=(mutated_group,) + mutated_fact.value_groups[1:],
        )
    }
    monkeypatch.setattr(
        ufl_fact_snapshot_service.fact_diff_service,
        "build_authenticated_fact_source_facts",
        lambda **_kwargs: malformed_facts,
    )
    factory = SessionFactory()
    _install_sources(
        monkeypatch,
        source_snapshot=fixture["source_snapshot"],
        rows=fixture["rows"],
    )

    with pytest.raises(
        ufl_fact_snapshot_service.OrchestrationUFLFactSnapshotInvariantError,
        match="orchestration_ufl_fact_snapshot_value_metadata_invalid",
    ):
        run_async(
            ufl_fact_snapshot_service.get_orchestration_ufl_fact_snapshot(
                factory,
                project_id=fixture["project_id"],
                orchestration_id=fixture["orchestration_id"],
            )
        )


def test_get_orchestration_ufl_fact_snapshot_uses_shared_single_side_auth_and_stays_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _rich_fixture()
    factory = SessionFactory()
    _install_sources(monkeypatch, source_snapshot=fixture["source_snapshot"], rows=fixture["rows"])

    snapshot = run_async(
        ufl_fact_snapshot_service.get_orchestration_ufl_fact_snapshot(
            factory,
            project_id=fixture["project_id"],
            orchestration_id=fixture["orchestration_id"],
        )
    )

    source = inspect.getsource(ufl_fact_snapshot_service)
    assert "build_authenticated_fact_source_facts" in source
    assert "current_value_id" not in source
    assert snapshot.algorithm_name == "orchestration_ufl_fact_snapshot"
    assert snapshot.algorithm_version == "1.0.0"
    assert all(session.commit_count == 0 for session in factory.sessions)
    assert all(session.rollback_count == 1 for session in factory.sessions)
