from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import uuid
from types import SimpleNamespace

import pytest

from app.repositories import consistency_check as consistency_check_repository
from app.schemas.consistency_check import (
    ConsistencyCheckPlannerConfig,
)
from app.schemas.fact_value_duplicate_grouping import (
    DuplicateGroupingApplicationLedger,
    FactValueConsistencyCandidateApplicationLedger,
    FactValueConsistencyCandidateLedger,
    FactValueConsistencyCandidateMemberLedger,
)
from app.services import consistency_check as consistency_check_service
from app.services import fact_value_duplicate_grouping as duplicate_grouping_service


def run_async(awaitable):
    return asyncio.run(awaitable)


class FakeSession:
    def __init__(self) -> None:
        self.rollback_count = 0

    async def rollback(self) -> None:
        self.rollback_count += 1


class SessionFactory:
    def __init__(self) -> None:
        self.open_count = 0
        self.sessions: list[FakeSession] = []

    def __call__(self):
        factory = self

        class _Context:
            async def __aenter__(self_inner):
                factory.open_count += 1
                session = FakeSession()
                factory.sessions.append(session)
                return session

            async def __aexit__(self_inner, exc_type, exc, tb):
                factory.open_count -= 1
                return False

        return _Context()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _member_spec(
    *,
    member_id: uuid.UUID | None = None,
    fact_value_id: uuid.UUID | None = None,
    source_batch_id: uuid.UUID | None = None,
    semantic_key_hash: str,
    value_type: str = "string",
    value_json=None,
    referenced_entity_id: uuid.UUID | None = None,
    evidences: list[dict] | None = None,
) -> dict:
    return {
        "member_id": member_id or uuid.uuid4(),
        "fact_value_id": fact_value_id or uuid.uuid4(),
        "source_batch_id": source_batch_id or uuid.uuid4(),
        "semantic_key_hash": semantic_key_hash,
        "value_type": value_type,
        "value_json": value_json,
        "referenced_entity_id": referenced_entity_id,
        "evidences": evidences or [],
    }


def _evidence_spec(
    *,
    evidence_link_id: uuid.UUID | None = None,
    evidence_id: uuid.UUID | None = None,
    role: str = "supporting",
    is_primary: bool = True,
    source_order: int,
    excerpt: str,
    block_id: uuid.UUID | None = None,
    location_key: str = "loc-1",
    page_no: int | None = 1,
    start_line: int | None = 1,
    end_line: int | None = 1,
    start_offset: int = 0,
    end_offset: int | None = None,
) -> dict:
    actual_end_offset = len(excerpt) if end_offset is None else end_offset
    return {
        "evidence_link_id": evidence_link_id or uuid.uuid4(),
        "evidence_id": evidence_id or uuid.uuid4(),
        "role": role,
        "is_primary": is_primary,
        "source_order": source_order,
        "excerpt": excerpt,
        "excerpt_hash": sha256(excerpt),
        "block_id": block_id or uuid.uuid4(),
        "location_key": location_key,
        "page_no": page_no,
        "start_line": start_line,
        "end_line": end_line,
        "start_offset": start_offset,
        "end_offset": actual_end_offset,
    }


def _candidate_spec(
    *,
    candidate_id: uuid.UUID | None = None,
    fact_id: uuid.UUID | None = None,
    candidate_kind: str = "multi_value",
    members: list[dict],
) -> dict:
    return {
        "candidate_id": candidate_id or uuid.uuid4(),
        "fact_id": fact_id or uuid.uuid4(),
        "candidate_kind": candidate_kind,
        "members": members,
    }


def make_authenticated_application(
    candidate_specs: list[dict],
    *,
    consistency_application_id: uuid.UUID | None = None,
    duplicate_grouping_application_id: uuid.UUID | None = None,
    orchestration_id: uuid.UUID | None = None,
    extraction_run_id: uuid.UUID | None = None,
    result_manifest_hash: str = "c" * 64,
) -> SimpleNamespace:
    application_id = consistency_application_id or uuid.uuid4()
    source_application_id = duplicate_grouping_application_id or uuid.uuid4()
    actual_orchestration_id = orchestration_id or uuid.uuid4()
    actual_extraction_run_id = extraction_run_id or uuid.uuid4()
    candidate_ledgers: list[FactValueConsistencyCandidateLedger] = []
    member_ledgers: list[FactValueConsistencyCandidateMemberLedger] = []

    for candidate in candidate_specs:
        members = candidate["members"]
        candidate_ledgers.append(
            FactValueConsistencyCandidateLedger(
                id=candidate["candidate_id"],
                consistency_application_id=application_id,
                fact_id=candidate["fact_id"],
                candidate_kind=candidate["candidate_kind"],
                member_count=len(members),
                distinct_semantic_key_count=len({member["semantic_key_hash"] for member in members}),
                distinct_batch_count=len({member["source_batch_id"] for member in members}),
                created_at=datetime.now(timezone.utc),
            )
        )
        for member in members:
            member_ledgers.append(
                FactValueConsistencyCandidateMemberLedger(
                    id=member["member_id"],
                    consistency_application_id=application_id,
                    candidate_id=candidate["candidate_id"],
                    orchestration_id=actual_orchestration_id,
                    fact_value_id=member["fact_value_id"],
                    source_batch_id=member["source_batch_id"],
                    semantic_key_hash=member["semantic_key_hash"],
                    created_at=datetime.now(timezone.utc),
                )
            )

    application = FactValueConsistencyCandidateApplicationLedger(
        id=application_id,
        duplicate_grouping_application_id=source_application_id,
        orchestration_id=actual_orchestration_id,
        extraction_run_id=actual_extraction_run_id,
        algorithm_version="cross_batch_multi_value_v1",
        input_manifest_hash="a" * 64,
        result_manifest_hash=result_manifest_hash,
        candidate_count=len(candidate_ledgers),
        member_count=len(member_ledgers),
        created_at=datetime.now(timezone.utc),
    )
    source_application = DuplicateGroupingApplicationLedger(
        id=source_application_id,
        orchestration_id=actual_orchestration_id,
        extraction_run_id=actual_extraction_run_id,
        algorithm_version="cross_batch_exact_v2",
        input_manifest_hash="b" * 64,
        result_manifest_hash="d" * 64,
        input_fact_value_count=len(member_ledgers),
        duplicate_group_count=0,
        duplicate_member_count=0,
        created_at=datetime.now(timezone.utc),
    )
    return SimpleNamespace(
        application=application,
        source_duplicate_grouping_application=source_application,
        write_plan=None,
        candidate_ledgers=tuple(candidate_ledgers),
        member_ledgers=tuple(member_ledgers),
    )


def make_rows(
    authenticated: SimpleNamespace,
    candidate_specs: list[dict],
) -> tuple[consistency_check_repository.ConsistencyCheckCandidateRow, ...]:
    rows: list[consistency_check_repository.ConsistencyCheckCandidateRow] = []
    for candidate in candidate_specs:
        for member in candidate["members"]:
            for evidence in member["evidences"]:
                rows.append(
                    consistency_check_repository.ConsistencyCheckCandidateRow(
                        candidate_id=candidate["candidate_id"],
                        consistency_application_id=authenticated.application.id,
                        candidate_fact_id=candidate["fact_id"],
                        candidate_kind=candidate["candidate_kind"],
                        member_id=member["member_id"],
                        member_candidate_id=candidate["candidate_id"],
                        member_consistency_application_id=authenticated.application.id,
                        member_orchestration_id=authenticated.application.orchestration_id,
                        member_fact_value_id=member["fact_value_id"],
                        member_source_batch_id=member["source_batch_id"],
                        member_semantic_key_hash=member["semantic_key_hash"],
                        fact_value_fact_id=candidate["fact_id"],
                        fact_value_value_type=member["value_type"],
                        fact_value_value_json=member["value_json"],
                        fact_value_referenced_entity_id=member["referenced_entity_id"],
                        batch_orchestration_id=authenticated.application.orchestration_id,
                        evidence_link_id=evidence["evidence_link_id"],
                        evidence_link_fact_value_id=member["fact_value_id"],
                        evidence_id=evidence["evidence_id"],
                        evidence_role=evidence["role"],
                        evidence_is_primary=evidence["is_primary"],
                        evidence_source_order=evidence["source_order"],
                        block_id=evidence["block_id"],
                        start_offset=evidence["start_offset"],
                        end_offset=evidence["end_offset"],
                        excerpt=evidence["excerpt"],
                        excerpt_hash=evidence["excerpt_hash"],
                        location_key=evidence["location_key"],
                        page_no=evidence["page_no"],
                        start_line=evidence["start_line"],
                        end_line=evidence["end_line"],
                    )
                )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                str(row.candidate_fact_id),
                str(row.candidate_id),
                row.member_semantic_key_hash,
                str(row.member_source_batch_id),
                str(row.member_fact_value_id),
                row.evidence_source_order if row.evidence_source_order is not None else 2**31 - 1,
                str(row.evidence_link_id) if row.evidence_link_id is not None else "z" * 36,
            ),
        )
    )


async def _return_rows(_session, *, consistency_application_id):
    raise AssertionError("test must monkeypatch this helper")


def test_build_consistency_check_plan_builds_full_evidence_bundle_for_two_value_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_specs = [
        _candidate_spec(
            fact_id=uuid.uuid4(),
            members=[
                _member_spec(
                    semantic_key_hash="1" * 64,
                    value_json="Alice",
                    evidences=[_evidence_spec(source_order=0, excerpt="Alice lives in City A", location_key="md:a")],
                ),
                _member_spec(
                    semantic_key_hash="2" * 64,
                    value_json="Alice lives elsewhere",
                    evidences=[_evidence_spec(source_order=0, excerpt="Alice lives in City B", location_key="md:b")],
                ),
            ],
        )
    ]
    authenticated = make_authenticated_application(candidate_specs)
    rows = make_rows(authenticated, candidate_specs)
    session_factory = SessionFactory()

    async def fake_auth(_session_factory, *, consistency_application_id):
        assert consistency_application_id == authenticated.application.id
        return authenticated

    async def fake_rows(_session, *, consistency_application_id):
        assert consistency_application_id == authenticated.application.id
        return rows

    monkeypatch.setattr(
        duplicate_grouping_service,
        "authenticate_fact_value_consistency_candidate_application",
        fake_auth,
    )
    monkeypatch.setattr(
        consistency_check_repository,
        "list_consistency_check_candidate_rows",
        fake_rows,
    )

    plan = run_async(
        consistency_check_service.build_consistency_check_plan(
            session_factory,
            consistency_application_id=authenticated.application.id,
            config=ConsistencyCheckPlannerConfig(
                max_candidates_per_batch=8,
                max_evidence_characters_per_batch=500,
            ),
        )
    )

    assert session_factory.open_count == 0
    assert plan.consistency_application_id == authenticated.application.id
    assert (
        plan.source_result_manifest_hash
        == authenticated.source_duplicate_grouping_application.result_manifest_hash
    )
    assert len(plan.batches) == 1
    batch = plan.batches[0]
    assert batch.candidate_ids == (candidate_specs[0]["candidate_id"],)
    assert batch.candidate_count == 1
    candidate = batch.candidates[0]
    assert candidate.candidate_id == candidate_specs[0]["candidate_id"]
    assert candidate.fact_id == candidate_specs[0]["fact_id"]
    assert candidate.candidate_kind == "multi_value"
    assert plan.source_result_manifest_hash == authenticated.source_duplicate_grouping_application.result_manifest_hash
    assert [member.semantic_key_hash for member in candidate.members] == ["1" * 64, "2" * 64]
    assert candidate.members[0].value_json == "Alice"
    assert candidate.members[0].evidences[0].excerpt == "Alice lives in City A"
    assert candidate.members[0].evidences[0].location_key == "md:a"
    assert candidate.members[0].evidences[0].document_block_id == candidate_specs[0]["members"][0]["evidences"][0]["block_id"]
    assert candidate.members[0].evidences[0].evidence_content_hash == sha256("Alice lives in City A")


def test_build_consistency_check_plan_preserves_member_and_evidence_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    high_uuid = uuid.UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")
    low_uuid = uuid.UUID("00000000-0000-0000-0000-000000000001")
    batch_a = uuid.UUID("00000000-0000-0000-0000-000000000010")
    batch_b = uuid.UUID("00000000-0000-0000-0000-000000000020")
    candidate_specs = [
        _candidate_spec(
            fact_id=uuid.uuid4(),
            members=[
                _member_spec(
                    semantic_key_hash="b" * 64,
                    source_batch_id=batch_b,
                    value_json={"kind": "course", "name": "Math"},
                    value_type="object",
                    evidences=[
                        _evidence_spec(
                            evidence_link_id=high_uuid,
                            source_order=2,
                            excerpt="second evidence",
                            location_key="md:2",
                        ),
                        _evidence_spec(
                            evidence_link_id=low_uuid,
                            source_order=1,
                            excerpt="first evidence",
                            location_key="md:1",
                        ),
                    ],
                ),
                _member_spec(
                    semantic_key_hash="a" * 64,
                    source_batch_id=batch_a,
                    value_json={"kind": "card", "name": "Ace"},
                    value_type="object",
                    evidences=[_evidence_spec(source_order=0, excerpt="alpha evidence", location_key="md:0")],
                ),
            ],
        )
    ]
    authenticated = make_authenticated_application(candidate_specs)
    rows = make_rows(authenticated, candidate_specs)
    session_factory = SessionFactory()

    async def fake_auth(_session_factory, *, consistency_application_id):
        return authenticated

    async def fake_rows(_session, *, consistency_application_id):
        return rows

    monkeypatch.setattr(
        duplicate_grouping_service,
        "authenticate_fact_value_consistency_candidate_application",
        fake_auth,
    )
    monkeypatch.setattr(
        consistency_check_repository,
        "list_consistency_check_candidate_rows",
        fake_rows,
    )

    plan = run_async(
        consistency_check_service.build_consistency_check_plan(
            session_factory,
            consistency_application_id=authenticated.application.id,
            config=ConsistencyCheckPlannerConfig(
                max_candidates_per_batch=8,
                max_evidence_characters_per_batch=500,
            ),
        )
    )

    members = plan.batches[0].candidates[0].members
    assert [member.semantic_key_hash for member in members] == ["a" * 64, "b" * 64]
    assert [evidence.excerpt for evidence in members[1].evidences] == ["first evidence", "second evidence"]
    assert [evidence.source_order for evidence in members[1].evidences] == [1, 2]


@pytest.mark.parametrize(
    "mutation",
    ["missing_evidence", "wrong_fact", "wrong_batch", "wrong_orchestration"],
)
def test_build_consistency_check_plan_fails_closed_on_invalid_evidence_bindings(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    sentinel = "SENSITIVE_EXCERPT_SENTINEL"
    candidate_specs = [
        _candidate_spec(
            fact_id=uuid.uuid4(),
            members=[
                _member_spec(
                    semantic_key_hash="1" * 64,
                    value_json="A",
                    evidences=[_evidence_spec(source_order=0, excerpt=sentinel, location_key="md:a")],
                ),
                _member_spec(
                    semantic_key_hash="2" * 64,
                    value_json="B",
                    evidences=[_evidence_spec(source_order=0, excerpt="normal evidence", location_key="md:b")],
                ),
            ],
        )
    ]
    authenticated = make_authenticated_application(candidate_specs)
    rows = list(make_rows(authenticated, candidate_specs))
    first = rows[0]
    if mutation == "missing_evidence":
        rows[0] = replace(first, evidence_link_id=None, evidence_id=None)
    elif mutation == "wrong_fact":
        rows[0] = replace(first, fact_value_fact_id=uuid.uuid4())
    elif mutation == "wrong_batch":
        rows[0] = replace(first, batch_orchestration_id=uuid.uuid4())
    else:
        rows[0] = replace(first, member_orchestration_id=uuid.uuid4())
    session_factory = SessionFactory()

    async def fake_auth(_session_factory, *, consistency_application_id):
        return authenticated

    async def fake_rows(_session, *, consistency_application_id):
        return tuple(rows)

    monkeypatch.setattr(
        duplicate_grouping_service,
        "authenticate_fact_value_consistency_candidate_application",
        fake_auth,
    )
    monkeypatch.setattr(
        consistency_check_repository,
        "list_consistency_check_candidate_rows",
        fake_rows,
    )

    with pytest.raises(consistency_check_service.ConsistencyCheckPlanInvariantError) as exc_info:
        run_async(
            consistency_check_service.build_consistency_check_plan(
                session_factory,
                consistency_application_id=authenticated.application.id,
                config=ConsistencyCheckPlannerConfig(
                    max_candidates_per_batch=8,
                    max_evidence_characters_per_batch=500,
                ),
            )
        )

    assert sentinel not in str(exc_info.value)


def test_build_consistency_check_plan_deduplicates_same_member_link_but_rejects_cross_member_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_link_id = uuid.uuid4()
    shared_evidence_id = uuid.uuid4()
    candidate_specs = [
        _candidate_spec(
            fact_id=uuid.uuid4(),
            members=[
                _member_spec(
                    semantic_key_hash="1" * 64,
                    value_json="A",
                    evidences=[
                        _evidence_spec(
                            evidence_link_id=shared_link_id,
                            evidence_id=shared_evidence_id,
                            source_order=0,
                            excerpt="same link",
                            location_key="md:a",
                        )
                    ],
                ),
                _member_spec(
                    semantic_key_hash="2" * 64,
                    value_json="B",
                    evidences=[_evidence_spec(source_order=0, excerpt="other link", location_key="md:b")],
                ),
            ],
        )
    ]
    authenticated = make_authenticated_application(candidate_specs)
    base_rows = list(make_rows(authenticated, candidate_specs))
    duplicate_same_member_rows = tuple([base_rows[0], base_rows[0], base_rows[1]])
    cross_member_row = replace(
        base_rows[1],
        evidence_link_id=shared_link_id,
        evidence_id=shared_evidence_id,
    )
    session_factory = SessionFactory()

    async def fake_auth(_session_factory, *, consistency_application_id):
        return authenticated

    async def fake_rows(_session, *, consistency_application_id):
        return duplicate_same_member_rows

    monkeypatch.setattr(
        duplicate_grouping_service,
        "authenticate_fact_value_consistency_candidate_application",
        fake_auth,
    )
    monkeypatch.setattr(
        consistency_check_repository,
        "list_consistency_check_candidate_rows",
        fake_rows,
    )

    plan = run_async(
        consistency_check_service.build_consistency_check_plan(
            session_factory,
            consistency_application_id=authenticated.application.id,
            config=ConsistencyCheckPlannerConfig(
                max_candidates_per_batch=8,
                max_evidence_characters_per_batch=500,
            ),
        )
    )
    assert len(plan.batches[0].candidates[0].members[0].evidences) == 1

    async def fake_cross_member_rows(_session, *, consistency_application_id):
        return (base_rows[0], cross_member_row)

    monkeypatch.setattr(
        consistency_check_repository,
        "list_consistency_check_candidate_rows",
        fake_cross_member_rows,
    )

    with pytest.raises(
        consistency_check_service.ConsistencyCheckPlanInvariantError,
        match="consistency_check_plan_cross_member_link_reuse",
    ):
        run_async(
            consistency_check_service.build_consistency_check_plan(
                session_factory,
                consistency_application_id=authenticated.application.id,
                config=ConsistencyCheckPlannerConfig(
                    max_candidates_per_batch=8,
                    max_evidence_characters_per_batch=500,
                ),
            )
        )


def test_build_consistency_check_plan_is_deterministic_for_same_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_specs = [
        _candidate_spec(
            fact_id=uuid.uuid4(),
            members=[
                _member_spec(
                    semantic_key_hash="1" * 64,
                    value_json="A",
                    evidences=[_evidence_spec(source_order=0, excerpt="alpha", location_key="md:a")],
                ),
                _member_spec(
                    semantic_key_hash="2" * 64,
                    value_json="B",
                    evidences=[_evidence_spec(source_order=0, excerpt="beta", location_key="md:b")],
                ),
            ],
        ),
        _candidate_spec(
            fact_id=uuid.uuid4(),
            members=[
                _member_spec(
                    semantic_key_hash="3" * 64,
                    value_type="entity_ref",
                    referenced_entity_id=uuid.uuid4(),
                    evidences=[_evidence_spec(source_order=0, excerpt="entity alpha", location_key="md:c")],
                ),
                _member_spec(
                    semantic_key_hash="4" * 64,
                    value_type="entity_ref",
                    referenced_entity_id=uuid.uuid4(),
                    evidences=[_evidence_spec(source_order=0, excerpt="entity beta", location_key="md:d")],
                ),
            ],
        ),
    ]
    authenticated = make_authenticated_application(candidate_specs)
    rows = make_rows(authenticated, candidate_specs)
    session_factory = SessionFactory()

    async def fake_auth(_session_factory, *, consistency_application_id):
        return authenticated

    async def fake_rows(_session, *, consistency_application_id):
        return rows

    monkeypatch.setattr(
        duplicate_grouping_service,
        "authenticate_fact_value_consistency_candidate_application",
        fake_auth,
    )
    monkeypatch.setattr(
        consistency_check_repository,
        "list_consistency_check_candidate_rows",
        fake_rows,
    )

    first = run_async(
        consistency_check_service.build_consistency_check_plan(
            session_factory,
            consistency_application_id=authenticated.application.id,
            config=ConsistencyCheckPlannerConfig(
                max_candidates_per_batch=1,
                max_evidence_characters_per_batch=100,
            ),
        )
    )
    second = run_async(
        consistency_check_service.build_consistency_check_plan(
            session_factory,
            consistency_application_id=authenticated.application.id,
            config=ConsistencyCheckPlannerConfig(
                max_candidates_per_batch=1,
                max_evidence_characters_per_batch=100,
            ),
        )
    )

    assert first.plan_manifest_hash == second.plan_manifest_hash
    assert [batch.batch_manifest_hash for batch in first.batches] == [
        batch.batch_manifest_hash for batch in second.batches
    ]


def test_build_consistency_check_plan_batches_without_splitting_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_specs = [
        _candidate_spec(
            candidate_id=uuid.UUID("00000000-0000-0000-0000-000000000101"),
            fact_id=uuid.UUID("00000000-0000-0000-0000-000000000201"),
            members=[
                _member_spec(
                    semantic_key_hash="1" * 64,
                    value_json="A",
                    evidences=[_evidence_spec(source_order=0, excerpt="abcd", location_key="md:a")],
                ),
                _member_spec(
                    semantic_key_hash="2" * 64,
                    value_json="B",
                    evidences=[_evidence_spec(source_order=0, excerpt="efgh", location_key="md:b")],
                ),
            ],
        ),
        _candidate_spec(
            candidate_id=uuid.UUID("00000000-0000-0000-0000-000000000102"),
            fact_id=uuid.UUID("00000000-0000-0000-0000-000000000202"),
            members=[
                _member_spec(
                    semantic_key_hash="3" * 64,
                    value_json="C",
                    evidences=[_evidence_spec(source_order=0, excerpt="ij", location_key="md:c")],
                ),
                _member_spec(
                    semantic_key_hash="4" * 64,
                    value_json="D",
                    evidences=[_evidence_spec(source_order=0, excerpt="kl", location_key="md:d")],
                ),
            ],
        ),
    ]
    authenticated = make_authenticated_application(candidate_specs)
    rows = make_rows(authenticated, candidate_specs)
    session_factory = SessionFactory()

    async def fake_auth(_session_factory, *, consistency_application_id):
        return authenticated

    async def fake_rows(_session, *, consistency_application_id):
        return rows

    monkeypatch.setattr(
        duplicate_grouping_service,
        "authenticate_fact_value_consistency_candidate_application",
        fake_auth,
    )
    monkeypatch.setattr(
        consistency_check_repository,
        "list_consistency_check_candidate_rows",
        fake_rows,
    )

    plan = run_async(
        consistency_check_service.build_consistency_check_plan(
            session_factory,
            consistency_application_id=authenticated.application.id,
            config=ConsistencyCheckPlannerConfig(
                max_candidates_per_batch=1,
                max_evidence_characters_per_batch=20,
            ),
        )
    )

    assert len(plan.batches) == 2
    assert all(batch.candidate_count == 1 for batch in plan.batches)
    assert plan.batches[0].candidate_ids == (candidate_specs[0]["candidate_id"],)
    assert plan.batches[1].candidate_ids == (candidate_specs[1]["candidate_id"],)


def test_build_consistency_check_plan_rejects_oversized_single_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_specs = [
        _candidate_spec(
            fact_id=uuid.uuid4(),
            members=[
                _member_spec(
                    semantic_key_hash="1" * 64,
                    value_json="A",
                    evidences=[_evidence_spec(source_order=0, excerpt="x" * 40, location_key="md:a")],
                ),
                _member_spec(
                    semantic_key_hash="2" * 64,
                    value_json="B",
                    evidences=[_evidence_spec(source_order=0, excerpt="y" * 40, location_key="md:b")],
                ),
            ],
        )
    ]
    authenticated = make_authenticated_application(candidate_specs)
    rows = make_rows(authenticated, candidate_specs)
    session_factory = SessionFactory()

    async def fake_auth(_session_factory, *, consistency_application_id):
        return authenticated

    async def fake_rows(_session, *, consistency_application_id):
        return rows

    monkeypatch.setattr(
        duplicate_grouping_service,
        "authenticate_fact_value_consistency_candidate_application",
        fake_auth,
    )
    monkeypatch.setattr(
        consistency_check_repository,
        "list_consistency_check_candidate_rows",
        fake_rows,
    )

    with pytest.raises(
        consistency_check_service.ConsistencyCheckPlanStateError,
        match="consistency_check_plan_candidate_too_large",
    ):
        run_async(
            consistency_check_service.build_consistency_check_plan(
                session_factory,
                consistency_application_id=authenticated.application.id,
                config=ConsistencyCheckPlannerConfig(
                    max_candidates_per_batch=8,
                    max_evidence_characters_per_batch=20,
                ),
            )
        )
