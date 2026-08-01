from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from decimal import Decimal
from enum import StrEnum
import uuid

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError

from app.models.fact_value_duplicate_grouping import (
    FactValueDuplicateGroup,
    FactValueDuplicateGroupMember,
    FactValueDuplicateGroupingApplication,
)
from app.repositories import fact_value_duplicate_grouping as duplicate_grouping_repository
from app.schemas.fact_value_duplicate_grouping import (
    CROSS_BATCH_DUPLICATE_ALGORITHM_VERSION,
    DuplicateCandidate,
    DuplicateGroupEvidenceProjection,
)
from app.services import fact_value_duplicate_grouping as duplicate_grouping_service


def run_async(awaitable):
    return asyncio.run(awaitable)


def make_integrity_error(constraint_name: str | None) -> IntegrityError:
    class _Orig:
        def __init__(self, value: str | None) -> None:
            self.diag = type("Diag", (), {"constraint_name": value})()

    return IntegrityError("stmt", {}, _Orig(constraint_name))


def candidate(
    *,
    extraction_run_id: uuid.UUID | None = None,
    fact_value_id: uuid.UUID | None = None,
    fact_id: uuid.UUID | None = None,
    source_batch_id: uuid.UUID | None = None,
    value_type: str = "object",
    value_json=None,
    referenced_entity_id: uuid.UUID | None = None,
    evidence_link_ids: tuple[uuid.UUID, ...] | None = None,
) -> DuplicateCandidate:
    return DuplicateCandidate(
        fact_value_id=fact_value_id or uuid.uuid4(),
        fact_id=fact_id or uuid.uuid4(),
        extraction_run_id=extraction_run_id or uuid.uuid4(),
        source_batch_id=source_batch_id or uuid.uuid4(),
        value_type=value_type,
        value_json={"amount": "10", "currency": "CNY"} if value_json is None else value_json,
        referenced_entity_id=referenced_entity_id,
        evidence_link_ids=evidence_link_ids or (uuid.uuid4(),),
    )


class StatusEnum(StrEnum):
    READY = "ready"


class FakeSession:
    def __init__(self) -> None:
        self.commit_count = 0
        self.rollback_count = 0

    async def commit(self) -> None:
        self.commit_count += 1

    async def rollback(self) -> None:
        self.rollback_count += 1


class SessionFactory:
    def __init__(self, sessions: list[FakeSession] | None = None) -> None:
        self.sessions = sessions or []
        self.created_sessions: list[FakeSession] = []

    def __call__(self):
        factory = self

        class _Context:
            async def __aenter__(self_inner):
                session = factory.sessions.pop(0) if factory.sessions else FakeSession()
                factory.created_sessions.append(session)
                return session

            async def __aexit__(self_inner, exc_type, exc, tb):
                return False

        return _Context()


def test_duplicate_grouping_tables_are_registered() -> None:
    assert FactValueDuplicateGroupingApplication.__table__.name == "fact_value_duplicate_grouping_applications"
    assert FactValueDuplicateGroup.__table__.name == "fact_value_duplicate_groups"
    assert FactValueDuplicateGroupMember.__table__.name == "fact_value_duplicate_group_members"


def test_duplicate_fingerprint_is_stable_for_same_fact_and_semantic_value() -> None:
    run_id = uuid.uuid4()
    fact_id = uuid.uuid4()
    left = candidate(
        extraction_run_id=run_id,
        fact_id=fact_id,
        source_batch_id=uuid.uuid4(),
        evidence_link_ids=(uuid.uuid4(), uuid.uuid4()),
        value_json={"b": 2, "a": 1},
    )
    right = candidate(
        extraction_run_id=run_id,
        fact_value_id=uuid.uuid4(),
        fact_id=fact_id,
        source_batch_id=uuid.uuid4(),
        evidence_link_ids=(uuid.uuid4(),),
        value_json={"a": 1, "b": 2},
    )

    left_fp = duplicate_grouping_service.build_duplicate_fingerprint(left)
    right_fp = duplicate_grouping_service.build_duplicate_fingerprint(right)

    assert left_fp.sha256_hex == right_fp.sha256_hex
    assert left_fp.canonical_bytes == right_fp.canonical_bytes


def test_duplicate_fingerprint_supports_decimal_uuid_date_time_and_enum() -> None:
    run_id = uuid.uuid4()
    fact_id = uuid.uuid4()
    entity_id = uuid.uuid4()
    left = candidate(
        extraction_run_id=run_id,
        fact_id=fact_id,
        referenced_entity_id=entity_id,
        value_json={
            "amount": Decimal("10.50"),
            "as_of_date": date(2026, 8, 1),
            "at_time": time(12, 0, 1),
            "at_datetime": datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc),
            "status": StatusEnum.READY,
            "uuid": entity_id,
        },
    )
    right = candidate(
        extraction_run_id=run_id,
        fact_value_id=uuid.uuid4(),
        fact_id=fact_id,
        referenced_entity_id=entity_id,
        value_json={
            "status": StatusEnum.READY,
            "uuid": entity_id,
            "at_datetime": datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc),
            "at_time": time(12, 0, 1),
            "as_of_date": date(2026, 8, 1),
            "amount": Decimal("10.50"),
        },
    )

    assert (
        duplicate_grouping_service.build_duplicate_fingerprint(left).sha256_hex
        == duplicate_grouping_service.build_duplicate_fingerprint(right).sha256_hex
    )


def test_duplicate_fingerprint_distinguishes_none_from_missing_field() -> None:
    with_none = duplicate_grouping_service._canonical_json_bytes({"value": None})
    without_field = duplicate_grouping_service._canonical_json_bytes({})

    assert with_none != without_field


def test_duplicate_fingerprint_does_not_fold_case_or_semantics() -> None:
    base = candidate(value_json="ACME", value_type="string")
    changed_case = candidate(
        extraction_run_id=base.extraction_run_id,
        fact_id=base.fact_id,
        value_json="acme",
        value_type="string",
    )
    changed_unit = candidate(
        extraction_run_id=base.extraction_run_id,
        fact_id=base.fact_id,
        value_json={"amount": "0.1", "unit": "ratio"},
        value_type="object",
    )

    assert (
        duplicate_grouping_service.build_duplicate_fingerprint(base).sha256_hex
        != duplicate_grouping_service.build_duplicate_fingerprint(changed_case).sha256_hex
    )
    assert (
        duplicate_grouping_service.build_duplicate_fingerprint(base).sha256_hex
        != duplicate_grouping_service.build_duplicate_fingerprint(changed_unit).sha256_hex
    )


def test_duplicate_fingerprint_detects_digest_collision(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeDigest:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def hexdigest(self) -> str:
            return "f" * 64

    monkeypatch.setattr(duplicate_grouping_service.hashlib, "sha256", _FakeDigest)

    digest_map: dict[str, bytes] = {}
    first = candidate(value_json={"value": 1})
    second = candidate(
        extraction_run_id=first.extraction_run_id,
        fact_id=first.fact_id,
        value_json={"value": 2},
    )

    duplicate_grouping_service.build_duplicate_fingerprint(first, digest_bytes_by_hash=digest_map)
    with pytest.raises(duplicate_grouping_service.CrossBatchDuplicateGroupingInvariantError):
        duplicate_grouping_service.build_duplicate_fingerprint(second, digest_bytes_by_hash=digest_map)


def test_build_duplicate_grouping_write_plan_creates_cross_batch_groups_only() -> None:
    run_id = uuid.uuid4()
    fact_id = uuid.uuid4()
    batch_one = uuid.uuid4()
    batch_two = uuid.uuid4()
    batch_three = uuid.uuid4()
    same_value = {"amount": "10", "currency": "CNY"}

    plan = duplicate_grouping_service.build_duplicate_grouping_write_plan(
        [
            candidate(
                extraction_run_id=run_id,
                fact_id=fact_id,
                source_batch_id=batch_one,
                value_json=same_value,
            ),
            candidate(
                extraction_run_id=run_id,
                fact_id=fact_id,
                source_batch_id=batch_two,
                value_json=same_value,
            ),
            candidate(
                extraction_run_id=run_id,
                fact_id=fact_id,
                source_batch_id=batch_two,
                value_json=same_value,
            ),
            candidate(
                extraction_run_id=run_id,
                fact_id=fact_id,
                source_batch_id=batch_three,
                value_json={"amount": "12", "currency": "CNY"},
            ),
        ]
    )

    assert plan.input_fact_value_count == 4
    assert plan.duplicate_group_count == 1
    assert plan.duplicate_member_count == 3
    assert plan.groups[0].distinct_batch_count == 2
    assert len(plan.groups[0].members) == 3


def test_build_duplicate_grouping_write_plan_keeps_zero_result_application_stable() -> None:
    run_id = uuid.uuid4()
    same_batch = uuid.uuid4()
    plan = duplicate_grouping_service.build_duplicate_grouping_write_plan(
        [
            candidate(extraction_run_id=run_id, source_batch_id=same_batch, value_json="A", value_type="string"),
            candidate(extraction_run_id=run_id, source_batch_id=same_batch, value_json="A", value_type="string"),
        ]
    )

    assert plan.input_fact_value_count == 2
    assert plan.duplicate_group_count == 0
    assert plan.duplicate_member_count == 0
    assert len(plan.input_manifest_hash) == 64
    assert len(plan.result_manifest_hash) == 64


def test_build_duplicate_grouping_write_plan_separates_different_fact_or_value() -> None:
    run_id = uuid.uuid4()
    batch_one = uuid.uuid4()
    batch_two = uuid.uuid4()
    shared_fact = uuid.uuid4()
    other_fact = uuid.uuid4()
    plan = duplicate_grouping_service.build_duplicate_grouping_write_plan(
        [
            candidate(extraction_run_id=run_id, fact_id=shared_fact, source_batch_id=batch_one, value_json="A", value_type="string"),
            candidate(extraction_run_id=run_id, fact_id=other_fact, source_batch_id=batch_two, value_json="A", value_type="string"),
            candidate(extraction_run_id=run_id, fact_id=shared_fact, source_batch_id=batch_two, value_json="B", value_type="string"),
        ]
    )

    assert plan.duplicate_group_count == 0
    assert plan.duplicate_member_count == 0


def test_build_duplicate_group_evidence_union_deduplicates_by_evidence_id() -> None:
    evidence_a = uuid.uuid4()
    evidence_b = uuid.uuid4()
    projections = (
        DuplicateGroupEvidenceProjection(
            group_id=uuid.uuid4(),
            duplicate_key_hash="a" * 64,
            fact_value_id=uuid.uuid4(),
            source_batch_id=uuid.uuid4(),
            evidence_link_ids=(uuid.uuid4(), uuid.uuid4()),
            evidence_ids=(evidence_a, evidence_b),
        ),
        DuplicateGroupEvidenceProjection(
            group_id=uuid.uuid4(),
            duplicate_key_hash="a" * 64,
            fact_value_id=uuid.uuid4(),
            source_batch_id=uuid.uuid4(),
            evidence_link_ids=(uuid.uuid4(),),
            evidence_ids=(evidence_b,),
        ),
    )

    assert duplicate_grouping_service.build_duplicate_group_evidence_union(projections) == (
        evidence_a,
        evidence_b,
    )


@dataclass(frozen=True, slots=True)
class FakeApplicationLedger:
    id: uuid.UUID
    extraction_run_id: uuid.UUID
    algorithm_version: str
    input_manifest_hash: str
    result_manifest_hash: str
    input_fact_value_count: int
    duplicate_group_count: int
    duplicate_member_count: int
    created_at: datetime


def make_run_state(extraction_run_id: uuid.UUID) -> duplicate_grouping_repository.DuplicateGroupingRunState:
    return duplicate_grouping_repository.DuplicateGroupingRunState(
        extraction_run_id=extraction_run_id,
        project_id=uuid.uuid4(),
        extraction_run_status="completed",
        extraction_run_outcome="success",
        latest_terminal_orchestration_status="completed",
        active_orchestration_count=0,
    )


def test_ensure_cross_batch_duplicate_grouping_creates_new_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = uuid.uuid4()
    first_candidate = candidate(extraction_run_id=run_id, source_batch_id=uuid.uuid4())
    second_candidate = candidate(
        extraction_run_id=run_id,
        fact_id=first_candidate.fact_id,
        source_batch_id=uuid.uuid4(),
        value_json=first_candidate.value_json,
        value_type=first_candidate.value_type,
        referenced_entity_id=first_candidate.referenced_entity_id,
    )
    sessions = [FakeSession(), FakeSession()]
    session_factory = SessionFactory(sessions)

    async def fake_state(_session, *, extraction_run_id):
        return make_run_state(extraction_run_id)

    async def fake_count(_session, *, extraction_run_id):
        assert extraction_run_id == run_id
        return 2

    async def fake_candidates(_session, *, extraction_run_id):
        assert extraction_run_id == run_id
        return (first_candidate, second_candidate)

    async def fake_existing_for_update(_session, *, extraction_run_id, algorithm_version):
        assert extraction_run_id == run_id
        assert algorithm_version == CROSS_BATCH_DUPLICATE_ALGORITHM_VERSION
        return None

    async def fake_create_application(_session, application):
        return application

    async def fake_create_groups(_session, groups):
        return groups

    async def fake_create_members(_session, members):
        return members

    monkeypatch.setattr(duplicate_grouping_repository, "get_duplicate_grouping_run_state", fake_state)
    monkeypatch.setattr(duplicate_grouping_repository, "count_ai_fact_values", fake_count)
    monkeypatch.setattr(duplicate_grouping_repository, "list_duplicate_candidates", fake_candidates)
    monkeypatch.setattr(duplicate_grouping_repository, "get_grouping_application_for_update", fake_existing_for_update)
    monkeypatch.setattr(duplicate_grouping_repository, "create_grouping_application", fake_create_application)
    monkeypatch.setattr(duplicate_grouping_repository, "create_duplicate_groups", fake_create_groups)
    monkeypatch.setattr(duplicate_grouping_repository, "create_duplicate_group_members", fake_create_members)

    result = run_async(
        duplicate_grouping_service.ensure_cross_batch_duplicate_grouping(
            session_factory,
            extraction_run_id=run_id,
        )
    )

    assert result.created_new is True
    assert result.algorithm_version == CROSS_BATCH_DUPLICATE_ALGORITHM_VERSION
    assert result.duplicate_group_count == 1
    assert result.duplicate_member_count == 2
    assert session_factory.created_sessions[1].commit_count == 1


def test_ensure_cross_batch_duplicate_grouping_returns_existing_immutable_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = uuid.uuid4()
    first_candidate = candidate(extraction_run_id=run_id, source_batch_id=uuid.uuid4())
    second_candidate = candidate(
        extraction_run_id=run_id,
        fact_id=first_candidate.fact_id,
        source_batch_id=uuid.uuid4(),
        value_json=first_candidate.value_json,
        value_type=first_candidate.value_type,
        referenced_entity_id=first_candidate.referenced_entity_id,
    )
    plan = duplicate_grouping_service.build_duplicate_grouping_write_plan((first_candidate, second_candidate))
    existing = FakeApplicationLedger(
        id=uuid.uuid4(),
        extraction_run_id=run_id,
        algorithm_version=plan.algorithm_version,
        input_manifest_hash=plan.input_manifest_hash,
        result_manifest_hash=plan.result_manifest_hash,
        input_fact_value_count=plan.input_fact_value_count,
        duplicate_group_count=plan.duplicate_group_count,
        duplicate_member_count=plan.duplicate_member_count,
        created_at=datetime.now(timezone.utc),
    )
    sessions = [FakeSession(), FakeSession()]
    session_factory = SessionFactory(sessions)

    async def fake_state(_session, *, extraction_run_id):
        return make_run_state(extraction_run_id)

    async def fake_count(_session, *, extraction_run_id):
        return 2

    async def fake_candidates(_session, *, extraction_run_id):
        return (first_candidate, second_candidate)

    async def fake_existing_for_update(_session, *, extraction_run_id, algorithm_version):
        return type(
            "ExistingApplication",
            (),
            {
                "id": existing.id,
                "extraction_run_id": existing.extraction_run_id,
                "algorithm_version": existing.algorithm_version,
                "input_manifest_hash": existing.input_manifest_hash,
                "result_manifest_hash": existing.result_manifest_hash,
                "input_fact_value_count": existing.input_fact_value_count,
                "duplicate_group_count": existing.duplicate_group_count,
                "duplicate_member_count": existing.duplicate_member_count,
                "created_at": existing.created_at,
            },
        )()

    monkeypatch.setattr(duplicate_grouping_repository, "get_duplicate_grouping_run_state", fake_state)
    monkeypatch.setattr(duplicate_grouping_repository, "count_ai_fact_values", fake_count)
    monkeypatch.setattr(duplicate_grouping_repository, "list_duplicate_candidates", fake_candidates)
    monkeypatch.setattr(duplicate_grouping_repository, "get_grouping_application_for_update", fake_existing_for_update)

    result = run_async(
        duplicate_grouping_service.ensure_cross_batch_duplicate_grouping(
            session_factory,
            extraction_run_id=run_id,
        )
    )

    assert result.created_new is False
    assert result.grouping_application_id == existing.id
    assert session_factory.created_sessions[1].commit_count == 1


def test_ensure_cross_batch_duplicate_grouping_fails_closed_on_manifest_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = uuid.uuid4()
    first_candidate = candidate(extraction_run_id=run_id, source_batch_id=uuid.uuid4())
    second_candidate = candidate(
        extraction_run_id=run_id,
        fact_id=first_candidate.fact_id,
        source_batch_id=uuid.uuid4(),
        value_json=first_candidate.value_json,
        value_type=first_candidate.value_type,
        referenced_entity_id=first_candidate.referenced_entity_id,
    )
    sessions = [FakeSession(), FakeSession()]
    session_factory = SessionFactory(sessions)

    async def fake_state(_session, *, extraction_run_id):
        return make_run_state(extraction_run_id)

    async def fake_count(_session, *, extraction_run_id):
        return 2

    async def fake_candidates(_session, *, extraction_run_id):
        return (first_candidate, second_candidate)

    async def fake_existing_for_update(_session, *, extraction_run_id, algorithm_version):
        return type(
            "ExistingApplication",
            (),
            {
                "id": uuid.uuid4(),
                "extraction_run_id": run_id,
                "algorithm_version": algorithm_version,
                "input_manifest_hash": "a" * 64,
                "result_manifest_hash": "b" * 64,
                "input_fact_value_count": 999,
                "duplicate_group_count": 0,
                "duplicate_member_count": 0,
                "created_at": datetime.now(timezone.utc),
            },
        )()

    monkeypatch.setattr(duplicate_grouping_repository, "get_duplicate_grouping_run_state", fake_state)
    monkeypatch.setattr(duplicate_grouping_repository, "count_ai_fact_values", fake_count)
    monkeypatch.setattr(duplicate_grouping_repository, "list_duplicate_candidates", fake_candidates)
    monkeypatch.setattr(duplicate_grouping_repository, "get_grouping_application_for_update", fake_existing_for_update)

    with pytest.raises(duplicate_grouping_service.CrossBatchDuplicateGroupingInvariantError):
        run_async(
            duplicate_grouping_service.ensure_cross_batch_duplicate_grouping(
                session_factory,
                extraction_run_id=run_id,
            )
        )


def test_ensure_cross_batch_duplicate_grouping_handles_target_constraint_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = uuid.uuid4()
    first_candidate = candidate(extraction_run_id=run_id, source_batch_id=uuid.uuid4())
    second_candidate = candidate(
        extraction_run_id=run_id,
        fact_id=first_candidate.fact_id,
        source_batch_id=uuid.uuid4(),
        value_json=first_candidate.value_json,
        value_type=first_candidate.value_type,
        referenced_entity_id=first_candidate.referenced_entity_id,
    )
    plan = duplicate_grouping_service.build_duplicate_grouping_write_plan((first_candidate, second_candidate))
    existing = FakeApplicationLedger(
        id=uuid.uuid4(),
        extraction_run_id=run_id,
        algorithm_version=plan.algorithm_version,
        input_manifest_hash=plan.input_manifest_hash,
        result_manifest_hash=plan.result_manifest_hash,
        input_fact_value_count=plan.input_fact_value_count,
        duplicate_group_count=plan.duplicate_group_count,
        duplicate_member_count=plan.duplicate_member_count,
        created_at=datetime.now(timezone.utc),
    )
    sessions = [FakeSession(), FakeSession(), FakeSession()]
    session_factory = SessionFactory(sessions)

    async def fake_state(_session, *, extraction_run_id):
        return make_run_state(extraction_run_id)

    async def fake_count(_session, *, extraction_run_id):
        return 2

    async def fake_candidates(_session, *, extraction_run_id):
        return (first_candidate, second_candidate)

    async def fake_existing_for_update(_session, *, extraction_run_id, algorithm_version):
        return None

    async def fake_create_application(_session, application):
        raise make_integrity_error("uq_dupgrp_app_run_alg")

    async def fake_get_existing(_session, *, extraction_run_id, algorithm_version):
        return existing

    monkeypatch.setattr(duplicate_grouping_repository, "get_duplicate_grouping_run_state", fake_state)
    monkeypatch.setattr(duplicate_grouping_repository, "count_ai_fact_values", fake_count)
    monkeypatch.setattr(duplicate_grouping_repository, "list_duplicate_candidates", fake_candidates)
    monkeypatch.setattr(duplicate_grouping_repository, "get_grouping_application_for_update", fake_existing_for_update)
    monkeypatch.setattr(duplicate_grouping_repository, "create_grouping_application", fake_create_application)
    monkeypatch.setattr(duplicate_grouping_repository, "get_grouping_application_ledger", fake_get_existing)

    result = run_async(
        duplicate_grouping_service.ensure_cross_batch_duplicate_grouping(
            session_factory,
            extraction_run_id=run_id,
        )
    )

    assert result.created_new is False
    assert result.grouping_application_id == existing.id
    assert session_factory.created_sessions[1].rollback_count == 1


def test_ensure_cross_batch_duplicate_grouping_does_not_swallow_unknown_integrity_error(monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = uuid.uuid4()
    first_candidate = candidate(extraction_run_id=run_id, source_batch_id=uuid.uuid4())
    second_candidate = candidate(
        extraction_run_id=run_id,
        fact_id=first_candidate.fact_id,
        source_batch_id=uuid.uuid4(),
        value_json=first_candidate.value_json,
        value_type=first_candidate.value_type,
        referenced_entity_id=first_candidate.referenced_entity_id,
    )
    sessions = [FakeSession(), FakeSession()]
    session_factory = SessionFactory(sessions)

    async def fake_state(_session, *, extraction_run_id):
        return make_run_state(extraction_run_id)

    async def fake_count(_session, *, extraction_run_id):
        return 2

    async def fake_candidates(_session, *, extraction_run_id):
        return (first_candidate, second_candidate)

    async def fake_existing_for_update(_session, *, extraction_run_id, algorithm_version):
        return None

    async def fake_create_application(_session, application):
        raise make_integrity_error("uq_other_constraint")

    monkeypatch.setattr(duplicate_grouping_repository, "get_duplicate_grouping_run_state", fake_state)
    monkeypatch.setattr(duplicate_grouping_repository, "count_ai_fact_values", fake_count)
    monkeypatch.setattr(duplicate_grouping_repository, "list_duplicate_candidates", fake_candidates)
    monkeypatch.setattr(duplicate_grouping_repository, "get_grouping_application_for_update", fake_existing_for_update)
    monkeypatch.setattr(duplicate_grouping_repository, "create_grouping_application", fake_create_application)

    with pytest.raises(IntegrityError):
        run_async(
            duplicate_grouping_service.ensure_cross_batch_duplicate_grouping(
                session_factory,
                extraction_run_id=run_id,
            )
        )


def test_ensure_cross_batch_duplicate_grouping_rolls_back_on_partial_write_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = uuid.uuid4()
    first_candidate = candidate(extraction_run_id=run_id, source_batch_id=uuid.uuid4())
    second_candidate = candidate(
        extraction_run_id=run_id,
        fact_id=first_candidate.fact_id,
        source_batch_id=uuid.uuid4(),
        value_json=first_candidate.value_json,
        value_type=first_candidate.value_type,
        referenced_entity_id=first_candidate.referenced_entity_id,
    )
    sessions = [FakeSession(), FakeSession()]
    session_factory = SessionFactory(sessions)

    async def fake_state(_session, *, extraction_run_id):
        return make_run_state(extraction_run_id)

    async def fake_count(_session, *, extraction_run_id):
        return 2

    async def fake_candidates(_session, *, extraction_run_id):
        return (first_candidate, second_candidate)

    async def fake_existing_for_update(_session, *, extraction_run_id, algorithm_version):
        return None

    async def fake_create_application(_session, application):
        return application

    async def fake_create_groups(_session, groups):
        return groups

    async def fake_create_members(_session, members):
        raise RuntimeError("boom")

    monkeypatch.setattr(duplicate_grouping_repository, "get_duplicate_grouping_run_state", fake_state)
    monkeypatch.setattr(duplicate_grouping_repository, "count_ai_fact_values", fake_count)
    monkeypatch.setattr(duplicate_grouping_repository, "list_duplicate_candidates", fake_candidates)
    monkeypatch.setattr(duplicate_grouping_repository, "get_grouping_application_for_update", fake_existing_for_update)
    monkeypatch.setattr(duplicate_grouping_repository, "create_grouping_application", fake_create_application)
    monkeypatch.setattr(duplicate_grouping_repository, "create_duplicate_groups", fake_create_groups)
    monkeypatch.setattr(duplicate_grouping_repository, "create_duplicate_group_members", fake_create_members)

    with pytest.raises(RuntimeError):
        run_async(
            duplicate_grouping_service.ensure_cross_batch_duplicate_grouping(
                session_factory,
                extraction_run_id=run_id,
            )
        )

    assert session_factory.created_sessions[1].rollback_count == 1
    assert session_factory.created_sessions[1].commit_count == 0


def test_ensure_cross_batch_duplicate_grouping_rejects_not_ready_run_without_leaking_values(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    run_id = uuid.uuid4()
    sentinel = "SENTINEL_DUP_VALUE"
    session_factory = SessionFactory([FakeSession()])

    async def fake_state(_session, *, extraction_run_id):
        return duplicate_grouping_repository.DuplicateGroupingRunState(
            extraction_run_id=extraction_run_id,
            project_id=uuid.uuid4(),
            extraction_run_status="completed",
            extraction_run_outcome="success",
            latest_terminal_orchestration_status="running",
            active_orchestration_count=1,
        )

    monkeypatch.setattr(duplicate_grouping_repository, "get_duplicate_grouping_run_state", fake_state)

    with pytest.raises(duplicate_grouping_service.CrossBatchDuplicateGroupingStateError) as exc_info:
        run_async(
            duplicate_grouping_service.ensure_cross_batch_duplicate_grouping(
                session_factory,
                extraction_run_id=run_id,
            )
        )

    assert sentinel not in str(exc_info.value)
    assert sentinel not in caplog.text


def test_duplicate_candidate_query_compiles_and_reads_evidence_order() -> None:
    class _FakeResult:
        def __init__(self, rows):
            self._rows = rows

        def all(self):
            return self._rows

    class _Session:
        def __init__(self, rows):
            self.rows = rows
            self.statements = []

        async def execute(self, statement):
            self.statements.append(statement)
            return _FakeResult(self.rows)

    run_id = uuid.uuid4()
    fact_value_id = uuid.uuid4()
    rows = [
        type(
            "Row",
            (),
            {
                "fact_value_id": fact_value_id,
                "fact_id": uuid.uuid4(),
                "extraction_run_id": run_id,
                "source_batch_id": uuid.uuid4(),
                "value_type": "string",
                "value_json": "A",
                "referenced_entity_id": None,
                "evidence_link_id": uuid.UUID("00000000-0000-0000-0000-000000000002"),
            },
        )(),
        type(
            "Row",
            (),
            {
                "fact_value_id": fact_value_id,
                "fact_id": uuid.uuid4(),
                "extraction_run_id": run_id,
                "source_batch_id": uuid.uuid4(),
                "value_type": "string",
                "value_json": "A",
                "referenced_entity_id": None,
                "evidence_link_id": uuid.UUID("00000000-0000-0000-0000-000000000003"),
            },
        )(),
    ]
    session = _Session(rows)

    candidates = run_async(
        duplicate_grouping_repository.list_duplicate_candidates(
            session,
            extraction_run_id=run_id,
        )
    )
    sql = str(session.statements[0].compile(dialect=postgresql.dialect()))

    assert len(candidates) == 1
    assert candidates[0].evidence_link_ids == (
        uuid.UUID("00000000-0000-0000-0000-000000000002"),
        uuid.UUID("00000000-0000-0000-0000-000000000003"),
    )
    assert "fact_extraction_batch_applications" in sql
    assert "fact_extraction_orch_batches" in sql
    assert "ORDER BY fact_values.id ASC" in sql


def test_duplicate_group_evidence_projection_reads_member_evidence_order() -> None:
    class _FakeResult:
        def __init__(self, rows):
            self._rows = rows

        def all(self):
            return self._rows

    class _Session:
        def __init__(self, rows):
            self.rows = rows

        async def execute(self, _statement):
            return _FakeResult(self.rows)

    group_id = uuid.uuid4()
    fact_value_id = uuid.uuid4()
    rows = [
        type(
            "Row",
            (),
            {
                "group_id": group_id,
                "duplicate_key_hash": "a" * 64,
                "fact_value_id": fact_value_id,
                "source_batch_id": uuid.uuid4(),
                "evidence_link_id": uuid.UUID("00000000-0000-0000-0000-000000000002"),
                "evidence_id": uuid.UUID("00000000-0000-0000-0000-000000000010"),
            },
        )(),
        type(
            "Row",
            (),
            {
                "group_id": group_id,
                "duplicate_key_hash": "a" * 64,
                "fact_value_id": fact_value_id,
                "source_batch_id": uuid.uuid4(),
                "evidence_link_id": uuid.UUID("00000000-0000-0000-0000-000000000003"),
                "evidence_id": uuid.UUID("00000000-0000-0000-0000-000000000011"),
            },
        )(),
    ]
    projections = run_async(
        duplicate_grouping_repository.list_duplicate_group_evidence_projections(
            _Session(rows),
            group_id=group_id,
        )
    )

    assert projections == (
        DuplicateGroupEvidenceProjection(
            group_id=group_id,
            duplicate_key_hash="a" * 64,
            fact_value_id=fact_value_id,
            source_batch_id=rows[0].source_batch_id,
            evidence_link_ids=(
                uuid.UUID("00000000-0000-0000-0000-000000000002"),
                uuid.UUID("00000000-0000-0000-0000-000000000003"),
            ),
            evidence_ids=(
                uuid.UUID("00000000-0000-0000-0000-000000000010"),
                uuid.UUID("00000000-0000-0000-0000-000000000011"),
            ),
        ),
    )
