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
    FactValueConsistencyCandidate,
    FactValueConsistencyCandidateApplication,
    FactValueConsistencyCandidateMember,
    FactValueDuplicateGroup,
    FactValueDuplicateGroupMember,
    FactValueDuplicateGroupingApplication,
    normalize_duplicate_grouping_algorithm_version as normalize_model_algorithm_version,
)
from app.repositories import fact_value_duplicate_grouping as duplicate_grouping_repository
from app.schemas.fact_value_duplicate_grouping import (
    CROSS_BATCH_DUPLICATE_ALGORITHM_VERSION,
    CROSS_BATCH_MULTI_VALUE_CANDIDATE_ALGORITHM_VERSION,
    DuplicateCandidate,
    DuplicateGroupingApplicationLedger,
    DuplicateGroupEvidenceProjection,
    FactValueConsistencyCandidateApplicationLedger,
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
    orchestration_id: uuid.UUID | None = None,
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
        orchestration_id=orchestration_id or uuid.uuid4(),
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


@dataclass(frozen=True, slots=True)
class FakeApplicationLedger:
    id: uuid.UUID
    orchestration_id: uuid.UUID
    extraction_run_id: uuid.UUID
    algorithm_version: str
    input_manifest_hash: str
    result_manifest_hash: str
    input_fact_value_count: int
    duplicate_group_count: int
    duplicate_member_count: int
    created_at: datetime


def make_duplicate_grouping_application_ledger(
    candidates: tuple[DuplicateCandidate, ...],
    *,
    grouping_application_id: uuid.UUID | None = None,
    algorithm_version: str = CROSS_BATCH_DUPLICATE_ALGORITHM_VERSION,
) -> DuplicateGroupingApplicationLedger:
    plan = duplicate_grouping_service.build_duplicate_grouping_write_plan(
        candidates,
        algorithm_version=algorithm_version,
    )
    return DuplicateGroupingApplicationLedger(
        id=grouping_application_id or uuid.uuid4(),
        orchestration_id=candidates[0].orchestration_id if candidates else uuid.uuid4(),
        extraction_run_id=candidates[0].extraction_run_id if candidates else uuid.uuid4(),
        algorithm_version=plan.algorithm_version,
        input_manifest_hash=plan.input_manifest_hash,
        result_manifest_hash=plan.result_manifest_hash,
        input_fact_value_count=plan.input_fact_value_count,
        duplicate_group_count=plan.duplicate_group_count,
        duplicate_member_count=plan.duplicate_member_count,
        created_at=datetime.now(timezone.utc),
    )


def make_orchestration_state(
    orchestration_id: uuid.UUID,
    *,
    extraction_run_id: uuid.UUID | None = None,
    orchestration_status: str = "completed",
) -> duplicate_grouping_repository.DuplicateGroupingOrchestrationState:
    return duplicate_grouping_repository.DuplicateGroupingOrchestrationState(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id or uuid.uuid4(),
        project_id=uuid.uuid4(),
        extraction_run_status="completed",
        extraction_run_outcome="success",
        orchestration_status=orchestration_status,
    )


async def return_second_arg(*args, **kwargs):
    return args[1]


def test_duplicate_grouping_tables_are_registered() -> None:
    assert FactValueDuplicateGroupingApplication.__table__.name == "fact_value_duplicate_grouping_applications"
    assert FactValueDuplicateGroup.__table__.name == "fact_value_duplicate_groups"
    assert FactValueDuplicateGroupMember.__table__.name == "fact_value_duplicate_group_members"


def test_duplicate_fingerprint_is_stable_for_same_orchestration_fact_and_semantic_value() -> None:
    run_id = uuid.uuid4()
    orchestration_id = uuid.uuid4()
    fact_id = uuid.uuid4()
    left = candidate(
        orchestration_id=orchestration_id,
        extraction_run_id=run_id,
        fact_id=fact_id,
        source_batch_id=uuid.uuid4(),
        evidence_link_ids=(uuid.uuid4(), uuid.uuid4()),
        value_json={"b": 2, "a": 1},
    )
    right = candidate(
        orchestration_id=orchestration_id,
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


def test_duplicate_fingerprint_v2_changes_when_orchestration_changes() -> None:
    extraction_run_id = uuid.uuid4()
    fact_id = uuid.uuid4()
    left = candidate(
        orchestration_id=uuid.uuid4(),
        extraction_run_id=extraction_run_id,
        fact_id=fact_id,
    )
    right = candidate(
        orchestration_id=uuid.uuid4(),
        extraction_run_id=extraction_run_id,
        fact_id=fact_id,
        value_json=left.value_json,
        value_type=left.value_type,
        referenced_entity_id=left.referenced_entity_id,
    )

    assert (
        duplicate_grouping_service.build_duplicate_fingerprint(left).sha256_hex
        != duplicate_grouping_service.build_duplicate_fingerprint(right).sha256_hex
    )


def test_duplicate_fingerprint_normalizes_algorithm_version_before_hashing() -> None:
    base = candidate()
    same = candidate(
        orchestration_id=base.orchestration_id,
        extraction_run_id=base.extraction_run_id,
        fact_id=base.fact_id,
        source_batch_id=uuid.uuid4(),
        value_json=base.value_json,
        value_type=base.value_type,
        referenced_entity_id=base.referenced_entity_id,
    )

    spaced = duplicate_grouping_service.build_duplicate_fingerprint(
        base,
        algorithm_version="  cross_batch_exact_v2  ",
    )
    normalized = duplicate_grouping_service.build_duplicate_fingerprint(
        same,
        algorithm_version="cross_batch_exact_v2",
    )

    assert spaced.sha256_hex == normalized.sha256_hex


def test_duplicate_grouping_algorithm_version_normalizes_and_rejects_invalid_values() -> None:
    assert duplicate_grouping_service.normalize_duplicate_grouping_algorithm_version(
        "  cross_batch_exact_v2  "
    ) == "cross_batch_exact_v2"
    assert normalize_model_algorithm_version("  cross_batch_exact_v2  ") == "cross_batch_exact_v2"

    with pytest.raises(duplicate_grouping_service.CrossBatchDuplicateGroupingError, match="invalid_algorithm_version"):
        duplicate_grouping_service.normalize_duplicate_grouping_algorithm_version("   ")

    with pytest.raises(duplicate_grouping_service.CrossBatchDuplicateGroupingError, match="invalid_algorithm_version"):
        duplicate_grouping_service.normalize_duplicate_grouping_algorithm_version("x" * 65)


def test_canonical_json_rejects_nfc_key_collision() -> None:
    with pytest.raises(
        duplicate_grouping_service.CrossBatchDuplicateGroupingInvariantError,
        match="cross_batch_duplicate_grouping_nfc_key_collision",
    ):
        duplicate_grouping_service._canonical_json_bytes(
            {
                "e\u0301": "left",
                "é": "right",
            }
        )


def test_duplicate_fingerprint_supports_decimal_uuid_date_time_and_enum() -> None:
    run_id = uuid.uuid4()
    orchestration_id = uuid.uuid4()
    fact_id = uuid.uuid4()
    entity_id = uuid.uuid4()
    left = candidate(
        orchestration_id=orchestration_id,
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
        orchestration_id=orchestration_id,
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
        orchestration_id=base.orchestration_id,
        extraction_run_id=base.extraction_run_id,
        fact_id=base.fact_id,
        value_json="acme",
        value_type="string",
    )
    changed_unit = candidate(
        orchestration_id=base.orchestration_id,
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
        orchestration_id=first.orchestration_id,
        extraction_run_id=first.extraction_run_id,
        fact_id=first.fact_id,
        value_json={"value": 2},
    )

    duplicate_grouping_service.build_duplicate_fingerprint(first, digest_bytes_by_hash=digest_map)
    with pytest.raises(duplicate_grouping_service.CrossBatchDuplicateGroupingInvariantError):
        duplicate_grouping_service.build_duplicate_fingerprint(second, digest_bytes_by_hash=digest_map)


def test_build_duplicate_grouping_write_plan_creates_cross_batch_groups_only() -> None:
    orchestration_id = uuid.uuid4()
    run_id = uuid.uuid4()
    fact_id = uuid.uuid4()
    batch_one = uuid.uuid4()
    batch_two = uuid.uuid4()
    batch_three = uuid.uuid4()
    same_value = {"amount": "10", "currency": "CNY"}

    plan = duplicate_grouping_service.build_duplicate_grouping_write_plan(
        [
            candidate(
                orchestration_id=orchestration_id,
                extraction_run_id=run_id,
                fact_id=fact_id,
                source_batch_id=batch_one,
                value_json=same_value,
            ),
            candidate(
                orchestration_id=orchestration_id,
                extraction_run_id=run_id,
                fact_id=fact_id,
                source_batch_id=batch_two,
                value_json=same_value,
            ),
            candidate(
                orchestration_id=orchestration_id,
                extraction_run_id=run_id,
                fact_id=fact_id,
                source_batch_id=batch_two,
                value_json=same_value,
            ),
            candidate(
                orchestration_id=orchestration_id,
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
    orchestration_id = uuid.uuid4()
    run_id = uuid.uuid4()
    same_batch = uuid.uuid4()
    plan = duplicate_grouping_service.build_duplicate_grouping_write_plan(
        [
            candidate(
                orchestration_id=orchestration_id,
                extraction_run_id=run_id,
                source_batch_id=same_batch,
                value_json="A",
                value_type="string",
            ),
            candidate(
                orchestration_id=orchestration_id,
                extraction_run_id=run_id,
                source_batch_id=same_batch,
                value_json="A",
                value_type="string",
            ),
        ]
    )

    assert plan.input_fact_value_count == 2
    assert plan.duplicate_group_count == 0
    assert plan.duplicate_member_count == 0
    assert len(plan.input_manifest_hash) == 64
    assert len(plan.result_manifest_hash) == 64


def test_build_duplicate_grouping_write_plan_rejects_cross_orchestration_candidates() -> None:
    run_id = uuid.uuid4()
    with pytest.raises(duplicate_grouping_service.CrossBatchDuplicateGroupingInvariantError):
        duplicate_grouping_service.build_duplicate_grouping_write_plan(
            [
                candidate(orchestration_id=uuid.uuid4(), extraction_run_id=run_id, value_json="A", value_type="string"),
                candidate(orchestration_id=uuid.uuid4(), extraction_run_id=run_id, value_json="A", value_type="string"),
            ]
        )


def test_build_duplicate_grouping_write_plan_rejects_duplicate_fact_value_id() -> None:
    duplicate_fact_value_id = uuid.uuid4()
    orchestration_id = uuid.uuid4()
    extraction_run_id = uuid.uuid4()
    with pytest.raises(
        duplicate_grouping_service.CrossBatchDuplicateGroupingInvariantError,
        match="cross_batch_duplicate_grouping_duplicate_fact_value_id",
    ):
        duplicate_grouping_service.build_duplicate_grouping_write_plan(
            [
                candidate(
                    orchestration_id=orchestration_id,
                    extraction_run_id=extraction_run_id,
                    fact_value_id=duplicate_fact_value_id,
                    value_json="A",
                    value_type="string",
                ),
                candidate(
                    orchestration_id=orchestration_id,
                    extraction_run_id=extraction_run_id,
                    fact_value_id=duplicate_fact_value_id,
                    value_json="B",
                    value_type="string",
                ),
            ]
        )


def test_build_duplicate_grouping_write_plan_rejects_duplicate_evidence_link_id() -> None:
    duplicate_evidence_link_id = uuid.uuid4()
    with pytest.raises(
        duplicate_grouping_service.CrossBatchDuplicateGroupingInvariantError,
        match="cross_batch_duplicate_grouping_duplicate_evidence_link_id",
    ):
        duplicate_grouping_service.build_duplicate_grouping_write_plan(
            [
                candidate(
                    evidence_link_ids=(duplicate_evidence_link_id, duplicate_evidence_link_id),
                ),
            ]
        )


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


def test_build_duplicate_group_evidence_union_preserves_first_seen_projection_order() -> None:
    evidence_first = uuid.UUID("ffffffff-ffff-ffff-ffff-fffffffffff0")
    evidence_second = uuid.UUID("00000000-0000-0000-0000-000000000001")
    evidence_third = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    projections = (
        DuplicateGroupEvidenceProjection(
            group_id=uuid.uuid4(),
            duplicate_key_hash="a" * 64,
            fact_value_id=uuid.uuid4(),
            source_batch_id=uuid.uuid4(),
            evidence_link_ids=(uuid.uuid4(),),
            evidence_ids=(evidence_first, evidence_second),
        ),
        DuplicateGroupEvidenceProjection(
            group_id=uuid.uuid4(),
            duplicate_key_hash="b" * 64,
            fact_value_id=uuid.uuid4(),
            source_batch_id=uuid.uuid4(),
            evidence_link_ids=(uuid.uuid4(),),
            evidence_ids=(evidence_second, evidence_third),
        ),
    )

    assert duplicate_grouping_service.build_duplicate_group_evidence_union(projections) == (
        evidence_first,
        evidence_second,
        evidence_third,
    )


def test_same_extraction_run_different_orchestrations_create_independent_v2_ledgers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction_run_id = uuid.uuid4()
    orchestration_a = uuid.uuid4()
    orchestration_b = uuid.uuid4()
    sessions = [FakeSession(), FakeSession(), FakeSession(), FakeSession()]
    session_factory = SessionFactory(sessions)
    created_applications: list[FactValueDuplicateGroupingApplication] = []

    candidate_a1 = candidate(
        orchestration_id=orchestration_a,
        extraction_run_id=extraction_run_id,
        source_batch_id=uuid.uuid4(),
    )
    candidate_a2 = candidate(
        orchestration_id=orchestration_a,
        extraction_run_id=extraction_run_id,
        fact_id=candidate_a1.fact_id,
        source_batch_id=uuid.uuid4(),
        value_json=candidate_a1.value_json,
        value_type=candidate_a1.value_type,
        referenced_entity_id=candidate_a1.referenced_entity_id,
    )
    candidate_b1 = candidate(
        orchestration_id=orchestration_b,
        extraction_run_id=extraction_run_id,
        source_batch_id=uuid.uuid4(),
    )
    candidate_b2 = candidate(
        orchestration_id=orchestration_b,
        extraction_run_id=extraction_run_id,
        fact_id=candidate_b1.fact_id,
        source_batch_id=uuid.uuid4(),
        value_json=candidate_b1.value_json,
        value_type=candidate_b1.value_type,
        referenced_entity_id=candidate_b1.referenced_entity_id,
    )
    candidate_b3 = candidate(
        orchestration_id=orchestration_b,
        extraction_run_id=extraction_run_id,
        source_batch_id=uuid.uuid4(),
        value_json="new-value",
        value_type="string",
    )

    async def fake_state(_session, *, orchestration_id):
        if orchestration_id == orchestration_a:
            return make_orchestration_state(orchestration_a, extraction_run_id=extraction_run_id, orchestration_status="partial")
        return make_orchestration_state(orchestration_b, extraction_run_id=extraction_run_id, orchestration_status="completed")

    async def fake_invalid_bindings(_session, *, orchestration_id):
        return False

    async def fake_count(_session, *, orchestration_id):
        return 2 if orchestration_id == orchestration_a else 3

    async def fake_candidates(_session, *, orchestration_id):
        if orchestration_id == orchestration_a:
            return (candidate_a1, candidate_a2)
        return (candidate_b1, candidate_b2, candidate_b3)

    async def fake_existing_for_update(_session, *, orchestration_id, algorithm_version):
        assert algorithm_version == CROSS_BATCH_DUPLICATE_ALGORITHM_VERSION
        return None

    async def fake_create_application(_session, application):
        created_applications.append(application)
        return application

    monkeypatch.setattr(duplicate_grouping_repository, "get_duplicate_grouping_orchestration_state", fake_state)
    monkeypatch.setattr(duplicate_grouping_repository, "has_invalid_completed_batch_bindings", fake_invalid_bindings)
    monkeypatch.setattr(duplicate_grouping_repository, "count_duplicate_candidate_fact_values", fake_count)
    monkeypatch.setattr(duplicate_grouping_repository, "list_duplicate_candidates", fake_candidates)
    monkeypatch.setattr(duplicate_grouping_repository, "get_grouping_application_for_update", fake_existing_for_update)
    monkeypatch.setattr(duplicate_grouping_repository, "create_grouping_application", fake_create_application)
    monkeypatch.setattr(duplicate_grouping_repository, "create_duplicate_groups", return_second_arg)
    monkeypatch.setattr(duplicate_grouping_repository, "create_duplicate_group_members", return_second_arg)

    result_a = run_async(
        duplicate_grouping_service.ensure_cross_batch_duplicate_grouping(
            session_factory,
            orchestration_id=orchestration_a,
        )
    )
    result_b = run_async(
        duplicate_grouping_service.ensure_cross_batch_duplicate_grouping(
            session_factory,
            orchestration_id=orchestration_b,
        )
    )

    assert result_a.created_new is True
    assert result_b.created_new is True
    assert result_a.grouping_application_id != result_b.grouping_application_id
    assert all(app.algorithm_version == "cross_batch_exact_v2" for app in created_applications)
    assert {app.orchestration_id for app in created_applications} == {orchestration_a, orchestration_b}


def test_ensure_cross_batch_duplicate_grouping_returns_existing_immutable_ledger_for_same_orchestration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestration_id = uuid.uuid4()
    extraction_run_id = uuid.uuid4()
    first_candidate = candidate(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        source_batch_id=uuid.uuid4(),
    )
    second_candidate = candidate(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        fact_id=first_candidate.fact_id,
        source_batch_id=uuid.uuid4(),
        value_json=first_candidate.value_json,
        value_type=first_candidate.value_type,
        referenced_entity_id=first_candidate.referenced_entity_id,
    )
    plan = duplicate_grouping_service.build_duplicate_grouping_write_plan((first_candidate, second_candidate))
    existing = FakeApplicationLedger(
        id=uuid.uuid4(),
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        algorithm_version=plan.algorithm_version,
        input_manifest_hash=plan.input_manifest_hash,
        result_manifest_hash=plan.result_manifest_hash,
        input_fact_value_count=plan.input_fact_value_count,
        duplicate_group_count=plan.duplicate_group_count,
        duplicate_member_count=plan.duplicate_member_count,
        created_at=datetime.now(timezone.utc),
    )
    session_factory = SessionFactory([FakeSession(), FakeSession()])

    async def fake_state(_session, *, orchestration_id):
        return make_orchestration_state(orchestration_id, extraction_run_id=extraction_run_id, orchestration_status="completed")

    async def fake_invalid_bindings(_session, *, orchestration_id):
        return False

    async def fake_count(_session, *, orchestration_id):
        return 2

    async def fake_candidates(_session, *, orchestration_id):
        return (first_candidate, second_candidate)

    async def fake_existing_for_update(_session, *, orchestration_id, algorithm_version):
        return type(
            "ExistingApplication",
            (),
            {
                "id": existing.id,
                "orchestration_id": existing.orchestration_id,
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

    monkeypatch.setattr(duplicate_grouping_repository, "get_duplicate_grouping_orchestration_state", fake_state)
    monkeypatch.setattr(duplicate_grouping_repository, "has_invalid_completed_batch_bindings", fake_invalid_bindings)
    monkeypatch.setattr(duplicate_grouping_repository, "count_duplicate_candidate_fact_values", fake_count)
    monkeypatch.setattr(duplicate_grouping_repository, "list_duplicate_candidates", fake_candidates)
    monkeypatch.setattr(duplicate_grouping_repository, "get_grouping_application_for_update", fake_existing_for_update)

    result = run_async(
        duplicate_grouping_service.ensure_cross_batch_duplicate_grouping(
            session_factory,
            orchestration_id=orchestration_id,
        )
    )

    assert result.created_new is False
    assert result.grouping_application_id == existing.id
    assert session_factory.created_sessions[1].commit_count == 1


def test_ensure_cross_batch_duplicate_grouping_fails_closed_on_manifest_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestration_id = uuid.uuid4()
    extraction_run_id = uuid.uuid4()
    first_candidate = candidate(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        source_batch_id=uuid.uuid4(),
    )
    second_candidate = candidate(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        fact_id=first_candidate.fact_id,
        source_batch_id=uuid.uuid4(),
        value_json=first_candidate.value_json,
        value_type=first_candidate.value_type,
        referenced_entity_id=first_candidate.referenced_entity_id,
    )
    session_factory = SessionFactory([FakeSession(), FakeSession()])

    async def fake_state(_session, *, orchestration_id):
        return make_orchestration_state(orchestration_id, extraction_run_id=extraction_run_id, orchestration_status="completed")

    async def fake_invalid_bindings(_session, *, orchestration_id):
        return False

    async def fake_count(_session, *, orchestration_id):
        return 2

    async def fake_candidates(_session, *, orchestration_id):
        return (first_candidate, second_candidate)

    async def fake_existing_for_update(_session, *, orchestration_id, algorithm_version):
        return type(
            "ExistingApplication",
            (),
            {
                "id": uuid.uuid4(),
                "orchestration_id": orchestration_id,
                "extraction_run_id": extraction_run_id,
                "algorithm_version": algorithm_version,
                "input_manifest_hash": "a" * 64,
                "result_manifest_hash": "b" * 64,
                "input_fact_value_count": 999,
                "duplicate_group_count": 0,
                "duplicate_member_count": 0,
                "created_at": datetime.now(timezone.utc),
            },
        )()

    monkeypatch.setattr(duplicate_grouping_repository, "get_duplicate_grouping_orchestration_state", fake_state)
    monkeypatch.setattr(duplicate_grouping_repository, "has_invalid_completed_batch_bindings", fake_invalid_bindings)
    monkeypatch.setattr(duplicate_grouping_repository, "count_duplicate_candidate_fact_values", fake_count)
    monkeypatch.setattr(duplicate_grouping_repository, "list_duplicate_candidates", fake_candidates)
    monkeypatch.setattr(duplicate_grouping_repository, "get_grouping_application_for_update", fake_existing_for_update)

    with pytest.raises(duplicate_grouping_service.CrossBatchDuplicateGroupingInvariantError):
        run_async(
            duplicate_grouping_service.ensure_cross_batch_duplicate_grouping(
                session_factory,
                orchestration_id=orchestration_id,
            )
        )


def test_ensure_cross_batch_duplicate_grouping_handles_target_constraint_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestration_id = uuid.uuid4()
    extraction_run_id = uuid.uuid4()
    first_candidate = candidate(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        source_batch_id=uuid.uuid4(),
    )
    second_candidate = candidate(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        fact_id=first_candidate.fact_id,
        source_batch_id=uuid.uuid4(),
        value_json=first_candidate.value_json,
        value_type=first_candidate.value_type,
        referenced_entity_id=first_candidate.referenced_entity_id,
    )
    plan = duplicate_grouping_service.build_duplicate_grouping_write_plan((first_candidate, second_candidate))
    existing = FakeApplicationLedger(
        id=uuid.uuid4(),
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        algorithm_version=plan.algorithm_version,
        input_manifest_hash=plan.input_manifest_hash,
        result_manifest_hash=plan.result_manifest_hash,
        input_fact_value_count=plan.input_fact_value_count,
        duplicate_group_count=plan.duplicate_group_count,
        duplicate_member_count=plan.duplicate_member_count,
        created_at=datetime.now(timezone.utc),
    )
    session_factory = SessionFactory([FakeSession(), FakeSession(), FakeSession()])

    async def fake_state(_session, *, orchestration_id):
        return make_orchestration_state(orchestration_id, extraction_run_id=extraction_run_id, orchestration_status="completed")

    async def fake_invalid_bindings(_session, *, orchestration_id):
        return False

    async def fake_count(_session, *, orchestration_id):
        return 2

    async def fake_candidates(_session, *, orchestration_id):
        return (first_candidate, second_candidate)

    async def fake_existing_for_update(_session, *, orchestration_id, algorithm_version):
        return None

    async def fake_create_application(_session, application):
        raise make_integrity_error("uq_dupgrp_app_orch_alg")

    async def fake_get_existing(_session, *, orchestration_id, algorithm_version):
        return existing

    monkeypatch.setattr(duplicate_grouping_repository, "get_duplicate_grouping_orchestration_state", fake_state)
    monkeypatch.setattr(duplicate_grouping_repository, "has_invalid_completed_batch_bindings", fake_invalid_bindings)
    monkeypatch.setattr(duplicate_grouping_repository, "count_duplicate_candidate_fact_values", fake_count)
    monkeypatch.setattr(duplicate_grouping_repository, "list_duplicate_candidates", fake_candidates)
    monkeypatch.setattr(duplicate_grouping_repository, "get_grouping_application_for_update", fake_existing_for_update)
    monkeypatch.setattr(duplicate_grouping_repository, "create_grouping_application", fake_create_application)
    monkeypatch.setattr(duplicate_grouping_repository, "get_grouping_application_ledger", fake_get_existing)

    result = run_async(
        duplicate_grouping_service.ensure_cross_batch_duplicate_grouping(
            session_factory,
            orchestration_id=orchestration_id,
        )
    )

    assert result.created_new is False
    assert result.grouping_application_id == existing.id
    assert session_factory.created_sessions[1].rollback_count == 1


def test_ensure_cross_batch_duplicate_grouping_does_not_swallow_unknown_integrity_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestration_id = uuid.uuid4()
    extraction_run_id = uuid.uuid4()
    first_candidate = candidate(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        source_batch_id=uuid.uuid4(),
    )
    second_candidate = candidate(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        fact_id=first_candidate.fact_id,
        source_batch_id=uuid.uuid4(),
        value_json=first_candidate.value_json,
        value_type=first_candidate.value_type,
        referenced_entity_id=first_candidate.referenced_entity_id,
    )
    session_factory = SessionFactory([FakeSession(), FakeSession()])

    async def fake_state(_session, *, orchestration_id):
        return make_orchestration_state(orchestration_id, extraction_run_id=extraction_run_id, orchestration_status="completed")

    async def fake_invalid_bindings(_session, *, orchestration_id):
        return False

    async def fake_count(_session, *, orchestration_id):
        return 2

    async def fake_candidates(_session, *, orchestration_id):
        return (first_candidate, second_candidate)

    async def fake_existing_for_update(_session, *, orchestration_id, algorithm_version):
        return None

    async def fake_create_application(_session, application):
        raise make_integrity_error("uq_other_constraint")

    monkeypatch.setattr(duplicate_grouping_repository, "get_duplicate_grouping_orchestration_state", fake_state)
    monkeypatch.setattr(duplicate_grouping_repository, "has_invalid_completed_batch_bindings", fake_invalid_bindings)
    monkeypatch.setattr(duplicate_grouping_repository, "count_duplicate_candidate_fact_values", fake_count)
    monkeypatch.setattr(duplicate_grouping_repository, "list_duplicate_candidates", fake_candidates)
    monkeypatch.setattr(duplicate_grouping_repository, "get_grouping_application_for_update", fake_existing_for_update)
    monkeypatch.setattr(duplicate_grouping_repository, "create_grouping_application", fake_create_application)

    with pytest.raises(IntegrityError):
        run_async(
            duplicate_grouping_service.ensure_cross_batch_duplicate_grouping(
                session_factory,
                orchestration_id=orchestration_id,
            )
        )


def test_ensure_cross_batch_duplicate_grouping_normalizes_algorithm_version_for_query_and_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestration_id = uuid.uuid4()
    extraction_run_id = uuid.uuid4()
    first_candidate = candidate(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        source_batch_id=uuid.uuid4(),
    )
    second_candidate = candidate(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        fact_id=first_candidate.fact_id,
        source_batch_id=uuid.uuid4(),
        value_json=first_candidate.value_json,
        value_type=first_candidate.value_type,
        referenced_entity_id=first_candidate.referenced_entity_id,
    )
    session_factory = SessionFactory([FakeSession(), FakeSession()])
    seen_algorithm_versions: list[str] = []
    created_application_versions: list[str] = []

    async def fake_state(_session, *, orchestration_id):
        return make_orchestration_state(
            orchestration_id,
            extraction_run_id=extraction_run_id,
            orchestration_status="completed",
        )

    async def fake_invalid_bindings(_session, *, orchestration_id):
        return False

    async def fake_count(_session, *, orchestration_id):
        return 2

    async def fake_candidates(_session, *, orchestration_id):
        return (first_candidate, second_candidate)

    async def fake_existing_for_update(_session, *, orchestration_id, algorithm_version):
        seen_algorithm_versions.append(algorithm_version)
        return None

    async def fake_create_application(_session, application):
        created_application_versions.append(application.algorithm_version)
        return application

    monkeypatch.setattr(duplicate_grouping_repository, "get_duplicate_grouping_orchestration_state", fake_state)
    monkeypatch.setattr(duplicate_grouping_repository, "has_invalid_completed_batch_bindings", fake_invalid_bindings)
    monkeypatch.setattr(duplicate_grouping_repository, "count_duplicate_candidate_fact_values", fake_count)
    monkeypatch.setattr(duplicate_grouping_repository, "list_duplicate_candidates", fake_candidates)
    monkeypatch.setattr(duplicate_grouping_repository, "get_grouping_application_for_update", fake_existing_for_update)
    monkeypatch.setattr(duplicate_grouping_repository, "create_grouping_application", fake_create_application)
    monkeypatch.setattr(duplicate_grouping_repository, "create_duplicate_groups", return_second_arg)
    monkeypatch.setattr(duplicate_grouping_repository, "create_duplicate_group_members", return_second_arg)

    result = run_async(
        duplicate_grouping_service.ensure_cross_batch_duplicate_grouping(
            session_factory,
            orchestration_id=orchestration_id,
            algorithm_version="  cross_batch_exact_v2  ",
        )
    )

    assert result.algorithm_version == "cross_batch_exact_v2"
    assert seen_algorithm_versions == ["cross_batch_exact_v2"]
    assert created_application_versions == ["cross_batch_exact_v2"]


def test_ensure_cross_batch_duplicate_grouping_rolls_back_on_partial_write_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestration_id = uuid.uuid4()
    extraction_run_id = uuid.uuid4()
    first_candidate = candidate(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        source_batch_id=uuid.uuid4(),
    )
    second_candidate = candidate(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        fact_id=first_candidate.fact_id,
        source_batch_id=uuid.uuid4(),
        value_json=first_candidate.value_json,
        value_type=first_candidate.value_type,
        referenced_entity_id=first_candidate.referenced_entity_id,
    )
    session_factory = SessionFactory([FakeSession(), FakeSession()])

    async def fake_state(_session, *, orchestration_id):
        return make_orchestration_state(orchestration_id, extraction_run_id=extraction_run_id, orchestration_status="completed")

    async def fake_invalid_bindings(_session, *, orchestration_id):
        return False

    async def fake_count(_session, *, orchestration_id):
        return 2

    async def fake_candidates(_session, *, orchestration_id):
        return (first_candidate, second_candidate)

    async def fake_existing_for_update(_session, *, orchestration_id, algorithm_version):
        return None

    async def fake_create_application(_session, application):
        return application

    async def fake_create_groups(_session, groups):
        return groups

    async def fake_create_members(_session, members):
        raise RuntimeError("boom")

    monkeypatch.setattr(duplicate_grouping_repository, "get_duplicate_grouping_orchestration_state", fake_state)
    monkeypatch.setattr(duplicate_grouping_repository, "has_invalid_completed_batch_bindings", fake_invalid_bindings)
    monkeypatch.setattr(duplicate_grouping_repository, "count_duplicate_candidate_fact_values", fake_count)
    monkeypatch.setattr(duplicate_grouping_repository, "list_duplicate_candidates", fake_candidates)
    monkeypatch.setattr(duplicate_grouping_repository, "get_grouping_application_for_update", fake_existing_for_update)
    monkeypatch.setattr(duplicate_grouping_repository, "create_grouping_application", fake_create_application)
    monkeypatch.setattr(duplicate_grouping_repository, "create_duplicate_groups", fake_create_groups)
    monkeypatch.setattr(duplicate_grouping_repository, "create_duplicate_group_members", fake_create_members)

    with pytest.raises(RuntimeError):
        run_async(
            duplicate_grouping_service.ensure_cross_batch_duplicate_grouping(
                session_factory,
                orchestration_id=orchestration_id,
            )
        )

    assert session_factory.created_sessions[1].rollback_count == 1
    assert session_factory.created_sessions[1].commit_count == 0


@pytest.mark.parametrize("orchestration_status", ["planned", "running", "failed"])
def test_ensure_cross_batch_duplicate_grouping_rejects_non_terminal_orchestration_status(
    monkeypatch: pytest.MonkeyPatch,
    orchestration_status: str,
) -> None:
    orchestration_id = uuid.uuid4()
    session_factory = SessionFactory([FakeSession()])

    async def fake_state(_session, *, orchestration_id):
        return make_orchestration_state(orchestration_id, orchestration_status=orchestration_status)

    monkeypatch.setattr(duplicate_grouping_repository, "get_duplicate_grouping_orchestration_state", fake_state)

    with pytest.raises(duplicate_grouping_service.CrossBatchDuplicateGroupingStateError):
        run_async(
            duplicate_grouping_service.ensure_cross_batch_duplicate_grouping(
                session_factory,
                orchestration_id=orchestration_id,
            )
        )


@pytest.mark.parametrize("orchestration_status", ["completed", "partial"])
def test_ensure_cross_batch_duplicate_grouping_allows_completed_and_partial_orchestrations(
    monkeypatch: pytest.MonkeyPatch,
    orchestration_status: str,
) -> None:
    orchestration_id = uuid.uuid4()
    extraction_run_id = uuid.uuid4()
    first_candidate = candidate(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        source_batch_id=uuid.uuid4(),
    )
    second_candidate = candidate(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        fact_id=first_candidate.fact_id,
        source_batch_id=uuid.uuid4(),
        value_json=first_candidate.value_json,
        value_type=first_candidate.value_type,
        referenced_entity_id=first_candidate.referenced_entity_id,
    )
    session_factory = SessionFactory([FakeSession(), FakeSession()])

    async def fake_state(_session, *, orchestration_id):
        return make_orchestration_state(orchestration_id, extraction_run_id=extraction_run_id, orchestration_status=orchestration_status)

    async def fake_invalid_bindings(_session, *, orchestration_id):
        return False

    async def fake_count(_session, *, orchestration_id):
        return 2

    async def fake_candidates(_session, *, orchestration_id):
        return (first_candidate, second_candidate)

    async def fake_existing_for_update(_session, *, orchestration_id, algorithm_version):
        return None

    monkeypatch.setattr(duplicate_grouping_repository, "get_duplicate_grouping_orchestration_state", fake_state)
    monkeypatch.setattr(duplicate_grouping_repository, "has_invalid_completed_batch_bindings", fake_invalid_bindings)
    monkeypatch.setattr(duplicate_grouping_repository, "count_duplicate_candidate_fact_values", fake_count)
    monkeypatch.setattr(duplicate_grouping_repository, "list_duplicate_candidates", fake_candidates)
    monkeypatch.setattr(duplicate_grouping_repository, "get_grouping_application_for_update", fake_existing_for_update)
    monkeypatch.setattr(duplicate_grouping_repository, "create_grouping_application", return_second_arg)
    monkeypatch.setattr(duplicate_grouping_repository, "create_duplicate_groups", return_second_arg)
    monkeypatch.setattr(duplicate_grouping_repository, "create_duplicate_group_members", return_second_arg)

    result = run_async(
        duplicate_grouping_service.ensure_cross_batch_duplicate_grouping(
            session_factory,
            orchestration_id=orchestration_id,
        )
    )

    assert result.created_new is True


def test_ensure_cross_batch_duplicate_grouping_fails_closed_on_invalid_completed_batch_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestration_id = uuid.uuid4()
    extraction_run_id = uuid.uuid4()
    session_factory = SessionFactory([FakeSession()])

    async def fake_state(_session, *, orchestration_id):
        return make_orchestration_state(orchestration_id, extraction_run_id=extraction_run_id, orchestration_status="completed")

    async def fake_invalid_bindings(_session, *, orchestration_id):
        return True

    monkeypatch.setattr(duplicate_grouping_repository, "get_duplicate_grouping_orchestration_state", fake_state)
    monkeypatch.setattr(duplicate_grouping_repository, "has_invalid_completed_batch_bindings", fake_invalid_bindings)

    with pytest.raises(duplicate_grouping_service.CrossBatchDuplicateGroupingInvariantError, match="binding_mismatch"):
        run_async(
            duplicate_grouping_service.ensure_cross_batch_duplicate_grouping(
                session_factory,
                orchestration_id=orchestration_id,
            )
        )


def test_ensure_cross_batch_duplicate_grouping_fails_closed_on_candidate_source_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestration_id = uuid.uuid4()
    extraction_run_id = uuid.uuid4()
    session_factory = SessionFactory([FakeSession()])

    async def fake_state(_session, *, orchestration_id):
        return make_orchestration_state(orchestration_id, extraction_run_id=extraction_run_id, orchestration_status="completed")

    async def fake_invalid_bindings(_session, *, orchestration_id):
        return False

    async def fake_count(_session, *, orchestration_id):
        return 2

    async def fake_candidates(_session, *, orchestration_id):
        return (
            candidate(
                orchestration_id=orchestration_id,
                extraction_run_id=extraction_run_id,
                source_batch_id=uuid.uuid4(),
            ),
        )

    monkeypatch.setattr(duplicate_grouping_repository, "get_duplicate_grouping_orchestration_state", fake_state)
    monkeypatch.setattr(duplicate_grouping_repository, "has_invalid_completed_batch_bindings", fake_invalid_bindings)
    monkeypatch.setattr(duplicate_grouping_repository, "count_duplicate_candidate_fact_values", fake_count)
    monkeypatch.setattr(duplicate_grouping_repository, "list_duplicate_candidates", fake_candidates)

    with pytest.raises(duplicate_grouping_service.CrossBatchDuplicateGroupingInvariantError, match="candidate_source_mismatch"):
        run_async(
            duplicate_grouping_service.ensure_cross_batch_duplicate_grouping(
                session_factory,
                orchestration_id=orchestration_id,
            )
        )


def test_ensure_cross_batch_duplicate_grouping_rejects_not_ready_without_leaking_values(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    orchestration_id = uuid.uuid4()
    sentinel = "SENTINEL_DUP_VALUE"
    session_factory = SessionFactory([FakeSession()])

    async def fake_state(_session, *, orchestration_id):
        return duplicate_grouping_repository.DuplicateGroupingOrchestrationState(
            orchestration_id=orchestration_id,
            extraction_run_id=uuid.uuid4(),
            project_id=uuid.uuid4(),
            extraction_run_status="completed",
            extraction_run_outcome="success",
            orchestration_status="running",
        )

    monkeypatch.setattr(duplicate_grouping_repository, "get_duplicate_grouping_orchestration_state", fake_state)

    with pytest.raises(duplicate_grouping_service.CrossBatchDuplicateGroupingStateError) as exc_info:
        run_async(
            duplicate_grouping_service.ensure_cross_batch_duplicate_grouping(
                session_factory,
                orchestration_id=orchestration_id,
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
            self.statements = []
            self.rows = rows

        async def execute(self, statement):
            self.statements.append(statement)
            return _FakeResult(self.rows)

    orchestration_id = uuid.uuid4()
    extraction_run_id = uuid.uuid4()
    fact_value_id = uuid.uuid4()
    fact_id = uuid.uuid4()
    source_batch_id = uuid.uuid4()
    rows = [
        type(
            "Row",
            (),
            {
                "fact_value_id": fact_value_id,
                "fact_id": fact_id,
                "orchestration_id": orchestration_id,
                "extraction_run_id": extraction_run_id,
                "source_batch_id": source_batch_id,
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
                "fact_id": fact_id,
                "orchestration_id": orchestration_id,
                "extraction_run_id": extraction_run_id,
                "source_batch_id": source_batch_id,
                "value_type": "string",
                "value_json": "A",
                "referenced_entity_id": None,
                "evidence_link_id": uuid.UUID("00000000-0000-0000-0000-000000000003"),
            },
        )(),
        type(
            "Row",
            (),
            {
                "fact_value_id": fact_value_id,
                "fact_id": fact_id,
                "orchestration_id": orchestration_id,
                "extraction_run_id": extraction_run_id,
                "source_batch_id": source_batch_id,
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
            orchestration_id=orchestration_id,
        )
    )
    sql = str(session.statements[0].compile(dialect=postgresql.dialect()))

    assert len(candidates) == 1
    assert candidates[0].orchestration_id == orchestration_id
    assert candidates[0].source_batch_id == source_batch_id
    assert candidates[0].evidence_link_ids == (
        uuid.UUID("00000000-0000-0000-0000-000000000002"),
        uuid.UUID("00000000-0000-0000-0000-000000000003"),
    )
    assert "fact_extraction_orch_batches.orchestration_id" in sql
    assert "fact_extraction_batch_applications" in sql
    assert "ORDER BY fact_values.id ASC" in sql


@pytest.mark.parametrize(
    ("field_name", "left_value", "right_value"),
    [
        ("fact_id", uuid.uuid4(), uuid.uuid4()),
        ("orchestration_id", uuid.uuid4(), uuid.uuid4()),
        ("extraction_run_id", uuid.uuid4(), uuid.uuid4()),
        ("source_batch_id", uuid.uuid4(), uuid.uuid4()),
        ("value_type", "string", "number"),
        ("value_json", {"value": "A"}, {"value": "B"}),
        ("referenced_entity_id", uuid.uuid4(), uuid.uuid4()),
    ],
)
def test_duplicate_candidate_query_fails_closed_when_stable_fields_diverge(
    field_name: str,
    left_value,
    right_value,
) -> None:
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

    fact_value_id = uuid.uuid4()
    base = {
        "fact_value_id": fact_value_id,
        "fact_id": uuid.uuid4(),
        "orchestration_id": uuid.uuid4(),
        "extraction_run_id": uuid.uuid4(),
        "source_batch_id": uuid.uuid4(),
        "value_type": "string",
        "value_json": {"value": "A"},
        "referenced_entity_id": None,
        "evidence_link_id": uuid.uuid4(),
    }
    first_row = type("Row", (), dict(base))()
    second_payload = dict(base)
    second_payload[field_name] = right_value
    second_payload["evidence_link_id"] = uuid.uuid4()
    if field_name != "referenced_entity_id":
        first_row = type("Row", (), {**base, field_name: left_value})()
    second_row = type("Row", (), second_payload)()

    with pytest.raises(
        duplicate_grouping_repository.DuplicateGroupingRepositoryInvariantError,
        match="cross_batch_duplicate_grouping_candidate_row_mismatch",
    ):
        run_async(
            duplicate_grouping_repository.list_duplicate_candidates(
                _Session([first_row, second_row]),
                orchestration_id=uuid.uuid4(),
            )
        )


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
    source_batch_id = uuid.uuid4()
    first_link_id = uuid.UUID("ffffffff-ffff-ffff-ffff-fffffffffff0")
    second_link_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
    first_evidence_id = uuid.UUID("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeee0")
    second_evidence_id = uuid.UUID("00000000-0000-0000-0000-000000000002")
    rows = [
        type(
            "Row",
            (),
            {
                "group_id": group_id,
                "duplicate_key_hash": "a" * 64,
                "fact_value_id": fact_value_id,
                "source_batch_id": source_batch_id,
                "evidence_link_id": first_link_id,
                "evidence_id": first_evidence_id,
            },
        )(),
        type(
            "Row",
            (),
            {
                "group_id": group_id,
                "duplicate_key_hash": "a" * 64,
                "fact_value_id": fact_value_id,
                "source_batch_id": source_batch_id,
                "evidence_link_id": second_link_id,
                "evidence_id": second_evidence_id,
            },
        )(),
        type(
            "Row",
            (),
            {
                "group_id": group_id,
                "duplicate_key_hash": "a" * 64,
                "fact_value_id": fact_value_id,
                "source_batch_id": source_batch_id,
                "evidence_link_id": first_link_id,
                "evidence_id": first_evidence_id,
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
            source_batch_id=source_batch_id,
            evidence_link_ids=(
                first_link_id,
                second_link_id,
            ),
            evidence_ids=(
                first_evidence_id,
                second_evidence_id,
            ),
        ),
    )


def test_build_fact_value_consistency_candidate_write_plan_detects_cross_batch_multi_value_without_duplicate_groups() -> None:
    orchestration_id = uuid.uuid4()
    extraction_run_id = uuid.uuid4()
    fact_id = uuid.uuid4()
    first = candidate(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        fact_id=fact_id,
        source_batch_id=uuid.uuid4(),
        value_json="Value A",
        value_type="string",
    )
    second = candidate(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        fact_id=fact_id,
        source_batch_id=uuid.uuid4(),
        value_json="Value B",
        value_type="string",
    )
    source_duplicate_plan = duplicate_grouping_service.build_duplicate_grouping_write_plan((first, second))
    source_application = make_duplicate_grouping_application_ledger((first, second))

    plan = duplicate_grouping_service.build_fact_value_consistency_candidate_write_plan(
        (first, second),
        source_duplicate_grouping_application=source_application,
    )

    assert source_duplicate_plan.duplicate_group_count == 0
    assert plan.algorithm_version == CROSS_BATCH_MULTI_VALUE_CANDIDATE_ALGORITHM_VERSION
    assert plan.source_duplicate_grouping_algorithm_version == CROSS_BATCH_DUPLICATE_ALGORITHM_VERSION
    assert plan.input_manifest_hash == source_application.input_manifest_hash
    assert plan.candidate_count == 1
    assert plan.member_count == 2
    assert plan.candidates[0].fact_id == fact_id
    assert plan.candidates[0].candidate_kind == "multi_value"
    assert plan.candidates[0].distinct_semantic_key_count == 2
    assert plan.candidates[0].distinct_batch_count == 2


def test_build_fact_value_consistency_candidate_write_plan_skips_same_value_across_batches() -> None:
    orchestration_id = uuid.uuid4()
    extraction_run_id = uuid.uuid4()
    fact_id = uuid.uuid4()
    first = candidate(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        fact_id=fact_id,
        source_batch_id=uuid.uuid4(),
        value_json="Value A",
        value_type="string",
    )
    second = candidate(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        fact_id=fact_id,
        source_batch_id=uuid.uuid4(),
        value_json="Value A",
        value_type="string",
    )
    source_application = make_duplicate_grouping_application_ledger((first, second))

    plan = duplicate_grouping_service.build_fact_value_consistency_candidate_write_plan(
        (first, second),
        source_duplicate_grouping_application=source_application,
    )

    assert plan.candidate_count == 0
    assert plan.member_count == 0


def test_build_fact_value_consistency_candidate_write_plan_skips_different_values_within_single_batch() -> None:
    orchestration_id = uuid.uuid4()
    extraction_run_id = uuid.uuid4()
    fact_id = uuid.uuid4()
    source_batch_id = uuid.uuid4()
    first = candidate(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        fact_id=fact_id,
        source_batch_id=source_batch_id,
        value_json="Value A",
        value_type="string",
    )
    second = candidate(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        fact_id=fact_id,
        source_batch_id=source_batch_id,
        value_json="Value B",
        value_type="string",
    )
    source_application = make_duplicate_grouping_application_ledger((first, second))

    plan = duplicate_grouping_service.build_fact_value_consistency_candidate_write_plan(
        (first, second),
        source_duplicate_grouping_application=source_application,
    )

    assert plan.candidate_count == 0
    assert plan.member_count == 0


def test_build_fact_value_consistency_candidate_write_plan_creates_fact_level_candidate_with_all_members() -> None:
    orchestration_id = uuid.uuid4()
    extraction_run_id = uuid.uuid4()
    fact_id = uuid.uuid4()
    candidates = (
        candidate(
            orchestration_id=orchestration_id,
            extraction_run_id=extraction_run_id,
            fact_id=fact_id,
            source_batch_id=uuid.uuid4(),
            value_json="Value A",
            value_type="string",
        ),
        candidate(
            orchestration_id=orchestration_id,
            extraction_run_id=extraction_run_id,
            fact_id=fact_id,
            source_batch_id=uuid.uuid4(),
            value_json="Value B",
            value_type="string",
        ),
        candidate(
            orchestration_id=orchestration_id,
            extraction_run_id=extraction_run_id,
            fact_id=fact_id,
            source_batch_id=uuid.uuid4(),
            value_json="Value C",
            value_type="string",
        ),
        candidate(
            orchestration_id=orchestration_id,
            extraction_run_id=extraction_run_id,
            fact_id=fact_id,
            source_batch_id=uuid.uuid4(),
            value_json="Value A",
            value_type="string",
        ),
    )
    source_application = make_duplicate_grouping_application_ledger(candidates)

    plan = duplicate_grouping_service.build_fact_value_consistency_candidate_write_plan(
        candidates,
        source_duplicate_grouping_application=source_application,
    )

    assert plan.candidate_count == 1
    assert plan.member_count == 4
    assert plan.candidates[0].fact_id == fact_id
    assert plan.candidates[0].member_count == 4
    assert plan.candidates[0].distinct_semantic_key_count == 3
    assert plan.candidates[0].distinct_batch_count == 4
    assert {member.fact_value_id for member in plan.candidates[0].members} == {
        item.fact_value_id for item in candidates
    }


def test_build_fact_value_consistency_candidate_write_plan_never_merges_different_facts() -> None:
    orchestration_id = uuid.uuid4()
    extraction_run_id = uuid.uuid4()
    first_fact_id = uuid.uuid4()
    second_fact_id = uuid.uuid4()
    candidates = (
        candidate(
            orchestration_id=orchestration_id,
            extraction_run_id=extraction_run_id,
            fact_id=first_fact_id,
            source_batch_id=uuid.uuid4(),
            value_json="A1",
            value_type="string",
        ),
        candidate(
            orchestration_id=orchestration_id,
            extraction_run_id=extraction_run_id,
            fact_id=first_fact_id,
            source_batch_id=uuid.uuid4(),
            value_json="A2",
            value_type="string",
        ),
        candidate(
            orchestration_id=orchestration_id,
            extraction_run_id=extraction_run_id,
            fact_id=second_fact_id,
            source_batch_id=uuid.uuid4(),
            value_json="B1",
            value_type="string",
        ),
        candidate(
            orchestration_id=orchestration_id,
            extraction_run_id=extraction_run_id,
            fact_id=second_fact_id,
            source_batch_id=uuid.uuid4(),
            value_json="B2",
            value_type="string",
        ),
    )
    source_application = make_duplicate_grouping_application_ledger(candidates)

    plan = duplicate_grouping_service.build_fact_value_consistency_candidate_write_plan(
        candidates,
        source_duplicate_grouping_application=source_application,
    )

    assert plan.candidate_count == 2
    assert {item.fact_id for item in plan.candidates} == {first_fact_id, second_fact_id}
    assert all(item.member_count == 2 for item in plan.candidates)


def test_build_fact_value_consistency_candidate_write_plan_fails_closed_on_source_manifest_mismatch() -> None:
    orchestration_id = uuid.uuid4()
    extraction_run_id = uuid.uuid4()
    first = candidate(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        source_batch_id=uuid.uuid4(),
        value_json="Value A",
        value_type="string",
    )
    second = candidate(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        fact_id=first.fact_id,
        source_batch_id=uuid.uuid4(),
        value_json="Value B",
        value_type="string",
    )
    base_application = make_duplicate_grouping_application_ledger((first, second))
    source_application = DuplicateGroupingApplicationLedger(
        id=base_application.id,
        orchestration_id=base_application.orchestration_id,
        extraction_run_id=base_application.extraction_run_id,
        algorithm_version=base_application.algorithm_version,
        input_manifest_hash="f" * 64,
        result_manifest_hash=base_application.result_manifest_hash,
        input_fact_value_count=base_application.input_fact_value_count,
        duplicate_group_count=base_application.duplicate_group_count,
        duplicate_member_count=base_application.duplicate_member_count,
        created_at=base_application.created_at,
    )

    with pytest.raises(
        duplicate_grouping_service.FactValueConsistencyCandidateInvariantError,
        match="source_input_manifest_mismatch",
    ):
        duplicate_grouping_service.build_fact_value_consistency_candidate_write_plan(
            (first, second),
            source_duplicate_grouping_application=source_application,
        )


def test_build_fact_value_consistency_candidate_write_plan_fails_closed_on_cross_orchestration_candidates() -> None:
    extraction_run_id = uuid.uuid4()
    first = candidate(
        orchestration_id=uuid.uuid4(),
        extraction_run_id=extraction_run_id,
        source_batch_id=uuid.uuid4(),
        value_json="Value A",
        value_type="string",
    )
    second = candidate(
        orchestration_id=uuid.uuid4(),
        extraction_run_id=extraction_run_id,
        fact_id=first.fact_id,
        source_batch_id=uuid.uuid4(),
        value_json="Value B",
        value_type="string",
    )
    source_application = make_duplicate_grouping_application_ledger((first,))

    with pytest.raises(
        duplicate_grouping_service.FactValueConsistencyCandidateInvariantError,
        match="single orchestration",
    ):
        duplicate_grouping_service.build_fact_value_consistency_candidate_write_plan(
            (first, second),
            source_duplicate_grouping_application=source_application,
        )


def test_consistency_candidate_member_metadata_keeps_fact_value_fk_for_evidence_roundtrip() -> None:
    member_foreign_keys = {
        tuple(constraint.column_keys): (
            constraint.name,
            tuple(element.column.name for element in constraint.elements),
        )
        for constraint in FactValueConsistencyCandidateMember.__table__.foreign_key_constraints
    }

    assert member_foreign_keys[("fact_value_id",)][1] == ("id",)


def test_ensure_cross_batch_multi_value_consistency_candidates_writes_zero_candidate_application_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestration_id = uuid.uuid4()
    extraction_run_id = uuid.uuid4()
    fact_id = uuid.uuid4()
    candidates = (
        candidate(
            orchestration_id=orchestration_id,
            extraction_run_id=extraction_run_id,
            fact_id=fact_id,
            source_batch_id=uuid.uuid4(),
            value_json="Value A",
            value_type="string",
        ),
        candidate(
            orchestration_id=orchestration_id,
            extraction_run_id=extraction_run_id,
            fact_id=fact_id,
            source_batch_id=uuid.uuid4(),
            value_json="Value A",
            value_type="string",
        ),
    )
    source_application = make_duplicate_grouping_application_ledger(candidates)
    sessions = [FakeSession(), FakeSession(), FakeSession(), FakeSession()]
    session_factory = SessionFactory(sessions)
    created_applications: list[FactValueConsistencyCandidateApplication] = []
    existing_application: dict[str, FactValueConsistencyCandidateApplication | None] = {"value": None}

    async def fake_source_application(_session, *, grouping_application_id):
        assert grouping_application_id == source_application.id
        return source_application

    async def fake_state(_session, *, orchestration_id):
        return make_orchestration_state(
            orchestration_id,
            extraction_run_id=extraction_run_id,
            orchestration_status="completed",
        )

    async def fake_invalid_bindings(_session, *, orchestration_id):
        return False

    async def fake_count(_session, *, orchestration_id):
        return len(candidates)

    async def fake_candidates(_session, *, orchestration_id):
        return candidates

    async def fake_existing_for_update(_session, *, duplicate_grouping_application_id, algorithm_version):
        assert duplicate_grouping_application_id == source_application.id
        assert algorithm_version == CROSS_BATCH_MULTI_VALUE_CANDIDATE_ALGORITHM_VERSION
        return existing_application["value"]

    async def fake_create_application(_session, application):
        created_applications.append(application)
        existing_application["value"] = application
        return application

    monkeypatch.setattr(duplicate_grouping_repository, "get_grouping_application_ledger_by_id", fake_source_application)
    monkeypatch.setattr(duplicate_grouping_repository, "get_duplicate_grouping_orchestration_state", fake_state)
    monkeypatch.setattr(duplicate_grouping_repository, "has_invalid_completed_batch_bindings", fake_invalid_bindings)
    monkeypatch.setattr(duplicate_grouping_repository, "count_duplicate_candidate_fact_values", fake_count)
    monkeypatch.setattr(duplicate_grouping_repository, "list_duplicate_candidates", fake_candidates)
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "get_consistency_candidate_application_for_update",
        fake_existing_for_update,
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "create_consistency_candidate_application",
        fake_create_application,
    )
    monkeypatch.setattr(duplicate_grouping_repository, "create_consistency_candidates", return_second_arg)
    monkeypatch.setattr(duplicate_grouping_repository, "create_consistency_candidate_members", return_second_arg)

    first_result = run_async(
        duplicate_grouping_service.ensure_cross_batch_multi_value_consistency_candidates(
            session_factory,
            duplicate_grouping_application_id=source_application.id,
        )
    )
    second_result = run_async(
        duplicate_grouping_service.ensure_cross_batch_multi_value_consistency_candidates(
            session_factory,
            duplicate_grouping_application_id=source_application.id,
        )
    )

    assert first_result.created_new is True
    assert first_result.candidate_count == 0
    assert first_result.member_count == 0
    assert second_result.created_new is False
    assert second_result.consistency_application_id == first_result.consistency_application_id
    assert len(created_applications) == 1


def test_ensure_cross_batch_multi_value_consistency_candidates_handles_target_constraint_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestration_id = uuid.uuid4()
    extraction_run_id = uuid.uuid4()
    fact_id = uuid.uuid4()
    candidates = (
        candidate(
            orchestration_id=orchestration_id,
            extraction_run_id=extraction_run_id,
            fact_id=fact_id,
            source_batch_id=uuid.uuid4(),
            value_json="Value A",
            value_type="string",
        ),
        candidate(
            orchestration_id=orchestration_id,
            extraction_run_id=extraction_run_id,
            fact_id=fact_id,
            source_batch_id=uuid.uuid4(),
            value_json="Value B",
            value_type="string",
        ),
    )
    source_application = make_duplicate_grouping_application_ledger(candidates)
    write_plan = duplicate_grouping_service.build_fact_value_consistency_candidate_write_plan(
        candidates,
        source_duplicate_grouping_application=source_application,
    )
    existing_application = FactValueConsistencyCandidateApplicationLedger(
        id=uuid.uuid4(),
        duplicate_grouping_application_id=source_application.id,
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        algorithm_version=write_plan.algorithm_version,
        input_manifest_hash=write_plan.input_manifest_hash,
        result_manifest_hash=write_plan.result_manifest_hash,
        candidate_count=write_plan.candidate_count,
        member_count=write_plan.member_count,
        created_at=datetime.now(timezone.utc),
    )
    session_factory = SessionFactory([FakeSession(), FakeSession(), FakeSession()])

    async def fake_source_application(_session, *, grouping_application_id):
        assert grouping_application_id == source_application.id
        return source_application

    async def fake_state(_session, *, orchestration_id):
        return make_orchestration_state(
            orchestration_id,
            extraction_run_id=extraction_run_id,
            orchestration_status="completed",
        )

    async def fake_invalid_bindings(_session, *, orchestration_id):
        return False

    async def fake_count(_session, *, orchestration_id):
        return len(candidates)

    async def fake_candidates(_session, *, orchestration_id):
        return candidates

    async def fake_existing_for_update(_session, *, duplicate_grouping_application_id, algorithm_version):
        return None

    async def fake_create_application(_session, application):
        raise make_integrity_error("uq_fvcca_dupgrp_alg")

    async def fake_get_existing(_session, *, duplicate_grouping_application_id, algorithm_version):
        assert duplicate_grouping_application_id == source_application.id
        assert algorithm_version == CROSS_BATCH_MULTI_VALUE_CANDIDATE_ALGORITHM_VERSION
        return existing_application

    monkeypatch.setattr(duplicate_grouping_repository, "get_grouping_application_ledger_by_id", fake_source_application)
    monkeypatch.setattr(duplicate_grouping_repository, "get_duplicate_grouping_orchestration_state", fake_state)
    monkeypatch.setattr(duplicate_grouping_repository, "has_invalid_completed_batch_bindings", fake_invalid_bindings)
    monkeypatch.setattr(duplicate_grouping_repository, "count_duplicate_candidate_fact_values", fake_count)
    monkeypatch.setattr(duplicate_grouping_repository, "list_duplicate_candidates", fake_candidates)
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "get_consistency_candidate_application_for_update",
        fake_existing_for_update,
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "create_consistency_candidate_application",
        fake_create_application,
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "get_consistency_candidate_application_ledger",
        fake_get_existing,
    )

    result = run_async(
        duplicate_grouping_service.ensure_cross_batch_multi_value_consistency_candidates(
            session_factory,
            duplicate_grouping_application_id=source_application.id,
        )
    )

    assert result.created_new is False
    assert result.consistency_application_id == existing_application.id
    assert session_factory.created_sessions[1].rollback_count == 1


def test_ensure_cross_batch_multi_value_consistency_candidates_fails_closed_on_source_manifest_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestration_id = uuid.uuid4()
    extraction_run_id = uuid.uuid4()
    fact_id = uuid.uuid4()
    candidates = (
        candidate(
            orchestration_id=orchestration_id,
            extraction_run_id=extraction_run_id,
            fact_id=fact_id,
            source_batch_id=uuid.uuid4(),
            value_json="Value A",
            value_type="string",
        ),
        candidate(
            orchestration_id=orchestration_id,
            extraction_run_id=extraction_run_id,
            fact_id=fact_id,
            source_batch_id=uuid.uuid4(),
            value_json="Value B",
            value_type="string",
        ),
    )
    base_application = make_duplicate_grouping_application_ledger(candidates)
    source_application = DuplicateGroupingApplicationLedger(
        id=base_application.id,
        orchestration_id=base_application.orchestration_id,
        extraction_run_id=base_application.extraction_run_id,
        algorithm_version=base_application.algorithm_version,
        input_manifest_hash="f" * 64,
        result_manifest_hash=base_application.result_manifest_hash,
        input_fact_value_count=base_application.input_fact_value_count,
        duplicate_group_count=base_application.duplicate_group_count,
        duplicate_member_count=base_application.duplicate_member_count,
        created_at=base_application.created_at,
    )
    session_factory = SessionFactory([FakeSession()])

    async def fake_source_application(_session, *, grouping_application_id):
        return source_application

    async def fake_state(_session, *, orchestration_id):
        return make_orchestration_state(
            orchestration_id,
            extraction_run_id=extraction_run_id,
            orchestration_status="completed",
        )

    async def fake_invalid_bindings(_session, *, orchestration_id):
        return False

    async def fake_count(_session, *, orchestration_id):
        return len(candidates)

    async def fake_candidates(_session, *, orchestration_id):
        return candidates

    monkeypatch.setattr(duplicate_grouping_repository, "get_grouping_application_ledger_by_id", fake_source_application)
    monkeypatch.setattr(duplicate_grouping_repository, "get_duplicate_grouping_orchestration_state", fake_state)
    monkeypatch.setattr(duplicate_grouping_repository, "has_invalid_completed_batch_bindings", fake_invalid_bindings)
    monkeypatch.setattr(duplicate_grouping_repository, "count_duplicate_candidate_fact_values", fake_count)
    monkeypatch.setattr(duplicate_grouping_repository, "list_duplicate_candidates", fake_candidates)

    with pytest.raises(
        duplicate_grouping_service.FactValueConsistencyCandidateInvariantError,
        match="source_input_manifest_mismatch",
    ):
        run_async(
            duplicate_grouping_service.ensure_cross_batch_multi_value_consistency_candidates(
                session_factory,
                duplicate_grouping_application_id=source_application.id,
            )
        )


def test_ensure_cross_batch_multi_value_consistency_candidates_fails_closed_when_other_orchestration_candidates_enter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction_run_id = uuid.uuid4()
    source_orchestration_id = uuid.uuid4()
    other_orchestration_id = uuid.uuid4()
    fact_id = uuid.uuid4()
    source_candidate = candidate(
        orchestration_id=source_orchestration_id,
        extraction_run_id=extraction_run_id,
        fact_id=fact_id,
        source_batch_id=uuid.uuid4(),
        value_json="Value A",
        value_type="string",
    )
    leaked_candidate = candidate(
        orchestration_id=other_orchestration_id,
        extraction_run_id=extraction_run_id,
        fact_id=fact_id,
        source_batch_id=uuid.uuid4(),
        value_json="Value B",
        value_type="string",
    )
    source_application = make_duplicate_grouping_application_ledger((source_candidate,))
    session_factory = SessionFactory([FakeSession()])

    async def fake_source_application(_session, *, grouping_application_id):
        return source_application

    async def fake_state(_session, *, orchestration_id):
        return make_orchestration_state(
            orchestration_id,
            extraction_run_id=extraction_run_id,
            orchestration_status="completed",
        )

    async def fake_invalid_bindings(_session, *, orchestration_id):
        return False

    async def fake_count(_session, *, orchestration_id):
        return 2

    async def fake_candidates(_session, *, orchestration_id):
        return (source_candidate, leaked_candidate)

    monkeypatch.setattr(duplicate_grouping_repository, "get_grouping_application_ledger_by_id", fake_source_application)
    monkeypatch.setattr(duplicate_grouping_repository, "get_duplicate_grouping_orchestration_state", fake_state)
    monkeypatch.setattr(duplicate_grouping_repository, "has_invalid_completed_batch_bindings", fake_invalid_bindings)
    monkeypatch.setattr(duplicate_grouping_repository, "count_duplicate_candidate_fact_values", fake_count)
    monkeypatch.setattr(duplicate_grouping_repository, "list_duplicate_candidates", fake_candidates)

    with pytest.raises(
        duplicate_grouping_service.FactValueConsistencyCandidateInvariantError,
        match="single orchestration",
    ):
        run_async(
            duplicate_grouping_service.ensure_cross_batch_multi_value_consistency_candidates(
                session_factory,
                duplicate_grouping_application_id=source_application.id,
            )
        )
