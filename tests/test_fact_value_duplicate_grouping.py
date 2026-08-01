from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
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
    FactValueConsistencyCandidateLedger,
    FactValueConsistencyCandidateMemberLedger,
)
from app.schemas.fact_extraction_persistence import (
    AuthenticatedCompletedFactExtractionApplicationSnapshot,
    AuthenticatedPersistedFactProposalItem,
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
    evidence_ids: tuple[uuid.UUID, ...] = (),
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
        evidence_ids=evidence_ids,
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
    batch_count: int | None = None,
    completed_batch_count: int | None = None,
    failed_batch_count: int | None = None,
) -> duplicate_grouping_repository.DuplicateGroupingOrchestrationState:
    if orchestration_status == "completed":
        resolved_batch_count = 1 if batch_count is None else batch_count
        resolved_completed_batch_count = (
            resolved_batch_count if completed_batch_count is None else completed_batch_count
        )
        resolved_failed_batch_count = 0 if failed_batch_count is None else failed_batch_count
    elif orchestration_status == "partial":
        resolved_batch_count = 2 if batch_count is None else batch_count
        resolved_completed_batch_count = 1 if completed_batch_count is None else completed_batch_count
        resolved_failed_batch_count = 1 if failed_batch_count is None else failed_batch_count
    else:
        resolved_batch_count = 1 if batch_count is None else batch_count
        resolved_completed_batch_count = 0 if completed_batch_count is None else completed_batch_count
        resolved_failed_batch_count = 0 if failed_batch_count is None else failed_batch_count
    return duplicate_grouping_repository.DuplicateGroupingOrchestrationState(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id or uuid.uuid4(),
        project_id=uuid.uuid4(),
        extraction_run_status="completed",
        extraction_run_outcome="success",
        orchestration_status=orchestration_status,
        batch_count=resolved_batch_count,
        completed_batch_count=resolved_completed_batch_count,
        failed_batch_count=resolved_failed_batch_count,
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


def make_application_snapshot(
    *,
    project_id: uuid.UUID,
    extraction_run_id: uuid.UUID,
    inference_run_id: uuid.UUID,
    input_batch_id: uuid.UUID | None = None,
    items: tuple[AuthenticatedPersistedFactProposalItem, ...],
    persistence_name: str = "persistence",
    persistence_version: str = "1.0.0",
    entity_resolution_policy_name: str = "entity-policy",
    entity_resolution_policy_version: str = "1.0.0",
) -> AuthenticatedCompletedFactExtractionApplicationSnapshot:
    return AuthenticatedCompletedFactExtractionApplicationSnapshot(
        application_id=uuid.uuid4(),
        project_id=project_id,
        extraction_run_id=extraction_run_id,
        inference_run_id=inference_run_id,
        input_batch_id=input_batch_id or uuid.uuid4(),
        persistence_name=persistence_name,
        persistence_version=persistence_version,
        entity_resolution_policy_name=entity_resolution_policy_name,
        entity_resolution_policy_version=entity_resolution_policy_version,
        items=items,
    )


def make_completed_batch_application(
    *,
    source_batch_id: uuid.UUID | None = None,
    batch_index: int = 0,
    application_id: uuid.UUID | None,
    current_input_batch_id: uuid.UUID | None,
    current_inference_run_id: uuid.UUID | None,
) -> duplicate_grouping_repository.CompletedOrchestrationBatchApplication:
    return duplicate_grouping_repository.CompletedOrchestrationBatchApplication(
        source_batch_id=source_batch_id or uuid.uuid4(),
        batch_index=batch_index,
        application_id=application_id,
        current_input_batch_id=current_input_batch_id,
        current_inference_run_id=current_inference_run_id,
    )


def install_authoritative_application_auth(
    monkeypatch: pytest.MonkeyPatch,
    *,
    candidates_by_orchestration: dict[uuid.UUID, tuple[DuplicateCandidate, ...]],
    state_by_orchestration: dict[
        uuid.UUID,
        duplicate_grouping_repository.DuplicateGroupingOrchestrationState,
    ],
) -> None:
    completed_batches_by_orchestration: dict[
        uuid.UUID,
        tuple[duplicate_grouping_repository.CompletedOrchestrationBatchApplication, ...],
    ] = {}
    snapshots_by_application_id: dict[
        uuid.UUID,
        AuthenticatedCompletedFactExtractionApplicationSnapshot,
    ] = {}

    for orchestration_id, candidates in candidates_by_orchestration.items():
        state = state_by_orchestration[orchestration_id]
        grouped_candidates: dict[uuid.UUID, list[DuplicateCandidate]] = {}
        for item in candidates:
            grouped_candidates.setdefault(item.source_batch_id, []).append(item)
        completed_batches: list[duplicate_grouping_repository.CompletedOrchestrationBatchApplication] = []
        for batch_index, source_batch_id in enumerate(sorted(grouped_candidates, key=str)):
            inference_run_id = uuid.uuid4()
            input_batch_id = uuid.uuid4()
            items = tuple(
                AuthenticatedPersistedFactProposalItem(
                    proposal_index=proposal_index,
                    fact_id=current.fact_id,
                    fact_value_id=current.fact_value_id,
                    subject_entity_id=None,
                    referenced_entity_id=current.referenced_entity_id,
                    evidence_ids=current.evidence_ids,
                )
                for proposal_index, current in enumerate(grouped_candidates[source_batch_id])
            )
            application_snapshot = make_application_snapshot(
                project_id=state.project_id,
                extraction_run_id=state.extraction_run_id,
                inference_run_id=inference_run_id,
                input_batch_id=input_batch_id,
                items=items,
                persistence_name=state.persistence_name,
                persistence_version=state.persistence_version,
                entity_resolution_policy_name=state.entity_resolution_policy_name,
                entity_resolution_policy_version=state.entity_resolution_policy_version,
            )
            snapshots_by_application_id[application_snapshot.application_id] = application_snapshot
            completed_batches.append(
                make_completed_batch_application(
                    source_batch_id=source_batch_id,
                    batch_index=batch_index,
                    application_id=application_snapshot.application_id,
                    current_input_batch_id=input_batch_id,
                    current_inference_run_id=inference_run_id,
                )
            )
        completed_batches_by_orchestration[orchestration_id] = tuple(completed_batches)

    monkeypatch.setattr(
        duplicate_grouping_repository,
        "list_completed_orchestration_batch_applications",
        lambda *_args, **_kwargs: asyncio.sleep(
            0,
            result=completed_batches_by_orchestration[_kwargs["orchestration_id"]],
        ),
    )
    monkeypatch.setattr(
        duplicate_grouping_service.fact_extraction_persistence_service,
        "authenticate_completed_fact_extraction_application",
        lambda *_args, **_kwargs: asyncio.sleep(
            0,
            result=snapshots_by_application_id[_kwargs["application_id"]],
        ),
    )


async def return_second_arg(*args, **kwargs):
    return args[1]


async def return_empty_tuple(*args, **kwargs):
    return ()


def make_consistency_candidate_application_ledger(
    *,
    source_application: DuplicateGroupingApplicationLedger,
    write_plan,
    consistency_application_id: uuid.UUID | None = None,
) -> FactValueConsistencyCandidateApplicationLedger:
    return FactValueConsistencyCandidateApplicationLedger(
        id=consistency_application_id or uuid.uuid4(),
        duplicate_grouping_application_id=source_application.id,
        orchestration_id=source_application.orchestration_id,
        extraction_run_id=source_application.extraction_run_id,
        algorithm_version=write_plan.algorithm_version,
        input_manifest_hash=write_plan.input_manifest_hash,
        result_manifest_hash=write_plan.result_manifest_hash,
        candidate_count=write_plan.candidate_count,
        member_count=write_plan.member_count,
        created_at=datetime.now(timezone.utc),
    )


def build_consistency_candidate_subledgers_from_plan(
    write_plan,
    *,
    application: FactValueConsistencyCandidateApplicationLedger,
) -> tuple[tuple[FactValueConsistencyCandidateLedger, ...], tuple[FactValueConsistencyCandidateMemberLedger, ...]]:
    candidate_ledgers: list[FactValueConsistencyCandidateLedger] = []
    member_ledgers: list[FactValueConsistencyCandidateMemberLedger] = []
    for candidate_plan in write_plan.candidates:
        candidate_id = uuid.uuid4()
        candidate_ledgers.append(
            FactValueConsistencyCandidateLedger(
                id=candidate_id,
                consistency_application_id=application.id,
                fact_id=candidate_plan.fact_id,
                candidate_kind=candidate_plan.candidate_kind,
                member_count=candidate_plan.member_count,
                distinct_semantic_key_count=candidate_plan.distinct_semantic_key_count,
                distinct_batch_count=candidate_plan.distinct_batch_count,
                created_at=datetime.now(timezone.utc),
            )
        )
        for member_plan in candidate_plan.members:
            member_ledgers.append(
                FactValueConsistencyCandidateMemberLedger(
                    id=uuid.uuid4(),
                    consistency_application_id=application.id,
                    candidate_id=candidate_id,
                    orchestration_id=application.orchestration_id,
                    fact_value_id=member_plan.fact_value_id,
                    source_batch_id=member_plan.source_batch_id,
                    semantic_key_hash=member_plan.semantic_key_hash,
                    created_at=datetime.now(timezone.utc),
                )
            )
    return tuple(candidate_ledgers), tuple(member_ledgers)


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
    state_a = make_orchestration_state(
        orchestration_a,
        extraction_run_id=extraction_run_id,
        orchestration_status="partial",
        batch_count=3,
        completed_batch_count=2,
        failed_batch_count=1,
    )
    state_b = make_orchestration_state(
        orchestration_b,
        extraction_run_id=extraction_run_id,
        orchestration_status="completed",
        batch_count=3,
        completed_batch_count=3,
        failed_batch_count=0,
    )

    async def fake_state(_session, *, orchestration_id):
        if orchestration_id == orchestration_a:
            return state_a
        return state_b

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
    install_authoritative_application_auth(
        monkeypatch,
        candidates_by_orchestration={
            orchestration_a: (candidate_a1, candidate_a2),
            orchestration_b: (candidate_b1, candidate_b2, candidate_b3),
        },
        state_by_orchestration={
            orchestration_a: state_a,
            orchestration_b: state_b,
        },
    )

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
    state = make_orchestration_state(
        orchestration_id,
        extraction_run_id=extraction_run_id,
        orchestration_status="completed",
        batch_count=2,
        completed_batch_count=2,
        failed_batch_count=0,
    )

    async def fake_state(_session, *, orchestration_id):
        return state

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
    install_authoritative_application_auth(
        monkeypatch,
        candidates_by_orchestration={orchestration_id: (first_candidate, second_candidate)},
        state_by_orchestration={orchestration_id: state},
    )

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
    state = make_orchestration_state(
        orchestration_id,
        extraction_run_id=extraction_run_id,
        orchestration_status="completed",
        batch_count=2,
        completed_batch_count=2,
        failed_batch_count=0,
    )

    async def fake_state(_session, *, orchestration_id):
        return state

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
    install_authoritative_application_auth(
        monkeypatch,
        candidates_by_orchestration={orchestration_id: (first_candidate, second_candidate)},
        state_by_orchestration={orchestration_id: state},
    )

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
    state = make_orchestration_state(
        orchestration_id,
        extraction_run_id=extraction_run_id,
        orchestration_status="completed",
        batch_count=2,
        completed_batch_count=2,
        failed_batch_count=0,
    )

    async def fake_state(_session, *, orchestration_id):
        return state

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
    install_authoritative_application_auth(
        monkeypatch,
        candidates_by_orchestration={orchestration_id: (first_candidate, second_candidate)},
        state_by_orchestration={orchestration_id: state},
    )

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
    state = make_orchestration_state(
        orchestration_id,
        extraction_run_id=extraction_run_id,
        orchestration_status="completed",
        batch_count=2,
        completed_batch_count=2,
        failed_batch_count=0,
    )

    async def fake_state(_session, *, orchestration_id):
        return state

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
    install_authoritative_application_auth(
        monkeypatch,
        candidates_by_orchestration={orchestration_id: (first_candidate, second_candidate)},
        state_by_orchestration={orchestration_id: state},
    )

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
    state = make_orchestration_state(
        orchestration_id,
        extraction_run_id=extraction_run_id,
        orchestration_status="completed",
        batch_count=2,
        completed_batch_count=2,
        failed_batch_count=0,
    )

    async def fake_state(_session, *, orchestration_id):
        return state

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
    install_authoritative_application_auth(
        monkeypatch,
        candidates_by_orchestration={orchestration_id: (first_candidate, second_candidate)},
        state_by_orchestration={orchestration_id: state},
    )

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
    state = make_orchestration_state(
        orchestration_id,
        extraction_run_id=extraction_run_id,
        orchestration_status="completed",
        batch_count=2,
        completed_batch_count=2,
        failed_batch_count=0,
    )

    async def fake_state(_session, *, orchestration_id):
        return state

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
    install_authoritative_application_auth(
        monkeypatch,
        candidates_by_orchestration={orchestration_id: (first_candidate, second_candidate)},
        state_by_orchestration={orchestration_id: state},
    )

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
    state = make_orchestration_state(
        orchestration_id,
        extraction_run_id=extraction_run_id,
        orchestration_status=orchestration_status,
        batch_count=2 if orchestration_status == "completed" else 3,
        completed_batch_count=2,
        failed_batch_count=0 if orchestration_status == "completed" else 1,
    )

    async def fake_state(_session, *, orchestration_id):
        return state

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
    install_authoritative_application_auth(
        monkeypatch,
        candidates_by_orchestration={orchestration_id: (first_candidate, second_candidate)},
        state_by_orchestration={orchestration_id: state},
    )

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
    matching_candidate = candidate(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        source_batch_id=uuid.uuid4(),
        evidence_ids=(uuid.uuid4(),),
    )
    state = make_orchestration_state(
        orchestration_id,
        extraction_run_id=extraction_run_id,
        orchestration_status="completed",
        batch_count=1,
        completed_batch_count=1,
        failed_batch_count=0,
    )

    async def fake_state(_session, *, orchestration_id):
        return state

    async def fake_invalid_bindings(_session, *, orchestration_id):
        return False

    async def fake_count(_session, *, orchestration_id):
        return 2

    async def fake_candidates(_session, *, orchestration_id):
        return (matching_candidate,)

    monkeypatch.setattr(duplicate_grouping_repository, "get_duplicate_grouping_orchestration_state", fake_state)
    monkeypatch.setattr(duplicate_grouping_repository, "has_invalid_completed_batch_bindings", fake_invalid_bindings)
    monkeypatch.setattr(duplicate_grouping_repository, "count_duplicate_candidate_fact_values", fake_count)
    monkeypatch.setattr(duplicate_grouping_repository, "list_duplicate_candidates", fake_candidates)
    install_authoritative_application_auth(
        monkeypatch,
        candidates_by_orchestration={orchestration_id: (matching_candidate,)},
        state_by_orchestration={orchestration_id: state},
    )

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
            batch_count=1,
            completed_batch_count=0,
            failed_batch_count=0,
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


def test_authenticate_duplicate_grouping_source_snapshot_accepts_matching_authoritative_application_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestration_id = uuid.uuid4()
    extraction_run_id = uuid.uuid4()
    project_id = uuid.uuid4()
    inference_run_id = uuid.uuid4()
    source_batch_id = uuid.uuid4()
    fact_value_id = uuid.uuid4()
    fact_id = uuid.uuid4()
    evidence_id = uuid.uuid4()
    evidence_link_id = uuid.uuid4()
    input_batch_id = uuid.uuid4()
    state = duplicate_grouping_repository.DuplicateGroupingOrchestrationState(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        project_id=project_id,
        extraction_run_status="completed",
        extraction_run_outcome="success",
        orchestration_status="completed",
        batch_count=1,
        completed_batch_count=1,
        failed_batch_count=0,
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
    source_candidate = candidate(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        fact_value_id=fact_value_id,
        fact_id=fact_id,
        source_batch_id=source_batch_id,
        value_type="string",
        value_json="same",
        evidence_link_ids=(evidence_link_id,),
        evidence_ids=(evidence_id,),
    )
    application_snapshot = make_application_snapshot(
        project_id=project_id,
        extraction_run_id=extraction_run_id,
        inference_run_id=inference_run_id,
        input_batch_id=input_batch_id,
        items=(
            AuthenticatedPersistedFactProposalItem(
                proposal_index=0,
                fact_id=fact_id,
                fact_value_id=fact_value_id,
                subject_entity_id=None,
                referenced_entity_id=None,
                evidence_ids=(evidence_id,),
            ),
        ),
    )

    monkeypatch.setattr(
        duplicate_grouping_repository,
        "get_duplicate_grouping_orchestration_state",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=state),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "has_invalid_completed_batch_bindings",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=False),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "list_completed_orchestration_batch_applications",
        lambda *_args, **_kwargs: asyncio.sleep(
            0,
            result=(
                make_completed_batch_application(
                    source_batch_id=source_batch_id,
                    batch_index=0,
                    application_id=application_snapshot.application_id,
                    current_input_batch_id=input_batch_id,
                    current_inference_run_id=inference_run_id,
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "count_duplicate_candidate_fact_values",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=1),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "list_duplicate_candidates",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=(source_candidate,)),
    )
    monkeypatch.setattr(
        duplicate_grouping_service.fact_extraction_persistence_service,
        "authenticate_completed_fact_extraction_application",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=application_snapshot),
    )
    session_factory = SessionFactory()

    snapshot = run_async(
        duplicate_grouping_service.authenticate_duplicate_grouping_source_snapshot(
            session_factory,
            orchestration_id=orchestration_id,
        )
    )

    assert snapshot.candidate_count == 1
    assert snapshot.candidates == (source_candidate,)
    assert snapshot.application_snapshots == (application_snapshot,)
    assert all(session.commit_count == 0 for session in session_factory.created_sessions)
    assert all(session.rollback_count == 1 for session in session_factory.created_sessions)


def test_authenticate_duplicate_grouping_source_snapshot_rejects_extra_ai_fact_value_outside_authoritative_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestration_id = uuid.uuid4()
    extraction_run_id = uuid.uuid4()
    project_id = uuid.uuid4()
    inference_run_id = uuid.uuid4()
    source_batch_id = uuid.uuid4()
    matching_fact_value_id = uuid.uuid4()
    extra_fact_value_id = uuid.uuid4()
    fact_id = uuid.uuid4()
    evidence_id = uuid.uuid4()
    input_batch_id = uuid.uuid4()
    state = duplicate_grouping_repository.DuplicateGroupingOrchestrationState(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        project_id=project_id,
        extraction_run_status="completed",
        extraction_run_outcome="success",
        orchestration_status="completed",
        batch_count=1,
        completed_batch_count=1,
        failed_batch_count=0,
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
    matching_candidate = candidate(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        fact_value_id=matching_fact_value_id,
        fact_id=fact_id,
        source_batch_id=source_batch_id,
        value_type="string",
        value_json="same",
        evidence_link_ids=(uuid.uuid4(),),
        evidence_ids=(evidence_id,),
    )
    extra_candidate = candidate(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        fact_value_id=extra_fact_value_id,
        fact_id=uuid.uuid4(),
        source_batch_id=source_batch_id,
        value_type="string",
        value_json="extra",
        evidence_link_ids=(uuid.uuid4(),),
        evidence_ids=(uuid.uuid4(),),
    )
    application_snapshot = make_application_snapshot(
        project_id=project_id,
        extraction_run_id=extraction_run_id,
        inference_run_id=inference_run_id,
        input_batch_id=input_batch_id,
        items=(
            AuthenticatedPersistedFactProposalItem(
                proposal_index=0,
                fact_id=fact_id,
                fact_value_id=matching_fact_value_id,
                subject_entity_id=None,
                referenced_entity_id=None,
                evidence_ids=(evidence_id,),
            ),
        ),
    )

    monkeypatch.setattr(
        duplicate_grouping_repository,
        "get_duplicate_grouping_orchestration_state",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=state),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "has_invalid_completed_batch_bindings",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=False),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "list_completed_orchestration_batch_applications",
        lambda *_args, **_kwargs: asyncio.sleep(
            0,
            result=(
                make_completed_batch_application(
                    source_batch_id=source_batch_id,
                    batch_index=0,
                    application_id=application_snapshot.application_id,
                    current_input_batch_id=input_batch_id,
                    current_inference_run_id=inference_run_id,
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "count_duplicate_candidate_fact_values",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=2),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "list_duplicate_candidates",
        lambda *_args, **_kwargs: asyncio.sleep(
            0,
            result=(matching_candidate, extra_candidate),
        ),
    )
    monkeypatch.setattr(
        duplicate_grouping_service.fact_extraction_persistence_service,
        "authenticate_completed_fact_extraction_application",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=application_snapshot),
    )

    with pytest.raises(
        duplicate_grouping_service.CrossBatchDuplicateGroupingInvariantError,
        match="candidate_source_mismatch",
    ):
        run_async(
            duplicate_grouping_service.authenticate_duplicate_grouping_source_snapshot(
                SessionFactory(),
                orchestration_id=orchestration_id,
            )
        )


@pytest.mark.parametrize(
    ("candidate_evidence_ids", "application_evidence_ids"),
    [
        ((), (uuid.uuid4(),)),
        ((uuid.UUID("00000000-0000-0000-0000-000000000001"), uuid.UUID("00000000-0000-0000-0000-000000000002")), (uuid.UUID("00000000-0000-0000-0000-000000000002"), uuid.UUID("00000000-0000-0000-0000-000000000001"))),
    ],
)
def test_authenticate_duplicate_grouping_source_snapshot_rejects_authoritative_membership_drift(
    monkeypatch: pytest.MonkeyPatch,
    candidate_evidence_ids: tuple[uuid.UUID, ...],
    application_evidence_ids: tuple[uuid.UUID, ...],
) -> None:
    orchestration_id = uuid.uuid4()
    extraction_run_id = uuid.uuid4()
    project_id = uuid.uuid4()
    inference_run_id = uuid.uuid4()
    source_batch_id = uuid.uuid4()
    fact_value_id = uuid.uuid4()
    fact_id = uuid.uuid4()
    input_batch_id = uuid.uuid4()
    state = duplicate_grouping_repository.DuplicateGroupingOrchestrationState(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        project_id=project_id,
        extraction_run_status="completed",
        extraction_run_outcome="success",
        orchestration_status="completed",
        batch_count=1,
        completed_batch_count=1,
        failed_batch_count=0,
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
    source_candidate = candidate(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        fact_value_id=fact_value_id,
        fact_id=fact_id,
        source_batch_id=source_batch_id,
        value_type="string",
        value_json="same",
        evidence_link_ids=(uuid.uuid4(),),
        evidence_ids=candidate_evidence_ids,
    )
    application_snapshot = make_application_snapshot(
        project_id=project_id,
        extraction_run_id=extraction_run_id,
        inference_run_id=inference_run_id,
        input_batch_id=input_batch_id,
        items=(
            AuthenticatedPersistedFactProposalItem(
                proposal_index=0,
                fact_id=fact_id,
                fact_value_id=fact_value_id,
                subject_entity_id=None,
                referenced_entity_id=None,
                evidence_ids=application_evidence_ids,
            ),
        ),
    )

    monkeypatch.setattr(
        duplicate_grouping_repository,
        "get_duplicate_grouping_orchestration_state",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=state),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "has_invalid_completed_batch_bindings",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=False),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "list_completed_orchestration_batch_applications",
        lambda *_args, **_kwargs: asyncio.sleep(
            0,
            result=(
                make_completed_batch_application(
                    source_batch_id=source_batch_id,
                    batch_index=0,
                    application_id=application_snapshot.application_id,
                    current_input_batch_id=input_batch_id,
                    current_inference_run_id=inference_run_id,
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "count_duplicate_candidate_fact_values",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=1),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "list_duplicate_candidates",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=(source_candidate,)),
    )
    monkeypatch.setattr(
        duplicate_grouping_service.fact_extraction_persistence_service,
        "authenticate_completed_fact_extraction_application",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=application_snapshot),
    )

    with pytest.raises(
        duplicate_grouping_service.CrossBatchDuplicateGroupingInvariantError,
        match="candidate_source_mismatch",
    ):
        run_async(
            duplicate_grouping_service.authenticate_duplicate_grouping_source_snapshot(
                SessionFactory(),
                orchestration_id=orchestration_id,
            )
        )


def test_authenticate_duplicate_grouping_source_snapshot_maps_application_replay_conflicts_to_redacted_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestration_id = uuid.uuid4()
    extraction_run_id = uuid.uuid4()
    state = duplicate_grouping_repository.DuplicateGroupingOrchestrationState(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        project_id=uuid.uuid4(),
        extraction_run_status="completed",
        extraction_run_outcome="success",
        orchestration_status="completed",
        batch_count=1,
        completed_batch_count=1,
        failed_batch_count=0,
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

    monkeypatch.setattr(
        duplicate_grouping_repository,
        "get_duplicate_grouping_orchestration_state",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=state),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "has_invalid_completed_batch_bindings",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=False),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "list_completed_orchestration_batch_applications",
        lambda *_args, **_kwargs: asyncio.sleep(
            0,
            result=(
                make_completed_batch_application(
                    batch_index=0,
                    application_id=uuid.uuid4(),
                    current_input_batch_id=uuid.uuid4(),
                    current_inference_run_id=uuid.uuid4(),
                ),
            ),
        ),
    )

    async def raise_replay_conflict(*_args, **_kwargs):
        raise duplicate_grouping_service.fact_extraction_persistence_service.FactExtractionApplicationReplayConflictError(
            "detailed internal mismatch"
        )

    monkeypatch.setattr(
        duplicate_grouping_service.fact_extraction_persistence_service,
        "authenticate_completed_fact_extraction_application",
        raise_replay_conflict,
    )

    with pytest.raises(
        duplicate_grouping_service.CrossBatchDuplicateGroupingInvariantError,
        match="application_result_mismatch",
    ) as exc_info:
        run_async(
            duplicate_grouping_service.authenticate_duplicate_grouping_source_snapshot(
                SessionFactory(),
                orchestration_id=orchestration_id,
            )
        )

    assert "detailed internal mismatch" not in str(exc_info.value)


@pytest.mark.parametrize(
    "field_name",
    [
        "planner_name",
        "planner_version",
        "agent_name",
        "agent_version",
        "prompt_contract_hash",
        "provider",
        "requested_model",
        "executor_name",
        "executor_version",
        "persistence_name",
        "persistence_version",
        "entity_resolution_policy_name",
        "entity_resolution_policy_version",
    ],
)
def test_authenticate_duplicate_grouping_source_snapshot_rejects_empty_or_invalid_identity_fields(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
) -> None:
    orchestration_id = uuid.uuid4()
    state = make_orchestration_state(
        orchestration_id,
        orchestration_status="completed",
        batch_count=1,
        completed_batch_count=1,
        failed_batch_count=0,
    )
    invalid_value = "" if field_name != "prompt_contract_hash" else "not-a-hash"
    state = replace(state, **{field_name: invalid_value})

    monkeypatch.setattr(
        duplicate_grouping_repository,
        "get_duplicate_grouping_orchestration_state",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=state),
    )

    with pytest.raises(
        duplicate_grouping_service.CrossBatchDuplicateGroupingInvariantError,
        match="orchestration_identity_invalid",
    ):
        run_async(
            duplicate_grouping_service.authenticate_duplicate_grouping_source_snapshot(
                SessionFactory(),
                orchestration_id=orchestration_id,
            )
        )


@pytest.mark.parametrize(
    ("orchestration_status", "batch_count", "completed_batch_count", "failed_batch_count"),
    [
        ("completed", 2, 1, 0),
        ("completed", 2, 2, 1),
        ("partial", 2, 0, 2),
        ("partial", 2, 1, 0),
        ("partial", 3, 1, 1),
    ],
)
def test_authenticate_duplicate_grouping_source_snapshot_rejects_invalid_batch_count_shapes(
    monkeypatch: pytest.MonkeyPatch,
    orchestration_status: str,
    batch_count: int,
    completed_batch_count: int,
    failed_batch_count: int,
) -> None:
    orchestration_id = uuid.uuid4()
    state = make_orchestration_state(
        orchestration_id,
        orchestration_status=orchestration_status,
        batch_count=batch_count,
        completed_batch_count=completed_batch_count,
        failed_batch_count=failed_batch_count,
    )

    monkeypatch.setattr(
        duplicate_grouping_repository,
        "get_duplicate_grouping_orchestration_state",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=state),
    )

    with pytest.raises(
        duplicate_grouping_service.CrossBatchDuplicateGroupingInvariantError,
        match="batch_count_mismatch",
    ):
        run_async(
            duplicate_grouping_service.authenticate_duplicate_grouping_source_snapshot(
                SessionFactory(),
                orchestration_id=orchestration_id,
            )
        )


@pytest.mark.parametrize(
    ("completed_batch_count", "rows", "expected_code"),
    [
        (
            2,
            lambda app_id, input_batch_id, inference_run_id: (
                make_completed_batch_application(
                    batch_index=0,
                    application_id=app_id,
                    current_input_batch_id=input_batch_id,
                    current_inference_run_id=inference_run_id,
                ),
            ),
            "batch_count_mismatch",
        ),
        (
            1,
            lambda app_id, input_batch_id, inference_run_id: (
                make_completed_batch_application(
                    source_batch_id=uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                    batch_index=0,
                    application_id=app_id,
                    current_input_batch_id=input_batch_id,
                    current_inference_run_id=inference_run_id,
                ),
                make_completed_batch_application(
                    source_batch_id=uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
                    batch_index=1,
                    application_id=app_id,
                    current_input_batch_id=input_batch_id,
                    current_inference_run_id=inference_run_id,
                ),
            ),
            "batch_count_mismatch",
        ),
        (
            2,
            lambda app_id, input_batch_id, inference_run_id: (
                make_completed_batch_application(
                    source_batch_id=uuid.UUID("cccccccc-cccc-cccc-cccc-cccccccccccc"),
                    batch_index=0,
                    application_id=app_id,
                    current_input_batch_id=input_batch_id,
                    current_inference_run_id=inference_run_id,
                ),
                make_completed_batch_application(
                    source_batch_id=uuid.UUID("cccccccc-cccc-cccc-cccc-cccccccccccc"),
                    batch_index=1,
                    application_id=app_id,
                    current_input_batch_id=input_batch_id,
                    current_inference_run_id=inference_run_id,
                ),
            ),
            "completed_batch_source_mismatch",
        ),
    ],
)
def test_authenticate_duplicate_grouping_source_snapshot_rejects_missing_extra_or_duplicate_completed_batches(
    monkeypatch: pytest.MonkeyPatch,
    completed_batch_count: int,
    rows,
    expected_code: str,
) -> None:
    orchestration_id = uuid.uuid4()
    state = make_orchestration_state(
        orchestration_id,
        orchestration_status="completed",
        batch_count=max(1, completed_batch_count),
        completed_batch_count=completed_batch_count,
        failed_batch_count=0,
    )
    application_snapshot = make_application_snapshot(
        project_id=state.project_id,
        extraction_run_id=state.extraction_run_id,
        inference_run_id=uuid.uuid4(),
        input_batch_id=uuid.uuid4(),
        items=(),
        persistence_name=state.persistence_name,
        persistence_version=state.persistence_version,
        entity_resolution_policy_name=state.entity_resolution_policy_name,
        entity_resolution_policy_version=state.entity_resolution_policy_version,
    )

    monkeypatch.setattr(
        duplicate_grouping_repository,
        "get_duplicate_grouping_orchestration_state",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=state),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "has_invalid_completed_batch_bindings",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=False),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "list_completed_orchestration_batch_applications",
        lambda *_args, **_kwargs: asyncio.sleep(
            0,
            result=rows(
                application_snapshot.application_id,
                application_snapshot.input_batch_id,
                application_snapshot.inference_run_id,
            ),
        ),
    )

    with pytest.raises(
        duplicate_grouping_service.CrossBatchDuplicateGroupingInvariantError,
        match=expected_code,
    ):
        run_async(
            duplicate_grouping_service.authenticate_duplicate_grouping_source_snapshot(
                SessionFactory(),
                orchestration_id=orchestration_id,
            )
        )


@pytest.mark.parametrize(
    ("batch_count", "completed_batch_count", "rows"),
    [
        (
            2,
            2,
            lambda app_id, input_batch_id, inference_run_id: (
                make_completed_batch_application(
                    batch_index=0,
                    application_id=app_id,
                    current_input_batch_id=input_batch_id,
                    current_inference_run_id=inference_run_id,
                ),
                make_completed_batch_application(
                    batch_index=0,
                    application_id=app_id,
                    current_input_batch_id=input_batch_id,
                    current_inference_run_id=inference_run_id,
                ),
            ),
        ),
        (
            1,
            1,
            lambda app_id, input_batch_id, inference_run_id: (
                make_completed_batch_application(
                    batch_index=2,
                    application_id=app_id,
                    current_input_batch_id=input_batch_id,
                    current_inference_run_id=inference_run_id,
                ),
            ),
        ),
    ],
)
def test_authenticate_duplicate_grouping_source_snapshot_rejects_duplicate_or_out_of_range_batch_index(
    monkeypatch: pytest.MonkeyPatch,
    batch_count: int,
    completed_batch_count: int,
    rows,
) -> None:
    orchestration_id = uuid.uuid4()
    state = make_orchestration_state(
        orchestration_id,
        orchestration_status="completed",
        batch_count=batch_count,
        completed_batch_count=completed_batch_count,
        failed_batch_count=0,
    )
    application_snapshot = make_application_snapshot(
        project_id=state.project_id,
        extraction_run_id=state.extraction_run_id,
        inference_run_id=uuid.uuid4(),
        input_batch_id=uuid.uuid4(),
        items=(),
        persistence_name=state.persistence_name,
        persistence_version=state.persistence_version,
        entity_resolution_policy_name=state.entity_resolution_policy_name,
        entity_resolution_policy_version=state.entity_resolution_policy_version,
    )

    monkeypatch.setattr(
        duplicate_grouping_repository,
        "get_duplicate_grouping_orchestration_state",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=state),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "has_invalid_completed_batch_bindings",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=False),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "list_completed_orchestration_batch_applications",
        lambda *_args, **_kwargs: asyncio.sleep(
            0,
            result=rows(
                application_snapshot.application_id,
                application_snapshot.input_batch_id,
                application_snapshot.inference_run_id,
            ),
        ),
    )

    with pytest.raises(
        duplicate_grouping_service.CrossBatchDuplicateGroupingInvariantError,
        match="completed_batch_source_mismatch",
    ):
        run_async(
            duplicate_grouping_service.authenticate_duplicate_grouping_source_snapshot(
                SessionFactory(),
                orchestration_id=orchestration_id,
            )
        )


def test_authenticate_duplicate_grouping_source_snapshot_rejects_missing_or_mismatched_input_batch_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestration_id = uuid.uuid4()
    state = make_orchestration_state(
        orchestration_id,
        orchestration_status="completed",
        batch_count=1,
        completed_batch_count=1,
        failed_batch_count=0,
    )
    application_snapshot = make_application_snapshot(
        project_id=state.project_id,
        extraction_run_id=state.extraction_run_id,
        inference_run_id=uuid.uuid4(),
        input_batch_id=uuid.uuid4(),
        items=(),
        persistence_name=state.persistence_name,
        persistence_version=state.persistence_version,
        entity_resolution_policy_name=state.entity_resolution_policy_name,
        entity_resolution_policy_version=state.entity_resolution_policy_version,
    )

    monkeypatch.setattr(
        duplicate_grouping_repository,
        "get_duplicate_grouping_orchestration_state",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=state),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "has_invalid_completed_batch_bindings",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=False),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "list_completed_orchestration_batch_applications",
        lambda *_args, **_kwargs: asyncio.sleep(
            0,
            result=(
                make_completed_batch_application(
                    batch_index=0,
                    application_id=application_snapshot.application_id,
                    current_input_batch_id=uuid.uuid4(),
                    current_inference_run_id=application_snapshot.inference_run_id,
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "count_duplicate_candidate_fact_values",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=0),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "list_duplicate_candidates",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=()),
    )
    monkeypatch.setattr(
        duplicate_grouping_service.fact_extraction_persistence_service,
        "authenticate_completed_fact_extraction_application",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=application_snapshot),
    )

    with pytest.raises(
        duplicate_grouping_service.CrossBatchDuplicateGroupingInvariantError,
        match="application_input_batch_mismatch",
    ):
        run_async(
            duplicate_grouping_service.authenticate_duplicate_grouping_source_snapshot(
                SessionFactory(),
                orchestration_id=orchestration_id,
            )
        )


def test_authenticate_duplicate_grouping_source_snapshot_rejects_missing_current_input_batch_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestration_id = uuid.uuid4()
    state = make_orchestration_state(
        orchestration_id,
        orchestration_status="completed",
        batch_count=1,
        completed_batch_count=1,
        failed_batch_count=0,
    )

    monkeypatch.setattr(
        duplicate_grouping_repository,
        "get_duplicate_grouping_orchestration_state",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=state),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "has_invalid_completed_batch_bindings",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=False),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "list_completed_orchestration_batch_applications",
        lambda *_args, **_kwargs: asyncio.sleep(
            0,
            result=(
                make_completed_batch_application(
                    batch_index=0,
                    application_id=uuid.uuid4(),
                    current_input_batch_id=None,
                    current_inference_run_id=uuid.uuid4(),
                ),
            ),
        ),
    )

    with pytest.raises(
        duplicate_grouping_service.CrossBatchDuplicateGroupingInvariantError,
        match="completed_batch_source_mismatch",
    ):
        run_async(
            duplicate_grouping_service.authenticate_duplicate_grouping_source_snapshot(
                SessionFactory(),
                orchestration_id=orchestration_id,
            )
        )


def test_authenticate_duplicate_grouping_source_snapshot_rejects_current_inference_run_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestration_id = uuid.uuid4()
    state = make_orchestration_state(
        orchestration_id,
        orchestration_status="completed",
        batch_count=1,
        completed_batch_count=1,
        failed_batch_count=0,
    )
    application_snapshot = make_application_snapshot(
        project_id=state.project_id,
        extraction_run_id=state.extraction_run_id,
        inference_run_id=uuid.uuid4(),
        input_batch_id=uuid.uuid4(),
        items=(),
        persistence_name=state.persistence_name,
        persistence_version=state.persistence_version,
        entity_resolution_policy_name=state.entity_resolution_policy_name,
        entity_resolution_policy_version=state.entity_resolution_policy_version,
    )

    monkeypatch.setattr(
        duplicate_grouping_repository,
        "get_duplicate_grouping_orchestration_state",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=state),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "has_invalid_completed_batch_bindings",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=False),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "list_completed_orchestration_batch_applications",
        lambda *_args, **_kwargs: asyncio.sleep(
            0,
            result=(
                make_completed_batch_application(
                    batch_index=0,
                    application_id=application_snapshot.application_id,
                    current_input_batch_id=application_snapshot.input_batch_id,
                    current_inference_run_id=uuid.uuid4(),
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "count_duplicate_candidate_fact_values",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=0),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "list_duplicate_candidates",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=()),
    )
    monkeypatch.setattr(
        duplicate_grouping_service.fact_extraction_persistence_service,
        "authenticate_completed_fact_extraction_application",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=application_snapshot),
    )

    with pytest.raises(
        duplicate_grouping_service.CrossBatchDuplicateGroupingInvariantError,
        match="application_inference_run_mismatch",
    ):
        run_async(
            duplicate_grouping_service.authenticate_duplicate_grouping_source_snapshot(
                SessionFactory(),
                orchestration_id=orchestration_id,
            )
        )


def test_authenticate_duplicate_grouping_source_snapshot_accepts_all_withheld_application_with_zero_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestration_id = uuid.uuid4()
    state = make_orchestration_state(
        orchestration_id,
        orchestration_status="completed",
        batch_count=1,
        completed_batch_count=1,
        failed_batch_count=0,
    )
    application_snapshot = make_application_snapshot(
        project_id=state.project_id,
        extraction_run_id=state.extraction_run_id,
        inference_run_id=uuid.uuid4(),
        input_batch_id=uuid.uuid4(),
        items=(),
        persistence_name=state.persistence_name,
        persistence_version=state.persistence_version,
        entity_resolution_policy_name=state.entity_resolution_policy_name,
        entity_resolution_policy_version=state.entity_resolution_policy_version,
    )

    monkeypatch.setattr(
        duplicate_grouping_repository,
        "get_duplicate_grouping_orchestration_state",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=state),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "has_invalid_completed_batch_bindings",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=False),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "list_completed_orchestration_batch_applications",
        lambda *_args, **_kwargs: asyncio.sleep(
            0,
            result=(
                make_completed_batch_application(
                    batch_index=0,
                    source_batch_id=uuid.uuid4(),
                    application_id=application_snapshot.application_id,
                    current_input_batch_id=application_snapshot.input_batch_id,
                    current_inference_run_id=application_snapshot.inference_run_id,
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "count_duplicate_candidate_fact_values",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=0),
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "list_duplicate_candidates",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=()),
    )
    monkeypatch.setattr(
        duplicate_grouping_service.fact_extraction_persistence_service,
        "authenticate_completed_fact_extraction_application",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=application_snapshot),
    )

    snapshot = run_async(
        duplicate_grouping_service.authenticate_duplicate_grouping_source_snapshot(
            SessionFactory(),
            orchestration_id=orchestration_id,
        )
    )

    assert snapshot.candidate_count == 0
    assert snapshot.candidates == ()
    assert snapshot.application_snapshots == (application_snapshot,)


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
    monkeypatch.setattr(duplicate_grouping_repository, "list_consistency_candidate_ledgers", return_empty_tuple)
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "list_consistency_candidate_member_ledgers",
        return_empty_tuple,
    )

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
    existing_candidate_ledgers, existing_member_ledgers = build_consistency_candidate_subledgers_from_plan(
        write_plan,
        application=existing_application,
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
    async def fake_candidate_ledgers(_session, *, consistency_application_id):
        return existing_candidate_ledgers

    async def fake_member_ledgers(_session, *, consistency_application_id):
        return existing_member_ledgers

    monkeypatch.setattr(
        duplicate_grouping_repository,
        "list_consistency_candidate_ledgers",
        fake_candidate_ledgers,
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "list_consistency_candidate_member_ledgers",
        fake_member_ledgers,
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


@pytest.mark.parametrize("source_algorithm_version", ["cross_batch_exact_v1", "cross_batch_exact_v9"])
def test_build_fact_value_consistency_candidate_write_plan_rejects_unsupported_source_algorithm(
    source_algorithm_version: str,
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
        algorithm_version=source_algorithm_version,
        input_manifest_hash=base_application.input_manifest_hash,
        result_manifest_hash=base_application.result_manifest_hash,
        input_fact_value_count=base_application.input_fact_value_count,
        duplicate_group_count=base_application.duplicate_group_count,
        duplicate_member_count=base_application.duplicate_member_count,
        created_at=base_application.created_at,
    )

    with pytest.raises(
        duplicate_grouping_service.FactValueConsistencyCandidateStateError,
        match="fact_value_consistency_candidate_source_algorithm_unsupported",
    ):
        duplicate_grouping_service.build_fact_value_consistency_candidate_write_plan(
            candidates,
            source_duplicate_grouping_application=source_application,
        )


@pytest.mark.parametrize("source_algorithm_version", ["cross_batch_exact_v1", "cross_batch_exact_v9"])
def test_ensure_cross_batch_multi_value_consistency_candidates_rejects_unsupported_source_algorithm(
    monkeypatch: pytest.MonkeyPatch,
    source_algorithm_version: str,
) -> None:
    source_application = DuplicateGroupingApplicationLedger(
        id=uuid.uuid4(),
        orchestration_id=uuid.uuid4(),
        extraction_run_id=uuid.uuid4(),
        algorithm_version=source_algorithm_version,
        input_manifest_hash="a" * 64,
        result_manifest_hash="b" * 64,
        input_fact_value_count=0,
        duplicate_group_count=0,
        duplicate_member_count=0,
        created_at=datetime.now(timezone.utc),
    )
    session_factory = SessionFactory([FakeSession()])

    async def fake_source_application(_session, *, grouping_application_id):
        return source_application

    monkeypatch.setattr(duplicate_grouping_repository, "get_grouping_application_ledger_by_id", fake_source_application)

    with pytest.raises(
        duplicate_grouping_service.FactValueConsistencyCandidateStateError,
        match="fact_value_consistency_candidate_source_algorithm_unsupported",
    ):
        run_async(
            duplicate_grouping_service.ensure_cross_batch_multi_value_consistency_candidates(
                session_factory,
                duplicate_grouping_application_id=source_application.id,
            )
        )


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("algorithm_version", "cross_batch_exact_v9"),
        ("input_manifest_hash", "f" * 64),
        ("input_fact_value_count", 999),
    ],
)
def test_ensure_cross_batch_multi_value_consistency_candidates_fails_closed_on_write_session_source_drift(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    changed_value,
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
    read_source_application = make_duplicate_grouping_application_ledger(candidates)
    write_source_application = DuplicateGroupingApplicationLedger(
        id=read_source_application.id,
        orchestration_id=read_source_application.orchestration_id,
        extraction_run_id=read_source_application.extraction_run_id,
        algorithm_version=read_source_application.algorithm_version,
        input_manifest_hash=read_source_application.input_manifest_hash,
        result_manifest_hash=read_source_application.result_manifest_hash,
        input_fact_value_count=read_source_application.input_fact_value_count,
        duplicate_group_count=read_source_application.duplicate_group_count,
        duplicate_member_count=read_source_application.duplicate_member_count,
        created_at=read_source_application.created_at,
    )
    write_source_application = DuplicateGroupingApplicationLedger(
        id=write_source_application.id,
        orchestration_id=write_source_application.orchestration_id,
        extraction_run_id=write_source_application.extraction_run_id,
        algorithm_version=changed_value if field_name == "algorithm_version" else write_source_application.algorithm_version,
        input_manifest_hash=changed_value if field_name == "input_manifest_hash" else write_source_application.input_manifest_hash,
        result_manifest_hash=write_source_application.result_manifest_hash,
        input_fact_value_count=changed_value if field_name == "input_fact_value_count" else write_source_application.input_fact_value_count,
        duplicate_group_count=write_source_application.duplicate_group_count,
        duplicate_member_count=write_source_application.duplicate_member_count,
        created_at=write_source_application.created_at,
    )
    session_factory = SessionFactory([FakeSession(), FakeSession()])
    source_reads = [read_source_application, write_source_application]

    async def fake_source_application(_session, *, grouping_application_id):
        return source_reads.pop(0)

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
        match="fact_value_consistency_candidate_source_snapshot_mismatch",
    ):
        run_async(
            duplicate_grouping_service.ensure_cross_batch_multi_value_consistency_candidates(
                session_factory,
                duplicate_grouping_application_id=read_source_application.id,
            )
        )


def test_ensure_cross_batch_multi_value_consistency_candidates_returns_existing_valid_non_zero_ledger(
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
    existing_application = make_consistency_candidate_application_ledger(
        source_application=source_application,
        write_plan=write_plan,
    )
    existing_candidate_ledgers, existing_member_ledgers = build_consistency_candidate_subledgers_from_plan(
        write_plan,
        application=existing_application,
    )
    session_factory = SessionFactory([FakeSession(), FakeSession()])

    async def fake_source_application(_session, *, grouping_application_id):
        return source_application

    async def fake_state(_session, *, orchestration_id):
        return make_orchestration_state(orchestration_id, extraction_run_id=extraction_run_id)

    async def fake_invalid_bindings(_session, *, orchestration_id):
        return False

    async def fake_count(_session, *, orchestration_id):
        return len(candidates)

    async def fake_candidates(_session, *, orchestration_id):
        return candidates

    async def fake_existing_for_update(_session, *, duplicate_grouping_application_id, algorithm_version):
        return type(
            "ExistingConsistencyApplication",
            (),
            {
                "id": existing_application.id,
                "duplicate_grouping_application_id": existing_application.duplicate_grouping_application_id,
                "orchestration_id": existing_application.orchestration_id,
                "extraction_run_id": existing_application.extraction_run_id,
                "algorithm_version": existing_application.algorithm_version,
                "input_manifest_hash": existing_application.input_manifest_hash,
                "result_manifest_hash": existing_application.result_manifest_hash,
                "candidate_count": existing_application.candidate_count,
                "member_count": existing_application.member_count,
                "created_at": existing_application.created_at,
            },
        )()

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
    async def fake_candidate_ledgers(_session, *, consistency_application_id):
        return existing_candidate_ledgers

    async def fake_member_ledgers(_session, *, consistency_application_id):
        return existing_member_ledgers

    monkeypatch.setattr(
        duplicate_grouping_repository,
        "list_consistency_candidate_ledgers",
        fake_candidate_ledgers,
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "list_consistency_candidate_member_ledgers",
        fake_member_ledgers,
    )

    result = run_async(
        duplicate_grouping_service.ensure_cross_batch_multi_value_consistency_candidates(
            session_factory,
            duplicate_grouping_application_id=source_application.id,
        )
    )

    assert result.created_new is False
    assert result.consistency_application_id == existing_application.id
    assert session_factory.created_sessions[1].commit_count == 1


@pytest.mark.parametrize("corruption_kind", ["missing_candidate", "extra_member", "member_semantic_hash", "member_source_batch", "member_candidate_binding", "candidate_stats"])
def test_ensure_cross_batch_multi_value_consistency_candidates_fails_closed_on_existing_subledger_corruption(
    monkeypatch: pytest.MonkeyPatch,
    corruption_kind: str,
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
    existing_application = make_consistency_candidate_application_ledger(
        source_application=source_application,
        write_plan=write_plan,
    )
    candidate_ledgers, member_ledgers = build_consistency_candidate_subledgers_from_plan(
        write_plan,
        application=existing_application,
    )
    candidate_ledgers = list(candidate_ledgers)
    member_ledgers = list(member_ledgers)
    if corruption_kind == "missing_candidate":
        candidate_ledgers = []
    elif corruption_kind == "extra_member":
        member_ledgers.append(
            FactValueConsistencyCandidateMemberLedger(
                id=uuid.uuid4(),
                consistency_application_id=existing_application.id,
                candidate_id=candidate_ledgers[0].id,
                orchestration_id=existing_application.orchestration_id,
                fact_value_id=uuid.uuid4(),
                source_batch_id=uuid.uuid4(),
                semantic_key_hash="a" * 64,
                created_at=datetime.now(timezone.utc),
            )
        )
    elif corruption_kind == "member_semantic_hash":
        member_ledgers[0] = FactValueConsistencyCandidateMemberLedger(
            id=member_ledgers[0].id,
            consistency_application_id=member_ledgers[0].consistency_application_id,
            candidate_id=member_ledgers[0].candidate_id,
            orchestration_id=member_ledgers[0].orchestration_id,
            fact_value_id=member_ledgers[0].fact_value_id,
            source_batch_id=member_ledgers[0].source_batch_id,
            semantic_key_hash="f" * 64,
            created_at=member_ledgers[0].created_at,
        )
    elif corruption_kind == "member_source_batch":
        member_ledgers[0] = FactValueConsistencyCandidateMemberLedger(
            id=member_ledgers[0].id,
            consistency_application_id=member_ledgers[0].consistency_application_id,
            candidate_id=member_ledgers[0].candidate_id,
            orchestration_id=member_ledgers[0].orchestration_id,
            fact_value_id=member_ledgers[0].fact_value_id,
            source_batch_id=uuid.uuid4(),
            semantic_key_hash=member_ledgers[0].semantic_key_hash,
            created_at=member_ledgers[0].created_at,
        )
    elif corruption_kind == "member_candidate_binding":
        member_ledgers[0] = FactValueConsistencyCandidateMemberLedger(
            id=member_ledgers[0].id,
            consistency_application_id=member_ledgers[0].consistency_application_id,
            candidate_id=uuid.uuid4(),
            orchestration_id=member_ledgers[0].orchestration_id,
            fact_value_id=member_ledgers[0].fact_value_id,
            source_batch_id=member_ledgers[0].source_batch_id,
            semantic_key_hash=member_ledgers[0].semantic_key_hash,
            created_at=member_ledgers[0].created_at,
        )
    elif corruption_kind == "candidate_stats":
        candidate_ledgers[0] = FactValueConsistencyCandidateLedger(
            id=candidate_ledgers[0].id,
            consistency_application_id=candidate_ledgers[0].consistency_application_id,
            fact_id=candidate_ledgers[0].fact_id,
            candidate_kind=candidate_ledgers[0].candidate_kind,
            member_count=99,
            distinct_semantic_key_count=candidate_ledgers[0].distinct_semantic_key_count,
            distinct_batch_count=candidate_ledgers[0].distinct_batch_count,
            created_at=candidate_ledgers[0].created_at,
        )

    session_factory = SessionFactory([FakeSession(), FakeSession()])

    async def fake_source_application(_session, *, grouping_application_id):
        return source_application

    async def fake_state(_session, *, orchestration_id):
        return make_orchestration_state(orchestration_id, extraction_run_id=extraction_run_id)

    async def fake_invalid_bindings(_session, *, orchestration_id):
        return False

    async def fake_count(_session, *, orchestration_id):
        return len(candidates)

    async def fake_candidates(_session, *, orchestration_id):
        return candidates

    async def fake_existing_for_update(_session, *, duplicate_grouping_application_id, algorithm_version):
        return type(
            "ExistingConsistencyApplication",
            (),
            {
                "id": existing_application.id,
                "duplicate_grouping_application_id": existing_application.duplicate_grouping_application_id,
                "orchestration_id": existing_application.orchestration_id,
                "extraction_run_id": existing_application.extraction_run_id,
                "algorithm_version": existing_application.algorithm_version,
                "input_manifest_hash": existing_application.input_manifest_hash,
                "result_manifest_hash": existing_application.result_manifest_hash,
                "candidate_count": existing_application.candidate_count,
                "member_count": existing_application.member_count,
                "created_at": existing_application.created_at,
            },
        )()

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
    async def fake_candidate_ledgers(_session, *, consistency_application_id):
        return tuple(candidate_ledgers)

    async def fake_member_ledgers(_session, *, consistency_application_id):
        return tuple(member_ledgers)

    monkeypatch.setattr(
        duplicate_grouping_repository,
        "list_consistency_candidate_ledgers",
        fake_candidate_ledgers,
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "list_consistency_candidate_member_ledgers",
        fake_member_ledgers,
    )

    with pytest.raises(
        duplicate_grouping_service.FactValueConsistencyCandidateInvariantError,
        match="fact_value_consistency_candidate_subledger_mismatch",
    ):
        run_async(
            duplicate_grouping_service.ensure_cross_batch_multi_value_consistency_candidates(
                session_factory,
                duplicate_grouping_application_id=source_application.id,
            )
        )


def test_ensure_cross_batch_multi_value_consistency_candidates_concurrency_readback_revalidates_subledgers(
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
    existing_application = make_consistency_candidate_application_ledger(
        source_application=source_application,
        write_plan=write_plan,
    )
    existing_candidate_ledgers, existing_member_ledgers = build_consistency_candidate_subledgers_from_plan(
        write_plan,
        application=existing_application,
    )
    corrupted_member_ledgers = (
        FactValueConsistencyCandidateMemberLedger(
            id=existing_member_ledgers[0].id,
            consistency_application_id=existing_member_ledgers[0].consistency_application_id,
            candidate_id=existing_member_ledgers[0].candidate_id,
            orchestration_id=existing_member_ledgers[0].orchestration_id,
            fact_value_id=existing_member_ledgers[0].fact_value_id,
            source_batch_id=existing_member_ledgers[0].source_batch_id,
            semantic_key_hash="f" * 64,
            created_at=existing_member_ledgers[0].created_at,
        ),
        existing_member_ledgers[1],
    )
    session_factory = SessionFactory([FakeSession(), FakeSession(), FakeSession()])

    async def fake_source_application(_session, *, grouping_application_id):
        return source_application

    async def fake_state(_session, *, orchestration_id):
        return make_orchestration_state(orchestration_id, extraction_run_id=extraction_run_id)

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
    async def fake_candidate_ledgers(_session, *, consistency_application_id):
        return existing_candidate_ledgers

    async def fake_member_ledgers(_session, *, consistency_application_id):
        return corrupted_member_ledgers

    monkeypatch.setattr(
        duplicate_grouping_repository,
        "list_consistency_candidate_ledgers",
        fake_candidate_ledgers,
    )
    monkeypatch.setattr(
        duplicate_grouping_repository,
        "list_consistency_candidate_member_ledgers",
        fake_member_ledgers,
    )

    with pytest.raises(
        duplicate_grouping_service.FactValueConsistencyCandidateInvariantError,
        match="fact_value_consistency_candidate_subledger_mismatch",
    ):
        run_async(
            duplicate_grouping_service.ensure_cross_batch_multi_value_consistency_candidates(
                session_factory,
                duplicate_grouping_application_id=source_application.id,
            )
        )
