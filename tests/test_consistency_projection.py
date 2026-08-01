from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
from datetime import datetime, timezone
from types import SimpleNamespace
import uuid

import pytest

from app.schemas.agent_consistency_check import ConsistencyCheckAssessment
from app.schemas.agent_consistency_check import ConsistencyCheckResponse
from app.schemas.consistency_check_execution import (
    ConsistencyCheckBatchExecutionResult,
)
from app.schemas.consistency_check_persistence import (
    ConsistencyAssessmentCitationLedgerRecord,
    ConsistencyAssessmentLedgerRecord,
    ConsistencyCheckApplicationLedgerRecord,
    ConsistencyCheckBatchLedgerRecord,
)
from app.schemas.fact_value_duplicate_grouping import (
    FactValueConsistencyCandidateApplicationLedger,
    FactValueConsistencyCandidateLedger,
    FactValueConsistencyCandidateMemberLedger,
)
from app.repositories.consistency_projection import (
    ConsistencyProjectionSnapshot,
    ConsistencyProjectionSourceRow,
)
from app.services import consistency_check as consistency_check_service
from app.services import consistency_check_execution as execution_service
from app.services import consistency_projection as projection_service
from app.services import consistency_review as review_service
from app.services import fact_value_duplicate_grouping as duplicate_grouping_service


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
        self.open_count = 0

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


def _uuid(seed: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, seed)


def _assessment_hash(
    *,
    candidate_id: uuid.UUID,
    verdict: str,
    severity: str,
    confidence: float,
    explanation: str,
    cited_evidence_link_ids: list[uuid.UUID],
    impact: list[str],
    recommended_actions: list[str],
) -> tuple[ConsistencyCheckAssessment, str]:
    assessment = ConsistencyCheckAssessment(
        candidate_id=candidate_id,
        verdict=verdict,
        severity=severity,
        confidence=confidence,
        explanation=explanation,
        cited_evidence_link_ids=cited_evidence_link_ids,
        impact=impact,
        recommended_actions=recommended_actions,
    )
    return (
        assessment,
        duplicate_grouping_service.hash_deterministic_payload(
            execution_service._assessment_manifest_payload(assessment)
        ),
    )


def _decision_manifest(
    *,
    project_id: uuid.UUID,
    application_id: uuid.UUID,
    assessment_id: uuid.UUID,
    source_application_id: uuid.UUID,
    source_candidate_id: uuid.UUID,
    actor_id: uuid.UUID,
    decision_no: int,
    supersedes_decision_id: uuid.UUID | None,
    decision_kind: str,
    comment: str | None,
    selected_fact_value_ids: tuple[uuid.UUID, ...],
) -> str:
    return review_service._build_decision_manifest_hash(
        project_id=project_id,
        consistency_check_application_id=application_id,
        assessment_id=assessment_id,
        source_consistency_application_id=source_application_id,
        source_consistency_candidate_id=source_candidate_id,
        actor_id=actor_id,
        decision_no=decision_no,
        supersedes_decision_id=supersedes_decision_id,
        decision_kind=decision_kind,
        comment=comment,
        selected_fact_value_ids=selected_fact_value_ids,
    )


def _decision_record(
    *,
    decision_id: uuid.UUID,
    project_id: uuid.UUID,
    application_id: uuid.UUID,
    assessment_id: uuid.UUID,
    source_application_id: uuid.UUID,
    source_candidate_id: uuid.UUID,
    actor_id: uuid.UUID,
    decision_no: int,
    supersedes_decision_id: uuid.UUID | None,
    decision_kind: str,
    comment: str | None,
    selected_fact_value_ids: tuple[uuid.UUID, ...],
    created_at: datetime,
):
    from app.schemas.consistency_review import (
        ConsistencyReviewDecisionLedgerRecord,
        ConsistencyReviewDecisionSelectionLedgerRecord,
    )

    decision = ConsistencyReviewDecisionLedgerRecord(
        id=decision_id,
        project_id=project_id,
        consistency_check_application_id=application_id,
        assessment_id=assessment_id,
        source_consistency_application_id=source_application_id,
        source_consistency_candidate_id=source_candidate_id,
        actor_id=actor_id,
        decision_no=decision_no,
        supersedes_decision_id=supersedes_decision_id,
        decision_kind=decision_kind,
        selected_value_count=len(selected_fact_value_ids),
        comment=comment,
        decision_manifest_hash=_decision_manifest(
            project_id=project_id,
            application_id=application_id,
            assessment_id=assessment_id,
            source_application_id=source_application_id,
            source_candidate_id=source_candidate_id,
            actor_id=actor_id,
            decision_no=decision_no,
            supersedes_decision_id=supersedes_decision_id,
            decision_kind=decision_kind,
            comment=comment,
            selected_fact_value_ids=selected_fact_value_ids,
        ),
        created_at=created_at,
    )
    selections = tuple(
        ConsistencyReviewDecisionSelectionLedgerRecord(
            id=_uuid(f"{decision_id}-selection-{selection_order}"),
            decision_id=decision_id,
            assessment_id=assessment_id,
            source_consistency_application_id=source_application_id,
            source_consistency_candidate_id=source_candidate_id,
            fact_value_id=fact_value_id,
            selection_order=selection_order,
            created_at=created_at,
        )
        for selection_order, fact_value_id in enumerate(selected_fact_value_ids)
    )
    return decision, selections


def _source_row(
    *,
    candidate_id: uuid.UUID,
    fact_id: uuid.UUID,
    candidate_kind: str,
    member_id: uuid.UUID,
    fact_value_id: uuid.UUID,
    source_batch_id: uuid.UUID,
    semantic_key_hash: str,
    value_type: str,
    value_json,
    normalized_value_text: str | None,
    referenced_entity_id: uuid.UUID | None,
    evidence_link_id: uuid.UUID,
    evidence_id: uuid.UUID,
    evidence_role: str,
    is_primary: bool,
    source_order: int,
    document_revision_id: uuid.UUID,
    document_block_id: uuid.UUID,
    location_key: str,
    page_no: int | None,
    start_line: int | None,
    end_line: int | None,
    start_offset: int,
    end_offset: int,
    excerpt: str,
) -> ConsistencyProjectionSourceRow:
    excerpt_hash = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
    return ConsistencyProjectionSourceRow(
        candidate_id=candidate_id,
        consistency_application_id=_uuid("source-app"),
        candidate_fact_id=fact_id,
        candidate_kind=candidate_kind,
        member_id=member_id,
        member_candidate_id=candidate_id,
        member_consistency_application_id=_uuid("source-app"),
        member_orchestration_id=_uuid("orchestration"),
        member_fact_value_id=fact_value_id,
        member_source_batch_id=source_batch_id,
        member_semantic_key_hash=semantic_key_hash,
        fact_value_fact_id=fact_id,
        fact_value_value_type=value_type,
        fact_value_value_json=value_json,
        fact_value_normalized_value_text=normalized_value_text,
        fact_value_referenced_entity_id=referenced_entity_id,
        batch_orchestration_id=_uuid("orchestration"),
        evidence_link_id=evidence_link_id,
        evidence_link_fact_value_id=fact_value_id,
        evidence_id=evidence_id,
        evidence_role=evidence_role,
        evidence_is_primary=is_primary,
        evidence_source_order=source_order,
        document_revision_id=document_revision_id,
        block_id=document_block_id,
        start_offset=start_offset,
        end_offset=end_offset,
        excerpt=excerpt,
        excerpt_hash=excerpt_hash,
        location_key=location_key,
        page_no=page_no,
        start_line=start_line,
        end_line=end_line,
    )


def _build_projection_fixture():
    created_at = datetime(2026, 8, 1, tzinfo=timezone.utc)
    project_id = _uuid("project")
    app_id = _uuid("cc-app")
    source_app_id = _uuid("source-app")
    orchestration_id = _uuid("orchestration")
    duplicate_grouping_application_id = _uuid("duplicate-grouping")
    extraction_run_id = _uuid("extraction-run")

    candidate_a = _uuid("candidate-a")
    candidate_b = _uuid("candidate-b")
    candidate_c = _uuid("candidate-c")
    fact_a = _uuid("fact-a")
    fact_b = _uuid("fact-b")
    fact_c = _uuid("fact-c")
    member_a1 = _uuid("member-a1")
    member_a2 = _uuid("member-a2")
    member_b1 = _uuid("member-b1")
    member_c1 = _uuid("member-c1")
    fv_a1 = _uuid("fv-a1")
    fv_a2 = _uuid("fv-a2")
    fv_b1 = _uuid("fv-b1")
    fv_c1 = _uuid("fv-c1")
    batch_source_0 = _uuid("source-batch-0")
    batch_source_1 = _uuid("source-batch-1")
    batch_source_2 = _uuid("source-batch-2")
    entity_ref = _uuid("entity-ref")

    source_rows = (
        _source_row(
            candidate_id=candidate_a,
            fact_id=fact_a,
            candidate_kind="multi_value",
            member_id=member_a1,
            fact_value_id=fv_a1,
            source_batch_id=batch_source_0,
            semantic_key_hash="1" * 64,
            value_type="string",
            value_json="alpha",
            normalized_value_text="alpha",
            referenced_entity_id=None,
            evidence_link_id=_uuid("evlink-a1-0"),
            evidence_id=_uuid("evidence-a1-0"),
            evidence_role="supporting",
            is_primary=True,
            source_order=0,
            document_revision_id=_uuid("revision-a"),
            document_block_id=_uuid("block-a0"),
            location_key="loc:a0",
            page_no=1,
            start_line=1,
            end_line=1,
            start_offset=0,
            end_offset=5,
            excerpt="alpha",
        ),
        _source_row(
            candidate_id=candidate_a,
            fact_id=fact_a,
            candidate_kind="multi_value",
            member_id=member_a1,
            fact_value_id=fv_a1,
            source_batch_id=batch_source_0,
            semantic_key_hash="1" * 64,
            value_type="string",
            value_json="alpha",
            normalized_value_text="alpha",
            referenced_entity_id=None,
            evidence_link_id=_uuid("evlink-a1-1"),
            evidence_id=_uuid("evidence-a1-1"),
            evidence_role="context",
            is_primary=False,
            source_order=1,
            document_revision_id=_uuid("revision-a"),
            document_block_id=_uuid("block-a1"),
            location_key="loc:a1",
            page_no=1,
            start_line=2,
            end_line=2,
            start_offset=6,
            end_offset=11,
            excerpt="alpha-context",
        ),
        _source_row(
            candidate_id=candidate_a,
            fact_id=fact_a,
            candidate_kind="multi_value",
            member_id=member_a2,
            fact_value_id=fv_a2,
            source_batch_id=batch_source_1,
            semantic_key_hash="2" * 64,
            value_type="object",
            value_json={"city": "Paris"},
            normalized_value_text=None,
            referenced_entity_id=None,
            evidence_link_id=_uuid("evlink-a2-0"),
            evidence_id=_uuid("evidence-a2-0"),
            evidence_role="supporting",
            is_primary=True,
            source_order=0,
            document_revision_id=_uuid("revision-a"),
            document_block_id=_uuid("block-a2"),
            location_key="loc:a2",
            page_no=1,
            start_line=3,
            end_line=3,
            start_offset=0,
            end_offset=12,
            excerpt="paris object",
        ),
        _source_row(
            candidate_id=candidate_b,
            fact_id=fact_b,
            candidate_kind="multi_value",
            member_id=member_b1,
            fact_value_id=fv_b1,
            source_batch_id=batch_source_0,
            semantic_key_hash="3" * 64,
            value_type="entity_ref",
            value_json=None,
            normalized_value_text="entity:42",
            referenced_entity_id=entity_ref,
            evidence_link_id=_uuid("evlink-b1-0"),
            evidence_id=_uuid("evidence-b1-0"),
            evidence_role="supporting",
            is_primary=True,
            source_order=0,
            document_revision_id=_uuid("revision-b"),
            document_block_id=_uuid("block-b0"),
            location_key="loc:b0",
            page_no=2,
            start_line=1,
            end_line=1,
            start_offset=0,
            end_offset=8,
            excerpt="entity ok",
        ),
        _source_row(
            candidate_id=candidate_c,
            fact_id=fact_c,
            candidate_kind="multi_value",
            member_id=member_c1,
            fact_value_id=fv_c1,
            source_batch_id=batch_source_2,
            semantic_key_hash="4" * 64,
            value_type="number",
            value_json=7,
            normalized_value_text="7",
            referenced_entity_id=None,
            evidence_link_id=_uuid("evlink-c1-0"),
            evidence_id=_uuid("evidence-c1-0"),
            evidence_role="supporting",
            is_primary=True,
            source_order=0,
            document_revision_id=_uuid("revision-c"),
            document_block_id=_uuid("block-c0"),
            location_key="loc:c0",
            page_no=3,
            start_line=1,
            end_line=1,
            start_offset=0,
            end_offset=1,
            excerpt="7",
        ),
        _source_row(
            candidate_id=candidate_c,
            fact_id=fact_c,
            candidate_kind="multi_value",
            member_id=member_c1,
            fact_value_id=fv_c1,
            source_batch_id=batch_source_2,
            semantic_key_hash="4" * 64,
            value_type="number",
            value_json=7,
            normalized_value_text="7",
            referenced_entity_id=None,
            evidence_link_id=_uuid("evlink-c1-1"),
            evidence_id=_uuid("evidence-c1-1"),
            evidence_role="context",
            is_primary=False,
            source_order=1,
            document_revision_id=_uuid("revision-c"),
            document_block_id=_uuid("block-c1"),
            location_key="loc:c1",
            page_no=3,
            start_line=2,
            end_line=2,
            start_offset=2,
            end_offset=3,
            excerpt="8",
        ),
    )

    authenticated_source = SimpleNamespace(
        project_id=project_id,
        application=FactValueConsistencyCandidateApplicationLedger(
            id=source_app_id,
            duplicate_grouping_application_id=duplicate_grouping_application_id,
            orchestration_id=orchestration_id,
            extraction_run_id=extraction_run_id,
            algorithm_version="cross_batch_multi_value_v1",
            input_manifest_hash="a" * 64,
            result_manifest_hash="b" * 64,
            candidate_count=3,
            member_count=4,
            created_at=created_at,
        ),
        write_plan=SimpleNamespace(),
        candidate_ledgers=(
            FactValueConsistencyCandidateLedger(
                id=candidate_a,
                consistency_application_id=source_app_id,
                fact_id=fact_a,
                candidate_kind="multi_value",
                member_count=2,
                distinct_semantic_key_count=2,
                distinct_batch_count=2,
                created_at=created_at,
            ),
            FactValueConsistencyCandidateLedger(
                id=candidate_b,
                consistency_application_id=source_app_id,
                fact_id=fact_b,
                candidate_kind="multi_value",
                member_count=1,
                distinct_semantic_key_count=1,
                distinct_batch_count=1,
                created_at=created_at,
            ),
            FactValueConsistencyCandidateLedger(
                id=candidate_c,
                consistency_application_id=source_app_id,
                fact_id=fact_c,
                candidate_kind="multi_value",
                member_count=1,
                distinct_semantic_key_count=1,
                distinct_batch_count=1,
                created_at=created_at,
            ),
        ),
        member_ledgers=(
            FactValueConsistencyCandidateMemberLedger(
                id=member_a1,
                consistency_application_id=source_app_id,
                candidate_id=candidate_a,
                orchestration_id=orchestration_id,
                fact_value_id=fv_a1,
                source_batch_id=batch_source_0,
                semantic_key_hash="1" * 64,
                created_at=created_at,
            ),
            FactValueConsistencyCandidateMemberLedger(
                id=member_a2,
                consistency_application_id=source_app_id,
                candidate_id=candidate_a,
                orchestration_id=orchestration_id,
                fact_value_id=fv_a2,
                source_batch_id=batch_source_1,
                semantic_key_hash="2" * 64,
                created_at=created_at,
            ),
            FactValueConsistencyCandidateMemberLedger(
                id=member_b1,
                consistency_application_id=source_app_id,
                candidate_id=candidate_b,
                orchestration_id=orchestration_id,
                fact_value_id=fv_b1,
                source_batch_id=batch_source_0,
                semantic_key_hash="3" * 64,
                created_at=created_at,
            ),
            FactValueConsistencyCandidateMemberLedger(
                id=member_c1,
                consistency_application_id=source_app_id,
                candidate_id=candidate_c,
                orchestration_id=orchestration_id,
                fact_value_id=fv_c1,
                source_batch_id=batch_source_2,
                semantic_key_hash="4" * 64,
                created_at=created_at,
            ),
        ),
    )

    candidate_bundles = consistency_check_service._build_candidate_bundles(
        authenticated_source,
        source_rows,
    )
    assessment_a, assessment_hash_a = _assessment_hash(
        candidate_id=candidate_a,
        verdict="conflict",
        severity="red",
        confidence=0.91,
        explanation="candidate a conflict",
        cited_evidence_link_ids=[_uuid("evlink-a2-0"), _uuid("evlink-a1-0")],
        impact=["scope_review"],
        recommended_actions=["review_source_scope"],
    )
    assessment_b, assessment_hash_b = _assessment_hash(
        candidate_id=candidate_b,
        verdict="compatible",
        severity="none",
        confidence=0.88,
        explanation="candidate b ok",
        cited_evidence_link_ids=[_uuid("evlink-b1-0")],
        impact=[],
        recommended_actions=[],
    )
    assessment_c, assessment_hash_c = _assessment_hash(
        candidate_id=candidate_c,
        verdict="insufficient_evidence",
        severity="none",
        confidence=0.4,
        explanation="candidate c insufficient",
        cited_evidence_link_ids=[_uuid("evlink-c1-0")],
        impact=["data_quality_review"],
        recommended_actions=["request_more_evidence"],
    )

    batches = (
        ConsistencyCheckBatchLedgerRecord(
            id=_uuid("batch-ledger-0"),
            consistency_check_application_id=app_id,
            batch_index=0,
            batch_manifest_hash="1" * 64,
            skipped_empty=False,
            input_batch_id=_uuid("input-batch-0"),
            inference_run_id=_uuid("inference-run-0"),
            request_hash="2" * 64,
            message_content_hash="3" * 64,
            created_at=created_at,
        ),
        ConsistencyCheckBatchLedgerRecord(
            id=_uuid("batch-ledger-1"),
            consistency_check_application_id=app_id,
            batch_index=1,
            batch_manifest_hash="4" * 64,
            skipped_empty=False,
            input_batch_id=_uuid("input-batch-1"),
            inference_run_id=_uuid("inference-run-1"),
            request_hash="5" * 64,
            message_content_hash="6" * 64,
            created_at=created_at,
        ),
    )
    assessments = (
        ConsistencyAssessmentLedgerRecord(
            id=_uuid("assessment-a"),
            consistency_check_application_id=app_id,
            source_consistency_application_id=source_app_id,
            source_consistency_candidate_id=candidate_a,
            batch_index=0,
            verdict=assessment_a.verdict,
            severity=assessment_a.severity,
            confidence=assessment_a.confidence,
            explanation=assessment_a.explanation,
            impact_json=tuple(assessment_a.impact),
            recommended_actions_json=tuple(assessment_a.recommended_actions),
            assessment_manifest_hash=assessment_hash_a,
            created_at=created_at,
        ),
        ConsistencyAssessmentLedgerRecord(
            id=_uuid("assessment-b"),
            consistency_check_application_id=app_id,
            source_consistency_application_id=source_app_id,
            source_consistency_candidate_id=candidate_b,
            batch_index=0,
            verdict=assessment_b.verdict,
            severity=assessment_b.severity,
            confidence=assessment_b.confidence,
            explanation=assessment_b.explanation,
            impact_json=tuple(assessment_b.impact),
            recommended_actions_json=tuple(assessment_b.recommended_actions),
            assessment_manifest_hash=assessment_hash_b,
            created_at=created_at,
        ),
        ConsistencyAssessmentLedgerRecord(
            id=_uuid("assessment-c"),
            consistency_check_application_id=app_id,
            source_consistency_application_id=source_app_id,
            source_consistency_candidate_id=candidate_c,
            batch_index=1,
            verdict=assessment_c.verdict,
            severity=assessment_c.severity,
            confidence=assessment_c.confidence,
            explanation=assessment_c.explanation,
            impact_json=tuple(assessment_c.impact),
            recommended_actions_json=tuple(assessment_c.recommended_actions),
            assessment_manifest_hash=assessment_hash_c,
            created_at=created_at,
        ),
    )
    citations = (
        ConsistencyAssessmentCitationLedgerRecord(
            id=_uuid("citation-a0"),
            assessment_id=_uuid("assessment-a"),
            source_consistency_application_id=source_app_id,
            source_consistency_candidate_id=candidate_a,
            source_fact_value_id=fv_a2,
            evidence_link_id=_uuid("evlink-a2-0"),
            citation_order=0,
            created_at=created_at,
        ),
        ConsistencyAssessmentCitationLedgerRecord(
            id=_uuid("citation-a1"),
            assessment_id=_uuid("assessment-a"),
            source_consistency_application_id=source_app_id,
            source_consistency_candidate_id=candidate_a,
            source_fact_value_id=fv_a1,
            evidence_link_id=_uuid("evlink-a1-0"),
            citation_order=1,
            created_at=created_at,
        ),
        ConsistencyAssessmentCitationLedgerRecord(
            id=_uuid("citation-b0"),
            assessment_id=_uuid("assessment-b"),
            source_consistency_application_id=source_app_id,
            source_consistency_candidate_id=candidate_b,
            source_fact_value_id=fv_b1,
            evidence_link_id=_uuid("evlink-b1-0"),
            citation_order=0,
            created_at=created_at,
        ),
        ConsistencyAssessmentCitationLedgerRecord(
            id=_uuid("citation-c0"),
            assessment_id=_uuid("assessment-c"),
            source_consistency_application_id=source_app_id,
            source_consistency_candidate_id=candidate_c,
            source_fact_value_id=fv_c1,
            evidence_link_id=_uuid("evlink-c1-0"),
            citation_order=0,
            created_at=created_at,
        ),
    )

    batch_results = (
        ConsistencyCheckBatchExecutionResult(
            project_id=project_id,
            consistency_application_id=source_app_id,
            source_result_manifest_hash=authenticated_source.application.result_manifest_hash,
            plan_manifest_hash="7" * 64,
            batch_index=0,
            batch_manifest_hash=batches[0].batch_manifest_hash,
            input_batch_id=batches[0].input_batch_id,
            inference_run_id=batches[0].inference_run_id,
            request_hash=batches[0].request_hash,
            message_content_hash=batches[0].message_content_hash,
            skipped_empty=False,
            reused_completed_run=False,
            response=ConsistencyCheckResponse(assessments=[assessment_a, assessment_b]),
            response_model=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
        ),
        ConsistencyCheckBatchExecutionResult(
            project_id=project_id,
            consistency_application_id=source_app_id,
            source_result_manifest_hash=authenticated_source.application.result_manifest_hash,
            plan_manifest_hash="7" * 64,
            batch_index=1,
            batch_manifest_hash=batches[1].batch_manifest_hash,
            input_batch_id=batches[1].input_batch_id,
            inference_run_id=batches[1].inference_run_id,
            request_hash=batches[1].request_hash,
            message_content_hash=batches[1].message_content_hash,
            skipped_empty=False,
            reused_completed_run=False,
            response=ConsistencyCheckResponse(assessments=[assessment_c]),
            response_model=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
        ),
    )
    result_manifest_hash = execution_service._build_plan_result_manifest_hash(
        authoritative_plan=SimpleNamespace(
            project_id=project_id,
            consistency_application_id=source_app_id,
            source_result_manifest_hash=authenticated_source.application.result_manifest_hash,
            plan_manifest_hash="7" * 64,
        ),
        ordered_batch_results=batch_results,
        ordered_assessments_by_batch=((assessment_a, assessment_b), (assessment_c,)),
    )
    application = ConsistencyCheckApplicationLedgerRecord(
        id=app_id,
        project_id=project_id,
        consistency_application_id=source_app_id,
        orchestration_id=orchestration_id,
        source_result_manifest_hash=authenticated_source.application.result_manifest_hash,
        plan_manifest_hash="7" * 64,
        execution_identity_hash="8" * 64,
        result_manifest_hash=result_manifest_hash,
        prompt_contract_hash="9" * 64,
        provider="openai",
        requested_model="gpt-4.1",
        executor_name=execution_service.CONSISTENCY_CHECK_EXECUTOR_NAME,
        executor_version=execution_service.CONSISTENCY_CHECK_EXECUTOR_VERSION,
        batch_count=2,
        executed_batch_count=2,
        skipped_empty_batch_count=0,
        inference_run_count=2,
        assessment_count=3,
        created_at=created_at,
    )
    snapshot = ConsistencyProjectionSnapshot(
        application=application,
        batches=batches,
        assessments=assessments,
        citations=citations,
        source_rows=source_rows,
    )
    return snapshot, authenticated_source, {
        "project_id": project_id,
        "application_id": app_id,
        "source_application_id": source_app_id,
        "assessment_a_id": _uuid("assessment-a"),
        "assessment_b_id": _uuid("assessment-b"),
        "assessment_c_id": _uuid("assessment-c"),
        "candidate_a_id": candidate_a,
        "candidate_b_id": candidate_b,
        "candidate_c_id": candidate_c,
        "fv_a1": fv_a1,
        "fv_a2": fv_a2,
        "fv_b1": fv_b1,
        "fv_c1": fv_c1,
        "actor_id": _uuid("review-actor"),
        "created_at": created_at,
    }


def _install_snapshot_monkeypatches(
    monkeypatch: pytest.MonkeyPatch,
    *,
    snapshot: ConsistencyProjectionSnapshot | None,
    authenticated_source,
    decisions=(),
    selections=(),
    call_log: list[tuple[str, uuid.UUID]] | None = None,
) -> None:
    async def fake_load_snapshot(_session, *, consistency_check_application_id):
        if snapshot is None or consistency_check_application_id != snapshot.application.id:
            return None
        return snapshot

    async def fake_authenticate_source(_session_factory, *, consistency_application_id):
        assert snapshot is not None
        assert consistency_application_id == snapshot.application.consistency_application_id
        return authenticated_source

    async def fake_list_decisions_by_application(_session, *, consistency_check_application_id):
        if call_log is not None:
            call_log.append(("decisions", consistency_check_application_id))
        assert snapshot is not None
        assert consistency_check_application_id == snapshot.application.id
        return tuple(decisions)

    async def fake_list_selections_by_application(_session, *, consistency_check_application_id):
        if call_log is not None:
            call_log.append(("selections", consistency_check_application_id))
        assert snapshot is not None
        assert consistency_check_application_id == snapshot.application.id
        return tuple(selections)

    async def _unexpected_per_assessment(*args, **kwargs):
        raise AssertionError("per-assessment decision query should not be used")

    monkeypatch.setattr(
        projection_service.persistence_service.consistency_projection_repository,
        "load_consistency_projection_snapshot",
        fake_load_snapshot,
    )
    monkeypatch.setattr(
        projection_service.persistence_service.duplicate_grouping_service,
        "authenticate_fact_value_consistency_candidate_application",
        fake_authenticate_source,
    )
    monkeypatch.setattr(
        projection_service.consistency_review_repository,
        "list_decision_ledgers_by_application",
        fake_list_decisions_by_application,
    )
    monkeypatch.setattr(
        projection_service.consistency_review_repository,
        "list_selection_ledgers_by_application",
        fake_list_selections_by_application,
    )
    monkeypatch.setattr(
        projection_service.consistency_review_repository,
        "list_decision_ledgers",
        _unexpected_per_assessment,
    )
    monkeypatch.setattr(
        projection_service.consistency_review_repository,
        "list_selection_ledgers",
        _unexpected_per_assessment,
    )


def test_get_consistency_review_projection_builds_complete_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, authenticated_source, fixture_ids = _build_projection_fixture()
    session_factory = SessionFactory()
    _install_snapshot_monkeypatches(
        monkeypatch,
        snapshot=snapshot,
        authenticated_source=authenticated_source,
    )

    result = run_async(
        projection_service.get_consistency_review_projection(
            session_factory,
            project_id=snapshot.application.project_id,
            consistency_check_application_id=snapshot.application.id,
        )
    )

    assert result.project_id == snapshot.application.project_id
    assert result.consistency_check_application_id == snapshot.application.id
    assert result.source_consistency_application_id == snapshot.application.consistency_application_id
    assert result.plan_manifest_hash == snapshot.application.plan_manifest_hash
    assert result.result_manifest_hash == snapshot.application.result_manifest_hash
    assert result.assessment_count == 3
    assert result.conflict_count == 1
    assert result.compatible_count == 1
    assert result.insufficient_evidence_count == 1
    assert result.red_count == 1
    assert result.yellow_count == 0
    assert result.pending_review_count == 2
    assert result.reviewed_count == 0
    assert result.deferred_count == 0
    assert result.not_required_count == 1
    assert result.decision_count == 0
    assert [item.candidate_id for item in result.items] == [
        _uuid("candidate-a"),
        _uuid("candidate-b"),
        _uuid("candidate-c"),
    ]
    first = result.items[0]
    assert first.assessment_id == fixture_ids["assessment_a_id"]
    assert first.review_status == "pending_review"
    assert first.current_decision is None
    assert first.decision_history == ()
    assert first.selected_fact_value_ids == ()
    assert [member.fact_value_id for member in first.members] == [_uuid("fv-a1"), _uuid("fv-a2")]
    assert [e.evidence_link_id for e in first.members[0].evidences] == [
        _uuid("evlink-a1-0"),
        _uuid("evlink-a1-1"),
    ]
    assert first.members[0].evidences[0].cited_by_assessment is True
    assert first.members[0].evidences[1].cited_by_assessment is False
    assert first.members[0].selected_by_current_decision is False
    assert first.members[0].current_selection_order is None
    assert first.members[1].value_type == "object"
    assert result.items[1].review_status == "not_required"
    assert result.items[1].members[0].referenced_entity_id == _uuid("entity-ref")
    assert result.items[2].review_status == "pending_review"
    assert result.items[2].members[0].evidences[0].document_revision_id == _uuid("revision-c")
    assert all(session.commit_count == 0 for session in session_factory.sessions)


@pytest.mark.parametrize(
    ("decision_kind", "selected_fact_value_ids", "expected_status", "expected_selected_ids"),
    [
        ("select_one", ("fv-a2",), "reviewed", (_uuid("fv-a2"),)),
        ("keep_multiple", ("fv-a2", "fv-a1"), "reviewed", (_uuid("fv-a2"), _uuid("fv-a1"))),
        ("confirm_compatible", (), "reviewed", ()),
        ("defer", (), "deferred", ()),
    ],
)
def test_get_consistency_review_projection_maps_leaf_decision_state_and_selection(
    monkeypatch: pytest.MonkeyPatch,
    decision_kind: str,
    selected_fact_value_ids: tuple[str, ...],
    expected_status: str,
    expected_selected_ids: tuple[uuid.UUID, ...],
) -> None:
    snapshot, authenticated_source, fixture_ids = _build_projection_fixture()
    decision, selections = _decision_record(
        decision_id=_uuid(f"{decision_kind}-decision"),
        project_id=fixture_ids["project_id"],
        application_id=fixture_ids["application_id"],
        assessment_id=fixture_ids["assessment_a_id"],
        source_application_id=fixture_ids["source_application_id"],
        source_candidate_id=fixture_ids["candidate_a_id"],
        actor_id=fixture_ids["actor_id"],
        decision_no=1,
        supersedes_decision_id=None,
        decision_kind=decision_kind,
        comment="manual review" if decision_kind == "select_one" else None,
        selected_fact_value_ids=tuple(_uuid(seed) for seed in selected_fact_value_ids),
        created_at=fixture_ids["created_at"],
    )
    _install_snapshot_monkeypatches(
        monkeypatch,
        snapshot=snapshot,
        authenticated_source=authenticated_source,
        decisions=(decision,),
        selections=selections,
    )

    result = run_async(
        projection_service.get_consistency_review_projection(
            SessionFactory(),
            project_id=snapshot.application.project_id,
            consistency_check_application_id=snapshot.application.id,
        )
    )

    item = result.items[0]
    assert item.review_status == expected_status
    assert item.current_decision is not None
    assert item.current_decision.decision_kind == decision_kind
    assert item.selected_fact_value_ids == expected_selected_ids
    assert [entry.decision_no for entry in item.decision_history] == [1]
    selection_map = {
        member.fact_value_id: (
            member.selected_by_current_decision,
            member.current_selection_order,
        )
        for member in item.members
    }
    if decision_kind == "select_one":
        assert selection_map[fixture_ids["fv_a2"]] == (True, 0)
        assert selection_map[fixture_ids["fv_a1"]] == (False, None)
    elif decision_kind == "keep_multiple":
        assert selection_map[fixture_ids["fv_a2"]] == (True, 0)
        assert selection_map[fixture_ids["fv_a1"]] == (True, 1)
    else:
        assert selection_map[fixture_ids["fv_a2"]] == (False, None)
        assert selection_map[fixture_ids["fv_a1"]] == (False, None)


def test_get_consistency_review_projection_exposes_current_leaf_and_ordered_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, authenticated_source, fixture_ids = _build_projection_fixture()
    first_decision, first_selections = _decision_record(
        decision_id=_uuid("decision-first"),
        project_id=fixture_ids["project_id"],
        application_id=fixture_ids["application_id"],
        assessment_id=fixture_ids["assessment_a_id"],
        source_application_id=fixture_ids["source_application_id"],
        source_candidate_id=fixture_ids["candidate_a_id"],
        actor_id=fixture_ids["actor_id"],
        decision_no=1,
        supersedes_decision_id=None,
        decision_kind="select_one",
        comment="first",
        selected_fact_value_ids=(fixture_ids["fv_a1"],),
        created_at=fixture_ids["created_at"],
    )
    second_decision, second_selections = _decision_record(
        decision_id=_uuid("decision-second"),
        project_id=fixture_ids["project_id"],
        application_id=fixture_ids["application_id"],
        assessment_id=fixture_ids["assessment_a_id"],
        source_application_id=fixture_ids["source_application_id"],
        source_candidate_id=fixture_ids["candidate_a_id"],
        actor_id=fixture_ids["actor_id"],
        decision_no=2,
        supersedes_decision_id=first_decision.id,
        decision_kind="keep_multiple",
        comment="second",
        selected_fact_value_ids=(fixture_ids["fv_a2"], fixture_ids["fv_a1"]),
        created_at=fixture_ids["created_at"],
    )
    _install_snapshot_monkeypatches(
        monkeypatch,
        snapshot=snapshot,
        authenticated_source=authenticated_source,
        decisions=(first_decision, second_decision),
        selections=first_selections + second_selections,
    )

    result = run_async(
        projection_service.get_consistency_review_projection(
            SessionFactory(),
            project_id=snapshot.application.project_id,
            consistency_check_application_id=snapshot.application.id,
        )
    )

    item = result.items[0]
    assert item.review_status == "reviewed"
    assert [entry.decision_no for entry in item.decision_history] == [1, 2]
    assert item.current_decision is not None
    assert item.current_decision.decision_id == second_decision.id
    assert item.selected_fact_value_ids == (fixture_ids["fv_a2"], fixture_ids["fv_a1"])
    assert result.reviewed_count == 1
    assert result.decision_count == 2


def test_get_consistency_review_projection_uses_batch_read_without_n_plus_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, authenticated_source, fixture_ids = _build_projection_fixture()
    decision, selections = _decision_record(
        decision_id=_uuid("decision-batch-read"),
        project_id=fixture_ids["project_id"],
        application_id=fixture_ids["application_id"],
        assessment_id=fixture_ids["assessment_a_id"],
        source_application_id=fixture_ids["source_application_id"],
        source_candidate_id=fixture_ids["candidate_a_id"],
        actor_id=fixture_ids["actor_id"],
        decision_no=1,
        supersedes_decision_id=None,
        decision_kind="select_one",
        comment=None,
        selected_fact_value_ids=(fixture_ids["fv_a1"],),
        created_at=fixture_ids["created_at"],
    )
    call_log: list[tuple[str, uuid.UUID]] = []
    _install_snapshot_monkeypatches(
        monkeypatch,
        snapshot=snapshot,
        authenticated_source=authenticated_source,
        decisions=(decision,),
        selections=selections,
        call_log=call_log,
    )

    run_async(
        projection_service.get_consistency_review_projection(
            SessionFactory(),
            project_id=snapshot.application.project_id,
            consistency_check_application_id=snapshot.application.id,
        )
    )

    assert call_log == [
        ("decisions", snapshot.application.id),
        ("selections", snapshot.application.id),
    ]


def test_get_consistency_review_projection_is_deterministic_across_repeated_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, authenticated_source, fixture_ids = _build_projection_fixture()
    decision, selections = _decision_record(
        decision_id=_uuid("repeat-decision"),
        project_id=fixture_ids["project_id"],
        application_id=fixture_ids["application_id"],
        assessment_id=fixture_ids["assessment_a_id"],
        source_application_id=fixture_ids["source_application_id"],
        source_candidate_id=fixture_ids["candidate_a_id"],
        actor_id=fixture_ids["actor_id"],
        decision_no=1,
        supersedes_decision_id=None,
        decision_kind="select_one",
        comment="repeatable",
        selected_fact_value_ids=(fixture_ids["fv_a1"],),
        created_at=fixture_ids["created_at"],
    )
    _install_snapshot_monkeypatches(
        monkeypatch,
        snapshot=snapshot,
        authenticated_source=authenticated_source,
        decisions=(decision,),
        selections=selections,
    )
    session_factory = SessionFactory()

    first = run_async(
        projection_service.get_consistency_review_projection(
            session_factory,
            project_id=snapshot.application.project_id,
            consistency_check_application_id=snapshot.application.id,
        )
    )
    second = run_async(
        projection_service.get_consistency_review_projection(
            session_factory,
            project_id=snapshot.application.project_id,
            consistency_check_application_id=snapshot.application.id,
        )
    )

    assert first == second


def test_get_consistency_review_projection_rejects_unknown_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_factory = SessionFactory()
    _install_snapshot_monkeypatches(
        monkeypatch,
        snapshot=None,
        authenticated_source=None,
    )

    with pytest.raises(
        projection_service.ConsistencyProjectionStateError,
        match="consistency_review_projection_application_not_found",
    ):
        run_async(
            projection_service.get_consistency_review_projection(
                session_factory,
                project_id=_uuid("project"),
                consistency_check_application_id=_uuid("missing"),
            )
        )


def test_get_consistency_review_projection_rejects_project_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, authenticated_source, _fixture_ids = _build_projection_fixture()
    session_factory = SessionFactory()
    _install_snapshot_monkeypatches(
        monkeypatch,
        snapshot=snapshot,
        authenticated_source=authenticated_source,
    )

    with pytest.raises(
        projection_service.ConsistencyProjectionStateError,
        match="consistency_review_projection_project_id_mismatch",
    ):
        run_async(
            projection_service.get_consistency_review_projection(
                session_factory,
                project_id=_uuid("other-project"),
                consistency_check_application_id=snapshot.application.id,
            )
        )


def test_get_consistency_review_projection_reads_exact_application_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, authenticated_source, _fixture_ids = _build_projection_fixture()
    other_snapshot, _, _ = _build_projection_fixture()
    requested_ids: list[uuid.UUID] = []

    async def fake_load_snapshot(_session, *, consistency_check_application_id):
        requested_ids.append(consistency_check_application_id)
        if consistency_check_application_id == snapshot.application.id:
            return snapshot
        return other_snapshot

    async def fake_authenticate_source(_session_factory, *, consistency_application_id):
        assert consistency_application_id == snapshot.application.consistency_application_id
        return authenticated_source

    async def fake_list_decisions_by_application(_session, *, consistency_check_application_id):
        assert consistency_check_application_id == snapshot.application.id
        return ()

    async def fake_list_selections_by_application(_session, *, consistency_check_application_id):
        assert consistency_check_application_id == snapshot.application.id
        return ()

    monkeypatch.setattr(
        projection_service.persistence_service.consistency_projection_repository,
        "load_consistency_projection_snapshot",
        fake_load_snapshot,
    )
    monkeypatch.setattr(
        projection_service.persistence_service.duplicate_grouping_service,
        "authenticate_fact_value_consistency_candidate_application",
        fake_authenticate_source,
    )
    monkeypatch.setattr(
        projection_service.consistency_review_repository,
        "list_decision_ledgers_by_application",
        fake_list_decisions_by_application,
    )
    monkeypatch.setattr(
        projection_service.consistency_review_repository,
        "list_selection_ledgers_by_application",
        fake_list_selections_by_application,
    )

    result = run_async(
        projection_service.get_consistency_review_projection(
            SessionFactory(),
            project_id=snapshot.application.project_id,
            consistency_check_application_id=snapshot.application.id,
        )
    )

    assert requested_ids == [snapshot.application.id]
    assert result.consistency_check_application_id == snapshot.application.id


def test_get_consistency_review_projection_zero_assessment_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_at = datetime(2026, 8, 1, tzinfo=timezone.utc)
    project_id = _uuid("project-zero")
    app_id = _uuid("cc-app-zero")
    source_app_id = _uuid("source-app-zero")
    orchestration_id = _uuid("orchestration-zero")
    snapshot = ConsistencyProjectionSnapshot(
        application=ConsistencyCheckApplicationLedgerRecord(
            id=app_id,
            project_id=project_id,
            consistency_application_id=source_app_id,
            orchestration_id=orchestration_id,
            source_result_manifest_hash="a" * 64,
            plan_manifest_hash="b" * 64,
            execution_identity_hash="c" * 64,
            result_manifest_hash=execution_service._build_plan_result_manifest_hash(
                authoritative_plan=SimpleNamespace(
                    project_id=project_id,
                    consistency_application_id=source_app_id,
                    source_result_manifest_hash="a" * 64,
                    plan_manifest_hash="b" * 64,
                ),
                ordered_batch_results=(
                    ConsistencyCheckBatchExecutionResult(
                        project_id=project_id,
                        consistency_application_id=source_app_id,
                        source_result_manifest_hash="a" * 64,
                        plan_manifest_hash="b" * 64,
                        batch_index=0,
                        batch_manifest_hash="d" * 64,
                        input_batch_id=None,
                        inference_run_id=None,
                        request_hash=None,
                        message_content_hash=None,
                        skipped_empty=True,
                        reused_completed_run=False,
                        response=ConsistencyCheckResponse(assessments=[]),
                        response_model=None,
                        prompt_tokens=None,
                        completion_tokens=None,
                        total_tokens=None,
                    ),
                ),
                ordered_assessments_by_batch=((),),
            ),
            prompt_contract_hash="e" * 64,
            provider="openai",
            requested_model="gpt-4.1",
            executor_name=execution_service.CONSISTENCY_CHECK_EXECUTOR_NAME,
            executor_version=execution_service.CONSISTENCY_CHECK_EXECUTOR_VERSION,
            batch_count=1,
            executed_batch_count=1,
            skipped_empty_batch_count=1,
            inference_run_count=0,
            assessment_count=0,
            created_at=created_at,
        ),
        batches=(
            ConsistencyCheckBatchLedgerRecord(
                id=_uuid("batch-zero"),
                consistency_check_application_id=app_id,
                batch_index=0,
                batch_manifest_hash="d" * 64,
                skipped_empty=True,
                input_batch_id=None,
                inference_run_id=None,
                request_hash=None,
                message_content_hash=None,
                created_at=created_at,
            ),
        ),
        assessments=(),
        citations=(),
        source_rows=(),
    )
    authenticated_source = SimpleNamespace(
        project_id=project_id,
        application=FactValueConsistencyCandidateApplicationLedger(
            id=source_app_id,
            duplicate_grouping_application_id=_uuid("dg-zero"),
            orchestration_id=orchestration_id,
            extraction_run_id=_uuid("er-zero"),
            algorithm_version="cross_batch_multi_value_v1",
            input_manifest_hash="f" * 64,
            result_manifest_hash="a" * 64,
            candidate_count=0,
            member_count=0,
            created_at=created_at,
        ),
        write_plan=SimpleNamespace(),
        candidate_ledgers=(),
        member_ledgers=(),
    )
    _install_snapshot_monkeypatches(
        monkeypatch,
        snapshot=snapshot,
        authenticated_source=authenticated_source,
    )

    result = run_async(
        projection_service.get_consistency_review_projection(
            SessionFactory(),
            project_id=project_id,
            consistency_check_application_id=app_id,
        )
    )

    assert result.items == ()
    assert result.assessment_count == 0
    assert result.conflict_count == 0
    assert result.compatible_count == 0
    assert result.insufficient_evidence_count == 0
    assert result.pending_review_count == 0
    assert result.reviewed_count == 0
    assert result.deferred_count == 0
    assert result.not_required_count == 0
    assert result.decision_count == 0


@pytest.mark.parametrize(
    "drift_kind",
    [
        "assessment_missing",
        "assessment_extra",
        "assessment_hash",
        "result_hash",
        "citation_binding",
        "decision_gap",
        "decision_wrong_predecessor",
        "decision_manifest",
        "selection_order",
        "selection_orphan",
        "decision_unknown_assessment",
        "decision_cross_application",
    ],
)
def test_get_consistency_review_projection_fails_closed_on_ledger_drift(
    monkeypatch: pytest.MonkeyPatch,
    drift_kind: str,
) -> None:
    snapshot, authenticated_source, fixture_ids = _build_projection_fixture()
    first_decision, first_selections = _decision_record(
        decision_id=_uuid("drift-first"),
        project_id=fixture_ids["project_id"],
        application_id=fixture_ids["application_id"],
        assessment_id=fixture_ids["assessment_a_id"],
        source_application_id=fixture_ids["source_application_id"],
        source_candidate_id=fixture_ids["candidate_a_id"],
        actor_id=fixture_ids["actor_id"],
        decision_no=1,
        supersedes_decision_id=None,
        decision_kind="select_one",
        comment=None,
        selected_fact_value_ids=(fixture_ids["fv_a1"],),
        created_at=fixture_ids["created_at"],
    )
    second_decision, second_selections = _decision_record(
        decision_id=_uuid("drift-second"),
        project_id=fixture_ids["project_id"],
        application_id=fixture_ids["application_id"],
        assessment_id=fixture_ids["assessment_a_id"],
        source_application_id=fixture_ids["source_application_id"],
        source_candidate_id=fixture_ids["candidate_a_id"],
        actor_id=fixture_ids["actor_id"],
        decision_no=2,
        supersedes_decision_id=first_decision.id,
        decision_kind="keep_multiple",
        comment=None,
        selected_fact_value_ids=(fixture_ids["fv_a2"], fixture_ids["fv_a1"]),
        created_at=fixture_ids["created_at"],
    )
    decisions = [first_decision, second_decision]
    selections = list(first_selections + second_selections)
    if drift_kind == "assessment_missing":
        snapshot = ConsistencyProjectionSnapshot(
            application=replace(snapshot.application, assessment_count=3),
            batches=snapshot.batches,
            assessments=snapshot.assessments[:-1],
            citations=snapshot.citations,
            source_rows=snapshot.source_rows,
        )
    elif drift_kind == "assessment_extra":
        extra = replace(
            snapshot.assessments[0],
            id=_uuid("assessment-extra"),
            source_consistency_candidate_id=_uuid("candidate-extra"),
        )
        snapshot = ConsistencyProjectionSnapshot(
            application=replace(snapshot.application, assessment_count=4),
            batches=snapshot.batches,
            assessments=snapshot.assessments + (extra,),
            citations=snapshot.citations,
            source_rows=snapshot.source_rows,
        )
    elif drift_kind == "assessment_hash":
        snapshot = ConsistencyProjectionSnapshot(
            application=snapshot.application,
            batches=snapshot.batches,
            assessments=(
                replace(snapshot.assessments[0], assessment_manifest_hash="0" * 64),
                *snapshot.assessments[1:],
            ),
            citations=snapshot.citations,
            source_rows=snapshot.source_rows,
        )
    elif drift_kind == "result_hash":
        snapshot = ConsistencyProjectionSnapshot(
            application=replace(snapshot.application, result_manifest_hash="0" * 64),
            batches=snapshot.batches,
            assessments=snapshot.assessments,
            citations=snapshot.citations,
            source_rows=snapshot.source_rows,
        )
    elif drift_kind == "citation_binding":
        snapshot = ConsistencyProjectionSnapshot(
            application=snapshot.application,
            batches=snapshot.batches,
            assessments=snapshot.assessments,
            citations=(
                replace(snapshot.citations[0], source_fact_value_id=_uuid("wrong-fv")),
                *snapshot.citations[1:],
            ),
            source_rows=snapshot.source_rows,
        )
    elif drift_kind == "decision_gap":
        decisions[1] = replace(decisions[1], decision_no=3)
    elif drift_kind == "decision_wrong_predecessor":
        decisions[1] = replace(
            decisions[1],
            supersedes_decision_id=_uuid("wrong-predecessor"),
        )
    elif drift_kind == "decision_manifest":
        decisions[0] = replace(decisions[0], decision_manifest_hash="0" * 64)
    elif drift_kind == "selection_order":
        selections[-1] = replace(selections[-1], selection_order=2)
    elif drift_kind == "selection_orphan":
        selections.append(
            replace(
                selections[0],
                id=_uuid("orphan-selection"),
                decision_id=_uuid("missing-decision"),
            )
        )
    elif drift_kind == "decision_unknown_assessment":
        decisions[0] = replace(
            decisions[0],
            assessment_id=_uuid("unknown-assessment"),
        )
    else:
        decisions[0] = replace(
            decisions[0],
            consistency_check_application_id=_uuid("other-application"),
        )
    _install_snapshot_monkeypatches(
        monkeypatch,
        snapshot=snapshot,
        authenticated_source=authenticated_source,
        decisions=tuple(decisions),
        selections=tuple(selections),
    )

    with pytest.raises(
        projection_service.ConsistencyProjectionInvariantError,
        match="consistency_review_projection_immutable_ledger_mismatch",
    ):
        run_async(
            projection_service.get_consistency_review_projection(
                SessionFactory(),
                project_id=snapshot.application.project_id,
                consistency_check_application_id=snapshot.application.id,
            )
        )


def test_get_consistency_review_projection_contract_is_shared_across_value_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, authenticated_source, _fixture_ids = _build_projection_fixture()
    _install_snapshot_monkeypatches(
        monkeypatch,
        snapshot=snapshot,
        authenticated_source=authenticated_source,
    )

    result = run_async(
        projection_service.get_consistency_review_projection(
            SessionFactory(),
            project_id=snapshot.application.project_id,
            consistency_check_application_id=snapshot.application.id,
        )
    )

    member_shapes = [
        (
            member.fact_value_id,
            member.value_type,
            hasattr(member, "value_json"),
            hasattr(member, "normalized_value_text"),
            hasattr(member, "referenced_entity_id"),
            isinstance(member.evidences, tuple),
        )
        for item in result.items
        for member in item.members
    ]
    assert all(shape[2:] == (True, True, True, True) for shape in member_shapes)


def test_get_consistency_review_projection_does_not_leak_sensitive_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, authenticated_source, _fixture_ids = _build_projection_fixture()
    sentinel = "SENSITIVE_EXPLANATION_SENTINEL"
    snapshot = ConsistencyProjectionSnapshot(
        application=replace(snapshot.application, result_manifest_hash="0" * 64),
        batches=snapshot.batches,
        assessments=(
            replace(snapshot.assessments[0], explanation=sentinel),
            *snapshot.assessments[1:],
        ),
        citations=snapshot.citations,
        source_rows=snapshot.source_rows,
    )
    _install_snapshot_monkeypatches(
        monkeypatch,
        snapshot=snapshot,
        authenticated_source=authenticated_source,
    )

    with pytest.raises(
        projection_service.ConsistencyProjectionInvariantError,
        match="consistency_review_projection_immutable_ledger_mismatch",
    ) as exc_info:
        run_async(
            projection_service.get_consistency_review_projection(
                SessionFactory(),
                project_id=snapshot.application.project_id,
                consistency_check_application_id=snapshot.application.id,
            )
        )

    assert sentinel not in str(exc_info.value)
