from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.consistency_check import validate_consistency_check_batch_plan, validate_consistency_check_prompt
from app.agents.prompt_registry import PromptDefinition
from app.models.consistency_check import (
    ConsistencyAssessmentCitation,
    ConsistencyAssessmentLedger,
    ConsistencyCheckApplication,
    ConsistencyCheckBatchLedger,
)
from app.repositories import consistency_check as consistency_check_repository
from app.repositories import fact_value_duplicate_grouping as duplicate_grouping_repository
from app.repositories import consistency_projection as consistency_projection_repository
from app.schemas.consistency_check import ConsistencyCheckPlan
from app.schemas.consistency_check_execution import (
    ConsistencyCheckAssessment,
    ConsistencyCheckResponse,
    ConsistencyCheckBatchExecutionResult,
    ConsistencyCheckPlanExecutionResult,
)
from app.schemas.consistency_check_persistence import (
    ConsistencyAssessmentCitationSpec,
    ConsistencyAssessmentCitationLedgerRecord,
    ConsistencyAssessmentSpec,
    ConsistencyAssessmentLedgerRecord,
    ConsistencyCheckApplicationLedgerRecord,
    ConsistencyCheckBatchLedgerRecord,
    ConsistencyCheckBatchSpec,
    ConsistencyCheckPersistencePlan,
    ConsistencyCheckPersistenceResult,
)
from app.services import consistency_check as consistency_check_service
from app.services import consistency_check_execution as execution_service
from app.services import fact_value_duplicate_grouping as duplicate_grouping_service
from app.services import inference as inference_service


_APPLICATION_UNIQUE_CONSTRAINT = "uq_ccapp_exec_identity_hash"


class ConsistencyCheckPersistenceError(Exception):
    """Base class for consistency-check ledger persistence failures."""


class ConsistencyCheckPersistenceStateError(ConsistencyCheckPersistenceError):
    """Raised when the input plan, result, or source ledger is not admissible."""


class ConsistencyCheckPersistenceInvariantError(ConsistencyCheckPersistenceError):
    """Raised when immutable persisted ledgers diverge from the authoritative projection."""


@dataclass(frozen=True, slots=True)
class AuthenticatedConsistencyCheckLedgerProjectionContext:
    application: ConsistencyCheckApplicationLedgerRecord
    authenticated_source: duplicate_grouping_service.AuthenticatedFactValueConsistencyCandidateApplication
    candidate_bundles: tuple[object, ...]
    source_rows: tuple[consistency_projection_repository.ConsistencyProjectionSourceRow, ...]
    batches: tuple[ConsistencyCheckBatchLedgerRecord, ...]
    assessments: tuple[ConsistencyAssessmentLedgerRecord, ...]
    citations: tuple[ConsistencyAssessmentCitationLedgerRecord, ...]


def _require_plan(plan: ConsistencyCheckPlan) -> ConsistencyCheckPlan:
    if not isinstance(plan, ConsistencyCheckPlan):
        raise ConsistencyCheckPersistenceStateError("consistency_check_persistence_plan_invalid")
    return plan


def _require_execution_result(
    execution_result: ConsistencyCheckPlanExecutionResult,
) -> ConsistencyCheckPlanExecutionResult:
    if not isinstance(execution_result, ConsistencyCheckPlanExecutionResult):
        raise ConsistencyCheckPersistenceStateError(
            "consistency_check_persistence_execution_result_invalid"
        )
    return execution_result


def _get_integrity_constraint_name(error: IntegrityError) -> str | None:
    diag = getattr(error.orig, "diag", None)
    constraint_name = getattr(diag, "constraint_name", None)
    if constraint_name is None or not isinstance(constraint_name, str):
        return None
    return constraint_name


def _immutable_mismatch() -> None:
    raise ConsistencyCheckPersistenceInvariantError(
        "consistency_check_persistence_immutable_ledger_mismatch"
    )


def _normalize_execution_identity_inputs(
    *,
    provider: str,
    requested_model: str,
) -> tuple[str, str]:
    try:
        return (
            inference_service.normalize_inference_identity_text(
                provider,
                field_name="provider",
            ),
            inference_service.normalize_inference_identity_text(
                requested_model,
                field_name="requested_model",
            ),
        )
    except inference_service.InvalidInferenceInputError:
        raise ConsistencyCheckPersistenceStateError(
            "consistency_check_persistence_execution_identity_invalid"
        ) from None


def _validate_source_bindings(
    *,
    plan: ConsistencyCheckPlan,
    authenticated_source: duplicate_grouping_service.AuthenticatedFactValueConsistencyCandidateApplication,
) -> None:
    if plan.project_id != authenticated_source.project_id:
        raise ConsistencyCheckPersistenceStateError(
            "consistency_check_persistence_project_id_mismatch"
        )
    if plan.consistency_application_id != authenticated_source.application.id:
        raise ConsistencyCheckPersistenceStateError(
            "consistency_check_persistence_source_application_mismatch"
        )
    if plan.source_result_manifest_hash != authenticated_source.application.result_manifest_hash:
        raise ConsistencyCheckPersistenceStateError(
            "consistency_check_persistence_source_manifest_mismatch"
        )


async def _rebuild_and_validate_authoritative_plan(
    session_factory: Callable[[], AsyncSession],
    *,
    plan: ConsistencyCheckPlan,
) -> ConsistencyCheckPlan:
    provided_plan = _require_plan(plan)
    authoritative_plan = await consistency_check_service.build_consistency_check_plan(
        session_factory,
        consistency_application_id=provided_plan.consistency_application_id,
        config=provided_plan.config,
    )
    for batch in authoritative_plan.batches:
        validate_consistency_check_batch_plan(plan=authoritative_plan, batch=batch)
    for batch in provided_plan.batches:
        validate_consistency_check_batch_plan(plan=provided_plan, batch=batch)
    if provided_plan != authoritative_plan:
        raise ConsistencyCheckPersistenceStateError(
            "consistency_check_persistence_plan_mismatch"
        )
    return authoritative_plan


async def _authenticate_batch_results(
    session_factory: Callable[[], AsyncSession],
    *,
    authoritative_plan: ConsistencyCheckPlan,
    execution_result: ConsistencyCheckPlanExecutionResult,
    prompt: PromptDefinition,
    provider: str,
    requested_model: str,
) -> tuple[ConsistencyCheckBatchExecutionResult, ...]:
    if len(execution_result.inference_run_ids) != len(authoritative_plan.batches):
        raise ConsistencyCheckPersistenceStateError(
            "consistency_check_persistence_inference_run_count_mismatch"
        )

    authenticated_batch_results: list[ConsistencyCheckBatchExecutionResult] = []
    for batch, inference_run_id in zip(
        authoritative_plan.batches,
        execution_result.inference_run_ids,
        strict=True,
    ):
        batch_result = await execution_service.authenticate_completed_consistency_check_batch_run(
            session_factory,
            authoritative_plan=authoritative_plan,
            batch=batch,
            prompt=prompt,
            provider=provider,
            requested_model=requested_model,
            inference_run_id=inference_run_id,
        )
        authenticated_batch_results.append(batch_result)
    return tuple(authenticated_batch_results)


def _authenticate_execution_result(
    *,
    authoritative_plan: ConsistencyCheckPlan,
    execution_result: ConsistencyCheckPlanExecutionResult,
    batch_results: Sequence[ConsistencyCheckBatchExecutionResult],
) -> ConsistencyCheckPlanExecutionResult:
    ordered_assessments_by_batch: list[tuple] = []
    for batch, batch_result in zip(authoritative_plan.batches, batch_results, strict=True):
        ordered_assessments_by_batch.append(
            execution_service._validate_and_order_batch_result(
                authoritative_plan=authoritative_plan,
                batch=batch,
                result=batch_result,
            )
        )
    expected_result = execution_service._build_plan_execution_result(
        authoritative_plan=authoritative_plan,
        ordered_batch_results=batch_results,
        ordered_assessments_by_batch=ordered_assessments_by_batch,
    )
    execution_service._validate_plan_execution_result_matches_expected(
        expected=expected_result,
        actual=execution_result,
    )
    return expected_result


def _build_execution_identity_hash(
    *,
    authoritative_plan: ConsistencyCheckPlan,
    prompt: PromptDefinition,
    provider: str,
    requested_model: str,
) -> str:
    return duplicate_grouping_service.hash_deterministic_payload(
        {
            "project_id": str(authoritative_plan.project_id),
            "consistency_application_id": str(authoritative_plan.consistency_application_id),
            "plan_manifest_hash": authoritative_plan.plan_manifest_hash,
            "prompt_contract_hash": prompt.contract_hash,
            "provider": provider,
            "requested_model": requested_model,
            "executor_name": execution_service.CONSISTENCY_CHECK_EXECUTOR_NAME,
            "executor_version": execution_service.CONSISTENCY_CHECK_EXECUTOR_VERSION,
        }
    )


def _build_citation_specs(
    *,
    candidate: object,
    assessment: object,
) -> tuple[tuple[uuid.UUID, uuid.UUID, int], ...]:
    fact_value_id_by_link_id: dict[uuid.UUID, uuid.UUID] = {}
    for member in candidate.members:
        for evidence in member.evidences:
            prior = fact_value_id_by_link_id.get(evidence.evidence_link_id)
            if prior is not None and prior != member.fact_value_id:
                raise ConsistencyCheckPersistenceInvariantError(
                    "consistency_check_persistence_candidate_evidence_binding_mismatch"
                )
            fact_value_id_by_link_id[evidence.evidence_link_id] = member.fact_value_id

    citation_specs: list[tuple[uuid.UUID, uuid.UUID, int]] = []
    for citation_order, evidence_link_id in enumerate(assessment.cited_evidence_link_ids):
        source_fact_value_id = fact_value_id_by_link_id.get(evidence_link_id)
        if source_fact_value_id is None:
            raise ConsistencyCheckPersistenceStateError(
                "consistency_check_persistence_citation_binding_mismatch"
            )
        citation_specs.append((source_fact_value_id, evidence_link_id, citation_order))
    return tuple(citation_specs)


def _build_persistence_plan(
    *,
    authoritative_plan: ConsistencyCheckPlan,
    authenticated_source: duplicate_grouping_service.AuthenticatedFactValueConsistencyCandidateApplication,
    authenticated_execution_result: ConsistencyCheckPlanExecutionResult,
    batch_results: Sequence[ConsistencyCheckBatchExecutionResult],
    prompt: PromptDefinition,
    provider: str,
    requested_model: str,
) -> ConsistencyCheckPersistencePlan:
    assessment_by_candidate_id = {
        assessment.candidate_id: assessment
        for assessment in authenticated_execution_result.assessments
    }

    batch_specs = tuple(
        ConsistencyCheckBatchSpec(
            batch_index=batch_result.batch_index,
            batch_manifest_hash=batch_result.batch_manifest_hash,
            skipped_empty=batch_result.skipped_empty,
            input_batch_id=batch_result.input_batch_id,
            inference_run_id=batch_result.inference_run_id,
            request_hash=batch_result.request_hash,
            message_content_hash=batch_result.message_content_hash,
        )
        for batch_result in batch_results
    )

    assessment_specs: list[ConsistencyAssessmentSpec] = []
    for batch in authoritative_plan.batches:
        for candidate in batch.candidates:
            assessment = assessment_by_candidate_id.get(candidate.candidate_id)
            if assessment is None:
                raise ConsistencyCheckPersistenceStateError(
                    "consistency_check_persistence_assessment_missing"
                )
            raw_citations = _build_citation_specs(candidate=candidate, assessment=assessment)
            citations = tuple(
                ConsistencyAssessmentCitationSpec(
                    assessment_source_consistency_application_id=authenticated_source.application.id,
                    assessment_source_consistency_candidate_id=candidate.candidate_id,
                    source_fact_value_id=citation[0],
                    evidence_link_id=citation[1],
                    citation_order=citation[2],
                )
                for citation in raw_citations
            )
            assessment_specs.append(
                ConsistencyAssessmentSpec(
                    source_consistency_application_id=authenticated_source.application.id,
                    source_consistency_candidate_id=candidate.candidate_id,
                    batch_index=batch.batch_index,
                    verdict=assessment.verdict,
                    severity=assessment.severity,
                    confidence=assessment.confidence,
                    explanation=assessment.explanation,
                    impact_json=tuple(assessment.impact),
                    recommended_actions_json=tuple(assessment.recommended_actions),
                    assessment_manifest_hash=duplicate_grouping_service.hash_deterministic_payload(
                        execution_service._assessment_manifest_payload(assessment)
                    ),
                    citations=citations,
                )
            )

    return ConsistencyCheckPersistencePlan(
        project_id=authoritative_plan.project_id,
        consistency_application_id=authoritative_plan.consistency_application_id,
        orchestration_id=authenticated_source.application.orchestration_id,
        source_result_manifest_hash=authoritative_plan.source_result_manifest_hash,
        plan_manifest_hash=authoritative_plan.plan_manifest_hash,
        execution_identity_hash=_build_execution_identity_hash(
            authoritative_plan=authoritative_plan,
            prompt=prompt,
            provider=provider,
            requested_model=requested_model,
        ),
        result_manifest_hash=authenticated_execution_result.result_manifest_hash,
        prompt_contract_hash=prompt.contract_hash,
        provider=provider,
        requested_model=requested_model,
        executor_name=execution_service.CONSISTENCY_CHECK_EXECUTOR_NAME,
        executor_version=execution_service.CONSISTENCY_CHECK_EXECUTOR_VERSION,
        batch_count=authenticated_execution_result.batch_count,
        executed_batch_count=authenticated_execution_result.executed_batch_count,
        skipped_empty_batch_count=authenticated_execution_result.skipped_empty_batch_count,
        inference_run_count=sum(1 for run_id in authenticated_execution_result.inference_run_ids if run_id is not None),
        assessment_count=len(authenticated_execution_result.assessments),
        batches=batch_specs,
        assessments=tuple(assessment_specs),
    )


async def _reauthenticate_source_in_write_transaction(
    session: AsyncSession,
    *,
    authenticated_source: duplicate_grouping_service.AuthenticatedFactValueConsistencyCandidateApplication,
) -> None:
    current_source_application = (
        await duplicate_grouping_repository.get_consistency_candidate_application_ledger_by_id(
            session,
            consistency_application_id=authenticated_source.application.id,
        )
    )
    if current_source_application is None:
        raise ConsistencyCheckPersistenceStateError(
            "consistency_check_persistence_source_application_not_found"
        )
    duplicate_grouping_service._assert_consistency_application_matches_plan(
        current_source_application,
        duplicate_grouping_application_id=authenticated_source.application.duplicate_grouping_application_id,
        orchestration_id=authenticated_source.application.orchestration_id,
        extraction_run_id=authenticated_source.application.extraction_run_id,
        plan=authenticated_source.write_plan,
    )
    state = await duplicate_grouping_repository.get_duplicate_grouping_orchestration_state(
        session,
        orchestration_id=current_source_application.orchestration_id,
    )
    state = duplicate_grouping_service._validate_run_state(
        state,
        orchestration_id=current_source_application.orchestration_id,
    )
    if state.project_id != authenticated_source.project_id:
        raise ConsistencyCheckPersistenceStateError(
            "consistency_check_persistence_source_project_drift"
        )
    if await duplicate_grouping_repository.has_invalid_completed_batch_bindings(
        session,
        orchestration_id=current_source_application.orchestration_id,
    ):
        raise ConsistencyCheckPersistenceInvariantError(
            "consistency_check_persistence_source_binding_mismatch"
        )
    candidate_ledgers = await duplicate_grouping_repository.list_consistency_candidate_ledgers(
        session,
        consistency_application_id=current_source_application.id,
    )
    member_ledgers = await duplicate_grouping_repository.list_consistency_candidate_member_ledgers(
        session,
        consistency_application_id=current_source_application.id,
    )
    duplicate_grouping_service._compute_consistency_candidate_result_manifest_hash(
        candidate_ledgers,
        member_ledgers,
        application=current_source_application,
        plan=authenticated_source.write_plan,
    )


def _assert_application_matches_plan(
    application: ConsistencyCheckApplicationLedgerRecord,
    *,
    persistence_plan: ConsistencyCheckPersistencePlan,
) -> None:
    if application.project_id != persistence_plan.project_id:
        _immutable_mismatch()
    if application.consistency_application_id != persistence_plan.consistency_application_id:
        _immutable_mismatch()
    if application.orchestration_id != persistence_plan.orchestration_id:
        _immutable_mismatch()
    if application.source_result_manifest_hash != persistence_plan.source_result_manifest_hash:
        _immutable_mismatch()
    if application.plan_manifest_hash != persistence_plan.plan_manifest_hash:
        _immutable_mismatch()
    if application.execution_identity_hash != persistence_plan.execution_identity_hash:
        _immutable_mismatch()
    if application.result_manifest_hash != persistence_plan.result_manifest_hash:
        _immutable_mismatch()
    if application.prompt_contract_hash != persistence_plan.prompt_contract_hash:
        _immutable_mismatch()
    if application.provider != persistence_plan.provider:
        _immutable_mismatch()
    if application.requested_model != persistence_plan.requested_model:
        _immutable_mismatch()
    if application.executor_name != persistence_plan.executor_name:
        _immutable_mismatch()
    if application.executor_version != persistence_plan.executor_version:
        _immutable_mismatch()
    if application.batch_count != persistence_plan.batch_count:
        _immutable_mismatch()
    if application.executed_batch_count != persistence_plan.executed_batch_count:
        _immutable_mismatch()
    if application.skipped_empty_batch_count != persistence_plan.skipped_empty_batch_count:
        _immutable_mismatch()
    if application.inference_run_count != persistence_plan.inference_run_count:
        _immutable_mismatch()
    if application.assessment_count != persistence_plan.assessment_count:
        _immutable_mismatch()


def _assert_batch_record_matches_spec(
    record: ConsistencyCheckBatchLedgerRecord,
    *,
    application_id: uuid.UUID,
    spec: ConsistencyCheckBatchSpec,
) -> None:
    if record.consistency_check_application_id != application_id:
        _immutable_mismatch()
    if record.batch_index != spec.batch_index:
        _immutable_mismatch()
    if record.batch_manifest_hash != spec.batch_manifest_hash:
        _immutable_mismatch()
    if record.skipped_empty != spec.skipped_empty:
        _immutable_mismatch()
    if record.input_batch_id != spec.input_batch_id:
        _immutable_mismatch()
    if record.inference_run_id != spec.inference_run_id:
        _immutable_mismatch()
    if record.request_hash != spec.request_hash:
        _immutable_mismatch()
    if record.message_content_hash != spec.message_content_hash:
        _immutable_mismatch()


def _build_unique_key_map(
    records: Sequence[object],
    *,
    key_builder: Callable[[object], object],
) -> dict[object, object]:
    keyed: dict[object, object] = {}
    for record in records:
        key = key_builder(record)
        if key in keyed:
            _immutable_mismatch()
        keyed[key] = record
    return keyed


async def _assert_existing_ledgers_match_plan(
    session: AsyncSession,
    *,
    application: ConsistencyCheckApplicationLedgerRecord,
    persistence_plan: ConsistencyCheckPersistencePlan,
) -> None:
    _assert_application_matches_plan(application, persistence_plan=persistence_plan)

    actual_batches = await consistency_check_repository.list_consistency_check_batch_ledgers(
        session,
        consistency_check_application_id=application.id,
    )
    actual_batch_by_index = _build_unique_key_map(
        actual_batches,
        key_builder=lambda record: record.batch_index,
    )
    expected_batch_by_index = _build_unique_key_map(
        persistence_plan.batches,
        key_builder=lambda spec: spec.batch_index,
    )
    if set(actual_batch_by_index) != set(expected_batch_by_index):
        _immutable_mismatch()
    for batch_index in sorted(expected_batch_by_index):
        _assert_batch_record_matches_spec(
            actual_batch_by_index[batch_index],
            application_id=application.id,
            spec=expected_batch_by_index[batch_index],
        )

    actual_assessments = await consistency_check_repository.list_consistency_assessment_ledgers(
        session,
        consistency_check_application_id=application.id,
    )
    actual_citations = await consistency_check_repository.list_consistency_assessment_citation_ledgers(
        session,
        consistency_check_application_id=application.id,
    )
    actual_assessment_by_key = _build_unique_key_map(
        actual_assessments,
        key_builder=lambda record: (
            record.batch_index,
            record.source_consistency_candidate_id,
        ),
    )
    expected_assessment_by_key = _build_unique_key_map(
        persistence_plan.assessments,
        key_builder=lambda spec: (
            spec.batch_index,
            spec.source_consistency_candidate_id,
        ),
    )
    if set(actual_assessment_by_key) != set(expected_assessment_by_key):
        _immutable_mismatch()
    if len(actual_citations) != sum(
        len(assessment.citations) for assessment in persistence_plan.assessments
    ):
        _immutable_mismatch()
    citations_by_assessment_id: dict[uuid.UUID, list] = {}
    for citation in actual_citations:
        citations_by_assessment_id.setdefault(citation.assessment_id, []).append(citation)
    if not set(citations_by_assessment_id).issubset(
        {record.id for record in actual_assessment_by_key.values()}
    ):
        _immutable_mismatch()

    for assessment_key in sorted(expected_assessment_by_key):
        actual_assessment = actual_assessment_by_key[assessment_key]
        expected_assessment = expected_assessment_by_key[assessment_key]
        if actual_assessment.consistency_check_application_id != application.id:
            _immutable_mismatch()
        if (
            actual_assessment.source_consistency_application_id
            != expected_assessment.source_consistency_application_id
        ):
            _immutable_mismatch()
        if (
            actual_assessment.source_consistency_candidate_id
            != expected_assessment.source_consistency_candidate_id
        ):
            _immutable_mismatch()
        if actual_assessment.batch_index != expected_assessment.batch_index:
            _immutable_mismatch()
        if actual_assessment.verdict != expected_assessment.verdict:
            _immutable_mismatch()
        if actual_assessment.severity != expected_assessment.severity:
            _immutable_mismatch()
        if actual_assessment.confidence != expected_assessment.confidence:
            _immutable_mismatch()
        if actual_assessment.explanation != expected_assessment.explanation:
            _immutable_mismatch()
        if actual_assessment.impact_json != expected_assessment.impact_json:
            _immutable_mismatch()
        if (
            actual_assessment.recommended_actions_json
            != expected_assessment.recommended_actions_json
        ):
            _immutable_mismatch()
        if (
            actual_assessment.assessment_manifest_hash
            != expected_assessment.assessment_manifest_hash
        ):
            _immutable_mismatch()

        actual_assessment_citations = citations_by_assessment_id.get(actual_assessment.id, [])
        actual_citation_by_order = _build_unique_key_map(
            actual_assessment_citations,
            key_builder=lambda record: record.citation_order,
        )
        expected_citation_by_order = _build_unique_key_map(
            expected_assessment.citations,
            key_builder=lambda spec: spec.citation_order,
        )
        if set(actual_citation_by_order) != set(expected_citation_by_order):
            _immutable_mismatch()
        for citation_order in sorted(expected_citation_by_order):
            actual_citation = actual_citation_by_order[citation_order]
            expected_citation = expected_citation_by_order[citation_order]
            if actual_citation.assessment_id != actual_assessment.id:
                _immutable_mismatch()
            if (
                actual_citation.source_consistency_application_id
                != expected_citation.assessment_source_consistency_application_id
            ):
                _immutable_mismatch()
            if (
                actual_citation.source_consistency_candidate_id
                != expected_citation.assessment_source_consistency_candidate_id
            ):
                _immutable_mismatch()
            if actual_citation.source_fact_value_id != expected_citation.source_fact_value_id:
                _immutable_mismatch()
            if actual_citation.evidence_link_id != expected_citation.evidence_link_id:
                _immutable_mismatch()
            if actual_citation.citation_order != expected_citation.citation_order:
                _immutable_mismatch()


async def authenticate_persisted_consistency_check_application(
    session_factory: Callable[[], AsyncSession],
    *,
    project_id: uuid.UUID,
    consistency_check_application_id: uuid.UUID,
) -> AuthenticatedConsistencyCheckLedgerProjectionContext:
    if not isinstance(project_id, uuid.UUID):
        raise ConsistencyCheckPersistenceStateError(
            "consistency_check_persistence_project_id_invalid"
        )
    if not isinstance(consistency_check_application_id, uuid.UUID):
        raise ConsistencyCheckPersistenceStateError(
            "consistency_check_persistence_application_id_invalid"
        )

    async with session_factory() as read_session:
        try:
            snapshot = await consistency_projection_repository.load_consistency_projection_snapshot(
                read_session,
                consistency_check_application_id=consistency_check_application_id,
            )
        except BaseException:
            await read_session.rollback()
            raise
        else:
            await read_session.rollback()
    if snapshot is None:
        raise ConsistencyCheckPersistenceStateError(
            "consistency_check_persistence_application_not_found"
        )
    if snapshot.application.project_id != project_id:
        raise ConsistencyCheckPersistenceStateError(
            "consistency_check_persistence_project_id_mismatch"
        )

    authenticated_source = (
        await duplicate_grouping_service.authenticate_fact_value_consistency_candidate_application(
            session_factory,
            consistency_application_id=snapshot.application.consistency_application_id,
        )
    )
    if snapshot.application.project_id != authenticated_source.project_id:
        raise ConsistencyCheckPersistenceStateError(
            "consistency_check_persistence_project_id_mismatch"
        )
    if snapshot.application.consistency_application_id != authenticated_source.application.id:
        raise ConsistencyCheckPersistenceStateError(
            "consistency_check_persistence_source_application_mismatch"
        )
    if snapshot.application.orchestration_id != authenticated_source.application.orchestration_id:
        raise ConsistencyCheckPersistenceStateError(
            "consistency_check_persistence_source_orchestration_mismatch"
        )
    if (
        snapshot.application.source_result_manifest_hash
        != authenticated_source.application.result_manifest_hash
    ):
        raise ConsistencyCheckPersistenceStateError(
            "consistency_check_persistence_source_manifest_mismatch"
        )

    candidate_bundles = consistency_check_service._build_candidate_bundles(
        authenticated_source,
        snapshot.source_rows,
    )
    candidate_by_id = _build_unique_key_map(
        candidate_bundles,
        key_builder=lambda candidate: candidate.candidate_id,
    )
    batch_by_index = _build_unique_key_map(
        snapshot.batches,
        key_builder=lambda batch: batch.batch_index,
    )
    assessment_by_key = _build_unique_key_map(
        snapshot.assessments,
        key_builder=lambda assessment: (
            assessment.batch_index,
            assessment.source_consistency_candidate_id,
        ),
    )
    assessment_by_id = _build_unique_key_map(
        snapshot.assessments,
        key_builder=lambda assessment: assessment.id,
    )
    if len(snapshot.batches) != snapshot.application.batch_count:
        _immutable_mismatch()
    if len(snapshot.assessments) != snapshot.application.assessment_count:
        _immutable_mismatch()
    if sum(1 for batch in snapshot.batches if batch.skipped_empty) != snapshot.application.skipped_empty_batch_count:
        _immutable_mismatch()
    if sum(1 for batch in snapshot.batches if batch.inference_run_id is not None) != snapshot.application.inference_run_count:
        _immutable_mismatch()
    if set(batch_by_index) != set(range(snapshot.application.batch_count)):
        _immutable_mismatch()
    if len(candidate_bundles) != len(snapshot.assessments):
        _immutable_mismatch()
    if {candidate.candidate_id for candidate in candidate_bundles} != {
        assessment.source_consistency_candidate_id for assessment in snapshot.assessments
    }:
        _immutable_mismatch()

    citations_by_assessment_id: dict[uuid.UUID, list[ConsistencyAssessmentCitationLedgerRecord]] = {}
    for citation in snapshot.citations:
        if citation.assessment_id not in assessment_by_id:
            _immutable_mismatch()
        citations_by_assessment_id.setdefault(citation.assessment_id, []).append(citation)

    ordered_batch_results: list[ConsistencyCheckBatchExecutionResult] = []
    ordered_assessments_by_batch: list[tuple[ConsistencyCheckAssessment, ...]] = []
    for batch_index in range(snapshot.application.batch_count):
        batch = batch_by_index[batch_index]
        batch_assessments: list[ConsistencyCheckAssessment] = []
        for candidate in candidate_bundles:
            assessment_record = assessment_by_key.get((batch_index, candidate.candidate_id))
            if assessment_record is None:
                continue
            if (
                assessment_record.consistency_check_application_id != snapshot.application.id
                or assessment_record.source_consistency_application_id
                != snapshot.application.consistency_application_id
                or candidate.candidate_id not in candidate_by_id
            ):
                _immutable_mismatch()

            member_by_fact_value_id = {
                member.fact_value_id: {
                    evidence.evidence_link_id for evidence in member.evidences
                }
                for member in candidate.members
            }
            citations = citations_by_assessment_id.get(assessment_record.id, [])
            citation_by_order = _build_unique_key_map(
                citations,
                key_builder=lambda citation: citation.citation_order,
            )
            cited_evidence_link_ids: list[uuid.UUID] = []
            for citation_order in sorted(citation_by_order):
                citation = citation_by_order[citation_order]
                if (
                    citation.source_consistency_application_id
                    != snapshot.application.consistency_application_id
                    or citation.source_consistency_candidate_id != candidate.candidate_id
                ):
                    _immutable_mismatch()
                member_evidence_ids = member_by_fact_value_id.get(citation.source_fact_value_id)
                if member_evidence_ids is None or citation.evidence_link_id not in member_evidence_ids:
                    _immutable_mismatch()
                cited_evidence_link_ids.append(citation.evidence_link_id)

            assessment_model = ConsistencyCheckAssessment(
                candidate_id=assessment_record.source_consistency_candidate_id,
                verdict=assessment_record.verdict,
                severity=assessment_record.severity,
                confidence=assessment_record.confidence,
                explanation=assessment_record.explanation,
                cited_evidence_link_ids=cited_evidence_link_ids,
                impact=list(assessment_record.impact_json),
                recommended_actions=list(assessment_record.recommended_actions_json),
            )
            recomputed_hash = duplicate_grouping_service.hash_deterministic_payload(
                execution_service._assessment_manifest_payload(assessment_model)
            )
            if recomputed_hash != assessment_record.assessment_manifest_hash:
                _immutable_mismatch()
            batch_assessments.append(assessment_model)

        if batch.skipped_empty != (len(batch_assessments) == 0):
            _immutable_mismatch()
        ordered_assessments = tuple(batch_assessments)
        ordered_assessments_by_batch.append(ordered_assessments)
        ordered_batch_results.append(
            ConsistencyCheckBatchExecutionResult(
                project_id=snapshot.application.project_id,
                consistency_application_id=snapshot.application.consistency_application_id,
                source_result_manifest_hash=snapshot.application.source_result_manifest_hash,
                plan_manifest_hash=snapshot.application.plan_manifest_hash,
                batch_index=batch.batch_index,
                batch_manifest_hash=batch.batch_manifest_hash,
                input_batch_id=batch.input_batch_id,
                inference_run_id=batch.inference_run_id,
                request_hash=batch.request_hash,
                message_content_hash=batch.message_content_hash,
                skipped_empty=batch.skipped_empty,
                reused_completed_run=False,
                response=ConsistencyCheckResponse(assessments=list(ordered_assessments)),
                response_model=None,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
            )
        )

    result_manifest_hash = execution_service._build_plan_result_manifest_hash(
        authoritative_plan=SimpleNamespace(
            project_id=snapshot.application.project_id,
            consistency_application_id=snapshot.application.consistency_application_id,
            source_result_manifest_hash=snapshot.application.source_result_manifest_hash,
            plan_manifest_hash=snapshot.application.plan_manifest_hash,
        ),
        ordered_batch_results=tuple(ordered_batch_results),
        ordered_assessments_by_batch=tuple(ordered_assessments_by_batch),
    )
    if result_manifest_hash != snapshot.application.result_manifest_hash:
        _immutable_mismatch()

    return AuthenticatedConsistencyCheckLedgerProjectionContext(
        application=snapshot.application,
        authenticated_source=authenticated_source,
        candidate_bundles=tuple(candidate_bundles),
        source_rows=snapshot.source_rows,
        batches=snapshot.batches,
        assessments=snapshot.assessments,
        citations=snapshot.citations,
    )

def _build_result(
    *,
    application_id: uuid.UUID,
    persistence_plan: ConsistencyCheckPersistencePlan,
    created_new: bool,
) -> ConsistencyCheckPersistenceResult:
    return ConsistencyCheckPersistenceResult(
        consistency_check_application_id=application_id,
        created_new=created_new,
        batch_count=persistence_plan.batch_count,
        assessment_count=persistence_plan.assessment_count,
    )


async def _read_existing_result(
    session: AsyncSession,
    *,
    execution_identity_hash: str,
    persistence_plan: ConsistencyCheckPersistencePlan,
) -> ConsistencyCheckPersistenceResult | None:
    existing_application = await consistency_check_repository.get_consistency_check_application_ledger_by_execution_identity(
        session,
        execution_identity_hash=execution_identity_hash,
    )
    if existing_application is None:
        return None
    await _assert_existing_ledgers_match_plan(
        session,
        application=existing_application,
        persistence_plan=persistence_plan,
    )
    return _build_result(
        application_id=existing_application.id,
        persistence_plan=persistence_plan,
        created_new=False,
    )


async def persist_consistency_check_plan_result(
    session_factory: Callable[[], AsyncSession],
    *,
    plan: ConsistencyCheckPlan,
    execution_result: ConsistencyCheckPlanExecutionResult,
    prompt: PromptDefinition,
    provider: str,
    requested_model: str,
) -> ConsistencyCheckPersistenceResult:
    validate_consistency_check_prompt(prompt)
    normalized_provider, normalized_requested_model = _normalize_execution_identity_inputs(
        provider=provider,
        requested_model=requested_model,
    )
    authoritative_plan = await _rebuild_and_validate_authoritative_plan(
        session_factory,
        plan=plan,
    )
    provided_execution_result = _require_execution_result(execution_result)

    authenticated_source = (
        await duplicate_grouping_service.authenticate_fact_value_consistency_candidate_application(
            session_factory,
            consistency_application_id=authoritative_plan.consistency_application_id,
        )
    )
    _validate_source_bindings(
        plan=authoritative_plan,
        authenticated_source=authenticated_source,
    )

    batch_results = await _authenticate_batch_results(
        session_factory,
        authoritative_plan=authoritative_plan,
        execution_result=provided_execution_result,
        prompt=prompt,
        provider=normalized_provider,
        requested_model=normalized_requested_model,
    )
    authenticated_execution_result = _authenticate_execution_result(
        authoritative_plan=authoritative_plan,
        execution_result=provided_execution_result,
        batch_results=batch_results,
    )
    persistence_plan = _build_persistence_plan(
        authoritative_plan=authoritative_plan,
        authenticated_source=authenticated_source,
        authenticated_execution_result=authenticated_execution_result,
        batch_results=batch_results,
        prompt=prompt,
        provider=normalized_provider,
        requested_model=normalized_requested_model,
    )

    async with session_factory() as write_session:
        try:
            await _reauthenticate_source_in_write_transaction(
                write_session,
                authenticated_source=authenticated_source,
            )
            existing_application = await consistency_check_repository.get_consistency_check_application_for_update(
                write_session,
                execution_identity_hash=persistence_plan.execution_identity_hash,
            )
            if existing_application is not None:
                existing_ledger = ConsistencyCheckApplicationLedgerRecord(
                    id=existing_application.id,
                    project_id=existing_application.project_id,
                    consistency_application_id=existing_application.consistency_application_id,
                    orchestration_id=existing_application.orchestration_id,
                    source_result_manifest_hash=existing_application.source_result_manifest_hash,
                    plan_manifest_hash=existing_application.plan_manifest_hash,
                    execution_identity_hash=existing_application.execution_identity_hash,
                    result_manifest_hash=existing_application.result_manifest_hash,
                    prompt_contract_hash=existing_application.prompt_contract_hash,
                    provider=existing_application.provider,
                    requested_model=existing_application.requested_model,
                    executor_name=existing_application.executor_name,
                    executor_version=existing_application.executor_version,
                    batch_count=existing_application.batch_count,
                    executed_batch_count=existing_application.executed_batch_count,
                    skipped_empty_batch_count=existing_application.skipped_empty_batch_count,
                    inference_run_count=existing_application.inference_run_count,
                    assessment_count=existing_application.assessment_count,
                    created_at=existing_application.created_at,
                )
                await _assert_existing_ledgers_match_plan(
                    write_session,
                    application=existing_ledger,
                    persistence_plan=persistence_plan,
                )
                await write_session.commit()
                return _build_result(
                    application_id=existing_ledger.id,
                    persistence_plan=persistence_plan,
                    created_new=False,
                )

            application = ConsistencyCheckApplication(
                id=uuid.uuid4(),
                project_id=persistence_plan.project_id,
                consistency_application_id=persistence_plan.consistency_application_id,
                orchestration_id=persistence_plan.orchestration_id,
                source_result_manifest_hash=persistence_plan.source_result_manifest_hash,
                plan_manifest_hash=persistence_plan.plan_manifest_hash,
                execution_identity_hash=persistence_plan.execution_identity_hash,
                result_manifest_hash=persistence_plan.result_manifest_hash,
                prompt_contract_hash=persistence_plan.prompt_contract_hash,
                provider=persistence_plan.provider,
                requested_model=persistence_plan.requested_model,
                executor_name=persistence_plan.executor_name,
                executor_version=persistence_plan.executor_version,
                batch_count=persistence_plan.batch_count,
                executed_batch_count=persistence_plan.executed_batch_count,
                skipped_empty_batch_count=persistence_plan.skipped_empty_batch_count,
                inference_run_count=persistence_plan.inference_run_count,
                assessment_count=persistence_plan.assessment_count,
            )
            await consistency_check_repository.create_consistency_check_application(
                write_session,
                application,
            )

            batch_rows = [
                ConsistencyCheckBatchLedger(
                    id=uuid.uuid4(),
                    consistency_check_application_id=application.id,
                    batch_index=batch.batch_index,
                    batch_manifest_hash=batch.batch_manifest_hash,
                    skipped_empty=batch.skipped_empty,
                    input_batch_id=batch.input_batch_id,
                    inference_run_id=batch.inference_run_id,
                    request_hash=batch.request_hash,
                    message_content_hash=batch.message_content_hash,
                )
                for batch in persistence_plan.batches
            ]
            if batch_rows:
                await consistency_check_repository.create_consistency_check_batches(
                    write_session,
                    batch_rows,
                )

            assessment_rows: list[ConsistencyAssessmentLedger] = []
            assessment_id_by_key: dict[tuple[uuid.UUID, int], uuid.UUID] = {}
            for assessment in persistence_plan.assessments:
                assessment_id = uuid.uuid4()
                assessment_id_by_key[
                    (
                        assessment.source_consistency_candidate_id,
                        assessment.batch_index,
                    )
                ] = assessment_id
                assessment_rows.append(
                    ConsistencyAssessmentLedger(
                        id=assessment_id,
                        consistency_check_application_id=application.id,
                        source_consistency_application_id=assessment.source_consistency_application_id,
                        source_consistency_candidate_id=assessment.source_consistency_candidate_id,
                        batch_index=assessment.batch_index,
                        verdict=assessment.verdict,
                        severity=assessment.severity,
                        confidence=assessment.confidence,
                        explanation=assessment.explanation,
                        impact_json=list(assessment.impact_json),
                        recommended_actions_json=list(assessment.recommended_actions_json),
                        assessment_manifest_hash=assessment.assessment_manifest_hash,
                    )
                )
            if assessment_rows:
                await consistency_check_repository.create_consistency_assessments(
                    write_session,
                    assessment_rows,
                )

            citation_rows: list[ConsistencyAssessmentCitation] = []
            for assessment in persistence_plan.assessments:
                assessment_id = assessment_id_by_key[
                    (
                        assessment.source_consistency_candidate_id,
                        assessment.batch_index,
                    )
                ]
                for citation in assessment.citations:
                    citation_rows.append(
                        ConsistencyAssessmentCitation(
                            id=uuid.uuid4(),
                            assessment_id=assessment_id,
                            source_consistency_application_id=citation.assessment_source_consistency_application_id,
                            source_consistency_candidate_id=citation.assessment_source_consistency_candidate_id,
                            source_fact_value_id=citation.source_fact_value_id,
                            evidence_link_id=citation.evidence_link_id,
                            citation_order=citation.citation_order,
                        )
                    )
            if citation_rows:
                await consistency_check_repository.create_consistency_assessment_citations(
                    write_session,
                    citation_rows,
                )

            await write_session.commit()
            return _build_result(
                application_id=application.id,
                persistence_plan=persistence_plan,
                created_new=True,
            )
        except IntegrityError as error:
            constraint_name = _get_integrity_constraint_name(error)
            await write_session.rollback()
            if constraint_name != _APPLICATION_UNIQUE_CONSTRAINT:
                raise
        except BaseException:
            await write_session.rollback()
            raise

    async with session_factory() as read_session:
        try:
            existing_result = await _read_existing_result(
                read_session,
                execution_identity_hash=persistence_plan.execution_identity_hash,
                persistence_plan=persistence_plan,
            )
        except BaseException:
            await read_session.rollback()
            raise
        else:
            await read_session.rollback()
    if existing_result is None:
        raise ConsistencyCheckPersistenceInvariantError(
            "consistency_check_persistence_concurrent_ledger_missing"
        )
    return existing_result
