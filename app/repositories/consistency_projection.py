from __future__ import annotations

from dataclasses import dataclass
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.consistency_check import ConsistencyCheckApplication
from app.models.document_content import DocumentBlock, ExtractionRun, SourceEvidence
from app.models.fact import FactEvidenceLink, FactValue
from app.models.fact_extraction_orchestration import FactExtractionOrchestrationBatch
from app.models.fact_value_duplicate_grouping import (
    FactValueConsistencyCandidate,
    FactValueConsistencyCandidateMember,
)
from app.repositories import consistency_check as consistency_check_repository
from app.schemas.consistency_check_persistence import (
    ConsistencyAssessmentCitationLedgerRecord,
    ConsistencyAssessmentLedgerRecord,
    ConsistencyCheckApplicationLedgerRecord,
    ConsistencyCheckBatchLedgerRecord,
)


@dataclass(frozen=True, slots=True)
class ConsistencyProjectionSourceRow:
    candidate_id: uuid.UUID
    consistency_application_id: uuid.UUID
    candidate_fact_id: uuid.UUID
    candidate_kind: str
    member_id: uuid.UUID
    member_candidate_id: uuid.UUID
    member_consistency_application_id: uuid.UUID
    member_orchestration_id: uuid.UUID
    member_fact_value_id: uuid.UUID
    member_source_batch_id: uuid.UUID
    member_semantic_key_hash: str
    fact_value_fact_id: uuid.UUID
    fact_value_value_type: str
    fact_value_value_json: object | None
    fact_value_normalized_value_text: str | None
    fact_value_referenced_entity_id: uuid.UUID | None
    batch_orchestration_id: uuid.UUID | None
    evidence_link_id: uuid.UUID | None
    evidence_link_fact_value_id: uuid.UUID | None
    evidence_id: uuid.UUID | None
    evidence_role: str | None
    evidence_is_primary: bool | None
    evidence_source_order: int | None
    document_revision_id: uuid.UUID | None
    block_id: uuid.UUID | None
    start_offset: int | None
    end_offset: int | None
    excerpt: str | None
    excerpt_hash: str | None
    location_key: str | None
    page_no: int | None
    start_line: int | None
    end_line: int | None


@dataclass(frozen=True, slots=True)
class ConsistencyProjectionSnapshot:
    application: ConsistencyCheckApplicationLedgerRecord
    batches: tuple[ConsistencyCheckBatchLedgerRecord, ...]
    assessments: tuple[ConsistencyAssessmentLedgerRecord, ...]
    citations: tuple[ConsistencyAssessmentCitationLedgerRecord, ...]
    source_rows: tuple[ConsistencyProjectionSourceRow, ...]


async def get_consistency_check_application_ledger_by_id(
    session: AsyncSession,
    *,
    consistency_check_application_id: uuid.UUID,
) -> ConsistencyCheckApplicationLedgerRecord | None:
    result = await session.execute(
        select(ConsistencyCheckApplication).where(
            ConsistencyCheckApplication.id == consistency_check_application_id
        )
    )
    application = result.scalar_one_or_none()
    if application is None:
        return None
    return ConsistencyCheckApplicationLedgerRecord(
        id=application.id,
        project_id=application.project_id,
        consistency_application_id=application.consistency_application_id,
        orchestration_id=application.orchestration_id,
        source_result_manifest_hash=application.source_result_manifest_hash,
        plan_manifest_hash=application.plan_manifest_hash,
        execution_identity_hash=application.execution_identity_hash,
        result_manifest_hash=application.result_manifest_hash,
        prompt_contract_hash=application.prompt_contract_hash,
        provider=application.provider,
        requested_model=application.requested_model,
        executor_name=application.executor_name,
        executor_version=application.executor_version,
        batch_count=application.batch_count,
        executed_batch_count=application.executed_batch_count,
        skipped_empty_batch_count=application.skipped_empty_batch_count,
        inference_run_count=application.inference_run_count,
        assessment_count=application.assessment_count,
        created_at=application.created_at,
    )


async def list_consistency_projection_source_rows(
    session: AsyncSession,
    *,
    source_consistency_application_id: uuid.UUID,
) -> tuple[ConsistencyProjectionSourceRow, ...]:
    result = await session.execute(
        select(
            FactValueConsistencyCandidate.id.label("candidate_id"),
            FactValueConsistencyCandidate.consistency_application_id.label("consistency_application_id"),
            FactValueConsistencyCandidate.fact_id.label("candidate_fact_id"),
            FactValueConsistencyCandidate.candidate_kind.label("candidate_kind"),
            FactValueConsistencyCandidateMember.id.label("member_id"),
            FactValueConsistencyCandidateMember.candidate_id.label("member_candidate_id"),
            FactValueConsistencyCandidateMember.consistency_application_id.label(
                "member_consistency_application_id"
            ),
            FactValueConsistencyCandidateMember.orchestration_id.label("member_orchestration_id"),
            FactValueConsistencyCandidateMember.fact_value_id.label("member_fact_value_id"),
            FactValueConsistencyCandidateMember.source_batch_id.label("member_source_batch_id"),
            FactValueConsistencyCandidateMember.semantic_key_hash.label("member_semantic_key_hash"),
            FactValue.fact_id.label("fact_value_fact_id"),
            FactValue.value_type.label("fact_value_value_type"),
            FactValue.value_json.label("fact_value_value_json"),
            FactValue.normalized_value_text.label("fact_value_normalized_value_text"),
            FactValue.referenced_entity_id.label("fact_value_referenced_entity_id"),
            FactExtractionOrchestrationBatch.orchestration_id.label("batch_orchestration_id"),
            FactEvidenceLink.id.label("evidence_link_id"),
            FactEvidenceLink.fact_value_id.label("evidence_link_fact_value_id"),
            FactEvidenceLink.evidence_id.label("evidence_id"),
            FactEvidenceLink.role.label("evidence_role"),
            FactEvidenceLink.is_primary.label("evidence_is_primary"),
            FactEvidenceLink.source_order.label("evidence_source_order"),
            ExtractionRun.revision_id.label("document_revision_id"),
            SourceEvidence.block_id.label("block_id"),
            SourceEvidence.start_offset.label("start_offset"),
            SourceEvidence.end_offset.label("end_offset"),
            SourceEvidence.excerpt.label("excerpt"),
            SourceEvidence.excerpt_hash.label("excerpt_hash"),
            DocumentBlock.location_key.label("location_key"),
            DocumentBlock.page_no.label("page_no"),
            DocumentBlock.start_line.label("start_line"),
            DocumentBlock.end_line.label("end_line"),
        )
        .select_from(FactValueConsistencyCandidate)
        .join(
            FactValueConsistencyCandidateMember,
            (
                FactValueConsistencyCandidateMember.candidate_id == FactValueConsistencyCandidate.id
            )
            & (
                FactValueConsistencyCandidateMember.consistency_application_id
                == FactValueConsistencyCandidate.consistency_application_id
            ),
        )
        .join(FactValue, FactValue.id == FactValueConsistencyCandidateMember.fact_value_id)
        .join(
            FactExtractionOrchestrationBatch,
            FactExtractionOrchestrationBatch.id == FactValueConsistencyCandidateMember.source_batch_id,
        )
        .outerjoin(FactEvidenceLink, FactEvidenceLink.fact_value_id == FactValue.id)
        .outerjoin(SourceEvidence, SourceEvidence.id == FactEvidenceLink.evidence_id)
        .outerjoin(DocumentBlock, DocumentBlock.id == SourceEvidence.block_id)
        .outerjoin(ExtractionRun, ExtractionRun.id == DocumentBlock.extraction_run_id)
        .where(FactValueConsistencyCandidate.consistency_application_id == source_consistency_application_id)
        .order_by(
            FactValueConsistencyCandidate.fact_id.asc(),
            FactValueConsistencyCandidate.id.asc(),
            FactValueConsistencyCandidateMember.semantic_key_hash.asc(),
            FactValueConsistencyCandidateMember.source_batch_id.asc(),
            FactValueConsistencyCandidateMember.fact_value_id.asc(),
            FactEvidenceLink.source_order.asc().nulls_last(),
            FactEvidenceLink.id.asc().nulls_last(),
        )
    )
    rows = list(result.all())
    return tuple(
        ConsistencyProjectionSourceRow(
            candidate_id=row.candidate_id,
            consistency_application_id=row.consistency_application_id,
            candidate_fact_id=row.candidate_fact_id,
            candidate_kind=row.candidate_kind,
            member_id=row.member_id,
            member_candidate_id=row.member_candidate_id,
            member_consistency_application_id=row.member_consistency_application_id,
            member_orchestration_id=row.member_orchestration_id,
            member_fact_value_id=row.member_fact_value_id,
            member_source_batch_id=row.member_source_batch_id,
            member_semantic_key_hash=row.member_semantic_key_hash,
            fact_value_fact_id=row.fact_value_fact_id,
            fact_value_value_type=row.fact_value_value_type,
            fact_value_value_json=row.fact_value_value_json,
            fact_value_normalized_value_text=row.fact_value_normalized_value_text,
            fact_value_referenced_entity_id=row.fact_value_referenced_entity_id,
            batch_orchestration_id=row.batch_orchestration_id,
            evidence_link_id=row.evidence_link_id,
            evidence_link_fact_value_id=row.evidence_link_fact_value_id,
            evidence_id=row.evidence_id,
            evidence_role=row.evidence_role,
            evidence_is_primary=row.evidence_is_primary,
            evidence_source_order=row.evidence_source_order,
            document_revision_id=row.document_revision_id,
            block_id=row.block_id,
            start_offset=row.start_offset,
            end_offset=row.end_offset,
            excerpt=row.excerpt,
            excerpt_hash=row.excerpt_hash,
            location_key=row.location_key,
            page_no=row.page_no,
            start_line=row.start_line,
            end_line=row.end_line,
        )
        for row in rows
    )


async def load_consistency_projection_snapshot(
    session: AsyncSession,
    *,
    consistency_check_application_id: uuid.UUID,
) -> ConsistencyProjectionSnapshot | None:
    application = await get_consistency_check_application_ledger_by_id(
        session,
        consistency_check_application_id=consistency_check_application_id,
    )
    if application is None:
        return None
    batches = await consistency_check_repository.list_consistency_check_batch_ledgers(
        session,
        consistency_check_application_id=application.id,
    )
    assessments = await consistency_check_repository.list_consistency_assessment_ledgers(
        session,
        consistency_check_application_id=application.id,
    )
    citations = await consistency_check_repository.list_consistency_assessment_citation_ledgers(
        session,
        consistency_check_application_id=application.id,
    )
    source_rows = await list_consistency_projection_source_rows(
        session,
        source_consistency_application_id=application.consistency_application_id,
    )
    return ConsistencyProjectionSnapshot(
        application=application,
        batches=batches,
        assessments=assessments,
        citations=citations,
        source_rows=source_rows,
    )
