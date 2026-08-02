from __future__ import annotations

from dataclasses import dataclass
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document_content import DocumentBlock, ExtractionRun, SourceEvidence
from app.models.fact import Fact, FactEvidenceLink, FactValue, FactValueSourceKind
from app.models.fact_extraction_application import FactExtractionBatchApplication
from app.models.fact_extraction_orchestration import (
    FactExtractionOrchestration,
    FactExtractionOrchestrationBatch,
)


@dataclass(frozen=True, slots=True)
class DocumentRevisionFactDiffSourceRow:
    fact_project_id: uuid.UUID
    fact_id: uuid.UUID
    fact_identity_hash: str
    subject_kind: str
    subject_key: str
    predicate_key: str
    scope_key: str | None
    subject_entity_id: uuid.UUID | None
    fact_value_id: uuid.UUID
    extraction_run_id: uuid.UUID
    inference_run_id: uuid.UUID
    source_batch_id: uuid.UUID
    application_project_id: uuid.UUID
    application_extraction_run_id: uuid.UUID
    application_inference_run_id: uuid.UUID
    orchestration_project_id: uuid.UUID
    orchestration_extraction_run_id: uuid.UUID
    batch_current_inference_run_id: uuid.UUID
    value_type: str
    value_json: object | None
    normalized_value_text: str | None
    fact_value_hash: str
    referenced_entity_id: uuid.UUID | None
    evidence_link_id: uuid.UUID | None
    evidence_link_source_order: int | None
    evidence_id: uuid.UUID | None
    document_block_id: uuid.UUID | None
    evidence_start_offset: int | None
    evidence_end_offset: int | None
    evidence_excerpt: str | None
    evidence_excerpt_hash: str | None
    block_extraction_run_id: uuid.UUID | None
    block_source_order: int | None
    block_location_key: str | None
    block_page_no: int | None
    block_start_line: int | None
    block_end_line: int | None
    block_table_index: int | None
    block_row_index: int | None
    block_raw_text: str | None
    application_id: uuid.UUID | None = None
    language_code: str | None = None
    confidence: float | None = None
    evidence_role: str | None = None
    evidence_is_primary: bool | None = None
    document_revision_id: uuid.UUID | None = None


async def list_document_revision_fact_diff_source_rows(
    session: AsyncSession,
    *,
    orchestration_id: uuid.UUID,
) -> tuple[DocumentRevisionFactDiffSourceRow, ...]:
    result = await session.execute(
        select(
            Fact.project_id.label("fact_project_id"),
            Fact.id.label("fact_id"),
            Fact.identity_hash.label("fact_identity_hash"),
            Fact.subject_kind.label("subject_kind"),
            Fact.subject_key.label("subject_key"),
            Fact.predicate_key.label("predicate_key"),
            Fact.scope_key.label("scope_key"),
            Fact.subject_entity_id.label("subject_entity_id"),
            FactValue.id.label("fact_value_id"),
            FactValue.extraction_run_id.label("extraction_run_id"),
            FactValue.inference_run_id.label("inference_run_id"),
            FactExtractionOrchestrationBatch.id.label("source_batch_id"),
            FactExtractionBatchApplication.id.label("application_id"),
            FactExtractionBatchApplication.project_id.label("application_project_id"),
            FactExtractionBatchApplication.extraction_run_id.label(
                "application_extraction_run_id"
            ),
            FactExtractionBatchApplication.inference_run_id.label(
                "application_inference_run_id"
            ),
            FactExtractionOrchestration.project_id.label("orchestration_project_id"),
            FactExtractionOrchestration.extraction_run_id.label(
                "orchestration_extraction_run_id"
            ),
            FactExtractionOrchestrationBatch.current_inference_run_id.label(
                "batch_current_inference_run_id"
            ),
            FactValue.value_type.label("value_type"),
            FactValue.value_json.label("value_json"),
            FactValue.normalized_value_text.label("normalized_value_text"),
            FactValue.value_hash.label("fact_value_hash"),
            FactValue.language_code.label("language_code"),
            FactValue.confidence.label("confidence"),
            FactValue.referenced_entity_id.label("referenced_entity_id"),
            FactEvidenceLink.id.label("evidence_link_id"),
            FactEvidenceLink.source_order.label("evidence_link_source_order"),
            SourceEvidence.id.label("evidence_id"),
            FactEvidenceLink.role.label("evidence_role"),
            FactEvidenceLink.is_primary.label("evidence_is_primary"),
            DocumentBlock.id.label("document_block_id"),
            ExtractionRun.revision_id.label("document_revision_id"),
            SourceEvidence.start_offset.label("evidence_start_offset"),
            SourceEvidence.end_offset.label("evidence_end_offset"),
            SourceEvidence.excerpt.label("evidence_excerpt"),
            SourceEvidence.excerpt_hash.label("evidence_excerpt_hash"),
            DocumentBlock.extraction_run_id.label("block_extraction_run_id"),
            DocumentBlock.source_order.label("block_source_order"),
            DocumentBlock.location_key.label("block_location_key"),
            DocumentBlock.page_no.label("block_page_no"),
            DocumentBlock.start_line.label("block_start_line"),
            DocumentBlock.end_line.label("block_end_line"),
            DocumentBlock.table_index.label("block_table_index"),
            DocumentBlock.row_index.label("block_row_index"),
            DocumentBlock.raw_text.label("block_raw_text"),
        )
        .select_from(FactExtractionOrchestrationBatch)
        .join(
            FactExtractionOrchestration,
            FactExtractionOrchestrationBatch.orchestration_id == FactExtractionOrchestration.id,
        )
        .join(
            FactExtractionBatchApplication,
            FactExtractionOrchestrationBatch.application_id == FactExtractionBatchApplication.id,
        )
        .join(
            FactValue,
            FactValue.inference_run_id == FactExtractionBatchApplication.inference_run_id,
        )
        .join(Fact, FactValue.fact_id == Fact.id)
        .outerjoin(FactEvidenceLink, FactEvidenceLink.fact_value_id == FactValue.id)
        .outerjoin(SourceEvidence, FactEvidenceLink.evidence_id == SourceEvidence.id)
        .outerjoin(DocumentBlock, SourceEvidence.block_id == DocumentBlock.id)
        .outerjoin(ExtractionRun, DocumentBlock.extraction_run_id == ExtractionRun.id)
        .where(
            FactExtractionOrchestrationBatch.orchestration_id == orchestration_id,
            FactExtractionOrchestrationBatch.status == "completed",
            FactExtractionBatchApplication.status == "completed",
            FactExtractionBatchApplication.project_id == FactExtractionOrchestration.project_id,
            FactExtractionBatchApplication.extraction_run_id
            == FactExtractionOrchestration.extraction_run_id,
            FactExtractionBatchApplication.inference_run_id
            == FactExtractionOrchestrationBatch.current_inference_run_id,
            FactValue.extraction_run_id == FactExtractionOrchestration.extraction_run_id,
            FactValue.source_kind == FactValueSourceKind.AI.value,
            Fact.project_id == FactExtractionOrchestration.project_id,
        )
        .order_by(
            FactValue.id.asc(),
            DocumentBlock.source_order.asc().nulls_last(),
            FactEvidenceLink.source_order.asc().nulls_last(),
            FactEvidenceLink.id.asc().nulls_last(),
        )
    )
    rows = result.all()
    return tuple(
        DocumentRevisionFactDiffSourceRow(
            fact_project_id=row.fact_project_id,
            fact_id=row.fact_id,
            fact_identity_hash=row.fact_identity_hash,
            subject_kind=row.subject_kind,
            subject_key=row.subject_key,
            predicate_key=row.predicate_key,
            scope_key=row.scope_key,
            subject_entity_id=row.subject_entity_id,
            fact_value_id=row.fact_value_id,
            extraction_run_id=row.extraction_run_id,
            inference_run_id=row.inference_run_id,
            source_batch_id=row.source_batch_id,
            application_id=row.application_id,
            application_project_id=row.application_project_id,
            application_extraction_run_id=row.application_extraction_run_id,
            application_inference_run_id=row.application_inference_run_id,
            orchestration_project_id=row.orchestration_project_id,
            orchestration_extraction_run_id=row.orchestration_extraction_run_id,
            batch_current_inference_run_id=row.batch_current_inference_run_id,
            value_type=row.value_type,
            value_json=row.value_json,
            normalized_value_text=row.normalized_value_text,
            fact_value_hash=row.fact_value_hash,
            language_code=row.language_code,
            confidence=row.confidence,
            referenced_entity_id=row.referenced_entity_id,
            evidence_link_id=row.evidence_link_id,
            evidence_link_source_order=row.evidence_link_source_order,
            evidence_id=row.evidence_id,
            evidence_role=row.evidence_role,
            evidence_is_primary=row.evidence_is_primary,
            document_block_id=row.document_block_id,
            document_revision_id=row.document_revision_id,
            evidence_start_offset=row.evidence_start_offset,
            evidence_end_offset=row.evidence_end_offset,
            evidence_excerpt=row.evidence_excerpt,
            evidence_excerpt_hash=row.evidence_excerpt_hash,
            block_extraction_run_id=row.block_extraction_run_id,
            block_source_order=row.block_source_order,
            block_location_key=row.block_location_key,
            block_page_no=row.block_page_no,
            block_start_line=row.block_start_line,
            block_end_line=row.block_end_line,
            block_table_index=row.block_table_index,
            block_row_index=row.block_row_index,
            block_raw_text=row.block_raw_text,
        )
        for row in rows
    )
