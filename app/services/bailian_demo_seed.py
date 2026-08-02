from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any
import uuid
from unittest.mock import patch

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agents.fact_extraction_planner import plan_fact_extraction_batches
from app.agents.prompt_registry import get_prompt
from app.core.config import get_settings
from app.models.document import Document, DocumentStatus
from app.models.document_content import DocumentBlock, ExtractionRun, SourceEvidence
from app.models.document_revision import (
    DocumentRevision,
    DocumentRevisionStatus,
    SourceAuthority,
    UploadIntent,
)
from app.models.dynamic_schema import DynamicSchema
from app.models.fact import Fact, FactEvidenceLink, FactValue
from app.models.fact_extraction_orchestration import FactExtractionOrchestration
from app.models.project import Project, ProjectStatus, ProjectVisibility
from app.models.project_member import ProjectMember, ProjectMemberRole
from app.models.project_version import ProjectVersion
from app.models.user import User, UserStatus
from app.schemas.agent_consistency_check import ConsistencyCheckResponse
from app.schemas.document_extraction import ExtractedBlock, ExtractedBlockType, ExtractedDocument, ExtractionOutcome
from app.schemas.dynamic_schema import (
    DynamicSchemaFieldInput,
    DynamicSchemaIdentityInput,
    DynamicSchemaVersionInput,
)
from app.schemas.dynamic_schema_commands import HumanSchemaDraftInput
from app.schemas.fact_extraction_plan import FactExtractionPlannerConfig
from app.schemas.bailian_demo_seed import BailianDemoSeedResult
from app.schemas.fact import FactValueInput
from app.schemas.fact_commands import AIProposalInput, FactEvidenceInput
from app.services import bailian_review_tools
from app.services import consistency_check as consistency_check_service
from app.services import consistency_check_execution
from app.services import consistency_check_persistence
from app.services import consistency_review
from app.services import document_content as document_content_service
from app.services import dynamic_schema as dynamic_schema_service
from app.services import dynamic_schema_knowledge_view
from app.services import dynamic_schema_review_projection
from app.services import fact as fact_service
from app.services import fact_extraction_orchestration
from app.services import fact_value_duplicate_grouping
from app.services import project_version as project_version_service
from app.services.llm import MockLLMClient
from app.utils.document_blocks import build_document_block_anchor_hash


_SEED_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "zhiwen:bailian-demo-seed")
_PRIMARY_DOCUMENT_TITLE = "织文产品说明（演示）"
_SECONDARY_DOCUMENT_TITLE = "织文评审备忘（演示）"
_PROJECT_NAME = "织文百炼联调演示"
_PROJECT_SLUG = "zhiwen-bailian-demo"
_SUBJECT_KIND = "product"
_SUBJECT_KEY = "织文/企业知识库"
_SCHEMA_KEY = "zhiwen-bailian-demo-schema"
_SCHEMA_NAME = "织文百炼联调演示 Schema"
_SEED_HANDLE = "bailian_demo"
_SEED_EMAIL = "bailian-demo@example.com"
_SCHEMA_SUMMARY = "百炼联调演示知识视图"
_VERSION_REASON = "百炼联调演示版本"
_EXTRACTOR_NAME = "bailian_demo_seed"
_EXTRACTOR_VERSION = "1.0.0"
_PROVIDER = "mock"
_REQUESTED_MODEL = "mock-local"


class BailianDemoSeedError(Exception):
    """Base class for bailian demo seed failures."""


class BailianDemoSeedStateError(BailianDemoSeedError):
    """Raised when the local environment forbids seeding."""


class BailianDemoSeedInconsistentError(BailianDemoSeedError):
    """Raised when existing seed data is partial or drifted."""


def _seed_uuid(seed_id: str, name: str) -> uuid.UUID:
    return uuid.uuid5(_SEED_NAMESPACE, f"{seed_id}:{name}")


def _seed_phase(seed_id: str, phase: str) -> str:
    return f"{seed_id}:{phase}"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_string(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class _DeterministicUUID4:
    def __init__(self, seed_id: str) -> None:
        self._namespace = _seed_uuid(seed_id, "uuid4")
        self._counts: dict[str, int] = {}

    def __call__(self) -> uuid.UUID:
        frame = inspect.stack(context=0)[1]
        location = f"{Path(frame.filename).name}:{frame.function}:{frame.lineno}"
        index = self._counts.get(location, 0)
        self._counts[location] = index + 1
        return uuid.uuid5(self._namespace, f"{location}:{index}")


@contextmanager
def _patch_deterministic_uuid4(seed_id: str):
    generator = _DeterministicUUID4(seed_id)
    with patch("uuid.uuid4", generator):
        yield


def _build_primary_block_texts() -> tuple[str, ...]:
    return (
        "织文正式发布日期计划为2026-08-08。生产部署区域为华北2（北京）。支持渠道为在线工单。",
        "当前版本用于百炼插件联调演示，内容均为虚构样例。",
        "另一份内部评审记录写到发布日期暂定2026-08-15。",
        "灰度部署记录也把生产部署区域写作华东1（杭州）。",
    )


def _evidence_offsets(text: str, excerpt: str) -> tuple[int, int]:
    start = text.index(excerpt)
    return start, start + len(excerpt)


def _build_extracted_document() -> ExtractedDocument:
    block_texts = _build_primary_block_texts()
    blocks = [
        ExtractedBlock(
            source_order=index,
            block_type=ExtractedBlockType.PARAGRAPH,
            raw_text=text,
            normalized_text=text,
            location_key=f"demo:{index + 1}",
            anchor_hash=build_document_block_anchor_hash(
                detected_format="md",
                location_key=f"demo:{index + 1}",
                raw_text=text,
            ),
            page_no=1,
            block_index=index,
            heading_path=[],
            start_line=index + 1,
            end_line=index + 1,
        )
        for index, text in enumerate(block_texts)
    ]
    return ExtractedDocument(
        outcome=ExtractionOutcome.SUCCESS,
        detected_format="md",
        detected_encoding="utf-8",
        page_count=1,
        character_count=sum(len(text) for text in block_texts),
        block_count=len(blocks),
        blocks=blocks,
        warnings=[],
        metadata={"seed": "bailian-demo"},
    )


def _build_fact_extraction_responses() -> list[str]:
    block0, _block1, block2, block3 = _build_primary_block_texts()
    release_start_1, release_end_1 = _evidence_offsets(block0, "2026-08-08")
    region_start_1, region_end_1 = _evidence_offsets(block0, "华北2（北京）")
    support_start, support_end = _evidence_offsets(block0, "在线工单")
    release_start_2, release_end_2 = _evidence_offsets(block2, "2026-08-15")
    region_start_2, region_end_2 = _evidence_offsets(block3, "华东1（杭州）")

    batch_one = {
        "facts": [
            {
                "subject_kind": _SUBJECT_KIND,
                "subject_key": _SUBJECT_KEY,
                "predicate_key": "release_date",
                "scope_key": "official",
                "value_type": "date",
                "value_json": "2026-08-08",
                "confidence": 0.95,
                "evidence": [
                    {
                        "block_ref": "B0001",
                        "start_offset": release_start_1,
                        "end_offset": release_end_1,
                        "role": "supporting",
                    }
                ],
            },
            {
                "subject_kind": _SUBJECT_KIND,
                "subject_key": _SUBJECT_KEY,
                "predicate_key": "deployment_region",
                "scope_key": "production",
                "value_type": "string",
                "value_json": "华北2（北京）",
                "confidence": 0.94,
                "evidence": [
                    {
                        "block_ref": "B0001",
                        "start_offset": region_start_1,
                        "end_offset": region_end_1,
                        "role": "supporting",
                    }
                ],
            },
            {
                "subject_kind": _SUBJECT_KIND,
                "subject_key": _SUBJECT_KEY,
                "predicate_key": "support_channel",
                "scope_key": None,
                "value_type": "string",
                "value_json": "在线工单",
                "confidence": 0.92,
                "evidence": [
                    {
                        "block_ref": "B0001",
                        "start_offset": support_start,
                        "end_offset": support_end,
                        "role": "supporting",
                    }
                ],
            },
        ],
        "batch_summary": "demo batch one",
        "uncertainties": [],
    }
    batch_two = {
        "facts": [
            {
                "subject_kind": _SUBJECT_KIND,
                "subject_key": _SUBJECT_KEY,
                "predicate_key": "release_date",
                "scope_key": "official",
                "value_type": "date",
                "value_json": "2026-08-15",
                "confidence": 0.88,
                "evidence": [
                    {
                        "block_ref": "B0001",
                        "start_offset": release_start_2,
                        "end_offset": release_end_2,
                        "role": "supporting",
                    }
                ],
            },
            {
                "subject_kind": _SUBJECT_KIND,
                "subject_key": _SUBJECT_KEY,
                "predicate_key": "deployment_region",
                "scope_key": "production",
                "value_type": "string",
                "value_json": "华东1（杭州）",
                "confidence": 0.87,
                "evidence": [
                    {
                        "block_ref": "B0002",
                        "start_offset": region_start_2,
                        "end_offset": region_end_2,
                        "role": "supporting",
                    }
                ],
            },
        ],
        "batch_summary": "demo batch two",
        "uncertainties": [],
    }
    return [
        _json_string(batch_one),
        _json_string(batch_two),
    ]


def _build_human_schema_input() -> HumanSchemaDraftInput:
    return HumanSchemaDraftInput(
        identity=DynamicSchemaIdentityInput(
            schema_key=_SCHEMA_KEY,
            name=_SCHEMA_NAME,
            subject_kind=_SUBJECT_KIND,
            description="百炼联调演示动态 Schema",
        ),
        version=DynamicSchemaVersionInput(
            summary=_SCHEMA_SUMMARY,
            layout_config={"seed": "bailian-demo"},
            fields=[
                DynamicSchemaFieldInput(
                    field_key="release_date",
                    label="发布日期",
                    predicate_key="release_date",
                    scope_key="official",
                    expected_value_type="date",
                    cardinality="one",
                    is_required=True,
                    display_order=0,
                ),
                DynamicSchemaFieldInput(
                    field_key="deployment_region",
                    label="部署区域",
                    predicate_key="deployment_region",
                    scope_key="production",
                    expected_value_type="string",
                    cardinality="one",
                    display_order=1,
                ),
                DynamicSchemaFieldInput(
                    field_key="support_channel",
                    label="支持渠道",
                    predicate_key="support_channel",
                    scope_key=None,
                    expected_value_type="string",
                    cardinality="one",
                    display_order=2,
                ),
            ],
        ),
    )


async def _get_project_by_slug(session: AsyncSession) -> Project | None:
    result = await session.execute(
        select(Project).where(Project.slug == _PROJECT_SLUG)
    )
    return result.scalar_one_or_none()


async def _get_user_by_handle(session: AsyncSession) -> User | None:
    result = await session.execute(select(User).where(User.handle == _SEED_HANDLE))
    return result.scalar_one_or_none()


async def _get_schema(session: AsyncSession, project_id: uuid.UUID) -> DynamicSchema | None:
    result = await session.execute(
        select(DynamicSchema).where(
            DynamicSchema.project_id == project_id,
            DynamicSchema.schema_key == _SCHEMA_KEY,
        )
    )
    return result.scalar_one_or_none()


async def _get_document_by_title(session: AsyncSession, project_id: uuid.UUID, title: str) -> Document | None:
    result = await session.execute(
        select(Document).where(
            Document.project_id == project_id,
            Document.title == title,
        )
    )
    return result.scalar_one_or_none()


async def _get_single_extraction_run(session: AsyncSession, revision_id: uuid.UUID) -> ExtractionRun | None:
    result = await session.execute(
        select(ExtractionRun)
        .where(ExtractionRun.revision_id == revision_id)
        .order_by(ExtractionRun.attempt_no.asc())
    )
    rows = list(result.scalars().all())
    if not rows:
        return None
    if len(rows) != 1:
        raise BailianDemoSeedInconsistentError("bailian_demo_seed_inconsistent")
    return rows[0]


async def _get_blocks_for_run(session: AsyncSession, extraction_run_id: uuid.UUID) -> list[DocumentBlock]:
    result = await session.execute(
        select(DocumentBlock)
        .where(DocumentBlock.extraction_run_id == extraction_run_id)
        .order_by(DocumentBlock.source_order.asc())
    )
    return list(result.scalars().all())


async def _create_base_objects(session_factory, seed_id: str) -> None:
    async with session_factory() as session:
        user = User(
            id=_seed_uuid(seed_id, "user"),
            handle=_SEED_HANDLE,
            display_name="百炼联调演示用户",
            email=_SEED_EMAIL,
            status=UserStatus.ACTIVE.value,
        )
        project = Project(
            id=_seed_uuid(seed_id, "project"),
            name=_PROJECT_NAME,
            slug=_PROJECT_SLUG,
            description="用于百炼插件联调的本地演示项目",
            visibility=ProjectVisibility.PRIVATE.value,
            status=ProjectStatus.ACTIVE.value,
            current_version_id=None,
            created_by_id=user.id,
        )
        member = ProjectMember(
            id=_seed_uuid(seed_id, "project-member"),
            project_id=project.id,
            user_id=user.id,
            role=ProjectMemberRole.OWNER.value,
        )
        primary_document = Document(
            id=_seed_uuid(seed_id, "document-primary"),
            project_id=project.id,
            title=_PRIMARY_DOCUMENT_TITLE,
            description="主演示来源文档",
            status=DocumentStatus.ACTIVE.value,
            created_by_id=user.id,
            current_revision_id=None,
            logical_order_kind="manual",
            logical_order_value="1",
            logical_order_index=Decimal("1"),
        )
        secondary_document = Document(
            id=_seed_uuid(seed_id, "document-secondary"),
            project_id=project.id,
            title=_SECONDARY_DOCUMENT_TITLE,
            description="附加演示文档",
            status=DocumentStatus.ACTIVE.value,
            created_by_id=user.id,
            current_revision_id=None,
            logical_order_kind="manual",
            logical_order_value="2",
            logical_order_index=Decimal("2"),
        )
        primary_content = "\n".join(_build_primary_block_texts())
        secondary_content = "这是一份额外的演示备忘，仅用于本地联调对象完整性。"
        primary_revision = DocumentRevision(
            id=_seed_uuid(seed_id, "revision-primary"),
            document_id=primary_document.id,
            revision_no=1,
            upload_intent=UploadIntent.NEW_DOCUMENT.value,
            supersedes_revision_id=None,
            original_filename="zhiwen-demo-primary.md",
            storage_key="demo/zhiwen-demo-primary.md",
            mime_type="text/markdown",
            file_size_bytes=len(primary_content.encode("utf-8")),
            sha256=_sha256_text(primary_content),
            detected_language="zh-CN",
            language_confidence=1.0,
            source_authority=SourceAuthority.FORMAL.value,
            status=DocumentRevisionStatus.COMPLETED.value,
            uploaded_by_id=user.id,
        )
        secondary_revision = DocumentRevision(
            id=_seed_uuid(seed_id, "revision-secondary"),
            document_id=secondary_document.id,
            revision_no=1,
            upload_intent=UploadIntent.NEW_DOCUMENT.value,
            supersedes_revision_id=None,
            original_filename="zhiwen-demo-secondary.md",
            storage_key="demo/zhiwen-demo-secondary.md",
            mime_type="text/markdown",
            file_size_bytes=len(secondary_content.encode("utf-8")),
            sha256=_sha256_text(secondary_content),
            detected_language="zh-CN",
            language_confidence=1.0,
            source_authority=SourceAuthority.REFERENCE.value,
            status=DocumentRevisionStatus.COMPLETED.value,
            uploaded_by_id=user.id,
        )
        session.add_all([user, project, member, primary_document, secondary_document])
        await session.flush()
        session.add_all([primary_revision, secondary_revision])
        await session.flush()
        primary_document.current_revision_id = primary_revision.id
        secondary_document.current_revision_id = secondary_revision.id
        await session.commit()


async def _persist_primary_extraction(session_factory, seed_id: str) -> ExtractionRun:
    async with session_factory() as session:
        revision = await session.get(DocumentRevision, _seed_uuid(seed_id, "revision-primary"))
        if revision is None:
            raise BailianDemoSeedInconsistentError("bailian_demo_seed_inconsistent")
        with _patch_deterministic_uuid4(_seed_phase(seed_id, "primary-extraction")):
            run = await document_content_service.persist_extraction_result(
                session,
                revision_id=revision.id,
                extracted_document=_build_extracted_document(),
                extractor_name=_EXTRACTOR_NAME,
                extractor_version=_EXTRACTOR_VERSION,
                started_at=datetime(2026, 8, 3, 8, 0, tzinfo=timezone.utc),
                completed_at=datetime(2026, 8, 3, 8, 1, tzinfo=timezone.utc),
            )
        return run


async def _load_primary_extraction(session_factory, seed_id: str) -> tuple[Project, ExtractionRun, list[DocumentBlock]]:
    async with session_factory() as session:
        project = await session.get(Project, _seed_uuid(seed_id, "project"))
        revision = await session.get(DocumentRevision, _seed_uuid(seed_id, "revision-primary"))
        if project is None or revision is None:
            raise BailianDemoSeedInconsistentError("bailian_demo_seed_inconsistent")
        extraction_run = await _get_single_extraction_run(session, revision.id)
        if extraction_run is None:
            raise BailianDemoSeedInconsistentError("bailian_demo_seed_inconsistent")
        blocks = await _get_blocks_for_run(session, extraction_run.id)
        if len(blocks) != 4:
            raise BailianDemoSeedInconsistentError("bailian_demo_seed_inconsistent")
        return project, extraction_run, blocks


async def _run_fact_pipeline(session_factory, seed_id: str) -> tuple[uuid.UUID, uuid.UUID]:
    project, extraction_run, blocks = await _load_primary_extraction(session_factory, seed_id)
    fact_prompt = get_prompt("agent1_fact_extraction", "1.0.0")
    plan = plan_fact_extraction_batches(
        extraction_run_id=extraction_run.id,
        blocks=blocks,
        prompt=fact_prompt,
        config=FactExtractionPlannerConfig(
            target_message_characters=5000,
            max_message_characters=6000,
            max_blocks_per_batch=2,
            overlap_block_count=0,
            include_preceding_heading=False,
        ),
    )
    if len(plan.batches) != 2:
        raise BailianDemoSeedError("expected exactly two extraction batches")

    with _patch_deterministic_uuid4(_seed_phase(seed_id, "fact-orchestration")):
        orchestration = await fact_extraction_orchestration.execute_fact_extraction_orchestration(
            session_factory,
            project_id=project.id,
            extraction_run_id=extraction_run.id,
            plan=plan,
            prompt=fact_prompt,
            llm_client=MockLLMClient(_build_fact_extraction_responses()),
            provider=_PROVIDER,
            requested_model=_REQUESTED_MODEL,
            worker_token=_seed_uuid(seed_id, "worker-token"),
        )
        duplicate_result = await fact_value_duplicate_grouping.ensure_cross_batch_duplicate_grouping(
            session_factory,
            orchestration_id=orchestration.orchestration_id,
        )
        candidate_result = await fact_value_duplicate_grouping.ensure_cross_batch_multi_value_consistency_candidates(
            session_factory,
            duplicate_grouping_application_id=duplicate_result.grouping_application_id,
        )

    consistency_prompt = get_prompt("agent2_consistency_check", "1.0.0")
    plan_result = await consistency_check_service.build_consistency_check_plan(
        session_factory,
        consistency_application_id=candidate_result.consistency_application_id,
        config=consistency_check_service.ConsistencyCheckPlannerConfig(
            max_candidates_per_batch=10,
            max_evidence_characters_per_batch=5000,
        ),
    )
    if len(plan_result.batches) != 1:
        raise BailianDemoSeedError("expected exactly one consistency batch")
    batch = plan_result.batches[0]
    response = {
        "assessments": [
            {
                "candidate_id": str(candidate.candidate_id),
                "verdict": "conflict",
                "severity": "yellow",
                "confidence": 0.92,
                "explanation": "演示数据中存在互斥候选值，需要人工确认。",
                "cited_evidence_link_ids": [
                    str(candidate.members[0].evidences[0].evidence_link_id),
                    str(candidate.members[-1].evidences[0].evidence_link_id),
                ],
                "impact": ["data_quality_review"],
                "recommended_actions": ["escalate_human_review"],
            }
            for candidate in batch.candidates
        ]
    }
    with _patch_deterministic_uuid4(_seed_phase(seed_id, "consistency-check")):
        execution = await consistency_check_execution.execute_consistency_check_plan(
            session_factory,
            project_id=plan_result.project_id,
            plan=plan_result,
            prompt=consistency_prompt,
            llm_client=MockLLMClient([_json_string(response)]),
            provider=_PROVIDER,
            requested_model=_REQUESTED_MODEL,
        )
        persisted = await consistency_check_persistence.persist_consistency_check_plan_result(
            session_factory,
            plan=plan_result,
            execution_result=execution,
            prompt=consistency_prompt,
            provider=_PROVIDER,
            requested_model=_REQUESTED_MODEL,
        )

    auth = await consistency_check_persistence.authenticate_persisted_consistency_check_application(
        session_factory,
        project_id=project.id,
        consistency_check_application_id=persisted.consistency_check_application_id,
    )
    async with session_factory() as session:
        facts = list(
            (
                await session.execute(
                    select(Fact)
                    .where(Fact.project_id == project.id)
                    .order_by(Fact.predicate_key.asc())
                )
            )
            .scalars()
            .all()
        )
        fact_by_predicate = {fact.predicate_key: fact for fact in facts}
        deployment_fact = fact_by_predicate["deployment_region"]
        release_fact = fact_by_predicate["release_date"]
        deployment_values = list(
            (
                await session.execute(
                    select(FactValue)
                    .where(FactValue.fact_id == deployment_fact.id)
                    .order_by(FactValue.version_no.asc())
                )
            )
            .scalars()
            .all()
        )
        chosen = next(
            value.id for value in deployment_values if value.normalized_value_text == "华北2（北京）"
        )

    assessment_id: uuid.UUID | None = None
    for assessment in auth.assessments:
        if assessment.source_consistency_candidate_id in {
            candidate.candidate_id
            for candidate in auth.candidate_bundles
            if candidate.fact_id == deployment_fact.id
        }:
            assessment_id = assessment.id
            break
    if assessment_id is None:
        raise BailianDemoSeedInconsistentError("bailian_demo_seed_inconsistent")

    with _patch_deterministic_uuid4(_seed_phase(seed_id, "consistency-review")):
        await consistency_review.append_consistency_review_decision(
            session_factory,
            project_id=project.id,
            consistency_check_application_id=persisted.consistency_check_application_id,
            assessment_id=assessment_id,
            actor_id=_seed_uuid(seed_id, "user"),
            expected_current_decision_id=None,
            decision_kind="select_one",
            selected_fact_value_ids=(chosen,),
            comment="演示场景中人工确认华北2（北京）为生效值。",
        )

    return orchestration.orchestration_id, persisted.consistency_check_application_id


async def _create_schema(session_factory, seed_id: str) -> tuple[uuid.UUID, uuid.UUID]:
    async with session_factory() as session:
        with _patch_deterministic_uuid4(_seed_phase(seed_id, "schema-draft")):
            version = await dynamic_schema_service.create_human_schema_draft(
                session,
                project_id=_seed_uuid(seed_id, "project"),
                actor_id=_seed_uuid(seed_id, "user"),
                payload=_build_human_schema_input(),
            )
    async with session_factory() as session:
        with _patch_deterministic_uuid4(_seed_phase(seed_id, "schema-activate")):
            active = await dynamic_schema_service.activate_dynamic_schema_version(
                session,
                project_id=_seed_uuid(seed_id, "project"),
                actor_id=_seed_uuid(seed_id, "user"),
                schema_id=version.schema_id,
                version_id=version.id,
            )
    return active.schema_id, active.id


async def _create_project_version(session_factory, seed_id: str, schema_id: uuid.UUID, schema_version_id: uuid.UUID) -> uuid.UUID:
    async with session_factory() as session:
        project = await session.get(Project, _seed_uuid(seed_id, "project"))
        if project is None:
            raise BailianDemoSeedInconsistentError("bailian_demo_seed_inconsistent")
        result = await session.execute(
            select(FactExtractionOrchestration)
            .where(FactExtractionOrchestration.project_id == project.id)
            .order_by(FactExtractionOrchestration.attempt_no.asc())
        )
        orchestrations = list(result.scalars().all())
        if len(orchestrations) != 1:
            raise BailianDemoSeedInconsistentError("bailian_demo_seed_inconsistent")
        orchestration_id = orchestrations[0].id
    review_projection = await dynamic_schema_review_projection.project_reviewed_orchestration_ufl_to_dynamic_schema(
        session_factory,
        project_id=_seed_uuid(seed_id, "project"),
        schema_id=schema_id,
        schema_version_id=schema_version_id,
        orchestration_id=orchestration_id,
        consistency_check_application_id=_lookup_consistency_check_application_id(session_factory, seed_id),
    )
    dynamic_schema_review_projection.authenticate_dynamic_schema_review_projection(
        review_projection,
        subject_keys=None,
    )
    knowledge_view = await dynamic_schema_knowledge_view.build_dynamic_schema_knowledge_view(
        session_factory,
        project_id=_seed_uuid(seed_id, "project"),
        schema_id=schema_id,
        schema_version_id=schema_version_id,
        orchestration_id=orchestration_id,
        consistency_check_application_id=_lookup_consistency_check_application_id(session_factory, seed_id),
        subject_keys=None,
    )
    dynamic_schema_knowledge_view.authenticate_dynamic_schema_knowledge_view(
        knowledge_view,
        subject_keys=None,
    )
    created = await project_version_service.create_project_version(
        session_factory,
        project_version_id=_seed_uuid(seed_id, "project-version"),
        project_id=_seed_uuid(seed_id, "project"),
        schema_id=schema_id,
        schema_version_id=schema_version_id,
        orchestration_id=orchestration_id,
        consistency_check_application_id=_lookup_consistency_check_application_id(session_factory, seed_id),
        created_by_id=_seed_uuid(seed_id, "user"),
        creation_kind="manual",
        reason=_VERSION_REASON,
    )
    return created.id


def _lookup_consistency_check_application_id(session_factory, seed_id: str) -> uuid.UUID:
    # This helper is only called after the consistency ledger has been created and
    # its ID is stable in the database.
    return _seed_uuid(seed_id, "consistency-check-app-placeholder")


async def _load_consistency_check_application_id(session_factory, seed_id: str) -> uuid.UUID:
    async with session_factory() as session:
        project = await session.get(Project, _seed_uuid(seed_id, "project"))
        if project is None:
            raise BailianDemoSeedInconsistentError("bailian_demo_seed_inconsistent")
        schema = await _get_schema(session, project.id)
        if schema is None or project.current_version_id is None:
            result = await session.execute(
                select(ProjectVersion)
                .where(ProjectVersion.project_id == project.id)
                .order_by(ProjectVersion.version_no.asc())
            )
            rows = list(result.scalars().all())
            if rows:
                return rows[-1].consistency_check_application_id
        result = await session.execute(
            select(ProjectVersion)
            .where(ProjectVersion.project_id == project.id)
            .order_by(ProjectVersion.version_no.asc())
        )
        rows = list(result.scalars().all())
        if rows:
            return rows[-1].consistency_check_application_id
    raise BailianDemoSeedInconsistentError("bailian_demo_seed_inconsistent")


async def _verify_and_build_result(session_factory, seed_id: str, *, created_new: bool) -> BailianDemoSeedResult:
    async with session_factory() as session:
        user = await _get_user_by_handle(session)
        project = await _get_project_by_slug(session)
        if user is None or project is None:
            raise BailianDemoSeedInconsistentError("bailian_demo_seed_inconsistent")
        schema = await _get_schema(session, project.id)
        if schema is None or schema.current_version_id is None:
            raise BailianDemoSeedInconsistentError("bailian_demo_seed_inconsistent")
        primary_document = await _get_document_by_title(session, project.id, _PRIMARY_DOCUMENT_TITLE)
        if primary_document is None or primary_document.current_revision_id is None:
            raise BailianDemoSeedInconsistentError("bailian_demo_seed_inconsistent")
        extraction_run = await _get_single_extraction_run(session, primary_document.current_revision_id)
        if extraction_run is None:
            raise BailianDemoSeedInconsistentError("bailian_demo_seed_inconsistent")
        result = await session.execute(
            select(FactExtractionOrchestration)
            .where(FactExtractionOrchestration.project_id == project.id)
            .order_by(FactExtractionOrchestration.attempt_no.asc())
        )
        orchestrations = list(result.scalars().all())
        if len(orchestrations) != 1:
            raise BailianDemoSeedInconsistentError("bailian_demo_seed_inconsistent")
        orchestration = orchestrations[0]
        version = await session.get(ProjectVersion, project.current_version_id)
        if version is None:
            raise BailianDemoSeedInconsistentError("bailian_demo_seed_inconsistent")

    reviewed = await dynamic_schema_review_projection.project_reviewed_orchestration_ufl_to_dynamic_schema(
        session_factory,
        project_id=project.id,
        schema_id=schema.id,
        schema_version_id=schema.current_version_id,
        orchestration_id=orchestration.id,
        consistency_check_application_id=version.consistency_check_application_id,
    )
    reviewed = dynamic_schema_review_projection.authenticate_dynamic_schema_review_projection(
        reviewed,
        subject_keys=None,
    )
    knowledge_view = await dynamic_schema_knowledge_view.build_dynamic_schema_knowledge_view(
        session_factory,
        project_id=project.id,
        schema_id=schema.id,
        schema_version_id=schema.current_version_id,
        orchestration_id=orchestration.id,
        consistency_check_application_id=version.consistency_check_application_id,
        subject_keys=None,
    )
    knowledge_view = dynamic_schema_knowledge_view.authenticate_dynamic_schema_knowledge_view(
        knowledge_view,
        subject_keys=None,
    )
    snapshot = await project_version_service.get_project_version_snapshot(
        session_factory,
        project_id=project.id,
        project_version_id=version.id,
    )
    snapshot = project_version_service.authenticate_project_version_snapshot(snapshot)

    facts_by_predicate: dict[str, Any] = {}
    for record in reviewed.records:
        for field in record.fields:
            for fact in field.reviewed_facts:
                facts_by_predicate.setdefault(fact.fact.predicate_key, fact)

    pending_fact = facts_by_predicate.get("release_date")
    resolved_fact = facts_by_predicate.get("deployment_region")
    observation_fact = facts_by_predicate.get("support_channel")
    if pending_fact is None or resolved_fact is None or observation_fact is None:
        raise BailianDemoSeedInconsistentError("bailian_demo_seed_inconsistent")
    if pending_fact.review_state != "pending_review":
        raise BailianDemoSeedInconsistentError("bailian_demo_seed_inconsistent")
    if resolved_fact.review_state != "resolved" or resolved_fact.resolution_basis != "human_selection":
        raise BailianDemoSeedInconsistentError("bailian_demo_seed_inconsistent")
    if observation_fact.review_state != "no_consistency_candidate":
        raise BailianDemoSeedInconsistentError("bailian_demo_seed_inconsistent")

    review_required = await bailian_review_tools.list_review_items(
        session_factory,
        project_id=str(project.id),
        schema_id=str(schema.id),
        schema_version_id=str(schema.current_version_id),
        orchestration_id=str(orchestration.id),
        consistency_check_application_id=str(version.consistency_check_application_id),
        state="review_required",
        limit=20,
    )
    resolved = await bailian_review_tools.list_review_items(
        session_factory,
        project_id=str(project.id),
        schema_id=str(schema.id),
        schema_version_id=str(schema.current_version_id),
        orchestration_id=str(orchestration.id),
        consistency_check_application_id=str(version.consistency_check_application_id),
        state="resolved",
        limit=20,
    )
    observation = await bailian_review_tools.list_review_items(
        session_factory,
        project_id=str(project.id),
        schema_id=str(schema.id),
        schema_version_id=str(schema.current_version_id),
        orchestration_id=str(orchestration.id),
        consistency_check_application_id=str(version.consistency_check_application_id),
        state="observation_only",
        limit=20,
    )
    if pending_fact.fact.fact_id not in {item.fact_id for item in review_required.items}:
        raise BailianDemoSeedInconsistentError("bailian_demo_seed_inconsistent")
    if resolved_fact.fact.fact_id not in {item.fact_id for item in resolved.items}:
        raise BailianDemoSeedInconsistentError("bailian_demo_seed_inconsistent")
    if observation_fact.fact.fact_id not in {item.fact_id for item in observation.items}:
        raise BailianDemoSeedInconsistentError("bailian_demo_seed_inconsistent")

    detail = await bailian_review_tools.get_review_item_detail(
        session_factory,
        project_id=str(project.id),
        fact_id=str(pending_fact.fact.fact_id),
        schema_id=str(schema.id),
        schema_version_id=str(schema.current_version_id),
        orchestration_id=str(orchestration.id),
        consistency_check_application_id=str(version.consistency_check_application_id),
    )
    if detail.semantic_value_count != 2 or len(detail.value_groups) != 2:
        raise BailianDemoSeedInconsistentError("bailian_demo_seed_inconsistent")

    record = await bailian_review_tools.get_version_record(
        session_factory,
        project_id=str(project.id),
        project_version_id=str(version.id),
        subject_key=_SUBJECT_KEY,
    )
    if record.subject_key != _SUBJECT_KEY:
        raise BailianDemoSeedInconsistentError("bailian_demo_seed_inconsistent")

    return BailianDemoSeedResult(
        seed_id=seed_id,
        created_new=created_new,
        user_id=user.id,
        project_id=project.id,
        schema_id=schema.id,
        schema_version_id=schema.current_version_id,
        orchestration_id=orchestration.id,
        extraction_run_id=extraction_run.id,
        consistency_check_application_id=version.consistency_check_application_id,
        project_version_id=version.id,
        pending_review_fact_id=pending_fact.fact.fact_id,
        resolved_fact_id=resolved_fact.fact.fact_id,
        observation_only_fact_id=observation_fact.fact.fact_id,
        pending_review_subject_key=pending_fact.fact.subject_key,
        version_record_subject_key=record.subject_key,
        reviewed_projection_manifest_hash=reviewed.reviewed_projection_manifest_hash,
        knowledge_view_manifest_hash=knowledge_view.knowledge_view_manifest_hash,
        project_version_manifest_hash=snapshot.version_manifest_hash,
    )


async def seed_bailian_demo(
    session_factory,
    *,
    seed_id: str = "bailian-demo-v1",
) -> BailianDemoSeedResult:
    if get_settings().is_production:
        raise BailianDemoSeedStateError("bailian_demo_seed_production_forbidden")

    async with session_factory() as session:
        has_any_seed_rows = (
            await _get_project_by_slug(session) is not None
            or await _get_user_by_handle(session) is not None
        )

    if has_any_seed_rows:
        return await _verify_and_build_result(session_factory, seed_id, created_new=False)

    await _create_base_objects(session_factory, seed_id)
    await _persist_primary_extraction(session_factory, seed_id)
    orchestration_id, consistency_application_id = await _run_fact_pipeline(session_factory, seed_id)
    schema_id, schema_version_id = await _create_schema(session_factory, seed_id)
    created_version_id = await project_version_service.create_project_version(
        session_factory,
        project_version_id=_seed_uuid(seed_id, "project-version"),
        project_id=_seed_uuid(seed_id, "project"),
        schema_id=schema_id,
        schema_version_id=schema_version_id,
        orchestration_id=orchestration_id,
        consistency_check_application_id=consistency_application_id,
        created_by_id=_seed_uuid(seed_id, "user"),
        creation_kind="manual",
        reason=_VERSION_REASON,
    )
    if not created_version_id.created_new and created_version_id.id != _seed_uuid(seed_id, "project-version"):
        raise BailianDemoSeedInconsistentError("bailian_demo_seed_inconsistent")
    return await _verify_and_build_result(session_factory, seed_id, created_new=True)
