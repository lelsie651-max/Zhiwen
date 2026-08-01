from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import inspect
import uuid

import pytest

from app.repositories.fact_value_duplicate_grouping import DuplicateGroupingOrchestrationState
from app.schemas.document_revision_diff import (
    DocumentRevisionBlockDiff,
    DocumentRevisionBlockDiffItem,
    DocumentRevisionDiffBlockSnapshot,
)
from app.schemas.fact import FactIdentityInput, FactValueInput
from app.schemas.fact_extraction_persistence import (
    AuthenticatedCompletedFactExtractionApplicationSnapshot,
    AuthenticatedPersistedFactProposalItem,
)
from app.schemas.fact_value_duplicate_grouping import (
    CROSS_BATCH_DUPLICATE_ALGORITHM_VERSION,
    DuplicateCandidate,
)
from app.services import document_revision_fact_diff as fact_diff_service
from app.services.fact_value_duplicate_grouping import (
    AuthenticatedDuplicateGroupingSourceSnapshot,
)


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
    return fact_diff_service.fact_service.build_fact_identity_hash(
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
    return fact_diff_service.fact_service.normalize_fact_value_input(
        FactValueInput(
            value_type=value_type,
            value_json=value_json,
            referenced_entity_id=referenced_entity_id,
            language_code=None,
            confidence=None,
        )
    )


def _block_snapshot(*, seed: str, source_order: int) -> DocumentRevisionDiffBlockSnapshot:
    block_id = _uuid(f"block-{seed}")
    raw_text = f"block text {seed}"
    return DocumentRevisionDiffBlockSnapshot(
        block_id=block_id,
        source_order=source_order,
        block_type="paragraph",
        location_key=f"loc-{seed}",
        page_no=1,
        start_line=source_order + 1,
        end_line=source_order + 1,
        table_index=None,
        row_index=None,
        anchor_hash="a" * 64,
        raw_text_hash=_hash_text(raw_text),
        normalized_text_hash=_hash_text(raw_text),
        raw_text=raw_text,
        normalized_text=raw_text,
    )


def _block_diff(fixture: dict[str, object], *, comparison_quality: str = "complete") -> DocumentRevisionBlockDiff:
    base_unchanged = _block_snapshot(seed="base-unchanged", source_order=0)
    target_unchanged = _block_snapshot(seed="target-unchanged", source_order=0)
    base_moved = _block_snapshot(seed="base-moved", source_order=2)
    target_moved = _block_snapshot(seed="target-moved", source_order=1)
    base_modified = _block_snapshot(seed="base-modified", source_order=1)
    target_modified = _block_snapshot(seed="target-modified", source_order=2)
    target_added = _block_snapshot(seed="target-added", source_order=3)
    base_removed = _block_snapshot(seed="base-removed", source_order=4)
    fixture["block_ids"] = {
        "base_unchanged": base_unchanged.block_id,
        "target_unchanged": target_unchanged.block_id,
        "base_moved": base_moved.block_id,
        "target_moved": target_moved.block_id,
        "base_modified": base_modified.block_id,
        "target_modified": target_modified.block_id,
        "target_added": target_added.block_id,
        "base_removed": base_removed.block_id,
    }
    return DocumentRevisionBlockDiff(
        project_id=fixture["project_id"],
        document_id=fixture["document_id"],
        base_revision_id=fixture["base_revision_id"],
        target_revision_id=fixture["target_revision_id"],
        base_extraction_run_id=fixture["base_run_id"],
        target_extraction_run_id=fixture["target_run_id"],
        base_revision_no=1,
        target_revision_no=2,
        algorithm_name="document_revision_block_diff",
        algorithm_version="1.0.0",
        extractor_name="deterministic-extractor",
        extractor_version="1.0.0",
        detected_format="md",
        comparison_quality=comparison_quality,
        unchanged_count=1,
        modified_count=1,
        moved_count=1,
        added_count=1,
        removed_count=1,
        items=(
            DocumentRevisionBlockDiffItem(
                change_kind="unchanged",
                base_block=base_unchanged,
                target_block=target_unchanged,
            ),
            DocumentRevisionBlockDiffItem(
                change_kind="moved",
                base_block=base_moved,
                target_block=target_moved,
            ),
            DocumentRevisionBlockDiffItem(
                change_kind="modified",
                base_block=base_modified,
                target_block=target_modified,
                raw_text_changed=True,
                normalized_text_changed=True,
                block_type_changed=False,
                locator_changed=False,
            ),
            DocumentRevisionBlockDiffItem(
                change_kind="added",
                base_block=None,
                target_block=target_added,
            ),
            DocumentRevisionBlockDiffItem(
                change_kind="removed",
                base_block=base_removed,
                target_block=None,
            ),
        ),
        diff_manifest_hash="b" * 64,
    )


def _state(
    *,
    orchestration_id: uuid.UUID,
    extraction_run_id: uuid.UUID,
    project_id: uuid.UUID,
    orchestration_status: str = "completed",
    provider: str = "provider",
) -> DuplicateGroupingOrchestrationState:
    return DuplicateGroupingOrchestrationState(
        orchestration_id=orchestration_id,
        extraction_run_id=extraction_run_id,
        project_id=project_id,
        extraction_run_status="completed",
        extraction_run_outcome="success",
        orchestration_status=orchestration_status,
        planner_name="planner",
        planner_version="1.0.0",
        agent_name="agent",
        agent_version="2.0.0",
        prompt_contract_hash="c" * 64,
        provider=provider,
        requested_model="model-x",
        executor_name="executor",
        executor_version="1.2.0",
        persistence_name="persistence",
        persistence_version="1.3.0",
        entity_resolution_policy_name="entity-policy",
        entity_resolution_policy_version="1.4.0",
    )


def _candidate(
    *,
    fact_value_id: uuid.UUID,
    fact_id: uuid.UUID,
    orchestration_id: uuid.UUID,
    extraction_run_id: uuid.UUID,
    source_batch_id: uuid.UUID,
    value_json: object | None,
    evidence_link_ids: tuple[uuid.UUID, ...],
    value_type: str = "string",
    referenced_entity_id: uuid.UUID | None = None,
    evidence_ids: tuple[uuid.UUID, ...] = (),
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


def _source_snapshot(
    *,
    state: DuplicateGroupingOrchestrationState,
    candidates: tuple[DuplicateCandidate, ...],
    application_snapshots: tuple[
        AuthenticatedCompletedFactExtractionApplicationSnapshot, ...
    ] = (),
) -> AuthenticatedDuplicateGroupingSourceSnapshot:
    return AuthenticatedDuplicateGroupingSourceSnapshot(
        state=state,
        candidate_count=len(candidates),
        candidates=candidates,
        application_snapshots=application_snapshots,
    )


def _row(
    *,
    project_id: uuid.UUID,
    fact_id: uuid.UUID,
    fact_identity_hash: str | None,
    fact_value_id: uuid.UUID,
    extraction_run_id: uuid.UUID,
    inference_run_id: uuid.UUID,
    source_batch_id: uuid.UUID,
    value_json: object | None,
    evidence_link_id: uuid.UUID,
    evidence_id: uuid.UUID,
    document_block_id: uuid.UUID,
    block_source_order: int,
    excerpt: str,
    application_project_id: uuid.UUID | None = None,
    application_extraction_run_id: uuid.UUID | None = None,
    application_inference_run_id: uuid.UUID | None = None,
    orchestration_project_id: uuid.UUID | None = None,
    orchestration_extraction_run_id: uuid.UUID | None = None,
    batch_current_inference_run_id: uuid.UUID | None = None,
    subject_kind: str = "subject",
    subject_key: str | None = None,
    predicate_key: str | None = None,
    scope_key: str | None = None,
    subject_entity_id: uuid.UUID | None = None,
    value_type: str = "string",
    referenced_entity_id: uuid.UUID | None = None,
) -> fact_diff_service.document_revision_fact_diff_repository.DocumentRevisionFactDiffSourceRow:
    actual_subject_key = subject_key or f"subject-{fact_id}"
    actual_predicate_key = predicate_key or f"predicate-{fact_id}"
    normalized_value = _normalized_fact_value(
        value_type=value_type,
        value_json=value_json,
        referenced_entity_id=referenced_entity_id,
    )
    return fact_diff_service.document_revision_fact_diff_repository.DocumentRevisionFactDiffSourceRow(
        fact_project_id=project_id,
        fact_id=fact_id,
        fact_identity_hash=(
            fact_identity_hash
            or _fact_identity_hash(
                subject_kind=subject_kind,
                subject_key=actual_subject_key,
                predicate_key=actual_predicate_key,
                scope_key=scope_key,
                subject_entity_id=subject_entity_id,
            )
        ),
        subject_kind=subject_kind,
        subject_key=actual_subject_key,
        predicate_key=actual_predicate_key,
        scope_key=scope_key,
        subject_entity_id=subject_entity_id,
        fact_value_id=fact_value_id,
        extraction_run_id=extraction_run_id,
        inference_run_id=inference_run_id,
        source_batch_id=source_batch_id,
        application_project_id=application_project_id or project_id,
        application_extraction_run_id=application_extraction_run_id or extraction_run_id,
        application_inference_run_id=application_inference_run_id or inference_run_id,
        orchestration_project_id=orchestration_project_id or project_id,
        orchestration_extraction_run_id=orchestration_extraction_run_id or extraction_run_id,
        batch_current_inference_run_id=batch_current_inference_run_id or inference_run_id,
        value_type=normalized_value.value_type,
        value_json=normalized_value.value_json,
        normalized_value_text=normalized_value.normalized_value_text,
        fact_value_hash=normalized_value.value_hash,
        referenced_entity_id=normalized_value.referenced_entity_id,
        evidence_link_id=evidence_link_id,
        evidence_link_source_order=0,
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
    )


def _application_snapshot(
    *,
    project_id: uuid.UUID,
    extraction_run_id: uuid.UUID,
    inference_run_id: uuid.UUID,
    fact_id: uuid.UUID,
    fact_value_id: uuid.UUID,
    evidence_ids: tuple[uuid.UUID, ...],
    subject_entity_id: uuid.UUID | None = None,
    referenced_entity_id: uuid.UUID | None = None,
) -> AuthenticatedCompletedFactExtractionApplicationSnapshot:
    return AuthenticatedCompletedFactExtractionApplicationSnapshot(
        application_id=uuid.uuid4(),
        project_id=project_id,
        extraction_run_id=extraction_run_id,
        inference_run_id=inference_run_id,
        input_batch_id=uuid.uuid4(),
        persistence_name="persistence",
        persistence_version="1.3.0",
        entity_resolution_policy_name="entity-policy",
        entity_resolution_policy_version="1.4.0",
        items=(
            AuthenticatedPersistedFactProposalItem(
                proposal_index=0,
                fact_id=fact_id,
                fact_value_id=fact_value_id,
                subject_entity_id=subject_entity_id,
                referenced_entity_id=referenced_entity_id,
                evidence_ids=evidence_ids,
            ),
        ),
    )


def _fixture() -> dict[str, object]:
    fixture: dict[str, object] = {
        "project_id": _uuid("project"),
        "document_id": _uuid("document"),
        "base_revision_id": _uuid("base-revision"),
        "target_revision_id": _uuid("target-revision"),
        "base_run_id": _uuid("base-run"),
        "target_run_id": _uuid("target-run"),
        "base_orchestration_id": _uuid("base-orchestration"),
        "target_orchestration_id": _uuid("target-orchestration"),
    }
    fixture["block_diff"] = _block_diff(fixture)
    base_state = _state(
        orchestration_id=fixture["base_orchestration_id"],
        extraction_run_id=fixture["base_run_id"],
        project_id=fixture["project_id"],
    )
    target_state = _state(
        orchestration_id=fixture["target_orchestration_id"],
        extraction_run_id=fixture["target_run_id"],
        project_id=fixture["project_id"],
    )
    fixture["base_state"] = base_state
    fixture["target_state"] = target_state

    fact_unchanged = _uuid("fact-unchanged")
    fact_modified = _uuid("fact-modified")
    fact_added = _uuid("fact-added")
    fact_removed = _uuid("fact-removed")
    fact_identity_hashes = {
        "unchanged": _fact_identity_hash(
            subject_kind="subject",
            subject_key="subject-unchanged",
            predicate_key="predicate-unchanged",
        ),
        "modified": _fact_identity_hash(
            subject_kind="subject",
            subject_key="subject-modified",
            predicate_key="predicate-modified",
        ),
        "added": _fact_identity_hash(
            subject_kind="subject",
            subject_key="subject-added",
            predicate_key="predicate-added",
        ),
        "removed": _fact_identity_hash(
            subject_kind="subject",
            subject_key="subject-removed",
            predicate_key="predicate-removed",
        ),
    }
    fixture["fact_ids"] = {
        "unchanged": fact_unchanged,
        "modified": fact_modified,
        "added": fact_added,
        "removed": fact_removed,
    }
    fixture["fact_identity_hashes"] = fact_identity_hashes

    base_candidates = (
        _candidate(
            fact_value_id=_uuid("fv-base-a1"),
            fact_id=fact_unchanged,
            orchestration_id=fixture["base_orchestration_id"],
            extraction_run_id=fixture["base_run_id"],
            source_batch_id=_uuid("base-batch-a1"),
            value_json="same",
            evidence_link_ids=(_uuid("el-base-a1"),),
            evidence_ids=(_uuid("e-base-a1"),),
        ),
        _candidate(
            fact_value_id=_uuid("fv-base-a2"),
            fact_id=fact_unchanged,
            orchestration_id=fixture["base_orchestration_id"],
            extraction_run_id=fixture["base_run_id"],
            source_batch_id=_uuid("base-batch-a2"),
            value_json="same",
            evidence_link_ids=(_uuid("el-base-a2"),),
            evidence_ids=(_uuid("e-base-a2"),),
        ),
        _candidate(
            fact_value_id=_uuid("fv-base-b1"),
            fact_id=fact_modified,
            orchestration_id=fixture["base_orchestration_id"],
            extraction_run_id=fixture["base_run_id"],
            source_batch_id=_uuid("base-batch-b1"),
            value_json="alpha",
            evidence_link_ids=(_uuid("el-base-b1"),),
            evidence_ids=(_uuid("e-base-b1"),),
        ),
        _candidate(
            fact_value_id=_uuid("fv-base-b2"),
            fact_id=fact_modified,
            orchestration_id=fixture["base_orchestration_id"],
            extraction_run_id=fixture["base_run_id"],
            source_batch_id=_uuid("base-batch-b2"),
            value_json="beta",
            evidence_link_ids=(_uuid("el-base-b2"),),
            evidence_ids=(_uuid("e-base-b2"),),
        ),
        _candidate(
            fact_value_id=_uuid("fv-base-d1"),
            fact_id=fact_removed,
            orchestration_id=fixture["base_orchestration_id"],
            extraction_run_id=fixture["base_run_id"],
            source_batch_id=_uuid("base-batch-d1"),
            value_json="removed",
            evidence_link_ids=(_uuid("el-base-d1"),),
            evidence_ids=(_uuid("e-base-d1"),),
        ),
    )
    target_candidates = (
        _candidate(
            fact_value_id=_uuid("fv-target-a1"),
            fact_id=fact_unchanged,
            orchestration_id=fixture["target_orchestration_id"],
            extraction_run_id=fixture["target_run_id"],
            source_batch_id=_uuid("target-batch-a1"),
            value_json="same",
            evidence_link_ids=(_uuid("el-target-a1"),),
            evidence_ids=(_uuid("e-target-a1"),),
        ),
        _candidate(
            fact_value_id=_uuid("fv-target-b1"),
            fact_id=fact_modified,
            orchestration_id=fixture["target_orchestration_id"],
            extraction_run_id=fixture["target_run_id"],
            source_batch_id=_uuid("target-batch-b1"),
            value_json="alpha",
            evidence_link_ids=(_uuid("el-target-b1"),),
            evidence_ids=(_uuid("e-target-b1"),),
        ),
        _candidate(
            fact_value_id=_uuid("fv-target-c1"),
            fact_id=fact_added,
            orchestration_id=fixture["target_orchestration_id"],
            extraction_run_id=fixture["target_run_id"],
            source_batch_id=_uuid("target-batch-c1"),
            value_json="added",
            evidence_link_ids=(_uuid("el-target-c1"),),
            evidence_ids=(_uuid("e-target-c1"),),
        ),
    )
    block_ids = fixture["block_ids"]
    fixture["base_rows"] = (
        _row(
            project_id=fixture["project_id"],
            fact_id=fact_unchanged,
            fact_identity_hash=fact_identity_hashes["unchanged"],
            fact_value_id=_uuid("fv-base-a1"),
            extraction_run_id=fixture["base_run_id"],
            inference_run_id=_uuid("ir-base-a1"),
            source_batch_id=_uuid("base-batch-a1"),
            value_json="same",
            evidence_link_id=_uuid("el-base-a1"),
            evidence_id=_uuid("e-base-a1"),
            document_block_id=block_ids["base_unchanged"],
            block_source_order=0,
            excerpt="same excerpt 1",
            subject_key="subject-unchanged",
            predicate_key="predicate-unchanged",
        ),
        _row(
            project_id=fixture["project_id"],
            fact_id=fact_unchanged,
            fact_identity_hash=fact_identity_hashes["unchanged"],
            fact_value_id=_uuid("fv-base-a2"),
            extraction_run_id=fixture["base_run_id"],
            inference_run_id=_uuid("ir-base-a2"),
            source_batch_id=_uuid("base-batch-a2"),
            value_json="same",
            evidence_link_id=_uuid("el-base-a2"),
            evidence_id=_uuid("e-base-a2"),
            document_block_id=block_ids["base_moved"],
            block_source_order=2,
            excerpt="same excerpt 2",
            subject_key="subject-unchanged",
            predicate_key="predicate-unchanged",
        ),
        _row(
            project_id=fixture["project_id"],
            fact_id=fact_modified,
            fact_identity_hash=fact_identity_hashes["modified"],
            fact_value_id=_uuid("fv-base-b1"),
            extraction_run_id=fixture["base_run_id"],
            inference_run_id=_uuid("ir-base-b1"),
            source_batch_id=_uuid("base-batch-b1"),
            value_json="alpha",
            evidence_link_id=_uuid("el-base-b1"),
            evidence_id=_uuid("e-base-b1"),
            document_block_id=block_ids["base_modified"],
            block_source_order=1,
            excerpt="alpha excerpt",
            subject_key="subject-modified",
            predicate_key="predicate-modified",
        ),
        _row(
            project_id=fixture["project_id"],
            fact_id=fact_modified,
            fact_identity_hash=fact_identity_hashes["modified"],
            fact_value_id=_uuid("fv-base-b2"),
            extraction_run_id=fixture["base_run_id"],
            inference_run_id=_uuid("ir-base-b2"),
            source_batch_id=_uuid("base-batch-b2"),
            value_json="beta",
            evidence_link_id=_uuid("el-base-b2"),
            evidence_id=_uuid("e-base-b2"),
            document_block_id=block_ids["base_moved"],
            block_source_order=2,
            excerpt="beta excerpt",
            subject_key="subject-modified",
            predicate_key="predicate-modified",
        ),
        _row(
            project_id=fixture["project_id"],
            fact_id=fact_removed,
            fact_identity_hash=fact_identity_hashes["removed"],
            fact_value_id=_uuid("fv-base-d1"),
            extraction_run_id=fixture["base_run_id"],
            inference_run_id=_uuid("ir-base-d1"),
            source_batch_id=_uuid("base-batch-d1"),
            value_json="removed",
            evidence_link_id=_uuid("el-base-d1"),
            evidence_id=_uuid("e-base-d1"),
            document_block_id=block_ids["base_removed"],
            block_source_order=4,
            excerpt="removed excerpt",
            subject_key="subject-removed",
            predicate_key="predicate-removed",
        ),
    )
    fixture["target_rows"] = (
        _row(
            project_id=fixture["project_id"],
            fact_id=fact_unchanged,
            fact_identity_hash=fact_identity_hashes["unchanged"],
            fact_value_id=_uuid("fv-target-a1"),
            extraction_run_id=fixture["target_run_id"],
            inference_run_id=_uuid("ir-target-a1"),
            source_batch_id=_uuid("target-batch-a1"),
            value_json="same",
            evidence_link_id=_uuid("el-target-a1"),
            evidence_id=_uuid("e-target-a1"),
            document_block_id=block_ids["target_unchanged"],
            block_source_order=0,
            excerpt="same excerpt target",
            subject_key="subject-unchanged",
            predicate_key="predicate-unchanged",
        ),
        _row(
            project_id=fixture["project_id"],
            fact_id=fact_modified,
            fact_identity_hash=fact_identity_hashes["modified"],
            fact_value_id=_uuid("fv-target-b1"),
            extraction_run_id=fixture["target_run_id"],
            inference_run_id=_uuid("ir-target-b1"),
            source_batch_id=_uuid("target-batch-b1"),
            value_json="alpha",
            evidence_link_id=_uuid("el-target-b1"),
            evidence_id=_uuid("e-target-b1"),
            document_block_id=block_ids["target_modified"],
            block_source_order=2,
            excerpt="alpha excerpt target",
            subject_key="subject-modified",
            predicate_key="predicate-modified",
        ),
        _row(
            project_id=fixture["project_id"],
            fact_id=fact_added,
            fact_identity_hash=fact_identity_hashes["added"],
            fact_value_id=_uuid("fv-target-c1"),
            extraction_run_id=fixture["target_run_id"],
            inference_run_id=_uuid("ir-target-c1"),
            source_batch_id=_uuid("target-batch-c1"),
            value_json="added",
            evidence_link_id=_uuid("el-target-c1"),
            evidence_id=_uuid("e-target-c1"),
            document_block_id=block_ids["target_added"],
            block_source_order=3,
            excerpt="added excerpt",
            subject_key="subject-added",
            predicate_key="predicate-added",
        ),
    )
    base_rows_by_fact_value = {
        row.fact_value_id: tuple(
            candidate_row
            for candidate_row in fixture["base_rows"]
            if candidate_row.fact_value_id == row.fact_value_id
        )
        for row in fixture["base_rows"]
    }
    target_rows_by_fact_value = {
        row.fact_value_id: tuple(
            candidate_row
            for candidate_row in fixture["target_rows"]
            if candidate_row.fact_value_id == row.fact_value_id
        )
        for row in fixture["target_rows"]
    }
    fixture["base_snapshot"] = _source_snapshot(
        state=base_state,
        candidates=base_candidates,
        application_snapshots=tuple(
            _application_snapshot(
                project_id=fixture["project_id"],
                extraction_run_id=fixture["base_run_id"],
                inference_run_id=base_rows_by_fact_value[candidate.fact_value_id][0].inference_run_id,
                fact_id=candidate.fact_id,
                fact_value_id=candidate.fact_value_id,
                evidence_ids=tuple(
                    row.evidence_id for row in base_rows_by_fact_value[candidate.fact_value_id]
                ),
            )
            for candidate in base_candidates
        ),
    )
    fixture["target_snapshot"] = _source_snapshot(
        state=target_state,
        candidates=target_candidates,
        application_snapshots=tuple(
            _application_snapshot(
                project_id=fixture["project_id"],
                extraction_run_id=fixture["target_run_id"],
                inference_run_id=target_rows_by_fact_value[candidate.fact_value_id][0].inference_run_id,
                fact_id=candidate.fact_id,
                fact_value_id=candidate.fact_value_id,
                evidence_ids=tuple(
                    row.evidence_id for row in target_rows_by_fact_value[candidate.fact_value_id]
                ),
            )
            for candidate in target_candidates
        ),
    )
    return fixture


def _install_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    fixture: dict[str, object],
    *,
    requested: dict[str, list[uuid.UUID]] | None = None,
) -> None:
    async def fake_block_diff(*args, **kwargs):
        if requested is not None:
            requested.setdefault("block_diff", []).append(kwargs["base_revision_id"])
        return fixture["block_diff"]

    async def fake_source_snapshot(_session_factory, *, orchestration_id):
        if requested is not None:
            requested.setdefault("snapshot", []).append(orchestration_id)
        if orchestration_id == fixture["base_orchestration_id"]:
            return fixture["base_snapshot"]
        if orchestration_id == fixture["target_orchestration_id"]:
            return fixture["target_snapshot"]
        raise AssertionError("unexpected orchestration id")

    async def fake_list_rows(_session, *, orchestration_id):
        if requested is not None:
            requested.setdefault("rows", []).append(orchestration_id)
        if orchestration_id == fixture["base_orchestration_id"]:
            return fixture["base_rows"]
        if orchestration_id == fixture["target_orchestration_id"]:
            return fixture["target_rows"]
        return ()

    monkeypatch.setattr(
        fact_diff_service.document_revision_diff_service,
        "get_document_revision_block_diff",
        fake_block_diff,
    )
    monkeypatch.setattr(
        fact_diff_service.duplicate_grouping_service,
        "authenticate_duplicate_grouping_source_snapshot",
        fake_source_snapshot,
    )
    monkeypatch.setattr(
        fact_diff_service.document_revision_fact_diff_repository,
        "list_document_revision_fact_diff_source_rows",
        fake_list_rows,
    )


def test_get_document_revision_fact_diff_classifies_items_groups_values_and_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    requested: dict[str, list[uuid.UUID]] = {}
    _install_dependencies(monkeypatch, fixture, requested=requested)
    session_factory = SessionFactory()

    first = run_async(
        fact_diff_service.get_document_revision_fact_diff(
            session_factory,
            project_id=fixture["project_id"],
            document_id=fixture["document_id"],
            base_revision_id=fixture["base_revision_id"],
            target_revision_id=fixture["target_revision_id"],
            base_extraction_run_id=fixture["base_run_id"],
            target_extraction_run_id=fixture["target_run_id"],
            base_orchestration_id=fixture["base_orchestration_id"],
            target_orchestration_id=fixture["target_orchestration_id"],
        )
    )
    second = run_async(
        fact_diff_service.get_document_revision_fact_diff(
            session_factory,
            project_id=fixture["project_id"],
            document_id=fixture["document_id"],
            base_revision_id=fixture["base_revision_id"],
            target_revision_id=fixture["target_revision_id"],
            base_extraction_run_id=fixture["base_run_id"],
            target_extraction_run_id=fixture["target_run_id"],
            base_orchestration_id=fixture["base_orchestration_id"],
            target_orchestration_id=fixture["target_orchestration_id"],
        )
    )

    assert first == second
    assert first.fact_diff_algorithm_name == fact_diff_service.DOCUMENT_REVISION_FACT_DIFF_ALGORITHM_NAME
    assert first.fact_diff_algorithm_version == fact_diff_service.DOCUMENT_REVISION_FACT_DIFF_ALGORITHM_VERSION
    assert first.semantic_fingerprint_algorithm_version == CROSS_BATCH_DUPLICATE_ALGORITHM_VERSION
    assert first.provider == "provider"
    assert first.comparison_quality == "complete"
    assert (
        first.unchanged_count,
        first.modified_count,
        first.added_count,
        first.removed_count,
    ) == (1, 1, 1, 1)
    assert [item.change_kind for item in first.items] == [
        "unchanged",
        "modified",
        "added",
        "removed",
    ]
    unchanged = first.items[0]
    assert unchanged.target_fact is not None
    assert len(unchanged.base_value_groups) == 1
    assert unchanged.base_value_groups[0].fact_value_ids == (
        _uuid("fv-base-a1"),
        _uuid("fv-base-a2"),
    )
    assert [e.block_change_kind for e in unchanged.base_value_groups[0].evidences] == [
        "unchanged",
        "moved",
    ]
    modified = first.items[1]
    assert [group.value_json for group in modified.base_value_groups] == ["alpha", "beta"]
    assert [group.value_json for group in modified.target_value_groups] == ["alpha"]
    added = first.items[2]
    assert added.base_fact is None
    assert added.target_value_groups[0].evidences[0].block_change_kind == "added"
    removed = first.items[3]
    assert removed.target_fact is None
    assert removed.base_value_groups[0].evidences[0].block_change_kind == "removed"
    assert requested == {
        "block_diff": [fixture["base_revision_id"], fixture["base_revision_id"]],
        "snapshot": [
            fixture["base_orchestration_id"],
            fixture["target_orchestration_id"],
            fixture["base_orchestration_id"],
            fixture["target_orchestration_id"],
        ],
        "rows": [
            fixture["base_orchestration_id"],
            fixture["target_orchestration_id"],
            fixture["base_orchestration_id"],
            fixture["target_orchestration_id"],
        ],
    }
    assert all(session.commit_count == 0 for session in session_factory.sessions)
    assert all(session.rollback_count == 1 for session in session_factory.sessions)


def test_get_document_revision_fact_diff_propagates_partial_quality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    fixture["block_diff"] = _block_diff(fixture, comparison_quality="partial")
    fixture["target_snapshot"] = replace(
        fixture["target_snapshot"],
        state=replace(fixture["target_state"], orchestration_status="partial"),
    )
    _install_dependencies(monkeypatch, fixture)

    result = run_async(
        fact_diff_service.get_document_revision_fact_diff(
            SessionFactory(),
            project_id=fixture["project_id"],
            document_id=fixture["document_id"],
            base_revision_id=fixture["base_revision_id"],
            target_revision_id=fixture["target_revision_id"],
            base_extraction_run_id=fixture["base_run_id"],
            target_extraction_run_id=fixture["target_run_id"],
            base_orchestration_id=fixture["base_orchestration_id"],
            target_orchestration_id=fixture["target_orchestration_id"],
        )
    )

    assert result.comparison_quality == "partial"


def test_get_document_revision_fact_diff_returns_empty_projection_for_zero_fact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    fixture["base_snapshot"] = _source_snapshot(state=fixture["base_state"], candidates=())
    fixture["target_snapshot"] = _source_snapshot(state=fixture["target_state"], candidates=())
    fixture["base_rows"] = ()
    fixture["target_rows"] = ()
    _install_dependencies(monkeypatch, fixture)

    result = run_async(
        fact_diff_service.get_document_revision_fact_diff(
            SessionFactory(),
            project_id=fixture["project_id"],
            document_id=fixture["document_id"],
            base_revision_id=fixture["base_revision_id"],
            target_revision_id=fixture["target_revision_id"],
            base_extraction_run_id=fixture["base_run_id"],
            target_extraction_run_id=fixture["target_run_id"],
            base_orchestration_id=fixture["base_orchestration_id"],
            target_orchestration_id=fixture["target_orchestration_id"],
        )
    )

    assert result.items == ()
    assert (
        result.unchanged_count,
        result.modified_count,
        result.added_count,
        result.removed_count,
    ) == (0, 0, 0, 0)


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("project", "document_revision_fact_diff_orchestration_project_mismatch"),
        ("base_run", "document_revision_fact_diff_base_orchestration_run_mismatch"),
        ("target_run", "document_revision_fact_diff_target_orchestration_run_mismatch"),
        ("agent", "document_revision_fact_diff_agent_identity_mismatch"),
    ],
)
def test_get_document_revision_fact_diff_rejects_orchestration_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_code: str,
) -> None:
    fixture = _fixture()
    if mutation == "project":
        fixture["base_snapshot"] = replace(
            fixture["base_snapshot"],
            state=replace(fixture["base_state"], project_id=_uuid("other-project")),
        )
    elif mutation == "base_run":
        fixture["base_snapshot"] = replace(
            fixture["base_snapshot"],
            state=replace(fixture["base_state"], extraction_run_id=_uuid("wrong-run")),
        )
    elif mutation == "target_run":
        fixture["target_snapshot"] = replace(
            fixture["target_snapshot"],
            state=replace(fixture["target_state"], extraction_run_id=_uuid("wrong-target-run")),
        )
    else:
        fixture["target_snapshot"] = replace(
            fixture["target_snapshot"],
            state=replace(fixture["target_state"], provider="other-provider"),
        )
    _install_dependencies(monkeypatch, fixture)

    with pytest.raises(
        fact_diff_service.DocumentRevisionFactDiffError,
        match=expected_code,
    ):
        run_async(
            fact_diff_service.get_document_revision_fact_diff(
                SessionFactory(),
                project_id=fixture["project_id"],
                document_id=fixture["document_id"],
                base_revision_id=fixture["base_revision_id"],
                target_revision_id=fixture["target_revision_id"],
                base_extraction_run_id=fixture["base_run_id"],
                target_extraction_run_id=fixture["target_run_id"],
                base_orchestration_id=fixture["base_orchestration_id"],
                target_orchestration_id=fixture["target_orchestration_id"],
            )
        )


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("missing_fact_value", "document_revision_fact_diff_fact_value_source_mismatch"),
        ("source_batch", "document_revision_fact_diff_fact_value_source_mismatch"),
        ("inference", "document_revision_fact_diff_fact_value_source_mismatch"),
        ("fact_project", "document_revision_fact_diff_fact_project_mismatch"),
        ("missing_block", "document_revision_fact_diff_block_mapping_missing"),
        ("excerpt", "document_revision_fact_diff_evidence_excerpt_mismatch"),
    ],
)
def test_get_document_revision_fact_diff_fails_closed_on_source_and_evidence_drift(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_code: str,
) -> None:
    fixture = _fixture()
    if mutation == "missing_fact_value":
        fixture["base_rows"] = fixture["base_rows"] + (
            _row(
                project_id=fixture["project_id"],
                fact_id=_uuid("extra-fact"),
                fact_identity_hash="9" * 64,
                fact_value_id=_uuid("extra-fv"),
                extraction_run_id=fixture["base_run_id"],
                inference_run_id=_uuid("extra-ir"),
                source_batch_id=_uuid("extra-batch"),
                value_json="extra",
                evidence_link_id=_uuid("extra-el"),
                evidence_id=_uuid("extra-e"),
                document_block_id=fixture["block_ids"]["base_unchanged"],
                block_source_order=0,
                excerpt="extra",
            ),
        )
    elif mutation == "source_batch":
        fixture["base_rows"] = (
            replace(fixture["base_rows"][0], source_batch_id=_uuid("wrong-batch")),
            *fixture["base_rows"][1:],
        )
    elif mutation == "inference":
        fixture["base_rows"] = (
            replace(
                fixture["base_rows"][0],
                application_inference_run_id=_uuid("wrong-inference"),
            ),
            *fixture["base_rows"][1:],
        )
    elif mutation == "fact_project":
        fixture["base_rows"] = (
            replace(fixture["base_rows"][0], fact_project_id=_uuid("other-project")),
            *fixture["base_rows"][1:],
        )
    elif mutation == "missing_block":
        fixture["base_rows"] = (
            replace(fixture["base_rows"][0], document_block_id=_uuid("missing-block")),
            *fixture["base_rows"][1:],
        )
    else:
        fixture["base_rows"] = (
            replace(fixture["base_rows"][0], evidence_excerpt="wrong excerpt"),
            *fixture["base_rows"][1:],
        )
    _install_dependencies(monkeypatch, fixture)

    with pytest.raises(
        fact_diff_service.DocumentRevisionFactDiffInvariantError,
        match=expected_code,
    ):
        run_async(
            fact_diff_service.get_document_revision_fact_diff(
                SessionFactory(),
                project_id=fixture["project_id"],
                document_id=fixture["document_id"],
                base_revision_id=fixture["base_revision_id"],
                target_revision_id=fixture["target_revision_id"],
                base_extraction_run_id=fixture["base_run_id"],
                target_extraction_run_id=fixture["target_run_id"],
                base_orchestration_id=fixture["base_orchestration_id"],
                target_orchestration_id=fixture["target_orchestration_id"],
            )
        )


@pytest.mark.parametrize("mutation", ["algorithm", "identity", "value", "evidence"])
def test_get_document_revision_fact_diff_manifest_changes_when_inputs_change(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    fixture = _fixture()
    _install_dependencies(monkeypatch, fixture)
    baseline = run_async(
        fact_diff_service.get_document_revision_fact_diff(
            SessionFactory(),
            project_id=fixture["project_id"],
            document_id=fixture["document_id"],
            base_revision_id=fixture["base_revision_id"],
            target_revision_id=fixture["target_revision_id"],
            base_extraction_run_id=fixture["base_run_id"],
            target_extraction_run_id=fixture["target_run_id"],
            base_orchestration_id=fixture["base_orchestration_id"],
            target_orchestration_id=fixture["target_orchestration_id"],
        )
    )

    mutated = _fixture()
    if mutation == "algorithm":
        monkeypatch.setattr(
            fact_diff_service,
            "DOCUMENT_REVISION_FACT_DIFF_ALGORITHM_VERSION",
            "1.0.1",
        )
    elif mutation == "identity":
        mutated["base_state"] = replace(mutated["base_state"], provider="provider-2")
        mutated["target_state"] = replace(mutated["target_state"], provider="provider-2")
        mutated["base_snapshot"] = replace(
            mutated["base_snapshot"],
            state=mutated["base_state"],
        )
        mutated["target_snapshot"] = replace(
            mutated["target_snapshot"],
            state=mutated["target_state"],
        )
    elif mutation == "value":
        normalized_value = _normalized_fact_value(
            value_type="string",
            value_json="alpha-2",
        )
        mutated["target_snapshot"] = replace(
            mutated["target_snapshot"],
            candidates=tuple(
                replace(candidate, value_json="alpha-2")
                if candidate.fact_value_id == _uuid("fv-target-b1")
                else candidate
                for candidate in mutated["target_snapshot"].candidates
            ),
            candidate_count=len(mutated["target_snapshot"].candidates),
        )
        mutated["target_rows"] = tuple(
            replace(
                row,
                value_json=normalized_value.value_json,
                normalized_value_text=normalized_value.normalized_value_text,
                fact_value_hash=normalized_value.value_hash,
            )
            if row.fact_value_id == _uuid("fv-target-b1")
            else row
            for row in mutated["target_rows"]
        )
    else:
        mutated["target_rows"] = tuple(
            replace(
                row,
                evidence_excerpt="added excerpt changed",
                evidence_end_offset=len("added excerpt changed"),
                evidence_excerpt_hash=_hash_text("added excerpt changed"),
                block_raw_text="added excerpt changed",
            )
            if row.fact_value_id == _uuid("fv-target-c1")
            else row
            for row in mutated["target_rows"]
        )
    _install_dependencies(monkeypatch, mutated)
    changed = run_async(
        fact_diff_service.get_document_revision_fact_diff(
            SessionFactory(),
            project_id=mutated["project_id"],
            document_id=mutated["document_id"],
            base_revision_id=mutated["base_revision_id"],
            target_revision_id=mutated["target_revision_id"],
            base_extraction_run_id=mutated["base_run_id"],
            target_extraction_run_id=mutated["target_run_id"],
            base_orchestration_id=mutated["base_orchestration_id"],
            target_orchestration_id=mutated["target_orchestration_id"],
        )
    )

    assert baseline.fact_diff_manifest_hash != changed.fact_diff_manifest_hash


def test_get_document_revision_fact_diff_does_not_write_and_does_not_leak_sensitive_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    sentinel = "SENSITIVE_FACT_DIFF_SENTINEL"
    fixture["base_rows"] = (
        replace(
            fixture["base_rows"][0],
            evidence_excerpt=sentinel,
        ),
        *fixture["base_rows"][1:],
    )
    session_factory = SessionFactory()
    _install_dependencies(monkeypatch, fixture)

    with pytest.raises(
        fact_diff_service.DocumentRevisionFactDiffInvariantError,
        match="document_revision_fact_diff_evidence_excerpt_mismatch",
    ) as exc_info:
        run_async(
            fact_diff_service.get_document_revision_fact_diff(
                session_factory,
                project_id=fixture["project_id"],
                document_id=fixture["document_id"],
                base_revision_id=fixture["base_revision_id"],
                target_revision_id=fixture["target_revision_id"],
                base_extraction_run_id=fixture["base_run_id"],
                target_extraction_run_id=fixture["target_run_id"],
                base_orchestration_id=fixture["base_orchestration_id"],
                target_orchestration_id=fixture["target_orchestration_id"],
            )
        )

    assert sentinel not in str(exc_info.value)
    assert all(session.commit_count == 0 for session in session_factory.sessions)
    assert all(session.rollback_count == 1 for session in session_factory.sessions)


def test_get_document_revision_fact_diff_rejects_fact_identity_hash_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    fixture["base_rows"] = (
        replace(fixture["base_rows"][0], fact_identity_hash="f" * 64),
        *fixture["base_rows"][1:],
    )
    _install_dependencies(monkeypatch, fixture)

    with pytest.raises(
        fact_diff_service.DocumentRevisionFactDiffInvariantError,
        match="document_revision_fact_diff_fact_identity_mismatch",
    ):
        run_async(
            fact_diff_service.get_document_revision_fact_diff(
                SessionFactory(),
                project_id=fixture["project_id"],
                document_id=fixture["document_id"],
                base_revision_id=fixture["base_revision_id"],
                target_revision_id=fixture["target_revision_id"],
                base_extraction_run_id=fixture["base_run_id"],
                target_extraction_run_id=fixture["target_run_id"],
                base_orchestration_id=fixture["base_orchestration_id"],
                target_orchestration_id=fixture["target_orchestration_id"],
            )
        )


@pytest.mark.parametrize(
    ("field_name", "field_value", "expected_code"),
    [
        ("fact_value_hash", "f" * 64, "document_revision_fact_diff_fact_value_mismatch"),
        ("normalized_value_text", "tampered", "document_revision_fact_diff_fact_value_mismatch"),
        ("value_json", {"unexpected": "shape"}, "document_revision_fact_diff_fact_value_invalid"),
    ],
)
def test_get_document_revision_fact_diff_rejects_fact_value_certification_drift(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    field_value: object,
    expected_code: str,
) -> None:
    fixture = _fixture()
    sentinel = "SENSITIVE_VALUE_JSON_SENTINEL"
    value = sentinel if field_name == "normalized_value_text" else field_value
    fixture["base_rows"] = (
        replace(fixture["base_rows"][0], **{field_name: value}),
        *fixture["base_rows"][1:],
    )
    _install_dependencies(monkeypatch, fixture)

    with pytest.raises(
        fact_diff_service.DocumentRevisionFactDiffInvariantError,
        match=expected_code,
    ) as exc_info:
        run_async(
            fact_diff_service.get_document_revision_fact_diff(
                SessionFactory(),
                project_id=fixture["project_id"],
                document_id=fixture["document_id"],
                base_revision_id=fixture["base_revision_id"],
                target_revision_id=fixture["target_revision_id"],
                base_extraction_run_id=fixture["base_run_id"],
                target_extraction_run_id=fixture["target_run_id"],
                base_orchestration_id=fixture["base_orchestration_id"],
                target_orchestration_id=fixture["target_orchestration_id"],
            )
        )

    assert sentinel not in str(exc_info.value)


def test_service_source_uses_block_diff_and_duplicate_grouping_snapshot_helpers() -> None:
    source = inspect.getsource(fact_diff_service)
    assert "get_document_revision_block_diff" in source
    assert "authenticate_duplicate_grouping_source_snapshot" in source
    assert "llm" not in source
    assert "latest" not in source
