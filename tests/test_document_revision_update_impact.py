from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import uuid

import pytest

from app.schemas.consistency_check_persistence import (
    ConsistencyCheckApplicationLedgerRecord,
)
from app.schemas.consistency_projection import ConsistencyReviewProjectionMember
from app.schemas.document_revision_fact_diff import (
    DocumentRevisionFactDiff,
    DocumentRevisionFactDiffFactSnapshot,
    DocumentRevisionFactDiffItem,
    DocumentRevisionFactDiffValueGroup,
)
from app.schemas.effective_fact_value import (
    EffectiveFactValueProjection,
    EffectiveFactValueProjectionItem,
)
from app.schemas.fact_value_duplicate_grouping import (
    DuplicateGroupingApplicationLedger,
    FactValueConsistencyCandidateApplicationLedger,
)
from app.services import document_revision_update_impact as impact_service
from app.services.consistency_check_persistence import (
    AuthenticatedConsistencyCheckLedgerProjectionContext,
)
from app.services.fact_value_duplicate_grouping import (
    AuthenticatedFactValueConsistencyCandidateApplication,
)


def run_async(awaitable):
    return asyncio.run(awaitable)


class SessionFactory:
    def __init__(self) -> None:
        self.open_count = 0

    def __call__(self):
        factory = self

        class _Context:
            async def __aenter__(self_inner):
                factory.open_count += 1
                return object()

            async def __aexit__(self_inner, exc_type, exc, tb):
                factory.open_count -= 1
                return False

        return _Context()


def _uuid(seed: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, seed)


def _fact(seed: str) -> DocumentRevisionFactDiffFactSnapshot:
    return DocumentRevisionFactDiffFactSnapshot(
        fact_id=_uuid(f"fact-{seed}"),
        identity_hash=f"{seed:0<64}"[:64],
        subject_kind="subject",
        subject_key=f"subject-{seed}",
        predicate_key=f"predicate-{seed}",
        scope_key=None,
        subject_entity_id=None,
    )


def _value_group(seed: str, *fact_value_seeds: str) -> DocumentRevisionFactDiffValueGroup:
    return DocumentRevisionFactDiffValueGroup(
        semantic_key_hash=f"{seed:0<64}"[:64],
        value_type="string",
        value_json=f"value-{seed}",
        referenced_entity_id=None,
        fact_value_ids=tuple(_uuid(fact_value_seed) for fact_value_seed in fact_value_seeds),
        evidences=(),
    )


def _candidate_member(fact_value_id: uuid.UUID) -> ConsistencyReviewProjectionMember:
    return ConsistencyReviewProjectionMember(
        fact_value_id=fact_value_id,
        value_type="string",
        value_json=f"value-{fact_value_id}",
        normalized_value_text=f"normalized-{fact_value_id}",
        referenced_entity_id=None,
        selected_by_current_decision=False,
        current_selection_order=None,
        evidences=(),
    )


def _effective_item(
    *,
    fact_id: uuid.UUID,
    assessment_seed: str,
    review_status: str,
    resolution_status: str,
    resolution_basis: str,
    effective_fact_value_ids: tuple[uuid.UUID, ...],
    current_decision_seed: str | None = None,
    current_decision_kind: str | None = None,
    candidate_members: tuple[ConsistencyReviewProjectionMember, ...] = (),
) -> EffectiveFactValueProjectionItem:
    return EffectiveFactValueProjectionItem(
        fact_id=fact_id,
        candidate_id=_uuid(f"candidate-{assessment_seed}"),
        assessment_id=_uuid(f"assessment-{assessment_seed}"),
        agent_verdict="conflict",
        review_status=review_status,
        resolution_status=resolution_status,
        resolution_basis=resolution_basis,
        current_decision_id=(
            None if current_decision_seed is None else _uuid(f"decision-{current_decision_seed}")
        ),
        current_decision_kind=current_decision_kind,
        effective_fact_value_ids=effective_fact_value_ids,
        candidate_members=candidate_members,
    )


def _authenticated_context(
    *,
    project_id: uuid.UUID,
    consistency_check_application_id: uuid.UUID,
    base_orchestration_id: uuid.UUID,
    base_extraction_run_id: uuid.UUID,
    result_manifest_hash: str,
    source_consistency_application_id: uuid.UUID,
) -> AuthenticatedConsistencyCheckLedgerProjectionContext:
    return AuthenticatedConsistencyCheckLedgerProjectionContext(
        application=ConsistencyCheckApplicationLedgerRecord(
            id=consistency_check_application_id,
            project_id=project_id,
            consistency_application_id=source_consistency_application_id,
            orchestration_id=_uuid("consistency-orchestration"),
            source_result_manifest_hash="s" * 64,
            plan_manifest_hash="p" * 64,
            execution_identity_hash="e" * 64,
            result_manifest_hash=result_manifest_hash,
            prompt_contract_hash="c" * 64,
            provider="provider",
            requested_model="model",
            executor_name="executor",
            executor_version="1.0.0",
            batch_count=1,
            executed_batch_count=1,
            skipped_empty_batch_count=0,
            inference_run_count=1,
            assessment_count=1,
            created_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
        ),
        authenticated_source=AuthenticatedFactValueConsistencyCandidateApplication(
            project_id=project_id,
            application=FactValueConsistencyCandidateApplicationLedger(
                id=source_consistency_application_id,
                duplicate_grouping_application_id=_uuid("duplicate-grouping-app"),
                orchestration_id=base_orchestration_id,
                extraction_run_id=base_extraction_run_id,
                algorithm_version="1.0.0",
                input_manifest_hash="1" * 64,
                result_manifest_hash="2" * 64,
                candidate_count=1,
                member_count=1,
                created_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
            ),
            source_duplicate_grouping_application=DuplicateGroupingApplicationLedger(
                id=_uuid("duplicate-grouping-app"),
                orchestration_id=base_orchestration_id,
                extraction_run_id=base_extraction_run_id,
                algorithm_version="1.0.0",
                input_manifest_hash="3" * 64,
                result_manifest_hash="4" * 64,
                input_fact_value_count=1,
                duplicate_group_count=1,
                duplicate_member_count=1,
                created_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
            ),
            write_plan=object(),
            candidate_ledgers=(),
            member_ledgers=(),
        ),
        candidate_bundles=(),
        source_rows=(),
        batches=(),
        assessments=(),
        citations=(),
    )


def _fixture() -> dict[str, object]:
    project_id = _uuid("project")
    document_id = _uuid("document")
    base_revision_id = _uuid("base-revision")
    target_revision_id = _uuid("target-revision")
    base_extraction_run_id = _uuid("base-run")
    target_extraction_run_id = _uuid("target-run")
    base_orchestration_id = _uuid("base-orchestration")
    target_orchestration_id = _uuid("target-orchestration")
    base_consistency_check_application_id = _uuid("base-consistency-check-app")
    source_consistency_application_id = _uuid("fvcc-app")
    result_manifest_hash = "a" * 64

    fact_resolved = _fact("resolved")
    fact_no_context = _fact("no-context")
    fact_unresolved = _fact("unresolved")
    fact_modified = _fact("modified")
    fact_added = _fact("added")
    fact_removed = _fact("removed")

    fact_diff = DocumentRevisionFactDiff(
        project_id=project_id,
        document_id=document_id,
        base_revision_id=base_revision_id,
        target_revision_id=target_revision_id,
        base_extraction_run_id=base_extraction_run_id,
        target_extraction_run_id=target_extraction_run_id,
        base_orchestration_id=base_orchestration_id,
        target_orchestration_id=target_orchestration_id,
        base_revision_no=1,
        target_revision_no=2,
        base_orchestration_status="completed",
        target_orchestration_status="completed",
        block_diff_manifest_hash="b" * 64,
        fact_diff_algorithm_name="document_revision_fact_diff",
        fact_diff_algorithm_version="1.0.0",
        semantic_fingerprint_algorithm_version="1.0.0",
        planner_name="planner",
        planner_version="1.0.0",
        agent_name="agent",
        agent_version="1.0.0",
        prompt_contract_hash="c" * 64,
        provider="provider",
        requested_model="model",
        executor_name="executor",
        executor_version="1.0.0",
        persistence_name="persistence",
        persistence_version="1.0.0",
        entity_resolution_policy_name="entity-policy",
        entity_resolution_policy_version="1.0.0",
        comparison_quality="complete",
        unchanged_count=3,
        modified_count=1,
        added_count=1,
        removed_count=1,
        items=(
            DocumentRevisionFactDiffItem(
                change_kind="unchanged",
                base_fact=fact_resolved,
                target_fact=replace(fact_resolved),
                base_value_groups=(_value_group("base-resolved", "base-fv-resolved"),),
                target_value_groups=(_value_group("target-resolved", "target-fv-resolved"),),
            ),
            DocumentRevisionFactDiffItem(
                change_kind="unchanged",
                base_fact=fact_no_context,
                target_fact=replace(fact_no_context),
                base_value_groups=(_value_group("base-no-context", "base-fv-no-context"),),
                target_value_groups=(_value_group("target-no-context", "target-fv-no-context"),),
            ),
            DocumentRevisionFactDiffItem(
                change_kind="unchanged",
                base_fact=fact_unresolved,
                target_fact=replace(fact_unresolved),
                base_value_groups=(_value_group("base-unresolved", "base-fv-unresolved"),),
                target_value_groups=(_value_group("target-unresolved", "target-fv-unresolved"),),
            ),
            DocumentRevisionFactDiffItem(
                change_kind="modified",
                base_fact=fact_modified,
                target_fact=replace(fact_modified),
                base_value_groups=(_value_group("base-modified", "base-fv-modified"),),
                target_value_groups=(_value_group("target-modified", "target-fv-modified"),),
            ),
            DocumentRevisionFactDiffItem(
                change_kind="added",
                base_fact=None,
                target_fact=fact_added,
                base_value_groups=(),
                target_value_groups=(_value_group("target-added", "target-fv-added"),),
            ),
            DocumentRevisionFactDiffItem(
                change_kind="removed",
                base_fact=fact_removed,
                target_fact=None,
                base_value_groups=(_value_group("base-removed", "base-fv-removed"),),
                target_value_groups=(),
            ),
        ),
        fact_diff_manifest_hash="d" * 64,
    )
    effective_projection = EffectiveFactValueProjection(
        project_id=project_id,
        consistency_check_application_id=base_consistency_check_application_id,
        source_consistency_application_id=source_consistency_application_id,
        result_manifest_hash=result_manifest_hash,
        fact_count=3,
        resolved_count=2,
        pending_count=1,
        deferred_count=0,
        items=(
            _effective_item(
                fact_id=fact_resolved.fact_id,
                assessment_seed="resolved",
                review_status="reviewed",
                resolution_status="resolved",
                resolution_basis="human_selection",
                effective_fact_value_ids=(_uuid("base-fv-resolved"),),
                current_decision_seed="resolved",
                current_decision_kind="select_one",
                candidate_members=(
                    _candidate_member(_uuid("base-fv-resolved")),
                ),
            ),
            _effective_item(
                fact_id=fact_unresolved.fact_id,
                assessment_seed="unresolved",
                review_status="pending_review",
                resolution_status="pending_review",
                resolution_basis="none",
                effective_fact_value_ids=(),
                candidate_members=(
                    _candidate_member(_uuid("base-fv-unresolved")),
                ),
            ),
            _effective_item(
                fact_id=fact_modified.fact_id,
                assessment_seed="modified",
                review_status="reviewed",
                resolution_status="resolved",
                resolution_basis="human_selection",
                effective_fact_value_ids=(_uuid("base-fv-modified"),),
                current_decision_seed="modified",
                current_decision_kind="select_one",
                candidate_members=(
                    _candidate_member(_uuid("base-fv-modified")),
                ),
            ),
        ),
    )
    authenticated_context = _authenticated_context(
        project_id=project_id,
        consistency_check_application_id=base_consistency_check_application_id,
        base_orchestration_id=base_orchestration_id,
        base_extraction_run_id=base_extraction_run_id,
        result_manifest_hash=result_manifest_hash,
        source_consistency_application_id=source_consistency_application_id,
    )
    return {
        "project_id": project_id,
        "document_id": document_id,
        "base_revision_id": base_revision_id,
        "target_revision_id": target_revision_id,
        "base_extraction_run_id": base_extraction_run_id,
        "target_extraction_run_id": target_extraction_run_id,
        "base_orchestration_id": base_orchestration_id,
        "target_orchestration_id": target_orchestration_id,
        "base_consistency_check_application_id": base_consistency_check_application_id,
        "fact_diff": fact_diff,
        "effective_projection": effective_projection,
        "authenticated_context": authenticated_context,
        "fact_ids": {
            "resolved": fact_resolved.fact_id,
            "no_context": fact_no_context.fact_id,
            "unresolved": fact_unresolved.fact_id,
            "modified": fact_modified.fact_id,
            "added": fact_added.fact_id,
            "removed": fact_removed.fact_id,
        },
    }


def _install_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fact_diff: DocumentRevisionFactDiff | Exception,
    authenticated_context: AuthenticatedConsistencyCheckLedgerProjectionContext | Exception,
    effective_projection: EffectiveFactValueProjection | Exception,
    requested: dict[str, list[uuid.UUID]] | None = None,
) -> None:
    async def fake_fact_diff(
        _session_factory,
        *,
        project_id,
        document_id,
        base_revision_id,
        target_revision_id,
        base_extraction_run_id,
        target_extraction_run_id,
        base_orchestration_id,
        target_orchestration_id,
    ):
        if requested is not None:
            requested.setdefault("fact_diff", []).append(base_revision_id)
        if isinstance(fact_diff, Exception):
            raise fact_diff
        return fact_diff

    async def fake_authenticated_application(
        _session_factory,
        *,
        project_id,
        consistency_check_application_id,
    ):
        if requested is not None:
            requested.setdefault("authenticated", []).append(consistency_check_application_id)
        if isinstance(authenticated_context, Exception):
            raise authenticated_context
        return authenticated_context

    async def fake_effective_projection(
        _session_factory,
        *,
        project_id,
        consistency_check_application_id,
    ):
        if requested is not None:
            requested.setdefault("effective", []).append(consistency_check_application_id)
        if isinstance(effective_projection, Exception):
            raise effective_projection
        return effective_projection

    monkeypatch.setattr(
        impact_service.fact_diff_service,
        "get_document_revision_fact_diff",
        fake_fact_diff,
    )
    monkeypatch.setattr(
        impact_service.consistency_check_persistence_service,
        "authenticate_persisted_consistency_check_application",
        fake_authenticated_application,
    )
    monkeypatch.setattr(
        impact_service.effective_fact_value_service,
        "get_effective_fact_value_projection",
        fake_effective_projection,
    )


def _call(service_session_factory: SessionFactory, fixture: dict[str, object]):
    return run_async(
        impact_service.get_document_revision_update_impact(
            service_session_factory,
            project_id=fixture["project_id"],
            document_id=fixture["document_id"],
            base_revision_id=fixture["base_revision_id"],
            target_revision_id=fixture["target_revision_id"],
            base_extraction_run_id=fixture["base_extraction_run_id"],
            target_extraction_run_id=fixture["target_extraction_run_id"],
            base_orchestration_id=fixture["base_orchestration_id"],
            target_orchestration_id=fixture["target_orchestration_id"],
            base_consistency_check_application_id=fixture["base_consistency_check_application_id"],
        )
    )


def _resign_impact(
    impact: impact_service.DocumentRevisionUpdateImpact,
) -> impact_service.DocumentRevisionUpdateImpact:
    return replace(
        impact,
        impact_manifest_hash=impact_service._build_manifest_hash(impact=impact),
    )


def test_get_document_revision_update_impact_maps_six_kinds_and_preserves_fact_diff_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    requested: dict[str, list[uuid.UUID]] = {}
    _install_dependencies(
        monkeypatch,
        fact_diff=fixture["fact_diff"],
        authenticated_context=fixture["authenticated_context"],
        effective_projection=fixture["effective_projection"],
        requested=requested,
    )
    session_factory = SessionFactory()

    first = _call(session_factory, fixture)
    second = _call(session_factory, fixture)

    assert first == second
    assert [item.fact_id for item in first.items] == [
        fixture["fact_ids"]["resolved"],
        fixture["fact_ids"]["no_context"],
        fixture["fact_ids"]["unresolved"],
        fixture["fact_ids"]["modified"],
        fixture["fact_ids"]["added"],
        fixture["fact_ids"]["removed"],
    ]
    assert [item.impact_kind for item in first.items] == [
        "unchanged_resolved",
        "unchanged_no_review_context",
        "unchanged_unresolved",
        "modified",
        "added",
        "removed",
    ]
    assert [item.requires_review for item in first.items] == [
        False,
        False,
        True,
        True,
        True,
        True,
    ]
    assert first.fact_count == 6
    assert first.review_required_count == 4
    assert first.unchanged_resolved_count == 1
    assert first.unchanged_no_review_context_count == 1
    assert first.unchanged_unresolved_count == 1
    assert first.modified_count == 1
    assert first.added_count == 1
    assert first.removed_count == 1
    assert first.comparison_quality == "complete"
    assert (
        first.base_source_consistency_application_id
        == fixture["authenticated_context"].application.consistency_application_id
    )
    resolved_item = first.items[0]
    assert resolved_item.base_assessment_id == _uuid("assessment-resolved")
    assert resolved_item.base_current_decision_kind == "select_one"
    assert resolved_item.base_effective_fact_value_ids == (_uuid("base-fv-resolved"),)
    modified_item = first.items[3]
    assert modified_item.base_effective_fact_value_ids == (_uuid("base-fv-modified"),)
    assert modified_item.target_value_groups[0].fact_value_ids == (_uuid("target-fv-modified"),)
    removed_item = first.items[5]
    assert removed_item.impact_kind == "removed"
    assert removed_item.base_fact is not None
    assert removed_item.target_fact is None
    assert requested == {
        "fact_diff": [fixture["base_revision_id"], fixture["base_revision_id"]],
        "authenticated": [
            fixture["base_consistency_check_application_id"],
            fixture["base_consistency_check_application_id"],
        ],
        "effective": [
            fixture["base_consistency_check_application_id"],
            fixture["base_consistency_check_application_id"],
        ],
    }
    assert session_factory.open_count == 0


@pytest.mark.parametrize(
    ("review_status", "resolution_status"),
    [
        ("pending_review", "pending_review"),
        ("deferred", "deferred"),
        ("not_required", "unreviewed_compatible"),
    ],
)
def test_get_document_revision_update_impact_marks_unchanged_unresolved_contexts_for_review(
    monkeypatch: pytest.MonkeyPatch,
    review_status: str,
    resolution_status: str,
) -> None:
    fixture = _fixture()
    fixture["effective_projection"] = replace(
        fixture["effective_projection"],
        items=(
            replace(
                fixture["effective_projection"].items[0],
                fact_id=fixture["fact_ids"]["unresolved"],
                review_status=review_status,
                resolution_status=resolution_status,
                resolution_basis="none",
                current_decision_id=(
                    None
                    if review_status != "deferred"
                    else _uuid("decision-deferred-unresolved")
                ),
                current_decision_kind=(
                    None if review_status != "deferred" else "defer"
                ),
                effective_fact_value_ids=(),
                candidate_members=(
                    _candidate_member(_uuid("base-fv-unresolved")),
                ),
            ),
        ),
        fact_count=1,
        resolved_count=0,
        pending_count=0 if review_status == "deferred" else 1,
        deferred_count=1 if review_status == "deferred" else 0,
    )
    _install_dependencies(
        monkeypatch,
        fact_diff=replace(
            fixture["fact_diff"],
            items=(fixture["fact_diff"].items[2],),
            unchanged_count=1,
            modified_count=0,
            added_count=0,
            removed_count=0,
        ),
        authenticated_context=fixture["authenticated_context"],
        effective_projection=fixture["effective_projection"],
    )

    result = _call(SessionFactory(), fixture)

    assert len(result.items) == 1
    assert result.items[0].impact_kind == "unchanged_unresolved"
    assert result.items[0].requires_review is True


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("application_project", "document_revision_update_impact_base_application_project_mismatch"),
        ("application_orchestration", "document_revision_update_impact_base_application_orchestration_mismatch"),
        ("application_run", "document_revision_update_impact_base_application_extraction_run_mismatch"),
    ],
)
def test_get_document_revision_update_impact_rejects_base_application_source_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_code: str,
) -> None:
    fixture = _fixture()
    context = fixture["authenticated_context"]
    if mutation == "application_project":
        context = replace(
            context,
            application=replace(context.application, project_id=_uuid("other-project")),
        )
    elif mutation == "application_orchestration":
        context = replace(
            context,
            authenticated_source=replace(
                context.authenticated_source,
                application=replace(
                    context.authenticated_source.application,
                    orchestration_id=_uuid("wrong-base-orchestration"),
                ),
            ),
        )
    else:
        context = replace(
            context,
            authenticated_source=replace(
                context.authenticated_source,
                source_duplicate_grouping_application=replace(
                    context.authenticated_source.source_duplicate_grouping_application,
                    extraction_run_id=_uuid("wrong-base-run"),
                ),
            ),
        )
    _install_dependencies(
        monkeypatch,
        fact_diff=fixture["fact_diff"],
        authenticated_context=context,
        effective_projection=fixture["effective_projection"],
    )

    with pytest.raises(
        impact_service.DocumentRevisionUpdateImpactError,
        match=expected_code,
    ):
        _call(SessionFactory(), fixture)


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("duplicate_fact", "document_revision_update_impact_effective_context_duplicate_fact"),
        ("unknown_fact", "document_revision_update_impact_effective_context_unknown_fact"),
    ],
)
def test_get_document_revision_update_impact_rejects_duplicate_or_unknown_base_effective_facts(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_code: str,
) -> None:
    fixture = _fixture()
    if mutation == "duplicate_fact":
        fixture["effective_projection"] = replace(
            fixture["effective_projection"],
            items=(
                fixture["effective_projection"].items[0],
                replace(
                    fixture["effective_projection"].items[1],
                    fact_id=fixture["effective_projection"].items[0].fact_id,
                    candidate_members=fixture["effective_projection"].items[0].candidate_members,
                    effective_fact_value_ids=(),
                    resolution_status="pending_review",
                    resolution_basis="none",
                    current_decision_id=None,
                    current_decision_kind=None,
                ),
            ),
            fact_count=2,
            resolved_count=1,
            pending_count=1,
            deferred_count=0,
        )
    else:
        fixture["effective_projection"] = replace(
            fixture["effective_projection"],
            items=(
                replace(
                    fixture["effective_projection"].items[0],
                    fact_id=_uuid("unknown-fact"),
                ),
            ),
            fact_count=1,
            resolved_count=1,
            pending_count=0,
        )
    _install_dependencies(
        monkeypatch,
        fact_diff=fixture["fact_diff"],
        authenticated_context=fixture["authenticated_context"],
        effective_projection=fixture["effective_projection"],
    )

    with pytest.raises(
        impact_service.DocumentRevisionUpdateImpactInvariantError,
        match=expected_code,
    ):
        _call(SessionFactory(), fixture)


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (
            "effective_not_in_base",
            "document_revision_update_impact_effective_fact_value_source_mismatch",
        ),
        (
            "candidate_not_in_base",
            "document_revision_update_impact_candidate_member_source_mismatch",
        ),
        (
            "effective_not_in_candidate",
            "document_revision_update_impact_effective_fact_value_source_mismatch",
        ),
        (
            "base_group_duplicate",
            "document_revision_update_impact_base_fact_value_duplicate",
        ),
        (
            "candidate_duplicate",
            "document_revision_update_impact_candidate_member_duplicate",
        ),
        (
            "effective_duplicate",
            "document_revision_update_impact_effective_fact_value_duplicate",
        ),
    ],
)
def test_get_document_revision_update_impact_rejects_cross_projection_fact_value_binding_drift(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_code: str,
) -> None:
    fixture = _fixture()
    if mutation == "effective_not_in_base":
        fixture["effective_projection"] = replace(
            fixture["effective_projection"],
            items=(
                replace(
                    fixture["effective_projection"].items[0],
                    effective_fact_value_ids=(_uuid("other-fv"),),
                    candidate_members=(_candidate_member(_uuid("base-fv-resolved")),),
                ),
            ),
            fact_count=1,
            resolved_count=1,
            pending_count=0,
            deferred_count=0,
        )
        fixture["fact_diff"] = replace(
            fixture["fact_diff"],
            items=(fixture["fact_diff"].items[0],),
            unchanged_count=1,
            modified_count=0,
            added_count=0,
            removed_count=0,
        )
    elif mutation == "candidate_not_in_base":
        fixture["effective_projection"] = replace(
            fixture["effective_projection"],
            items=(
                replace(
                    fixture["effective_projection"].items[0],
                    candidate_members=(_candidate_member(_uuid("other-fv")),),
                    effective_fact_value_ids=(),
                    resolution_status="pending_review",
                    resolution_basis="none",
                    current_decision_id=None,
                    current_decision_kind=None,
                ),
            ),
            fact_count=1,
            resolved_count=0,
            pending_count=1,
            deferred_count=0,
        )
        fixture["fact_diff"] = replace(
            fixture["fact_diff"],
            items=(fixture["fact_diff"].items[0],),
            unchanged_count=1,
            modified_count=0,
            added_count=0,
            removed_count=0,
        )
    elif mutation == "effective_not_in_candidate":
        fixture["effective_projection"] = replace(
            fixture["effective_projection"],
            items=(
                replace(
                    fixture["effective_projection"].items[0],
                    effective_fact_value_ids=(_uuid("base-fv-resolved"),),
                    candidate_members=(_candidate_member(_uuid("base-fv-resolved-alt")),),
                ),
            ),
            fact_count=1,
            resolved_count=1,
            pending_count=0,
            deferred_count=0,
        )
        fixture["fact_diff"] = replace(
            fixture["fact_diff"],
            items=(
                replace(
                    fixture["fact_diff"].items[0],
                    base_value_groups=(
                        _value_group(
                            "base-resolved-expanded",
                            "base-fv-resolved",
                            "base-fv-resolved-alt",
                        ),
                    ),
                ),
            ),
            unchanged_count=1,
            modified_count=0,
            added_count=0,
            removed_count=0,
        )
    elif mutation == "base_group_duplicate":
        fixture["fact_diff"] = replace(
            fixture["fact_diff"],
            items=(
                replace(
                    fixture["fact_diff"].items[0],
                    base_value_groups=(
                        _value_group(
                            "dup-a",
                            "base-fv-resolved",
                        ),
                        _value_group(
                            "dup-b",
                            "base-fv-resolved",
                        ),
                    ),
                ),
            ),
            unchanged_count=1,
            modified_count=0,
            added_count=0,
            removed_count=0,
        )
        fixture["effective_projection"] = replace(
            fixture["effective_projection"],
            items=(fixture["effective_projection"].items[0],),
            fact_count=1,
            resolved_count=1,
            pending_count=0,
            deferred_count=0,
        )
    elif mutation == "candidate_duplicate":
        fixture["effective_projection"] = replace(
            fixture["effective_projection"],
            items=(
                replace(
                    fixture["effective_projection"].items[0],
                    candidate_members=(
                        _candidate_member(_uuid("base-fv-resolved")),
                        _candidate_member(_uuid("base-fv-resolved")),
                    ),
                ),
            ),
            fact_count=1,
            resolved_count=1,
            pending_count=0,
            deferred_count=0,
        )
        fixture["fact_diff"] = replace(
            fixture["fact_diff"],
            items=(fixture["fact_diff"].items[0],),
            unchanged_count=1,
            modified_count=0,
            added_count=0,
            removed_count=0,
        )
    else:
        fixture["effective_projection"] = replace(
            fixture["effective_projection"],
            items=(
                replace(
                    fixture["effective_projection"].items[0],
                    effective_fact_value_ids=(
                        _uuid("base-fv-resolved"),
                        _uuid("base-fv-resolved"),
                    ),
                ),
            ),
            fact_count=1,
            resolved_count=1,
            pending_count=0,
            deferred_count=0,
        )
        fixture["fact_diff"] = replace(
            fixture["fact_diff"],
            items=(fixture["fact_diff"].items[0],),
            unchanged_count=1,
            modified_count=0,
            added_count=0,
            removed_count=0,
        )
    _install_dependencies(
        monkeypatch,
        fact_diff=fixture["fact_diff"],
        authenticated_context=fixture["authenticated_context"],
        effective_projection=fixture["effective_projection"],
    )

    with pytest.raises(
        impact_service.DocumentRevisionUpdateImpactInvariantError,
        match=expected_code,
    ):
        _call(SessionFactory(), fixture)


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (
            "source_mismatch",
            "document_revision_update_impact_effective_projection_source_mismatch",
        ),
        (
            "fact_count",
            "document_revision_update_impact_effective_projection_count_mismatch",
        ),
        (
            "resolved_count",
            "document_revision_update_impact_effective_projection_count_mismatch",
        ),
        (
            "pending_count",
            "document_revision_update_impact_effective_projection_count_mismatch",
        ),
        (
            "deferred_count",
            "document_revision_update_impact_effective_projection_count_mismatch",
        ),
    ],
)
def test_get_document_revision_update_impact_rejects_effective_projection_source_or_count_drift(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_code: str,
) -> None:
    fixture = _fixture()
    if mutation == "source_mismatch":
        fixture["effective_projection"] = replace(
            fixture["effective_projection"],
            source_consistency_application_id=_uuid("other-source-app"),
        )
    elif mutation == "fact_count":
        fixture["effective_projection"] = replace(
            fixture["effective_projection"],
            fact_count=99,
        )
    elif mutation == "resolved_count":
        fixture["effective_projection"] = replace(
            fixture["effective_projection"],
            resolved_count=99,
        )
    elif mutation == "pending_count":
        fixture["effective_projection"] = replace(
            fixture["effective_projection"],
            pending_count=99,
        )
    else:
        fixture["effective_projection"] = replace(
            fixture["effective_projection"],
            deferred_count=99,
        )
    _install_dependencies(
        monkeypatch,
        fact_diff=fixture["fact_diff"],
        authenticated_context=fixture["authenticated_context"],
        effective_projection=fixture["effective_projection"],
    )

    with pytest.raises(
        impact_service.DocumentRevisionUpdateImpactInvariantError,
        match=expected_code,
    ):
        _call(SessionFactory(), fixture)


def test_get_document_revision_update_impact_hash_changes_when_source_manifest_or_item_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    _install_dependencies(
        monkeypatch,
        fact_diff=fixture["fact_diff"],
        authenticated_context=fixture["authenticated_context"],
        effective_projection=fixture["effective_projection"],
    )
    baseline = _call(SessionFactory(), fixture)

    mutated = _fixture()
    mutated["fact_diff"] = replace(mutated["fact_diff"], fact_diff_manifest_hash="f" * 64)
    _install_dependencies(
        monkeypatch,
        fact_diff=mutated["fact_diff"],
        authenticated_context=mutated["authenticated_context"],
        effective_projection=mutated["effective_projection"],
    )
    changed_manifest = _call(SessionFactory(), mutated)
    assert baseline.impact_manifest_hash != changed_manifest.impact_manifest_hash

    mutated = _fixture()
    mutated["effective_projection"] = replace(
        mutated["effective_projection"],
        items=(
            replace(
                mutated["effective_projection"].items[2],
                fact_id=mutated["fact_diff"].items[2].base_fact.fact_id,
                review_status="deferred",
                resolution_status="deferred",
                resolution_basis="none",
                current_decision_id=_uuid("decision-deferred-hash"),
                current_decision_kind="defer",
                effective_fact_value_ids=(),
                candidate_members=(
                    _candidate_member(_uuid("base-fv-unresolved")),
                ),
            ),
        ),
        fact_count=1,
        resolved_count=0,
        pending_count=0,
        deferred_count=1,
    )
    mutated["fact_diff"] = replace(
        mutated["fact_diff"],
        items=(mutated["fact_diff"].items[2],),
        unchanged_count=1,
        modified_count=0,
        added_count=0,
        removed_count=0,
    )
    _install_dependencies(
        monkeypatch,
        fact_diff=mutated["fact_diff"],
        authenticated_context=mutated["authenticated_context"],
        effective_projection=mutated["effective_projection"],
    )
    changed_item = _call(SessionFactory(), mutated)
    assert baseline.impact_manifest_hash != changed_item.impact_manifest_hash

    mutated = _fixture()
    mutated["fact_diff"] = replace(
        mutated["fact_diff"],
        comparison_quality="partial",
    )
    _install_dependencies(
        monkeypatch,
        fact_diff=mutated["fact_diff"],
        authenticated_context=mutated["authenticated_context"],
        effective_projection=mutated["effective_projection"],
    )
    changed_quality = _call(SessionFactory(), mutated)
    assert changed_quality.comparison_quality == "partial"
    assert baseline.impact_manifest_hash != changed_quality.impact_manifest_hash


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("fact_count", "document_revision_update_impact_fact_count_mismatch"),
        (
            "review_required_count",
            "document_revision_update_impact_review_required_count_mismatch",
        ),
        ("kind_count", "document_revision_update_impact_kind_count_mismatch"),
        ("manifest", "document_revision_update_impact_manifest_mismatch"),
    ],
)
def test_authenticate_document_revision_update_impact_projection_rejects_count_or_manifest_drift(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_code: str,
) -> None:
    fixture = _fixture()
    _install_dependencies(
        monkeypatch,
        fact_diff=fixture["fact_diff"],
        authenticated_context=fixture["authenticated_context"],
        effective_projection=fixture["effective_projection"],
    )
    impact = _call(SessionFactory(), fixture)
    if mutation == "fact_count":
        impact = replace(impact, fact_count=999)
    elif mutation == "review_required_count":
        impact = replace(impact, review_required_count=999)
    elif mutation == "kind_count":
        impact = replace(impact, modified_count=999)
    else:
        impact = replace(impact, impact_manifest_hash="0" * 64)

    with pytest.raises(
        impact_service.DocumentRevisionUpdateImpactInvariantError,
        match=expected_code,
    ):
        impact_service.authenticate_document_revision_update_impact_projection(impact)


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("nonzero_empty", "document_revision_update_impact_fact_count_mismatch"),
        ("duplicate_fact", "document_revision_update_impact_duplicate_fact_id"),
        ("invalid_mapping", "document_revision_update_impact_kind_mapping_invalid"),
        (
            "non_bool_requires_review",
            "document_revision_update_impact_requires_review_invalid",
        ),
        (
            "invalid_requires_review",
            "document_revision_update_impact_requires_review_invalid",
        ),
        ("invalid_shape", "document_revision_update_impact_fact_shape_invalid"),
    ],
)
def test_authenticate_document_revision_update_impact_projection_rejects_item_shape_drift(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_code: str,
) -> None:
    fixture = _fixture()
    _install_dependencies(
        monkeypatch,
        fact_diff=fixture["fact_diff"],
        authenticated_context=fixture["authenticated_context"],
        effective_projection=fixture["effective_projection"],
    )
    impact = _call(SessionFactory(), fixture)
    if mutation == "nonzero_empty":
        impact = replace(
            impact,
            items=(),
            fact_count=1,
            review_required_count=1,
            unchanged_resolved_count=0,
            unchanged_no_review_context_count=0,
            unchanged_unresolved_count=0,
            modified_count=1,
            added_count=0,
            removed_count=0,
            impact_manifest_hash="0" * 64,
        )
    elif mutation == "duplicate_fact":
        duplicate_item = replace(impact.items[1], fact_id=impact.items[0].fact_id)
        impact = replace(
            impact,
            items=(impact.items[0], duplicate_item, *impact.items[2:]),
            impact_manifest_hash="0" * 64,
        )
    elif mutation == "invalid_mapping":
        impact = replace(
            impact,
            items=(
                replace(impact.items[0], fact_change_kind="modified"),
                *impact.items[1:],
            ),
            impact_manifest_hash="0" * 64,
        )
    elif mutation == "non_bool_requires_review":
        impact = replace(
            impact,
            items=(
                *impact.items[:3],
                replace(impact.items[3], requires_review=1),
                *impact.items[4:],
            ),
            impact_manifest_hash="0" * 64,
        )
    elif mutation == "invalid_requires_review":
        impact = replace(
            impact,
            items=(
                *impact.items[:3],
                replace(impact.items[3], requires_review=False),
                *impact.items[4:],
            ),
            impact_manifest_hash="0" * 64,
        )
    else:
        impact = replace(
            impact,
            items=(
                *impact.items[:4],
                replace(impact.items[4], base_fact=impact.items[4].target_fact),
                impact.items[5],
            ),
            impact_manifest_hash="0" * 64,
        )

    with pytest.raises(
        impact_service.DocumentRevisionUpdateImpactInvariantError,
        match=expected_code,
    ):
        impact_service.authenticate_document_revision_update_impact_projection(impact)


@pytest.mark.parametrize(
    ("review_status", "resolution_status", "resolution_basis", "decision_kind", "effective_ids"),
    [
        ("reviewed", "resolved", "human_selection", "select_one", (_uuid("base-fv-resolved"),)),
        ("pending_review", "pending_review", "none", None, ()),
        ("deferred", "deferred", "none", "defer", ()),
        ("not_required", "unreviewed_compatible", "none", None, ()),
    ],
)
def test_authenticate_document_revision_update_impact_projection_accepts_valid_base_review_context_shapes(
    monkeypatch: pytest.MonkeyPatch,
    review_status: str,
    resolution_status: str,
    resolution_basis: str,
    decision_kind: str | None,
    effective_ids: tuple[uuid.UUID, ...],
) -> None:
    fixture = _fixture()
    _install_dependencies(
        monkeypatch,
        fact_diff=fixture["fact_diff"],
        authenticated_context=fixture["authenticated_context"],
        effective_projection=fixture["effective_projection"],
    )
    impact = _call(SessionFactory(), fixture)
    target_index = 0 if resolution_status == "resolved" else 2
    decision_id = None if decision_kind is None else _uuid(f"decision-{resolution_status}")
    impact = replace(
        impact,
        items=(
            *impact.items[:target_index],
            replace(
                impact.items[target_index],
                base_review_status=review_status,
                base_resolution_status=resolution_status,
                base_resolution_basis=resolution_basis,
                base_current_decision_id=decision_id,
                base_current_decision_kind=decision_kind,
                base_effective_fact_value_ids=effective_ids,
            ),
            *impact.items[target_index + 1 :],
        ),
    )
    impact = _resign_impact(impact)

    assert (
        impact_service.authenticate_document_revision_update_impact_projection(impact)
        == impact
    )


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (
            "pending_as_resolved",
            "document_revision_update_impact_base_review_context_invalid",
        ),
        (
            "resolved_as_unresolved",
            "document_revision_update_impact_base_review_context_invalid",
        ),
        (
            "no_review_context_with_assessment",
            "document_revision_update_impact_base_review_context_invalid",
        ),
        (
            "decision_basis_mismatch",
            "document_revision_update_impact_base_review_context_invalid",
        ),
        (
            "effective_duplicate",
            "document_revision_update_impact_base_effective_fact_value_invalid",
        ),
        (
            "effective_unknown",
            "document_revision_update_impact_base_effective_fact_value_invalid",
        ),
        (
            "effective_not_in_group",
            "document_revision_update_impact_base_effective_fact_value_invalid",
        ),
        (
            "added_with_base_context",
            "document_revision_update_impact_base_review_context_invalid",
        ),
        (
            "base_none_with_groups",
            "document_revision_update_impact_fact_shape_invalid",
        ),
        (
            "target_none_with_groups",
            "document_revision_update_impact_fact_shape_invalid",
        ),
    ],
)
def test_authenticate_document_revision_update_impact_projection_rejects_semantic_context_drift_even_with_resigned_manifest(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_code: str,
) -> None:
    fixture = _fixture()
    _install_dependencies(
        monkeypatch,
        fact_diff=fixture["fact_diff"],
        authenticated_context=fixture["authenticated_context"],
        effective_projection=fixture["effective_projection"],
    )
    impact = _call(SessionFactory(), fixture)

    if mutation == "pending_as_resolved":
        impact = replace(
            impact,
            items=(
                *impact.items[:2],
                replace(
                    impact.items[2],
                    impact_kind="unchanged_resolved",
                    requires_review=False,
                ),
                *impact.items[3:],
            ),
            review_required_count=3,
            unchanged_resolved_count=2,
            unchanged_unresolved_count=0,
        )
    elif mutation == "resolved_as_unresolved":
        impact = replace(
            impact,
            items=(
                replace(
                    impact.items[0],
                    impact_kind="unchanged_unresolved",
                    requires_review=True,
                ),
                *impact.items[1:],
            ),
            review_required_count=5,
            unchanged_resolved_count=0,
            unchanged_unresolved_count=2,
        )
    elif mutation == "no_review_context_with_assessment":
        impact = replace(
            impact,
            items=(
                impact.items[0],
                replace(
                    impact.items[1],
                    base_assessment_id=_uuid("fake-assessment"),
                ),
                *impact.items[2:],
            ),
        )
    elif mutation == "decision_basis_mismatch":
        impact = replace(
            impact,
            items=(
                replace(
                    impact.items[0],
                    base_resolution_basis="human_confirmed_compatibility",
                    base_current_decision_kind="select_one",
                ),
                *impact.items[1:],
            ),
        )
    elif mutation == "effective_duplicate":
        impact = replace(
            impact,
            items=(
                replace(
                    impact.items[0],
                    base_effective_fact_value_ids=(
                        _uuid("base-fv-resolved"),
                        _uuid("base-fv-resolved"),
                    ),
                ),
                *impact.items[1:],
            ),
        )
    elif mutation == "effective_unknown":
        impact = replace(
            impact,
            items=(
                replace(
                    impact.items[0],
                    base_effective_fact_value_ids=(_uuid("unknown-effective"),),
                ),
                *impact.items[1:],
            ),
        )
    elif mutation == "effective_not_in_group":
        impact = replace(
            impact,
            items=(
                replace(
                    impact.items[0],
                    base_value_groups=(
                        impact.items[0].base_value_groups[0],
                        _value_group("other", "base-fv-other"),
                    ),
                    base_effective_fact_value_ids=(_uuid("base-fv-missing"),),
                ),
                *impact.items[1:],
            ),
        )
    elif mutation == "added_with_base_context":
        impact = replace(
            impact,
            items=(
                *impact.items[:4],
                replace(
                    impact.items[4],
                    base_assessment_id=_uuid("added-context"),
                    base_review_status="reviewed",
                    base_resolution_status="resolved",
                    base_resolution_basis="human_selection",
                    base_current_decision_id=_uuid("added-decision"),
                    base_current_decision_kind="select_one",
                    base_effective_fact_value_ids=(_uuid("target-fv-added"),),
                ),
                impact.items[5],
            ),
        )
    elif mutation == "base_none_with_groups":
        impact = replace(
            impact,
            items=(
                *impact.items[:4],
                replace(
                    impact.items[4],
                    base_value_groups=(_value_group("unexpected-base", "fake-base-fv"),),
                ),
                impact.items[5],
            ),
        )
    else:
        impact = replace(
            impact,
            items=(
                *impact.items[:5],
                replace(
                    impact.items[5],
                    target_value_groups=(_value_group("unexpected-target", "fake-target-fv"),),
                ),
            ),
        )

    impact = _resign_impact(impact)

    with pytest.raises(
        impact_service.DocumentRevisionUpdateImpactInvariantError,
        match=expected_code,
    ):
        impact_service.authenticate_document_revision_update_impact_projection(impact)


def test_get_document_revision_update_impact_returns_empty_projection_for_zero_fact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    fixture["fact_diff"] = replace(
        fixture["fact_diff"],
        unchanged_count=0,
        modified_count=0,
        added_count=0,
        removed_count=0,
        items=(),
    )
    fixture["effective_projection"] = replace(
        fixture["effective_projection"],
        fact_count=0,
        resolved_count=0,
        pending_count=0,
        deferred_count=0,
        items=(),
    )
    _install_dependencies(
        monkeypatch,
        fact_diff=fixture["fact_diff"],
        authenticated_context=fixture["authenticated_context"],
        effective_projection=fixture["effective_projection"],
    )

    result = _call(SessionFactory(), fixture)

    assert result.items == ()
    assert result.fact_count == 0
    assert result.review_required_count == 0
    assert result.comparison_quality == "complete"


def test_get_document_revision_update_impact_does_not_leak_sensitive_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    sentinel = "SENSITIVE_IMPACT_SENTINEL"
    fixture["effective_projection"] = replace(
        fixture["effective_projection"],
        result_manifest_hash=sentinel,
    )
    _install_dependencies(
        monkeypatch,
        fact_diff=fixture["fact_diff"],
        authenticated_context=fixture["authenticated_context"],
        effective_projection=fixture["effective_projection"],
    )
    session_factory = SessionFactory()

    with pytest.raises(
        impact_service.DocumentRevisionUpdateImpactInvariantError,
        match="document_revision_update_impact_effective_projection_source_mismatch",
    ) as exc_info:
        _call(session_factory, fixture)

    assert sentinel not in str(exc_info.value)
    assert session_factory.open_count == 0
