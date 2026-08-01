from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import uuid

import pytest

from app.schemas.consistency_check_persistence import (
    ConsistencyCheckApplicationLedgerRecord,
)
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
        candidate_members=(),
    )


def _authenticated_context(
    *,
    project_id: uuid.UUID,
    consistency_check_application_id: uuid.UUID,
    base_orchestration_id: uuid.UUID,
    base_extraction_run_id: uuid.UUID,
    result_manifest_hash: str,
) -> AuthenticatedConsistencyCheckLedgerProjectionContext:
    return AuthenticatedConsistencyCheckLedgerProjectionContext(
        application=ConsistencyCheckApplicationLedgerRecord(
            id=consistency_check_application_id,
            project_id=project_id,
            consistency_application_id=_uuid("source-consistency-app"),
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
                id=_uuid("fvcc-app"),
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
        source_consistency_application_id=_uuid("source-consistency-app"),
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
                effective_fact_value_ids=(_uuid("base-effective-resolved"),),
                current_decision_seed="resolved",
                current_decision_kind="select_one",
            ),
            _effective_item(
                fact_id=fact_unresolved.fact_id,
                assessment_seed="unresolved",
                review_status="pending_review",
                resolution_status="pending_review",
                resolution_basis="none",
                effective_fact_value_ids=(),
            ),
            _effective_item(
                fact_id=fact_modified.fact_id,
                assessment_seed="modified",
                review_status="reviewed",
                resolution_status="resolved",
                resolution_basis="human_selection",
                effective_fact_value_ids=(_uuid("base-effective-modified"),),
                current_decision_seed="modified",
                current_decision_kind="select_one",
            ),
        ),
    )
    authenticated_context = _authenticated_context(
        project_id=project_id,
        consistency_check_application_id=base_consistency_check_application_id,
        base_orchestration_id=base_orchestration_id,
        base_extraction_run_id=base_extraction_run_id,
        result_manifest_hash=result_manifest_hash,
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
    resolved_item = first.items[0]
    assert resolved_item.base_assessment_id == _uuid("assessment-resolved")
    assert resolved_item.base_current_decision_kind == "select_one"
    assert resolved_item.base_effective_fact_value_ids == (_uuid("base-effective-resolved"),)
    modified_item = first.items[3]
    assert modified_item.base_effective_fact_value_ids == (_uuid("base-effective-modified"),)
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
                current_decision_id=None,
                current_decision_kind=None,
                effective_fact_value_ids=(),
            ),
        ),
        fact_count=1,
        resolved_count=0,
        pending_count=1,
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
                ),
            ),
            fact_count=2,
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
