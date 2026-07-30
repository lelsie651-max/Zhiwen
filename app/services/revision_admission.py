from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document_revision import DocumentRevision, DocumentRevisionStatus, SourceAuthority
from app.models.ingestion_validation import (
    IngestionValidationReport,
    ValidationRecommendedAuthority,
    ValidationReportOutcome,
    ValidationReportStatus,
    ValidationUserDecision,
)
from app.models.project_member import ProjectMemberRole
from app.repositories import revision_admission as revision_admission_repository
from app.schemas.revision_admission import RevisionAdmissionDecisionInput, RevisionAdmissionResult

TERMINAL_ADMISSION_STATUSES = {
    DocumentRevisionStatus.REJECTED.value,
    DocumentRevisionStatus.QUARANTINED.value,
    DocumentRevisionStatus.NEEDS_CONFIRMATION.value,
    DocumentRevisionStatus.ACCEPTED.value,
}
POST_ADMISSION_STATUSES = {
    DocumentRevisionStatus.PARSING.value,
    DocumentRevisionStatus.EXTRACTING.value,
    DocumentRevisionStatus.AWAITING_REVIEW.value,
    DocumentRevisionStatus.COMPLETED.value,
    DocumentRevisionStatus.FAILED.value,
}


class RevisionAdmissionError(Exception):
    """Raised when revision admission cannot proceed."""


class RevisionAdmissionNotFoundError(RevisionAdmissionError):
    """Raised when the target revision is unavailable within the project."""


class RevisionAdmissionPermissionError(RevisionAdmissionError):
    """Raised when the actor cannot decide revision admission."""


class RevisionAdmissionStateError(RevisionAdmissionError):
    """Raised when revision/report state prevents the requested transition."""


class RevisionAdmissionStateCorruptionError(RevisionAdmissionError):
    """Raised when revision/report state violates invariants."""


async def resolve_revision_admission(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    revision_id: uuid.UUID,
) -> RevisionAdmissionResult:
    revision, report = await _load_revision_and_latest_report_for_update(
        session,
        project_id=project_id,
        revision_id=revision_id,
    )
    _validate_report_ready_for_admission(report)
    _reject_post_admission_status(revision)
    _validate_pass_report_decision_state(report)
    if report.outcome == ValidationReportOutcome.WARNING.value and report.user_decision is not None:
        _validate_existing_decision_consistency(revision=revision, report=report)
        return _build_result(revision=revision, report=report)

    target_status = _map_report_outcome_to_revision_status(report.outcome)
    target_authority = _resolve_pass_authority(revision=revision, report=report)

    if _is_resolve_idempotent(revision=revision, report=report, target_status=target_status, target_authority=target_authority):
        return _build_result(revision=revision, report=report)

    if revision.status != DocumentRevisionStatus.PREFLIGHT_CHECKING.value:
        raise RevisionAdmissionStateError("Revision must be preflight_checking before admission resolution.")

    if target_status == DocumentRevisionStatus.NEEDS_CONFIRMATION.value:
        revision.status = DocumentRevisionStatus.NEEDS_CONFIRMATION.value
    elif target_status == DocumentRevisionStatus.ACCEPTED.value:
        revision.status = DocumentRevisionStatus.ACCEPTED.value
        revision.source_authority = target_authority
    elif target_status == DocumentRevisionStatus.REJECTED.value:
        revision.status = DocumentRevisionStatus.REJECTED.value
    elif target_status == DocumentRevisionStatus.QUARANTINED.value:
        revision.status = DocumentRevisionStatus.QUARANTINED.value
    else:
        raise RevisionAdmissionStateError("Unsupported admission target status.")

    try:
        await session.flush()
        await session.commit()
    except Exception:
        await session.rollback()
        raise

    return _build_result(revision=revision, report=report)


async def decide_revision_admission(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    actor_id: uuid.UUID,
    revision_id: uuid.UUID,
    payload: RevisionAdmissionDecisionInput,
) -> RevisionAdmissionResult:
    revision, report = await _load_revision_and_latest_report_for_update(
        session,
        project_id=project_id,
        revision_id=revision_id,
    )
    _validate_report_ready_for_admission(report)
    _reject_post_admission_status(revision)

    actor = await revision_admission_repository.get_active_user_by_id(session, user_id=actor_id)
    if actor is None:
        raise RevisionAdmissionPermissionError("Only active users can decide revision admission.")
    membership = await revision_admission_repository.get_project_member_for_project(
        session,
        project_id=project_id,
        user_id=actor_id,
    )
    if membership is None or membership.role not in {
        ProjectMemberRole.OWNER.value,
        ProjectMemberRole.EDITOR.value,
    }:
        raise RevisionAdmissionPermissionError("Only owners and editors can decide revision admission.")

    existing_decision = report.user_decision
    if existing_decision is not None:
        _validate_existing_decision_consistency(revision=revision, report=report)
        if existing_decision == payload.decision.value:
            return _build_result(revision=revision, report=report)
        raise RevisionAdmissionStateError("A different admission decision already exists and cannot be overwritten.")

    if report.outcome != ValidationReportOutcome.WARNING.value:
        raise RevisionAdmissionStateError("Only warning reports require manual admission decisions.")
    if revision.status != DocumentRevisionStatus.NEEDS_CONFIRMATION.value:
        raise RevisionAdmissionStateError("Revision must be needs_confirmation before manual decision.")

    decision_time = datetime.now(timezone.utc)
    report.user_decision = payload.decision.value
    report.decided_by_id = actor.id
    report.decided_at = decision_time

    if payload.decision == ValidationUserDecision.CONTINUE_FORMAL:
        revision.source_authority = SourceAuthority.FORMAL.value
        revision.status = DocumentRevisionStatus.ACCEPTED.value
    elif payload.decision == ValidationUserDecision.CONTINUE_REFERENCE:
        revision.source_authority = SourceAuthority.REFERENCE.value
        revision.status = DocumentRevisionStatus.ACCEPTED.value
    elif payload.decision == ValidationUserDecision.CANCEL:
        revision.status = DocumentRevisionStatus.REJECTED.value
    else:
        raise RevisionAdmissionStateError("Unsupported admission decision.")

    try:
        await session.flush()
        await session.commit()
    except Exception:
        await session.rollback()
        raise

    return _build_result(revision=revision, report=report)


async def _load_revision_and_latest_report_for_update(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    revision_id: uuid.UUID,
) -> tuple[DocumentRevision, IngestionValidationReport]:
    revision = await revision_admission_repository.get_revision_for_admission_update(
        session,
        project_id=project_id,
        revision_id=revision_id,
    )
    if revision is None:
        raise RevisionAdmissionNotFoundError("Revision must belong to the target project.")

    report = await revision_admission_repository.get_latest_validation_report_for_update(
        session,
        revision_id=revision.id,
    )
    if report is None:
        raise RevisionAdmissionStateCorruptionError("Revision has no validation report.")
    if report.revision_id != revision.id:
        raise RevisionAdmissionStateCorruptionError("Latest report does not belong to the locked revision.")
    return revision, report


def _validate_report_ready_for_admission(report: IngestionValidationReport) -> None:
    if report.status != ValidationReportStatus.COMPLETED.value:
        raise RevisionAdmissionStateError("Latest validation report must be completed.")
    if report.outcome == ValidationReportOutcome.PENDING.value:
        raise RevisionAdmissionStateError("Latest validation report outcome must not be pending.")


def _reject_post_admission_status(revision: DocumentRevision) -> None:
    if revision.status in POST_ADMISSION_STATUSES:
        raise RevisionAdmissionStateCorruptionError(
            "Admission cannot run after downstream processing has already started."
        )


def _validate_pass_report_decision_state(report: IngestionValidationReport) -> None:
    if report.outcome == ValidationReportOutcome.PASS.value and report.user_decision is not None:
        raise RevisionAdmissionStateCorruptionError("Pass reports must not carry manual admission decisions.")


def _map_report_outcome_to_revision_status(report_outcome: str) -> str:
    mapping = {
        ValidationReportOutcome.PASS.value: DocumentRevisionStatus.ACCEPTED.value,
        ValidationReportOutcome.WARNING.value: DocumentRevisionStatus.NEEDS_CONFIRMATION.value,
        ValidationReportOutcome.REJECT.value: DocumentRevisionStatus.REJECTED.value,
        ValidationReportOutcome.QUARANTINE.value: DocumentRevisionStatus.QUARANTINED.value,
    }
    try:
        return mapping[report_outcome]
    except KeyError as exc:
        raise RevisionAdmissionStateError("Unsupported validation report outcome.") from exc


def _resolve_pass_authority(
    *,
    revision: DocumentRevision,
    report: IngestionValidationReport,
) -> str:
    if report.outcome != ValidationReportOutcome.PASS.value:
        return revision.source_authority
    if report.recommended_authority in {
        ValidationRecommendedAuthority.FORMAL.value,
        ValidationRecommendedAuthority.REFERENCE.value,
    }:
        return report.recommended_authority
    return revision.source_authority


def _is_resolve_idempotent(
    *,
    revision: DocumentRevision,
    report: IngestionValidationReport,
    target_status: str,
    target_authority: str,
) -> bool:
    if revision.status in POST_ADMISSION_STATUSES:
        return False
    if revision.status != target_status:
        return False
    if target_status == DocumentRevisionStatus.ACCEPTED.value:
        if revision.source_authority != target_authority:
            return False
    if target_status == DocumentRevisionStatus.NEEDS_CONFIRMATION.value:
        _validate_warning_decision_consistency(revision=revision, report=report)
    if target_status == DocumentRevisionStatus.REJECTED.value and report.user_decision == ValidationUserDecision.CANCEL.value:
        _validate_existing_decision_consistency(revision=revision, report=report)
    return True


def _validate_warning_decision_consistency(
    *,
    revision: DocumentRevision,
    report: IngestionValidationReport,
) -> None:
    if report.outcome != ValidationReportOutcome.WARNING.value:
        return
    if report.user_decision is None:
        return
    _validate_existing_decision_consistency(revision=revision, report=report)


def _validate_existing_decision_consistency(
    *,
    revision: DocumentRevision,
    report: IngestionValidationReport,
) -> None:
    if report.user_decision == ValidationUserDecision.CONTINUE_FORMAL.value:
        if revision.status != DocumentRevisionStatus.ACCEPTED.value:
            raise RevisionAdmissionStateCorruptionError(
                "Formal decision requires revision status accepted."
            )
        if revision.source_authority != SourceAuthority.FORMAL.value:
            raise RevisionAdmissionStateCorruptionError(
                "Formal decision requires source_authority=formal."
            )
        return
    if report.user_decision == ValidationUserDecision.CONTINUE_REFERENCE.value:
        if revision.status != DocumentRevisionStatus.ACCEPTED.value:
            raise RevisionAdmissionStateCorruptionError(
                "Reference decision requires revision status accepted."
            )
        if revision.source_authority != SourceAuthority.REFERENCE.value:
            raise RevisionAdmissionStateCorruptionError(
                "Reference decision requires source_authority=reference."
            )
        return
    if report.user_decision == ValidationUserDecision.CANCEL.value:
        if revision.status != DocumentRevisionStatus.REJECTED.value:
            raise RevisionAdmissionStateCorruptionError(
                "Cancel decision requires revision status rejected."
            )
        return
    raise RevisionAdmissionStateCorruptionError("Unsupported stored admission decision.")


def _build_result(
    *,
    revision: DocumentRevision,
    report: IngestionValidationReport,
) -> RevisionAdmissionResult:
    return RevisionAdmissionResult(
        revision_id=revision.id,
        report_id=report.id,
        revision_status=revision.status,
        source_authority=revision.source_authority,
        user_decision=report.user_decision,
        decision_required=revision.status == DocumentRevisionStatus.NEEDS_CONFIRMATION.value,
    )
