from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import math
import uuid

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document_content import ExtractionRunOutcome, ExtractionRunStatus
from app.models.fact import (
    Fact,
    FactEvidenceLink,
    FactStatus,
    FactValue,
    FactValueSourceKind,
    FactValueStatus,
    FactValueType,
)
from app.models.inference import InferenceRunStatus, InferenceTaskType
from app.models.project_member import ProjectMemberRole
from app.models.user import UserStatus
from app.repositories import fact as fact_repository
from app.repositories import inference as inference_repository
from app.repositories import user as user_repository
from app.schemas.fact import FactIdentityInput, FactValueInput
from app.schemas.fact_commands import (
    AIProposalInput,
    FactValueDecisionInput,
    HumanFactValueInput,
)
from app.utils.validation import normalize_text


class FactProposalError(Exception):
    """Raised when an AI fact proposal cannot be completed."""


class FactProposalRunNotFoundError(FactProposalError):
    """Raised when the target extraction run does not exist."""


class InvalidFactProposalError(FactProposalError):
    """Raised when the AI proposal payload or related resources are invalid."""


class InferenceRunNotEligibleForFactError(FactProposalError):
    """Raised when an inference run cannot be used as AI fact provenance."""


class FactInferenceProjectMismatchError(FactProposalError):
    """Raised when inference provenance belongs to a different project."""


class FactInferenceSourceMismatchError(FactProposalError):
    """Raised when inference provenance does not cover the target extraction run."""


class FactInferenceEvidenceMismatchError(FactProposalError):
    """Raised when evidence blocks are not present in the inference input batch."""


class RetiredFactError(FactProposalError):
    """Raised when a proposal targets a retired fact."""


class FactPermissionError(FactProposalError):
    """Raised when the actor does not have permission to modify facts."""


class FactNotFoundError(FactProposalError):
    """Raised when the target fact does not exist within the project."""


class FactValueNotFoundError(FactProposalError):
    """Raised when the target fact value does not exist within the project."""


@dataclass(frozen=True)
class NormalizedFactValue:
    value_type: str
    value_json: object | None
    normalized_value_text: str
    value_hash: str


async def propose_ai_fact_value(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    extraction_run_id: uuid.UUID,
    inference_run_id: uuid.UUID,
    payload: AIProposalInput,
) -> FactValue:
    try:
        extraction_run = await fact_repository.get_extraction_run_with_project(session, extraction_run_id)
        if extraction_run is None:
            raise FactProposalRunNotFoundError("Target extraction run not found.")

        run_project = extraction_run.revision.document.project
        if run_project.id != project_id:
            raise InvalidFactProposalError("Extraction run does not belong to the requested project.")

        if extraction_run.status != ExtractionRunStatus.COMPLETED.value:
            raise InvalidFactProposalError("Only completed extraction runs can produce AI fact proposals.")

        if extraction_run.outcome not in {
            ExtractionRunOutcome.SUCCESS.value,
            ExtractionRunOutcome.PARTIAL.value,
        }:
            raise InvalidFactProposalError("Only successful or partial extraction runs can produce AI fact proposals.")

        inference_context = await inference_repository.get_completed_fact_extraction_run_context(
            session,
            inference_run_id=inference_run_id,
        )
        if inference_context is None:
            raise InferenceRunNotEligibleForFactError("Inference run not found.")
        if inference_context.status != InferenceRunStatus.COMPLETED.value:
            raise InferenceRunNotEligibleForFactError(
                "Only completed inference runs can produce AI fact proposals."
            )
        if inference_context.task_type != InferenceTaskType.FACT_EXTRACTION.value:
            raise InferenceRunNotEligibleForFactError(
                "Only fact_extraction inference runs can produce AI fact proposals."
            )
        if inference_context.project_id != project_id:
            raise FactInferenceProjectMismatchError(
                "Inference run does not belong to the requested project."
            )
        if extraction_run_id not in inference_context.extraction_run_id_snapshots:
            raise FactInferenceSourceMismatchError(
                "Inference run input batch does not include the requested extraction run."
            )

        evidence_ids = [evidence.evidence_id for evidence in payload.evidences]
        evidence_records = await fact_repository.list_source_evidences_with_context(session, evidence_ids)
        if len(evidence_records) != len(evidence_ids):
            raise InvalidFactProposalError("All evidences must exist.")

        evidence_by_id = {evidence.id: evidence for evidence in evidence_records}
        for evidence_id in evidence_ids:
            evidence = evidence_by_id[evidence_id]
            evidence_run = evidence.block.extraction_run
            evidence_project = evidence_run.revision.document.project
            if evidence_project.id != project_id:
                raise InvalidFactProposalError("All evidences must belong to the requested project.")
            if evidence_run.id != extraction_run_id:
                raise InvalidFactProposalError("All evidences must belong to the current extraction run.")
            if evidence.block.id not in inference_context.source_block_id_snapshots:
                raise FactInferenceEvidenceMismatchError(
                    "All evidences must come from source blocks present in the inference input batch."
                )

        identity_hash = build_fact_identity_hash(payload.identity)
        normalized_value = normalize_fact_value_input(payload.value)

        fact = await _get_or_create_fact_for_update(
            session,
            project_id=project_id,
            identity=payload.identity,
            identity_hash=identity_hash,
            created_by_id=None,
        )
        if fact.status == FactStatus.RETIRED.value:
            raise RetiredFactError("Retired facts cannot accept new AI proposals.")

        version_no = await fact_repository.get_next_fact_version_no(session, fact.id)
        fact_value = FactValue(
            fact_id=fact.id,
            version_no=version_no,
            value_type=normalized_value.value_type,
            value_json=normalized_value.value_json,
            normalized_value_text=normalized_value.normalized_value_text,
            value_hash=normalized_value.value_hash,
            language_code=payload.value.language_code,
            status=FactValueStatus.PROPOSED.value,
            source_kind=FactValueSourceKind.AI.value,
            extraction_run_id=extraction_run_id,
            inference_run_id=inference_run_id,
            confidence=payload.value.confidence,
            created_by_id=None,
            decided_by_id=None,
            decided_at=None,
        )

        await fact_repository.create_fact_value(session, fact_value)
        evidence_links = [
            FactEvidenceLink(
                fact_value_id=fact_value.id,
                evidence_id=evidence_input.evidence_id,
                role=evidence_input.role.value,
                is_primary=evidence_input.is_primary,
                source_order=index,
            )
            for index, evidence_input in enumerate(payload.evidences)
        ]
        if evidence_links:
            await fact_repository.create_fact_evidence_links(session, evidence_links)
            fact_value.evidence_links.extend(evidence_links)
        fact.values.append(fact_value)
        await session.commit()
        return fact_value
    except BaseException:
        await session.rollback()
        raise


async def create_human_fact_value(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    actor_id: uuid.UUID,
    payload: HumanFactValueInput,
) -> FactValue:
    actor = await _require_fact_actor(session, project_id=project_id, actor_id=actor_id)
    normalized_value = normalize_fact_value_input(payload.value)
    identity_hash = build_fact_identity_hash(payload.identity)
    evidence_records = await _load_project_evidences(
        session,
        project_id=project_id,
        evidence_ids=[evidence.evidence_id for evidence in payload.evidences],
    )

    fact = await _get_or_create_fact_for_update(
        session,
        project_id=project_id,
        identity=payload.identity,
        identity_hash=identity_hash,
        created_by_id=actor.id,
    )
    if fact.status == FactStatus.RETIRED.value:
        raise RetiredFactError("Retired facts cannot accept new human values.")

    current_timestamp = datetime.now(timezone.utc)
    version_no = await fact_repository.get_next_fact_version_no(session, fact.id)
    fact_value = FactValue(
        fact_id=fact.id,
        version_no=version_no,
        value_type=normalized_value.value_type,
        value_json=normalized_value.value_json,
        normalized_value_text=normalized_value.normalized_value_text,
        value_hash=normalized_value.value_hash,
        language_code=payload.value.language_code,
        status=FactValueStatus.ACCEPTED.value,
        source_kind=FactValueSourceKind.HUMAN.value,
        extraction_run_id=None,
        inference_run_id=None,
        confidence=payload.value.confidence,
        created_by_id=actor.id,
        decided_by_id=actor.id,
        decided_at=current_timestamp,
    )

    try:
        await fact_repository.create_fact_value(session, fact_value)
        evidence_links = _build_evidence_links(
            fact_value_id=fact_value.id,
            evidences=payload.evidences,
        )
        if evidence_links:
            await fact_repository.create_fact_evidence_links(session, evidence_links)
            fact_value.evidence_links.extend(evidence_links)

        await _replace_current_value(
            session,
            fact=fact,
            new_current_value=fact_value,
        )
        fact.values.append(fact_value)
        await session.commit()
    except Exception:
        await session.rollback()
        raise

    return fact_value


async def accept_fact_value(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    actor_id: uuid.UUID,
    fact_value_id: uuid.UUID,
    command: FactValueDecisionInput,
) -> FactValue:
    actor = await _require_fact_actor(session, project_id=project_id, actor_id=actor_id)
    fact_value = await fact_repository.get_fact_value_for_update(
        session,
        project_id=project_id,
        fact_value_id=fact_value_id,
    )
    if fact_value is None:
        raise FactValueNotFoundError("Target fact value not found.")

    fact = await fact_repository.get_fact_by_id_for_update(
        session,
        project_id=project_id,
        fact_id=fact_value.fact_id,
    )
    if fact is None:
        raise FactNotFoundError("Target fact not found.")
    if fact.status == FactStatus.RETIRED.value:
        raise RetiredFactError("Retired facts cannot accept new values.")

    if fact_value.status == FactValueStatus.ACCEPTED.value and fact.current_value_id == fact_value.id:
        return fact_value
    if fact_value.status != FactValueStatus.PROPOSED.value:
        raise InvalidFactProposalError("Only proposed fact values can be accepted.")

    current_timestamp = datetime.now(timezone.utc)
    fact_value.status = FactValueStatus.ACCEPTED.value
    fact_value.decided_by_id = actor.id
    fact_value.decided_at = current_timestamp

    try:
        await _replace_current_value(
            session,
            fact=fact,
            new_current_value=fact_value,
            replace_existing=command.replace_current,
        )
        await fact_repository.update_fact_value(session, fact_value)
        await session.commit()
    except Exception:
        await session.rollback()
        raise

    return fact_value


async def reject_fact_value(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    actor_id: uuid.UUID,
    fact_value_id: uuid.UUID,
) -> FactValue:
    actor = await _require_fact_actor(session, project_id=project_id, actor_id=actor_id)
    fact_value = await fact_repository.get_fact_value_for_update(
        session,
        project_id=project_id,
        fact_value_id=fact_value_id,
    )
    if fact_value is None:
        raise FactValueNotFoundError("Target fact value not found.")

    fact = await fact_repository.get_fact_by_id_for_update(
        session,
        project_id=project_id,
        fact_id=fact_value.fact_id,
    )
    if fact is None:
        raise FactNotFoundError("Target fact not found.")
    if fact.status == FactStatus.RETIRED.value:
        raise RetiredFactError("Retired facts cannot reject values.")

    if fact_value.status == FactValueStatus.REJECTED.value:
        return fact_value
    if fact_value.status != FactValueStatus.PROPOSED.value:
        raise InvalidFactProposalError("Only proposed fact values can be rejected.")

    fact_value.status = FactValueStatus.REJECTED.value
    fact_value.decided_by_id = actor.id
    fact_value.decided_at = datetime.now(timezone.utc)

    try:
        await fact_repository.update_fact_value(session, fact_value)
        await session.commit()
    except Exception:
        await session.rollback()
        raise

    return fact_value


def build_fact_identity_hash(identity: FactIdentityInput) -> str:
    payload = {
        "predicate_key": identity.predicate_key,
        "scope_key": identity.scope_key,
        "subject_key": identity.subject_key,
        "subject_kind": identity.subject_kind,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def normalize_fact_value_input(value: FactValueInput) -> NormalizedFactValue:
    normalized_value = _normalize_value_by_type(value.value_type.value, value.value_json)
    normalized_value_text = _build_normalized_value_text(
        value_type=value.value_type.value,
        normalized_value=normalized_value,
    )
    serialized_payload = json.dumps(
        {
            "value": normalized_value,
            "value_type": value.value_type.value,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    value_hash = hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()

    return NormalizedFactValue(
        value_type=value.value_type.value,
        value_json=normalized_value,
        normalized_value_text=normalized_value_text,
        value_hash=value_hash,
    )


def _normalize_value_by_type(value_type: str, raw_value: object | None) -> object | None:
    if value_type == FactValueType.STRING.value:
        if not isinstance(raw_value, str):
            raise InvalidFactProposalError("string values must be strings.")
        normalized = normalize_text(raw_value)
        if not normalized:
            raise InvalidFactProposalError("string values must not be empty.")
        return normalized

    if value_type == FactValueType.NUMBER.value:
        if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
            raise InvalidFactProposalError("number values must be finite integers or floats.")
        if not math.isfinite(float(raw_value)):
            raise InvalidFactProposalError("number values must be finite.")
        return raw_value

    if value_type == FactValueType.BOOLEAN.value:
        if not isinstance(raw_value, bool):
            raise InvalidFactProposalError("boolean values must be booleans.")
        return raw_value

    if value_type == FactValueType.DATE.value:
        if not isinstance(raw_value, str):
            raise InvalidFactProposalError("date values must be ISO date strings.")
        try:
            normalized_date = date.fromisoformat(raw_value)
        except ValueError as exc:
            raise InvalidFactProposalError("date values must use YYYY-MM-DD.") from exc
        return normalized_date.isoformat()

    if value_type == FactValueType.DATETIME.value:
        if not isinstance(raw_value, str):
            raise InvalidFactProposalError("datetime values must be timezone-aware ISO 8601 strings.")
        try:
            parsed_datetime = datetime.fromisoformat(raw_value)
        except ValueError as exc:
            raise InvalidFactProposalError("datetime values must be valid ISO 8601 strings.") from exc
        if parsed_datetime.tzinfo is None or parsed_datetime.utcoffset() is None:
            raise InvalidFactProposalError("datetime values must include timezone information.")
        return parsed_datetime.astimezone(timezone.utc).isoformat()

    if value_type == FactValueType.ENTITY_REF.value:
        if not isinstance(raw_value, dict):
            raise InvalidFactProposalError("entity_ref values must be objects.")
        keys = set(raw_value.keys())
        if keys != {"kind", "key"}:
            raise InvalidFactProposalError("entity_ref values must only contain kind and key.")
        kind = raw_value.get("kind")
        key = raw_value.get("key")
        if not isinstance(kind, str) or not isinstance(key, str):
            raise InvalidFactProposalError("entity_ref kind and key must be strings.")
        normalized_kind = normalize_text(kind)
        normalized_key = normalize_text(key)
        if not normalized_kind or not normalized_key:
            raise InvalidFactProposalError("entity_ref kind and key must not be empty.")
        return {
            "key": normalized_key,
            "kind": normalized_kind,
        }

    if value_type == FactValueType.LIST.value:
        if not isinstance(raw_value, list):
            raise InvalidFactProposalError("list values must be lists.")
        return raw_value

    if value_type == FactValueType.OBJECT.value:
        if not isinstance(raw_value, dict):
            raise InvalidFactProposalError("object values must be objects.")
        return raw_value

    if value_type == FactValueType.NULL.value:
        if raw_value is not None:
            raise InvalidFactProposalError("null values must use None.")
        return None

    raise InvalidFactProposalError("Unsupported fact value type.")


def _build_normalized_value_text(*, value_type: str, normalized_value: object | None) -> str:
    if value_type == FactValueType.STRING.value:
        return normalized_value
    if value_type == FactValueType.DATE.value:
        return normalized_value
    if value_type == FactValueType.DATETIME.value:
        return normalized_value
    if value_type == FactValueType.NULL.value:
        return "null"
    return json.dumps(
        normalized_value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


async def _require_fact_actor(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    actor_id: uuid.UUID,
):
    actor = await user_repository.get_user_by_id(session, actor_id)
    if actor is None or actor.status != UserStatus.ACTIVE.value:
        raise FactPermissionError("Actor must be an active user.")

    project_member = await fact_repository.get_project_member_for_project(
        session,
        project_id=project_id,
        user_id=actor_id,
    )
    if project_member is None:
        raise FactPermissionError("Actor must belong to the target project.")
    if project_member.role not in {
        ProjectMemberRole.OWNER.value,
        ProjectMemberRole.EDITOR.value,
    }:
        raise FactPermissionError("Actor does not have permission to modify facts.")

    return actor


async def _load_project_evidences(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    evidence_ids: list[uuid.UUID],
) -> dict[uuid.UUID, object]:
    if not evidence_ids:
        return {}

    evidence_records = await fact_repository.list_source_evidences_with_context(session, evidence_ids)
    if len(evidence_records) != len(evidence_ids):
        raise InvalidFactProposalError("All evidences must exist.")

    evidence_by_id = {evidence.id: evidence for evidence in evidence_records}
    for evidence_id in evidence_ids:
        evidence = evidence_by_id[evidence_id]
        evidence_project = evidence.block.extraction_run.revision.document.project
        if evidence_project.id != project_id:
            raise InvalidFactProposalError("All evidences must belong to the requested project.")
    return evidence_by_id


async def _get_or_create_fact_for_update(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    identity: FactIdentityInput,
    identity_hash: str,
    created_by_id: uuid.UUID | None,
) -> Fact:
    fact = await fact_repository.get_fact_by_identity_for_update(
        session,
        project_id=project_id,
        identity_hash=identity_hash,
    )
    if fact is not None:
        return fact

    savepoint = await session.begin_nested()
    try:
        fact = Fact(
            project_id=project_id,
            subject_kind=identity.subject_kind,
            subject_key=identity.subject_key,
            predicate_key=identity.predicate_key,
            scope_key=identity.scope_key,
            identity_hash=identity_hash,
            status=FactStatus.ACTIVE.value,
            created_by_id=created_by_id,
        )
        await fact_repository.create_fact(session, fact)
    except IntegrityError:
        await savepoint.rollback()
        fact = await fact_repository.get_fact_by_identity_for_update(
            session,
            project_id=project_id,
            identity_hash=identity_hash,
        )
        if fact is None:
            raise
    else:
        await savepoint.commit()
    return fact


def _build_evidence_links(
    *,
    fact_value_id: uuid.UUID,
    evidences,
) -> list[FactEvidenceLink]:
    return [
        FactEvidenceLink(
            fact_value_id=fact_value_id,
            evidence_id=evidence_input.evidence_id,
            role=evidence_input.role.value,
            is_primary=evidence_input.is_primary,
            source_order=index,
        )
        for index, evidence_input in enumerate(evidences)
    ]


async def _replace_current_value(
    session: AsyncSession,
    *,
    fact: Fact,
    new_current_value: FactValue,
    replace_existing: bool = True,
) -> None:
    if fact.current_value_id is None:
        fact.current_value_id = new_current_value.id
        fact.current_value = new_current_value
        await fact_repository.update_fact(session, fact)
        return

    if fact.current_value_id == new_current_value.id:
        if new_current_value.status != FactValueStatus.ACCEPTED.value:
            raise InvalidFactProposalError("Only accepted fact values may be current.")
        return

    current_value = await fact_repository.get_fact_value_for_update(
        session,
        project_id=fact.project_id,
        fact_value_id=fact.current_value_id,
    )
    if current_value is None:
        raise InvalidFactProposalError("Current fact value must belong to the same project.")
    if current_value.fact_id != fact.id:
        raise InvalidFactProposalError("Current fact value must belong to the same fact.")
    if current_value.status != FactValueStatus.ACCEPTED.value:
        raise InvalidFactProposalError("Current fact value must be accepted.")
    if not replace_existing:
        raise InvalidFactProposalError("Replacing the current fact value requires explicit confirmation.")

    current_value.status = FactValueStatus.SUPERSEDED.value
    await fact_repository.update_fact_value(session, current_value)
    fact.current_value_id = new_current_value.id
    fact.current_value = new_current_value
    await fact_repository.update_fact(session, fact)
