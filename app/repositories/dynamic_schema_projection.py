from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload

from app.models.document_content import DocumentBlock, SourceEvidence
from app.models.dynamic_schema import DynamicSchema, DynamicSchemaField, DynamicSchemaVersion
from app.models.fact import Fact, FactEvidenceLink, FactValue


async def get_dynamic_schema_by_id(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    schema_id: uuid.UUID,
) -> DynamicSchema | None:
    result = await session.execute(
        select(DynamicSchema).where(
            DynamicSchema.project_id == project_id,
            DynamicSchema.id == schema_id,
        )
    )
    return result.scalar_one_or_none()


async def get_dynamic_schema_version_with_fields_for_projection(
    session: AsyncSession,
    *,
    schema_id: uuid.UUID,
    version_id: uuid.UUID,
) -> DynamicSchemaVersion | None:
    result = await session.execute(
        select(DynamicSchemaVersion)
        .where(
            DynamicSchemaVersion.schema_id == schema_id,
            DynamicSchemaVersion.id == version_id,
        )
        .options(
            joinedload(DynamicSchemaVersion.schema),
            selectinload(DynamicSchemaVersion.fields),
        )
    )
    return result.scalar_one_or_none()


async def list_facts_for_projection(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    subject_kind: str,
    predicate_keys: list[str],
    subject_keys: list[str] | None = None,
) -> list[Fact]:
    statement = (
        select(Fact)
        .where(
            Fact.project_id == project_id,
            Fact.status == "active",
            Fact.subject_kind == subject_kind,
            Fact.predicate_key.in_(predicate_keys),
        )
        .options(
            joinedload(Fact.current_value).options(
                selectinload(FactValue.evidence_links).options(
                    joinedload(FactEvidenceLink.evidence).options(
                        joinedload(SourceEvidence.block)
                    )
                )
            )
        )
    )
    if subject_keys is not None:
        statement = statement.where(Fact.subject_key.in_(subject_keys))
    result = await session.execute(statement)
    return list(result.scalars().unique().all())
