from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.ingestion_validation import IngestionRiskFlag, IngestionValidationReport


async def create_validation_report(
    session: AsyncSession,
    report: IngestionValidationReport,
) -> IngestionValidationReport:
    session.add(report)
    await session.flush()
    return report


async def create_risk_flags(
    session: AsyncSession,
    risk_flags: list[IngestionRiskFlag],
) -> list[IngestionRiskFlag]:
    session.add_all(risk_flags)
    await session.flush()
    return risk_flags
