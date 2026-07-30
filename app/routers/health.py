from sqlalchemy.ext.asyncio import AsyncSession

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.core.database import get_db_session
from app.services import health as health_service

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "zhiwen"}


@router.get("/ready")
async def ready_check(
    session: AsyncSession = Depends(get_db_session),
) -> JSONResponse:
    if await health_service.is_database_ready(session):
        return JSONResponse(
            status_code=200,
            content={"status": "ready", "database": "ok"},
        )

    return JSONResponse(
        status_code=503,
        content={"status": "not_ready", "database": "unavailable"},
    )
