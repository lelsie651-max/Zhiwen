from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger


logger = get_logger(__name__)


async def is_database_ready(session: AsyncSession) -> bool:
    """Run a lightweight readiness query without exposing internals."""

    try:
        await session.execute(text("SELECT 1"))
    except Exception:
        logger.warning("Database readiness check failed.")
        return False
    return True
