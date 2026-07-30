from __future__ import annotations

import asyncio
from collections.abc import Callable
import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.processing_job_executor import execute_revision_extraction_job
from app.storage import FileStorage

logger = logging.getLogger(__name__)


async def dispatch_revision_extraction_job(
    *,
    job_id: uuid.UUID,
    session_factory: Callable[[], AsyncSession],
    storage: FileStorage,
) -> None:
    try:
        result = await execute_revision_extraction_job(
            session_factory,
            job_id=job_id,
            storage=storage,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning(
            "Background extraction dispatch failed safely: job_id=%s status=dispatch_failed",
            job_id,
        )
        return

    logger.info(
        "Background extraction dispatch finished safely: job_id=%s status=%s",
        job_id,
        result.execution_status,
    )
