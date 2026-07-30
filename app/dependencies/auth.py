import uuid

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.models.user import User
from app.services import identity as identity_service


async def get_optional_current_user(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> User | None:
    raw_user_id = request.session.get("current_user_id")
    if raw_user_id is None:
        return None

    try:
        user_id = uuid.UUID(str(raw_user_id))
    except (TypeError, ValueError):
        request.session.clear()
        return None

    user = await identity_service.get_active_user_by_id(session, user_id)
    if user is None:
        request.session.clear()
        return None

    return user
