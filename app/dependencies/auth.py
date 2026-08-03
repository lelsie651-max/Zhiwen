import uuid
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.models.user import User
from app.services import identity as identity_service
from app.utils.csrf import verify_csrf_token


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


async def require_current_user(
    current_user: User | None = Depends(get_optional_current_user),
) -> User:
    if current_user is None:
        raise HTTPException(status_code=401, detail="authentication_required")
    return current_user


def verify_api_csrf_token(
    request: Request,
    csrf_token: Annotated[str | None, Header(alias="X-CSRF-Token")] = None,
) -> None:
    verify_csrf_token(request, csrf_token)
