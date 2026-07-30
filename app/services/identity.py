import uuid

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User, UserStatus
from app.repositories import user as user_repository
from app.schemas.user import UserCreate


class IdentityConflictError(Exception):
    def __init__(self, field: str, message: str):
        super().__init__(message)
        self.field = field
        self.message = message


class DuplicateHandleError(IdentityConflictError):
    def __init__(self) -> None:
        super().__init__("handle", "该 handle 已被占用，请更换。")


class DuplicateEmailError(IdentityConflictError):
    def __init__(self) -> None:
        super().__init__("email", "该邮箱已被占用，请更换。")


async def get_active_user_by_id(
    session: AsyncSession,
    user_id: uuid.UUID,
) -> User | None:
    user = await user_repository.get_user_by_id(session, user_id)
    if user is None or user.status != UserStatus.ACTIVE.value:
        return None
    return user


async def register_user(session: AsyncSession, payload: UserCreate) -> User:
    try:
        if await user_repository.get_user_by_handle(session, payload.handle):
            raise DuplicateHandleError()
        if payload.email and await user_repository.get_user_by_email(session, str(payload.email)):
            raise DuplicateEmailError()

        user = User(**payload.model_dump(mode="python"))
        await user_repository.create_user(session, user)
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        message = str(exc)
        if "uq_users_handle" in message:
            raise DuplicateHandleError() from exc
        if "uq_users_email" in message:
            raise DuplicateEmailError() from exc
        raise
    except Exception:
        await session.rollback()
        raise

    return user
