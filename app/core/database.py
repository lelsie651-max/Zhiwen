from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings


settings = get_settings()

# The engine is configured once and reused across the app.
engine = create_async_engine(
    settings.database_url,
    echo=settings.runtime_debug,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
async_session_factory = AsyncSessionLocal


def get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    return async_session_factory


async def get_db_session() -> AsyncIterator[AsyncSession]:
    async with AsyncSessionLocal() as session:
        yield session
