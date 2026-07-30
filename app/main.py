from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.routers.app import router as app_router
from app.routers.health import router as health_router
from app.routers.projects import router as projects_router
from app.routers.web import router as web_router


BASE_DIR = Path(__file__).resolve().parent
settings = get_settings()

configure_logging(settings)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(
        "Starting application name=%s env=%s port=%s",
        settings.app_name,
        settings.app_env,
        settings.port,
    )
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        debug=settings.runtime_debug,
        lifespan=lifespan,
    )
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.secret_key,
        same_site="lax",
        https_only=settings.is_production,
    )

    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
    app.include_router(web_router)
    app.include_router(health_router)
    app.include_router(app_router)
    app.include_router(projects_router)

    return app


app = create_app()
