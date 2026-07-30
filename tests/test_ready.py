from collections.abc import AsyncIterator

from starlette.testclient import TestClient

from app.core.database import get_db_session
from app.main import app
from app.services import health as health_service


async def override_db_session() -> AsyncIterator[object]:
    yield object()


def test_ready_returns_200_when_database_is_available(monkeypatch) -> None:
    async def fake_is_database_ready(_session: object) -> bool:
        return True

    app.dependency_overrides[get_db_session] = override_db_session
    monkeypatch.setattr(health_service, "is_database_ready", fake_is_database_ready)

    with TestClient(app) as client:
        response = client.get("/ready")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "database": "ok"}


def test_ready_returns_503_when_database_is_unavailable(monkeypatch) -> None:
    async def fake_is_database_ready(_session: object) -> bool:
        return False

    app.dependency_overrides[get_db_session] = override_db_session
    monkeypatch.setattr(health_service, "is_database_ready", fake_is_database_ready)

    with TestClient(app) as client:
        response = client.get("/ready")

    app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready", "database": "unavailable"}
