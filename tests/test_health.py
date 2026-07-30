from starlette.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_returns_expected_payload() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "zhiwen"}
