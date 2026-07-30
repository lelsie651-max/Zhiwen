from starlette.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_home_page_renders_core_copy() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "织文" in response.text
    assert "把散乱文档，变成能发现矛盾的知识库" in response.text
