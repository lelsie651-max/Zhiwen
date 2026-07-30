from collections.abc import AsyncIterator
import asyncio
from datetime import datetime, timezone
import re
import uuid

import pytest
from starlette.testclient import TestClient

from app.core.database import get_db_session
from app.main import app
from app.models.project import Project, ProjectStatus, ProjectVisibility
from app.models.project_member import ProjectMemberRole
from app.models.user import User, UserStatus
from app.schemas.project import ProjectCreate
from app.services import identity as identity_service
from app.services import project as project_service


def extract_csrf_token(html: str) -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', html)
    assert match is not None
    return match.group(1)


def build_user(
    *,
    user_id: uuid.UUID | None = None,
    handle: str = "writer01",
    display_name: str = "织文用户",
) -> User:
    now = datetime.now(timezone.utc)
    return User(
        id=user_id or uuid.uuid4(),
        handle=handle,
        display_name=display_name,
        email=None,
        avatar_url=None,
        locale="zh-CN",
        timezone="Asia/Shanghai",
        status=UserStatus.ACTIVE.value,
        created_at=now,
        updated_at=now,
    )


def build_project(
    *,
    owner_id: uuid.UUID,
    slug: str = "first-project",
    name: str = "第一个项目",
) -> Project:
    now = datetime.now(timezone.utc)
    return Project(
        id=uuid.uuid4(),
        name=name,
        slug=slug,
        description="项目说明",
        visibility=ProjectVisibility.PRIVATE.value,
        status=ProjectStatus.ACTIVE.value,
        created_by_id=owner_id,
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def fake_db_session() -> object:
    return object()


@pytest.fixture(autouse=True)
def override_db_dependency(fake_db_session: object):
    async def _override() -> AsyncIterator[object]:
        yield fake_db_session

    app.dependency_overrides[get_db_session] = _override
    yield fake_db_session
    app.dependency_overrides.clear()


def login_client(
    client: TestClient,
    monkeypatch,
    user: User,
    *,
    list_projects: list[Project] | None = None,
) -> dict[str, uuid.UUID]:
    captured: dict[str, uuid.UUID] = {}

    async def fake_register_user(_session: object, _payload) -> User:
        return user

    async def fake_get_active_user_by_id(_session: object, user_id: uuid.UUID) -> User | None:
        captured["user_id"] = user_id
        if user_id == user.id:
            return user
        return None

    async def fake_list_projects_for_user(_session: object, _user_id: uuid.UUID) -> list[Project]:
        return list_projects or []

    monkeypatch.setattr(identity_service, "register_user", fake_register_user)
    monkeypatch.setattr(identity_service, "get_active_user_by_id", fake_get_active_user_by_id)
    monkeypatch.setattr(project_service, "list_projects_for_user", fake_list_projects_for_user)

    setup_page = client.get("/setup")
    csrf_token = extract_csrf_token(setup_page.text)

    response = client.post(
        "/setup",
        data={
            "handle": user.handle,
            "display_name": user.display_name,
            "email": "",
            "locale": "zh-CN",
            "timezone": "Asia/Shanghai",
            "csrf_token": csrf_token,
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/app"
    return captured


def test_app_redirects_to_setup_when_identity_is_missing() -> None:
    with TestClient(app) as client:
        response = client.get("/app", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/setup"


def test_setup_success_persists_session_user_id(monkeypatch) -> None:
    user = build_user()

    with TestClient(app) as client:
        captured = login_client(client, monkeypatch, user)
        response = client.get("/app")

    assert response.status_code == 200
    assert captured["user_id"] == user.id
    assert user.display_name in response.text


def test_duplicate_handle_is_shown_as_form_error(monkeypatch) -> None:
    async def fake_register_user(_session: object, _payload):
        raise identity_service.DuplicateHandleError()

    monkeypatch.setattr(identity_service, "register_user", fake_register_user)

    with TestClient(app) as client:
        csrf_token = extract_csrf_token(client.get("/setup").text)
        response = client.post(
            "/setup",
            data={
                "handle": "writer01",
                "display_name": "织文用户",
                "email": "",
                "locale": "zh-CN",
                "timezone": "Asia/Shanghai",
                "csrf_token": csrf_token,
            },
        )

    assert response.status_code == 400
    assert "该 handle 已被占用，请更换。" in response.text


def test_logged_in_user_can_see_project_list(monkeypatch) -> None:
    user = build_user()
    project = build_project(owner_id=user.id)

    with TestClient(app) as client:
        login_client(client, monkeypatch, user, list_projects=[project])
        response = client.get("/app")

    assert response.status_code == 200
    assert project.name in response.text
    assert project.slug in response.text


class FakeSession:
    def __init__(self) -> None:
        self.commit_called = False
        self.rollback_called = False

    async def commit(self) -> None:
        self.commit_called = True

    async def rollback(self) -> None:
        self.rollback_called = True


def test_project_creation_creates_owner_membership(monkeypatch) -> None:
    session = FakeSession()
    created: dict[str, object] = {}
    user_id = uuid.uuid4()

    async def fake_get_project_by_slug(_session, _slug: str):
        return None

    async def fake_create_project(_session, project: Project) -> Project:
        project.id = uuid.uuid4()
        created["project"] = project
        return project

    async def fake_create_project_member(_session, project_member) -> object:
        created["project_member"] = project_member
        return project_member

    monkeypatch.setattr(project_service.project_repository, "get_project_by_slug", fake_get_project_by_slug)
    monkeypatch.setattr(project_service.project_repository, "create_project", fake_create_project)
    monkeypatch.setattr(project_service.project_repository, "create_project_member", fake_create_project_member)

    project = asyncio.run(
        project_service.create_project_for_owner(
            session,
            user_id,
            ProjectCreate(
                name="事务项目",
                slug="transaction-project",
                description="测试事务",
                visibility="private",
            ),
        )
    )

    assert session.commit_called is True
    assert created["project"] is project
    assert created["project"].created_by_id == user_id
    assert created["project_member"].user_id == user_id
    assert created["project_member"].role == ProjectMemberRole.OWNER


def test_project_creation_rolls_back_when_member_creation_fails(monkeypatch) -> None:
    session = FakeSession()
    user_id = uuid.uuid4()

    async def fake_get_project_by_slug(_session, _slug: str):
        return None

    async def fake_create_project(_session, project: Project) -> Project:
        project.id = uuid.uuid4()
        return project

    async def fake_create_project_member(_session, _project_member) -> object:
        raise RuntimeError("member create failed")

    monkeypatch.setattr(project_service.project_repository, "get_project_by_slug", fake_get_project_by_slug)
    monkeypatch.setattr(project_service.project_repository, "create_project", fake_create_project)
    monkeypatch.setattr(project_service.project_repository, "create_project_member", fake_create_project_member)

    with pytest.raises(RuntimeError, match="member create failed"):
        asyncio.run(
            project_service.create_project_for_owner(
                session,
                user_id,
                ProjectCreate(
                    name="失败事务项目",
                    slug="rollback-project",
                    description="测试回滚",
                    visibility="private",
                ),
            )
        )

    assert session.commit_called is False
    assert session.rollback_called is True


def test_duplicate_slug_is_shown_as_form_error(monkeypatch) -> None:
    user = build_user()

    async def fake_create_project_for_owner(_session: object, _user_id: uuid.UUID, _payload) -> Project:
        raise project_service.DuplicateProjectSlugError()

    async def fake_list_projects_for_user(
        _session: object,
        _user_id: uuid.UUID,
    ) -> list[Project]:
        return []

    with TestClient(app) as client:
        login_client(client, monkeypatch, user)
        monkeypatch.setattr(project_service, "create_project_for_owner", fake_create_project_for_owner)
        monkeypatch.setattr(project_service, "list_projects_for_user", fake_list_projects_for_user)
        dashboard_page = client.get("/app")
        csrf_token = extract_csrf_token(dashboard_page.text)
        response = client.post(
            "/projects",
            data={
                "name": "重复项目",
                "slug": "first-project",
                "description": "重复 slug",
                "visibility": "private",
                "csrf_token": csrf_token,
            },
        )

    assert response.status_code == 400
    assert "该 slug 已存在，请更换。" in response.text


def test_disabled_user_invalidates_session(monkeypatch) -> None:
    user = build_user()

    with TestClient(app) as client:
        login_client(client, monkeypatch, user)

        async def fake_get_active_user_by_id(_session: object, _user_id: uuid.UUID) -> None:
            return None

        monkeypatch.setattr(identity_service, "get_active_user_by_id", fake_get_active_user_by_id)
        response = client.get("/app", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/setup"
    assert "session=null" in response.headers.get("set-cookie", "")
    assert "expires=Thu, 01 Jan 1970" in response.headers.get("set-cookie", "")


def test_logout_clears_session(monkeypatch) -> None:
    user = build_user()

    with TestClient(app) as client:
        login_client(client, monkeypatch, user)
        dashboard_page = client.get("/app")
        csrf_token = extract_csrf_token(dashboard_page.text)
        response = client.post(
            "/logout",
            data={"csrf_token": csrf_token},
            follow_redirects=False,
        )

    assert response.status_code == 303
    assert response.headers["location"] == "/setup"
    assert "session=null" in response.headers.get("set-cookie", "")
    assert "expires=Thu, 01 Jan 1970" in response.headers.get("set-cookie", "")


def test_get_pages_include_csrf_hidden_field(monkeypatch) -> None:
    user = build_user()

    with TestClient(app) as client:
        setup_response = client.get("/setup")
        login_client(client, monkeypatch, user)
        dashboard_response = client.get("/app")

    assert 'name="csrf_token"' in setup_response.text
    assert 'name="csrf_token"' in dashboard_response.text


def test_setup_rejects_missing_csrf_token() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/setup",
            data={
                "handle": "writer01",
                "display_name": "织文用户",
                "email": "",
                "locale": "zh-CN",
                "timezone": "Asia/Shanghai",
            },
        )

    assert response.status_code == 403


def test_setup_rejects_invalid_csrf_token() -> None:
    with TestClient(app) as client:
        client.get("/setup")
        response = client.post(
            "/setup",
            data={
                "handle": "writer01",
                "display_name": "织文用户",
                "email": "",
                "locale": "zh-CN",
                "timezone": "Asia/Shanghai",
                "csrf_token": "bad-token",
            },
        )

    assert response.status_code == 403


def test_projects_post_accepts_valid_csrf_token(monkeypatch) -> None:
    user = build_user()
    project = build_project(owner_id=user.id)

    async def fake_create_project_for_owner(_session: object, _user_id: uuid.UUID, payload: ProjectCreate) -> Project:
        assert payload.name == "新项目"
        return project

    with TestClient(app) as client:
        login_client(client, monkeypatch, user)
        monkeypatch.setattr(project_service, "create_project_for_owner", fake_create_project_for_owner)
        dashboard_page = client.get("/app")
        csrf_token = extract_csrf_token(dashboard_page.text)
        response = client.post(
            "/projects",
            data={
                "name": "新项目",
                "slug": project.slug,
                "description": "项目说明",
                "visibility": "private",
                "csrf_token": csrf_token,
            },
            follow_redirects=False,
        )

    assert response.status_code == 303
    assert response.headers["location"] == f"/projects/{project.slug}"


def test_projects_post_rejects_invalid_csrf_token(monkeypatch) -> None:
    user = build_user()

    with TestClient(app) as client:
        login_client(client, monkeypatch, user)
        client.get("/app")
        response = client.post(
            "/projects",
            data={
                "name": "新项目",
                "slug": "new-project",
                "description": "",
                "visibility": "private",
                "csrf_token": "bad-token",
            },
        )

    assert response.status_code == 403


def test_logout_requires_csrf_token(monkeypatch) -> None:
    user = build_user()

    with TestClient(app) as client:
        login_client(client, monkeypatch, user)
        response = client.post("/logout", data={"csrf_token": "bad-token"}, follow_redirects=False)

    assert response.status_code == 403


def test_duplicate_slug_is_mapped_without_relying_on_exception_string(monkeypatch) -> None:
    session = FakeSession()
    user_id = uuid.uuid4()
    existing_project = build_project(owner_id=user_id, slug="taken-slug")
    call_count = {"count": 0}

    async def fake_get_project_by_slug(_session, slug: str):
        call_count["count"] += 1
        if call_count["count"] == 1:
            return None
        if slug == "taken-slug":
            return existing_project
        return None

    async def fake_create_project(_session, _project: Project) -> Project:
        raise project_service.IntegrityError("insert failed", params=None, orig=ValueError("db"))

    monkeypatch.setattr(project_service.project_repository, "get_project_by_slug", fake_get_project_by_slug)
    monkeypatch.setattr(project_service.project_repository, "create_project", fake_create_project)

    with pytest.raises(project_service.DuplicateProjectSlugError):
        asyncio.run(
            project_service.create_project_for_owner(
                session,
                user_id,
                ProjectCreate(
                    name="重复项目",
                    slug="taken-slug",
                    description="重复",
                    visibility="private",
                ),
            )
        )

    assert session.rollback_called is True
