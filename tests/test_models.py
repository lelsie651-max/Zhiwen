import uuid
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import UniqueConstraint

from app.models import Base
from app.models.base import TimestampMixin, utc_now
from app.models.project_member import ProjectMember
from app.schemas.project import ProjectCreate
from app.schemas.project_member import ProjectMemberCreate
from app.schemas.user import UserCreate


def test_identity_tables_are_registered() -> None:
    assert {"users", "projects", "project_members"} <= set(Base.metadata.tables)


def test_handle_unique_constraint_exists() -> None:
    users_table = Base.metadata.tables["users"]
    assert any(
        isinstance(constraint, UniqueConstraint)
        and tuple(constraint.columns.keys()) == ("handle",)
        for constraint in users_table.constraints
    )


def test_slug_unique_constraint_exists() -> None:
    projects_table = Base.metadata.tables["projects"]
    assert any(
        isinstance(constraint, UniqueConstraint)
        and tuple(constraint.columns.keys()) == ("slug",)
        for constraint in projects_table.constraints
    )


def test_project_member_composite_unique_constraint_exists() -> None:
    project_members_table = Base.metadata.tables["project_members"]
    assert any(
        isinstance(constraint, UniqueConstraint)
        and tuple(constraint.columns.keys()) == ("project_id", "user_id")
        for constraint in project_members_table.constraints
    )


@pytest.mark.parametrize("role", ["owner", "editor", "viewer"])
def test_all_project_member_roles_pass_schema(role: str) -> None:
    payload = ProjectMemberCreate(
        project_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        role=role,
    )

    assert payload.role.value == role


def test_invalid_handle_is_rejected() -> None:
    with pytest.raises(ValidationError):
        UserCreate(handle="Bad Handle", display_name="织文用户")


def test_invalid_visibility_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ProjectCreate(
            name="知识库项目",
            slug="zhiwen-demo",
            visibility="team-only",
        )


def test_invalid_role_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ProjectMemberCreate(
            project_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            role="guest",
        )


def test_project_create_rejects_created_by_id_input() -> None:
    with pytest.raises(ValidationError):
        ProjectCreate(
            name="知识库项目",
            slug="zhiwen-demo",
            created_by_id=uuid.uuid4(),
        )


@pytest.mark.parametrize("column_name", ["created_at", "updated_at"])
def test_timestamp_mixin_columns_use_timezone_and_current_timestamp(
    column_name: str,
) -> None:
    column = TimestampMixin.__dict__[column_name].column

    assert column.type.timezone is True
    assert str(column.server_default.arg) == "CURRENT_TIMESTAMP"
    assert "TIMEZONE(" not in str(column.server_default.arg)


def test_joined_at_uses_timezone_and_current_timestamp() -> None:
    column = ProjectMember.__table__.c.joined_at

    assert column.type.timezone is True
    assert str(column.server_default.arg) == "CURRENT_TIMESTAMP"
    assert "TIMEZONE(" not in str(column.server_default.arg)


def test_python_utc_now_default_is_timezone_aware() -> None:
    value = utc_now()

    assert value.tzinfo is not None
    assert value.utcoffset() is not None
    assert value.utcoffset().total_seconds() == 0


def test_migration_uses_current_timestamp_without_timezone_wrapper() -> None:
    migration_path = Path(__file__).resolve().parents[1] / "alembic" / "versions" / "202607300201_initial_identity_models.py"
    content = migration_path.read_text(encoding="utf-8")

    assert "TIMEZONE(" not in content
    assert content.count('server_default=sa.text("CURRENT_TIMESTAMP")') == 5
    assert content.count("sa.DateTime(timezone=True)") == 5
