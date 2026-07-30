import uuid

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.project import Project
from app.models.project_member import ProjectMember, ProjectMemberRole
from app.repositories import project as project_repository
from app.schemas.project import ProjectCreate
from app.schemas.project_member import ProjectMemberCreate


class ProjectConflictError(Exception):
    def __init__(self, field: str, message: str):
        super().__init__(message)
        self.field = field
        self.message = message


class DuplicateProjectSlugError(ProjectConflictError):
    def __init__(self) -> None:
        super().__init__("slug", "该 slug 已存在，请更换。")


async def list_projects_for_user(session: AsyncSession, user_id: uuid.UUID) -> list[Project]:
    return await project_repository.list_projects_for_user(session, user_id)


async def get_project_for_user_by_slug(
    session: AsyncSession,
    user_id: uuid.UUID,
    slug: str,
) -> Project | None:
    return await project_repository.get_project_for_user_by_slug(session, user_id, slug)


async def create_project_for_owner(
    session: AsyncSession,
    current_user_id: uuid.UUID,
    payload: ProjectCreate,
) -> Project:
    try:
        if await project_repository.get_project_by_slug(session, payload.slug):
            raise DuplicateProjectSlugError()

        project = Project(
            **payload.model_dump(mode="python"),
            created_by_id=current_user_id,
        )
        await project_repository.create_project(session, project)

        project_member = ProjectMember(
            **ProjectMemberCreate(
                project_id=project.id,
                user_id=current_user_id,
                role=ProjectMemberRole.OWNER,
            ).model_dump(mode="python")
        )
        await project_repository.create_project_member(session, project_member)
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        if await project_repository.get_project_by_slug(session, payload.slug):
            raise DuplicateProjectSlugError() from exc
        raise
    except Exception:
        await session.rollback()
        raise

    return project
