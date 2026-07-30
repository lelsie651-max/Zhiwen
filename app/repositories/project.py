import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.models.project import Project
from app.models.project_member import ProjectMember


async def list_projects_for_user(session: AsyncSession, user_id: uuid.UUID) -> list[Project]:
    result = await session.execute(
        select(Project)
        .join(ProjectMember, ProjectMember.project_id == Project.id)
        .where(ProjectMember.user_id == user_id)
        .order_by(Project.created_at.desc())
    )
    return list(result.scalars().all())


async def get_project_by_slug(session: AsyncSession, slug: str) -> Project | None:
    result = await session.execute(select(Project).where(Project.slug == slug))
    return result.scalar_one_or_none()


async def get_project_for_user_by_slug(
    session: AsyncSession,
    user_id: uuid.UUID,
    slug: str,
) -> Project | None:
    result = await session.execute(
        select(Project)
        .join(ProjectMember, ProjectMember.project_id == Project.id)
        .where(Project.slug == slug, ProjectMember.user_id == user_id)
    )
    return result.scalar_one_or_none()


async def get_project_member_for_user_by_slug(
    session: AsyncSession,
    user_id: uuid.UUID,
    slug: str,
) -> ProjectMember | None:
    result = await session.execute(
        select(ProjectMember)
        .join(Project, ProjectMember.project_id == Project.id)
        .where(Project.slug == slug, ProjectMember.user_id == user_id)
        .options(joinedload(ProjectMember.project))
    )
    return result.scalar_one_or_none()


async def create_project(session: AsyncSession, project: Project) -> Project:
    session.add(project)
    await session.flush()
    return project


async def create_project_member(
    session: AsyncSession,
    project_member: ProjectMember,
) -> ProjectMember:
    session.add(project_member)
    await session.flush()
    return project_member
