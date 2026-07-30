from typing import Any

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.dependencies.auth import get_optional_current_user
from app.models.user import User
from app.routers.web import templates
from app.schemas.user import UserCreate
from app.services import identity as identity_service
from app.services import project as project_service
from app.utils.csrf import ensure_csrf_token, verify_csrf_token


router = APIRouter(tags=["app"])


def normalize_validation_errors(exc: ValidationError) -> dict[str, str]:
    errors: dict[str, str] = {}
    for error in exc.errors():
        location = error.get("loc", [])
        field = str(location[-1]) if location else "form"
        errors[field] = error.get("msg", "输入不合法。")
    return errors


def base_setup_form() -> dict[str, str]:
    return {
        "handle": "",
        "display_name": "",
        "email": "",
        "locale": "zh-CN",
        "timezone": "Asia/Shanghai",
    }


def render_setup_template(
    request: Request,
    *,
    errors: dict[str, str] | None = None,
    form: dict[str, str] | None = None,
    status_code: int = 200,
) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="setup.html",
        context={
            "title": "首次身份设置",
            "errors": errors or {},
            "form": form or base_setup_form(),
            "csrf_token": ensure_csrf_token(request),
        },
        status_code=status_code,
    )


async def render_dashboard(
    request: Request,
    session: AsyncSession,
    current_user: User,
    *,
    errors: dict[str, str] | None = None,
    project_form: dict[str, str] | None = None,
    status_code: int = 200,
) -> HTMLResponse:
    projects = await project_service.list_projects_for_user(session, current_user.id)
    context = {
        "title": "织文工作区",
        "current_user": current_user,
        "projects": projects,
        "errors": errors or {},
        "csrf_token": ensure_csrf_token(request),
        "project_form": project_form
        or {
            "name": "",
            "slug": "",
            "description": "",
            "visibility": "private",
        },
    }
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context=context,
        status_code=status_code,
    )


@router.get("/setup", response_class=HTMLResponse)
async def setup_page(
    request: Request,
    current_user: User | None = Depends(get_optional_current_user),
) -> Response:
    if current_user is not None:
        return RedirectResponse(url="/app", status_code=303)

    return render_setup_template(request)


@router.post("/setup", response_class=HTMLResponse)
async def setup_submit(
    request: Request,
    handle: str = Form(...),
    display_name: str = Form(...),
    email: str = Form(""),
    locale: str = Form("zh-CN"),
    timezone: str = Form("Asia/Shanghai"),
    csrf_token: str | None = Form(None),
    current_user: User | None = Depends(get_optional_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    if current_user is not None:
        return RedirectResponse(url="/app", status_code=303)

    verify_csrf_token(request, csrf_token)

    form_data = {
        "handle": handle,
        "display_name": display_name,
        "email": email,
        "locale": locale,
        "timezone": timezone,
    }

    try:
        payload = UserCreate(**form_data)
        user = await identity_service.register_user(session, payload)
    except ValidationError as exc:
        return render_setup_template(
            request,
            errors=normalize_validation_errors(exc),
            form=form_data,
            status_code=400,
        )
    except identity_service.IdentityConflictError as exc:
        return render_setup_template(
            request,
            errors={exc.field: exc.message},
            form=form_data,
            status_code=400,
        )

    request.session["current_user_id"] = str(user.id)
    return RedirectResponse(url="/app", status_code=303)


@router.get("/app", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    current_user: User | None = Depends(get_optional_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    if current_user is None:
        return RedirectResponse(url="/setup", status_code=303)

    return await render_dashboard(request, session, current_user)


@router.post("/logout")
async def logout(
    request: Request,
    csrf_token: str | None = Form(None),
) -> Response:
    verify_csrf_token(request, csrf_token)
    request.session.clear()
    return RedirectResponse(url="/setup", status_code=303)
