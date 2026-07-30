import logging
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_async_session_factory, get_db_session
from app.dependencies.auth import get_optional_current_user
from app.models.document_revision import DocumentRevisionStatus, UploadIntent
from app.models.ingestion_validation import ValidationUserDecision
from app.models.processing_job import ProcessingJob, ProcessingJobStatus
from app.models.project_member import ProjectMemberRole
from app.models.user import User
from app.routers.app import normalize_validation_errors, render_dashboard
from app.routers.web import templates
from app.schemas.document_upload import DocumentUploadSubmit
from app.schemas.project import ProjectCreate
from app.schemas.revision_admission import RevisionAdmissionDecisionInput
from app.services import document_upload as document_upload_service
from app.services import file_ingestion as file_ingestion_service
from app.services import processing_job as processing_job_service
from app.services import revision_admission as revision_admission_service
from app.services import project as project_service
from app.services.processing_job_dispatch import dispatch_revision_extraction_job
from app.utils.csrf import ensure_csrf_token, verify_csrf_token


router = APIRouter(tags=["projects"])
logger = logging.getLogger(__name__)

PROCESSING_NOTICE_MESSAGES = {
    "enqueue_failed": "处理任务已保存到当前状态，但排队或后台调度未完成，请稍后重试。",
    "queued": "处理任务已进入队列。",
    "retry_queued": "重试任务已进入队列。",
    "recovery_queued": "恢复任务已进入队列。",
}
MANAGE_PROCESSING_ROLES = {
    ProjectMemberRole.OWNER.value,
    ProjectMemberRole.EDITOR.value,
}


def base_upload_form() -> dict[str, str]:
    return {
        "upload_intent": UploadIntent.NEW_DOCUMENT.value,
        "title": "",
        "description": "",
        "logical_order_kind": "",
        "logical_order_value": "",
        "logical_order_index": "",
        "effective_from": "",
        "effective_to": "",
        "document_id": "",
    }

async def render_upload_page(
    request: Request,
    session: AsyncSession,
    current_user: User,
    slug: str,
    *,
    errors: dict[str, str] | None = None,
    form: dict[str, str] | None = None,
    status_code: int = 200,
) -> HTMLResponse:
    try:
        context = await document_upload_service.get_project_upload_context(
            session,
            current_user.id,
            slug,
        )
    except document_upload_service.UploadProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found.") from exc

    if not context.can_upload:
        raise HTTPException(status_code=403, detail="Only owners and editors can upload documents.")

    return templates.TemplateResponse(
        request=request,
        name="document_upload.html",
        context={
            "title": f"上传文档 - {context.project.name}",
            "current_user": current_user,
            "project": context.project,
            "documents": context.documents,
            "errors": errors or {},
            "upload_form": form or base_upload_form(),
            "csrf_token": ensure_csrf_token(request),
        },
        status_code=status_code,
    )


def _build_upload_form_data(
    *,
    upload_intent: str,
    title: str,
    description: str,
    logical_order_kind: str,
    logical_order_value: str,
    logical_order_index: str,
    effective_from: str,
    effective_to: str,
    document_id: str,
) -> dict[str, str]:
    return {
        "upload_intent": upload_intent,
        "title": title,
        "description": description,
        "logical_order_kind": logical_order_kind,
        "logical_order_value": logical_order_value,
        "logical_order_index": logical_order_index,
        "effective_from": effective_from,
        "effective_to": effective_to,
        "document_id": document_id,
    }


async def iter_upload_file_chunks(upload_file: UploadFile, chunk_size: int) -> AsyncIterator[bytes]:
    while True:
        chunk = await upload_file.read(chunk_size)
        if not chunk:
            break
        yield chunk


def build_revision_redirect_url(
    *,
    project_slug: str,
    document_id: uuid.UUID,
    revision_id: uuid.UUID,
    processing_notice: str | None = None,
) -> str:
    base_url = f"/projects/{project_slug}/documents/{document_id}/revisions/{revision_id}"
    if processing_notice is None:
        return base_url
    return f"{base_url}?{urlencode({'processing_notice': processing_notice})}"


def _schedule_processing_job(
    *,
    background_tasks: BackgroundTasks,
    scheduled_job_ids: set[uuid.UUID],
    job_id: uuid.UUID,
    storage,
) -> bool:
    if job_id in scheduled_job_ids:
        return False
    background_tasks.add_task(
        dispatch_revision_extraction_job,
        job_id=job_id,
        session_factory=get_async_session_factory(),
        storage=storage,
    )
    scheduled_job_ids.add(job_id)
    return True


def schedule_processing_job_once(
    *,
    background_tasks: BackgroundTasks,
    schedule_job_ids: set[uuid.UUID],
    job_id: uuid.UUID,
    session_factory,
    storage,
) -> bool:
    return _schedule_processing_job(
        background_tasks=background_tasks,
        scheduled_job_ids=schedule_job_ids,
        job_id=job_id,
        storage=storage,
    )


def _build_processing_notice_message(processing_notice: str | None) -> str | None:
    if processing_notice is None:
        return None
    return PROCESSING_NOTICE_MESSAGES.get(processing_notice)


def _build_stale_before() -> datetime:
    settings = get_settings()
    return datetime.now(timezone.utc) - timedelta(minutes=settings.processing_stale_minutes)


def _latest_running_job_is_expired(latest_processing_job: ProcessingJob | None, *, now: datetime) -> bool:
    return bool(
        latest_processing_job is not None
        and latest_processing_job.status == ProcessingJobStatus.RUNNING.value
        and latest_processing_job.lease_expires_at is not None
        and latest_processing_job.lease_expires_at <= now
    )


def _show_recover_action(
    *,
    revision_status: str,
    revision_updated_at: datetime,
    latest_processing_job: ProcessingJob | None,
    stale_before: datetime,
    now: datetime,
) -> bool:
    if _latest_running_job_is_expired(latest_processing_job, now=now):
        return True
    return revision_status in {
        DocumentRevisionStatus.PARSING.value,
        DocumentRevisionStatus.EXTRACTING.value,
    } and revision_updated_at <= stale_before


async def _get_revision_detail_context_or_404(
    *,
    session: AsyncSession,
    current_user: User,
    slug: str,
    document_id: uuid.UUID,
    revision_id: uuid.UUID,
) -> document_upload_service.RevisionDetailContext:
    try:
        return await document_upload_service.get_revision_detail_for_user(
            session,
            current_user.id,
            slug,
            document_id,
            revision_id,
        )
    except document_upload_service.UploadProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found.") from exc
    except document_upload_service.UploadRevisionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Revision not found.") from exc


def _require_processing_role(context: document_upload_service.RevisionDetailContext) -> None:
    if not context.can_manage_processing:
        raise HTTPException(status_code=403, detail="Only owners and editors can manage revision processing.")


def _redirect_to_revision_detail(
    *,
    context: document_upload_service.RevisionDetailContext,
    processing_notice: str | None = None,
) -> RedirectResponse:
    return RedirectResponse(
        url=build_revision_redirect_url(
            project_slug=context.project.slug,
            document_id=context.revision.document.id,
            revision_id=context.revision.id,
            processing_notice=processing_notice,
        ),
        status_code=303,
    )


def _schedule_manual_processing_job(
    *,
    background_tasks: BackgroundTasks,
    schedule_job_ids: set[uuid.UUID],
    job: ProcessingJob,
    storage,
) -> str | None:
    if job.status == ProcessingJobStatus.RUNNING.value:
        return None
    _schedule_processing_job(
        background_tasks=background_tasks,
        scheduled_job_ids=schedule_job_ids,
        job_id=job.id,
        storage=storage,
    )
    return None


def _log_safe_processing_failure(*, revision_id: uuid.UUID, status: str) -> None:
    logger.warning("Revision processing action failed safely: revision_id=%s status=%s", revision_id, status)


async def handle_post_upload_processing(
    *,
    session: AsyncSession,
    background_tasks: BackgroundTasks,
    schedule_job_ids: set[uuid.UUID],
    project_id: uuid.UUID,
    revision_id: uuid.UUID,
    current_revision_status: str,
    storage,
) -> str | None:
    if current_revision_status in {
        DocumentRevisionStatus.REJECTED.value,
        DocumentRevisionStatus.QUARANTINED.value,
    }:
        return None

    try:
        admission = await revision_admission_service.resolve_revision_admission(
            session,
            project_id=project_id,
            revision_id=revision_id,
        )
        if admission.revision_status != DocumentRevisionStatus.ACCEPTED.value:
            return None

        job = await processing_job_service.enqueue_revision_extraction_job(
            session,
            project_id=project_id,
            revision_id=revision_id,
            trigger_kind=processing_job_service.ProcessingTriggerKind.AUTOMATIC,
            actor_id=None,
        )
        _schedule_processing_job(
            background_tasks=background_tasks,
            scheduled_job_ids=schedule_job_ids,
            job_id=job.id,
            storage=storage,
        )
        return None
    except Exception:
        _log_safe_processing_failure(revision_id=revision_id, status="enqueue_failed")
        return "enqueue_failed"


@router.post("/projects", response_class=HTMLResponse)
async def create_project(
    request: Request,
    name: str = Form(...),
    slug: str = Form(...),
    description: str = Form(""),
    visibility: str = Form("private"),
    csrf_token: str | None = Form(None),
    current_user: User | None = Depends(get_optional_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    if current_user is None:
        return RedirectResponse(url="/setup", status_code=303)

    verify_csrf_token(request, csrf_token)

    form_data = {
        "name": name,
        "slug": slug,
        "description": description,
        "visibility": visibility,
    }

    try:
        payload = ProjectCreate(**form_data)
        project = await project_service.create_project_for_owner(
            session,
            current_user.id,
            payload,
        )
    except ValidationError as exc:
        return await render_dashboard(
            request,
            session,
            current_user,
            errors=normalize_validation_errors(exc),
            project_form=form_data,
            status_code=400,
        )
    except project_service.ProjectConflictError as exc:
        return await render_dashboard(
            request,
            session,
            current_user,
            errors={exc.field: exc.message},
            project_form=form_data,
            status_code=400,
        )

    return RedirectResponse(url=f"/projects/{project.slug}", status_code=303)


@router.get("/projects/{slug}", response_class=HTMLResponse)
async def project_detail(
    request: Request,
    slug: str,
    current_user: User | None = Depends(get_optional_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    if current_user is None:
        return RedirectResponse(url="/setup", status_code=303)

    try:
        context, document_summaries = await document_upload_service.get_project_workspace_context(
            session,
            current_user.id,
            slug,
        )
    except document_upload_service.UploadProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found.") from exc

    return templates.TemplateResponse(
        request=request,
        name="project_detail.html",
        context={
            "title": context.project.name,
            "current_user": current_user,
            "project": context.project,
            "document_summaries": document_summaries,
            "can_upload": context.can_upload,
        },
    )


@router.get("/projects/{slug}/documents/upload", response_class=HTMLResponse)
async def document_upload_page(
    request: Request,
    slug: str,
    current_user: User | None = Depends(get_optional_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    if current_user is None:
        return RedirectResponse(url="/setup", status_code=303)

    return await render_upload_page(request, session, current_user, slug)


@router.post("/projects/{slug}/documents/upload", response_class=HTMLResponse)
async def document_upload_submit(
    request: Request,
    background_tasks: BackgroundTasks,
    slug: str,
    upload_intent: str = Form(...),
    title: str = Form(""),
    description: str = Form(""),
    logical_order_kind: str = Form(""),
    logical_order_value: str = Form(""),
    logical_order_index: str = Form(""),
    effective_from: str = Form(""),
    effective_to: str = Form(""),
    document_id: str = Form(""),
    file: UploadFile | None = File(None),
    csrf_token: str | None = Form(None),
    current_user: User | None = Depends(get_optional_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    if current_user is None:
        return RedirectResponse(url="/setup", status_code=303)

    verify_csrf_token(request, csrf_token)

    form_data = _build_upload_form_data(
        upload_intent=upload_intent,
        title=title,
        description=description,
        logical_order_kind=logical_order_kind,
        logical_order_value=logical_order_value,
        logical_order_index=logical_order_index,
        effective_from=effective_from,
        effective_to=effective_to,
        document_id=document_id,
    )
    if file is None:
        return await render_upload_page(
            request,
            session,
            current_user,
            slug,
            errors={"file": "请选择要上传的文件。"},
            form=form_data,
            status_code=400,
        )

    try:
        storage_backend = file_ingestion_service.get_local_file_storage()
        if not file.filename:
            return await render_upload_page(
                request,
                session,
                current_user,
                slug,
                errors={"file": "请选择要上传的文件。"},
                form=form_data,
                status_code=400,
            )

        try:
            payload = DocumentUploadSubmit(**form_data)
            settings = get_settings()
            result = await document_upload_service.upload_document_for_project(
                session,
                user_id=current_user.id,
                project_slug=slug,
                payload=payload,
                chunks=iter_upload_file_chunks(file, settings.upload_chunk_bytes),
                original_filename=file.filename,
                declared_mime=file.content_type,
                storage=storage_backend,
            )
        except ValidationError as exc:
            return await render_upload_page(
                request,
                session,
                current_user,
                slug,
                errors=normalize_validation_errors(exc),
                form=form_data,
                status_code=400,
            )
        except document_upload_service.UploadProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Project not found.") from exc
        except document_upload_service.UploadAccessError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except document_upload_service.UploadFormError as exc:
            return await render_upload_page(
                request,
                session,
                current_user,
                slug,
                errors={exc.field: exc.message},
                form=form_data,
                status_code=400,
            )
        except (
            file_ingestion_service.FileIngestionError,
            file_ingestion_service.TechnicalPreflightError,
        ) as exc:
            return await render_upload_page(
                request,
                session,
                current_user,
                slug,
                errors={"file": str(exc)},
                form=form_data,
                status_code=400,
            )
    finally:
        await file.close()

    schedule_job_ids: set[uuid.UUID] = set()
    processing_notice = await handle_post_upload_processing(
        session=session,
        background_tasks=background_tasks,
        schedule_job_ids=schedule_job_ids,
        project_id=result.project.id,
        revision_id=result.revision.id,
        current_revision_status=result.revision.status,
        storage=storage_backend,
    )

    return RedirectResponse(
        url=build_revision_redirect_url(
            project_slug=result.project.slug,
            document_id=result.document.id,
            revision_id=result.revision.id,
            processing_notice=processing_notice,
        ),
        status_code=303,
    )


@router.get(
    "/projects/{slug}/documents/{document_id}/revisions/{revision_id}",
    response_class=HTMLResponse,
)
async def revision_detail(
    request: Request,
    slug: str,
    document_id: uuid.UUID,
    revision_id: uuid.UUID,
    current_user: User | None = Depends(get_optional_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    if current_user is None:
        return RedirectResponse(url="/setup", status_code=303)

    context = await _get_revision_detail_context_or_404(
        session=session,
        current_user=current_user,
        slug=slug,
        document_id=document_id,
        revision_id=revision_id,
    )
    now = datetime.now(timezone.utc)
    stale_before = _build_stale_before()
    latest_processing_job = context.latest_processing_job
    running_job_active = (
        latest_processing_job is not None
        and latest_processing_job.status == ProcessingJobStatus.RUNNING.value
        and not _latest_running_job_is_expired(latest_processing_job, now=now)
    )
    can_manage_processing = context.can_manage_processing
    processing_notice_message = _build_processing_notice_message(
        request.query_params.get("processing_notice")
    )
    show_recover_processing = can_manage_processing and _show_recover_action(
        revision_status=context.revision.status,
        revision_updated_at=context.revision.updated_at,
        latest_processing_job=latest_processing_job,
        stale_before=stale_before,
        now=now,
    )

    return templates.TemplateResponse(
        request=request,
        name="revision_detail.html",
        context={
            "title": f"{context.revision.document.title} · 修订详情",
            "current_user": current_user,
            "project": context.project,
            "document": context.revision.document,
            "revision": context.revision,
            "report": context.report,
            "risk_flags": context.risk_flags,
            "latest_processing_job": latest_processing_job,
            "latest_extraction_run": context.latest_extraction_run,
            "sha256_short": context.revision.sha256[:12],
            "csrf_token": ensure_csrf_token(request),
            "processing_notice_message": processing_notice_message,
            "can_manage_processing": can_manage_processing,
            "show_admission_actions": can_manage_processing
            and context.revision.status == DocumentRevisionStatus.NEEDS_CONFIRMATION.value,
            "show_start_processing": can_manage_processing
            and context.revision.status == DocumentRevisionStatus.ACCEPTED.value
            and not running_job_active,
            "show_retry_processing": can_manage_processing
            and context.revision.status == DocumentRevisionStatus.FAILED.value,
            "show_recover_processing": show_recover_processing,
        },
    )


@router.post(
    "/projects/{slug}/documents/{document_id}/revisions/{revision_id}/admission",
    response_class=HTMLResponse,
)
async def revision_admission_submit(
    request: Request,
    background_tasks: BackgroundTasks,
    slug: str,
    document_id: uuid.UUID,
    revision_id: uuid.UUID,
    decision: str = Form(...),
    csrf_token: str | None = Form(None),
    current_user: User | None = Depends(get_optional_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    if current_user is None:
        return RedirectResponse(url="/setup", status_code=303)

    verify_csrf_token(request, csrf_token)
    context = await _get_revision_detail_context_or_404(
        session=session,
        current_user=current_user,
        slug=slug,
        document_id=document_id,
        revision_id=revision_id,
    )
    _require_processing_role(context)
    try:
        payload = RevisionAdmissionDecisionInput(decision=decision)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail="Invalid admission decision.") from exc

    try:
        result = await revision_admission_service.decide_revision_admission(
            session,
            project_id=context.project.id,
            actor_id=current_user.id,
            revision_id=context.revision.id,
            payload=payload,
        )
    except revision_admission_service.RevisionAdmissionPermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except revision_admission_service.RevisionAdmissionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Revision not found.") from exc
    except (
        revision_admission_service.RevisionAdmissionStateError,
        revision_admission_service.RevisionAdmissionStateCorruptionError,
    ) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if result.user_decision == ValidationUserDecision.CANCEL:
        return _redirect_to_revision_detail(context=context)

    schedule_job_ids: set[uuid.UUID] = set()
    storage_backend = file_ingestion_service.get_local_file_storage()
    try:
        job = await processing_job_service.enqueue_revision_extraction_job(
            session,
            project_id=context.project.id,
            revision_id=context.revision.id,
            trigger_kind=processing_job_service.ProcessingTriggerKind.MANUAL,
            actor_id=current_user.id,
        )
        if job.status != ProcessingJobStatus.RUNNING.value:
            _schedule_processing_job(
                background_tasks=background_tasks,
                scheduled_job_ids=schedule_job_ids,
                job_id=job.id,
                storage=storage_backend,
            )
    except Exception:
        _log_safe_processing_failure(revision_id=context.revision.id, status="enqueue_failed")
        return _redirect_to_revision_detail(context=context, processing_notice="enqueue_failed")
    return _redirect_to_revision_detail(
        context=context,
        processing_notice=None if job.status == ProcessingJobStatus.RUNNING.value else "queued",
    )


@router.post(
    "/projects/{slug}/documents/{document_id}/revisions/{revision_id}/processing/start",
    response_class=HTMLResponse,
)
async def revision_processing_start(
    request: Request,
    background_tasks: BackgroundTasks,
    slug: str,
    document_id: uuid.UUID,
    revision_id: uuid.UUID,
    csrf_token: str | None = Form(None),
    current_user: User | None = Depends(get_optional_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    if current_user is None:
        return RedirectResponse(url="/setup", status_code=303)

    verify_csrf_token(request, csrf_token)
    context = await _get_revision_detail_context_or_404(
        session=session,
        current_user=current_user,
        slug=slug,
        document_id=document_id,
        revision_id=revision_id,
    )
    _require_processing_role(context)

    try:
        job = await processing_job_service.enqueue_revision_extraction_job(
            session,
            project_id=context.project.id,
            revision_id=context.revision.id,
            trigger_kind=processing_job_service.ProcessingTriggerKind.MANUAL,
            actor_id=current_user.id,
        )
    except processing_job_service.ProcessingJobPermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except processing_job_service.ProcessingJobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except processing_job_service.ProcessingJobStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if job.status == ProcessingJobStatus.RUNNING.value:
        return _redirect_to_revision_detail(context=context)

    try:
        _schedule_processing_job(
            background_tasks=background_tasks,
            scheduled_job_ids=set(),
            job_id=job.id,
            storage=file_ingestion_service.get_local_file_storage(),
        )
    except Exception:
        _log_safe_processing_failure(revision_id=context.revision.id, status="enqueue_failed")
        return _redirect_to_revision_detail(context=context, processing_notice="enqueue_failed")
    return _redirect_to_revision_detail(context=context, processing_notice="queued")


@router.post(
    "/projects/{slug}/documents/{document_id}/revisions/{revision_id}/processing/retry",
    response_class=HTMLResponse,
)
async def revision_processing_retry(
    request: Request,
    background_tasks: BackgroundTasks,
    slug: str,
    document_id: uuid.UUID,
    revision_id: uuid.UUID,
    csrf_token: str | None = Form(None),
    current_user: User | None = Depends(get_optional_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    if current_user is None:
        return RedirectResponse(url="/setup", status_code=303)

    verify_csrf_token(request, csrf_token)
    context = await _get_revision_detail_context_or_404(
        session=session,
        current_user=current_user,
        slug=slug,
        document_id=document_id,
        revision_id=revision_id,
    )
    _require_processing_role(context)

    try:
        job = await processing_job_service.retry_failed_revision_extraction(
            session,
            project_id=context.project.id,
            revision_id=context.revision.id,
            actor_id=current_user.id,
        )
    except processing_job_service.ProcessingJobPermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except processing_job_service.ProcessingJobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except processing_job_service.ProcessingJobStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    try:
        _schedule_processing_job(
            background_tasks=background_tasks,
            scheduled_job_ids=set(),
            job_id=job.id,
            storage=file_ingestion_service.get_local_file_storage(),
        )
    except Exception:
        _log_safe_processing_failure(revision_id=context.revision.id, status="enqueue_failed")
        return _redirect_to_revision_detail(context=context, processing_notice="enqueue_failed")
    return _redirect_to_revision_detail(context=context, processing_notice="retry_queued")


@router.post(
    "/projects/{slug}/documents/{document_id}/revisions/{revision_id}/processing/recover",
    response_class=HTMLResponse,
)
async def revision_processing_recover(
    request: Request,
    background_tasks: BackgroundTasks,
    slug: str,
    document_id: uuid.UUID,
    revision_id: uuid.UUID,
    csrf_token: str | None = Form(None),
    current_user: User | None = Depends(get_optional_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    if current_user is None:
        return RedirectResponse(url="/setup", status_code=303)

    verify_csrf_token(request, csrf_token)
    context = await _get_revision_detail_context_or_404(
        session=session,
        current_user=current_user,
        slug=slug,
        document_id=document_id,
        revision_id=revision_id,
    )
    _require_processing_role(context)

    try:
        job = await processing_job_service.recover_stale_revision_extraction(
            session,
            project_id=context.project.id,
            revision_id=context.revision.id,
            actor_id=current_user.id,
            stale_before=_build_stale_before(),
        )
    except processing_job_service.ProcessingJobPermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except processing_job_service.ProcessingJobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except processing_job_service.ProcessingJobStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    try:
        _schedule_processing_job(
            background_tasks=background_tasks,
            scheduled_job_ids=set(),
            job_id=job.id,
            storage=file_ingestion_service.get_local_file_storage(),
        )
    except Exception:
        _log_safe_processing_failure(revision_id=context.revision.id, status="enqueue_failed")
        return _redirect_to_revision_detail(context=context, processing_notice="enqueue_failed")
    return _redirect_to_revision_detail(context=context, processing_notice="recovery_queued")
