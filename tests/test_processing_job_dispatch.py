import asyncio
import uuid

import pytest

from app.services import processing_job_dispatch as processing_job_dispatch_service


def run_async(awaitable):
    return asyncio.run(awaitable)


def test_dispatch_calls_executor(monkeypatch) -> None:
    called: dict[str, object] = {}

    async def fake_execute_revision_extraction_job(session_factory, *, job_id, storage):
        called["session_factory"] = session_factory
        called["job_id"] = job_id
        called["storage"] = storage
        return type("Result", (), {"execution_status": "completed"})()

    monkeypatch.setattr(
        processing_job_dispatch_service,
        "execute_revision_extraction_job",
        fake_execute_revision_extraction_job,
    )

    session_factory = object()
    storage = object()
    job_id = uuid.uuid4()

    run_async(
        processing_job_dispatch_service.dispatch_revision_extraction_job(
            job_id=job_id,
            session_factory=session_factory,  # type: ignore[arg-type]
            storage=storage,  # type: ignore[arg-type]
        )
    )

    assert called == {
        "session_factory": session_factory,
        "job_id": job_id,
        "storage": storage,
    }


def test_dispatch_swallows_ordinary_exception_with_safe_logging(monkeypatch, caplog) -> None:
    async def fake_execute_revision_extraction_job(session_factory, *, job_id, storage):
        raise RuntimeError("lease_token=secret storage_key=hidden body text")

    monkeypatch.setattr(
        processing_job_dispatch_service,
        "execute_revision_extraction_job",
        fake_execute_revision_extraction_job,
    )

    with caplog.at_level("WARNING"):
        run_async(
            processing_job_dispatch_service.dispatch_revision_extraction_job(
                job_id=uuid.uuid4(),
                session_factory=object(),  # type: ignore[arg-type]
                storage=object(),  # type: ignore[arg-type]
            )
        )

    assert "dispatch_failed" in caplog.text
    assert "lease_token" not in caplog.text
    assert "storage_key" not in caplog.text
    assert "body text" not in caplog.text


def test_dispatch_propagates_cancelled_error(monkeypatch) -> None:
    async def fake_execute_revision_extraction_job(session_factory, *, job_id, storage):
        raise asyncio.CancelledError

    monkeypatch.setattr(
        processing_job_dispatch_service,
        "execute_revision_extraction_job",
        fake_execute_revision_extraction_job,
    )

    with pytest.raises(asyncio.CancelledError):
        run_async(
            processing_job_dispatch_service.dispatch_revision_extraction_job(
                job_id=uuid.uuid4(),
                session_factory=object(),  # type: ignore[arg-type]
                storage=object(),  # type: ignore[arg-type]
            )
        )
