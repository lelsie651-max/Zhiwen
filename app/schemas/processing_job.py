from __future__ import annotations

from datetime import datetime
import uuid

from pydantic import BaseModel, ConfigDict

from app.models.processing_job import ProcessingJobStatus, ProcessingJobType, ProcessingTriggerKind


class ProcessingJobRead(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    revision_id: uuid.UUID
    job_type: ProcessingJobType
    status: ProcessingJobStatus
    trigger_kind: ProcessingTriggerKind
    attempt_no: int
    requested_by_id: uuid.UUID | None = None
    lease_expires_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result_extraction_run_id: uuid.UUID | None = None
    failure_code: str | None = None
    failure_message: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, extra="forbid")
