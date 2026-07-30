from __future__ import annotations

from enum import StrEnum
import uuid

from pydantic import BaseModel, ConfigDict

from app.models.document_revision import DocumentRevisionStatus
from app.models.processing_job import ProcessingJobStatus


class ProcessingJobExecutionStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ProcessingJobExecutionResult(BaseModel):
    job_id: uuid.UUID
    revision_id: uuid.UUID
    execution_status: ProcessingJobExecutionStatus
    job_status: ProcessingJobStatus
    extraction_run_id: uuid.UUID | None = None
    revision_status: DocumentRevisionStatus | None = None
    failure_code: str | None = None
    skip_reason: str | None = None

    model_config = ConfigDict(extra="forbid")
