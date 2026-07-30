from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field

from app.models.document_revision import DocumentRevisionStatus
from app.schemas.document_extraction import ExtractionOutcome


class RevisionExtractionResult(BaseModel):
    revision_id: uuid.UUID
    extraction_run_id: uuid.UUID
    outcome: ExtractionOutcome
    revision_status: DocumentRevisionStatus
    block_count: int
    character_count: int
    page_count: int | None = None
    warnings: list[str] = Field(default_factory=list)
    requires_ocr: bool

    model_config = ConfigDict(extra="forbid")
