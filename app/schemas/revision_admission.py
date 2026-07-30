from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict

from app.models.document_revision import DocumentRevisionStatus, SourceAuthority
from app.models.ingestion_validation import ValidationUserDecision


class RevisionAdmissionResult(BaseModel):
    revision_id: uuid.UUID
    report_id: uuid.UUID
    revision_status: DocumentRevisionStatus
    source_authority: SourceAuthority
    user_decision: ValidationUserDecision | None = None
    decision_required: bool

    model_config = ConfigDict(extra="forbid")


class RevisionAdmissionDecisionInput(BaseModel):
    decision: ValidationUserDecision

    model_config = ConfigDict(extra="forbid")
