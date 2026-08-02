from __future__ import annotations

from dataclasses import dataclass
import uuid


@dataclass(frozen=True, slots=True)
class BailianDemoSeedResult:
    seed_id: str
    created_new: bool
    user_id: uuid.UUID
    project_id: uuid.UUID
    schema_id: uuid.UUID
    schema_version_id: uuid.UUID
    orchestration_id: uuid.UUID
    extraction_run_id: uuid.UUID
    consistency_check_application_id: uuid.UUID
    project_version_id: uuid.UUID
    pending_review_fact_id: uuid.UUID
    resolved_fact_id: uuid.UUID
    observation_only_fact_id: uuid.UUID
    pending_review_subject_key: str
    version_record_subject_key: str
    reviewed_projection_manifest_hash: str
    knowledge_view_manifest_hash: str
    project_version_manifest_hash: str
