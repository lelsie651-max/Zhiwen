from __future__ import annotations

from dataclasses import dataclass
import uuid


@dataclass(frozen=True, slots=True)
class ConsistencyCheckWorkflowResult:
    project_id: uuid.UUID
    consistency_application_id: uuid.UUID
    plan_manifest_hash: str
    execution_result_manifest_hash: str
    consistency_check_application_id: uuid.UUID
    created_new: bool
    batch_count: int
    assessment_count: int
