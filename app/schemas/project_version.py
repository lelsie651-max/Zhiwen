from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import uuid
from typing import Literal


ProjectVersionCreationKindLiteral = Literal[
    "manual",
    "automatic",
    "pre_publish",
    "rollback",
]
ProjectVersionKnowledgeAlgorithmName = Literal["dynamic_schema_knowledge_view"]
ProjectVersionKnowledgeAlgorithmVersion = Literal["1.0.0"]
ProjectVersionSnapshotFormatVersion = Literal["1.0.0"]


@dataclass(frozen=True, slots=True)
class ProjectVersionSnapshot:
    id: uuid.UUID
    project_id: uuid.UUID
    version_no: int
    created_by_id: uuid.UUID
    creation_kind: ProjectVersionCreationKindLiteral
    copied_from_version_id: uuid.UUID | None
    reason: str | None
    schema_id: uuid.UUID
    schema_version_id: uuid.UUID
    orchestration_id: uuid.UUID
    extraction_run_id: uuid.UUID
    consistency_check_application_id: uuid.UUID
    source_consistency_application_id: uuid.UUID
    schema_definition_manifest_hash: str
    ufl_source_manifest_hash: str
    consistency_result_manifest_hash: str
    raw_projection_manifest_hash: str
    reviewed_projection_manifest_hash: str
    knowledge_view_manifest_hash: str
    knowledge_view_algorithm_name: ProjectVersionKnowledgeAlgorithmName
    knowledge_view_algorithm_version: ProjectVersionKnowledgeAlgorithmVersion
    snapshot_format_version: ProjectVersionSnapshotFormatVersion
    snapshot_json: Mapping[str, object]
    snapshot_json_hash: str
    version_manifest_hash: str
    record_count: int
    section_count: int
    field_count: int
    missing_field_count: int
    review_required_field_count: int
    resolved_field_count: int
    observation_only_field_count: int
    mixed_field_count: int
    created_at: datetime
    is_current: bool


@dataclass(frozen=True, slots=True)
class ProjectVersionCreateResult(ProjectVersionSnapshot):
    created_new: bool
