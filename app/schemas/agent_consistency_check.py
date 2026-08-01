"""Agent 2 consistency-check structured output contract."""

from __future__ import annotations

import math
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ConsistencyVerdict = Literal["conflict", "compatible", "insufficient_evidence"]
ConsistencySeverity = Literal["red", "yellow", "none"]
ConsistencyImpact = Literal[
    "data_quality_review",
    "downstream_consumer_review",
    "timeline_review",
    "scope_review",
    "unit_review",
    "entity_resolution_review",
]
ConsistencyRecommendedAction = Literal[
    "request_more_evidence",
    "normalize_units",
    "normalize_time_scope",
    "review_entity_resolution",
    "review_source_scope",
    "escalate_human_review",
    "leave_as_is",
]


class ConsistencyCheckAssessment(BaseModel):
    candidate_id: uuid.UUID
    verdict: ConsistencyVerdict
    severity: ConsistencySeverity
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=1, max_length=2000)
    cited_evidence_link_ids: list[uuid.UUID] = Field(min_length=1, max_length=200)
    impact: list[ConsistencyImpact] = Field(default_factory=list, max_length=20)
    recommended_actions: list[ConsistencyRecommendedAction] = Field(
        default_factory=list,
        max_length=20,
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence_must_be_finite_number(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("confidence must be a finite number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("confidence must be a finite number")
        return numeric

    @field_validator("explanation")
    @classmethod
    def _explanation_must_be_non_blank(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("explanation must be a non-empty string")
        return value.strip()

    @field_validator(
        "cited_evidence_link_ids",
        "impact",
        "recommended_actions",
        mode="before",
    )
    @classmethod
    def _lists_must_be_lists(cls, value: Any) -> Any:
        if not isinstance(value, list):
            raise ValueError("field must be a list")
        return value

    @model_validator(mode="after")
    def validate_assessment_semantics(self) -> "ConsistencyCheckAssessment":
        if self.verdict == "conflict" and self.severity not in {"red", "yellow"}:
            raise ValueError("conflict verdict requires red or yellow severity")
        if self.verdict in {"compatible", "insufficient_evidence"} and self.severity != "none":
            raise ValueError("non-conflict verdict requires none severity")
        cited_ids = list(self.cited_evidence_link_ids)
        if len(set(cited_ids)) != len(cited_ids):
            raise ValueError("cited_evidence_link_ids must be unique")
        return self


class ConsistencyCheckResponse(BaseModel):
    assessments: list[ConsistencyCheckAssessment] = Field(min_length=1, max_length=200)

    model_config = ConfigDict(extra="forbid")

    @field_validator("assessments", mode="before")
    @classmethod
    def _assessments_must_be_list(cls, value: Any) -> Any:
        if not isinstance(value, list):
            raise ValueError("assessments must be a list")
        return value

    @model_validator(mode="after")
    def validate_unique_candidates(self) -> "ConsistencyCheckResponse":
        candidate_ids = [assessment.candidate_id for assessment in self.assessments]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("assessments must contain unique candidate_id values")
        return self
