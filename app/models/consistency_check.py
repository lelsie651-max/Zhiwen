from __future__ import annotations

from datetime import datetime
import json
import math
import re
import uuid
from typing import TYPE_CHECKING, get_args

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKeyConstraint,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.models.base import Base, UUIDPrimaryKeyMixin, utc_now
from app.schemas.agent_consistency_check import (
    ConsistencyImpact,
    ConsistencyRecommendedAction,
    ConsistencySeverity,
    ConsistencyVerdict,
)
from app.utils.validation import normalize_text

if TYPE_CHECKING:
    from app.models.consistency_review import ConsistencyReviewDecision
    from app.models.fact import FactEvidenceLink
    from app.models.fact_extraction_orchestration import FactExtractionOrchestration
    from app.models.fact_value_duplicate_grouping import (
        FactValueConsistencyCandidate,
        FactValueConsistencyCandidateApplication,
        FactValueConsistencyCandidateMember,
    )
    from app.models.inference import InferenceInputBatch, InferenceRun
    from app.models.project import Project


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_VERDICTS = tuple(get_args(ConsistencyVerdict))
_ALLOWED_SEVERITIES = tuple(get_args(ConsistencySeverity))
_ALLOWED_IMPACTS = tuple(get_args(ConsistencyImpact))
_ALLOWED_ACTIONS = tuple(get_args(ConsistencyRecommendedAction))


def _sql_string_tuple(values: tuple[str, ...]) -> str:
    return f"({', '.join(repr(value) for value in values)})"


def _sql_jsonb_array(values: tuple[str, ...]) -> str:
    return f"'{json.dumps(list(values), separators=(',', ':'), ensure_ascii=True)}'::jsonb"


_VERDICT_SQL = _sql_string_tuple(_ALLOWED_VERDICTS)
_SEVERITY_SQL = _sql_string_tuple(_ALLOWED_SEVERITIES)
_ALLOWED_IMPACTS_JSONB_SQL = _sql_jsonb_array(_ALLOWED_IMPACTS)
_ALLOWED_ACTIONS_JSONB_SQL = _sql_jsonb_array(_ALLOWED_ACTIONS)
_BATCH_SHAPE_SQL = (
    "("
    "skipped_empty = true "
    "AND input_batch_id IS NULL "
    "AND inference_run_id IS NULL "
    "AND request_hash IS NULL "
    "AND message_content_hash IS NULL"
    ") OR ("
    "skipped_empty = false "
    "AND input_batch_id IS NOT NULL "
    "AND inference_run_id IS NOT NULL "
    "AND request_hash IS NOT NULL "
    "AND message_content_hash IS NOT NULL"
    ")"
)
_VERDICT_SEVERITY_SQL = (
    "("
    "verdict = 'conflict' AND severity IN ('red', 'yellow')"
    ") OR ("
    "verdict IN ('compatible', 'insufficient_evidence') AND severity = 'none'"
    ")"
)
_CONFIDENCE_SQL = (
    "confidence >= 0.0 "
    "AND confidence <= 1.0 "
    "AND CAST(confidence AS TEXT) NOT IN ('NaN', 'Infinity', '-Infinity')"
)
_IMPACT_JSON_SQL = (
    "CASE WHEN jsonb_typeof(impact_json) = 'array' "
    f"THEN jsonb_array_length(impact_json) <= 20 AND impact_json <@ {_ALLOWED_IMPACTS_JSONB_SQL} "
    "ELSE FALSE END"
)
_ACTIONS_JSON_SQL = (
    "CASE WHEN jsonb_typeof(recommended_actions_json) = 'array' "
    f"THEN jsonb_array_length(recommended_actions_json) <= 20 "
    f"AND recommended_actions_json <@ {_ALLOWED_ACTIONS_JSONB_SQL} "
    "ELSE FALSE END"
)


def _normalize_short_text(value: str, *, field_name: str, max_length: int) -> str:
    normalized = normalize_text(value)
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if len(normalized) > max_length:
        raise ValueError(f"{field_name} must be at most {max_length} characters")
    return normalized


def _normalize_hash(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.lower()
    if not SHA256_PATTERN.fullmatch(normalized):
        raise ValueError(f"{field_name} must be a 64-character lowercase hexadecimal string")
    return normalized


def _normalize_enum_value(
    value: str,
    *,
    field_name: str,
    max_length: int,
    allowed_values: tuple[str, ...],
) -> str:
    normalized = _normalize_short_text(value, field_name=field_name, max_length=max_length)
    if normalized not in allowed_values:
        raise ValueError(f"{field_name} must be one of: {', '.join(allowed_values)}")
    return normalized


def _validate_string_list(
    value: list[str],
    *,
    field_name: str,
    allowed_values: tuple[str, ...],
) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    if len(value) > 20:
        raise ValueError(f"{field_name} must contain at most 20 items")

    normalized_values: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} items must be strings")
        normalized = normalize_text(item)
        if not normalized:
            raise ValueError(f"{field_name} items must not be empty")
        if normalized not in allowed_values:
            raise ValueError(f"{field_name} items must be one of: {', '.join(allowed_values)}")
        normalized_values.append(normalized)

    return normalized_values


class ConsistencyCheckApplication(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "consistency_check_applications"
    __table_args__ = (
        ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name="fk_ccapp_project_id_projects",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["consistency_application_id", "orchestration_id"],
            [
                "fact_value_consistency_candidate_applications.id",
                "fact_value_consistency_candidate_applications.orchestration_id",
            ],
            name="fk_ccapp_srcapp_orch_fvcca",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["orchestration_id", "project_id"],
            [
                "fact_extraction_orchestrations.id",
                "fact_extraction_orchestrations.project_id",
            ],
            name="fk_ccapp_orch_project_feo",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "execution_identity_hash",
            name="uq_ccapp_exec_identity_hash",
        ),
        UniqueConstraint(
            "id",
            "project_id",
            name="uq_ccapp_id_project",
        ),
        UniqueConstraint(
            "id",
            "consistency_application_id",
            name="uq_ccapp_id_srcapp",
        ),
        UniqueConstraint(
            "id",
            "project_id",
            "orchestration_id",
            "consistency_application_id",
            name="uq_ccapp_id_proj_orch_src",
        ),
        CheckConstraint(
            "source_result_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="src_result_hash_fmt",
        ),
        CheckConstraint(
            "plan_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="plan_manifest_hash_fmt",
        ),
        CheckConstraint(
            "execution_identity_hash ~ '^[0-9a-f]{64}$'",
            name="exec_identity_hash_fmt",
        ),
        CheckConstraint(
            "result_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="result_manifest_hash_fmt",
        ),
        CheckConstraint(
            "prompt_contract_hash ~ '^[0-9a-f]{64}$'",
            name="prompt_contract_hash_fmt",
        ),
        CheckConstraint("char_length(provider) BETWEEN 1 AND 128", name="provider_len"),
        CheckConstraint(
            "char_length(requested_model) BETWEEN 1 AND 128",
            name="requested_model_len",
        ),
        CheckConstraint(
            "char_length(executor_name) BETWEEN 1 AND 64",
            name="executor_name_len",
        ),
        CheckConstraint(
            "char_length(executor_version) BETWEEN 1 AND 32",
            name="executor_version_len",
        ),
        CheckConstraint("batch_count >= 0", name="batch_count_nn"),
        CheckConstraint(
            "executed_batch_count >= 0",
            name="executed_count_nn",
        ),
        CheckConstraint(
            "skipped_empty_batch_count >= 0",
            name="skipped_count_nn",
        ),
        CheckConstraint(
            "inference_run_count >= 0",
            name="run_count_nn",
        ),
        CheckConstraint("assessment_count >= 0", name="assessment_count_nn"),
        CheckConstraint(
            "executed_batch_count = batch_count",
            name="executed_eq_batch",
        ),
        CheckConstraint(
            "inference_run_count + skipped_empty_batch_count = batch_count",
            name="run_skipped_eq_batch",
        ),
    )

    project_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    consistency_application_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    orchestration_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    source_result_manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    plan_manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    execution_identity_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    result_manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_contract_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str] = mapped_column(String(128), nullable=False)
    requested_model: Mapped[str] = mapped_column(String(128), nullable=False)
    executor_name: Mapped[str] = mapped_column(String(64), nullable=False)
    executor_version: Mapped[str] = mapped_column(String(32), nullable=False)
    batch_count: Mapped[int] = mapped_column(Integer, nullable=False)
    executed_batch_count: Mapped[int] = mapped_column(Integer, nullable=False)
    skipped_empty_batch_count: Mapped[int] = mapped_column(Integer, nullable=False)
    inference_run_count: Mapped[int] = mapped_column(Integer, nullable=False)
    assessment_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    project: Mapped["Project"] = relationship(foreign_keys=[project_id])
    orchestration: Mapped["FactExtractionOrchestration"] = relationship(
        foreign_keys=[orchestration_id, project_id],
        overlaps="project,source_consistency_application",
    )
    source_consistency_application: Mapped["FactValueConsistencyCandidateApplication"] = relationship(
        foreign_keys=[consistency_application_id, orchestration_id],
        overlaps="orchestration",
    )
    batches: Mapped[list["ConsistencyCheckBatchLedger"]] = relationship(
        back_populates="application",
        foreign_keys="ConsistencyCheckBatchLedger.consistency_check_application_id",
        order_by="ConsistencyCheckBatchLedger.batch_index",
    )
    assessments: Mapped[list["ConsistencyAssessmentLedger"]] = relationship(
        back_populates="application",
        foreign_keys="ConsistencyAssessmentLedger.consistency_check_application_id",
        order_by=(
            "ConsistencyAssessmentLedger.batch_index, "
            "ConsistencyAssessmentLedger.source_consistency_candidate_id"
        ),
    )

    @validates(
        "source_result_manifest_hash",
        "plan_manifest_hash",
        "execution_identity_hash",
        "result_manifest_hash",
        "prompt_contract_hash",
    )
    def validate_hash(self, key: str, value: str) -> str:
        return _normalize_hash(value, field_name=key)

    @validates("provider")
    def validate_provider(self, _key: str, value: str) -> str:
        return _normalize_short_text(value, field_name="provider", max_length=128)

    @validates("requested_model")
    def validate_requested_model(self, _key: str, value: str) -> str:
        return _normalize_short_text(value, field_name="requested_model", max_length=128)

    @validates("executor_name")
    def validate_executor_name(self, _key: str, value: str) -> str:
        return _normalize_short_text(value, field_name="executor_name", max_length=64)

    @validates("executor_version")
    def validate_executor_version(self, _key: str, value: str) -> str:
        return _normalize_short_text(value, field_name="executor_version", max_length=32)


class ConsistencyCheckBatchLedger(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "consistency_check_batches"
    __table_args__ = (
        ForeignKeyConstraint(
            ["consistency_check_application_id"],
            ["consistency_check_applications.id"],
            name="fk_ccbatch_app_id_ccapp",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["input_batch_id"],
            ["inference_input_batches.id"],
            name="fk_ccbatch_input_batch_id_iib",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["inference_run_id", "input_batch_id"],
            ["inference_runs.id", "inference_runs.input_batch_id"],
            name="fk_ccbatch_run_input_ir",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "consistency_check_application_id",
            "batch_index",
            name="uq_ccbatch_app_batch_index",
        ),
        UniqueConstraint(
            "consistency_check_application_id",
            "inference_run_id",
            name="uq_ccbatch_app_inference_run_id",
        ),
        CheckConstraint("batch_index >= 0", name="batch_index_nn"),
        CheckConstraint("batch_manifest_hash ~ '^[0-9a-f]{64}$'", name="batch_manifest_hash_fmt"),
        CheckConstraint(
            "request_hash IS NULL OR request_hash ~ '^[0-9a-f]{64}$'",
            name="request_hash_fmt",
        ),
        CheckConstraint(
            "message_content_hash IS NULL OR message_content_hash ~ '^[0-9a-f]{64}$'",
            name="message_content_hash_fmt",
        ),
        CheckConstraint(_BATCH_SHAPE_SQL, name="batch_shape_valid"),
    )

    consistency_check_application_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    batch_index: Mapped[int] = mapped_column(Integer, nullable=False)
    batch_manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    skipped_empty: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    input_batch_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    inference_run_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    request_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    message_content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    application: Mapped["ConsistencyCheckApplication"] = relationship(
        back_populates="batches",
        foreign_keys=[consistency_check_application_id],
    )
    input_batch: Mapped["InferenceInputBatch | None"] = relationship(foreign_keys=[input_batch_id])
    inference_run: Mapped["InferenceRun | None"] = relationship(
        foreign_keys=[inference_run_id, input_batch_id],
        overlaps="input_batch",
    )
    assessments: Mapped[list["ConsistencyAssessmentLedger"]] = relationship(
        back_populates="batch",
        foreign_keys="[ConsistencyAssessmentLedger.consistency_check_application_id, ConsistencyAssessmentLedger.batch_index]",
        order_by="ConsistencyAssessmentLedger.source_consistency_candidate_id",
        overlaps="application,assessments",
    )

    @validates("batch_manifest_hash")
    def validate_batch_manifest_hash(self, _key: str, value: str) -> str:
        return _normalize_hash(value, field_name="batch_manifest_hash")

    @validates("request_hash", "message_content_hash")
    def validate_optional_hash(self, key: str, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalize_hash(value, field_name=key)


class ConsistencyAssessmentLedger(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "consistency_assessments"
    __table_args__ = (
        ForeignKeyConstraint(
            ["consistency_check_application_id", "source_consistency_application_id"],
            [
                "consistency_check_applications.id",
                "consistency_check_applications.consistency_application_id",
            ],
            name="fk_ccasmt_app_srcapp_ccapp",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["source_consistency_candidate_id", "source_consistency_application_id"],
            [
                "fact_value_consistency_candidates.id",
                "fact_value_consistency_candidates.consistency_application_id",
            ],
            name="fk_ccasmt_candidate_srcapp_fvcc",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["consistency_check_application_id", "batch_index"],
            [
                "consistency_check_batches.consistency_check_application_id",
                "consistency_check_batches.batch_index",
            ],
            name="fk_ccasmt_app_batch_index_ccbatch",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "consistency_check_application_id",
            "source_consistency_candidate_id",
            name="uq_ccasmt_app_candidate_id",
        ),
        UniqueConstraint(
            "id",
            "source_consistency_application_id",
            "source_consistency_candidate_id",
            name="uq_ccasmt_id_srcapp_candidate",
        ),
        UniqueConstraint(
            "id",
            "consistency_check_application_id",
            "source_consistency_application_id",
            "source_consistency_candidate_id",
            name="uq_ccasmt_id_app_srccand",
        ),
        CheckConstraint("batch_index >= 0", name="batch_index_nn"),
        CheckConstraint(f"verdict IN {_VERDICT_SQL}", name="verdict_valid"),
        CheckConstraint(f"severity IN {_SEVERITY_SQL}", name="severity_valid"),
        CheckConstraint(_VERDICT_SEVERITY_SQL, name="verdict_severity_pair_valid"),
        CheckConstraint(_CONFIDENCE_SQL, name="confidence_valid"),
        CheckConstraint("char_length(explanation) BETWEEN 1 AND 2000", name="explanation_len"),
        CheckConstraint(
            _IMPACT_JSON_SQL,
            name="impact_contract_valid",
        ),
        CheckConstraint(
            _ACTIONS_JSON_SQL,
            name="actions_contract_valid",
        ),
        CheckConstraint(
            "assessment_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="assessment_manifest_hash_fmt",
        ),
    )

    consistency_check_application_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    source_consistency_application_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    source_consistency_candidate_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    batch_index: Mapped[int] = mapped_column(Integer, nullable=False)
    verdict: Mapped[str] = mapped_column(String(32), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    impact_json: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )
    recommended_actions_json: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )
    assessment_manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    application: Mapped["ConsistencyCheckApplication"] = relationship(
        back_populates="assessments",
        foreign_keys=[consistency_check_application_id],
        overlaps="assessments",
    )
    source_consistency_candidate: Mapped["FactValueConsistencyCandidate"] = relationship(
        foreign_keys=[source_consistency_candidate_id, source_consistency_application_id],
    )
    batch: Mapped["ConsistencyCheckBatchLedger"] = relationship(
        back_populates="assessments",
        foreign_keys=[consistency_check_application_id, batch_index],
        overlaps="application,assessments",
    )
    citations: Mapped[list["ConsistencyAssessmentCitation"]] = relationship(
        back_populates="assessment",
        foreign_keys=(
            "[ConsistencyAssessmentCitation.assessment_id, "
            "ConsistencyAssessmentCitation.source_consistency_application_id, "
            "ConsistencyAssessmentCitation.source_consistency_candidate_id]"
        ),
        order_by="ConsistencyAssessmentCitation.citation_order",
    )
    review_decisions: Mapped[list["ConsistencyReviewDecision"]] = relationship(
        back_populates="assessment",
        foreign_keys=(
            "[ConsistencyReviewDecision.assessment_id, "
            "ConsistencyReviewDecision.consistency_check_application_id, "
            "ConsistencyReviewDecision.source_consistency_application_id, "
            "ConsistencyReviewDecision.source_consistency_candidate_id]"
        ),
        order_by="ConsistencyReviewDecision.decision_no",
    )

    @validates("verdict")
    def validate_verdict(self, _key: str, value: str) -> str:
        return _normalize_enum_value(
            value,
            field_name="verdict",
            max_length=32,
            allowed_values=_ALLOWED_VERDICTS,
        )

    @validates("severity")
    def validate_severity(self, _key: str, value: str) -> str:
        return _normalize_enum_value(
            value,
            field_name="severity",
            max_length=16,
            allowed_values=_ALLOWED_SEVERITIES,
        )

    @validates("confidence")
    def validate_confidence(self, _key: str, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("confidence must be a finite number between 0 and 1")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
            raise ValueError("confidence must be a finite number between 0 and 1")
        return numeric

    @validates("explanation")
    def validate_explanation(self, _key: str, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("explanation must be a string")
        normalized = value.strip()
        if not normalized:
            raise ValueError("explanation must not be empty")
        if len(normalized) > 2000:
            raise ValueError("explanation must be at most 2000 characters")
        return normalized

    @validates("impact_json")
    def validate_impact_json(self, _key: str, value: list[str]) -> list[str]:
        return _validate_string_list(
            value,
            field_name="impact_json",
            allowed_values=_ALLOWED_IMPACTS,
        )

    @validates("recommended_actions_json")
    def validate_recommended_actions_json(self, _key: str, value: list[str]) -> list[str]:
        return _validate_string_list(
            value,
            field_name="recommended_actions_json",
            allowed_values=_ALLOWED_ACTIONS,
        )

    @validates("assessment_manifest_hash")
    def validate_assessment_manifest_hash(self, _key: str, value: str) -> str:
        return _normalize_hash(value, field_name="assessment_manifest_hash")


class ConsistencyAssessmentCitation(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "consistency_assessment_citations"
    __table_args__ = (
        ForeignKeyConstraint(
            [
                "assessment_id",
                "source_consistency_application_id",
                "source_consistency_candidate_id",
            ],
            [
                "consistency_assessments.id",
                "consistency_assessments.source_consistency_application_id",
                "consistency_assessments.source_consistency_candidate_id",
            ],
            name="fk_cccite_asmt_srccand_ccasmt",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            [
                "source_consistency_application_id",
                "source_consistency_candidate_id",
                "source_fact_value_id",
            ],
            [
                "fact_value_consistency_candidate_members.consistency_application_id",
                "fact_value_consistency_candidate_members.candidate_id",
                "fact_value_consistency_candidate_members.fact_value_id",
            ],
            name="fk_cccite_srccand_fv_fvccm",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["evidence_link_id", "source_fact_value_id"],
            ["fact_evidence_links.id", "fact_evidence_links.fact_value_id"],
            name="fk_cccite_evid_fv_fel",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "assessment_id",
            "evidence_link_id",
            name="uq_cccite_assessment_evidence_link_id",
        ),
        UniqueConstraint(
            "assessment_id",
            "citation_order",
            name="uq_cccite_assessment_citation_order",
        ),
        CheckConstraint("citation_order BETWEEN 0 AND 199", name="citation_order_range"),
    )

    assessment_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    source_consistency_application_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    source_consistency_candidate_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    source_fact_value_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    evidence_link_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    citation_order: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    assessment: Mapped["ConsistencyAssessmentLedger"] = relationship(
        back_populates="citations",
        foreign_keys=[
            assessment_id,
            source_consistency_application_id,
            source_consistency_candidate_id,
        ],
        overlaps="source_candidate_member,evidence_link",
    )
    source_candidate_member: Mapped["FactValueConsistencyCandidateMember"] = relationship(
        foreign_keys=[
            source_consistency_application_id,
            source_consistency_candidate_id,
            source_fact_value_id,
        ],
        overlaps="assessment,citations,evidence_link",
    )
    evidence_link: Mapped["FactEvidenceLink"] = relationship(
        foreign_keys=[evidence_link_id, source_fact_value_id],
        overlaps="source_candidate_member",
    )

    @validates("citation_order")
    def validate_citation_order(self, _key: str, value: int) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("citation_order must be an integer between 0 and 199")
        if value < 0 or value > 199:
            raise ValueError("citation_order must be an integer between 0 and 199")
        return value


Index("ix_ccapp_project_id", ConsistencyCheckApplication.project_id)
Index("ix_ccapp_consistency_application_id", ConsistencyCheckApplication.consistency_application_id)
Index("ix_ccapp_orchestration_id", ConsistencyCheckApplication.orchestration_id)
Index("ix_ccbatch_consistency_check_application_id", ConsistencyCheckBatchLedger.consistency_check_application_id)
Index("ix_ccbatch_input_batch_id", ConsistencyCheckBatchLedger.input_batch_id)
Index("ix_ccbatch_inference_run_id", ConsistencyCheckBatchLedger.inference_run_id)
Index("ix_ccasmt_consistency_check_application_id", ConsistencyAssessmentLedger.consistency_check_application_id)
Index("ix_ccasmt_source_consistency_application_id", ConsistencyAssessmentLedger.source_consistency_application_id)
Index("ix_ccasmt_source_consistency_candidate_id", ConsistencyAssessmentLedger.source_consistency_candidate_id)
Index("ix_ccasmt_batch_index", ConsistencyAssessmentLedger.batch_index)
Index("ix_cccite_assessment_id", ConsistencyAssessmentCitation.assessment_id)
Index("ix_cccite_evidence_link_id", ConsistencyAssessmentCitation.evidence_link_id)
Index("ix_cccite_source_fact_value_id", ConsistencyAssessmentCitation.source_fact_value_id)
