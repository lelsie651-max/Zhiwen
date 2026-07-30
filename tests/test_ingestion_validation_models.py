import uuid

import pytest
from pydantic import ValidationError
from sqlalchemy import CheckConstraint, UniqueConstraint

from app.models import Base
from app.models.ingestion_validation import IngestionRiskFlag, IngestionValidationReport
from app.schemas.ingestion_validation import (
    RiskFlagCreate,
    RiskFlagRead,
    ValidationReportCreate,
    ValidationReportInternalRead,
    ValidationReportRead,
)


def test_ingestion_validation_tables_are_registered() -> None:
    assert {"ingestion_validation_reports", "ingestion_risk_flags"} <= set(Base.metadata.tables)


def test_validation_report_revision_attempt_unique_constraint_exists() -> None:
    reports_table = Base.metadata.tables["ingestion_validation_reports"]
    assert any(
        isinstance(constraint, UniqueConstraint)
        and tuple(constraint.columns.keys()) == ("revision_id", "attempt_no")
        for constraint in reports_table.constraints
    )


def test_ratio_and_confidence_range_constraints_exist() -> None:
    reports_table = Base.metadata.tables["ingestion_validation_reports"]
    risk_flags_table = Base.metadata.tables["ingestion_risk_flags"]

    report_checks = {
        str(constraint.sqltext)
        for constraint in reports_table.constraints
        if isinstance(constraint, CheckConstraint)
    }
    risk_flag_checks = {
        str(constraint.sqltext)
        for constraint in risk_flags_table.constraints
        if isinstance(constraint, CheckConstraint)
    }

    assert any("content_confidence" in sql and "<= 1" in sql for sql in report_checks)
    assert any("printable_ratio" in sql and "<= 1" in sql for sql in report_checks)
    assert any("garbled_ratio" in sql and "<= 1" in sql for sql in report_checks)
    assert any("repetition_ratio" in sql and "<= 1" in sql for sql in report_checks)
    assert any("confidence" in sql and "<= 1" in sql for sql in risk_flag_checks)


def test_decision_actor_time_pair_constraints_exist() -> None:
    reports_table = Base.metadata.tables["ingestion_validation_reports"]
    check_sql = {
        str(constraint.sqltext)
        for constraint in reports_table.constraints
        if isinstance(constraint, CheckConstraint)
    }

    assert any("decided_by_id IS NULL" in sql and "decided_at IS NULL" in sql for sql in check_sql)
    assert any("user_decision IS NULL" in sql and "decided_by_id IS NULL" in sql for sql in check_sql)


@pytest.mark.parametrize(
    ("schema_class", "payload"),
    [
        (
            ValidationReportCreate,
            {
                "revision_id": uuid.uuid4(),
                "attempt_no": 1,
                "status": "bad-status",
            },
        ),
        (
            ValidationReportCreate,
            {
                "revision_id": uuid.uuid4(),
                "attempt_no": 1,
                "outcome": "bad-outcome",
            },
        ),
        (
            RiskFlagCreate,
            {
                "report_id": uuid.uuid4(),
                "category": "bad-category",
                "kind": "garbled_text",
                "severity": "warning",
                "detector": "deterministic",
                "message": "发现乱码",
            },
        ),
        (
            RiskFlagCreate,
            {
                "report_id": uuid.uuid4(),
                "category": "text_quality",
                "kind": "garbled_text",
                "severity": "bad-severity",
                "detector": "deterministic",
                "message": "发现乱码",
            },
        ),
    ],
)
def test_invalid_status_outcome_severity_and_category_are_rejected(
    schema_class,
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        schema_class(**payload)


def test_validation_report_create_rejects_decided_by_id() -> None:
    with pytest.raises(ValidationError):
        ValidationReportCreate(
            revision_id=uuid.uuid4(),
            attempt_no=1,
            decided_by_id=uuid.uuid4(),
        )


def test_safe_read_hides_failure_message() -> None:
    assert "failure_message" not in ValidationReportRead.model_fields


def test_internal_read_includes_failure_message() -> None:
    assert "failure_message" in ValidationReportInternalRead.model_fields


def test_json_default_values_are_independent() -> None:
    report_one = ValidationReportCreate(revision_id=uuid.uuid4(), attempt_no=1)
    report_two = ValidationReportCreate(revision_id=uuid.uuid4(), attempt_no=2)
    report_one.file_checks["encoding"] = "utf-8"

    risk_one = RiskFlagCreate(
        report_id=uuid.uuid4(),
        category="text_quality",
        kind="garbled_text",
        severity="warning",
        detector="deterministic",
        message="发现乱码",
    )
    risk_two = RiskFlagCreate(
        report_id=uuid.uuid4(),
        category="privacy",
        kind="pii_leak",
        severity="warning",
        detector="manual",
        message="可能包含隐私信息",
    )
    risk_one.details["rule"] = "garbled"

    assert report_two.file_checks == {}
    assert report_one.file_checks is not report_two.file_checks
    assert risk_two.details == {}
    assert risk_one.details is not risk_two.details


def test_validation_report_and_risk_flag_time_columns_use_timezone() -> None:
    report_columns = (
        IngestionValidationReport.__table__.c.decided_at,
        IngestionValidationReport.__table__.c.started_at,
        IngestionValidationReport.__table__.c.completed_at,
        IngestionValidationReport.__table__.c.created_at,
        IngestionValidationReport.__table__.c.updated_at,
    )
    risk_flag_column = IngestionRiskFlag.__table__.c.created_at

    assert all(column.type.timezone is True for column in report_columns)
    assert risk_flag_column.type.timezone is True


def test_risk_flag_read_exposes_safe_fields() -> None:
    assert "details" in RiskFlagRead.model_fields
