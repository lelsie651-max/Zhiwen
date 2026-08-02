from __future__ import annotations

from pathlib import Path
import subprocess
import uuid

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import CheckConstraint, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable

from app.models import (
    Base,
    ConsistencyAssessmentLedger,
    ConsistencyCheckApplication,
    ConsistencyReviewDecision,
    ConsistencyReviewDecisionKind,
    ConsistencyReviewDecisionSelection,
)


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


def test_consistency_review_tables_registered() -> None:
    assert {
        "consistency_review_decisions",
        "consistency_review_decision_selections",
    } <= set(Base.metadata.tables)


def test_single_migration_head_is_consistency_review() -> None:
    root = Path(__file__).resolve().parents[1]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    script = ScriptDirectory.from_config(config)
    assert list(script.get_heads()) == ["202608010600"]


def test_consistency_review_tables_compile_with_postgresql_offline_ddl() -> None:
    dialect = postgresql.dialect()

    app_sql = str(CreateTable(ConsistencyCheckApplication.__table__).compile(dialect=dialect))
    assessment_sql = str(CreateTable(ConsistencyAssessmentLedger.__table__).compile(dialect=dialect))
    decision_sql = str(CreateTable(ConsistencyReviewDecision.__table__).compile(dialect=dialect))
    selection_sql = str(CreateTable(ConsistencyReviewDecisionSelection.__table__).compile(dialect=dialect))

    assert "uq_ccapp_id_project" in app_sql
    assert "uq_ccasmt_id_app_srccand" in assessment_sql
    assert "fk_ccrevd_app_project_ccapp" in decision_sql
    assert "fk_ccrevd_asmt_app_src_ccasmt" in decision_sql
    assert "fk_ccrevd_actor_id_users" in decision_sql
    assert "fk_ccrevd_prev_asmt_self" in decision_sql
    assert "uq_ccrevd_manifest_hash" in decision_sql
    assert "ccrevd_sel_count_shape" in decision_sql
    assert "ccrevd_revision_shape" in decision_sql
    assert "fk_ccrevs_decision_src_ccrevd" in selection_sql
    assert "fk_ccrevs_srccand_fv_fvccm" in selection_sql
    assert "uq_ccrevs_decision_fv" in selection_sql
    assert "uq_ccrevs_decision_order" in selection_sql
    assert "ON DELETE RESTRICT" in decision_sql
    assert "ON DELETE RESTRICT" in selection_sql


def test_consistency_review_composite_foreign_keys_and_restricts() -> None:
    decision_table = Base.metadata.tables["consistency_review_decisions"]
    selection_table = Base.metadata.tables["consistency_review_decision_selections"]

    decision_foreign_keys = {
        tuple(constraint.column_keys): (
            constraint.name,
            tuple(element.column.name for element in constraint.elements),
            constraint.ondelete,
        )
        for constraint in decision_table.foreign_key_constraints
    }
    selection_foreign_keys = {
        tuple(constraint.column_keys): (
            constraint.name,
            tuple(element.column.name for element in constraint.elements),
            constraint.ondelete,
        )
        for constraint in selection_table.foreign_key_constraints
    }

    assert decision_foreign_keys[("project_id",)] == (
        "fk_ccrevd_project_id_projects",
        ("id",),
        "RESTRICT",
    )
    assert decision_foreign_keys[("consistency_check_application_id", "project_id")] == (
        "fk_ccrevd_app_project_ccapp",
        ("id", "project_id"),
        "RESTRICT",
    )
    assert decision_foreign_keys[
        (
            "assessment_id",
            "consistency_check_application_id",
            "source_consistency_application_id",
            "source_consistency_candidate_id",
        )
    ] == (
        "fk_ccrevd_asmt_app_src_ccasmt",
        (
            "id",
            "consistency_check_application_id",
            "source_consistency_application_id",
            "source_consistency_candidate_id",
        ),
        "RESTRICT",
    )
    assert decision_foreign_keys[("actor_id",)] == (
        "fk_ccrevd_actor_id_users",
        ("id",),
        "RESTRICT",
    )
    assert decision_foreign_keys[("supersedes_decision_id", "assessment_id")] == (
        "fk_ccrevd_prev_asmt_self",
        ("id", "assessment_id"),
        "RESTRICT",
    )
    assert selection_foreign_keys[
        (
            "decision_id",
            "assessment_id",
            "source_consistency_application_id",
            "source_consistency_candidate_id",
        )
    ] == (
        "fk_ccrevs_decision_src_ccrevd",
        (
            "id",
            "assessment_id",
            "source_consistency_application_id",
            "source_consistency_candidate_id",
        ),
        "RESTRICT",
    )
    assert selection_foreign_keys[
        (
            "source_consistency_application_id",
            "source_consistency_candidate_id",
            "fact_value_id",
        )
    ] == (
        "fk_ccrevs_srccand_fv_fvccm",
        ("consistency_application_id", "candidate_id", "fact_value_id"),
        "RESTRICT",
    )


def test_consistency_review_unique_constraints_cover_expected_keys() -> None:
    app_table = Base.metadata.tables["consistency_check_applications"]
    assessment_table = Base.metadata.tables["consistency_assessments"]
    decision_table = Base.metadata.tables["consistency_review_decisions"]
    selection_table = Base.metadata.tables["consistency_review_decision_selections"]

    app_uniques = {
        tuple(constraint.columns.keys()): constraint.name
        for constraint in app_table.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    assessment_uniques = {
        tuple(constraint.columns.keys()): constraint.name
        for constraint in assessment_table.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    decision_uniques = {
        tuple(constraint.columns.keys()): constraint.name
        for constraint in decision_table.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    selection_uniques = {
        tuple(constraint.columns.keys()): constraint.name
        for constraint in selection_table.constraints
        if isinstance(constraint, UniqueConstraint)
    }

    assert app_uniques[("id", "project_id")] == "uq_ccapp_id_project"
    assert assessment_uniques[
        (
            "id",
            "consistency_check_application_id",
            "source_consistency_application_id",
            "source_consistency_candidate_id",
        )
    ] == "uq_ccasmt_id_app_srccand"
    assert decision_uniques[("assessment_id", "decision_no")] == "uq_ccrevd_asmt_dec_no"
    assert decision_uniques[("supersedes_decision_id",)] == "uq_ccrevd_supersedes_id"
    assert decision_uniques[("decision_manifest_hash",)] == "uq_ccrevd_manifest_hash"
    assert decision_uniques[("id", "assessment_id")] == "uq_ccrevd_id_asmt"
    assert decision_uniques[
        (
            "id",
            "assessment_id",
            "source_consistency_application_id",
            "source_consistency_candidate_id",
        )
    ] == "uq_ccrevd_id_asmt_src"
    assert selection_uniques[("decision_id", "fact_value_id")] == "uq_ccrevs_decision_fv"
    assert selection_uniques[("decision_id", "selection_order")] == "uq_ccrevs_decision_order"


def test_consistency_review_constraints_cover_revision_shape_and_selection_shape() -> None:
    decision_checks = {
        constraint.name: str(constraint.sqltext)
        for constraint in Base.metadata.tables["consistency_review_decisions"].constraints
        if isinstance(constraint, CheckConstraint)
    }
    selection_checks = {
        constraint.name: str(constraint.sqltext)
        for constraint in Base.metadata.tables["consistency_review_decision_selections"].constraints
        if isinstance(constraint, CheckConstraint)
    }

    decision_no_sql = next(
        sql for name, sql in decision_checks.items() if name is not None and name.endswith("ccrevd_dec_no_pos")
    )
    decision_kind_sql = next(
        sql for name, sql in decision_checks.items() if name is not None and name.endswith("ccrevd_kind_valid")
    )
    selected_count_sql = next(
        sql for name, sql in decision_checks.items() if name is not None and name.endswith("ccrevd_sel_count_shape")
    )
    revision_shape_sql = next(
        sql for name, sql in decision_checks.items() if name is not None and name.endswith("ccrevd_revision_shape")
    )
    comment_sql = next(
        sql for name, sql in decision_checks.items() if name is not None and name.endswith("ccrevd_comment_len")
    )
    manifest_sql = next(
        sql for name, sql in decision_checks.items() if name is not None and name.endswith("ccrevd_manifest_hash_fmt")
    )
    selection_order_sql = next(
        sql for name, sql in selection_checks.items() if name is not None and name.endswith("ccrevs_sel_order_rng")
    )

    assert "decision_no > 0" in decision_no_sql
    assert "select_one" in decision_kind_sql
    assert "keep_multiple" in selected_count_sql
    assert "confirm_compatible" in selected_count_sql
    assert "decision_no = 1 AND supersedes_decision_id IS NULL" in revision_shape_sql
    assert "decision_no > 1 AND supersedes_decision_id IS NOT NULL" in revision_shape_sql
    assert "char_length(comment) BETWEEN 1 AND 2000" in comment_sql
    assert "decision_manifest_hash ~ '^[0-9a-f]{64}$'" in manifest_sql
    assert "selection_order BETWEEN 0 AND 199" in selection_order_sql


def test_consistency_review_validators_normalize_and_enforce_boundaries() -> None:
    decision = ConsistencyReviewDecision(
        project_id=_uuid(),
        consistency_check_application_id=_uuid(),
        assessment_id=_uuid(),
        source_consistency_application_id=_uuid(),
        source_consistency_candidate_id=_uuid(),
        actor_id=_uuid(),
        decision_no=2,
        supersedes_decision_id=_uuid(),
        decision_kind=ConsistencyReviewDecisionKind.SELECT_ONE.value,
        selected_value_count=1,
        comment="  keep this one  ",
        decision_manifest_hash="A" * 64,
    )
    selection = ConsistencyReviewDecisionSelection(
        decision_id=_uuid(),
        assessment_id=_uuid(),
        source_consistency_application_id=_uuid(),
        source_consistency_candidate_id=_uuid(),
        fact_value_id=_uuid(),
        selection_order=0,
    )

    assert decision.comment == "keep this one"
    assert decision.decision_manifest_hash == "a" * 64
    assert selection.selection_order == 0

    with pytest.raises(ValueError, match="decision_kind must be one of"):
        ConsistencyReviewDecision(
            project_id=_uuid(),
            consistency_check_application_id=_uuid(),
            assessment_id=_uuid(),
            source_consistency_application_id=_uuid(),
            source_consistency_candidate_id=_uuid(),
            actor_id=_uuid(),
            decision_no=1,
            supersedes_decision_id=None,
            decision_kind="wrong",
            selected_value_count=1,
            comment=None,
            decision_manifest_hash="a" * 64,
        )
    with pytest.raises(ValueError, match="decision_no must be a positive integer"):
        ConsistencyReviewDecision(
            project_id=_uuid(),
            consistency_check_application_id=_uuid(),
            assessment_id=_uuid(),
            source_consistency_application_id=_uuid(),
            source_consistency_candidate_id=_uuid(),
            actor_id=_uuid(),
            decision_no=0,
            supersedes_decision_id=None,
            decision_kind=ConsistencyReviewDecisionKind.DEFER.value,
            selected_value_count=0,
            comment=None,
            decision_manifest_hash="a" * 64,
        )
    with pytest.raises(ValueError, match="selected_value_count must be between 0 and 200"):
        ConsistencyReviewDecision(
            project_id=_uuid(),
            consistency_check_application_id=_uuid(),
            assessment_id=_uuid(),
            source_consistency_application_id=_uuid(),
            source_consistency_candidate_id=_uuid(),
            actor_id=_uuid(),
            decision_no=1,
            supersedes_decision_id=None,
            decision_kind=ConsistencyReviewDecisionKind.DEFER.value,
            selected_value_count=201,
            comment=None,
            decision_manifest_hash="a" * 64,
        )
    with pytest.raises(ValueError, match="comment must not be empty"):
        ConsistencyReviewDecision(
            project_id=_uuid(),
            consistency_check_application_id=_uuid(),
            assessment_id=_uuid(),
            source_consistency_application_id=_uuid(),
            source_consistency_candidate_id=_uuid(),
            actor_id=_uuid(),
            decision_no=1,
            supersedes_decision_id=None,
            decision_kind=ConsistencyReviewDecisionKind.DEFER.value,
            selected_value_count=0,
            comment="   ",
            decision_manifest_hash="a" * 64,
        )
    with pytest.raises(ValueError, match="decision_manifest_hash must be a 64-character lowercase hexadecimal string"):
        ConsistencyReviewDecision(
            project_id=_uuid(),
            consistency_check_application_id=_uuid(),
            assessment_id=_uuid(),
            source_consistency_application_id=_uuid(),
            source_consistency_candidate_id=_uuid(),
            actor_id=_uuid(),
            decision_no=1,
            supersedes_decision_id=None,
            decision_kind=ConsistencyReviewDecisionKind.DEFER.value,
            selected_value_count=0,
            comment=None,
            decision_manifest_hash="z" * 64,
        )
    with pytest.raises(ValueError, match="selection_order must be an integer between 0 and 199"):
        ConsistencyReviewDecisionSelection(
            decision_id=_uuid(),
            assessment_id=_uuid(),
            source_consistency_application_id=_uuid(),
            source_consistency_candidate_id=_uuid(),
            fact_value_id=_uuid(),
            selection_order=200,
        )


def test_consistency_review_constraint_and_index_names_fit_postgresql_limit() -> None:
    for table_name in (
        "consistency_review_decisions",
        "consistency_review_decision_selections",
    ):
        table = Base.metadata.tables[table_name]
        for constraint in table.constraints:
            if constraint.name is not None:
                assert len(constraint.name) <= 63
        for index in table.indexes:
            assert len(index.name) <= 63


def test_consistency_review_indexes_compile_for_postgresql() -> None:
    dialect = postgresql.dialect()
    for table_name in (
        "consistency_review_decisions",
        "consistency_review_decision_selections",
    ):
        table = Base.metadata.tables[table_name]
        for index in table.indexes:
            sql = str(CreateIndex(index).compile(dialect=dialect))
            assert "CREATE INDEX" in sql


def test_consistency_review_migration_upgrade_sql_creates_tables() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["alembic", "upgrade", "202608010400:202608010500", "--sql"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "-- Running upgrade 202608010400 -> 202608010500" in result.stdout
    assert "ALTER TABLE consistency_check_applications ADD CONSTRAINT uq_ccapp_id_project" in result.stdout
    assert "ALTER TABLE consistency_assessments ADD CONSTRAINT uq_ccasmt_id_app_srccand" in result.stdout
    assert "CREATE TABLE consistency_review_decisions" in result.stdout
    assert "CREATE TABLE consistency_review_decision_selections" in result.stdout


def test_consistency_review_migration_downgrade_sql_drops_tables_in_dependency_order() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["alembic", "downgrade", "202608010500:202608010400", "--sql"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    output = result.stdout
    assert output.index("DROP TABLE consistency_review_decision_selections") < output.index(
        "DROP TABLE consistency_review_decisions"
    )
    assert "ALTER TABLE consistency_assessments DROP CONSTRAINT uq_ccasmt_id_app_srccand" in output
    assert "ALTER TABLE consistency_check_applications DROP CONSTRAINT uq_ccapp_id_project" in output


def test_consistency_review_migration_source_declares_expected_revision_and_tables() -> None:
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "202608010500_consistency_review_decisions.py"
    )
    content = migration_path.read_text(encoding="utf-8")

    assert 'revision: str = "202608010500"' in content
    assert 'down_revision: str | None = "202608010400"' in content
    assert '"consistency_review_decisions"' in content
    assert '"consistency_review_decision_selections"' in content
    assert '"uq_ccapp_id_project"' in content
    assert '"uq_ccasmt_id_app_srccand"' in content
