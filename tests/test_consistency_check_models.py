from __future__ import annotations

from pathlib import Path
import subprocess

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import CheckConstraint, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable

from app.models import (
    Base,
    ConsistencyAssessmentCitation,
    ConsistencyAssessmentLedger,
    ConsistencyCheckApplication,
    ConsistencyCheckBatchLedger,
)
from app.models.fact import FactEvidenceLink
from app.models.fact_extraction_orchestration import FactExtractionOrchestration
from app.models.fact_value_duplicate_grouping import FactValueConsistencyCandidateMember
from app.models.inference import InferenceRun


def test_consistency_check_tables_registered() -> None:
    assert {
        "consistency_check_applications",
        "consistency_check_batches",
        "consistency_assessments",
        "consistency_assessment_citations",
    } <= set(Base.metadata.tables)


def test_single_migration_head_is_consistency_check_ledger() -> None:
    root = Path(__file__).resolve().parents[1]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    script = ScriptDirectory.from_config(config)
    assert list(script.get_heads()) == ["202608010600"]


def test_consistency_check_tables_compile_with_postgresql_offline_ddl() -> None:
    dialect = postgresql.dialect()

    app_sql = str(CreateTable(ConsistencyCheckApplication.__table__).compile(dialect=dialect))
    batch_sql = str(CreateTable(ConsistencyCheckBatchLedger.__table__).compile(dialect=dialect))
    assessment_sql = str(CreateTable(ConsistencyAssessmentLedger.__table__).compile(dialect=dialect))
    citation_sql = str(CreateTable(ConsistencyAssessmentCitation.__table__).compile(dialect=dialect))
    orchestration_sql = str(CreateTable(FactExtractionOrchestration.__table__).compile(dialect=dialect))
    inference_run_sql = str(CreateTable(InferenceRun.__table__).compile(dialect=dialect))
    evidence_link_sql = str(CreateTable(FactEvidenceLink.__table__).compile(dialect=dialect))
    candidate_member_sql = str(
        CreateTable(FactValueConsistencyCandidateMember.__table__).compile(dialect=dialect)
    )

    assert "uq_ccapp_exec_identity_hash" in app_sql
    assert "fk_ccapp_srcapp_orch_fvcca" in app_sql
    assert "fk_ccapp_orch_project_feo" in app_sql
    assert "ON DELETE RESTRICT" in app_sql
    assert "uq_ccbatch_app_batch_index" in batch_sql
    assert "uq_ccbatch_app_inference_run_id" in batch_sql
    assert "fk_ccbatch_run_input_ir" in batch_sql
    assert "batch_shape_valid" in batch_sql
    assert "fk_ccasmt_candidate_srcapp_fvcc" in assessment_sql
    assert "fk_ccasmt_app_batch_index_ccbatch" in assessment_sql
    assert "verdict_severity_pair_valid" in assessment_sql
    assert "confidence_valid" in assessment_sql
    assert "uq_ccasmt_id_srcapp_candidate" in assessment_sql
    assert "uq_cccite_assessment_evidence_link_id" in citation_sql
    assert "uq_cccite_assessment_citation_order" in citation_sql
    assert "fk_cccite_asmt_srccand_ccasmt" in citation_sql
    assert "fk_cccite_srccand_fv_fvccm" in citation_sql
    assert "fk_cccite_evid_fv_fel" in citation_sql
    assert "uq_feo_id_project" in orchestration_sql
    assert "uq_ir_id_input_batch" in inference_run_sql
    assert "uq_fel_id_fact_value" in evidence_link_sql
    assert "uq_fvccm_app_cand_fv" in candidate_member_sql


def test_consistency_check_composite_foreign_keys_and_restricts() -> None:
    app_table = Base.metadata.tables["consistency_check_applications"]
    batch_table = Base.metadata.tables["consistency_check_batches"]
    assessment_table = Base.metadata.tables["consistency_assessments"]
    citation_table = Base.metadata.tables["consistency_assessment_citations"]

    app_foreign_keys = {
        tuple(constraint.column_keys): (
            constraint.name,
            tuple(element.column.name for element in constraint.elements),
            constraint.ondelete,
        )
        for constraint in app_table.foreign_key_constraints
    }
    batch_foreign_keys = {
        tuple(constraint.column_keys): (
            constraint.name,
            tuple(element.column.name for element in constraint.elements),
            constraint.ondelete,
        )
        for constraint in batch_table.foreign_key_constraints
    }
    assessment_foreign_keys = {
        tuple(constraint.column_keys): (
            constraint.name,
            tuple(element.column.name for element in constraint.elements),
            constraint.ondelete,
        )
        for constraint in assessment_table.foreign_key_constraints
    }
    citation_foreign_keys = {
        tuple(constraint.column_keys): (
            constraint.name,
            tuple(element.column.name for element in constraint.elements),
            constraint.ondelete,
        )
        for constraint in citation_table.foreign_key_constraints
    }

    assert app_foreign_keys[("project_id",)] == ("fk_ccapp_project_id_projects", ("id",), "RESTRICT")
    assert app_foreign_keys[("consistency_application_id", "orchestration_id")] == (
        "fk_ccapp_srcapp_orch_fvcca",
        ("id", "orchestration_id"),
        "RESTRICT",
    )
    assert app_foreign_keys[("orchestration_id", "project_id")] == (
        "fk_ccapp_orch_project_feo",
        ("id", "project_id"),
        "RESTRICT",
    )
    assert batch_foreign_keys[("consistency_check_application_id",)] == (
        "fk_ccbatch_app_id_ccapp",
        ("id",),
        "RESTRICT",
    )
    assert batch_foreign_keys[("input_batch_id",)] == (
        "fk_ccbatch_input_batch_id_iib",
        ("id",),
        "RESTRICT",
    )
    assert batch_foreign_keys[("inference_run_id", "input_batch_id")] == (
        "fk_ccbatch_run_input_ir",
        ("id", "input_batch_id"),
        "RESTRICT",
    )
    assert assessment_foreign_keys[
        ("consistency_check_application_id", "source_consistency_application_id")
    ] == (
        "fk_ccasmt_app_srcapp_ccapp",
        ("id", "consistency_application_id"),
        "RESTRICT",
    )
    assert assessment_foreign_keys[
        ("source_consistency_candidate_id", "source_consistency_application_id")
    ] == (
        "fk_ccasmt_candidate_srcapp_fvcc",
        ("id", "consistency_application_id"),
        "RESTRICT",
    )
    assert assessment_foreign_keys[("consistency_check_application_id", "batch_index")] == (
        "fk_ccasmt_app_batch_index_ccbatch",
        ("consistency_check_application_id", "batch_index"),
        "RESTRICT",
    )
    assert citation_foreign_keys[
        ("assessment_id", "source_consistency_application_id", "source_consistency_candidate_id")
    ] == (
        "fk_cccite_asmt_srccand_ccasmt",
        ("id", "source_consistency_application_id", "source_consistency_candidate_id"),
        "RESTRICT",
    )
    assert citation_foreign_keys[
        ("source_consistency_application_id", "source_consistency_candidate_id", "source_fact_value_id")
    ] == (
        "fk_cccite_srccand_fv_fvccm",
        ("consistency_application_id", "candidate_id", "fact_value_id"),
        "RESTRICT",
    )
    assert citation_foreign_keys[("evidence_link_id", "source_fact_value_id")] == (
        "fk_cccite_evid_fv_fel",
        ("id", "fact_value_id"),
        "RESTRICT",
    )


def test_consistency_check_unique_constraints_cover_expected_keys() -> None:
    app_table = Base.metadata.tables["consistency_check_applications"]
    batch_table = Base.metadata.tables["consistency_check_batches"]
    assessment_table = Base.metadata.tables["consistency_assessments"]
    citation_table = Base.metadata.tables["consistency_assessment_citations"]

    app_uniques = {
        tuple(constraint.columns.keys()): constraint.name
        for constraint in app_table.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    batch_uniques = {
        tuple(constraint.columns.keys()): constraint.name
        for constraint in batch_table.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    assessment_uniques = {
        tuple(constraint.columns.keys()): constraint.name
        for constraint in assessment_table.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    citation_uniques = {
        tuple(constraint.columns.keys()): constraint.name
        for constraint in citation_table.constraints
        if isinstance(constraint, UniqueConstraint)
    }

    assert app_uniques[("execution_identity_hash",)] == "uq_ccapp_exec_identity_hash"
    assert app_uniques[("id", "consistency_application_id")] == "uq_ccapp_id_srcapp"
    assert batch_uniques[("consistency_check_application_id", "batch_index")] == "uq_ccbatch_app_batch_index"
    assert batch_uniques[("consistency_check_application_id", "inference_run_id")] == "uq_ccbatch_app_inference_run_id"
    assert assessment_uniques[("consistency_check_application_id", "source_consistency_candidate_id")] == (
        "uq_ccasmt_app_candidate_id"
    )
    assert assessment_uniques[("id", "source_consistency_application_id", "source_consistency_candidate_id")] == (
        "uq_ccasmt_id_srcapp_candidate"
    )
    assert citation_uniques[("assessment_id", "evidence_link_id")] == (
        "uq_cccite_assessment_evidence_link_id"
    )
    assert citation_uniques[("assessment_id", "citation_order")] == (
        "uq_cccite_assessment_citation_order"
    )


def test_consistency_check_constraints_cover_source_bindings_and_result_contracts() -> None:
    app_checks = {
        constraint.name: str(constraint.sqltext)
        for constraint in Base.metadata.tables["consistency_check_applications"].constraints
        if isinstance(constraint, CheckConstraint)
    }
    batch_checks = {
        constraint.name: str(constraint.sqltext)
        for constraint in Base.metadata.tables["consistency_check_batches"].constraints
        if isinstance(constraint, CheckConstraint)
    }
    assessment_checks = {
        constraint.name: str(constraint.sqltext)
        for constraint in Base.metadata.tables["consistency_assessments"].constraints
        if isinstance(constraint, CheckConstraint)
    }

    batch_shape_sql = next(
        sql for name, sql in batch_checks.items() if name is not None and name.endswith("batch_shape_valid")
    )
    executed_sql = next(
        sql for name, sql in app_checks.items() if name is not None and name.endswith("executed_eq_batch")
    )
    run_skipped_sql = next(
        sql for name, sql in app_checks.items() if name is not None and name.endswith("run_skipped_eq_batch")
    )
    verdict_severity_sql = next(
        sql
        for name, sql in assessment_checks.items()
        if name is not None and name.endswith("verdict_severity_pair_valid")
    )
    confidence_sql = next(
        sql for name, sql in assessment_checks.items() if name is not None and name.endswith("confidence_valid")
    )
    explanation_sql = next(
        sql for name, sql in assessment_checks.items() if name is not None and name.endswith("explanation_len")
    )
    impact_sql = next(
        sql
        for name, sql in assessment_checks.items()
        if name is not None and name.endswith("impact_contract_valid")
    )
    actions_sql = next(
        sql
        for name, sql in assessment_checks.items()
        if name is not None and name.endswith("actions_contract_valid")
    )
    citation_sql = next(
        str(constraint.sqltext)
        for constraint in Base.metadata.tables["consistency_assessment_citations"].constraints
        if isinstance(constraint, CheckConstraint) and constraint.name is not None and constraint.name.endswith("citation_order_range")
    )

    assert "executed_batch_count = batch_count" in executed_sql
    assert "inference_run_count + skipped_empty_batch_count = batch_count" in run_skipped_sql
    assert "skipped_empty = true" in batch_shape_sql
    assert "input_batch_id IS NOT NULL" in batch_shape_sql
    assert "verdict = 'conflict'" in verdict_severity_sql
    assert "severity = 'none'" in verdict_severity_sql
    assert "confidence <= 1.0" in confidence_sql
    assert "CAST(confidence AS TEXT) NOT IN ('NaN', 'Infinity', '-Infinity')" in confidence_sql
    assert "char_length(explanation) BETWEEN 1 AND 2000" in explanation_sql
    assert "jsonb_array_length(impact_json) <= 20" in impact_sql
    assert "entity_resolution_review" in impact_sql
    assert "jsonb_array_length(recommended_actions_json) <= 20" in actions_sql
    assert "leave_as_is" in actions_sql
    assert "citation_order BETWEEN 0 AND 199" in citation_sql


def test_existing_helper_unique_constraints_support_composite_references() -> None:
    orchestration_uniques = {
        tuple(constraint.columns.keys()): constraint.name
        for constraint in Base.metadata.tables["fact_extraction_orchestrations"].constraints
        if isinstance(constraint, UniqueConstraint)
    }
    inference_run_uniques = {
        tuple(constraint.columns.keys()): constraint.name
        for constraint in Base.metadata.tables["inference_runs"].constraints
        if isinstance(constraint, UniqueConstraint)
    }
    evidence_link_uniques = {
        tuple(constraint.columns.keys()): constraint.name
        for constraint in Base.metadata.tables["fact_evidence_links"].constraints
        if isinstance(constraint, UniqueConstraint)
    }
    candidate_member_uniques = {
        tuple(constraint.columns.keys()): constraint.name
        for constraint in Base.metadata.tables["fact_value_consistency_candidate_members"].constraints
        if isinstance(constraint, UniqueConstraint)
    }

    assert orchestration_uniques[("id", "project_id")] == "uq_feo_id_project"
    assert inference_run_uniques[("id", "input_batch_id")] == "uq_ir_id_input_batch"
    assert evidence_link_uniques[("id", "fact_value_id")] == "uq_fel_id_fact_value"
    assert candidate_member_uniques[("consistency_application_id", "candidate_id", "fact_value_id")] == (
        "uq_fvccm_app_cand_fv"
    )


def test_consistency_check_constraint_and_index_names_fit_postgresql_limit() -> None:
    targeted_existing_constraints = {
        "uq_feo_id_project",
        "uq_ir_id_input_batch",
        "uq_fel_id_fact_value",
        "uq_fvccm_app_cand_fv",
    }
    for table_name in (
        "fact_extraction_orchestrations",
        "inference_runs",
        "fact_evidence_links",
        "fact_value_consistency_candidate_members",
    ):
        table = Base.metadata.tables[table_name]
        for constraint in table.constraints:
            if constraint.name in targeted_existing_constraints:
                assert len(constraint.name) <= 63

    for table_name in (
        "consistency_check_applications",
        "consistency_check_batches",
        "consistency_assessments",
        "consistency_assessment_citations",
    ):
        table = Base.metadata.tables[table_name]
        for constraint in table.constraints:
            if constraint.name is not None:
                assert len(constraint.name) <= 63
        for index in table.indexes:
            assert len(index.name) <= 63


def test_consistency_check_indexes_compile_for_postgresql() -> None:
    dialect = postgresql.dialect()
    for table_name in (
        "consistency_check_applications",
        "consistency_check_batches",
        "consistency_assessments",
        "consistency_assessment_citations",
    ):
        table = Base.metadata.tables[table_name]
        for index in table.indexes:
            sql = str(CreateIndex(index).compile(dialect=dialect))
            assert "CREATE INDEX" in sql


def test_consistency_check_migration_upgrade_sql_creates_tables() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["alembic", "upgrade", "202608010300:202608010400", "--sql"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "-- Running upgrade 202608010300 -> 202608010400" in result.stdout
    assert "ALTER TABLE fact_extraction_orchestrations ADD CONSTRAINT uq_feo_id_project" in result.stdout
    assert "ALTER TABLE inference_runs ADD CONSTRAINT uq_ir_id_input_batch" in result.stdout
    assert "ALTER TABLE fact_value_consistency_candidate_members ADD CONSTRAINT uq_fvccm_app_cand_fv" in result.stdout
    assert "ALTER TABLE fact_evidence_links ADD CONSTRAINT uq_fel_id_fact_value" in result.stdout
    assert "CREATE TABLE consistency_check_applications" in result.stdout
    assert "CREATE TABLE consistency_check_batches" in result.stdout
    assert "CREATE TABLE consistency_assessments" in result.stdout
    assert "CREATE TABLE consistency_assessment_citations" in result.stdout


def test_consistency_check_migration_downgrade_sql_drops_tables_in_dependency_order() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["alembic", "downgrade", "202608010400:202608010300", "--sql"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    output = result.stdout
    assert output.index("DROP TABLE consistency_assessment_citations") < output.index(
        "DROP TABLE consistency_assessments"
    )
    assert output.index("DROP TABLE consistency_assessments") < output.index(
        "DROP TABLE consistency_check_batches"
    )
    assert output.index("DROP TABLE consistency_check_batches") < output.index(
        "DROP TABLE consistency_check_applications"
    )
    assert "ALTER TABLE fact_evidence_links DROP CONSTRAINT uq_fel_id_fact_value" in output
    assert "ALTER TABLE inference_runs DROP CONSTRAINT uq_ir_id_input_batch" in output
    assert "ALTER TABLE fact_extraction_orchestrations DROP CONSTRAINT uq_feo_id_project" in output


def test_consistency_check_migration_source_declares_expected_revision_and_tables() -> None:
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "202608010400_consistency_check_ledgers.py"
    )
    content = migration_path.read_text(encoding="utf-8")

    assert 'revision: str = "202608010400"' in content
    assert 'down_revision: str | None = "202608010300"' in content
    assert '"consistency_check_applications"' in content
    assert '"consistency_check_batches"' in content
    assert '"consistency_assessments"' in content
    assert '"consistency_assessment_citations"' in content
    assert '"uq_feo_id_project"' in content
    assert '"uq_ir_id_input_batch"' in content
    assert '"uq_fvccm_app_cand_fv"' in content
    assert '"uq_fel_id_fact_value"' in content
