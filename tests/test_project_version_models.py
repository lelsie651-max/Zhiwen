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
    ConsistencyCheckApplication,
    DynamicSchema,
    DynamicSchemaVersion,
    FactExtractionOrchestration,
    Project,
    ProjectVersion,
    ProjectVersionCreationKind,
)


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


def test_project_version_tables_registered() -> None:
    assert {"project_versions"} <= set(Base.metadata.tables)


def test_single_migration_head_is_project_version() -> None:
    root = Path(__file__).resolve().parents[1]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    script = ScriptDirectory.from_config(config)
    assert list(script.get_heads()) == ["202608010600"]


def test_project_version_tables_compile_with_postgresql_offline_ddl() -> None:
    dialect = postgresql.dialect()

    project_sql = str(CreateTable(Project.__table__).compile(dialect=dialect))
    project_version_sql = str(CreateTable(ProjectVersion.__table__).compile(dialect=dialect))
    schema_sql = str(CreateTable(DynamicSchema.__table__).compile(dialect=dialect))
    schema_version_sql = str(CreateTable(DynamicSchemaVersion.__table__).compile(dialect=dialect))
    orchestration_sql = str(CreateTable(FactExtractionOrchestration.__table__).compile(dialect=dialect))
    app_sql = str(CreateTable(ConsistencyCheckApplication.__table__).compile(dialect=dialect))

    assert "fk_projects_cur_ver_projver" in project_sql
    assert "ON DELETE RESTRICT" in project_sql
    assert "uq_dynschema_id_project" in schema_sql
    assert "uq_dynsver_id_schema" in schema_version_sql
    assert "uq_feo_id_proj_run" in orchestration_sql
    assert "uq_ccapp_id_proj_orch_src" in app_sql
    assert "JSONB" in project_version_sql
    assert "fk_projver_schema_proj_dynschema" in project_version_sql
    assert "fk_projver_sver_schema_dynsver" in project_version_sql
    assert "fk_projver_orch_proj_run_feo" in project_version_sql
    assert "fk_projver_ccapp_proj_orch_src" in project_version_sql
    assert "fk_projver_prev_ver_projver" in project_version_sql
    assert "uq_projver_project_verno" in project_version_sql
    assert "uq_projver_manifest_hash" in project_version_sql
    assert "projver_field_count_sum" in project_version_sql
    assert "projver_rollback_shape" in project_version_sql
    assert "ON DELETE RESTRICT" in project_version_sql


def test_project_version_helper_unique_constraints_support_composite_bindings() -> None:
    schema_uniques = {
        tuple(constraint.columns.keys()): constraint.name
        for constraint in Base.metadata.tables["dynamic_schemas"].constraints
        if isinstance(constraint, UniqueConstraint)
    }
    schema_version_uniques = {
        tuple(constraint.columns.keys()): constraint.name
        for constraint in Base.metadata.tables["dynamic_schema_versions"].constraints
        if isinstance(constraint, UniqueConstraint)
    }
    orchestration_uniques = {
        tuple(constraint.columns.keys()): constraint.name
        for constraint in Base.metadata.tables["fact_extraction_orchestrations"].constraints
        if isinstance(constraint, UniqueConstraint)
    }
    app_uniques = {
        tuple(constraint.columns.keys()): constraint.name
        for constraint in Base.metadata.tables["consistency_check_applications"].constraints
        if isinstance(constraint, UniqueConstraint)
    }

    assert schema_uniques[("id", "project_id")] == "uq_dynschema_id_project"
    assert schema_version_uniques[("id", "schema_id")] == "uq_dynsver_id_schema"
    assert orchestration_uniques[("id", "project_id", "extraction_run_id")] == "uq_feo_id_proj_run"
    assert app_uniques[
        ("id", "project_id", "orchestration_id", "consistency_application_id")
    ] == "uq_ccapp_id_proj_orch_src"


def test_project_version_composite_foreign_keys_and_restricts() -> None:
    project_table = Base.metadata.tables["projects"]
    version_table = Base.metadata.tables["project_versions"]

    project_foreign_keys = {
        tuple(constraint.column_keys): (
            constraint.name,
            tuple(element.column.name for element in constraint.elements),
            constraint.ondelete,
        )
        for constraint in project_table.foreign_key_constraints
    }
    version_foreign_keys = {
        tuple(constraint.column_keys): (
            constraint.name,
            tuple(element.column.name for element in constraint.elements),
            constraint.ondelete,
        )
        for constraint in version_table.foreign_key_constraints
    }

    assert project_foreign_keys[("current_version_id", "id")] == (
        "fk_projects_cur_ver_projver",
        ("id", "project_id"),
        "RESTRICT",
    )
    assert version_foreign_keys[("project_id",)] == (
        "fk_projver_project_id_projects",
        ("id",),
        "RESTRICT",
    )
    assert version_foreign_keys[("created_by_id",)] == (
        "fk_projver_created_by_id_users",
        ("id",),
        "RESTRICT",
    )
    assert version_foreign_keys[("schema_id", "project_id")] == (
        "fk_projver_schema_proj_dynschema",
        ("id", "project_id"),
        "RESTRICT",
    )
    assert version_foreign_keys[("schema_version_id", "schema_id")] == (
        "fk_projver_sver_schema_dynsver",
        ("id", "schema_id"),
        "RESTRICT",
    )
    assert version_foreign_keys[("orchestration_id", "project_id", "extraction_run_id")] == (
        "fk_projver_orch_proj_run_feo",
        ("id", "project_id", "extraction_run_id"),
        "RESTRICT",
    )
    assert version_foreign_keys[
        (
            "consistency_check_application_id",
            "project_id",
            "orchestration_id",
            "source_consistency_application_id",
        )
    ] == (
        "fk_projver_ccapp_proj_orch_src",
        ("id", "project_id", "orchestration_id", "consistency_application_id"),
        "RESTRICT",
    )
    assert version_foreign_keys[("copied_from_version_id", "project_id")] == (
        "fk_projver_prev_ver_projver",
        ("id", "project_id"),
        "RESTRICT",
    )


def test_project_version_unique_constraints_cover_expected_keys() -> None:
    version_uniques = {
        tuple(constraint.columns.keys()): constraint.name
        for constraint in Base.metadata.tables["project_versions"].constraints
        if isinstance(constraint, UniqueConstraint)
    }

    assert version_uniques[("project_id", "version_no")] == "uq_projver_project_verno"
    assert version_uniques[("version_manifest_hash",)] == "uq_projver_manifest_hash"
    assert version_uniques[("id", "project_id")] == "uq_projver_id_project"
    assert ("snapshot_json_hash",) not in version_uniques


def test_project_current_version_relationship_uses_post_update() -> None:
    assert Project.current_version.property.post_update is True
    assert Project.current_version.property.uselist is False


def test_project_version_constraints_cover_counts_hashes_reason_json_and_rollback() -> None:
    checks = {
        constraint.name: str(constraint.sqltext)
        for constraint in Base.metadata.tables["project_versions"].constraints
        if isinstance(constraint, CheckConstraint)
    }

    version_no_sql = next(
        sql for name, sql in checks.items() if name is not None and name.endswith("projver_ver_no_pos")
    )
    creation_kind_sql = next(
        sql
        for name, sql in checks.items()
        if name is not None and name.endswith("projver_creation_kind_valid")
    )
    reason_sql = next(
        sql for name, sql in checks.items() if name is not None and name.endswith("projver_reason_len")
    )
    snapshot_obj_sql = next(
        sql for name, sql in checks.items() if name is not None and name.endswith("projver_snapshot_obj")
    )
    knowledge_manifest_sql = next(
        sql
        for name, sql in checks.items()
        if name is not None and name.endswith("projver_know_manifest_fmt")
    )
    snapshot_hash_sql = next(
        sql
        for name, sql in checks.items()
        if name is not None and name.endswith("projver_snapshot_hash_fmt")
    )
    field_count_sum_sql = next(
        sql
        for name, sql in checks.items()
        if name is not None and name.endswith("projver_field_count_sum")
    )
    rollback_shape_sql = next(
        sql
        for name, sql in checks.items()
        if name is not None and name.endswith("projver_rollback_shape")
    )

    assert "version_no > 0" in version_no_sql
    assert "creation_kind IN ('manual', 'automatic', 'pre_publish', 'rollback')" in creation_kind_sql
    assert "char_length(reason) BETWEEN 1 AND 2000" in reason_sql
    assert "jsonb_typeof(snapshot_json) = 'object'" in snapshot_obj_sql
    assert "knowledge_view_manifest_hash ~ '^[0-9a-f]{64}$'" in knowledge_manifest_sql
    assert "snapshot_json_hash ~ '^[0-9a-f]{64}$'" in snapshot_hash_sql
    assert (
        "missing_field_count + review_required_field_count + resolved_field_count + "
        "observation_only_field_count + mixed_field_count = field_count"
    ) in field_count_sum_sql
    assert "creation_kind = 'rollback' AND copied_from_version_id IS NOT NULL" in rollback_shape_sql
    assert "creation_kind <> 'rollback' AND copied_from_version_id IS NULL" in rollback_shape_sql


def test_project_version_validators_normalize_and_enforce_boundaries() -> None:
    version = ProjectVersion(
        project_id=_uuid(),
        version_no=1,
        created_by_id=_uuid(),
        creation_kind=ProjectVersionCreationKind.MANUAL.value,
        copied_from_version_id=None,
        reason="  publish snapshot  ",
        schema_id=_uuid(),
        schema_version_id=_uuid(),
        orchestration_id=_uuid(),
        extraction_run_id=_uuid(),
        consistency_check_application_id=_uuid(),
        source_consistency_application_id=_uuid(),
        schema_definition_manifest_hash="A" * 64,
        ufl_source_manifest_hash="B" * 64,
        consistency_result_manifest_hash="C" * 64,
        raw_projection_manifest_hash="D" * 64,
        reviewed_projection_manifest_hash="E" * 64,
        knowledge_view_manifest_hash="F" * 64,
        knowledge_view_algorithm_name="  dynamic_schema_knowledge_view  ",
        knowledge_view_algorithm_version="  1.0.0  ",
        snapshot_format_version="  1.0.0  ",
        snapshot_json={"records": []},
        snapshot_json_hash="1" * 64,
        version_manifest_hash="2" * 64,
        record_count=1,
        section_count=2,
        field_count=5,
        missing_field_count=1,
        review_required_field_count=1,
        resolved_field_count=1,
        observation_only_field_count=1,
        mixed_field_count=1,
    )

    assert version.reason == "publish snapshot"
    assert version.schema_definition_manifest_hash == "a" * 64
    assert version.knowledge_view_algorithm_name == "dynamic_schema_knowledge_view"
    assert version.snapshot_format_version == "1.0.0"

    with pytest.raises(ValueError, match="creation_kind must be one of"):
        ProjectVersion(
            project_id=_uuid(),
            version_no=1,
            created_by_id=_uuid(),
            creation_kind="wrong",
            copied_from_version_id=None,
            reason=None,
            schema_id=_uuid(),
            schema_version_id=_uuid(),
            orchestration_id=_uuid(),
            extraction_run_id=_uuid(),
            consistency_check_application_id=_uuid(),
            source_consistency_application_id=_uuid(),
            schema_definition_manifest_hash="a" * 64,
            ufl_source_manifest_hash="b" * 64,
            consistency_result_manifest_hash="c" * 64,
            raw_projection_manifest_hash="d" * 64,
            reviewed_projection_manifest_hash="e" * 64,
            knowledge_view_manifest_hash="f" * 64,
            knowledge_view_algorithm_name="dynamic_schema_knowledge_view",
            knowledge_view_algorithm_version="1.0.0",
            snapshot_format_version="1.0.0",
            snapshot_json={},
            snapshot_json_hash="1" * 64,
            version_manifest_hash="2" * 64,
            record_count=0,
            section_count=0,
            field_count=0,
            missing_field_count=0,
            review_required_field_count=0,
            resolved_field_count=0,
            observation_only_field_count=0,
            mixed_field_count=0,
        )
    with pytest.raises(ValueError, match="version_no must be a positive integer"):
        ProjectVersion(
            project_id=_uuid(),
            version_no=True,
            created_by_id=_uuid(),
            creation_kind=ProjectVersionCreationKind.MANUAL.value,
            copied_from_version_id=None,
            reason=None,
            schema_id=_uuid(),
            schema_version_id=_uuid(),
            orchestration_id=_uuid(),
            extraction_run_id=_uuid(),
            consistency_check_application_id=_uuid(),
            source_consistency_application_id=_uuid(),
            schema_definition_manifest_hash="a" * 64,
            ufl_source_manifest_hash="b" * 64,
            consistency_result_manifest_hash="c" * 64,
            raw_projection_manifest_hash="d" * 64,
            reviewed_projection_manifest_hash="e" * 64,
            knowledge_view_manifest_hash="f" * 64,
            knowledge_view_algorithm_name="dynamic_schema_knowledge_view",
            knowledge_view_algorithm_version="1.0.0",
            snapshot_format_version="1.0.0",
            snapshot_json={},
            snapshot_json_hash="1" * 64,
            version_manifest_hash="2" * 64,
            record_count=0,
            section_count=0,
            field_count=0,
            missing_field_count=0,
            review_required_field_count=0,
            resolved_field_count=0,
            observation_only_field_count=0,
            mixed_field_count=0,
        )
    with pytest.raises(ValueError, match="record_count must be a non-negative integer"):
        ProjectVersion(
            project_id=_uuid(),
            version_no=1,
            created_by_id=_uuid(),
            creation_kind=ProjectVersionCreationKind.MANUAL.value,
            copied_from_version_id=None,
            reason=None,
            schema_id=_uuid(),
            schema_version_id=_uuid(),
            orchestration_id=_uuid(),
            extraction_run_id=_uuid(),
            consistency_check_application_id=_uuid(),
            source_consistency_application_id=_uuid(),
            schema_definition_manifest_hash="a" * 64,
            ufl_source_manifest_hash="b" * 64,
            consistency_result_manifest_hash="c" * 64,
            raw_projection_manifest_hash="d" * 64,
            reviewed_projection_manifest_hash="e" * 64,
            knowledge_view_manifest_hash="f" * 64,
            knowledge_view_algorithm_name="dynamic_schema_knowledge_view",
            knowledge_view_algorithm_version="1.0.0",
            snapshot_format_version="1.0.0",
            snapshot_json={},
            snapshot_json_hash="1" * 64,
            version_manifest_hash="2" * 64,
            record_count=False,
            section_count=0,
            field_count=0,
            missing_field_count=0,
            review_required_field_count=0,
            resolved_field_count=0,
            observation_only_field_count=0,
            mixed_field_count=0,
        )
    with pytest.raises(
        ValueError,
        match="knowledge_view_manifest_hash must be a 64-character lowercase hexadecimal string",
    ):
        ProjectVersion(
            project_id=_uuid(),
            version_no=1,
            created_by_id=_uuid(),
            creation_kind=ProjectVersionCreationKind.MANUAL.value,
            copied_from_version_id=None,
            reason=None,
            schema_id=_uuid(),
            schema_version_id=_uuid(),
            orchestration_id=_uuid(),
            extraction_run_id=_uuid(),
            consistency_check_application_id=_uuid(),
            source_consistency_application_id=_uuid(),
            schema_definition_manifest_hash="a" * 64,
            ufl_source_manifest_hash="b" * 64,
            consistency_result_manifest_hash="c" * 64,
            raw_projection_manifest_hash="d" * 64,
            reviewed_projection_manifest_hash="e" * 64,
            knowledge_view_manifest_hash="z" * 64,
            knowledge_view_algorithm_name="dynamic_schema_knowledge_view",
            knowledge_view_algorithm_version="1.0.0",
            snapshot_format_version="1.0.0",
            snapshot_json={},
            snapshot_json_hash="1" * 64,
            version_manifest_hash="2" * 64,
            record_count=0,
            section_count=0,
            field_count=0,
            missing_field_count=0,
            review_required_field_count=0,
            resolved_field_count=0,
            observation_only_field_count=0,
            mixed_field_count=0,
        )
    with pytest.raises(ValueError, match="reason must not be empty"):
        ProjectVersion(
            project_id=_uuid(),
            version_no=1,
            created_by_id=_uuid(),
            creation_kind=ProjectVersionCreationKind.MANUAL.value,
            copied_from_version_id=None,
            reason="   ",
            schema_id=_uuid(),
            schema_version_id=_uuid(),
            orchestration_id=_uuid(),
            extraction_run_id=_uuid(),
            consistency_check_application_id=_uuid(),
            source_consistency_application_id=_uuid(),
            schema_definition_manifest_hash="a" * 64,
            ufl_source_manifest_hash="b" * 64,
            consistency_result_manifest_hash="c" * 64,
            raw_projection_manifest_hash="d" * 64,
            reviewed_projection_manifest_hash="e" * 64,
            knowledge_view_manifest_hash="f" * 64,
            knowledge_view_algorithm_name="dynamic_schema_knowledge_view",
            knowledge_view_algorithm_version="1.0.0",
            snapshot_format_version="1.0.0",
            snapshot_json={},
            snapshot_json_hash="1" * 64,
            version_manifest_hash="2" * 64,
            record_count=0,
            section_count=0,
            field_count=0,
            missing_field_count=0,
            review_required_field_count=0,
            resolved_field_count=0,
            observation_only_field_count=0,
            mixed_field_count=0,
        )
    with pytest.raises(ValueError, match="snapshot_json must be a JSON object"):
        ProjectVersion(
            project_id=_uuid(),
            version_no=1,
            created_by_id=_uuid(),
            creation_kind=ProjectVersionCreationKind.MANUAL.value,
            copied_from_version_id=None,
            reason=None,
            schema_id=_uuid(),
            schema_version_id=_uuid(),
            orchestration_id=_uuid(),
            extraction_run_id=_uuid(),
            consistency_check_application_id=_uuid(),
            source_consistency_application_id=_uuid(),
            schema_definition_manifest_hash="a" * 64,
            ufl_source_manifest_hash="b" * 64,
            consistency_result_manifest_hash="c" * 64,
            raw_projection_manifest_hash="d" * 64,
            reviewed_projection_manifest_hash="e" * 64,
            knowledge_view_manifest_hash="f" * 64,
            knowledge_view_algorithm_name="dynamic_schema_knowledge_view",
            knowledge_view_algorithm_version="1.0.0",
            snapshot_format_version="1.0.0",
            snapshot_json=[],
            snapshot_json_hash="1" * 64,
            version_manifest_hash="2" * 64,
            record_count=0,
            section_count=0,
            field_count=0,
            missing_field_count=0,
            review_required_field_count=0,
            resolved_field_count=0,
            observation_only_field_count=0,
            mixed_field_count=0,
        )


def test_project_version_constraint_and_index_names_fit_postgresql_limit() -> None:
    targeted_existing_constraints = {
        "fk_projects_cur_ver_projver",
        "uq_dynschema_id_project",
        "uq_dynsver_id_schema",
        "uq_feo_id_proj_run",
        "uq_ccapp_id_proj_orch_src",
    }
    project_table = Base.metadata.tables["projects"]
    version_table = Base.metadata.tables["project_versions"]

    for constraint in project_table.constraints:
        if constraint.name in targeted_existing_constraints:
            assert len(constraint.name) <= 63
    for index in project_table.indexes:
        if index.name == "ix_projects_current_version_id":
            assert len(index.name) <= 63

    for table_name in (
        "dynamic_schemas",
        "dynamic_schema_versions",
        "fact_extraction_orchestrations",
        "consistency_check_applications",
    ):
        table = Base.metadata.tables[table_name]
        for constraint in table.constraints:
            if constraint.name in targeted_existing_constraints:
                assert len(constraint.name) <= 63
    for constraint in version_table.constraints:
        if constraint.name is not None:
            assert len(constraint.name) <= 63
    for index in version_table.indexes:
        assert len(index.name) <= 63


def test_project_version_indexes_compile_for_postgresql() -> None:
    dialect = postgresql.dialect()
    for table_name in ("projects", "project_versions"):
        table = Base.metadata.tables[table_name]
        for index in table.indexes:
            sql = str(CreateIndex(index).compile(dialect=dialect))
            assert "CREATE INDEX" in sql


def test_project_version_migration_upgrade_sql_creates_table_and_project_pointer() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["alembic", "upgrade", "202608010500:202608010600", "--sql"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    output = result.stdout
    assert "-- Running upgrade 202608010500 -> 202608010600" in output
    assert "ALTER TABLE dynamic_schemas ADD CONSTRAINT uq_dynschema_id_project" in output
    assert "ALTER TABLE dynamic_schema_versions ADD CONSTRAINT uq_dynsver_id_schema" in output
    assert "ALTER TABLE fact_extraction_orchestrations ADD CONSTRAINT uq_feo_id_proj_run" in output
    assert "ALTER TABLE consistency_check_applications ADD CONSTRAINT uq_ccapp_id_proj_orch_src" in output
    assert "CREATE TABLE project_versions" in output
    assert "ALTER TABLE projects ADD COLUMN current_version_id UUID" in output
    assert "ALTER TABLE projects ADD CONSTRAINT fk_projects_cur_ver_projver" in output


def test_project_version_migration_downgrade_sql_drops_pointer_table_and_helper_uniques_in_reverse_order() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["alembic", "downgrade", "202608010600:202608010500", "--sql"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    output = result.stdout
    assert output.index("ALTER TABLE projects DROP CONSTRAINT fk_projects_cur_ver_projver") < output.index(
        "DROP TABLE project_versions"
    )
    assert "ALTER TABLE consistency_check_applications DROP CONSTRAINT uq_ccapp_id_proj_orch_src" in output
    assert "ALTER TABLE fact_extraction_orchestrations DROP CONSTRAINT uq_feo_id_proj_run" in output
    assert "ALTER TABLE dynamic_schema_versions DROP CONSTRAINT uq_dynsver_id_schema" in output
    assert "ALTER TABLE dynamic_schemas DROP CONSTRAINT uq_dynschema_id_project" in output


def test_project_version_full_history_upgrade_head_sql_succeeds() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["alembic", "upgrade", "head", "--sql"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "-- Running upgrade 202608010500 -> 202608010600" in result.stdout
    assert "CREATE TABLE project_versions" in result.stdout


def test_project_version_migration_source_declares_expected_revision_and_tables() -> None:
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "202608010600_project_versions.py"
    )
    content = migration_path.read_text(encoding="utf-8")

    assert 'revision: str = "202608010600"' in content
    assert 'down_revision: str | None = "202608010500"' in content
    assert '"project_versions"' in content
    assert '"uq_dynschema_id_project"' in content
    assert '"uq_dynsver_id_schema"' in content
    assert '"uq_feo_id_proj_run"' in content
    assert '"uq_ccapp_id_proj_orch_src"' in content
    assert '"fk_projects_cur_ver_projver"' in content
