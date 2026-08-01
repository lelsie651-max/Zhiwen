from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import CheckConstraint, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.schema import CreateTable

from app.models import Base
from app.models.entity import Entity
from app.models.fact import Fact, FactEvidenceLink, FactValue
from app.models.fact_value_duplicate_grouping import (
    FactValueDuplicateGroup,
    FactValueDuplicateGroupMember,
    FactValueDuplicateGroupingApplication,
)
from app.schemas.fact import FactIdentityInput, FactRead, FactValueInput, FactValueRead


def test_fact_tables_are_registered() -> None:
    assert {"facts", "fact_values", "fact_evidence_links"} <= set(Base.metadata.tables)


def test_duplicate_grouping_tables_are_registered() -> None:
    assert {
        "fact_value_duplicate_grouping_applications",
        "fact_value_duplicate_groups",
        "fact_value_duplicate_group_members",
    } <= set(Base.metadata.tables)


def test_fact_identity_unique_constraint_exists() -> None:
    table = Base.metadata.tables["facts"]
    assert any(
        isinstance(constraint, UniqueConstraint)
        and tuple(constraint.columns.keys()) == ("project_id", "identity_hash")
        for constraint in table.constraints
    )


def test_fact_current_value_unique_constraint_exists() -> None:
    table = Base.metadata.tables["facts"]
    assert any(
        isinstance(constraint, UniqueConstraint)
        and tuple(constraint.columns.keys()) == ("current_value_id",)
        for constraint in table.constraints
    )


def test_fact_current_value_relationship_uses_post_update() -> None:
    assert Fact.current_value.property.post_update is True


def test_fact_subject_entity_foreign_key_uses_restrict_and_is_indexed() -> None:
    table = Base.metadata.tables["facts"]
    subject_entity_fk = next(
        constraint
        for constraint in table.foreign_key_constraints
        if tuple(constraint.column_keys) == ("subject_entity_id",)
    )
    indexes = {tuple(index.columns.keys()) for index in table.indexes}

    assert subject_entity_fk.ondelete == "RESTRICT"
    assert ("subject_entity_id",) in indexes


def test_fact_value_version_unique_constraint_exists() -> None:
    table = Base.metadata.tables["fact_values"]
    assert any(
        isinstance(constraint, UniqueConstraint)
        and tuple(constraint.columns.keys()) == ("fact_id", "version_no")
        for constraint in table.constraints
    )


def test_fact_value_value_type_and_null_rules_exist() -> None:
    table = Base.metadata.tables["fact_values"]
    check_sql = {
        str(constraint.sqltext)
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }

    assert any("value_type IN ('string', 'number', 'boolean', 'date', 'datetime', 'entity_ref', 'list', 'object', 'null')" in sql for sql in check_sql)
    assert any("value_type = 'null'" in sql and "value_json IS NULL" in sql for sql in check_sql)


def test_fact_value_value_json_uses_none_as_null() -> None:
    value_json_type = FactValue.__table__.c.value_json.type

    assert isinstance(value_json_type, JSONB)
    assert value_json_type.none_as_null is True


def test_fact_value_human_source_requires_created_by_constraint_exists() -> None:
    table = Base.metadata.tables["fact_values"]
    check_sql = {
        str(constraint.sqltext)
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }
    assert any("source_kind <> 'human' OR created_by_id IS NOT NULL" in sql for sql in check_sql)


def test_fact_value_ai_source_requires_extraction_run_constraint_exists() -> None:
    table = Base.metadata.tables["fact_values"]
    check_sql = {
        str(constraint.sqltext)
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }
    assert any(
        "source_kind <> 'ai' OR (extraction_run_id IS NOT NULL AND inference_run_id IS NOT NULL)"
        in sql
        for sql in check_sql
    )


def test_fact_value_non_ai_source_forbids_inference_run_constraint_exists() -> None:
    table = Base.metadata.tables["fact_values"]
    check_sql = {
        str(constraint.sqltext)
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }
    assert any("source_kind = 'ai' OR inference_run_id IS NULL" in sql for sql in check_sql)


def test_fact_value_extraction_run_foreign_key_uses_restrict() -> None:
    table = Base.metadata.tables["fact_values"]
    extraction_run_fk = next(
        constraint
        for constraint in table.foreign_key_constraints
        if tuple(constraint.column_keys) == ("extraction_run_id",)
    )

    assert extraction_run_fk.ondelete == "RESTRICT"


def test_fact_value_inference_run_foreign_key_uses_restrict() -> None:
    table = Base.metadata.tables["fact_values"]
    inference_run_fk = next(
        constraint
        for constraint in table.foreign_key_constraints
        if tuple(constraint.column_keys) == ("inference_run_id",)
    )

    assert inference_run_fk.ondelete == "RESTRICT"


def test_fact_value_inference_run_column_is_indexed() -> None:
    table = Base.metadata.tables["fact_values"]
    indexes = {tuple(index.columns.keys()) for index in table.indexes}
    assert ("inference_run_id",) in indexes


def test_fact_value_referenced_entity_foreign_key_uses_restrict_and_is_indexed() -> None:
    table = Base.metadata.tables["fact_values"]
    referenced_entity_fk = next(
        constraint
        for constraint in table.foreign_key_constraints
        if tuple(constraint.column_keys) == ("referenced_entity_id",)
    )
    indexes = {tuple(index.columns.keys()) for index in table.indexes}

    assert referenced_entity_fk.ondelete == "RESTRICT"
    assert ("referenced_entity_id",) in indexes


def test_fact_value_and_inference_run_relationships_are_bidirectional() -> None:
    assert FactValue.inference_run.property.back_populates == "fact_values"
    assert FactValue.inference_run.property._user_defined_foreign_keys
    assert "inference_run_id" in {
        column.key for column in FactValue.inference_run.property._user_defined_foreign_keys
    }


def test_fact_and_entity_relationships_are_bidirectional_without_ambiguity() -> None:
    assert Fact.subject_entity.property.back_populates == "subject_facts"
    assert Entity.subject_facts.property.back_populates == "subject_entity"
    assert Fact.subject_entity.property._user_defined_foreign_keys
    assert "subject_entity_id" in {
        column.key for column in Fact.subject_entity.property._user_defined_foreign_keys
    }
    assert FactValue.referenced_entity.property.back_populates == "referenced_fact_values"
    assert Entity.referenced_fact_values.property.back_populates == "referenced_entity"
    assert FactValue.referenced_entity.property._user_defined_foreign_keys
    assert "referenced_entity_id" in {
        column.key for column in FactValue.referenced_entity.property._user_defined_foreign_keys
    }


def test_fact_value_read_exposes_only_inference_run_id_for_ai_provenance() -> None:
    assert "inference_run_id" in FactValueRead.model_fields
    assert "referenced_entity_id" in FactValueRead.model_fields
    assert "response_json" not in FactValueRead.model_fields
    assert "failure_message" not in FactValueRead.model_fields
    assert "prompt" not in FactValueRead.model_fields


def test_fact_read_exposes_only_subject_entity_id_for_subject_link() -> None:
    assert "subject_entity_id" in FactRead.model_fields
    assert "aliases" not in FactRead.model_fields
    assert "merged_from" not in FactRead.model_fields


def test_fact_value_decision_constraints_exist() -> None:
    table = Base.metadata.tables["fact_values"]
    check_sql = {
        str(constraint.sqltext)
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }
    assert any("decided_by_id IS NULL AND decided_at IS NULL" in sql for sql in check_sql)
    assert any("status <> 'proposed'" in sql for sql in check_sql)
    assert any("status NOT IN ('accepted', 'rejected')" in sql for sql in check_sql)


def test_fact_value_entity_ref_pair_constraint_exists_and_matches_model_sql() -> None:
    table = Base.metadata.tables["fact_values"]
    pair_checks = [
        str(constraint.sqltext)
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint) and constraint.name.endswith("ck_fv_entity_ref_pair")
    ]

    assert pair_checks == [
        "((value_type = 'entity_ref' AND referenced_entity_id IS NOT NULL) OR "
        "(value_type <> 'entity_ref' AND referenced_entity_id IS NULL))"
    ]


def test_fact_evidence_link_unique_and_source_order_constraints_exist() -> None:
    table = Base.metadata.tables["fact_evidence_links"]
    unique_columns = {
        tuple(constraint.columns.keys())
        for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    check_sql = {
        str(constraint.sqltext)
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }

    assert ("fact_value_id", "evidence_id", "role") in unique_columns
    assert any("source_order >= 0" in sql for sql in check_sql)


def test_fact_identity_input_rejects_identity_hash_and_status_fields() -> None:
    with pytest.raises(ValidationError):
        FactIdentityInput(
            subject_kind="company",
            subject_key="acme",
            predicate_key="name",
            scope_key=None,
            identity_hash="a" * 64,
            status="active",
        )


def test_fact_identity_input_allows_nullable_subject_entity_id() -> None:
    payload = FactIdentityInput(
        subject_kind="company",
        subject_key="acme",
        subject_entity_id=None,
        predicate_key="legal_name",
        scope_key=None,
    )

    assert payload.subject_entity_id is None


def test_fact_value_input_rejects_hash_status_version_and_actor_fields() -> None:
    with pytest.raises(ValidationError):
        FactValueInput(
            value_type="string",
            value_json="Acme",
            language_code="zh-CN",
            confidence=0.9,
            value_hash="a" * 64,
            version_no=1,
            status="accepted",
            source_kind="ai",
            current_value_id="00000000-0000-0000-0000-000000000000",
            created_by_id="00000000-0000-0000-0000-000000000000",
            decided_by_id="00000000-0000-0000-0000-000000000000",
        )


def test_fact_value_input_validates_value_type_and_null_rule() -> None:
    payload = FactValueInput(
        value_type="null",
        value_json=None,
        language_code=None,
        confidence=None,
    )

    assert payload.value_json is None

    with pytest.raises(ValidationError):
        FactValueInput(
            value_type="null",
            value_json={"unexpected": True},
            language_code=None,
            confidence=None,
        )

    with pytest.raises(ValidationError):
        FactValueInput(
            value_type="string",
            value_json=None,
            language_code=None,
            confidence=None,
        )


def test_fact_value_input_validates_entity_ref_pair_rule() -> None:
    with pytest.raises(ValidationError):
        FactValueInput(
            value_type="entity_ref",
            value_json={"kind": "company", "key": "acme"},
            referenced_entity_id=None,
            language_code=None,
            confidence=None,
        )

    with pytest.raises(ValidationError):
        FactValueInput(
            value_type="string",
            value_json="Acme",
            referenced_entity_id="00000000-0000-0000-0000-000000000000",
            language_code=None,
            confidence=None,
        )


def test_fact_value_input_validates_confidence_range() -> None:
    with pytest.raises(ValidationError):
        FactValueInput(
            value_type="number",
            value_json=1,
            language_code=None,
            confidence=1.5,
        )


def test_fact_migration_handles_circular_foreign_key_in_order() -> None:
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "202607301630_fact_models.py"
    )
    content = migration_path.read_text(encoding="utf-8")

    assert content.index('op.create_table(\n        "facts"') < content.index(
        'op.create_table(\n        "fact_values"'
    )
    assert content.index('op.create_table(\n        "fact_values"') < content.index(
        'op.create_table(\n        "fact_evidence_links"'
    )
    assert content.index('op.create_table(\n        "fact_evidence_links"') < content.index(
        'op.create_foreign_key(\n        op.f("fk_facts_current_value_id_fact_values")'
    )


def test_fact_migration_uses_restrict_for_ai_extraction_run_foreign_key() -> None:
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "202607301630_fact_models.py"
    )
    content = migration_path.read_text(encoding="utf-8")

    assert 'name=op.f("fk_fact_values_extraction_run_id_extraction_runs"),\n            ondelete="RESTRICT"' in content


def test_fact_value_inference_provenance_migration_adds_fk_and_constraints() -> None:
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "202607311030_fact_value_inference_provenance.py"
    )
    content = migration_path.read_text(encoding="utf-8")

    assert 'down_revision: str | None = "202607310330"' in content
    assert 'sa.Column("inference_run_id", sa.Uuid(), nullable=True)' in content
    assert 'op.f("ix_fact_values_inference_run_id")' in content
    assert 'op.f("fk_fact_values_inference_run_id_inference_runs")' in content
    assert "Cannot add fact_value inference provenance" in content
    assert "explicit inference_run_id backfill" in content
    assert "extraction_run_id IS NOT NULL AND inference_run_id IS NOT NULL" in content
    assert "source_kind = 'ai' OR inference_run_id IS NULL" in content


def test_fact_value_inference_provenance_migration_downgrade_restores_old_constraint() -> None:
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "202607311030_fact_value_inference_provenance.py"
    )
    content = migration_path.read_text(encoding="utf-8")

    assert 'op.drop_constraint(op.f(_NON_AI_CHECK), "fact_values", type_="check")' in content
    assert 'op.drop_constraint(op.f(_NEW_AI_CHECK), "fact_values", type_="check")' in content
    assert "\"(source_kind <> 'ai' OR extraction_run_id IS NOT NULL)\"" in content
    assert 'op.drop_column("fact_values", "inference_run_id")' in content


def test_fact_entity_links_migration_adds_columns_constraints_and_safe_abort() -> None:
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "202607311430_fact_entity_links.py"
    )
    content = migration_path.read_text(encoding="utf-8")

    assert 'down_revision: str | None = "202607311230"' in content
    assert 'sa.Column("subject_entity_id", sa.Uuid(), nullable=True)' in content
    assert 'sa.Column("referenced_entity_id", sa.Uuid(), nullable=True)' in content
    assert 'op.f("fk_facts_subject_entity_id_entities")' in content
    assert 'op.f("fk_fact_values_referenced_entity_id_entities")' in content
    assert "Cannot add fact entity links" in content
    assert "explicit referenced_entity_id backfill" in content
    assert "((value_type = 'entity_ref' AND referenced_entity_id IS NOT NULL) OR " in content


def test_fact_entity_links_migration_downgrade_is_reversible() -> None:
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "202607311430_fact_entity_links.py"
    )
    content = migration_path.read_text(encoding="utf-8")

    assert 'op.drop_constraint(op.f(_ENTITY_REF_PAIR_CHECK), "fact_values", type_="check")' in content
    assert 'op.drop_constraint(' in content
    assert 'op.drop_index(op.f("ix_fact_values_referenced_entity_id"), table_name="fact_values")' in content
    assert 'op.drop_index(op.f("ix_facts_subject_entity_id"), table_name="facts")' in content
    assert 'op.drop_column("fact_values", "referenced_entity_id")' in content
    assert 'op.drop_column("facts", "subject_entity_id")' in content


def test_fact_migrations_include_inference_provenance_followup() -> None:
    versions_dir = Path(__file__).resolve().parents[1] / "alembic" / "versions"
    expected_migrations = [
        "202607301630_fact_models.py",
        "202607311030_fact_value_inference_provenance.py",
        "202607311430_fact_entity_links.py",
        "202607311800_fact_value_inference_replay.py",
        "202607312200_fact_extraction_orchestration.py",
        "202607312230_orchestration_recovery_hardening.py",
        "202608010100_fact_value_duplicate_grouping.py",
    ]
    fact_migrations = sorted(
        path.name
        for path in versions_dir.glob("*.py")
        if path.name in expected_migrations
    )

    assert fact_migrations == expected_migrations


def test_fact_tables_compile_with_postgresql_offline_ddl() -> None:
    dialect = postgresql.dialect()

    facts_sql = str(CreateTable(Fact.__table__).compile(dialect=dialect))
    fact_values_sql = str(CreateTable(FactValue.__table__).compile(dialect=dialect))
    dupgrp_app_sql = str(CreateTable(FactValueDuplicateGroupingApplication.__table__).compile(dialect=dialect))
    dupgrp_group_sql = str(CreateTable(FactValueDuplicateGroup.__table__).compile(dialect=dialect))
    dupgrp_member_sql = str(CreateTable(FactValueDuplicateGroupMember.__table__).compile(dialect=dialect))

    assert "subject_entity_id" in facts_sql
    assert "referenced_entity_id" in fact_values_sql
    assert "ck_fv_entity_ref_pair" in fact_values_sql
    assert "uq_dupgrp_app_run_alg" in dupgrp_app_sql
    assert "uq_dupgrp_group_app_key" in dupgrp_group_sql
    assert "uq_dupgrp_member_app_fv" in dupgrp_member_sql
    assert "fk_dupgrp_member_group_id_grouping_application_id_dupgrp_group" in dupgrp_member_sql


def test_duplicate_grouping_migration_adds_tables_constraints_and_indexes() -> None:
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "202608010100_fact_value_duplicate_grouping.py"
    )
    content = migration_path.read_text(encoding="utf-8")

    assert 'down_revision: str | None = "202607312230"' in content
    assert '"fact_value_duplicate_grouping_applications"' in content
    assert '"fact_value_duplicate_groups"' in content
    assert '"fact_value_duplicate_group_members"' in content
    assert 'name="uq_dupgrp_app_run_alg"' in content
    assert 'name="uq_dupgrp_group_app_key"' in content
    assert 'name="uq_dupgrp_member_app_fv"' in content
    assert 'name="uq_dupgrp_member_group_fv"' in content
    assert '"ix_dupgrp_app_extraction_run_id"' in content
    assert '"ix_dupgrp_member_source_batch_id"' in content


def test_duplicate_grouping_migration_downgrade_drops_tables_in_dependency_order() -> None:
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "202608010100_fact_value_duplicate_grouping.py"
    )
    content = migration_path.read_text(encoding="utf-8")

    assert content.index('op.drop_table("fact_value_duplicate_group_members")') < content.index(
        'op.drop_table("fact_value_duplicate_groups")'
    )
    assert content.index('op.drop_table("fact_value_duplicate_groups")') < content.index(
        'op.drop_table("fact_value_duplicate_grouping_applications")'
    )
