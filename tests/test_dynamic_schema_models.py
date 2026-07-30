from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import CheckConstraint, UniqueConstraint

from app.models import Base
from app.models.dynamic_schema import DynamicSchema
from app.schemas.dynamic_schema import (
    DynamicSchemaFieldInput,
    DynamicSchemaIdentityInput,
    DynamicSchemaVersionInput,
)


def test_dynamic_schema_tables_are_registered() -> None:
    assert {
        "dynamic_schemas",
        "dynamic_schema_versions",
        "dynamic_schema_fields",
    } <= set(Base.metadata.tables)


def test_dynamic_schema_identity_unique_constraint_exists() -> None:
    table = Base.metadata.tables["dynamic_schemas"]
    assert any(
        isinstance(constraint, UniqueConstraint)
        and tuple(constraint.columns.keys()) == ("project_id", "schema_key")
        for constraint in table.constraints
    )


def test_dynamic_schema_current_version_unique_constraint_exists() -> None:
    table = Base.metadata.tables["dynamic_schemas"]
    assert any(
        isinstance(constraint, UniqueConstraint)
        and tuple(constraint.columns.keys()) == ("current_version_id",)
        for constraint in table.constraints
    )


def test_dynamic_schema_current_version_relationship_uses_post_update() -> None:
    assert DynamicSchema.current_version.property.post_update is True
    assert DynamicSchema.current_version.property.uselist is False


def test_dynamic_schema_version_number_unique_constraint_exists() -> None:
    table = Base.metadata.tables["dynamic_schema_versions"]
    assert any(
        isinstance(constraint, UniqueConstraint)
        and tuple(constraint.columns.keys()) == ("schema_id", "version_no")
        for constraint in table.constraints
    )


def test_dynamic_schema_version_activation_constraints_exist() -> None:
    table = Base.metadata.tables["dynamic_schema_versions"]
    check_sql = {
        str(constraint.sqltext)
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }

    assert any("activated_by_id IS NULL AND activated_at IS NULL" in sql for sql in check_sql)
    assert any("status NOT IN ('draft', 'proposed')" in sql for sql in check_sql)
    assert any("status NOT IN ('active', 'retired')" in sql for sql in check_sql)


def test_dynamic_schema_version_human_source_requires_created_by_constraint_exists() -> None:
    table = Base.metadata.tables["dynamic_schema_versions"]
    check_sql = {
        str(constraint.sqltext)
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }

    assert any("source_kind <> 'human' OR created_by_id IS NOT NULL" in sql for sql in check_sql)


def test_dynamic_schema_field_unique_and_display_order_constraints_exist() -> None:
    table = Base.metadata.tables["dynamic_schema_fields"]
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

    assert ("schema_version_id", "field_key") in unique_columns
    assert any("display_order >= 0" in sql for sql in check_sql)


def test_dynamic_schema_identity_input_rejects_status_version_actor_and_current_fields() -> None:
    with pytest.raises(ValidationError):
        DynamicSchemaIdentityInput(
            schema_key="profile.main",
            name="Profile Main",
            subject_kind="person",
            description=None,
            status="active",
            current_version_id="00000000-0000-0000-0000-000000000000",
            created_by_id="00000000-0000-0000-0000-000000000000",
            version_no=1,
        )


def test_dynamic_schema_version_input_rejects_status_actor_and_current_fields() -> None:
    with pytest.raises(ValidationError):
        DynamicSchemaVersionInput(
            summary="Profile layout",
            layout_config={},
            fields=[
                DynamicSchemaFieldInput(
                    field_key="title",
                    label="Title",
                    predicate_key="person.name",
                    display_order=0,
                )
            ],
            status="active",
            source_kind="human",
            current_version_id="00000000-0000-0000-0000-000000000000",
            created_by_id="00000000-0000-0000-0000-000000000000",
            activated_at="2026-07-30T12:00:00+00:00",
        )


def test_dynamic_schema_version_input_rejects_duplicate_field_keys() -> None:
    with pytest.raises(ValidationError):
        DynamicSchemaVersionInput(
            summary=None,
            layout_config={},
            fields=[
                DynamicSchemaFieldInput(
                    field_key="title",
                    label="Title",
                    predicate_key="person.name",
                    display_order=0,
                ),
                DynamicSchemaFieldInput(
                    field_key="title",
                    label="Alias",
                    predicate_key="person.alias",
                    display_order=1,
                ),
            ],
        )


def test_dynamic_schema_version_input_rejects_duplicate_display_order() -> None:
    with pytest.raises(ValidationError):
        DynamicSchemaVersionInput(
            summary=None,
            layout_config={},
            fields=[
                DynamicSchemaFieldInput(
                    field_key="title",
                    label="Title",
                    predicate_key="person.name",
                    display_order=0,
                ),
                DynamicSchemaFieldInput(
                    field_key="summary",
                    label="Summary",
                    predicate_key="person.summary",
                    display_order=0,
                ),
            ],
        )


def test_dynamic_schema_version_input_allows_at_most_one_title_field() -> None:
    with pytest.raises(ValidationError):
        DynamicSchemaVersionInput(
            summary=None,
            layout_config={},
            fields=[
                DynamicSchemaFieldInput(
                    field_key="title",
                    label="Title",
                    predicate_key="person.name",
                    display_order=0,
                    is_title=True,
                ),
                DynamicSchemaFieldInput(
                    field_key="headline",
                    label="Headline",
                    predicate_key="person.headline",
                    display_order=1,
                    is_title=True,
                ),
            ],
        )


def test_dynamic_schema_field_input_rejects_hidden_title() -> None:
    with pytest.raises(ValidationError):
        DynamicSchemaFieldInput(
            field_key="title",
            label="Title",
            predicate_key="person.name",
            display_order=0,
            is_title=True,
            is_hidden=True,
        )


def test_dynamic_schema_tables_do_not_store_fact_values_or_evidence_fields() -> None:
    forbidden_columns = {"value_json", "fact_value_id", "evidence_id", "source_evidence_id"}

    for table_name in ("dynamic_schemas", "dynamic_schema_versions", "dynamic_schema_fields"):
        column_names = set(Base.metadata.tables[table_name].columns.keys())
        assert forbidden_columns.isdisjoint(column_names)


def test_dynamic_schema_migration_handles_circular_foreign_key_in_order() -> None:
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "202607301800_dynamic_schema_models.py"
    )
    content = migration_path.read_text(encoding="utf-8")

    assert content.index('op.create_table(\n        "dynamic_schemas"') < content.index(
        'op.create_table(\n        "dynamic_schema_versions"'
    )
    assert content.index('op.create_table(\n        "dynamic_schema_versions"') < content.index(
        'op.create_table(\n        "dynamic_schema_fields"'
    )
    assert content.index('op.create_table(\n        "dynamic_schema_fields"') < content.index(
        'op.create_foreign_key(\n        op.f("fk_dynamic_schemas_current_version_id_dynamic_schema_versions")'
    )
