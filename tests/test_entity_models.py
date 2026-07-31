from pathlib import Path
import uuid

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from pydantic import ValidationError
from sqlalchemy import CheckConstraint, Enum as SAEnum, String, UniqueConstraint

from app.models import Base
from app.models.entity import Entity, EntityAlias, normalize_entity_alias
from app.schemas.entity import EntityAliasCreateInput, EntityCreateInput
from app.services.entity import build_entity_identity_hash


def test_entity_tables_are_registered() -> None:
    assert {"entities", "entity_aliases"} <= set(Base.metadata.tables)


def test_single_migration_head_includes_entity_models() -> None:
    root = Path(__file__).resolve().parents[1]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    script = ScriptDirectory.from_config(config)

    assert list(script.get_heads()) == ["202607311900"]


def test_entity_status_columns_use_string_and_check_not_native_enum() -> None:
    assert isinstance(Entity.__table__.c.status.type, String)
    assert isinstance(EntityAlias.__table__.c.status.type, String)
    assert isinstance(EntityAlias.__table__.c.alias_kind.type, String)
    assert not isinstance(Entity.__table__.c.status.type, SAEnum)
    assert not isinstance(EntityAlias.__table__.c.status.type, SAEnum)
    assert not isinstance(EntityAlias.__table__.c.alias_kind.type, SAEnum)


def test_entity_unique_constraints_exist() -> None:
    table = Base.metadata.tables["entities"]
    unique_columns = {
        tuple(constraint.columns.keys())
        for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint)
    }

    assert ("project_id", "entity_type", "canonical_key") in unique_columns
    assert ("project_id", "identity_hash") in unique_columns


def test_entity_alias_unique_and_partial_primary_constraints_exist() -> None:
    table = Base.metadata.tables["entity_aliases"]
    unique_columns = {
        tuple(constraint.columns.keys())
        for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    assert unique_columns == {("entity_id", "normalized_alias", "language_code")}

    primary_index = next(index for index in table.indexes if index.name == "uq_ea_active_primary")
    assert primary_index.unique is True
    assert tuple(primary_index.columns.keys()) == ("entity_id",)
    assert str(primary_index.dialect_options["postgresql"]["where"]) == "status = 'active' AND is_primary = true"


def test_entity_merge_and_alias_constraints_exist() -> None:
    entity_checks = {
        str(constraint.sqltext)
        for constraint in Base.metadata.tables["entities"].constraints
        if isinstance(constraint, CheckConstraint)
    }
    alias_checks = {
        str(constraint.sqltext)
        for constraint in Base.metadata.tables["entity_aliases"].constraints
        if isinstance(constraint, CheckConstraint)
    }

    assert any("status = 'merged'" in sql and "merged_into_entity_id IS NOT NULL" in sql for sql in entity_checks)
    assert any("merged_into_entity_id <> id" in sql for sql in entity_checks)
    assert any("alias_kind = 'canonical'" in sql and "status = 'active'" in sql for sql in alias_checks)
    assert any("status <> 'retired' OR NOT is_primary" in sql for sql in alias_checks)


def test_relationships_are_wired_without_ambiguity() -> None:
    assert Entity.project.property.back_populates == "entities"
    assert Entity.aliases.property.back_populates == "entity"
    assert Entity.subject_facts.property.back_populates == "subject_entity"
    assert Entity.subject_facts.property.passive_deletes is True
    assert Entity.referenced_fact_values.property.back_populates == "referenced_entity"
    assert Entity.referenced_fact_values.property.passive_deletes is True
    assert Entity.merged_into.property.back_populates == "merged_from"
    assert EntityAlias.entity.property.back_populates == "aliases"
    assert Entity.created_by.property.back_populates == "created_entities"
    assert EntityAlias.created_by.property.back_populates == "created_entity_aliases"


def test_constraint_and_index_names_stay_within_postgres_limit() -> None:
    for table_name in ("entities", "entity_aliases"):
        table = Base.metadata.tables[table_name]
        for constraint in table.constraints:
            if constraint.name is not None:
                assert len(constraint.name) <= 63
        for index in table.indexes:
            assert len(index.name) <= 63


def test_entity_identity_hash_is_deterministic() -> None:
    project_id = uuid.uuid4()
    hash_one = build_entity_identity_hash(
        project_id=project_id,
        entity_type="person",
        canonical_key="zhang san",
    )
    hash_two = build_entity_identity_hash(
        project_id=project_id,
        entity_type="person",
        canonical_key="zhang san",
    )
    changed = build_entity_identity_hash(
        project_id=project_id,
        entity_type="person",
        canonical_key="li si",
    )

    assert hash_one == hash_two
    assert hash_one != changed


def test_entity_identity_hash_normalizes_equivalent_inputs() -> None:
    project_id = uuid.uuid4()
    hash_one = build_entity_identity_hash(
        project_id=project_id,
        entity_type="  Person  ",
        canonical_key="  Ａ\u3000B  ",
    )
    hash_two = build_entity_identity_hash(
        project_id=project_id,
        entity_type="Person",
        canonical_key="a b",
    )

    assert hash_one == hash_two


def test_entity_identity_hash_rejects_non_uuid_project_id() -> None:
    with pytest.raises(ValueError):
        build_entity_identity_hash(
            project_id="not-a-uuid",
            entity_type="person",
            canonical_key="zhang san",
        )


def test_normalize_entity_alias_applies_nfkc_casefold_and_whitespace_rules() -> None:
    assert normalize_entity_alias("  Ａ\u3000B\tC  ") == "a b c"
    assert normalize_entity_alias("Straße") == "strasse"


def test_normalize_entity_alias_rejects_empty_or_too_long_results() -> None:
    with pytest.raises(ValueError):
        normalize_entity_alias("   ")

    with pytest.raises(ValueError):
        normalize_entity_alias("a" * 256)


def test_entity_create_and_alias_input_use_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        EntityCreateInput(
            entity_type="person",
            canonical_key="zhang san",
            display_name="张三",
            identity_hash="x" * 64,
        )

    with pytest.raises(ValidationError):
        EntityAliasCreateInput(
            alias_text="张三",
            normalized_alias="zhang san",
        )


def test_entity_create_input_normalizes_canonical_key() -> None:
    payload = EntityCreateInput(
        entity_type="person",
        canonical_key="  Ａ\u3000B ",
        display_name="Alpha Beta",
    )

    assert payload.canonical_key == "a b"


@pytest.mark.parametrize("invalid_value", [0, 1, "true"])
def test_entity_alias_create_input_requires_strict_bool_for_is_primary(invalid_value: object) -> None:
    with pytest.raises(ValidationError):
        EntityAliasCreateInput(alias_text="张三", is_primary=invalid_value)


def test_entity_model_migration_creates_tables_and_partial_unique_index() -> None:
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "202607311230_entity_models.py"
    )
    content = migration_path.read_text(encoding="utf-8")

    assert 'down_revision: str | None = "202607311030"' in content
    assert 'op.create_table(\n        "entities"' in content
    assert 'op.create_table(\n        "entity_aliases"' in content
    assert 'name="uq_ent_proj_type_key"' in content
    assert 'name="uq_ent_proj_hash"' in content
    assert 'name="uq_ea_ent_norm_lang"' in content
    assert '"uq_ea_active_primary"' in content
    assert "sa.Enum(" not in content
