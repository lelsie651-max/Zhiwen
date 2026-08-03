from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TypeVar
from urllib.parse import urlsplit, urlunsplit
import uuid

from alembic import command
from alembic.config import Config
import psycopg
from psycopg import sql
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings, get_settings
from app.main import create_app
from app.models.document import Document
from app.models.project import Project
from app.models.project_version import ProjectVersion
from app.models.user import User
from app.services import bailian_demo_seed as seed_service
from app.services import bailian_review_tools
from app.services import consistency_review as consistency_review_service
from scripts import seed_bailian_demo as seed_cli


def run_async(awaitable):
    return asyncio.run(awaitable)


@dataclass(frozen=True, slots=True)
class DemoDatabase:
    name: str
    async_url: str
    sync_url: str


@dataclass(frozen=True, slots=True)
class SeededDemo:
    database: DemoDatabase
    result: seed_service.BailianDemoSeedResult


T = TypeVar("T")


def _replace_database_name(url: str, db_name: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, f"/{db_name}", parsed.query, parsed.fragment))


def _as_psycopg_conninfo(url: str) -> str:
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


def _create_demo_database(name: str) -> DemoDatabase:
    settings = Settings()
    async_url = _replace_database_name(settings.database_url, name)
    sync_url = _replace_database_name(settings.database_url_sync, name)
    admin_url = _replace_database_name(settings.database_url_sync, "postgres")
    root = Path(__file__).resolve().parents[1]

    with psycopg.connect(_as_psycopg_conninfo(admin_url), autocommit=True) as connection:
        connection.execute(
            sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
        )
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))

    previous_database_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = async_url
    get_settings.cache_clear()
    try:
        config = Config(str(root / "alembic.ini"))
        config.set_main_option("script_location", str(root / "alembic"))
        config.set_main_option("sqlalchemy.url", sync_url)
        command.upgrade(config, "head")
    finally:
        if previous_database_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous_database_url
        get_settings.cache_clear()

    return DemoDatabase(name=name, async_url=async_url, sync_url=sync_url)


def _drop_demo_database(database: DemoDatabase) -> None:
    admin_url = _replace_database_name(Settings().database_url_sync, "postgres")
    with psycopg.connect(_as_psycopg_conninfo(admin_url), autocommit=True) as connection:
        connection.execute(
            sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(database.name))
        )


async def _run_with_session_factory(
    database: DemoDatabase,
    operation: Callable[[async_sessionmaker[AsyncSession]], Awaitable[T]],
) -> T:
    # Each async call gets its own engine so Windows asyncpg connections never
    # outlive the event loop created by asyncio.run() in these sync tests.
    engine = create_async_engine(database.async_url, pool_pre_ping=True)
    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    try:
        return await operation(session_factory)
    finally:
        await engine.dispose()


def run_with_database(
    database: DemoDatabase,
    operation: Callable[[async_sessionmaker[AsyncSession]], Awaitable[T]],
) -> T:
    return run_async(_run_with_session_factory(database, operation))


@pytest.fixture
def demo_database() -> DemoDatabase:
    database = _create_demo_database(f"zhiwen_test_bailian_seed_{uuid.uuid4().hex[:8]}")
    try:
        yield database
    finally:
        _drop_demo_database(database)


@pytest.fixture
def seeded_demo(demo_database: DemoDatabase) -> SeededDemo:
    result = run_with_database(demo_database, seed_service.seed_bailian_demo)
    return SeededDemo(database=demo_database, result=result)


def _tool_args(result: seed_service.BailianDemoSeedResult) -> dict[str, str]:
    return {
        "project_id": str(result.project_id),
        "schema_id": str(result.schema_id),
        "schema_version_id": str(result.schema_version_id),
        "orchestration_id": str(result.orchestration_id),
        "consistency_check_application_id": str(result.consistency_check_application_id),
    }


def test_seed_rejects_production_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        seed_service,
        "get_settings",
        lambda: SimpleNamespace(is_production=True),
    )

    with pytest.raises(seed_service.BailianDemoSeedStateError, match="bailian_demo_seed_production_forbidden"):
        run_async(seed_service.seed_bailian_demo(lambda: None))


def test_cli_requires_confirm_local_demo_flag() -> None:
    with pytest.raises(SystemExit, match="bailian_demo_seed_confirmation_required"):
        run_async(seed_cli._main_async(confirm_local_demo=False))


def test_app_does_not_register_seed_http_route() -> None:
    app = create_app()
    paths = {route.path for route in app.routes if hasattr(route, "path")}

    assert all("seed" not in path for path in paths)


def test_seed_source_does_not_patch_global_uuid_or_use_unittest_mock() -> None:
    source = (Path(__file__).resolve().parents[1] / "app" / "services" / "bailian_demo_seed.py").read_text(
        encoding="utf-8"
    )

    assert "unittest.mock" not in source
    assert "patch(\"uuid.uuid4\"" not in source
    assert "_DeterministicUUID4" not in source
    assert "_patch_deterministic_uuid4" not in source


def test_fact_repository_lock_queries_use_selectinload_for_parent_fact() -> None:
    get_value_source = (Path(__file__).resolve().parents[1] / "app" / "repositories" / "fact.py").read_text(
        encoding="utf-8"
    )

    assert "selectinload(FactValue.fact)" in get_value_source
    assert "get_fact_value_for_update" in get_value_source


def test_consistency_review_member_snapshot_falls_back_to_authoritative_source_ids() -> None:
    source_application_id = uuid.uuid4()
    source_candidate_id = uuid.uuid4()
    member = SimpleNamespace(
        fact_value_id=uuid.uuid4(),
        source_batch_id=uuid.uuid4(),
        semantic_key_hash="a" * 64,
    )

    snapshot = consistency_review_service._build_member_snapshot(
        (member,),
        source_consistency_application_id=source_application_id,
        source_consistency_candidate_id=source_candidate_id,
    )

    assert snapshot[0].consistency_application_id == source_application_id
    assert snapshot[0].candidate_id == source_candidate_id


def test_empty_database_first_seed_succeeds(demo_database: DemoDatabase) -> None:
    result = run_with_database(demo_database, seed_service.seed_bailian_demo)

    assert result.created_new is True
    assert result.version_record_subject_key == "织文/企业知识库"


def test_second_seed_is_idempotent_and_reuses_same_database_ids(demo_database: DemoDatabase) -> None:
    first = run_with_database(demo_database, seed_service.seed_bailian_demo)
    second = run_with_database(demo_database, seed_service.seed_bailian_demo)

    assert first.created_new is True
    assert second.created_new is False
    assert second.user_id == first.user_id
    assert second.project_id == first.project_id
    assert second.schema_id == first.schema_id
    assert second.schema_version_id == first.schema_version_id
    assert second.orchestration_id == first.orchestration_id
    assert second.consistency_check_application_id == first.consistency_check_application_id
    assert second.project_version_id == first.project_version_id
    assert second.pending_review_fact_id == first.pending_review_fact_id
    assert second.resolved_fact_id == first.resolved_fact_id
    assert second.observation_only_fact_id == first.observation_only_fact_id


@pytest.mark.parametrize("partial_kind", ["user_only", "user_and_project_only"])
def test_partial_existing_seed_rows_fail_closed(
    demo_database: DemoDatabase,
    partial_kind: str,
) -> None:
    seed_id = "bailian-demo-v1"
    expected_user_id = seed_service._seed_uuid(seed_id, "user")
    expected_project_id = seed_service._seed_uuid(seed_id, "project")

    async def _prepare_with_factory(session_factory: async_sessionmaker[AsyncSession]) -> None:
        async with session_factory() as session:
            user = User(
                id=expected_user_id,
                handle="bailian_demo",
                display_name="百炼联调演示用户",
                email="bailian-demo@example.com",
                status="active",
            )
            session.add(user)
            if partial_kind == "user_and_project_only":
                session.add(
                    Project(
                        id=expected_project_id,
                        name="织文百炼联调演示",
                        slug="zhiwen-bailian-demo",
                        description="partial",
                        visibility="private",
                        status="active",
                        current_version_id=None,
                        created_by_id=user.id,
                    )
                )
            await session.commit()

    run_with_database(demo_database, _prepare_with_factory)

    with pytest.raises(seed_service.BailianDemoSeedInconsistentError, match="bailian_demo_seed_inconsistent"):
        run_with_database(demo_database, seed_service.seed_bailian_demo)


@pytest.mark.parametrize("mutator_name", ["clear_project_current_version", "clear_document_current_revision"])
def test_seed_fails_closed_when_bindings_or_current_pointers_drift(
    seeded_demo: SeededDemo,
    mutator_name: str,
) -> None:
    async def _mutate(session_factory: async_sessionmaker[AsyncSession]) -> None:
        async with session_factory() as session:
            project = await session.get(Project, seeded_demo.result.project_id)
            document = await session.get(
                Document,
                seed_service._seed_uuid(seeded_demo.result.seed_id, "document-primary"),
            )
            version = await session.get(ProjectVersion, seeded_demo.result.project_version_id)
            assert project is not None
            assert document is not None
            assert version is not None

            if mutator_name == "clear_project_current_version":
                project.current_version_id = None
            else:
                document.current_revision_id = None

            await session.commit()

    run_with_database(seeded_demo.database, _mutate)

    with pytest.raises(seed_service.BailianDemoSeedInconsistentError, match="bailian_demo_seed_inconsistent"):
        run_with_database(seeded_demo.database, seed_service.seed_bailian_demo)


def test_review_item_lists_return_expected_fact_ids(seeded_demo: SeededDemo) -> None:
    tool_args = _tool_args(seeded_demo.result)
    review_required = run_with_database(
        seeded_demo.database,
        lambda session_factory: bailian_review_tools.list_review_items(
            session_factory,
            **tool_args,
            state="review_required",
            limit=20,
        )
    )
    resolved = run_with_database(
        seeded_demo.database,
        lambda session_factory: bailian_review_tools.list_review_items(
            session_factory,
            **tool_args,
            state="resolved",
            limit=20,
        )
    )
    observation = run_with_database(
        seeded_demo.database,
        lambda session_factory: bailian_review_tools.list_review_items(
            session_factory,
            **tool_args,
            state="observation_only",
            limit=20,
        )
    )

    assert {item.fact_id for item in review_required.items} == {seeded_demo.result.pending_review_fact_id}
    assert {item.fact_id for item in resolved.items} == {seeded_demo.result.resolved_fact_id}
    assert {item.fact_id for item in observation.items} == {seeded_demo.result.observation_only_fact_id}


def test_pending_detail_has_two_groups_and_evidence(seeded_demo: SeededDemo) -> None:
    detail = run_with_database(
        seeded_demo.database,
        lambda session_factory: bailian_review_tools.get_review_item_detail(
            session_factory,
            **_tool_args(seeded_demo.result),
            fact_id=str(seeded_demo.result.pending_review_fact_id),
        )
    )

    assert detail.review_state == "pending_review"
    assert detail.semantic_value_count == 2
    assert len(detail.value_groups) == 2
    assert all(value_group["evidences"] for value_group in detail.value_groups)


def test_resolved_detail_only_exposes_selected_effective_value(seeded_demo: SeededDemo) -> None:
    detail = run_with_database(
        seeded_demo.database,
        lambda session_factory: bailian_review_tools.get_review_item_detail(
            session_factory,
            **_tool_args(seeded_demo.result),
            fact_id=str(seeded_demo.result.resolved_fact_id),
        )
    )

    assert detail.review_state == "resolved"
    assert detail.resolution_basis == "human_selection"
    assert detail.current_decision_kind == "select_one"
    assert len(detail.effective_fact_value_ids) == 1


def test_version_record_supports_enterprise_knowledge_subject_key(seeded_demo: SeededDemo) -> None:
    record = run_with_database(
        seeded_demo.database,
        lambda session_factory: bailian_review_tools.get_version_record(
            session_factory,
            project_id=str(seeded_demo.result.project_id),
            project_version_id=str(seeded_demo.result.project_version_id),
            subject_key="织文/企业知识库",
        )
    )

    assert record.subject_key == "织文/企业知识库"
    assert record.record_json["subject_key"] == "织文/企业知识库"


def test_cli_json_does_not_expose_tokens_database_url_or_source_content(seeded_demo: SeededDemo) -> None:
    payload = json.dumps(
        seed_cli._to_jsonable(seeded_demo.result),
        ensure_ascii=False,
        sort_keys=True,
    )

    assert "database_url" not in payload
    assert Settings().database_url not in payload
    assert "token" not in payload.lower()
    assert "excerpt" not in payload
    assert "在线工单" not in payload
    assert "华北2（北京）" not in payload
    assert "2026-08-08" not in payload


def test_seed_creates_single_document_single_revision_cross_batch_demo(seeded_demo: SeededDemo) -> None:
    async def _load_document_count(session_factory: async_sessionmaker[AsyncSession]) -> int:
        async with session_factory() as session:
            documents = list(
                (
                    await session.execute(
                        select(Document).where(Document.project_id == seeded_demo.result.project_id)
                    )
                )
                .scalars()
                .all()
            )
            return len(documents)

    assert run_with_database(seeded_demo.database, _load_document_count) == 1
