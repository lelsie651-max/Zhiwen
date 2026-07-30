import uuid
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import CheckConstraint, UniqueConstraint

from app.models import Base
from app.models.document import Document
from app.models.document_revision import DocumentRevision
from app.schemas.document import DocumentCreate
from app.schemas.document_revision import (
    DocumentRevisionCreate,
    DocumentRevisionInternalRead,
    DocumentRevisionRead,
)


def test_document_tables_are_registered() -> None:
    assert {"documents", "document_revisions"} <= set(Base.metadata.tables)


def test_document_revision_composite_unique_constraint_exists() -> None:
    document_revisions_table = Base.metadata.tables["document_revisions"]
    assert any(
        isinstance(constraint, UniqueConstraint)
        and tuple(constraint.columns.keys()) == ("document_id", "revision_no")
        for constraint in document_revisions_table.constraints
    )


def test_storage_key_unique_constraint_exists() -> None:
    document_revisions_table = Base.metadata.tables["document_revisions"]
    assert any(
        isinstance(constraint, UniqueConstraint)
        and tuple(constraint.columns.keys()) == ("storage_key",)
        for constraint in document_revisions_table.constraints
    )


def test_supersedes_revision_unique_constraint_exists() -> None:
    document_revisions_table = Base.metadata.tables["document_revisions"]
    assert any(
        isinstance(constraint, UniqueConstraint)
        and tuple(constraint.columns.keys()) == ("supersedes_revision_id",)
        for constraint in document_revisions_table.constraints
    )


def test_revision_number_and_upload_intent_check_exists() -> None:
    document_revisions_table = Base.metadata.tables["document_revisions"]
    check_sql = {
        str(constraint.sqltext)
        for constraint in document_revisions_table.constraints
        if isinstance(constraint, CheckConstraint)
    }

    assert any(
        "revision_no = 1" in sql
        and "upload_intent = 'new_document'" in sql
        and "revision_no > 1" in sql
        and "upload_intent = 'revision'" in sql
        for sql in check_sql
    )


def test_revision_supersedes_presence_check_exists() -> None:
    document_revisions_table = Base.metadata.tables["document_revisions"]
    check_sql = {
        str(constraint.sqltext)
        for constraint in document_revisions_table.constraints
        if isinstance(constraint, CheckConstraint)
    }

    assert any(
        "revision_no = 1" in sql
        and "supersedes_revision_id IS NULL" in sql
        and "revision_no > 1" in sql
        and "supersedes_revision_id IS NOT NULL" in sql
        for sql in check_sql
    )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("sha256", "not-a-sha256"),
        ("file_size_bytes", 0),
        ("language_confidence", 1.5),
        ("status", "unknown"),
    ],
)
def test_invalid_revision_fields_are_rejected(field_name: str, field_value: object) -> None:
    payload = {
        "document_id": uuid.uuid4(),
        "revision_no": 1,
        "upload_intent": "new_document",
        "original_filename": "meeting-notes.pdf",
        "storage_key": "documents/01/meeting-notes.bin",
        "mime_type": "application/pdf",
        "file_size_bytes": 1024,
        "sha256": "a" * 64,
    }
    payload[field_name] = field_value

    with pytest.raises(ValidationError):
        DocumentRevisionCreate(**payload)


def test_document_create_rejects_project_creator_and_current_revision_fields() -> None:
    for field_name in ("project_id", "created_by_id", "current_revision_id"):
        with pytest.raises(ValidationError):
            DocumentCreate(
                title="会议纪要",
                **{field_name: uuid.uuid4()},
            )


def test_document_revision_create_rejects_uploaded_by_id() -> None:
    with pytest.raises(ValidationError):
        DocumentRevisionCreate(
            document_id=uuid.uuid4(),
            revision_no=1,
            upload_intent="new_document",
            original_filename="meeting-notes.pdf",
            storage_key="documents/01/meeting-notes.bin",
            mime_type="application/pdf",
            file_size_bytes=1024,
            sha256="a" * 64,
            uploaded_by_id=uuid.uuid4(),
        )


def test_revision_one_rejects_supersedes_revision_id() -> None:
    with pytest.raises(ValidationError):
        DocumentRevisionCreate(
            document_id=uuid.uuid4(),
            revision_no=1,
            upload_intent="new_document",
            supersedes_revision_id=uuid.uuid4(),
            original_filename="meeting-notes.pdf",
            storage_key="documents/01/meeting-notes.bin",
            mime_type="application/pdf",
            file_size_bytes=1024,
            sha256="a" * 64,
        )


def test_revision_above_one_requires_supersedes_revision_id() -> None:
    with pytest.raises(ValidationError):
        DocumentRevisionCreate(
            document_id=uuid.uuid4(),
            revision_no=2,
            upload_intent="revision",
            original_filename="meeting-notes-v2.pdf",
            storage_key="documents/02/meeting-notes.bin",
            mime_type="application/pdf",
            file_size_bytes=1024,
            sha256="a" * 64,
        )


@pytest.mark.parametrize(
    ("revision_no", "upload_intent"),
    [
        (1, "revision"),
        (2, "new_document"),
    ],
)
def test_revision_no_and_upload_intent_mismatch_is_rejected(
    revision_no: int,
    upload_intent: str,
) -> None:
    with pytest.raises(ValidationError):
        DocumentRevisionCreate(
            document_id=uuid.uuid4(),
            revision_no=revision_no,
            upload_intent=upload_intent,
            supersedes_revision_id=uuid.uuid4() if revision_no > 1 else None,
            original_filename="meeting-notes.pdf",
            storage_key="documents/03/meeting-notes.bin",
            mime_type="application/pdf",
            file_size_bytes=1024,
            sha256="a" * 64,
        )


def test_original_filename_rejects_nul_character() -> None:
    with pytest.raises(ValidationError):
        DocumentRevisionCreate(
            document_id=uuid.uuid4(),
            revision_no=1,
            upload_intent="new_document",
            original_filename="bad\x00name.pdf",
            storage_key="documents/01/meeting-notes.bin",
            mime_type="application/pdf",
            file_size_bytes=1024,
            sha256="a" * 64,
        )


def test_logical_order_index_uses_decimal_numeric_mapping() -> None:
    column = Document.__table__.c.logical_order_index
    annotation = str(Document.__annotations__["logical_order_index"])

    assert "Decimal" in annotation
    assert column.type.python_type is Decimal
    assert column.type.scale == 6


def test_current_revision_unique_constraint_exists() -> None:
    documents_table = Base.metadata.tables["documents"]
    assert any(
        isinstance(constraint, UniqueConstraint)
        and tuple(constraint.columns.keys()) == ("current_revision_id",)
        for constraint in documents_table.constraints
    )


def test_document_and_revision_time_columns_use_timezone() -> None:
    document_columns = (
        Document.__table__.c.created_at,
        Document.__table__.c.updated_at,
        Document.__table__.c.effective_from,
        Document.__table__.c.effective_to,
    )
    revision_columns = (
        DocumentRevision.__table__.c.uploaded_at,
        DocumentRevision.__table__.c.updated_at,
    )

    assert all(column.type.timezone is True for column in (*document_columns, *revision_columns))


def test_document_revision_read_hides_storage_key() -> None:
    assert "storage_key" not in DocumentRevisionRead.model_fields


def test_document_revision_internal_read_includes_storage_key() -> None:
    assert "storage_key" in DocumentRevisionInternalRead.model_fields


def test_document_migration_handles_circular_foreign_key_in_order() -> None:
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "202607300430_document_models.py"
    )
    content = migration_path.read_text(encoding="utf-8")

    assert content.index('op.create_table(\n        "documents"') < content.index(
        'op.create_table(\n        "document_revisions"'
    )
    assert content.index('op.create_table(\n        "document_revisions"') < content.index(
        'op.create_foreign_key(\n        op.f("fk_documents_current_revision_id_document_revisions")'
    )
