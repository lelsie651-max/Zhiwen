import asyncio
import hashlib
import io
import warnings
import zipfile
from pathlib import Path

import pytest
from pypdf import PdfWriter

from app.schemas.file_ingestion import DetectedFileFormat, TechnicalPreflightOutcome
from app.services.file_ingestion import (
    EmptyUploadError,
    FileTooLargeError,
    receive_upload_stream,
    run_technical_preflight,
)
from app.storage import InvalidStorageKeyError, LocalFileStorage, StorageObjectNotFoundError


class ByteChunkStream:
    def __init__(self, data: bytes, *, chunk_size: int = 7) -> None:
        self._data = data
        self._chunk_size = chunk_size
        self._offset = 0

    def __aiter__(self) -> "ByteChunkStream":
        self._offset = 0
        return self

    async def __anext__(self) -> bytes:
        if self._offset >= len(self._data):
            raise StopAsyncIteration
        chunk = self._data[self._offset : self._offset + self._chunk_size]
        self._offset += self._chunk_size
        return chunk


def run_async(awaitable):
    return asyncio.run(awaitable)


async def collect_chunks(async_iterable) -> bytes:
    chunks: list[bytes] = []
    async for chunk in async_iterable:
        chunks.append(chunk)
    return b"".join(chunks)


def build_pdf_bytes(*, encrypt: bool = False) -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    if encrypt:
        writer.encrypt("secret")

    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def build_docx_bytes(*, suspicious: bool = False) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types></Types>")
        archive.writestr("_rels/.rels", "<Relationships></Relationships>")
        archive.writestr("word/document.xml", "<w:document></w:document>")
        if suspicious:
            archive.writestr("word/media/payload.bin", b"A" * (2 * 1024 * 1024))
    return buffer.getvalue()


def build_invalid_docx_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("not-docx.txt", "hello")
    return buffer.getvalue()


def build_docx_with_duplicate_entries() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types></Types>")
        archive.writestr("word/document.xml", "<w:document></w:document>")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            archive.writestr("word/styles.xml", "<w:styles></w:styles>")
            archive.writestr("word/styles.xml", "<w:styles><w:style/></w:styles>")
    return buffer.getvalue()


def make_storage(tmp_path: Path) -> LocalFileStorage:
    return LocalFileStorage(
        temp_root=tmp_path / "tmp",
        storage_root=tmp_path / "files",
    )


@pytest.mark.parametrize(
    ("filename", "data", "declared_mime", "expected_format", "expected_mime"),
    [
        (
            "paper.pdf",
            build_pdf_bytes(),
            "application/pdf",
            DetectedFileFormat.PDF,
            "application/pdf",
        ),
        (
            "chapter.docx",
            build_docx_bytes(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            DetectedFileFormat.DOCX,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
        (
            "notes.txt",
            "plain text content\n".encode("utf-8"),
            "text/plain",
            DetectedFileFormat.TXT,
            "text/plain",
        ),
        (
            "readme.md",
            "# Title\n\ncontent\n".encode("utf-8"),
            "text/markdown",
            DetectedFileFormat.MD,
            "text/markdown",
        ),
    ],
)
def test_supported_samples_pass(
    tmp_path: Path,
    filename: str,
    data: bytes,
    declared_mime: str,
    expected_format: DetectedFileFormat,
    expected_mime: str,
) -> None:
    storage = make_storage(tmp_path)
    received = run_async(
        receive_upload_stream(
            ByteChunkStream(data, chunk_size=9),
            original_filename=filename,
            declared_mime=declared_mime,
            storage=storage,
            max_upload_bytes=5 * 1024 * 1024,
            upload_chunk_bytes=4,
        )
    )
    result = run_async(run_technical_preflight(received, storage=storage))

    assert received.file_size == len(data)
    assert received.sha256 == hashlib.sha256(data).hexdigest()
    assert result.outcome == TechnicalPreflightOutcome.PASS
    assert result.detected_format == expected_format
    assert result.detected_mime == expected_mime
    assert not any(flag.severity == "blocker" for flag in result.risk_flags)


def test_empty_file_is_rejected_and_temp_file_is_deleted(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)

    with pytest.raises(EmptyUploadError):
        run_async(
            receive_upload_stream(
                ByteChunkStream(b"", chunk_size=3),
                original_filename="empty.txt",
                storage=storage,
                max_upload_bytes=1024,
                upload_chunk_bytes=8,
            )
        )

    assert list((tmp_path / "tmp").iterdir()) == []


def test_oversized_file_is_rejected_and_temp_file_is_deleted(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)

    with pytest.raises(FileTooLargeError):
        run_async(
            receive_upload_stream(
                ByteChunkStream(b"0123456789abcdef", chunk_size=16),
                original_filename="too-large.txt",
                storage=storage,
                max_upload_bytes=10,
                upload_chunk_bytes=4,
            )
        )

    assert list((tmp_path / "tmp").iterdir()) == []


@pytest.mark.parametrize(
    ("max_upload_bytes", "upload_chunk_bytes"),
    [
        (0, None),
        (-1, None),
        (1024, 0),
        (1024, -1),
    ],
)
def test_non_positive_explicit_limits_are_rejected(
    tmp_path: Path,
    max_upload_bytes: int | None,
    upload_chunk_bytes: int | None,
) -> None:
    storage = make_storage(tmp_path)

    with pytest.raises(ValueError):
        run_async(
            receive_upload_stream(
                ByteChunkStream(b"hello", chunk_size=5),
                original_filename="notes.txt",
                storage=storage,
                max_upload_bytes=max_upload_bytes,
                upload_chunk_bytes=upload_chunk_bytes,
            )
        )


def test_extension_and_real_format_mismatch_is_rejected(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)
    received = run_async(
        receive_upload_stream(
            ByteChunkStream(build_pdf_bytes()),
            original_filename="fake.txt",
            declared_mime="text/plain",
            storage=storage,
        )
    )

    result = run_async(run_technical_preflight(received, storage=storage))

    assert result.outcome == TechnicalPreflightOutcome.REJECT
    assert any(flag.kind == "format_mismatch" for flag in result.risk_flags)


def test_encrypted_pdf_is_rejected(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)
    received = run_async(
        receive_upload_stream(
            ByteChunkStream(build_pdf_bytes(encrypt=True)),
            original_filename="secret.pdf",
            storage=storage,
        )
    )

    result = run_async(run_technical_preflight(received, storage=storage))

    assert result.outcome == TechnicalPreflightOutcome.REJECT
    assert any(flag.kind == "encrypted_pdf" for flag in result.risk_flags)


def test_invalid_docx_is_rejected(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)
    received = run_async(
        receive_upload_stream(
            ByteChunkStream(build_invalid_docx_bytes()),
            original_filename="broken.docx",
            storage=storage,
        )
    )

    result = run_async(run_technical_preflight(received, storage=storage))

    assert result.outcome == TechnicalPreflightOutcome.REJECT
    assert any(flag.kind == "invalid_docx" for flag in result.risk_flags)


def test_suspicious_docx_is_quarantined(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)
    received = run_async(
        receive_upload_stream(
            ByteChunkStream(build_docx_bytes(suspicious=True)),
            original_filename="suspicious.docx",
            storage=storage,
        )
    )

    result = run_async(run_technical_preflight(received, storage=storage))

    assert result.outcome == TechnicalPreflightOutcome.QUARANTINE
    assert any(flag.kind == "suspicious_docx_archive" for flag in result.risk_flags)


def test_docx_duplicate_entries_count_is_not_deduplicated_and_warns(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)
    received = run_async(
        receive_upload_stream(
            ByteChunkStream(build_docx_with_duplicate_entries()),
            original_filename="duplicate.docx",
            storage=storage,
        )
    )

    result = run_async(run_technical_preflight(received, storage=storage))

    assert result.outcome == TechnicalPreflightOutcome.PASS
    assert result.file_checks["entry_count"] == 4
    assert result.file_checks["duplicate_entry_count"] == 1
    assert any(flag.kind == "duplicate_docx_entries" for flag in result.risk_flags)


def test_path_traversal_filename_is_sanitized_without_affecting_storage_path(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)
    pdf_bytes = build_pdf_bytes()
    received = run_async(
        receive_upload_stream(
            ByteChunkStream(pdf_bytes),
            original_filename="../../secrets/report.pdf",
            declared_mime="application/octet-stream",
            storage=storage,
        )
    )
    result = run_async(run_technical_preflight(received, storage=storage))
    storage_key = run_async(storage.promote_temp_file(received.temp_file, suffix=received.extension))
    stored_bytes = run_async(collect_chunks(storage.read_file_chunks(storage_key, chunk_size=8)))

    assert received.safe_original_filename == "report.pdf"
    assert received.filename_sanitized is True
    assert result.outcome == TechnicalPreflightOutcome.PASS
    assert any(flag.kind == "unsafe_filename_sanitized" for flag in result.risk_flags)
    assert any(flag.kind == "declared_mime_mismatch" for flag in result.risk_flags)
    assert "report" not in storage_key
    assert "secrets" not in storage_key
    assert stored_bytes == pdf_bytes


def test_text_encoding_metrics_are_numeric_and_semantically_consistent(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)
    text_bytes = "你好，世界\n".encode("utf-8")
    received = run_async(
        receive_upload_stream(
            ByteChunkStream(text_bytes),
            original_filename="notes.txt",
            storage=storage,
        )
    )

    result = run_async(run_technical_preflight(received, storage=storage))

    assert result.outcome == TechnicalPreflightOutcome.PASS
    assert result.file_checks["encoding"]
    assert isinstance(result.file_checks["encoding_chaos"], float)
    assert isinstance(result.file_checks["encoding_confidence"], float)
    assert result.file_checks["encoding_confidence"] == pytest.approx(
        max(0.0, min(1.0, 1.0 - result.file_checks["encoding_chaos"]))
    )
    assert 0.0 <= result.file_checks["encoding_confidence"] <= 1.0


def test_declared_mime_with_charset_parameter_does_not_warn_when_type_matches(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)
    received = run_async(
        receive_upload_stream(
            ByteChunkStream("hello".encode("utf-8")),
            original_filename="notes.txt",
            declared_mime=" Text/Plain ; charset=UTF-8 ",
            storage=storage,
        )
    )

    result = run_async(run_technical_preflight(received, storage=storage))

    assert result.outcome == TechnicalPreflightOutcome.PASS
    assert not any(flag.kind == "declared_mime_mismatch" for flag in result.risk_flags)


def test_true_declared_mime_mismatch_still_warns(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)
    received = run_async(
        receive_upload_stream(
            ByteChunkStream("hello".encode("utf-8")),
            original_filename="notes.txt",
            declared_mime="application/json; charset=utf-8",
            storage=storage,
        )
    )

    result = run_async(run_technical_preflight(received, storage=storage))

    assert result.outcome == TechnicalPreflightOutcome.PASS
    assert any(flag.kind == "declared_mime_mismatch" for flag in result.risk_flags)


def test_storage_key_cannot_escape_storage_root(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)

    for unsafe_key in (
        "../evil.bin",
        "files/../evil.bin",
        "/absolute/evil.bin",
        r"files\aa\evil.bin",
        "C:/evil.bin",
    ):
        with pytest.raises(InvalidStorageKeyError):
            storage.parse_storage_key(unsafe_key)


def test_delete_formal_file_removes_object(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)
    pdf_bytes = build_pdf_bytes()
    received = run_async(
        receive_upload_stream(
            ByteChunkStream(pdf_bytes),
            original_filename="paper.pdf",
            storage=storage,
        )
    )
    storage_key = run_async(storage.promote_temp_file(received.temp_file, suffix=received.extension))

    run_async(storage.delete_file(storage_key))

    with pytest.raises(StorageObjectNotFoundError):
        run_async(collect_chunks(storage.read_file_chunks(storage_key, chunk_size=16)))
