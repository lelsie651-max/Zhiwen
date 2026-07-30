from __future__ import annotations

import hashlib
import io
import re
import zipfile
from collections.abc import AsyncIterable

from charset_normalizer import from_bytes
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.config import Settings, get_settings
from app.models.ingestion_validation import (
    IngestionRiskCategory,
    IngestionRiskDetector,
    IngestionRiskSeverity,
)
from app.schemas.file_ingestion import (
    DetectedFileFormat,
    ReceivedUpload,
    TechnicalPreflightOutcome,
    TechnicalPreflightResult,
    TechnicalRiskFlag,
    TempFileReference,
)
from app.storage import FileStorage, LocalFileStorage, StorageError
from app.utils.validation import normalize_text


PDF_HEADER = b"%PDF-"
SUPPORTED_EXTENSIONS = {
    ".pdf": DetectedFileFormat.PDF,
    ".docx": DetectedFileFormat.DOCX,
    ".txt": DetectedFileFormat.TXT,
    ".md": DetectedFileFormat.MD,
}
DETECTED_MIME_BY_FORMAT = {
    DetectedFileFormat.PDF: "application/pdf",
    DetectedFileFormat.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    DetectedFileFormat.TXT: "text/plain",
    DetectedFileFormat.MD: "text/markdown",
}
DOCX_REQUIRED_ENTRIES = {"[Content_Types].xml", "word/document.xml"}
DOCX_MAX_ENTRY_COUNT = 2048
DOCX_MAX_UNCOMPRESSED_BYTES = 100 * 1024 * 1024
DOCX_MAX_COMPRESSION_RATIO = 100.0
TEXT_CONTROL_BYTE_WHITELIST = {0x09, 0x0A, 0x0C, 0x0D}


class FileIngestionError(Exception):
    """Base class for upload stream and preflight failures."""


class InvalidOriginalFilenameError(FileIngestionError):
    """Raised when the provided original filename is unsafe."""


class EmptyUploadError(FileIngestionError):
    """Raised when the received upload contains no bytes."""


class FileTooLargeError(FileIngestionError):
    """Raised when the upload exceeds the configured size limit."""


class TechnicalPreflightError(FileIngestionError):
    """Raised when technical preflight cannot complete due to an internal failure."""


def get_local_file_storage(settings: Settings | None = None) -> LocalFileStorage:
    config = settings or get_settings()
    return LocalFileStorage(
        temp_root=config.upload_temp_dir_path,
        storage_root=config.file_storage_root_path,
    )


def sanitize_original_filename(original_filename: str) -> tuple[str, bool]:
    if "\x00" in original_filename:
        raise InvalidOriginalFilenameError("original_filename must not contain NUL characters")

    normalized = normalize_text(original_filename)
    if not normalized:
        raise InvalidOriginalFilenameError("original_filename must not be empty")

    parts = [part for part in re.split(r"[\\/]+", normalized) if part not in {"", ".", ".."}]
    if not parts:
        raise InvalidOriginalFilenameError("original_filename must contain a safe basename")

    basename = parts[-1]
    if len(basename) > 255:
        raise InvalidOriginalFilenameError("original_filename basename must be at most 255 characters")
    return basename, basename != normalized


async def receive_upload_stream(
    chunks: AsyncIterable[bytes],
    *,
    original_filename: str,
    declared_mime: str | None = None,
    storage: FileStorage | None = None,
    max_upload_bytes: int | None = None,
    upload_chunk_bytes: int | None = None,
) -> ReceivedUpload:
    settings = get_settings()
    storage_backend = storage or get_local_file_storage(settings)
    size_limit = settings.max_upload_bytes if max_upload_bytes is None else max_upload_bytes
    chunk_limit = settings.upload_chunk_bytes if upload_chunk_bytes is None else upload_chunk_bytes
    if size_limit <= 0:
        raise ValueError("max_upload_bytes must be greater than 0")
    if chunk_limit <= 0:
        raise ValueError("upload_chunk_bytes must be greater than 0")

    safe_original_filename, filename_sanitized = sanitize_original_filename(original_filename)
    extension = _extract_extension(safe_original_filename)
    normalized_declared_mime = _normalize_optional_text(declared_mime)
    temp_file = await storage_backend.create_temp_file(suffix=".upload")
    has_bytes = False
    total_size = 0
    sha256 = hashlib.sha256()

    try:
        async for raw_chunk in chunks:
            if not isinstance(raw_chunk, (bytes, bytearray)):
                raise FileIngestionError("Upload chunks must be bytes.")
            if not raw_chunk:
                continue

            for chunk in _split_chunk(bytes(raw_chunk), chunk_limit):
                if not chunk:
                    continue
                has_bytes = True
                total_size += len(chunk)
                if total_size > size_limit:
                    raise FileTooLargeError("Upload exceeds the configured maximum size.")
                sha256.update(chunk)
                await storage_backend.write_temp_chunk(temp_file, chunk)

        if not has_bytes:
            raise EmptyUploadError("Upload must not be empty.")

        return ReceivedUpload(
            temp_file=temp_file,
            safe_original_filename=safe_original_filename,
            filename_sanitized=filename_sanitized,
            file_size=total_size,
            sha256=sha256.hexdigest(),
            declared_mime=normalized_declared_mime,
            extension=extension,
        )
    except Exception:
        await _safe_delete_temp_file(storage_backend, temp_file)
        raise


async def run_technical_preflight(
    received_upload: ReceivedUpload,
    *,
    storage: FileStorage | None = None,
) -> TechnicalPreflightResult:
    settings = get_settings()
    storage_backend = storage or get_local_file_storage(settings)

    try:
        file_bytes = await _read_temp_bytes(
            storage_backend,
            received_upload.temp_file,
            chunk_size=settings.upload_chunk_bytes,
        )
        return _inspect_bytes(received_upload, file_bytes)
    except FileIngestionError:
        await _safe_delete_temp_file(storage_backend, received_upload.temp_file)
        raise
    except StorageError:
        await _safe_delete_temp_file(storage_backend, received_upload.temp_file)
        raise
    except Exception as exc:
        await _safe_delete_temp_file(storage_backend, received_upload.temp_file)
        raise TechnicalPreflightError("Technical preflight failed unexpectedly.") from exc


def _inspect_bytes(received_upload: ReceivedUpload, file_bytes: bytes) -> TechnicalPreflightResult:
    warnings = _build_common_warning_flags(received_upload)
    extension = received_upload.extension.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        return TechnicalPreflightResult(
            outcome=TechnicalPreflightOutcome.REJECT,
            detected_format=None,
            detected_mime=None,
            detected_encoding=None,
            file_checks={
                "extension": extension,
                "supported_extension": False,
                "reason": "unsupported_extension",
            },
            risk_flags=warnings
            + [
                _make_risk_flag(
                    category=IngestionRiskCategory.FILE,
                    kind="unsupported_extension",
                    severity=IngestionRiskSeverity.BLOCKER,
                    message="当前上传仅支持 PDF、DOCX、TXT 和 MD 文件。",
                    details={"extension": extension},
                )
            ],
        )

    if extension == ".pdf":
        return _inspect_pdf(received_upload, file_bytes, warnings)
    if extension == ".docx":
        return _inspect_docx(received_upload, file_bytes, warnings)
    if extension in {".txt", ".md"}:
        return _inspect_text_like(received_upload, file_bytes, warnings)

    raise TechnicalPreflightError("Unsupported extension dispatch.")


def _inspect_pdf(
    received_upload: ReceivedUpload,
    file_bytes: bytes,
    warnings: list[TechnicalRiskFlag],
) -> TechnicalPreflightResult:
    file_checks = {
        "extension": received_upload.extension,
        "header_valid": file_bytes.startswith(PDF_HEADER),
    }
    if not file_checks["header_valid"]:
        return TechnicalPreflightResult(
            outcome=TechnicalPreflightOutcome.REJECT,
            detected_format=None,
            detected_mime=None,
            detected_encoding=None,
            file_checks={**file_checks, "reason": "format_mismatch"},
            risk_flags=warnings
            + [
                _make_risk_flag(
                    category=IngestionRiskCategory.FILE,
                    kind="format_mismatch",
                    severity=IngestionRiskSeverity.BLOCKER,
                    message="文件扩展名为 PDF，但文件头与真实格式不匹配。",
                    details={"expected_extension": ".pdf"},
                )
            ],
        )

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        encrypted = bool(reader.is_encrypted)
        readable = False
        if not encrypted:
            _ = len(reader.pages)
            readable = True
    except PdfReadError:
        return TechnicalPreflightResult(
            outcome=TechnicalPreflightOutcome.REJECT,
            detected_format=DetectedFileFormat.PDF,
            detected_mime=DETECTED_MIME_BY_FORMAT[DetectedFileFormat.PDF],
            detected_encoding=None,
            file_checks={**file_checks, "reason": "corrupted_pdf", "encrypted": False, "readable": False},
            risk_flags=warnings
            + [
                _make_risk_flag(
                    category=IngestionRiskCategory.FILE,
                    kind="corrupted_pdf",
                    severity=IngestionRiskSeverity.BLOCKER,
                    message="PDF 文件结构损坏，无法读取。",
                )
            ],
        )

    file_checks["encrypted"] = encrypted
    file_checks["readable"] = readable
    detected_mime = DETECTED_MIME_BY_FORMAT[DetectedFileFormat.PDF]
    if encrypted:
        return TechnicalPreflightResult(
            outcome=TechnicalPreflightOutcome.REJECT,
            detected_format=DetectedFileFormat.PDF,
            detected_mime=detected_mime,
            detected_encoding=None,
            file_checks={**file_checks, "reason": "encrypted_pdf"},
            risk_flags=warnings
            + [
                _make_risk_flag(
                    category=IngestionRiskCategory.FILE,
                    kind="encrypted_pdf",
                    severity=IngestionRiskSeverity.BLOCKER,
                    message="加密 PDF 暂不支持导入。",
                )
            ],
        )

    return TechnicalPreflightResult(
        outcome=TechnicalPreflightOutcome.PASS,
        detected_format=DetectedFileFormat.PDF,
        detected_mime=detected_mime,
        detected_encoding=None,
        file_checks=file_checks,
        risk_flags=_append_declared_mime_warning(warnings, received_upload.declared_mime, detected_mime),
    )


def _inspect_docx(
    received_upload: ReceivedUpload,
    file_bytes: bytes,
    warnings: list[TechnicalRiskFlag],
) -> TechnicalPreflightResult:
    file_checks = {
        "extension": received_upload.extension,
        "zip_valid": zipfile.is_zipfile(io.BytesIO(file_bytes)),
    }
    if not file_checks["zip_valid"]:
        return TechnicalPreflightResult(
            outcome=TechnicalPreflightOutcome.REJECT,
            detected_format=None,
            detected_mime=None,
            detected_encoding=None,
            file_checks={**file_checks, "reason": "format_mismatch"},
            risk_flags=warnings
            + [
                _make_risk_flag(
                    category=IngestionRiskCategory.FILE,
                    kind="format_mismatch",
                    severity=IngestionRiskSeverity.BLOCKER,
                    message="文件扩展名为 DOCX，但文件不是合法 ZIP 容器。",
                    details={"expected_extension": ".docx"},
                )
            ],
        )

    with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
        entries = archive.infolist()
        entry_names = [entry.filename for entry in entries]
        unique_entry_names = set(entry_names)
        duplicate_entry_count = len(entries) - len(unique_entry_names)
        file_checks["required_entries_present"] = DOCX_REQUIRED_ENTRIES <= unique_entry_names
        file_checks["entry_count"] = len(entries)
        file_checks["duplicate_entry_count"] = duplicate_entry_count
        uncompressed_bytes = sum(entry.file_size for entry in entries)
        compressed_bytes = sum(max(entry.compress_size, 1) for entry in entries)
        compression_ratio = uncompressed_bytes / compressed_bytes if compressed_bytes else float("inf")
        file_checks["uncompressed_bytes"] = uncompressed_bytes
        file_checks["compression_ratio"] = round(compression_ratio, 2)

        if not file_checks["required_entries_present"]:
            return TechnicalPreflightResult(
                outcome=TechnicalPreflightOutcome.REJECT,
                detected_format=None,
                detected_mime=None,
                detected_encoding=None,
                file_checks={**file_checks, "reason": "invalid_docx"},
                risk_flags=warnings
                + [
                    _make_risk_flag(
                        category=IngestionRiskCategory.FILE,
                        kind="invalid_docx",
                        severity=IngestionRiskSeverity.BLOCKER,
                        message="DOCX 缺少必要的 OpenXML 结构文件。",
                    )
                ],
            )

        if (
            file_checks["entry_count"] > DOCX_MAX_ENTRY_COUNT
            or uncompressed_bytes > DOCX_MAX_UNCOMPRESSED_BYTES
            or compression_ratio > DOCX_MAX_COMPRESSION_RATIO
        ):
            return TechnicalPreflightResult(
                outcome=TechnicalPreflightOutcome.QUARANTINE,
                detected_format=DetectedFileFormat.DOCX,
                detected_mime=DETECTED_MIME_BY_FORMAT[DetectedFileFormat.DOCX],
                detected_encoding=None,
                file_checks={**file_checks, "reason": "suspicious_docx_archive"},
                risk_flags=warnings
                + [
                    _make_risk_flag(
                        category=IngestionRiskCategory.SECURITY,
                        kind="suspicious_docx_archive",
                        severity=IngestionRiskSeverity.BLOCKER,
                        message="DOCX 压缩特征异常，已标记为隔离。",
                        details={
                            "entry_count": file_checks["entry_count"],
                            "duplicate_entry_count": duplicate_entry_count,
                            "uncompressed_bytes": uncompressed_bytes,
                            "compression_ratio": file_checks["compression_ratio"],
                        },
                    )
                ],
            )

        if duplicate_entry_count > 0:
            warnings = warnings + [
                _make_risk_flag(
                    category=IngestionRiskCategory.FILE,
                    kind="duplicate_docx_entries",
                    severity=IngestionRiskSeverity.WARNING,
                    message="DOCX 容器中存在重复条目名，建议人工复核。",
                    details={"duplicate_entry_count": duplicate_entry_count},
                )
            ]

    detected_mime = DETECTED_MIME_BY_FORMAT[DetectedFileFormat.DOCX]
    return TechnicalPreflightResult(
        outcome=TechnicalPreflightOutcome.PASS,
        detected_format=DetectedFileFormat.DOCX,
        detected_mime=detected_mime,
        detected_encoding=None,
        file_checks=file_checks,
        risk_flags=_append_declared_mime_warning(warnings, received_upload.declared_mime, detected_mime),
    )


def _inspect_text_like(
    received_upload: ReceivedUpload,
    file_bytes: bytes,
    warnings: list[TechnicalRiskFlag],
) -> TechnicalPreflightResult:
    expected_format = SUPPORTED_EXTENSIONS[received_upload.extension]
    if file_bytes.startswith(PDF_HEADER) or zipfile.is_zipfile(io.BytesIO(file_bytes)):
        return TechnicalPreflightResult(
            outcome=TechnicalPreflightOutcome.REJECT,
            detected_format=None,
            detected_mime=None,
            detected_encoding=None,
            file_checks={
                "extension": received_upload.extension,
                "binary_like": True,
                "reason": "format_mismatch",
            },
            risk_flags=warnings
            + [
                _make_risk_flag(
                    category=IngestionRiskCategory.FILE,
                    kind="format_mismatch",
                    severity=IngestionRiskSeverity.BLOCKER,
                    message="文本扩展名与真实文件格式不匹配。",
                    details={"expected_extension": received_upload.extension},
                )
            ],
        )

    binary_like = _looks_binary(file_bytes)
    if binary_like:
        return TechnicalPreflightResult(
            outcome=TechnicalPreflightOutcome.REJECT,
            detected_format=None,
            detected_mime=None,
            detected_encoding=None,
            file_checks={
                "extension": received_upload.extension,
                "binary_like": True,
                "reason": "binary_text_content",
            },
            risk_flags=warnings
            + [
                _make_risk_flag(
                    category=IngestionRiskCategory.FILE,
                    kind="binary_text_content",
                    severity=IngestionRiskSeverity.BLOCKER,
                    message="TXT/MD 文件包含明显二进制内容。",
                )
            ],
        )

    match = from_bytes(file_bytes).best()
    if match is None or not match.encoding:
        return TechnicalPreflightResult(
            outcome=TechnicalPreflightOutcome.REJECT,
            detected_format=None,
            detected_mime=None,
            detected_encoding=None,
            file_checks={
                "extension": received_upload.extension,
                "binary_like": False,
                "reason": "encoding_detection_failed",
            },
            risk_flags=warnings
            + [
                _make_risk_flag(
                    category=IngestionRiskCategory.FILE,
                    kind="encoding_detection_failed",
                    severity=IngestionRiskSeverity.BLOCKER,
                    message="无法可靠识别文本编码。",
                )
            ],
        )

    detected_mime = DETECTED_MIME_BY_FORMAT[expected_format]
    return TechnicalPreflightResult(
        outcome=TechnicalPreflightOutcome.PASS,
        detected_format=expected_format,
        detected_mime=detected_mime,
        detected_encoding=match.encoding,
        file_checks={
            "extension": received_upload.extension,
            "binary_like": False,
            "encoding": match.encoding,
            "encoding_chaos": float(match.chaos),
            "encoding_confidence": max(0.0, min(1.0, 1.0 - float(match.chaos))),
        },
        risk_flags=_append_declared_mime_warning(warnings, received_upload.declared_mime, detected_mime),
    )


def _build_common_warning_flags(received_upload: ReceivedUpload) -> list[TechnicalRiskFlag]:
    warnings: list[TechnicalRiskFlag] = []
    if received_upload.filename_sanitized:
        warnings.append(
            _make_risk_flag(
                category=IngestionRiskCategory.FILE,
                kind="unsafe_filename_sanitized",
                severity=IngestionRiskSeverity.WARNING,
                message="原始文件名包含路径信息，已降级为安全 basename。",
                details={"safe_original_filename": received_upload.safe_original_filename},
            )
        )
    return warnings


def _append_declared_mime_warning(
    warnings: list[TechnicalRiskFlag],
    declared_mime: str | None,
    detected_mime: str,
) -> list[TechnicalRiskFlag]:
    normalized_declared_mime = _normalize_declared_mime(declared_mime)
    if normalized_declared_mime is None or normalized_declared_mime == detected_mime:
        return warnings
    return warnings + [
        _make_risk_flag(
            category=IngestionRiskCategory.FILE,
            kind="declared_mime_mismatch",
            severity=IngestionRiskSeverity.WARNING,
            message="浏览器声明的 MIME 与文件真实格式不一致，已按真实格式处理。",
            details={"declared_mime": normalized_declared_mime, "detected_mime": detected_mime},
        )
    ]


def _make_risk_flag(
    *,
    category: IngestionRiskCategory,
    kind: str,
    severity: IngestionRiskSeverity,
    message: str,
    details: dict[str, object] | None = None,
) -> TechnicalRiskFlag:
    return TechnicalRiskFlag(
        category=category,
        kind=kind,
        severity=severity,
        detector=IngestionRiskDetector.DETERMINISTIC,
        message=message,
        details=details or {},
    )


def _extract_extension(filename: str) -> str:
    dot_index = filename.rfind(".")
    if dot_index <= 0:
        return ""
    return filename[dot_index:].lower()


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = normalize_text(value)
    return normalized or None


def _normalize_declared_mime(value: str | None) -> str | None:
    normalized = _normalize_optional_text(value)
    if normalized is None:
        return None
    mime_type = normalized.split(";", 1)[0].strip().lower()
    return mime_type or None


def _split_chunk(chunk: bytes, chunk_limit: int) -> list[bytes]:
    return [chunk[index : index + chunk_limit] for index in range(0, len(chunk), chunk_limit)]


def _looks_binary(file_bytes: bytes) -> bool:
    if not file_bytes:
        return True
    if b"\x00" in file_bytes:
        return True

    control_bytes = sum(
        1
        for value in file_bytes
        if value < 32 and value not in TEXT_CONTROL_BYTE_WHITELIST
    )
    return (control_bytes / len(file_bytes)) > 0.3


async def _read_temp_bytes(
    storage: FileStorage,
    temp_file: TempFileReference,
    *,
    chunk_size: int,
) -> bytes:
    chunks: list[bytes] = []
    async for chunk in storage.read_temp_chunks(temp_file, chunk_size=chunk_size):
        chunks.append(chunk)
    return b"".join(chunks)


async def _safe_delete_temp_file(storage: FileStorage, temp_file: TempFileReference) -> None:
    try:
        await storage.delete_temp_file(temp_file)
    except StorageError:
        return
