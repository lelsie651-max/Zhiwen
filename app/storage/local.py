from __future__ import annotations

import asyncio
import re
import uuid
from collections.abc import AsyncIterator
from pathlib import Path, PurePosixPath

from app.schemas.file_ingestion import TempFileReference
from app.storage.base import (
    FileStorage,
    InvalidStorageKeyError,
    ParsedStorageKey,
    StorageObjectNotFoundError,
)


SAFE_SUFFIX_PATTERN = re.compile(r"^\.[a-z0-9]{1,16}$")
TEMP_FILENAME_PATTERN = re.compile(r"^[0-9a-f]{32}(?:\.[a-z0-9]{1,16})?$")
STORAGE_FILENAME_PATTERN = re.compile(r"^[0-9a-f]{32}(?:\.[a-z0-9]{1,16})?$")


class LocalFileStorage(FileStorage):
    TEMP_NAMESPACE = "tmp"
    FILE_NAMESPACE = "files"

    def __init__(self, *, temp_root: Path, storage_root: Path) -> None:
        self._temp_root = temp_root.expanduser().resolve()
        self._storage_root = storage_root.expanduser().resolve()
        self._temp_root.mkdir(parents=True, exist_ok=True)
        self._storage_root.mkdir(parents=True, exist_ok=True)

    async def create_temp_file(self, *, suffix: str = ".upload") -> TempFileReference:
        normalized_suffix = self._normalize_suffix(suffix)
        token = uuid.uuid4().hex
        temp_key = f"{self.TEMP_NAMESPACE}/{token}{normalized_suffix}"
        temp_path = self._resolve_temp_path(temp_key)
        await asyncio.to_thread(temp_path.touch, exist_ok=False)
        return TempFileReference(temp_key=temp_key)

    async def write_temp_chunk(self, temp_file: TempFileReference, chunk: bytes) -> None:
        temp_path = self._resolve_temp_path(temp_file.temp_key)
        if not temp_path.exists():
            raise StorageObjectNotFoundError("Temporary file does not exist.")
        await asyncio.to_thread(self._append_bytes, temp_path, bytes(chunk))

    async def read_temp_chunks(
        self,
        temp_file: TempFileReference,
        *,
        chunk_size: int,
    ) -> AsyncIterator[bytes]:
        temp_path = self._resolve_temp_path(temp_file.temp_key)
        async for chunk in self._iter_file_chunks(temp_path, chunk_size=chunk_size):
            yield chunk

    async def read_file_chunks(
        self,
        storage_key: str,
        *,
        chunk_size: int,
    ) -> AsyncIterator[bytes]:
        file_path = self._resolve_storage_path(storage_key)
        async for chunk in self._iter_file_chunks(file_path, chunk_size=chunk_size):
            yield chunk

    async def promote_temp_file(self, temp_file: TempFileReference, *, suffix: str = "") -> str:
        temp_path = self._resolve_temp_path(temp_file.temp_key)
        if not temp_path.exists():
            raise StorageObjectNotFoundError("Temporary file does not exist.")

        normalized_suffix = self._normalize_suffix(suffix)
        token = uuid.uuid4().hex
        storage_key = f"{self.FILE_NAMESPACE}/{token[:2]}/{token}{normalized_suffix}"
        destination = self._resolve_storage_path(storage_key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(temp_path.replace, destination)
        return storage_key

    async def delete_temp_file(self, temp_file: TempFileReference) -> None:
        temp_path = self._resolve_temp_path(temp_file.temp_key)
        if temp_path.exists():
            await asyncio.to_thread(temp_path.unlink)

    async def delete_file(self, storage_key: str) -> None:
        file_path = self._resolve_storage_path(storage_key)
        if file_path.exists():
            await asyncio.to_thread(file_path.unlink)

    def parse_storage_key(self, storage_key: str) -> ParsedStorageKey:
        raw_key = storage_key.strip()
        if not raw_key:
            raise InvalidStorageKeyError("Storage key must not be empty.")
        if "\\" in raw_key or ":" in raw_key:
            raise InvalidStorageKeyError("Storage key must use normalized relative segments.")

        path = PurePosixPath(raw_key)
        if path.is_absolute():
            raise InvalidStorageKeyError("Absolute storage keys are not allowed.")

        parts = path.parts
        if len(parts) != 3 or parts[0] != self.FILE_NAMESPACE:
            raise InvalidStorageKeyError("Storage key namespace is invalid.")
        if any(part in {"", ".", ".."} for part in parts):
            raise InvalidStorageKeyError("Storage key contains unsafe path segments.")
        if not re.fullmatch(r"[0-9a-f]{2}", parts[1]):
            raise InvalidStorageKeyError("Storage key shard is invalid.")
        if not STORAGE_FILENAME_PATTERN.fullmatch(parts[2]):
            raise InvalidStorageKeyError("Storage key filename is invalid.")

        filename = parts[2]
        return ParsedStorageKey(
            namespace=parts[0],
            segments=parts,
            filename=filename,
            suffix=PurePosixPath(filename).suffix.lower(),
        )

    def _resolve_temp_path(self, temp_key: str) -> Path:
        raw_key = temp_key.strip()
        if not raw_key:
            raise InvalidStorageKeyError("Temp key must not be empty.")
        if "\\" in raw_key or ":" in raw_key:
            raise InvalidStorageKeyError("Temp key must use normalized relative segments.")

        path = PurePosixPath(raw_key)
        if path.is_absolute():
            raise InvalidStorageKeyError("Absolute temp keys are not allowed.")

        parts = path.parts
        if len(parts) != 2 or parts[0] != self.TEMP_NAMESPACE:
            raise InvalidStorageKeyError("Temp key namespace is invalid.")
        if any(part in {"", ".", ".."} for part in parts):
            raise InvalidStorageKeyError("Temp key contains unsafe path segments.")
        if not TEMP_FILENAME_PATTERN.fullmatch(parts[1]):
            raise InvalidStorageKeyError("Temp key filename is invalid.")

        candidate = (self._temp_root / parts[1]).resolve()
        if not candidate.is_relative_to(self._temp_root):
            raise InvalidStorageKeyError("Temp key resolves outside the temp root.")
        return candidate

    def _resolve_storage_path(self, storage_key: str) -> Path:
        parsed = self.parse_storage_key(storage_key)
        candidate = (self._storage_root / parsed.segments[1] / parsed.filename).resolve()
        if not candidate.is_relative_to(self._storage_root):
            raise InvalidStorageKeyError("Storage key resolves outside the storage root.")
        return candidate

    @staticmethod
    def _normalize_suffix(suffix: str) -> str:
        normalized = suffix.strip().lower()
        if not normalized:
            return ""
        if not SAFE_SUFFIX_PATTERN.fullmatch(normalized):
            raise InvalidStorageKeyError("Unsupported suffix.")
        return normalized

    @staticmethod
    def _append_bytes(path: Path, chunk: bytes) -> None:
        with path.open("ab") as file_obj:
            file_obj.write(chunk)

    async def _iter_file_chunks(self, path: Path, *, chunk_size: int) -> AsyncIterator[bytes]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if not path.exists():
            raise StorageObjectNotFoundError("Stored object does not exist.")

        file_obj = await asyncio.to_thread(path.open, "rb")
        try:
            while True:
                chunk = await asyncio.to_thread(file_obj.read, chunk_size)
                if not chunk:
                    break
                yield chunk
        finally:
            await asyncio.to_thread(file_obj.close)
