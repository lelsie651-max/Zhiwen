from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from app.schemas.file_ingestion import TempFileReference


class StorageError(Exception):
    """Base class for storage-related failures."""


class InvalidStorageKeyError(StorageError):
    """Raised when a storage key or temp key is malformed or unsafe."""


class StorageObjectNotFoundError(StorageError):
    """Raised when the requested temp object or stored object does not exist."""


@dataclass(frozen=True)
class ParsedStorageKey:
    namespace: str
    segments: tuple[str, ...]
    filename: str
    suffix: str


class FileStorage(ABC):
    @abstractmethod
    async def create_temp_file(self, *, suffix: str = ".upload") -> TempFileReference:
        """Create an empty temporary file and return its opaque reference."""

    @abstractmethod
    async def write_temp_chunk(self, temp_file: TempFileReference, chunk: bytes) -> None:
        """Append a chunk to a temporary file."""

    @abstractmethod
    async def read_temp_chunks(
        self,
        temp_file: TempFileReference,
        *,
        chunk_size: int,
    ) -> AsyncIterator[bytes]:
        """Read a temporary file in chunks."""

    @abstractmethod
    async def read_file_chunks(
        self,
        storage_key: str,
        *,
        chunk_size: int,
    ) -> AsyncIterator[bytes]:
        """Read a stored file in chunks."""

    @abstractmethod
    async def promote_temp_file(self, temp_file: TempFileReference, *, suffix: str = "") -> str:
        """Promote a temporary file to a permanent storage key."""

    @abstractmethod
    async def delete_temp_file(self, temp_file: TempFileReference) -> None:
        """Delete a temporary file if it exists."""

    @abstractmethod
    async def delete_file(self, storage_key: str) -> None:
        """Delete a permanent stored file if it exists."""

    @abstractmethod
    def parse_storage_key(self, storage_key: str) -> ParsedStorageKey:
        """Validate and parse a permanent storage key."""
