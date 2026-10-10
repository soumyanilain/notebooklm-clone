"""Storage abstraction.

Everything above this layer talks in *keys* (posix-style relative paths such as
``notebooks/ab12/chats/history.jsonl``) and never builds filesystem paths itself.
That keeps the notebook layout in one place (NotebookStore) and lets the backend
be swapped: the app ships with LocalStorage; a cloud bucket backend only needs to
implement this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class StorageError(Exception):
    """Raised for unsafe keys or storage failures, with a user-readable message."""


class StorageBackend(ABC):
    @abstractmethod
    def read_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def write_bytes(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def read_text(self, key: str) -> str: ...

    @abstractmethod
    def write_text(self, key: str, text: str) -> None:
        """Atomic write: readers never see a half-written file."""

    @abstractmethod
    def append_text(self, key: str, text: str) -> None: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def list_dir(self, key: str) -> list[str]:
        """Names directly under ``key`` (files and folders). Empty if missing."""

    @abstractmethod
    def is_dir(self, key: str) -> bool: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    @abstractmethod
    def delete_tree(self, key: str) -> None: ...

    @abstractmethod
    def copy_in(self, src: Path, key: str) -> None:
        """Copy a local file (e.g. a Gradio upload) into storage."""

    @abstractmethod
    def local_path(self, key: str) -> Path:
        """A real filesystem path for the key, for libraries that need one
        (Chroma's persistent directory, Gradio file downloads)."""

    @abstractmethod
    def size(self, key: str) -> int: ...
