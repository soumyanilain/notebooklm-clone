"""Local-disk implementation of the storage interface."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path, PurePosixPath

from .base import StorageBackend, StorageError


class LocalStorage(StorageBackend):
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # -- key handling -------------------------------------------------------
    def _resolve(self, key: str) -> Path:
        """Map a key to a path and refuse anything that escapes the root."""
        if not key or key.strip() == "":
            return self.root
        posix = PurePosixPath(key.replace("\\", "/"))
        if posix.is_absolute() or ".." in posix.parts:
            raise StorageError(f"Unsafe storage key: {key!r}")
        path = (self.root / Path(*posix.parts)).resolve()
        if path != self.root and self.root not in path.parents:
            raise StorageError(f"Storage key escapes the data folder: {key!r}")
        return path

    def local_path(self, key: str) -> Path:
        return self._resolve(key)

    # -- reads ----------------------------------------------------------------
    def read_bytes(self, key: str) -> bytes:
        return self._resolve(key).read_bytes()

    def read_text(self, key: str) -> str:
        return self._resolve(key).read_text(encoding="utf-8")

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()

    def is_dir(self, key: str) -> bool:
        return self._resolve(key).is_dir()

    def list_dir(self, key: str) -> list[str]:
        path = self._resolve(key)
        if not path.is_dir():
            return []
        return sorted(p.name for p in path.iterdir())

    def size(self, key: str) -> int:
        path = self._resolve(key)
        if path.is_file():
            return path.stat().st_size
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())

    # -- writes ---------------------------------------------------------------
    def write_bytes(self, key: str, data: bytes) -> None:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            self._replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def write_text(self, key: str, text: str) -> None:
        self.write_bytes(key, text.encode("utf-8"))

    def append_text(self, key: str, text: str) -> None:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(text)

    def copy_in(self, src: Path, key: str) -> None:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, path)

    def delete(self, key: str) -> None:
        path = self._resolve(key)
        if path.is_file():
            path.unlink()

    def delete_tree(self, key: str) -> None:
        path = self._resolve(key)
        if path == self.root:
            raise StorageError("Refusing to delete the whole data folder")
        if not path.exists():
            return
        # Windows can hold file locks for a moment after a database closes.
        last_error: Exception | None = None
        for _ in range(5):
            try:
                shutil.rmtree(path)
                return
            except OSError as exc:
                last_error = exc
                time.sleep(0.3)
        raise StorageError(f"Could not delete {key}: {last_error}")

    @staticmethod
    def _replace(tmp: str, path: Path) -> None:
        for attempt in range(5):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.1)
