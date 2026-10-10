"""Per-notebook data layout, built on top of a StorageBackend.

Layout (every key is relative to DATA_DIR)::

    notebooks/<notebook_id>/
        notebook.json                 id, name, timestamps
        sources/sources.json          source records (type, origin, counts, enabled)
        sources/raw/<sid>__<file>     original upload or fetched page
        sources/text/<sid>.json       extracted text as citable segments
        sources/text/<sid>.txt        same text, human readable
        chunks/<sid>.jsonl            chunks + citation metadata (source of truth)
        vectordb/                     Chroma index (derived, rebuildable from chunks)
        chats/history.jsonl           one chat message per line
        artifacts/index.json          artifact records
        artifacts/<file>.md           generated reports / quizzes / flashcards

This class is the only place that knows the layout.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Iterable
from pathlib import Path

from ..models import Artifact, ChatMessage, Chunk, Notebook, Segment, Source, new_id, now_iso
from .base import StorageBackend, StorageError

NOTEBOOKS = "notebooks"
DELETED_MARKER = ".deleted"
MAX_NAME_LEN = 80


class NotebookNotFound(StorageError):
    pass


def safe_filename(name: str, max_len: int = 80) -> str:
    """Strip directories and odd characters from a user-supplied file name."""
    name = name.replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .")
    if not name:
        name = "file"
    if len(name) > max_len:
        stem, dot, ext = name.rpartition(".")
        if dot and len(ext) <= 8:
            name = stem[: max_len - len(ext) - 1] + "." + ext
        else:
            name = name[:max_len]
    return name


class NotebookStore:
    def __init__(self, backend: StorageBackend, vector_root: Path | None = None):
        self.backend = backend
        self.vector_root = vector_root
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()
        # Hooks so other components (vector store cache, cloud sync) can react.
        self.on_notebook_deleted: list[Callable[[str], None]] = []
        self.on_before_notebook_delete: list[Callable[[str], None]] = []
        self.purge_deleted()

    # -- keys ---------------------------------------------------------------
    @staticmethod
    def _nb(nb_id: str, *parts: str) -> str:
        if not re.fullmatch(r"[a-z0-9_]{4,40}", nb_id or ""):
            raise StorageError(f"Invalid notebook id: {nb_id!r}")
        return "/".join([NOTEBOOKS, nb_id, *parts])

    def lock(self, nb_id: str) -> threading.RLock:
        with self._locks_guard:
            return self._locks.setdefault(nb_id, threading.RLock())

    def notebook_key(self, nb_id: str) -> str:
        return self._nb(nb_id)

    def vector_dir(self, nb_id: str) -> Path:
        if self.vector_root is not None:
            path = self.vector_root / nb_id
        else:
            path = self.backend.local_path(self._nb(nb_id, "vectordb"))
        path.mkdir(parents=True, exist_ok=True)
        return path

    # -- json helpers ---------------------------------------------------------
    def _read_json(self, key: str, default):
        if not self.backend.exists(key):
            return default
        try:
            return json.loads(self.backend.read_text(key))
        except (json.JSONDecodeError, OSError):
            return default

    def _write_json(self, key: str, data) -> None:
        self.backend.write_text(key, json.dumps(data, indent=2, ensure_ascii=False))

    # -- notebooks --------------------------------------------------------------
    def list_notebooks(self) -> list[Notebook]:
        notebooks: list[Notebook] = []
        for nb_id in self.backend.list_dir(NOTEBOOKS):
            key = f"{NOTEBOOKS}/{nb_id}"
            if not self.backend.is_dir(key) or self.backend.exists(f"{key}/{DELETED_MARKER}"):
                continue
            data = self._read_json(f"{key}/notebook.json", None)
            if data:
                notebooks.append(Notebook.from_dict(data))
        notebooks.sort(key=lambda nb: nb.updated_at, reverse=True)
        return notebooks

    def get_notebook(self, nb_id: str) -> Notebook:
        key = self._nb(nb_id)
        data = self._read_json(f"{key}/notebook.json", None)
        if not data or self.backend.exists(f"{key}/{DELETED_MARKER}"):
            raise NotebookNotFound("That notebook no longer exists. Pick another one from the list.")
        return Notebook.from_dict(data)

    def _validate_name(self, name: str, exclude_id: str | None = None) -> str:
        name = re.sub(r"\s+", " ", (name or "")).strip()
        if not name:
            raise StorageError("Please enter a notebook name.")
        if len(name) > MAX_NAME_LEN:
            raise StorageError(f"Notebook names can be at most {MAX_NAME_LEN} characters.")
        for nb in self.list_notebooks():
            if nb.id != exclude_id and nb.name.lower() == name.lower():
                raise StorageError(f"A notebook called '{nb.name}' already exists.")
        return name

    def create_notebook(self, name: str) -> Notebook:
        name = self._validate_name(name)
        ts = now_iso()
        nb = Notebook(id=new_id("nb_"), name=name, created_at=ts, updated_at=ts)
        self._write_json(self._nb(nb.id, "notebook.json"), nb.to_dict())
        self._write_json(self._nb(nb.id, "sources", "sources.json"), [])
        self._write_json(self._nb(nb.id, "artifacts", "index.json"), [])
        self.backend.write_text(self._nb(nb.id, "chats", "history.jsonl"), "")
        return nb

    def rename_notebook(self, nb_id: str, name: str) -> Notebook:
        with self.lock(nb_id):
            nb = self.get_notebook(nb_id)
            nb.name = self._validate_name(name, exclude_id=nb_id)
            nb.updated_at = now_iso()
            self._write_json(self._nb(nb_id, "notebook.json"), nb.to_dict())
            return nb

    def touch(self, nb_id: str) -> None:
        with self.lock(nb_id):
            nb = self.get_notebook(nb_id)
            nb.updated_at = now_iso()
            self._write_json(self._nb(nb_id, "notebook.json"), nb.to_dict())

    def delete_notebook(self, nb_id: str) -> None:
        with self.lock(nb_id):
            self.get_notebook(nb_id)
            for hook in self.on_before_notebook_delete:
                hook(nb_id)
            # Mark first so the notebook disappears even if a file is still locked.
            self.backend.write_text(self._nb(nb_id, DELETED_MARKER), now_iso())
            try:
                self.backend.delete_tree(self._nb(nb_id))
                if self.vector_root is not None:
                    vdir = self.vector_root / nb_id
                    if vdir.exists():
                        import shutil

                        shutil.rmtree(vdir, ignore_errors=True)
            except StorageError:
                pass  # leftovers are purged on the next start
        for hook in self.on_notebook_deleted:
            hook(nb_id)

    def purge_deleted(self) -> None:
        for nb_id in self.backend.list_dir(NOTEBOOKS):
            key = f"{NOTEBOOKS}/{nb_id}"
            if self.backend.exists(f"{key}/{DELETED_MARKER}"):
                try:
                    self.backend.delete_tree(key)
                except StorageError:
                    pass

    # -- sources ------------------------------------------------------------------
    def list_sources(self, nb_id: str) -> list[Source]:
        return [Source.from_dict(s) for s in self._read_json(self._nb(nb_id, "sources", "sources.json"), [])]

    def get_source(self, nb_id: str, source_id: str) -> Source:
        for src in self.list_sources(nb_id):
            if src.id == source_id:
                return src
        raise StorageError("That source no longer exists.")

    def save_source(self, source: Source) -> None:
        with self.lock(source.notebook_id):
            sources = [s for s in self.list_sources(source.notebook_id) if s.id != source.id]
            sources.append(source)
            sources.sort(key=lambda s: s.added_at)
            self._write_json(self._nb(source.notebook_id, "sources", "sources.json"), [s.to_dict() for s in sources])

    def set_sources_enabled(self, nb_id: str, enabled_ids: Iterable[str]) -> None:
        enabled = set(enabled_ids)
        with self.lock(nb_id):
            sources = self.list_sources(nb_id)
            for s in sources:
                s.enabled = s.id in enabled
            self._write_json(self._nb(nb_id, "sources", "sources.json"), [s.to_dict() for s in sources])

    def delete_source_files(self, nb_id: str, source_id: str) -> None:
        with self.lock(nb_id):
            src = self.get_source(nb_id, source_id)
            for key in (src.raw_key, src.text_key):
                if key:
                    self.backend.delete(key)
            self.backend.delete(self._nb(nb_id, "sources", "text", f"{source_id}.txt"))
            self.backend.delete(self._nb(nb_id, "chunks", f"{source_id}.jsonl"))
            sources = [s for s in self.list_sources(nb_id) if s.id != source_id]
            self._write_json(self._nb(nb_id, "sources", "sources.json"), [s.to_dict() for s in sources])

    def raw_key(self, nb_id: str, source_id: str, filename: str) -> str:
        return self._nb(nb_id, "sources", "raw", f"{source_id}__{safe_filename(filename)}")

    def save_raw_file(self, nb_id: str, source_id: str, filename: str, src: Path) -> str:
        key = self.raw_key(nb_id, source_id, filename)
        self.backend.copy_in(src, key)
        return key

    def save_raw_bytes(self, nb_id: str, source_id: str, filename: str, data: bytes) -> str:
        key = self.raw_key(nb_id, source_id, filename)
        self.backend.write_bytes(key, data)
        return key

    def save_extracted_text(self, nb_id: str, source: Source, segments: list[Segment]) -> str:
        key = self._nb(nb_id, "sources", "text", f"{source.id}.json")
        self._write_json(
            key,
            {"source_id": source.id, "name": source.name, "type": source.type, "origin": source.origin,
             "segments": [s.to_dict() for s in segments]},
        )
        readable = "\n\n".join(f"===== {s.label} =====\n{s.text}" for s in segments)
        self.backend.write_text(self._nb(nb_id, "sources", "text", f"{source.id}.txt"), readable)
        return key

    def load_segments(self, nb_id: str, source_id: str) -> list[Segment]:
        data = self._read_json(self._nb(nb_id, "sources", "text", f"{source_id}.json"), {"segments": []})
        return [Segment.from_dict(s) for s in data.get("segments", [])]

    # -- chunks ---------------------------------------------------------------------
    def save_chunks(self, nb_id: str, source_id: str, chunks: list[Chunk]) -> None:
        lines = "\n".join(json.dumps(c.to_dict(), ensure_ascii=False) for c in chunks)
        self.backend.write_text(self._nb(nb_id, "chunks", f"{source_id}.jsonl"), lines + ("\n" if lines else ""))

    def load_chunks(self, nb_id: str, source_ids: Iterable[str] | None = None) -> list[Chunk]:
        wanted = set(source_ids) if source_ids is not None else None
        chunks: list[Chunk] = []
        for src in self.list_sources(nb_id):
            if wanted is not None and src.id not in wanted:
                continue
            key = self._nb(nb_id, "chunks", f"{src.id}.jsonl")
            if not self.backend.exists(key):
                continue
            for line in self.backend.read_text(key).splitlines():
                if line.strip():
                    chunks.append(Chunk.from_dict(json.loads(line)))
        return chunks

    # -- chat history ---------------------------------------------------------------
    def append_message(self, nb_id: str, message: ChatMessage) -> None:
        with self.lock(nb_id):
            self.backend.append_text(
                self._nb(nb_id, "chats", "history.jsonl"), json.dumps(message.to_dict(), ensure_ascii=False) + "\n"
            )

    def load_history(self, nb_id: str) -> list[ChatMessage]:
        key = self._nb(nb_id, "chats", "history.jsonl")
        if not self.backend.exists(key):
            return []
        messages = []
        for line in self.backend.read_text(key).splitlines():
            if line.strip():
                try:
                    messages.append(ChatMessage.from_dict(json.loads(line)))
                except json.JSONDecodeError:
                    continue  # skip a torn line rather than losing the whole history
        return messages

    def clear_history(self, nb_id: str) -> None:
        with self.lock(nb_id):
            self.backend.write_text(self._nb(nb_id, "chats", "history.jsonl"), "")

    # -- artifacts ------------------------------------------------------------------------
    def list_artifacts(self, nb_id: str) -> list[Artifact]:
        items = [Artifact.from_dict(a) for a in self._read_json(self._nb(nb_id, "artifacts", "index.json"), [])]
        items.sort(key=lambda a: a.created_at, reverse=True)
        return items

    def save_artifact(self, nb_id: str, artifact_type: str, title: str, markdown: str, *,
                      model: str | None, source_ids: list[str], meta: dict | None = None) -> Artifact:
        with self.lock(nb_id):
            art_id = new_id("a_")
            stamp = now_iso().replace(":", "").replace("-", "")[:15]
            slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or artifact_type
            filename = f"{stamp}_{artifact_type}_{slug}.md"
            self.backend.write_text(self._nb(nb_id, "artifacts", filename), markdown)
            artifact = Artifact(id=art_id, notebook_id=nb_id, type=artifact_type, title=title, filename=filename,
                                created_at=now_iso(), model=model, source_ids=source_ids, meta=meta or {})
            items = [a.to_dict() for a in reversed(self.list_artifacts(nb_id))]
            items.append(artifact.to_dict())
            self._write_json(self._nb(nb_id, "artifacts", "index.json"), items)
            return artifact

    def get_artifact(self, nb_id: str, artifact_id: str) -> Artifact:
        for art in self.list_artifacts(nb_id):
            if art.id == artifact_id:
                return art
        raise StorageError("That artifact no longer exists.")

    def read_artifact(self, nb_id: str, artifact_id: str) -> str:
        art = self.get_artifact(nb_id, artifact_id)
        return self.backend.read_text(self._nb(nb_id, "artifacts", art.filename))

    def artifact_path(self, nb_id: str, artifact_id: str) -> Path:
        art = self.get_artifact(nb_id, artifact_id)
        return self.backend.local_path(self._nb(nb_id, "artifacts", art.filename))

    def delete_artifact(self, nb_id: str, artifact_id: str) -> None:
        with self.lock(nb_id):
            art = self.get_artifact(nb_id, artifact_id)
            self.backend.delete(self._nb(nb_id, "artifacts", art.filename))
            items = [a.to_dict() for a in reversed(self.list_artifacts(nb_id)) if a.id != artifact_id]
            self._write_json(self._nb(nb_id, "artifacts", "index.json"), items)

    # -- inspection ---------------------------------------------------------------------
    def storage_tree(self, nb_id: str, max_files: int = 12) -> str:
        """A text tree of the notebook folder, shown in the UI's Storage tab."""
        root = self._nb(nb_id)
        lines = [f"{root}/"]

        def walk(key: str, prefix: str, depth: int) -> None:
            names = [n for n in self.backend.list_dir(key) if not n.startswith(".tmp_")]
            if key.endswith("/vectordb"):
                lines.append(f"{prefix}└── … ({len(names)} index files, {self._fmt(self.backend.size(key))})")
                return
            shown = names[:max_files]
            for i, name in enumerate(shown):
                child = f"{key}/{name}"
                last = i == len(shown) - 1 and len(names) <= max_files
                branch = "└── " if last else "├── "
                if self.backend.is_dir(child):
                    lines.append(f"{prefix}{branch}{name}/")
                    walk(child, prefix + ("    " if last else "│   "), depth + 1)
                else:
                    lines.append(f"{prefix}{branch}{name}  ({self._fmt(self.backend.size(child))})")
            if len(names) > max_files:
                lines.append(f"{prefix}└── … {len(names) - max_files} more")

        walk(root, "", 0)
        return "\n".join(lines)

    @staticmethod
    def _fmt(size: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024 or unit == "GB":
                return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
            size /= 1024
        return f"{size} B"
