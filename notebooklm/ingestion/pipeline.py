"""Ingestion: file/URL -> extracted text -> chunks -> embeddings -> vector index.

Each stage's output is saved in the notebook folder (raw file, extracted text,
chunks), so the vector index can always be rebuilt and every citation can be
traced back to the original page, slide or section.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from ..config import Settings
from ..models import Source, new_id
from ..retrieval import Embedder, Retriever
from ..retrieval.vector_store import ChromaVectorStore
from ..storage import NotebookStore
from .chunker import chunk_segments
from .extractors import ExtractionError, detect_type, extract_file, fetch_url, normalize_url

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    source: Source
    seconds: float


class IngestionService:
    def __init__(self, settings: Settings, store: NotebookStore, embedder: Embedder,
                 vstore: ChromaVectorStore, retriever: Retriever):
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.vstore = vstore
        self.retriever = retriever

    # -- public API ---------------------------------------------------------------
    def add_file(self, nb_id: str, path: str | Path, original_name: str | None = None) -> IngestResult:
        started = time.perf_counter()
        path = Path(path)
        name = original_name or path.name
        if not path.exists():
            raise ExtractionError(f"Upload of '{name}' failed; please try again.")
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self.settings.max_upload_mb:
            raise ExtractionError(f"'{name}' is {size_mb:.1f} MB; the limit is {self.settings.max_upload_mb} MB.")
        detect_type(name)  # fail fast on unsupported types
        data = path.read_bytes()
        if not data:
            raise ExtractionError(f"'{name}' is empty.")

        digest = hashlib.sha256(data).hexdigest()
        for existing in self.store.list_sources(nb_id):
            if existing.sha256 == digest:
                raise ExtractionError(f"'{name}' is already in this notebook (as '{existing.name}').")

        kind, _title, segments = extract_file(data, name)
        source = Source(id=new_id("s_"), notebook_id=nb_id, name=name, type=kind, origin=name, sha256=digest)
        source.raw_key = self.store.save_raw_file(nb_id, source.id, name, path)
        self._index(nb_id, source, segments, url=None)
        return IngestResult(source, time.perf_counter() - started)

    def add_url(self, nb_id: str, url: str) -> IngestResult:
        started = time.perf_counter()
        url = normalize_url(url)
        for existing in self.store.list_sources(nb_id):
            if existing.type == "url" and existing.origin.rstrip("/") == url.rstrip("/"):
                raise ExtractionError("That URL is already in this notebook.")
        page = fetch_url(url, timeout=self.settings.url_timeout, max_mb=self.settings.max_url_mb)
        title = (page.title or page.url)[:120]
        source = Source(id=new_id("s_"), notebook_id=nb_id, name=title, type="url", origin=page.url,
                        sha256=hashlib.sha256(page.raw).hexdigest())
        source.raw_key = self.store.save_raw_bytes(nb_id, source.id, f"page{page.raw_ext}", page.raw)
        self._index(nb_id, source, page.segments, url=page.url)
        return IngestResult(source, time.perf_counter() - started)

    def remove_source(self, nb_id: str, source_id: str) -> Source:
        with self.store.lock(nb_id):
            source = self.store.get_source(nb_id, source_id)
            self.retriever.ensure_index(nb_id)
            self.vstore.delete_source(nb_id, source_id)
            self.store.delete_source_files(nb_id, source_id)
            self.retriever.invalidate(nb_id)
            self.store.touch(nb_id)
            return source

    # -- internals ----------------------------------------------------------------
    def _index(self, nb_id: str, source: Source, segments, url: str | None) -> None:
        chunks = chunk_segments(source, segments, self.settings.chunk_size, self.settings.chunk_overlap, url=url)
        if not chunks:
            raise ExtractionError(f"'{source.name}' produced no usable text.")
        with self.store.lock(nb_id):
            try:
                # Make sure the index already matches existing chunks before adding new ones.
                self.retriever.ensure_index(nb_id)
                source.text_key = self.store.save_extracted_text(nb_id, source, segments)
                embeddings = self.embedder.embed_documents([c.text for c in chunks])
                self.store.save_chunks(nb_id, source.id, chunks)
                source.n_segments = len(segments)
                source.n_chunks = len(chunks)
                source.n_chars = sum(len(s.text) for s in segments)
                self.store.save_source(source)
                self.vstore.add(nb_id, chunks, embeddings)
            except Exception:
                log.exception("Ingestion failed for %s", source.name)
                self._cleanup_failed(nb_id, source)
                raise
            finally:
                self.retriever.invalidate(nb_id)
            self.store.touch(nb_id)

    def _cleanup_failed(self, nb_id: str, source: Source) -> None:
        try:
            if any(s.id == source.id for s in self.store.list_sources(nb_id)):
                self.store.delete_source_files(nb_id, source.id)
            else:
                for key in (source.raw_key, source.text_key):
                    if key:
                        self.store.backend.delete(key)
            self.vstore.delete_source(nb_id, source.id)
        except Exception:  # noqa: BLE001
            log.debug("cleanup after failed ingestion also failed", exc_info=True)
