"""Vector store abstraction + ChromaDB implementation.

Each notebook gets its own persistent Chroma directory (see NotebookStore.vector_dir),
so notebooks are isolated and deleting one removes its index with it.

The index is *derived* data: chunks/*.jsonl is the source of truth. If an index is
missing or out of sync (e.g. after a cold start on a host with ephemeral disk, or
when VECTOR_DIR points somewhere temporary) it is rebuilt from the chunk files.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable

from ..models import Chunk
from ..storage import NotebookStore

log = logging.getLogger(__name__)

COLLECTION = "chunks"


class VectorStore(ABC):
    @abstractmethod
    def add(self, nb_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    @abstractmethod
    def query(self, nb_id: str, embedding: list[float], k: int,
              source_ids: list[str] | None = None) -> list[tuple[str, float]]:
        """Return [(chunk_id, cosine_similarity)] best first."""

    @abstractmethod
    def delete_source(self, nb_id: str, source_id: str) -> None: ...

    @abstractmethod
    def count(self, nb_id: str) -> int: ...

    @abstractmethod
    def close(self, nb_id: str) -> None: ...

    @abstractmethod
    def reset(self, nb_id: str) -> None: ...


class ChromaVectorStore(VectorStore):
    def __init__(self, store: NotebookStore):
        self.store = store
        self._clients: dict[str, object] = {}
        self._lock = threading.RLock()
        store.on_before_notebook_delete.append(self.close)

    def _client(self, nb_id: str):
        with self._lock:
            client = self._clients.get(nb_id)
            if client is None:
                import chromadb
                from chromadb.config import Settings as ChromaSettings

                client = chromadb.PersistentClient(
                    path=str(self.store.vector_dir(nb_id)),
                    settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
                )
                self._clients[nb_id] = client
            return client

    def _collection(self, nb_id: str):
        return self._client(nb_id).get_or_create_collection(
            name=COLLECTION, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )

    def add(self, nb_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        col = self._collection(nb_id)
        batch = 500
        for i in range(0, len(chunks), batch):
            part = chunks[i: i + batch]
            col.upsert(
                ids=[c.id for c in part],
                embeddings=embeddings[i: i + batch],
                documents=[c.text for c in part],
                metadatas=[{"source_id": c.source_id, "chunk_index": c.chunk_index, "locator": c.locator}
                           for c in part],
            )

    def query(self, nb_id: str, embedding: list[float], k: int,
              source_ids: list[str] | None = None) -> list[tuple[str, float]]:
        col = self._collection(nb_id)
        total = col.count()
        if total == 0:
            return []
        where = None
        if source_ids is not None:
            if not source_ids:
                return []
            where = {"source_id": {"$in": list(source_ids)}} if len(source_ids) > 1 else {"source_id": source_ids[0]}
        res = col.query(query_embeddings=[embedding], n_results=min(k, total), where=where,
                        include=["distances"])
        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]
        return [(cid, 1.0 - float(d)) for cid, d in zip(ids, dists)]

    def delete_source(self, nb_id: str, source_id: str) -> None:
        self._collection(nb_id).delete(where={"source_id": source_id})

    def count(self, nb_id: str) -> int:
        return self._collection(nb_id).count()

    def reset(self, nb_id: str) -> None:
        client = self._client(nb_id)
        try:
            client.delete_collection(COLLECTION)
        except Exception:  # noqa: BLE001 - missing collection is fine
            pass

    def close(self, nb_id: str) -> None:
        """Release the Chroma client so its files can be deleted (matters on Windows)."""
        with self._lock:
            client = self._clients.pop(nb_id, None)
        if client is not None:
            try:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
            except Exception:  # noqa: BLE001
                log.debug("Chroma close failed for %s", nb_id, exc_info=True)
            import gc

            gc.collect()

    def ensure_synced(self, nb_id: str, chunks: list[Chunk],
                      embed: Callable[[list[str]], list[list[float]]]) -> bool:
        """Rebuild the index from the chunk files if counts disagree. Returns True if rebuilt."""
        if self.count(nb_id) == len(chunks):
            return False
        log.info("Rebuilding vector index for %s (%d chunks)", nb_id, len(chunks))
        self.reset(nb_id)
        if chunks:
            self.add(nb_id, chunks, embed([c.text for c in chunks]))
        return True
