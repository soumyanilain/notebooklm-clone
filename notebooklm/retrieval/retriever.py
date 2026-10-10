"""Retrieval strategies compared in the evaluation.

vector         dense cosine similarity (bge-small) top-k                 baseline
hybrid         dense top-N + BM25 top-N fused with Reciprocal Rank Fusion
rerank         dense top-N re-scored by a cross-encoder, keep top-k
hybrid_rerank  hybrid top-N re-scored by a cross-encoder, keep top-k
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from ..config import RETRIEVAL_METHODS, Settings
from ..models import Chunk, RetrievedChunk
from ..storage import NotebookStore
from .bm25 import BM25Cache
from .embeddings import Embedder
from .reranker import Reranker
from .vector_store import ChromaVectorStore

RRF_K = 60


@dataclass
class RetrievalResult:
    method: str
    query: str
    chunks: list[RetrievedChunk]
    timings_ms: dict[str, float] = field(default_factory=dict)
    candidates: int = 0

    def to_dict(self) -> dict:
        return {"method": self.method, "query": self.query, "timings_ms": self.timings_ms,
                "candidates": self.candidates, "chunks": [c.to_dict() for c in self.chunks]}


class Retriever:
    def __init__(self, settings: Settings, store: NotebookStore, embedder: Embedder,
                 vstore: ChromaVectorStore, reranker: Reranker):
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.vstore = vstore
        self.reranker = reranker
        self.bm25 = BM25Cache()
        self._chunk_cache: dict[str, dict[str, Chunk]] = {}
        self._synced: set[str] = set()
        self._lock = threading.Lock()
        store.on_notebook_deleted.append(self.invalidate)

    # -- caches -----------------------------------------------------------------
    def invalidate(self, nb_id: str) -> None:
        with self._lock:
            self._chunk_cache.pop(nb_id, None)
            self._synced.discard(nb_id)
        self.bm25.invalidate(nb_id)

    def all_chunks(self, nb_id: str) -> dict[str, Chunk]:
        with self._lock:
            cached = self._chunk_cache.get(nb_id)
        if cached is not None:
            return cached
        chunks = {c.id: c for c in self.store.load_chunks(nb_id)}
        with self._lock:
            self._chunk_cache[nb_id] = chunks
        return chunks

    def ensure_index(self, nb_id: str) -> None:
        if nb_id in self._synced:
            return
        chunks = list(self.all_chunks(nb_id).values())
        self.vstore.ensure_synced(nb_id, chunks, self.embedder.embed_documents)
        with self._lock:
            self._synced.add(nb_id)

    def enabled_source_ids(self, nb_id: str) -> list[str]:
        return [s.id for s in self.store.list_sources(nb_id) if s.enabled]

    # -- main entry point ---------------------------------------------------------------
    def retrieve(self, nb_id: str, query: str, method: str | None = None, k: int | None = None,
                 source_ids: list[str] | None = None) -> RetrievalResult:
        method = method if method in RETRIEVAL_METHODS else self.settings.default_method
        k = k or self.settings.top_k
        n_cand = max(self.settings.candidate_k, k)
        timings: dict[str, float] = {}
        t_start = time.perf_counter()

        if source_ids is None:
            source_ids = self.enabled_source_ids(nb_id)
        if not source_ids:
            return RetrievalResult(method, query, [], {"total": 0.0})

        chunk_map = self.all_chunks(nb_id)
        self.ensure_index(nb_id)

        t = time.perf_counter()
        q_emb = self.embedder.embed_query(query)
        timings["embed_query"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        dense = self.vstore.query(nb_id, q_emb, n_cand if method != "vector" else k, source_ids)
        timings["vector_search"] = (time.perf_counter() - t) * 1000
        dense_scores = dict(dense)

        bm25_scores: dict[str, float] = {}
        if method in {"hybrid", "hybrid_rerank"}:
            t = time.perf_counter()
            index = self.bm25.get(nb_id, list(chunk_map.values()))
            sparse = index.search(query, n_cand, set(source_ids))
            bm25_scores = dict(sparse)
            timings["bm25_search"] = (time.perf_counter() - t) * 1000
            fused = self._rrf([[cid for cid, _ in dense], [cid for cid, _ in sparse]])
            candidates = [(cid, score) for cid, score in fused[:n_cand]]
        else:
            candidates = dense

        if method in {"rerank", "hybrid_rerank"} and candidates:
            t = time.perf_counter()
            cand_ids = [cid for cid, _ in candidates if cid in chunk_map]
            scores = self.reranker.score(query, [chunk_map[cid].text for cid in cand_ids])
            first_stage = dict(candidates)
            ranked = sorted(zip(cand_ids, scores), key=lambda x: x[1], reverse=True)[:k]
            timings["rerank"] = (time.perf_counter() - t) * 1000
            final = [(cid, s, {"rerank": s, "first_stage": first_stage.get(cid, 0.0)}) for cid, s in ranked]
        else:
            final = [(cid, s, {}) for cid, s in candidates[:k]]

        results: list[RetrievedChunk] = []
        for rank, (cid, score, extra) in enumerate(final, start=1):
            chunk = chunk_map.get(cid)
            if chunk is None:
                continue
            breakdown = {"vector": round(dense_scores.get(cid, 0.0), 4)}
            if bm25_scores:
                breakdown["bm25"] = round(bm25_scores.get(cid, 0.0), 4)
            breakdown.update({key: round(val, 4) for key, val in extra.items()})
            results.append(RetrievedChunk(chunk=chunk, score=round(float(score), 4), rank=rank, scores=breakdown))

        timings["total"] = (time.perf_counter() - t_start) * 1000
        timings = {key: round(val, 1) for key, val in timings.items()}
        return RetrievalResult(method, query, results, timings, candidates=len(candidates))

    @staticmethod
    def _rrf(rankings: list[list[str]]) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for ranking in rankings:
            for position, cid in enumerate(ranking, start=1):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + position)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # -- artifact context -------------------------------------------------------------------
    def coverage_sample(self, nb_id: str, source_ids: list[str], budget_chars: int) -> list[Chunk]:
        """Pick chunks spread evenly across every selected source (for reports/quizzes),
        so long documents don't crowd out short ones."""
        by_source: dict[str, list[Chunk]] = {}
        for chunk in self.all_chunks(nb_id).values():
            if chunk.source_id in source_ids:
                by_source.setdefault(chunk.source_id, []).append(chunk)
        if not by_source:
            return []
        for chunks in by_source.values():
            chunks.sort(key=lambda c: c.chunk_index)
        total_chars = sum(len(c.text) for chunks in by_source.values() for c in chunks)
        if total_chars <= budget_chars:
            return [c for sid in source_ids for c in by_source.get(sid, [])]

        per_source = budget_chars / len(by_source)
        picked: list[Chunk] = []
        for sid in source_ids:
            chunks = by_source.get(sid, [])
            if not chunks:
                continue
            avg = sum(len(c.text) for c in chunks) / len(chunks)
            n = max(1, min(len(chunks), int(per_source // max(avg, 1))))
            step = len(chunks) / n
            picked.extend(chunks[int(i * step)] for i in range(n))
        return picked
