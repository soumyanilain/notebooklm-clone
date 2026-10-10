"""Keyword retrieval with BM25 (rank_bm25), cached per notebook."""

from __future__ import annotations

import re
import threading

from rank_bm25 import BM25Okapi

from ..models import Chunk

STOPWORDS = set(
    """a about above after again against all am an and any are as at be because been before being below between
    both but by can could did do does doing down during each few for from further had has have having he her here
    hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off
    on once only or other our ours ourselves out over own same she should so some such than that the their theirs
    them themselves then there these they this those through to too under until up very was we were what when where
    which while who whom why will with would you your yours yourself yourselves""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.text) or ["_"] for c in chunks]) if chunks else None

    def search(self, query: str, k: int, source_ids: set[str] | None = None) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in ranked:
            if scores[i] <= 0:
                break
            if source_ids is not None and self.chunks[i].source_id not in source_ids:
                continue
            out.append((self.chunks[i].id, float(scores[i])))
            if len(out) >= k:
                break
        return out


class BM25Cache:
    """One BM25 index per notebook, rebuilt when the chunk set changes."""

    def __init__(self):
        self._cache: dict[str, tuple[tuple, BM25Index]] = {}
        self._lock = threading.Lock()

    def get(self, nb_id: str, chunks: list[Chunk]) -> BM25Index:
        signature = (len(chunks), chunks[0].id if chunks else "", chunks[-1].id if chunks else "")
        with self._lock:
            cached = self._cache.get(nb_id)
            if cached and cached[0] == signature:
                return cached[1]
            index = BM25Index(chunks)
            self._cache[nb_id] = (signature, index)
            return index

    def invalidate(self, nb_id: str) -> None:
        with self._lock:
            self._cache.pop(nb_id, None)
