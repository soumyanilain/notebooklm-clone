"""Second-stage reranking.

CrossEncoderReranker scores (query, passage) pairs jointly with
ms-marco-MiniLM-L-6-v2 (ONNX via fastembed). It reads the query and passage
together, so it is better than the bi-encoder at judging relevance, but it costs
one model pass per candidate, which is why it only reranks a short candidate list.
LexicalReranker is an offline stand-in for tests.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod

from ..config import Settings
from .bm25 import tokenize

log = logging.getLogger(__name__)


class Reranker(ABC):
    name: str

    @abstractmethod
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str, cache_dir: str):
        self.name = model_name
        self._cache_dir = cache_dir
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed.rerank.cross_encoder import TextCrossEncoder

                    log.info("Loading reranker %s", self.name)
                    self._model = TextCrossEncoder(model_name=self.name, cache_dir=self._cache_dir)
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return [float(s) for s in self._load().rerank(query, passages, batch_size=32)]


class LexicalReranker(Reranker):
    name = "lexical-overlap"

    def score(self, query: str, passages: list[str]) -> list[float]:
        q = set(tokenize(query))
        return [len(q & set(tokenize(p))) / (len(q) or 1) for p in passages]


_reranker: Reranker | None = None
_reranker_lock = threading.Lock()


def get_reranker(settings: Settings) -> Reranker:
    global _reranker
    with _reranker_lock:
        if _reranker is None:
            if settings.reranker_backend == "lexical":
                _reranker = LexicalReranker()
            else:
                _reranker = CrossEncoderReranker(settings.reranker_model, str(settings.model_cache_dir))
        return _reranker
