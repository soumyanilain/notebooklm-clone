"""Embedding models.

FastEmbedEmbedder runs BAAI/bge-small-en-v1.5 through ONNX Runtime (no PyTorch),
which keeps the install and the container small and works on any CPU.
HashEmbedder is a dependency-free stand-in used by the offline test suite.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
from abc import ABC, abstractmethod

from ..config import Settings

log = logging.getLogger(__name__)


class Embedder(ABC):
    name: str
    dim: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class FastEmbedEmbedder(Embedder):
    def __init__(self, model_name: str, cache_dir: str):
        self.name = model_name
        self._cache_dir = cache_dir
        self._model = None
        self._lock = threading.Lock()
        self.dim = 384

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed import TextEmbedding

                    log.info("Loading embedding model %s", self.name)
                    self._model = TextEmbedding(model_name=self.name, cache_dir=self._cache_dir)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        vectors = [_normalize(v.tolist()) for v in model.embed(texts, batch_size=32)]
        self.dim = len(vectors[0])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        model = self._load()
        vec = next(iter(model.query_embed(text)))
        return _normalize(vec.tolist())


class HashEmbedder(Embedder):
    """Feature-hashed bag of words + bigrams. Deterministic and offline; only for tests."""

    def __init__(self, dim: int = 512):
        self.name = f"hash-{dim}"
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        feats = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        vec = [0.0] * self.dim
        for feat in feats:
            h = int(hashlib.md5(feat.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        return _normalize(vec)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


_embedder: Embedder | None = None
_embedder_lock = threading.Lock()


def get_embedder(settings: Settings) -> Embedder:
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            if settings.embedding_backend == "hash":
                _embedder = HashEmbedder()
            else:
                _embedder = FastEmbedEmbedder(settings.embedding_model, str(settings.model_cache_dir))
        return _embedder
