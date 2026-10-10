"""Download the embedding and reranker models into MODEL_CACHE_DIR (used by the Dockerfile)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notebooklm.config import get_settings  # noqa: E402
from notebooklm.retrieval import get_embedder, get_reranker  # noqa: E402

settings = get_settings()
print("Embedding dims:", len(get_embedder(settings).embed_query("hello")))
print("Reranker score:", get_reranker(settings).score("hello", ["hello world"]))
print("Models cached in", settings.model_cache_dir)
