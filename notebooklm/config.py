"""Application settings, read once from environment variables (and .env locally).

Every tunable lives here so the rest of the code never touches os.environ directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Retrieval strategies the app supports. Keys are stored with chat messages and
# used by the evaluation script, so keep them stable.
RETRIEVAL_METHODS: dict[str, str] = {
    "vector": "Vector search (dense only)",
    "hybrid": "Hybrid: BM25 + vector (RRF)",
    "rerank": "Vector + cross-encoder rerank",
    "hybrid_rerank": "Hybrid + cross-encoder rerank",
}


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = _env(name)
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    # --- storage ---
    data_dir: Path
    # When set, Chroma indexes live here (one sub-folder per notebook) instead of
    # inside each notebook folder. Useful when DATA_DIR is a synced/slow volume.
    vector_dir: Path | None

    # --- LLM ---
    llm_backend: str  # "groq" or "fake" (offline tests)
    groq_api_key: str | None
    chat_models: list[str]
    judge_models: list[str]
    llm_temperature: float
    chat_max_tokens: int
    artifact_max_tokens: int

    # --- embeddings / reranking ---
    embedding_backend: str  # "fastembed" or "hash" (offline tests)
    embedding_model: str
    reranker_backend: str  # "fastembed" or "lexical" (offline tests)
    reranker_model: str
    model_cache_dir: Path

    # --- RAG parameters ---
    chunk_size: int
    chunk_overlap: int
    top_k: int
    candidate_k: int
    default_method: str
    history_turns: int
    artifact_context_chars: int

    # --- ingestion limits ---
    max_upload_mb: int
    url_timeout: int
    max_url_mb: int

    # --- optional persistence on Hugging Face (dataset repo sync) ---
    hf_dataset_repo: str | None
    hf_token: str | None
    hf_sync_minutes: int

    extra: dict = field(default_factory=dict)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_dir = Path(_env("DATA_DIR", str(PROJECT_ROOT / "data"))).expanduser().resolve()
    vector_dir_raw = _env("VECTOR_DIR")
    default_method = _env("RETRIEVAL_METHOD", "hybrid")  # chosen by the evaluation, see docs/RAG_EVALUATION.md
    if default_method not in RETRIEVAL_METHODS:
        default_method = "hybrid"
    try:
        temperature = float(_env("LLM_TEMPERATURE", "0.2"))
    except ValueError:
        temperature = 0.2

    return Settings(
        data_dir=data_dir,
        vector_dir=Path(vector_dir_raw).expanduser().resolve() if vector_dir_raw else None,
        llm_backend=(_env("LLM_BACKEND", "groq") or "groq").lower(),
        groq_api_key=_env("GROQ_API_KEY"),
        # First model is preferred; the rest are fallbacks used on rate limits or
        # if Groq retires a model.
        chat_models=_env_list(
            "GROQ_MODELS",
            ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        ),
        judge_models=_env_list(
            "GROQ_JUDGE_MODELS", ["openai/gpt-oss-20b", "llama-3.1-8b-instant", "openai/gpt-oss-120b"]
        ),
        llm_temperature=temperature,
        chat_max_tokens=_env_int("CHAT_MAX_TOKENS", 1500),
        artifact_max_tokens=_env_int("ARTIFACT_MAX_TOKENS", 3500),
        embedding_backend=(_env("EMBEDDING_BACKEND", "fastembed") or "fastembed").lower(),
        embedding_model=_env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        reranker_backend=(_env("RERANKER_BACKEND", "fastembed") or "fastembed").lower(),
        reranker_model=_env("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"),
        model_cache_dir=Path(_env("MODEL_CACHE_DIR", str(PROJECT_ROOT / ".model_cache"))).expanduser().resolve(),
        chunk_size=_env_int("CHUNK_SIZE", 1000),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 150),
        top_k=_env_int("TOP_K", 5),
        candidate_k=_env_int("CANDIDATE_K", 20),
        default_method=default_method,
        history_turns=_env_int("HISTORY_TURNS", 2),
        artifact_context_chars=_env_int("ARTIFACT_CONTEXT_CHARS", 14000),
        max_upload_mb=_env_int("MAX_UPLOAD_MB", 25),
        url_timeout=_env_int("URL_TIMEOUT", 20),
        max_url_mb=_env_int("MAX_URL_MB", 15),
        hf_dataset_repo=_env("HF_DATASET_REPO"),
        hf_token=_env("HF_TOKEN"),
        hf_sync_minutes=_env_int("HF_SYNC_MINUTES", 2),
    )
