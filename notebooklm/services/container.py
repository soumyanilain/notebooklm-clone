"""Composition root: builds every component once and wires them together.

The UI, the evaluation script and the tests all get their objects from here, so
there is exactly one place that decides which storage, vector store, embedder,
reranker and LLM implementations are used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..config import Settings, get_settings
from ..ingestion import IngestionService
from ..llm import LLMClient, get_llm
from ..retrieval import ChromaVectorStore, Embedder, Reranker, Retriever, get_embedder, get_reranker
from ..storage import LocalStorage, NotebookStore
from .artifacts import ArtifactService
from .chat import ChatService

log = logging.getLogger(__name__)


@dataclass
class Services:
    settings: Settings
    store: NotebookStore
    embedder: Embedder
    reranker: Reranker
    vstore: ChromaVectorStore
    retriever: Retriever
    llm: LLMClient
    ingestion: IngestionService
    chat: ChatService
    artifacts: ArtifactService
    sync: object | None = None


def build_services(settings: Settings | None = None) -> Services:
    settings = settings or get_settings()
    sync = None
    if settings.hf_dataset_repo:
        # Optional: restore notebooks from a private HF dataset repo before anything reads them.
        from ..storage.hf_sync import HFDatasetSync

        sync = HFDatasetSync(settings)
        sync.restore()

    backend = LocalStorage(settings.data_dir)
    store = NotebookStore(backend, vector_root=settings.vector_dir)
    embedder = get_embedder(settings)
    reranker = get_reranker(settings)
    vstore = ChromaVectorStore(store)
    retriever = Retriever(settings, store, embedder, vstore, reranker)
    llm = get_llm(settings)
    services = Services(
        settings=settings, store=store, embedder=embedder, reranker=reranker, vstore=vstore,
        retriever=retriever, llm=llm,
        ingestion=IngestionService(settings, store, embedder, vstore, retriever),
        chat=ChatService(settings, store, retriever, llm),
        artifacts=ArtifactService(settings, store, retriever, llm),
        sync=sync,
    )
    if sync is not None:
        sync.attach(store)
    log.info("Data dir: %s | embeddings: %s | reranker: %s | LLM backend: %s",
             settings.data_dir, embedder.name, reranker.name, settings.llm_backend)
    return services


def warm_up(services: Services) -> None:
    """Load the ONNX models up front so the first question isn't slow."""
    try:
        services.embedder.embed_query("warm up")
        services.reranker.score("warm up", ["warm up"])
    except Exception:  # noqa: BLE001 - warm-up is best effort
        log.warning("Model warm-up failed; models will load on first use", exc_info=True)
