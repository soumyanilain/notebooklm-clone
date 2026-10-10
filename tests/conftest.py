"""Tests run fully offline: hashed embeddings, lexical reranker and a fake LLM."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.update({"EMBEDDING_BACKEND": "hash", "RERANKER_BACKEND": "lexical", "LLM_BACKEND": "fake",
                   "GROQ_API_KEY": "", "HF_DATASET_REPO": ""})

CORPUS = ROOT / "eval" / "corpus"


@pytest.fixture()
def services(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    from notebooklm.config import get_settings
    from notebooklm.services import build_services

    get_settings.cache_clear()
    svc = build_services(get_settings())
    yield svc
    get_settings.cache_clear()


@pytest.fixture()
def loaded(services):
    nb = services.store.create_notebook("Halcyon")
    for path in sorted(CORPUS.iterdir()):
        services.ingestion.add_file(nb.id, path, path.name)
    return services, nb
