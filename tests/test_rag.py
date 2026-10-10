from notebooklm.config import RETRIEVAL_METHODS, get_settings
from notebooklm.services import build_services
from tests.conftest import CORPUS


def test_every_retrieval_method_returns_cited_chunks(loaded):
    services, nb = loaded
    for method in RETRIEVAL_METHODS:
        result = services.retriever.retrieve(nb.id, "root cause of INC-2047 expired certificate", method, k=4)
        assert len(result.chunks) == 4, method
        assert result.timings_ms["total"] >= 0
        assert any("INC-2047" in rc.chunk.text for rc in result.chunks), method
        assert all(rc.chunk.locator for rc in result.chunks)


def test_chat_saves_history_with_citations(loaded):
    services, nb = loaded
    turn = services.chat.ask(nb.id, "What was the root cause of INC-2047?")
    assert turn.citations and turn.citations[0]["source_name"]
    history = services.store.load_history(nb.id)
    assert [m.role for m in history] == ["user", "assistant"]
    assert history[1].citations == turn.citations
    assert "<summary>Source" in services.chat.history_for_display(nb.id)[1]["content"]


def test_chat_without_sources_does_not_call_llm(services):
    nb = services.store.create_notebook("Empty")
    turn = services.chat.ask(nb.id, "anything?")
    assert turn.model is None and "no sources" in turn.answer.lower()


def test_disabled_sources_are_excluded(loaded):
    services, nb = loaded
    deck = next(s for s in services.store.list_sources(nb.id) if s.type == "pptx")
    services.store.set_sources_enabled(nb.id, [deck.id])
    result = services.retriever.retrieve(nb.id, "telemetry outage", "hybrid", k=5)
    assert {rc.chunk.source_id for rc in result.chunks} == {deck.id}


def test_artifacts_are_saved_as_markdown(loaded):
    services, nb = loaded
    report, md = services.artifacts.generate_report(nb.id)
    assert md.startswith("# Report") and "## Sources" in md
    quiz, qmd = services.artifacts.generate_quiz(nb.id, 5)
    assert "## Answer key" in qmd and quiz.meta["answer_key"]
    cards, cmd = services.artifacts.generate_flashcards(nb.id, 5)
    assert "| Front | Back |" in cmd
    assert {a.type for a in services.store.list_artifacts(nb.id)} == {"report", "quiz", "flashcards"}
    path = services.store.artifact_path(nb.id, quiz.id)
    assert path.suffix == ".md" and path.read_text(encoding="utf-8") == qmd


def test_notebooks_are_isolated(loaded):
    services, nb = loaded
    other = services.store.create_notebook("Other")
    assert services.retriever.retrieve(other.id, "INC-2047", "vector").chunks == []


def test_remove_source_updates_index(loaded):
    services, nb = loaded
    src = next(s for s in services.store.list_sources(nb.id) if s.name == "postmortem_INC-2047.txt")
    services.ingestion.remove_source(nb.id, src.id)
    result = services.retriever.retrieve(nb.id, "INC-2047 tg-east-2", "hybrid", k=10)
    assert all(rc.chunk.source_id != src.id for rc in result.chunks)
    assert services.vstore.count(nb.id) == len(services.store.load_chunks(nb.id))


def test_duplicate_upload_is_rejected(loaded):
    import pytest

    from notebooklm.ingestion import ExtractionError

    services, nb = loaded
    with pytest.raises(ExtractionError):
        path = CORPUS / "postmortem_INC-2047.txt"
        services.ingestion.add_file(nb.id, path, path.name)


def test_data_survives_restart_and_index_rebuilds(loaded):
    services, nb = loaded
    services.chat.ask(nb.id, "What is the HX-7 flight time?")
    services.artifacts.generate_report(nb.id)
    services.vstore.close(nb.id)
    # simulate a host that lost the vector index (ephemeral disk / VECTOR_DIR on tmp)
    import shutil

    shutil.rmtree(services.store.vector_dir(nb.id))
    restarted = build_services(get_settings())
    assert [n.name for n in restarted.store.list_notebooks()] == ["Halcyon"]
    assert len(restarted.store.load_history(nb.id)) == 2
    assert len(restarted.store.list_artifacts(nb.id)) == 1
    result = restarted.retriever.retrieve(nb.id, "INC-2047", "vector", k=3)
    assert len(result.chunks) == 3


def test_ui_builds(services):
    from notebooklm.ui.gradio_app import build_app

    assert build_app(services) is not None


def test_citation_styles_are_normalized():
    from notebooklm.llm import normalize_citations

    assert normalize_citations("Root cause【1†L9-L13】【5†L1-L3】") == "Root cause [1][5]"
    assert normalize_citations("A [1, 3] B [2-4] C [Source 2] D [1†L2]") == "A [1][3] B [2][3][4] C [2] D [1]"
    assert normalize_citations("plain [1] stays") == "plain [1] stays"


def test_gpt_oss_style_citations_become_source_list(loaded):
    from notebooklm.services.chat import build_citations

    services, nb = loaded
    result = services.retriever.retrieve(nb.id, "INC-2047 root cause", "hybrid", k=5)
    answer, citations = build_citations("Expired certificate【1†L9-L13】【5†L1-L3】【9†L1】", result.chunks)
    assert answer == "Expired certificate [1][5]"
    assert [c["n"] for c in citations] == [1, 5]


def test_quiz_answers_are_shuffled():
    from notebooklm.services.artifacts import ArtifactService

    qs = [{"question": f"Q{i}", "options": ["right", "w1", "w2", "w3"], "answer": "A", "explanation": "", "source": 1}
          for i in range(40)]
    shuffled = ArtifactService._shuffle_options(qs, seed=7)
    letters = {q["answer"] for q in shuffled}
    assert len(letters) >= 3
    assert all(q["options"]["ABCD".index(q["answer"])] == "right" for q in shuffled)
