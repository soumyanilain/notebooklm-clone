"""Gradio front end. Every callback is a thin adapter: read inputs, call a service,
turn the result (or a friendly error) into component updates. No business logic here."""

from __future__ import annotations

import functools
import html
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import gradio as gr

from ..ingestion import ExtractionError
from ..llm import LLMError
from ..services import ArtifactError, Services, render_assistant
from ..services.chat import mark_citations, pretty_locator
from ..storage import StorageError
from .styles import active_style, style_css

log = logging.getLogger(__name__)

APP_NAME = os.getenv("APP_NAME", "📓 Sourcebook")
TAGLINE = "Add your documents, ask questions, and get answers that cite the page they came from."

AUTHOR = {
    "name": "Soumyanil Ain",
    "role": "MS Computer Science, UNC Charlotte",
    "course": "Course project for ITCS 5010",
    "links": [
        ("GitHub", "https://github.com/soumyanilain"),
        ("LinkedIn", "https://www.linkedin.com/in/soumyanilain"),
        ("Portfolio", "https://soumyanilain-portfolio.netlify.app"),
        ("Email", "mailto:soumyanilain@gmail.com"),
        ("Source code", "https://github.com/soumyanilain/notebooklm-clone"),
    ],
}

# Plain-language names for the retrieval strategies (keys match config.RETRIEVAL_METHODS).
METHOD_LABELS = {
    "hybrid": "Meaning + keywords (recommended)",
    "vector": "Meaning only",
    "rerank": "Meaning, then re-ranked (slower)",
    "hybrid_rerank": "Meaning + keywords, then re-ranked (slowest)",
}
TYPE_NAMES = {"pdf": "PDF", "pptx": "Slides", "txt": "Text", "url": "Web page"}
EXPECTED_ERRORS = (StorageError, ExtractionError, LLMError, ArtifactError, ValueError)

_STYLE = active_style()
THEME = _STYLE.theme
CSS = style_css(_STYLE)
HEAD = ""

HELP_MD = """
### Getting started
1. **Create a notebook** in the left column. A notebook keeps its own sources, chat history and study material, separate from every other notebook.
2. **Add sources.** Upload PDFs, PowerPoint decks (.pptx) or text files, or paste a link to a web page. Each source is read, split into short passages and indexed so it can be searched.
3. **Ask questions** in the Chat tab. Answers use only your sources. The highlighted numbers are citations; the list under each answer shows the file and the page, slide or line range each one points to.
4. **Make study material** in the Study material tab: a report, a quiz with an answer key, or flashcards. Each one is saved in the notebook as a Markdown file you can download.
5. **Choose what counts.** Untick a source in the left column to leave it out of answers and study material without deleting it.

### How answers are found
Each question is matched against your passages before the language model writes anything.

- **Meaning + keywords** (recommended) combines a meaning-based search with an exact keyword search, so it finds paraphrases *and* exact terms like form numbers, IDs and names.
- **Meaning only** compares the meaning of your question with every passage.
- **Re-ranked** options take the 20 best passages and re-score each one against your question with a second model. They're more careful but take a few seconds longer.

Open *How this answer was found* under the chat to see the passages that were used, their scores and how long each step took.

### Where your data is kept
Everything a notebook owns lives in its own folder: the original files, the extracted text, the passages, the search index, the chat history and the study material. The Storage tab shows that folder. The public cloud deployment runs on Google Cloud Run, whose disk is temporary: notebooks there disappear when the service restarts or scales to zero (the README explains how to keep them).
"""


def _friendly(n_out: int = 1, status_at: int | None = None):
    """Turn errors into a toast instead of breaking the outputs.

    Raising inside a Gradio callback marks every output component with an error
    badge, so expected problems (bad file, rate limit, missing key...) are shown as a
    warning toast and the outputs are left untouched (``gr.skip()``). If the event has
    a status box, the message is also written there.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except gr.Error as exc:
                message = str(getattr(exc, "message", exc))
            except EXPECTED_ERRORS as exc:
                message = str(exc)
            except Exception as exc:  # noqa: BLE001
                log.exception("Unexpected error in %s", fn.__name__)
                message = f"Something went wrong ({exc.__class__.__name__}: {exc}). The server log has details."
            gr.Warning(message, duration=10)
            outputs = [gr.skip()] * n_out
            if status_at is not None:
                outputs[status_at] = f"- {message}"
            return outputs[0] if n_out == 1 else tuple(outputs)

        return wrapper

    return decorator


def _plural(n: int, word: str, plural: str | None = None) -> str:
    return f"{n} {word if n == 1 else (plural or word + 's')}"


def _footer_html() -> str:
    links = "".join(f'<a href="{html.escape(url)}" target="_blank" rel="noopener">{html.escape(label)}</a>'
                    for label, url in AUTHOR["links"])
    return (f'<div class="app-footer"><div>Built by <strong>{AUTHOR["name"]}</strong>, {AUTHOR["role"]}. '
            f'{AUTHOR["course"]}.</div><nav aria-label="Author links">{links}</nav></div>')


def build_app(services: Services) -> gr.Blocks:
    store = services.store
    settings = services.settings

    # ------------------------------------------------------------------ view helpers
    def _require(nb_id: str | None) -> str:
        if not nb_id:
            raise gr.Error("Create a notebook first, or pick one from the list.")
        return nb_id

    def _source_choices(nb_id: str):
        sources = store.list_sources(nb_id)
        choices = [(f"{s.name} ({TYPE_NAMES.get(s.type, s.type)}, {_plural(s.n_chunks, 'passage')})", s.id)
                   for s in sources]
        return sources, choices

    def _artifact_choices(nb_id: str):
        arts = store.list_artifacts(nb_id)
        choices = []
        for a in arts:
            dt = datetime.fromisoformat(a.created_at).astimezone()
            when = f"{dt:%b} {dt.day}, {dt:%H:%M}"
            choices.append((f"{a.title}  ({when})", a.id))
        return arts, choices

    def _artifact_view(nb_id: str | None, art_id: str | None):
        if not nb_id or not art_id:
            return ("Nothing here yet. Choose a report, quiz or flashcards above and it will appear here, "
                    "ready to read or download."), None
        try:
            return mark_citations(store.read_artifact(nb_id, art_id)), str(store.artifact_path(nb_id, art_id))
        except (StorageError, OSError):
            return "This file couldn't be opened. It may have been deleted outside the app.", None

    def _summary(nb_id: str | None) -> str:
        if not nb_id:
            return ('<div class="nb-head"><div class="nb-title">👋 Start your first notebook</div>'
                    '<p class="nb-empty">Give it a name in the left column, then add a PDF, a slide deck, a text '
                    'file or a web page. Your questions will be answered from those sources only.</p></div>')
        nb = store.get_notebook(nb_id)
        sources = store.list_sources(nb_id)
        enabled = sum(1 for s in sources if s.enabled)
        passages = sum(s.n_chunks for s in sources if s.enabled)
        n_art = len(store.list_artifacts(nb_id))
        dt = datetime.fromisoformat(nb.created_at).astimezone()
        if not sources:
            chips = ["📎 No sources yet: add one on the left"]
        else:
            used = _plural(len(sources), "source") if enabled == len(sources) else f"{enabled} of {len(sources)} sources in use"
            chips = [f"📎 {used}", f"🧩 {_plural(passages, 'passage')}", f"🧾 {n_art} saved"]
        chips.append(f"🗓️ Created {dt:%b} {dt.day}, {dt.year}")
        chip_html = "".join(f"<span>{html.escape(c)}</span>" for c in chips)
        return f'<div class="nb-head"><div class="nb-title">{html.escape(nb.name)}</div><div class="nb-chips">{chip_html}</div></div>'

    def refresh_all(nb_id: str | None, art_id: str | None = None):
        notebooks = store.list_notebooks()
        nb_choices = [(nb.name, nb.id) for nb in notebooks]
        ids = {nb.id for nb in notebooks}
        if nb_id not in ids:
            nb_id = notebooks[0].id if notebooks else None
        if nb_id is None:
            empty_md, _ = _artifact_view(None, None)
            return (gr.update(choices=[], value=None), gr.update(choices=[], value=[]),
                    gr.update(choices=[], value=None), [], gr.update(choices=[], value=None), empty_md, None, "",
                    _summary(None), "")
        sources, src_choices = _source_choices(nb_id)
        arts, art_choices = _artifact_choices(nb_id)
        if art_id not in {a.id for a in arts}:
            art_id = arts[0].id if arts else None
        art_md, art_file = _artifact_view(nb_id, art_id)
        return (
            gr.update(choices=nb_choices, value=nb_id),
            gr.update(choices=src_choices, value=[s.id for s in sources if s.enabled]),
            gr.update(choices=src_choices, value=None),
            services.chat.history_for_display(nb_id),
            gr.update(choices=art_choices, value=art_id),
            art_md,
            art_file,
            store.storage_tree(nb_id),
            _summary(nb_id),
            "",
        )

    def retrieved_markdown(turn) -> str:
        r = turn.retrieval
        if not r.chunks:
            return "No passages were used for the last question."
        t = r.timings_ms
        steps = [f"embedding {t.get('embed_query', 0):.0f} ms", f"vector search {t.get('vector_search', 0):.0f} ms"]
        if "bm25_search" in t:
            steps.append(f"keyword search {t['bm25_search']:.0f} ms")
        if "rerank" in t:
            steps.append(f"re-ranking {t['rerank']:.0f} ms")
        lines = [
            f"Searched with **{METHOD_LABELS.get(r.method, r.method)}** and passed the top "
            f"{_plural(len(r.chunks), 'passage')} to `{turn.model}`.",
            "",
            f"<span class='timing'>Search took {t.get('total', 0):.0f} ms ({', '.join(steps)}); "
            f"writing the answer took {turn.llm_seconds:.2f} s; {turn.total_seconds:.2f} s in total.</span>",
            "",
        ]
        cited = {c["chunk_id"] for c in turn.citations}
        for rc in r.chunks:
            c = rc.chunk
            scores = ", ".join(f"{k.replace('_', ' ')} {v:.3f}" for k, v in rc.scores.items())
            snippet = re.sub(r"\s+", " ", c.text)[:300].rsplit(" ", 1)[0]
            used = "  (cited in the answer)" if c.id in cited else ""
            lines.append(f"<mark>{rc.rank}</mark> **{c.source_name}**, {pretty_locator(c.locator)}{used}  \n"
                         f"<span class='timing'>score {rc.score:.3f} ({scores})</span>\n\n> {snippet}…\n")
        return "\n".join(lines)

    # ------------------------------------------------------------------ callbacks
    @_friendly(10)
    def on_load():
        return refresh_all(None)

    @_friendly(11)
    def on_select(nb_id):
        return (*refresh_all(nb_id), "")

    @_friendly(11)
    def on_create(name):
        nb = store.create_notebook(name)
        gr.Info(f"Created “{nb.name}”. Add a source to get started.")
        return (*refresh_all(nb.id), "")

    @_friendly(11)
    def on_rename(nb_id, name):
        nb = store.rename_notebook(_require(nb_id), name)
        gr.Info(f"Renamed to “{nb.name}”.")
        return (*refresh_all(nb.id), "")

    @_friendly(2)
    def on_delete_ask(nb_id):
        nb = store.get_notebook(_require(nb_id))
        return gr.update(visible=True), (f"Delete **{nb.name}**? Its sources, chat history and study material "
                                         "will be removed for good.")

    def on_delete_cancel():
        return gr.update(visible=False), ""

    @_friendly(13)
    def on_delete_confirm(nb_id):
        nb = store.get_notebook(_require(nb_id))
        store.delete_notebook(nb.id)
        gr.Info(f"Deleted “{nb.name}”.")
        return (*refresh_all(None), gr.update(visible=False), "", "")

    @_friendly(12, status_at=11)
    def on_add_files(nb_id, files, progress=gr.Progress()):
        nb_id = _require(nb_id)
        if not files:
            raise gr.Error("Choose at least one PDF, PowerPoint or text file first.")
        files = files if isinstance(files, list) else [files]
        ok, failed = [], []
        for i, f in enumerate(files):
            path = Path(f if isinstance(f, str) else f.name)
            name = path.name
            progress((i, len(files)), desc=f"Reading {name}")
            try:
                result = services.ingestion.add_file(nb_id, path, name)
                s = result.source
                unit = {"pptx": "slide", "pdf": "page"}.get(s.type, "section")
                ok.append(f"- Added **{s.name}**: {_plural(s.n_segments, unit)}, "
                          f"{_plural(s.n_chunks, 'passage')} ({result.seconds:.1f} s)")
            except EXPECTED_ERRORS as exc:
                failed.append(f"- Skipped **{name}**: {exc}")
                gr.Warning(str(exc))
        if ok:
            gr.Info(f"Added {_plural(len(ok), 'source')}.")
        return (*refresh_all(nb_id), None, "\n".join(ok + failed))

    @_friendly(12, status_at=11)
    def on_add_url(nb_id, url):
        nb_id = _require(nb_id)
        result = services.ingestion.add_url(nb_id, url)
        s = result.source
        gr.Info(f"Added “{s.name}”.")
        status = (f"- Added **{s.name}**: {_plural(s.n_segments, 'section')}, {_plural(s.n_chunks, 'passage')} "
                  f"({result.seconds:.1f} s)")
        return (*refresh_all(nb_id), "", status)

    @_friendly(1)
    def on_toggle_sources(nb_id, enabled_ids):
        if nb_id:
            store.set_sources_enabled(nb_id, enabled_ids or [])
        return _summary(nb_id)

    @_friendly(11, status_at=10)
    def on_remove_source(nb_id, source_id):
        nb_id = _require(nb_id)
        if not source_id:
            raise gr.Error("Pick the source you want to remove.")
        src = services.ingestion.remove_source(nb_id, source_id)
        gr.Info(f"Removed “{src.name}”.")
        return (*refresh_all(nb_id), f"- Removed **{src.name}**.")

    def on_user_message(nb_id, message, history):
        message = (message or "").strip()
        if not nb_id or not message:
            return gr.update(), history or []
        return "", (history or []) + [{"role": "user", "content": message}]

    def on_answer(nb_id, history, method, top_k):
        history = history or []
        if not nb_id:
            gr.Warning("Create a notebook first, or pick one from the list.")
            return history, gr.update(), gr.update()
        if not history or history[-1]["role"] != "user":
            return history, gr.update(), gr.update()
        question = history[-1]["content"]
        if isinstance(question, list):  # defensive: multimodal message format
            question = " ".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in question)
        try:
            turn = services.chat.ask(nb_id, str(question), method=method, k=int(top_k))
        except EXPECTED_ERRORS as exc:
            gr.Warning(str(exc))
            return history + [{"role": "assistant", "content": f"Couldn't answer: {exc}"}], gr.update(), gr.update()
        except Exception as exc:  # noqa: BLE001
            log.exception("chat failed")
            return (history + [{"role": "assistant",
                                "content": f"Couldn't answer because of an unexpected error ({exc.__class__.__name__})."}],
                    gr.update(), gr.update())
        reply = render_assistant(turn.answer, turn.citations)
        return history + [{"role": "assistant", "content": reply}], retrieved_markdown(turn), store.storage_tree(nb_id)

    @_friendly(3)
    def on_clear_chat(nb_id):
        nb_id = _require(nb_id)
        store.clear_history(nb_id)
        gr.Info("Chat cleared.")
        return [], "", store.storage_tree(nb_id)

    def _after_artifact(nb_id, artifact, markdown, verb):
        _arts, choices = _artifact_choices(nb_id)
        gr.Info(f"{verb} and saved to this notebook.")
        return (gr.update(choices=choices, value=artifact.id), mark_citations(markdown),
                str(store.artifact_path(nb_id, artifact.id)), store.storage_tree(nb_id), _summary(nb_id))

    @_friendly(5)
    def on_report(nb_id, focus):
        nb_id = _require(nb_id)
        artifact, markdown = services.artifacts.generate_report(nb_id, focus)
        return _after_artifact(nb_id, artifact, markdown, "Report written")

    @_friendly(5)
    def on_quiz(nb_id, focus, n_questions, difficulty):
        nb_id = _require(nb_id)
        artifact, markdown = services.artifacts.generate_quiz(nb_id, int(n_questions), difficulty, focus)
        return _after_artifact(nb_id, artifact, markdown, "Quiz made")

    @_friendly(5)
    def on_flashcards(nb_id, focus, n_cards):
        nb_id = _require(nb_id)
        artifact, markdown = services.artifacts.generate_flashcards(nb_id, int(n_cards), focus)
        return _after_artifact(nb_id, artifact, markdown, "Flashcards made")

    @_friendly(2)
    def on_pick_artifact(nb_id, art_id):
        return _artifact_view(nb_id, art_id)

    @_friendly(4)
    def on_delete_artifact(nb_id, art_id):
        nb_id = _require(nb_id)
        if not art_id:
            raise gr.Error("Pick the file you want to delete.")
        title = store.get_artifact(nb_id, art_id).title
        store.delete_artifact(nb_id, art_id)
        gr.Info(f"Deleted “{title}”.")
        _arts, choices = _artifact_choices(nb_id)
        new_id = choices[0][1] if choices else None
        md, path = _artifact_view(nb_id, new_id)
        return gr.update(choices=choices, value=new_id), md, path, store.storage_tree(nb_id)

    @_friendly(1)
    def on_refresh_storage(nb_id):
        return store.storage_tree(nb_id) if nb_id else ""

    # ------------------------------------------------------------------ layout
    with gr.Blocks(title=f"{APP_NAME.lstrip('📓 ')}: research notebooks with cited answers") as demo:
        gr.HTML(f'<div class="app-hero"><div class="app-title">{APP_NAME}</div><div class="app-sub">{TAGLINE}</div>'
                '<div class="app-tags"><span>📄 PDF, PowerPoint, text and web pages</span>'
                '<span>🔎 Answers cite the exact page</span><span>🧾 Reports, quizzes and flashcards</span></div></div>')

        with gr.Row(equal_height=False):
            # ---------------- left column: notebooks + sources
            with gr.Column(scale=1, min_width=320):
                gr.HTML('<div class="side-title">📚 Notebooks</div>')
                nb_dd = gr.Dropdown(label="Open notebook", choices=[], value=None, interactive=True)
                with gr.Row(equal_height=True):
                    new_name = gr.Textbox(placeholder="Name a new notebook", show_label=False, scale=3,
                                          max_length=80, container=False)
                    create_btn = gr.Button("Create", variant="primary", scale=1, min_width=86)
                with gr.Accordion("Rename or delete this notebook", open=False):
                    with gr.Row(equal_height=True):
                        rename_box = gr.Textbox(placeholder="New name", show_label=False, scale=3, max_length=80,
                                                container=False)
                        rename_btn = gr.Button("Rename", scale=1, min_width=86)
                    delete_btn = gr.Button("Delete notebook", variant="stop", size="sm")
                    with gr.Group(visible=False) as confirm_box:
                        confirm_md = gr.Markdown()
                        with gr.Row():
                            confirm_yes = gr.Button("Delete for good", variant="stop", size="sm")
                            confirm_no = gr.Button("Keep it", size="sm")

                gr.HTML('<div class="side-title">📎 Sources</div>'
                        '<p class="side-note">PDF, PowerPoint (.pptx) or text files up to '
                        f'{settings.max_upload_mb} MB each, or any public web page.</p>')
                files_in = gr.File(label="Upload files", file_count="multiple",
                                   file_types=[".pdf", ".pptx", ".txt", ".md"], height=110)
                add_files_btn = gr.Button("Add files to notebook", variant="primary")
                url_box = gr.Textbox(label="Add a web page", placeholder="https://…", submit_btn="Add page")
                ingest_status = gr.Markdown(elem_classes="status")
                sources_cb = gr.CheckboxGroup(label="Use these sources",
                                              info="Untick a source to leave it out of answers and study material.",
                                              choices=[], value=[], interactive=True)
                with gr.Accordion("Remove a source", open=False):
                    remove_dd = gr.Dropdown(label="Source to remove", choices=[], value=None)
                    remove_btn = gr.Button("Remove source", variant="stop", size="sm")

            # ---------------- right column: notebook view
            with gr.Column(scale=3):
                summary_md = gr.HTML(_summary(None))
                with gr.Tabs():
                    with gr.Tab("💬 Chat"):
                        chatbot = gr.Chatbot(label="Chat", show_label=False, height=520, layout="panel",
                                             elem_id="chat", buttons=["copy"],
                                             placeholder="**Ask anything about this notebook's sources.**\n\n"
                                                         "Answers come only from your documents, and every claim "
                                                         "points to the page, slide or lines it came from.")
                        with gr.Row(equal_height=True):
                            question = gr.Textbox(placeholder="Ask about your sources, e.g. “What are the key "
                                                              "findings?”", show_label=False, scale=6, lines=1,
                                                  max_lines=5, container=False)
                            ask_btn = gr.Button("Ask", variant="primary", scale=1, min_width=96)
                        with gr.Row():
                            method_dd = gr.Dropdown(label="How to search your sources",
                                                    choices=[(label, key) for key, label in METHOD_LABELS.items()],
                                                    value=settings.default_method, scale=3, min_width=250)
                            topk = gr.Slider(label="Passages per answer", minimum=2, maximum=10, step=1,
                                             value=settings.top_k, scale=2, min_width=200)
                            clear_btn = gr.Button("Clear chat", size="sm", scale=1, min_width=96)
                        with gr.Accordion("How this answer was found", open=False):
                            retrieved_md = gr.Markdown("Ask a question to see which passages were used.",
                                                       elem_classes="trace")

                    with gr.Tab("🧾 Study material"):
                        gr.Markdown("Turn the sources you're using into something to study from. Everything you "
                                    "make is saved in this notebook and can be downloaded as a Markdown file.")
                        focus_box = gr.Textbox(label="Topic (optional)",
                                               info="Leave blank to cover all sources, or name a topic to focus on.",
                                               placeholder="e.g. battery safety rules")
                        with gr.Row(equal_height=False):
                            with gr.Column(min_width=220, elem_classes="tool-card"):
                                gr.HTML('<p class="tool-title">📄 Report</p><p class="tool-note">A structured '
                                        'summary: key themes, important facts and open questions, with citations.</p>')
                                report_btn = gr.Button("Write report", variant="primary")
                            with gr.Column(min_width=220, elem_classes="tool-card"):
                                gr.HTML('<p class="tool-title">❓ Quiz</p><p class="tool-note">Multiple-choice '
                                        'questions with an answer key and the source of each answer.</p>')
                                with gr.Row():
                                    quiz_n = gr.Slider(label="Questions", minimum=3, maximum=15, step=1, value=8)
                                    quiz_level = gr.Dropdown(label="Difficulty", choices=["easy", "medium", "hard"],
                                                             value="medium")
                                quiz_btn = gr.Button("Make quiz", variant="primary")
                            with gr.Column(min_width=220, elem_classes="tool-card"):
                                gr.HTML('<p class="tool-title">🃏 Flashcards</p><p class="tool-note">Short '
                                        'term-and-definition cards for quick review.</p>')
                                cards_n = gr.Slider(label="Cards", minimum=5, maximum=20, step=1, value=10)
                                cards_btn = gr.Button("Make flashcards", variant="primary")
                        with gr.Row(equal_height=False):
                            art_dd = gr.Dropdown(label="Saved in this notebook", choices=[], value=None, scale=5)
                            art_delete = gr.Button("Delete", variant="stop", size="sm", scale=1, min_width=86)
                        art_file = gr.File(label="Download as Markdown", interactive=False, height=60)
                        art_view = gr.Markdown(elem_id="doc", max_height=680, buttons=["copy"])

                    with gr.Tab("🗂️ Storage"):
                        gr.Markdown("This is the notebook's folder on disk. It holds the original files, the "
                                    "extracted text, the passages with their page and slide references, the search "
                                    "index, the chat history and the study material.")
                        storage_tree = gr.Code(label="Notebook folder", language=None, lines=22,
                                               elem_id="tree", interactive=False)
                        storage_refresh = gr.Button("Refresh", size="sm")

                    with gr.Tab("❔ How it works"):
                        gr.Markdown(HELP_MD)

        gr.HTML(_footer_html())

        # ------------------------------------------------------------------ wiring
        view_outputs = [nb_dd, sources_cb, remove_dd, chatbot, art_dd, art_view, art_file, storage_tree,
                        summary_md, retrieved_md]

        demo.load(on_load, outputs=view_outputs)
        nb_dd.input(on_select, inputs=nb_dd, outputs=view_outputs + [ingest_status])
        create_btn.click(on_create, inputs=new_name, outputs=view_outputs + [new_name])
        new_name.submit(on_create, inputs=new_name, outputs=view_outputs + [new_name])
        rename_btn.click(on_rename, inputs=[nb_dd, rename_box], outputs=view_outputs + [rename_box])
        rename_box.submit(on_rename, inputs=[nb_dd, rename_box], outputs=view_outputs + [rename_box])
        delete_btn.click(on_delete_ask, inputs=nb_dd, outputs=[confirm_box, confirm_md])
        confirm_no.click(on_delete_cancel, outputs=[confirm_box, confirm_md])
        confirm_yes.click(on_delete_confirm, inputs=nb_dd,
                          outputs=view_outputs + [confirm_box, confirm_md, ingest_status])

        add_files_btn.click(on_add_files, inputs=[nb_dd, files_in],
                            outputs=view_outputs + [files_in, ingest_status])
        url_box.submit(on_add_url, inputs=[nb_dd, url_box], outputs=view_outputs + [url_box, ingest_status])
        sources_cb.input(on_toggle_sources, inputs=[nb_dd, sources_cb], outputs=summary_md)
        remove_btn.click(on_remove_source, inputs=[nb_dd, remove_dd], outputs=view_outputs + [ingest_status])

        for trigger in (ask_btn.click, question.submit):
            trigger(on_user_message, inputs=[nb_dd, question, chatbot], outputs=[question, chatbot],
                    queue=False).then(on_answer, inputs=[nb_dd, chatbot, method_dd, topk],
                                      outputs=[chatbot, retrieved_md, storage_tree])
        clear_btn.click(on_clear_chat, inputs=nb_dd, outputs=[chatbot, retrieved_md, storage_tree])

        art_outputs = [art_dd, art_view, art_file, storage_tree, summary_md]
        report_btn.click(on_report, inputs=[nb_dd, focus_box], outputs=art_outputs)
        quiz_btn.click(on_quiz, inputs=[nb_dd, focus_box, quiz_n, quiz_level], outputs=art_outputs)
        cards_btn.click(on_flashcards, inputs=[nb_dd, focus_box, cards_n], outputs=art_outputs)
        art_dd.input(on_pick_artifact, inputs=[nb_dd, art_dd], outputs=[art_view, art_file])
        art_delete.click(on_delete_artifact, inputs=[nb_dd, art_dd], outputs=[art_dd, art_view, art_file, storage_tree])
        storage_refresh.click(on_refresh_storage, inputs=nb_dd, outputs=storage_tree)

    return demo
