"""RAG chat: question -> retrieve -> grounded answer with citations -> saved history."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from ..config import Settings
from ..llm import LLMClient, normalize_citations
from ..llm.prompts import ANSWER_SYSTEM, answer_user_prompt, format_context
from ..models import ChatMessage, RetrievedChunk, new_id, now_iso
from ..retrieval import RetrievalResult, Retriever
from ..storage import NotebookStore

CITE_RE = re.compile(r"\[(\d{1,2})\]")


@dataclass
class ChatTurn:
    question: str
    answer: str
    citations: list[dict]
    retrieval: RetrievalResult
    model: str | None
    llm_seconds: float
    total_seconds: float
    meta: dict = field(default_factory=dict)


def build_citations(answer: str, retrieved: list[RetrievedChunk]) -> tuple[str, list[dict]]:
    """Keep only citation markers that point at a real excerpt; return the cited excerpts."""
    valid = {str(i) for i in range(1, len(retrieved) + 1)}
    answer = normalize_citations(answer)
    answer = CITE_RE.sub(lambda m: m.group(0) if m.group(1) in valid else "", answer)
    used = sorted({int(n) for n in CITE_RE.findall(answer)})
    citations = []
    for n in used:
        rc = retrieved[n - 1]
        snippet = re.sub(r"\s+", " ", rc.chunk.text).strip()
        citations.append({
            "n": n, "source_id": rc.chunk.source_id, "source_name": rc.chunk.source_name,
            "source_type": rc.chunk.source_type, "locator": rc.chunk.locator, "chunk_id": rc.chunk.id,
            "url": rc.chunk.url, "score": rc.score,
            "snippet": snippet[:280] + ("…" if len(snippet) > 280 else ""),
        })
    return answer, citations


def pretty_locator(locator: str) -> str:
    """'p. 6' -> 'page 6', 'lines 1-38' -> 'lines 1–38', '§ Intro' -> 'section “Intro”'."""
    if locator.startswith("p. "):
        return "page " + locator[3:]
    if locator.startswith("§ "):
        return f"section “{locator[2:]}”"
    if locator.startswith("lines "):
        return locator.replace("-", "–")
    return locator


def mark_citations(text: str) -> str:
    """Show [n] markers as highlighted chips in the UI (the saved text keeps plain [n])."""
    return CITE_RE.sub(lambda m: f"<mark>{m.group(1)}</mark>", text)


def citation_line(c: dict) -> str:
    name = f"[{c['source_name']}]({c['url']})" if c.get("url") else f"**{c['source_name']}**"
    snippet = c["snippet"] if len(c["snippet"]) <= 200 else c["snippet"][:200].rsplit(" ", 1)[0] + "…"
    return f"<mark>{c['n']}</mark> {name}, {pretty_locator(c['locator'])}: “{snippet}”"


def render_assistant(content: str, citations: list[dict]) -> str:
    """Markdown shown in the chat window: the answer with highlighted citation markers,
    followed by the list of sources it cited."""
    body = mark_citations(content)
    if not citations:
        return body
    lines = "\n".join(f"- {citation_line(c)}" for c in citations)
    label = "Source" if len(citations) == 1 else f"Sources ({len(citations)})"
    return f"{body}\n\n<details open><summary>{label}</summary>\n\n{lines}\n\n</details>"


class ChatService:
    def __init__(self, settings: Settings, store: NotebookStore, retriever: Retriever, llm: LLMClient):
        self.settings = settings
        self.store = store
        self.retriever = retriever
        self.llm = llm

    def _history_text(self, history: list[ChatMessage]) -> str:
        turns = history[-2 * self.settings.history_turns:] if self.settings.history_turns else []
        lines = []
        for msg in turns:
            text = CITE_RE.sub("", msg.content)
            text = re.sub(r"\s+", " ", text).strip()
            lines.append(f"{msg.role.upper()}: {text[:600]}")
        return "\n".join(lines)

    @staticmethod
    def _retrieval_query(question: str, history: list[ChatMessage]) -> str:
        """Short follow-ups ("what about the second one?") retrieve poorly on their own,
        so borrow the previous question for context."""
        words = question.split()
        followup = len(words) <= 6 or re.search(r"\b(it|its|that|this|they|them|those|these|he|she)\b", question, re.I)
        if followup:
            previous = [m.content for m in history if m.role == "user"]
            if previous:
                return f"{previous[-1]} {question}"
        return question

    def ask(self, nb_id: str, question: str, method: str | None = None, k: int | None = None,
            save: bool = True) -> ChatTurn:
        started = time.perf_counter()
        question = (question or "").strip()
        if not question:
            raise ValueError("Type a question first.")
        notebook = self.store.get_notebook(nb_id)
        history = self.store.load_history(nb_id)

        enabled = self.retriever.enabled_source_ids(nb_id)
        if not self.store.list_sources(nb_id):
            answer, citations, model, llm_s = (
                "This notebook has no sources yet. Upload a PDF, PPTX or TXT file, or add a URL, then ask again.",
                [], None, 0.0)
            retrieval = RetrievalResult(method or self.settings.default_method, question, [], {"total": 0.0})
        elif not enabled:
            answer, citations, model, llm_s = (
                "All sources in this notebook are switched off. Tick at least one source in the Sources panel.",
                [], None, 0.0)
            retrieval = RetrievalResult(method or self.settings.default_method, question, [], {"total": 0.0})
        else:
            retrieval = self.retriever.retrieve(nb_id, self._retrieval_query(question, history), method, k, enabled)
            context = format_context(retrieval.chunks)
            messages = [
                {"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": answer_user_prompt(question, context, self._history_text(history))},
            ]
            response = self.llm.complete(messages, max_tokens=self.settings.chat_max_tokens)
            answer, citations = build_citations(response.text, retrieval.chunks)
            model, llm_s = response.model, response.latency_s

        total_s = time.perf_counter() - started
        meta = {
            "notebook": notebook.name, "method": retrieval.method, "model": model,
            "retrieval_ms": retrieval.timings_ms, "llm_s": round(llm_s, 2), "total_s": round(total_s, 2),
            "retrieved": [{"chunk_id": rc.chunk.id, "score": rc.score, "source": rc.chunk.source_name,
                           "locator": rc.chunk.locator} for rc in retrieval.chunks],
        }
        if save:
            self.store.append_message(nb_id, ChatMessage(id=new_id("m_"), role="user", content=question,
                                                         created_at=now_iso()))
            self.store.append_message(nb_id, ChatMessage(id=new_id("m_"), role="assistant", content=answer,
                                                         created_at=now_iso(), citations=citations, meta=meta))
            self.store.touch(nb_id)
        return ChatTurn(question, answer, citations, retrieval, model, llm_s, total_s, meta)

    def history_for_display(self, nb_id: str) -> list[dict]:
        out = []
        for msg in self.store.load_history(nb_id):
            content = (render_assistant(normalize_citations(msg.content), msg.citations)
                       if msg.role == "assistant" else msg.content)
            out.append({"role": msg.role, "content": content})
        return out
