"""Artifact generation: report, quiz with answer key, flashcards. Saved as Markdown."""

from __future__ import annotations

import random
import re
from datetime import datetime

from ..config import Settings
from ..llm import LLMClient, LLMError, normalize_citations, parse_json
from ..llm.prompts import (FLASHCARD_SYSTEM, QUIZ_SYSTEM, REPORT_SYSTEM, flashcards_prompt, format_context,
                           quiz_prompt, report_prompt)
from ..models import Artifact, Chunk
from ..retrieval import Retriever
from ..storage import NotebookStore

ARTIFACT_TYPES = {"report": "Report", "quiz": "Quiz + answer key", "flashcards": "Flashcards"}
LETTERS = "ABCD"


class ArtifactError(Exception):
    pass


class ArtifactService:
    def __init__(self, settings: Settings, store: NotebookStore, retriever: Retriever, llm: LLMClient):
        self.settings = settings
        self.store = store
        self.retriever = retriever
        self.llm = llm

    # -- context selection ------------------------------------------------------------
    def _context(self, nb_id: str, focus: str | None) -> list[Chunk]:
        enabled = self.retriever.enabled_source_ids(nb_id)
        if not self.store.list_sources(nb_id):
            raise ArtifactError("Add at least one source before generating artifacts.")
        if not enabled:
            raise ArtifactError("All sources are switched off. Tick at least one source first.")
        if focus:
            result = self.retriever.retrieve(nb_id, focus, k=14, source_ids=enabled)
            chunks = [rc.chunk for rc in result.chunks]
            if chunks:
                return chunks
        return self.retriever.coverage_sample(nb_id, enabled, self.settings.artifact_context_chars)

    @staticmethod
    def _header(kind: str, notebook_name: str, chunks: list[Chunk], model: str, focus: str | None) -> str:
        n_sources = len({c.source_id for c in chunks})
        now = datetime.now()
        when = f"{now:%B} {now.day}, {now.year} at {now:%H:%M}"
        sources = f"{n_sources} source" + ("" if n_sources == 1 else "s")
        focus_txt = f" Focus: {focus}." if focus else ""
        return (f"# {kind}: {notebook_name}\n\n"
                f"_Generated on {when} from {sources} ({len(chunks)} excerpts) with {model}.{focus_txt}_\n")

    @staticmethod
    def _sources_section(chunks: list[Chunk], cited: set[int] | None = None) -> str:
        lines = []
        for i, c in enumerate(chunks, start=1):
            if cited and i not in cited:
                continue
            name = f"[{c.source_name}]({c.url})" if c.url else c.source_name
            lines.append(f"- **[{i}]** {name} · {c.locator}")
        return "## Sources\n\n" + "\n".join(lines) + "\n"

    def _json_call(self, system: str, prompt: str) -> tuple[dict, str]:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        last_error = None
        for attempt in range(2):
            response = self.llm.complete(messages, max_tokens=self.settings.artifact_max_tokens, json_mode=True,
                                         temperature=0.3)
            try:
                return parse_json(response.text), response.model
            except (ValueError, TypeError) as exc:
                last_error = exc
                messages = messages + [
                    {"role": "assistant", "content": response.text[:2000]},
                    {"role": "user", "content": "That was not valid JSON. Reply with ONLY the JSON object."},
                ]
        raise LLMError(f"The model did not return valid JSON ({last_error}). Please try again.")

    # -- report -------------------------------------------------------------------------
    def generate_report(self, nb_id: str, focus: str | None = None) -> tuple[Artifact, str]:
        notebook = self.store.get_notebook(nb_id)
        focus = (focus or "").strip() or None
        chunks = self._context(nb_id, focus)
        context = format_context(chunks, max_chars_per_chunk=1400)
        messages = [{"role": "system", "content": REPORT_SYSTEM},
                    {"role": "user", "content": report_prompt(notebook.name, context, focus)}]
        response = self.llm.complete(messages, max_tokens=self.settings.artifact_max_tokens, temperature=0.3)
        body = re.sub(r"^#\s+.*\n", "", normalize_citations(response.text).strip(), count=1)  # drop model title
        valid = set(range(1, len(chunks) + 1))
        body = re.sub(r"\[(\d{1,2})\]", lambda m: m.group(0) if int(m.group(1)) in valid else "", body)
        cited = {int(n) for n in re.findall(r"\[(\d{1,2})\]", body)}
        markdown = (self._header("Report", notebook.name, chunks, response.model, focus) + "\n" + body.strip()
                    + "\n\n---\n\n" + self._sources_section(chunks, cited or None))
        title = f"Report{' - ' + focus if focus else ''}"
        artifact = self.store.save_artifact(nb_id, "report", title, markdown, model=response.model,
                                            source_ids=sorted({c.source_id for c in chunks}),
                                            meta={"focus": focus, "excerpts": len(chunks)})
        return artifact, markdown

    # -- quiz -----------------------------------------------------------------------------
    @staticmethod
    def _normalize_question(q: dict) -> dict | None:
        if not isinstance(q, dict):
            return None
        question = str(q.get("question", "")).strip()
        options = [str(o).strip() for o in (q.get("options") or []) if str(o).strip()]
        options = [re.sub(r"^[A-Da-d][\).:]\s+", "", o) for o in options][:4]
        if not question or len(options) < 2:
            return None
        answer = str(q.get("answer", "")).strip()
        letter = None
        if len(answer) >= 1 and answer[0].upper() in LETTERS[: len(options)] and (len(answer) == 1 or not answer[1].isalpha()):
            letter = answer[0].upper()
        else:
            for i, opt in enumerate(options):
                if answer and answer.lower() == opt.lower():
                    letter = LETTERS[i]
        if letter is None:
            return None
        try:
            source = int(q.get("source")) if q.get("source") is not None else None
        except (TypeError, ValueError):
            source = None
        return {"question": question, "options": options, "answer": letter,
                "explanation": str(q.get("explanation", "")).strip(), "source": source}

    @staticmethod
    def _shuffle_options(questions: list[dict], seed: int | None = None) -> list[dict]:
        """LLMs put the right answer in the same slot most of the time (usually B), so the
        options are shuffled here and the answer letter is remapped."""
        rng = random.Random(seed)
        for q in questions:
            correct = q["options"][LETTERS.index(q["answer"])]
            for _ in range(5):
                rng.shuffle(q["options"])
                if len(q["options"]) < 3 or q["options"].index(correct) != LETTERS.index(q["answer"]):
                    break
            q["answer"] = LETTERS[q["options"].index(correct)]
        return questions

    def generate_quiz(self, nb_id: str, n_questions: int = 8, difficulty: str = "medium",
                      focus: str | None = None) -> tuple[Artifact, str]:
        notebook = self.store.get_notebook(nb_id)
        n_questions = max(3, min(int(n_questions), 20))
        focus = (focus or "").strip() or None
        chunks = self._context(nb_id, focus)
        context = format_context(chunks, max_chars_per_chunk=1400)
        data, model = self._json_call(QUIZ_SYSTEM, quiz_prompt(notebook.name, context, n_questions, difficulty, focus))
        questions = [q for q in (self._normalize_question(q) for q in data.get("questions", [])) if q]
        if len(questions) < min(3, n_questions):
            raise LLMError("The model returned too few usable quiz questions. Please try again.")
        questions = self._shuffle_options(questions)

        lines = [self._header("Quiz", notebook.name, chunks, model, focus),
                 f"\n**{len(questions)} questions, {difficulty} difficulty.** The answer key is at the end.\n",
                 "## Questions\n"]
        for i, q in enumerate(questions, start=1):
            lines.append(f"**{i}. {q['question']}**\n")
            lines.extend(f"- {LETTERS[j]}) {opt}" for j, opt in enumerate(q["options"]))
            lines.append("")
        lines.append("---\n\n## Answer key\n")
        for i, q in enumerate(questions, start=1):
            idx = LETTERS.index(q["answer"])
            ref = ""
            if q["source"] and 1 <= q["source"] <= len(chunks):
                c = chunks[q["source"] - 1]
                ref = f" *(Source: {c.source_name} · {c.locator})*"
            lines.append(f"{i}. **{q['answer']}) {q['options'][idx]}** — {q['explanation']}{ref}")
        markdown = "\n".join(lines) + "\n"
        title = f"Quiz ({len(questions)} questions{', ' + focus if focus else ''})"
        artifact = self.store.save_artifact(nb_id, "quiz", title, markdown, model=model,
                                            source_ids=sorted({c.source_id for c in chunks}),
                                            meta={"questions": len(questions), "difficulty": difficulty,
                                                  "focus": focus, "answer_key": [q["answer"] for q in questions]})
        return artifact, markdown

    # -- flashcards -----------------------------------------------------------------------
    def generate_flashcards(self, nb_id: str, n_cards: int = 10, focus: str | None = None) -> tuple[Artifact, str]:
        notebook = self.store.get_notebook(nb_id)
        n_cards = max(3, min(int(n_cards), 30))
        focus = (focus or "").strip() or None
        chunks = self._context(nb_id, focus)
        context = format_context(chunks, max_chars_per_chunk=1400)
        data, model = self._json_call(FLASHCARD_SYSTEM, flashcards_prompt(notebook.name, context, n_cards, focus))
        cards = [c for c in data.get("cards", []) if isinstance(c, dict) and c.get("front") and c.get("back")]
        if not cards:
            raise LLMError("The model returned no usable flashcards. Please try again.")

        def cell(text: str) -> str:
            return re.sub(r"\s+", " ", str(text)).replace("|", "\\|").strip()

        lines = [self._header("Flashcards", notebook.name, chunks, model, focus), "",
                 "| # | Front | Back | Source |", "|---|---|---|---|"]
        for i, card in enumerate(cards, start=1):
            ref = ""
            try:
                n = int(card.get("source"))
                if 1 <= n <= len(chunks):
                    ref = f"{chunks[n - 1].source_name} · {chunks[n - 1].locator}"
            except (TypeError, ValueError):
                pass
            back = re.sub(r"\[\d{1,2}\]", "", normalize_citations(str(card["back"])))
            lines.append(f"| {i} | {cell(card['front'])} | {cell(back)} | {cell(ref)} |")
        markdown = "\n".join(lines) + "\n"
        artifact = self.store.save_artifact(nb_id, "flashcards", f"Flashcards ({len(cards)} cards)", markdown,
                                            model=model, source_ids=sorted({c.source_id for c in chunks}),
                                            meta={"cards": len(cards), "focus": focus})
        return artifact, markdown
