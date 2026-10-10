"""LLM access through Groq, with model fallback and readable errors.

Groq's free tier has per-model rate limits and occasionally retires models, so the
client walks an ordered list of models: if the first is rate-limited or gone, it
tries the next one instead of failing the user's request.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..config import Settings

log = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised with a message that is safe to show to the user."""


@dataclass
class LLMResponse:
    text: str
    model: str
    latency_s: float
    usage: dict = field(default_factory=dict)


THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


def strip_reasoning(text: str) -> str:
    text = THINK_RE.sub("", text or "")
    if "<think>" in text.lower():  # unterminated block: drop everything before the answer
        text = text.split("</think>")[-1] if "</think>" in text else ""
    return text.strip()


# Models write citations in several styles: gpt-oss uses its own 【1†L9-L13】 markers,
# others write [1, 3], [1-2], [Source 2] or [Excerpt 2]. Normalize all of them to [1][3].
_CITE_FULLWIDTH = re.compile(r"【\s*(\d{1,2})(?:\s*†[^】]*)?\s*】")
_CITE_DAGGER = re.compile(r"\[(\d{1,2})\s*†[^\]]*\]")
_CITE_LABELLED = re.compile(r"\[(?:source|excerpt|doc(?:ument)?|ref)\s*#?\s*(\d{1,2})\]", re.I)
_CITE_LIST = re.compile(r"\[(\d{1,2}(?:\s*[,;]\s*\d{1,2})+)\]")
_CITE_RANGE = re.compile(r"\[(\d{1,2})\s*[-–]\s*(\d{1,2})\]")


def normalize_citations(text: str) -> str:
    text = _CITE_FULLWIDTH.sub(r"[\1]", text or "")
    text = _CITE_DAGGER.sub(r"[\1]", text)
    text = _CITE_LABELLED.sub(r"[\1]", text)
    text = _CITE_LIST.sub(lambda m: "".join(f"[{n.strip()}]" for n in re.split(r"[,;]", m.group(1))), text)

    def expand(m: re.Match) -> str:
        a, b = int(m.group(1)), int(m.group(2))
        return "".join(f"[{n}]" for n in range(a, b + 1)) if 0 < a < b <= a + 10 else m.group(0)

    text = _CITE_RANGE.sub(expand, text)
    text = re.sub(r"(?<=[\w)])(\[\d{1,2}\])", r" \1", text)  # "cause[1]" -> "cause [1]"
    return re.sub(r"[ \t]+(\[\d{1,2}\])", r" \1", text)


def parse_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating code fences or extra prose."""
    cleaned = strip_reasoning(text)
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned.strip(), flags=re.I)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            return json.loads(cleaned[start: end + 1])
        raise


class LLMClient(ABC):
    @abstractmethod
    def complete(self, messages: list[dict], *, max_tokens: int, temperature: float | None = None,
                 json_mode: bool = False, models: list[str] | None = None) -> LLMResponse: ...


class GroqLLM(LLMClient):
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None
        self._lock = threading.Lock()

    def _get_client(self):
        if not self.settings.groq_api_key:
            raise LLMError(
                "GROQ_API_KEY is not set. Add it to your .env file locally, or to Secret Manager for the "
                "Cloud Run deployment (see scripts/gcp_setup.sh), then restart the app."
            )
        with self._lock:
            if self._client is None:
                from groq import Groq

                self._client = Groq(api_key=self.settings.groq_api_key, max_retries=2, timeout=90.0)
        return self._client

    def complete(self, messages: list[dict], *, max_tokens: int, temperature: float | None = None,
                 json_mode: bool = False, models: list[str] | None = None) -> LLMResponse:
        import groq

        client = self._get_client()
        models = models or self.settings.chat_models
        temperature = self.settings.llm_temperature if temperature is None else temperature
        errors: list[str] = []

        for model in models:
            kwargs: dict = {"model": model, "messages": messages, "temperature": temperature,
                            "max_completion_tokens": max_tokens}
            extra_body: dict = {}
            if model.startswith("openai/gpt-oss"):
                extra_body["reasoning_effort"] = "low"
                extra_body["include_reasoning"] = False
            elif "qwen3" in model:
                extra_body["reasoning_format"] = "hidden"
            if extra_body:
                kwargs["extra_body"] = extra_body
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            started = time.perf_counter()
            resp = None
            for _attempt in range(3):
                try:
                    resp = client.chat.completions.create(**kwargs)
                    break
                except groq.AuthenticationError as exc:
                    raise LLMError("Groq rejected the API key (401). Check GROQ_API_KEY.") from exc
                except groq.BadRequestError as exc:
                    msg = str(exc).lower()
                    # Drop optional parameters this model rejects, then retry the same model.
                    if "response_format" in kwargs and ("json" in msg or "response_format" in msg):
                        kwargs.pop("response_format")
                        continue
                    if "extra_body" in kwargs and ("reasoning" in msg or "property" in msg):
                        kwargs.pop("extra_body")
                        continue
                    if "decommissioned" in msg or "does not exist" in msg or "model_not_found" in msg:
                        errors.append(f"{model}: model retired or unavailable")
                    else:
                        errors.append(f"{model}: {self._short(str(exc))}")
                    log.warning("Bad request on %s: %s", model, exc)
                    break
                except groq.RateLimitError:
                    errors.append(f"{model}: rate limited")
                    log.warning("Rate limited on %s, trying next model", model)
                    break
                except groq.NotFoundError:
                    errors.append(f"{model}: model not available")
                    break
                except groq.APITimeoutError:
                    errors.append(f"{model}: timed out")
                    break
                except groq.APIConnectionError as exc:
                    raise LLMError("Could not connect to Groq. Check your internet connection.") from exc
                except groq.APIError as exc:
                    errors.append(f"{model}: {self._short(str(exc))}")
                    break
            if resp is None:
                continue

            text = strip_reasoning(resp.choices[0].message.content or "")
            if not text:
                errors.append(f"{model}: empty response")
                continue
            usage = {}
            if getattr(resp, "usage", None):
                usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}
            return LLMResponse(text=text, model=model, latency_s=time.perf_counter() - started, usage=usage)

        detail = "; ".join(errors) or "no models configured"
        if errors and all("rate limited" in e for e in errors):
            raise LLMError("Groq's free-tier rate limit was hit on every model. Wait a minute and try again.")
        raise LLMError(f"The language model request failed ({detail}).")

    @staticmethod
    def _short(msg: str, limit: int = 160) -> str:
        msg = re.sub(r"\s+", " ", msg)
        return msg if len(msg) <= limit else msg[:limit] + "…"


class FakeLLM(LLMClient):
    """Deterministic offline model for tests and CI (LLM_BACKEND=fake)."""

    def complete(self, messages: list[dict], *, max_tokens: int, temperature: float | None = None,
                 json_mode: bool = False, models: list[str] | None = None) -> LLMResponse:
        prompt = messages[-1]["content"]
        system = messages[0]["content"].lower() if messages and messages[0]["role"] == "system" else ""
        if json_mode and "flashcard" in prompt.lower():
            body = {"cards": [{"front": f"Term {i}", "back": f"Definition {i} [1]", "source": 1} for i in range(1, 6)]}
            text = json.dumps(body)
        elif json_mode:
            questions = [
                {"question": f"Sample question {i}?", "options": ["Alpha", "Beta", "Gamma", "Delta"],
                 "answer": "B", "explanation": "Because the source says so.", "source": 1}
                for i in range(1, 6)
            ]
            text = json.dumps({"title": "Sample quiz", "questions": questions})
        elif "study reports" in system:
            text = "## Executive summary\nThe sources describe the topic [1].\n\n## Key themes\n### Theme one\nDetail [1]."
        elif "no sources were retrieved" in prompt.lower():
            text = "I couldn't find that in this notebook's sources."
        else:
            text = "According to the notebook's sources, here is the answer [1]."
        return LLMResponse(text=text, model="fake-llm", latency_s=0.01)


_llm: LLMClient | None = None
_llm_lock = threading.Lock()


def get_llm(settings: Settings) -> LLMClient:
    global _llm
    with _llm_lock:
        if _llm is None:
            _llm = FakeLLM() if settings.llm_backend == "fake" else GroqLLM(settings)
        return _llm
