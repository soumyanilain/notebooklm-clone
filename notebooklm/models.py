"""Plain data records shared by every layer. They serialize to JSON with to_dict()."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


class _Record:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        # Ignore unknown keys so older/newer files still load.
        allowed = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass
class Notebook(_Record):
    id: str
    name: str
    created_at: str
    updated_at: str


@dataclass
class Source(_Record):
    id: str
    notebook_id: str
    name: str  # display name: file name or page title
    type: str  # pdf | pptx | txt | url
    origin: str  # original file name or URL
    raw_key: str | None = None  # storage key of the original file / fetched HTML
    text_key: str | None = None  # storage key of the extracted text (JSON)
    sha256: str | None = None
    n_segments: int = 0  # pages / slides / sections
    n_chunks: int = 0
    n_chars: int = 0
    enabled: bool = True
    added_at: str = field(default_factory=now_iso)


@dataclass
class Segment(_Record):
    """A citable unit of extracted text: a PDF page, a slide, a web section..."""

    text: str
    label: str  # human readable locator, e.g. "p. 3", "slide 5", "§ Overview"
    page: int | None = None  # page or slide number when it exists


@dataclass
class Chunk(_Record):
    id: str
    notebook_id: str
    source_id: str
    source_name: str
    source_type: str
    text: str
    chunk_index: int
    locator: str  # e.g. "p. 3", "slide 5", "lines 10-24"
    page: int | None = None
    url: str | None = None

    def citation_label(self) -> str:
        return f"{self.source_name} · {self.locator}" if self.locator else self.source_name


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int
    scores: dict[str, float] = field(default_factory=dict)  # vector / bm25 / rrf / rerank

    def to_dict(self) -> dict[str, Any]:
        return {"chunk": self.chunk.to_dict(), "score": self.score, "rank": self.rank, "scores": self.scores}


@dataclass
class ChatMessage(_Record):
    id: str
    role: str  # user | assistant
    content: str
    created_at: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Artifact(_Record):
    id: str
    notebook_id: str
    type: str  # report | quiz | flashcards
    title: str
    filename: str
    created_at: str
    model: str | None = None
    source_ids: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
