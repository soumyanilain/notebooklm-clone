"""Chunking: split each citable segment into overlapping windows.

Chunks never cross a segment boundary, so every chunk keeps exactly one locator
(page, slide, line range or web section) and citations stay precise.

Within a segment we use a sliding window that snaps its end to the most natural
break available (paragraph > line > sentence > word) and starts the next window
``overlap`` characters earlier, aligned to a word boundary.
"""

from __future__ import annotations

import re

from ..models import Chunk, Segment, Source

_BREAKS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]


def split_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[tuple[int, int]]:
    """Return (start, end) character spans covering ``text``."""
    text_len = len(text)
    if text_len == 0:
        return []
    if text_len <= chunk_size:
        return [(0, text_len)]
    overlap = max(0, min(overlap, chunk_size // 2))
    spans: list[tuple[int, int]] = []
    start = 0
    while start < text_len:
        hard_end = min(start + chunk_size, text_len)
        end = hard_end
        if hard_end < text_len:
            window_floor = start + int(chunk_size * 0.55)
            for sep in _BREAKS:
                pos = text.rfind(sep, window_floor, hard_end)
                if pos != -1:
                    end = pos + len(sep)
                    break
        spans.append((start, end))
        if end >= text_len:
            break
        next_start = max(end - overlap, start + 1)
        # move forward to the start of a word so chunks don't begin mid-word
        m = re.search(r"\s", text[next_start:end])
        if m and overlap:
            next_start = next_start + m.end()
        start = next_start
        # avoid a tiny trailing chunk that is pure overlap
        if text_len - start <= overlap:
            last_start, _ = spans[-1]
            spans[-1] = (last_start, text_len)
            break
    return spans


def chunk_segments(source: Source, segments: list[Segment], chunk_size: int, overlap: int,
                   url: str | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    for segment in segments:
        for start, end in split_text(segment.text, chunk_size, overlap):
            piece = segment.text[start:end].strip()
            if len(piece) < 20 and chunks:
                continue  # drop fragments too small to be useful
            chunks.append(
                Chunk(
                    id=f"{source.id}-{len(chunks):04d}",
                    notebook_id=source.notebook_id,
                    source_id=source.id,
                    source_name=source.name,
                    source_type=source.type,
                    text=piece,
                    chunk_index=len(chunks),
                    locator=segment.label,
                    page=segment.page,
                    url=url,
                )
            )
    return chunks
