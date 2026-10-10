"""Text extraction for each supported source type.

Every extractor returns ``(title, segments)`` where each Segment is a citable unit
(PDF page, slide, text block, web-page section) carrying its own locator label, so
citations can point at "p. 4" or "slide 7" instead of just a file name.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

from ..models import Segment

SUPPORTED_EXTENSIONS = {".pdf": "pdf", ".pptx": "pptx", ".txt": "txt", ".md": "txt"}
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36 NotebookLMClone/1.0"
)


class ExtractionError(Exception):
    """Raised with a message that is safe to show to the user."""


def detect_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted({e for e in SUPPORTED_EXTENSIONS}))
        raise ExtractionError(f"'{Path(filename).name}' is not a supported file type. Supported: {allowed}")
    return SUPPORTED_EXTENSIONS[ext]


def clean_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words hyphenated across lines
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- PDF
def extract_pdf(data: bytes, filename: str) -> tuple[str, list[Segment]]:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:  # noqa: BLE001
                raise ExtractionError(f"'{filename}' is password-protected.") from exc
        pages = list(reader.pages)
    except ExtractionError:
        raise
    except (PdfReadError, ValueError, KeyError, OSError) as exc:
        raise ExtractionError(f"'{filename}' could not be read as a PDF ({exc.__class__.__name__}).") from exc

    segments: list[Segment] = []
    for number, page in enumerate(pages, start=1):
        try:
            text = clean_text(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - one bad page should not sink the file
            continue
        if text:
            segments.append(Segment(text=text, label=f"p. {number}", page=number))
    if not segments:
        raise ExtractionError(
            f"No text could be extracted from '{filename}'. It may be a scanned PDF (images only); "
            "try a text-based PDF or export the text to a .txt file."
        )
    title = filename
    try:
        meta_title = (reader.metadata.title or "").strip() if reader.metadata else ""
        if 3 <= len(meta_title) <= 120:
            title = meta_title
    except Exception:  # noqa: BLE001
        pass
    return title, segments


# -------------------------------------------------------------------------- PPTX
def _shape_texts(shape) -> list[str]:
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    texts: list[str] = []
    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        for child in shape.shapes:
            texts.extend(_shape_texts(child))
        return texts
    if getattr(shape, "has_table", False) and shape.has_table:
        for row in shape.table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                texts.append(" | ".join(cells))
        return texts
    if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
        for para in shape.text_frame.paragraphs:
            line = (para.text or "").replace("\v", " ").strip()
            if line:
                texts.append(("  " * para.level + "- " + line) if para.level else line)
    return texts


def extract_pptx(data: bytes, filename: str) -> tuple[str, list[Segment]]:
    from pptx import Presentation

    try:
        prs = Presentation(BytesIO(data))
    except Exception as exc:  # noqa: BLE001 - python-pptx raises many types
        raise ExtractionError(f"'{filename}' could not be opened as a PowerPoint file.") from exc

    segments: list[Segment] = []
    deck_title = ""
    for number, slide in enumerate(prs.slides, start=1):
        title = ""
        if slide.shapes.title is not None and slide.shapes.title.has_text_frame:
            title = slide.shapes.title.text_frame.text.strip()
        body: list[str] = []
        for shape in slide.shapes:
            if slide.shapes.title is not None and shape.shape_id == slide.shapes.title.shape_id:
                continue
            body.extend(_shape_texts(shape))
        notes = ""
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            notes = slide.notes_slide.notes_text_frame.text.strip()
        parts = [f"Slide {number}: {title}" if title else f"Slide {number}"]
        if body:
            parts.append("\n".join(body))
        if notes:
            parts.append(f"Speaker notes: {notes}")
        text = clean_text("\n".join(parts))
        if body or notes or title:
            segments.append(Segment(text=text, label=f"slide {number}", page=number))
        if not deck_title and title:
            deck_title = title
    if not segments:
        raise ExtractionError(f"'{filename}' has no text on its slides.")
    return filename, segments


# --------------------------------------------------------------------------- TXT
def decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            text = data.decode(encoding)
            if encoding == "utf-16" and not data[:2] in (b"\xff\xfe", b"\xfe\xff"):
                continue  # only trust utf-16 when there is a BOM
            return text
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def extract_txt(data: bytes, filename: str, block_lines: int = 40) -> tuple[str, list[Segment]]:
    """Plain text has no pages, so cite line ranges. Lines are grouped into blocks
    at blank lines close to ``block_lines`` so each block is a readable unit."""
    text = decode_text(data).replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = text.split("\n")
    if not text.strip():
        raise ExtractionError(f"'{filename}' is empty.")

    segments: list[Segment] = []
    start = 0
    while start < len(lines):
        end = min(start + block_lines, len(lines))
        if end < len(lines):
            # prefer to end the block on a blank line within the last third
            for j in range(end, start + block_lines // 3, -1):
                if not lines[j - 1].strip():
                    end = j
                    break
        block = clean_text("\n".join(lines[start:end]))
        if block:
            first = start + 1 + next((i for i, ln in enumerate(lines[start:end]) if ln.strip()), 0)
            last = start + max((i for i, ln in enumerate(lines[start:end]) if ln.strip()), default=0) + 1
            segments.append(Segment(text=block, label=f"lines {first}-{last}", page=None))
        start = end
    return filename, segments


# --------------------------------------------------------------------------- URL
@dataclass
class FetchedPage:
    url: str
    title: str
    segments: list[Segment]
    raw: bytes
    raw_ext: str
    kind: str  # html | pdf | txt


def _check_public_host(url: str) -> None:
    """Refuse URLs that point at the server itself or a private network (SSRF)."""
    host = urlparse(url).hostname or ""
    if host in {"localhost"} or host.endswith(".local") or host.endswith(".internal"):
        raise ExtractionError("URLs pointing at local or internal hosts are not allowed.")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ExtractionError(f"Could not resolve '{host}'. Check the URL.") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ExtractionError("URLs pointing at local or private network addresses are not allowed.")


def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        raise ExtractionError("Please paste a URL first.")
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or "." not in parsed.netloc:
        raise ExtractionError(f"'{url}' doesn't look like a valid web address.")
    return url


BLOCK_TAGS = ["h1", "h2", "h3", "h4", "p", "li", "pre", "blockquote", "td", "th", "dd", "dt", "figcaption"]
NOISE_TAGS = ["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg", "iframe",
              "button", "select", "input", "template", "canvas"]
NOISE_SELECTORS = [".mw-editsection", "sup.reference", ".reference", ".navbox", ".sidebar", ".infobox",
                   ".toc", "#toc", ".cookie", "[role=navigation]", "[aria-hidden=true]", ".advert", ".ads"]


def html_to_segments(html: str, max_section_chars: int = 4000) -> tuple[str, list[Segment]]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    og = soup.find("meta", attrs={"property": "og:title"})
    if not title and og and og.get("content"):
        title = og["content"].strip()

    for tag in soup(NOISE_TAGS):
        tag.decompose()
    for selector in NOISE_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()

    main = soup.find("article") or soup.find("main") or soup.find(attrs={"role": "main"}) or soup.body or soup
    # Pick top-level text blocks in document order (skip blocks nested in other blocks).
    blocks: list[tuple[str, str]] = []
    for el in main.find_all(BLOCK_TAGS):
        if el.find_parent(BLOCK_TAGS) is not None:
            continue
        text = re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()
        if not text:
            continue
        if el.name in {"li", "dd", "dt"}:
            text = "- " + text
        blocks.append((el.name, text))

    body_chars = sum(len(t) for _, t in blocks)
    if body_chars < 200:
        # Unusual markup: fall back to all visible text.
        text = clean_text(main.get_text("\n"))
        blocks = [("p", line) for line in text.split("\n") if line.strip()]

    segments: list[Segment] = []
    heading = ""
    buffer: list[str] = []

    def flush() -> None:
        text = clean_text("\n".join(buffer))
        if len(text) >= 40:
            label = f"§ {heading[:60]}" if heading else f"part {len(segments) + 1}"
            segments.append(Segment(text=(f"{heading}\n{text}" if heading else text), label=label))
        buffer.clear()

    for name, text in blocks:
        if name in {"h1", "h2", "h3"}:
            flush()
            heading = text
            continue
        buffer.append(text)
        if sum(len(b) for b in buffer) > max_section_chars:
            flush()
    flush()

    if not segments:
        raise ExtractionError("No readable text found on that page. It may need JavaScript or a login.")
    return title, segments


def fetch_url(url: str, timeout: int = 20, max_mb: int = 15) -> FetchedPage:
    import requests

    url = normalize_url(url)
    _check_public_host(url)
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,application/pdf,text/plain;q=0.9,*/*;q=0.5"}
    current = url
    # Follow redirects by hand so every hop is checked against private addresses.
    for _hop in range(6):
        try:
            resp = requests.get(current, headers=headers, timeout=timeout, stream=True, allow_redirects=False)
        except requests.exceptions.Timeout as exc:
            raise ExtractionError(f"The page took longer than {timeout}s to respond.") from exc
        except requests.exceptions.SSLError as exc:
            raise ExtractionError("The site's HTTPS certificate could not be verified.") from exc
        except requests.exceptions.RequestException as exc:
            raise ExtractionError(f"Could not reach that URL ({exc.__class__.__name__}).") from exc
        if resp.is_redirect and resp.headers.get("Location"):
            from urllib.parse import urljoin

            current = normalize_url(urljoin(current, resp.headers["Location"]))
            _check_public_host(current)
            resp.close()
            continue
        break
    else:
        raise ExtractionError("That URL redirected too many times.")

    if resp.status_code >= 400:
        hint = {401: "it requires a login", 403: "the site blocked automated access",
                404: "the page was not found", 429: "the site is rate-limiting requests"}.get(resp.status_code, "")
        raise ExtractionError(f"The site returned HTTP {resp.status_code}" + (f": {hint}." if hint else "."))

    limit = max_mb * 1024 * 1024
    data = bytearray()
    for part in resp.iter_content(chunk_size=65536):
        data.extend(part)
        if len(data) > limit:
            raise ExtractionError(f"That page is larger than {max_mb} MB.")
    data = bytes(data)
    ctype = (resp.headers.get("Content-Type") or "").lower()
    final_url = resp.url

    if "application/pdf" in ctype or final_url.lower().split("?")[0].endswith(".pdf"):
        name = Path(urlparse(final_url).path).name or "document.pdf"
        title, segments = extract_pdf(data, name)
        return FetchedPage(final_url, title, segments, data, ".pdf", "pdf")
    if "text/plain" in ctype:
        name = Path(urlparse(final_url).path).name or "page.txt"
        title, segments = extract_txt(data, name)
        return FetchedPage(final_url, title, segments, data, ".txt", "txt")
    if ctype and "html" not in ctype and "xml" not in ctype:
        raise ExtractionError(f"Unsupported content type at that URL: {ctype.split(';')[0]}.")

    encoding = resp.encoding if resp.encoding and resp.encoding.lower() != "iso-8859-1" else None
    html = data.decode(encoding or resp.apparent_encoding or "utf-8", errors="replace")
    title, segments = html_to_segments(html)
    return FetchedPage(final_url, title or urlparse(final_url).netloc, segments, data, ".html", "html")


def extract_file(data: bytes, filename: str) -> tuple[str, str, list[Segment]]:
    """Dispatch on extension. Returns (type, title, segments)."""
    kind = detect_type(filename)
    if kind == "pdf":
        title, segs = extract_pdf(data, filename)
    elif kind == "pptx":
        title, segs = extract_pptx(data, filename)
    else:
        title, segs = extract_txt(data, filename)
    return kind, title, segs
