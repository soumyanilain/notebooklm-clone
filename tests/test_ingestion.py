import pytest

from notebooklm.ingestion.chunker import split_text
from notebooklm.ingestion.extractors import (ExtractionError, _check_public_host, extract_file, html_to_segments,
                                             normalize_url)
from tests.conftest import CORPUS


def test_split_text_covers_everything_with_overlap():
    text = " ".join(f"Sentence number {i} talks about drones." for i in range(200))
    spans = split_text(text, chunk_size=500, overlap=80)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(end - start <= 500 for start, end in spans)
    for (s1, e1), (s2, _e2) in zip(spans, spans[1:]):
        assert s2 < e1, "consecutive chunks should overlap"


def test_short_text_is_one_chunk():
    assert split_text("hello world", 1000, 150) == [(0, 11)]


def test_pdf_pages_are_citable():
    kind, title, segs = extract_file((CORPUS / "halcyon_field_operations_handbook.pdf").read_bytes(), "h.pdf")
    assert kind == "pdf" and segs[0].label == "p. 1"
    assert any("OPS-19" in s.text for s in segs)
    assert "Halcyon" in title


def test_pptx_includes_speaker_notes():
    kind, _title, segs = extract_file((CORPUS / "project_kestrel_q3_review.pptx").read_bytes(), "deck.pptx")
    assert kind == "pptx" and len(segs) == 10 and segs[8].label == "slide 9"
    assert "Brightwater Utilities" in segs[8].text  # only present in the notes


def test_txt_uses_line_ranges():
    kind, _title, segs = extract_file((CORPUS / "postmortem_INC-2047.txt").read_bytes(), "pm.txt")
    assert kind == "txt" and segs[0].label.startswith("lines 1-")


def test_unsupported_and_empty_files():
    with pytest.raises(ExtractionError):
        extract_file(b"data", "notes.docx")
    with pytest.raises(ExtractionError):
        extract_file(b"   ", "empty.txt")
    with pytest.raises(ExtractionError):
        extract_file(b"not a pdf", "broken.pdf")


def test_html_sections_use_headings():
    html = """<html><head><title>Test page</title></head><body><nav>menu</nav><main>
    <h1>Intro</h1><p>""" + "Retrieval augmented generation grounds answers. " * 5 + """</p>
    <h2>Details</h2><p>""" + "Chunks are embedded and stored in a vector database. " * 5 + """</p></main>
    <footer>footer text</footer></body></html>"""
    title, segs = html_to_segments(html)
    assert title == "Test page"
    assert [s.label for s in segs] == ["§ Intro", "§ Details"]
    assert all("menu" not in s.text and "footer" not in s.text for s in segs)


def test_url_validation_blocks_private_hosts():
    assert normalize_url("example.com/a") == "https://example.com/a"
    with pytest.raises(ExtractionError):
        normalize_url("ftp://example.com")
    with pytest.raises(ExtractionError):
        _check_public_host("http://localhost:7860/")
    with pytest.raises(ExtractionError):
        _check_public_host("http://127.0.0.1/")
