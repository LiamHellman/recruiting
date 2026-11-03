import io
import zipfile

import pytest

from app.extraction.text import (
    DOCX,
    MAX_BYTES,
    PDF,
    DocumentError,
    clean_text,
    extract_document,
)
from tests.conftest import fixture_cv
from tests.docgen import make_docx, make_pdf


def squash(text: str) -> str:
    return " ".join(text.split())


def test_pdf_text_pages_and_embedded_links():
    pdf = make_pdf(
        [["Amira Haddad", "Backend Developer — Montréal"], ["SKILLS", "Python, Django"]],
        links=["https://github.com/amira-h"],
    )
    doc = extract_document(pdf)
    assert doc.content_type == PDF
    assert doc.text == "Amira Haddad\nBackend Developer — Montréal\n\nSKILLS\nPython, Django"
    assert [doc.text[s:e] for s, e in doc.pages] == [
        "Amira Haddad\nBackend Developer — Montréal",
        "SKILLS\nPython, Django",
    ]
    assert doc.links == ["https://github.com/amira-h"]


def test_fixture_cv_survives_a_pdf_round_trip():
    original = fixture_cv("02-ben-okafor")
    doc = extract_document(make_pdf([original.splitlines()]))
    assert squash(doc.text) == squash(original)


def test_docx_reads_header_body_and_tables_in_order_with_hyperlinks():
    data = make_docx(
        ["Experience", "Built APIs in FastAPI."],
        header=["Ben Okafor", "Toronto, ON"],
        table=[["Python", "PostgreSQL"], ["Docker", "AWS"]],
        links={"GitHub": "https://github.com/ben-okafor"},
    )
    doc = extract_document(data)
    assert doc.content_type == DOCX
    assert doc.text.splitlines() == [
        "Ben Okafor",
        "Toronto, ON",
        "Experience",
        "Python | PostgreSQL",
        "Docker | AWS",
        "Built APIs in FastAPI.",
        "Links: GitHub",
    ]
    assert doc.pages == [(0, len(doc.text))]
    assert doc.links == ["https://github.com/ben-okafor"]


def test_fixture_cv_survives_a_docx_round_trip_exactly():
    original = fixture_cv("01-amira-haddad")
    doc = extract_document(make_docx(original.splitlines()))
    assert doc.text == clean_text(original)


def test_clean_text_folds_ligatures_and_blank_runs():
    assert clean_text("ﬁnance\r\n\n\n\nnext  \x00line\t") == "finance\n\nnext  line"


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"hello world", "unsupported file type"),
        (b"\xd0\xcf\x11\xe0legacy .doc", "unsupported file type"),
        (b"%PDF-1.4\nnot really a pdf", "could not read the PDF"),
        (make_pdf([[]]), "no text found"),
    ],
)
def test_unreadable_files_are_rejected_with_a_safe_message(data: bytes, message: str):
    with pytest.raises(DocumentError, match=message):
        extract_document(data)


def test_zip_that_is_not_a_docx_is_rejected():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("hello.txt", "hi")
    with pytest.raises(DocumentError, match="unsupported file type"):
        extract_document(buffer.getvalue())


def test_docx_zip_bomb_is_rejected_before_parsing():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", "")
        archive.writestr("word/padding.bin", b"\0" * (51 * 1024 * 1024))
    with pytest.raises(DocumentError, match="expands to more than"):
        extract_document(buffer.getvalue())


def test_oversized_files_are_rejected():
    with pytest.raises(DocumentError, match="larger than"):
        extract_document(b"%PDF-" + b"\0" * MAX_BYTES)
