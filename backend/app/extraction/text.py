"""PDF/DOCX -> plain text with stable character offsets.

The returned `text` is the reference for every CV locator (`cv:<doc>#start-end`), so it is
stored as-is and never re-derived. Only text is read: images (photos) are ignored.
pdfplumber for PDFs (PyMuPDF is AGPL); python-docx for DOCX. Legacy .doc is not supported.
"""

import io
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from typing import Any, Literal

import pdfplumber
from docx import Document as open_docx
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.table import Table
from docx.text.paragraph import Paragraph

MAX_BYTES = 10 * 1024 * 1024
MAX_PAGES = 30
MAX_DOCX_UNCOMPRESSED = 50 * 1024 * 1024  # zip-bomb guard
MAX_CHARS = 200_000

ContentType = Literal[
    "application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
]
PDF: ContentType = "application/pdf"
DOCX: ContentType = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class DocumentError(ValueError):
    """The file is not a document we can read. The message is safe to show the user."""


@dataclass(frozen=True)
class ExtractedDocument:
    content_type: ContentType
    text: str
    pages: list[tuple[int, int]]  # [start, end) of each PDF page in `text`; one span for DOCX
    links: list[str]  # hyperlink targets embedded in the file (not those in the text)


def sniff(data: bytes) -> ContentType:
    if data.startswith(b"%PDF-"):
        return PDF
    if data.startswith(b"PK\x03\x04"):
        return DOCX
    raise DocumentError("unsupported file type: upload a PDF or DOCX")


def extract_document(data: bytes) -> ExtractedDocument:
    if len(data) > MAX_BYTES:
        raise DocumentError(f"file is larger than {MAX_BYTES // (1024 * 1024)} MB")
    content_type = sniff(data)
    chunks, links = _read_pdf(data) if content_type == PDF else _read_docx(data)

    parts: list[str] = []
    pages: list[tuple[int, int]] = []
    position = 0
    for chunk in (clean_text(c) for c in chunks):
        if parts:
            parts.append("\n\n")
            position += 2
        parts.append(chunk)
        pages.append((position, position + len(chunk)))
        position += len(chunk)
    text = "".join(parts)
    if not text.strip():
        raise DocumentError("no text found (a scanned PDF?); text recognition is not supported")
    if len(text) > MAX_CHARS:
        raise DocumentError(f"document has more than {MAX_CHARS} characters of text")
    return ExtractedDocument(
        content_type=content_type, text=text, pages=pages, links=list(dict.fromkeys(links))
    )


_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def clean_text(text: str) -> str:
    """NFKC (folds PDF ligatures like "ﬁ"), \\n line ends, no control characters, no
    trailing spaces, at most one blank line in a row."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    text = _CONTROL.sub("", text.replace("\t", " "))
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip("\n")


def _read_pdf(data: bytes) -> tuple[list[str], list[str]]:
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            if len(pdf.pages) > MAX_PAGES:
                raise DocumentError(f"PDF has more than {MAX_PAGES} pages")
            chunks: list[str] = []
            links: list[str] = []
            for page in pdf.pages:
                chunks.append(page.extract_text() or "")
                links.extend(str(h["uri"]) for h in page.hyperlinks if h.get("uri"))
            return chunks, links
    except DocumentError:
        raise
    except Exception as exc:  # pdfminer raises many types for broken or encrypted files
        raise DocumentError("could not read the PDF (damaged or password-protected?)") from exc


def _read_docx(data: bytes) -> tuple[list[str], list[str]]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
            if "word/document.xml" not in names:
                raise DocumentError("unsupported file type: upload a PDF or DOCX")
            if sum(info.file_size for info in archive.infolist()) > MAX_DOCX_UNCOMPRESSED:
                raise DocumentError("DOCX expands to more than 50 MB")
        doc = open_docx(io.BytesIO(data))
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError("could not read the DOCX (damaged?)") from exc

    lines: list[str] = []
    parts: list[Any] = []
    seen_headers: set[int] = set()
    for section in doc.sections:
        header = section.header
        if header.is_linked_to_previous or id(header.part) in seen_headers:
            continue
        seen_headers.add(id(header.part))
        lines.extend(_block_lines(header))
        parts.append(header.part)
    lines.extend(_block_lines(doc))
    parts.append(doc.part)

    links = [
        str(rel.target_ref)
        for part in parts
        for rel in part.rels.values()
        if rel.is_external and rel.reltype == RELATIONSHIP_TYPE.HYPERLINK
    ]
    return ["\n".join(lines)], links


def _block_lines(container: Any) -> list[str]:
    """Paragraph and table text in document order. Table rows become one line, cells
    joined by " | "; merged cells are read once."""
    lines: list[str] = []
    for block in container.iter_inner_content():
        if isinstance(block, Paragraph):
            lines.append(block.text)
        elif isinstance(block, Table):
            for row in block.rows:
                seen: set[int] = set()
                cells: list[str] = []
                for cell in row.cells:
                    if id(cell._tc) in seen:
                        continue
                    seen.add(id(cell._tc))
                    text = " / ".join(line for line in _block_lines(cell) if line.strip())
                    if text:
                        cells.append(text)
                if cells:
                    lines.append(" | ".join(cells))
    return lines
