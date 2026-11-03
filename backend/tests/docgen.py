"""Build small PDF and DOCX files in memory, so fixture CVs live in git as plain text."""

import io

from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


def _pdf_string(text: str) -> bytes:
    raw = text.encode("cp1252")
    return b"(" + raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)") + b")"


def make_pdf(pages: list[list[str]], links: list[str] | None = None) -> bytes:
    """A text-only PDF: one Helvetica line per string. `links` become URI annotations on
    the first page (invisible, like a "GitHub" label that links somewhere)."""
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    catalog = add(b"")  # filled in below
    pages_obj = add(b"")
    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    page_ids = []
    for page_no, lines in enumerate(pages):
        stream = (
            b"BT /F1 10 Tf 14 TL 50 780 Td "
            + b" ".join(_pdf_string(line) + b" Tj T*" for line in lines)
            + b" ET"
        )
        content = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        annots = b""
        if page_no == 0 and links:
            annot_ids = [
                add(
                    b"<< /Type /Annot /Subtype /Link /Rect [50 50 150 60] /Border [0 0 0] "
                    b"/A << /S /URI /URI " + _pdf_string(url) + b" >> >>"
                )
                for url in links
            ]
            annots = b" /Annots [" + b" ".join(b"%d 0 R" % i for i in annot_ids) + b"]"
        page_ids.append(
            add(
                b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] "
                b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R%s >>"
                % (pages_obj, font, content, annots)
            )
        )
    objects[catalog - 1] = b"<< /Type /Catalog /Pages %d 0 R >>" % pages_obj
    kids = b" ".join(b"%d 0 R" % i for i in page_ids)
    objects[pages_obj - 1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, len(page_ids))

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n" % number + body + b"\nendobj\n")
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
    for offset in offsets:
        out.write(b"%010d 00000 n \n" % offset)
    out.write(
        b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n"
        % (len(objects) + 1, catalog, xref)
    )
    return out.getvalue()


def _add_hyperlink(paragraph, url: str, text: str) -> None:
    rel_id = paragraph.part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), rel_id)
    run = OxmlElement("w:r")
    text_el = OxmlElement("w:t")
    text_el.text = text
    run.append(text_el)
    link.append(run)
    paragraph._p.append(link)


def make_docx(
    paragraphs: list[str],
    *,
    header: list[str] | None = None,
    table: list[list[str]] | None = None,
    links: dict[str, str] | None = None,
) -> bytes:
    """A DOCX with body paragraphs, an optional page header, an optional table after the
    first paragraph, and optional hyperlinks (label -> url) in a final paragraph."""
    doc = Document()
    if header:
        section_header = doc.sections[0].header
        section_header.paragraphs[0].text = header[0]
        for line in header[1:]:
            section_header.add_paragraph(line)
    for i, text in enumerate(paragraphs):
        doc.add_paragraph(text)
        if i == 0 and table:
            grid = doc.add_table(rows=len(table), cols=len(table[0]))
            for r, row in enumerate(table):
                for c, cell in enumerate(row):
                    grid.cell(r, c).text = cell
    if links:
        paragraph = doc.add_paragraph("Links: ")
        for label, url in links.items():
            _add_hyperlink(paragraph, url, label)
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()
