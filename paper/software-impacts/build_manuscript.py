# SPDX-License-Identifier: AGPL-3.0-only
"""Build the Software Impacts working manuscript as a DOCX file."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

FONT_NAME = "Times New Roman"
BLACK = RGBColor(0, 0, 0)
DARK_HEADER = "404040"
LIGHT_ROW = "F2F2F2"
BORDER = "D9D9D9"


def set_font(run, size: float, *, bold: bool = False) -> None:
    """Apply a deterministic font to a run."""
    run.font.name = FONT_NAME
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = BLACK
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), FONT_NAME)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), FONT_NAME)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), FONT_NAME)


def set_cell_shading(cell, fill: str) -> None:
    """Set a table cell background fill."""
    properties = cell._tc.get_or_add_tcPr()
    shading = properties.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        properties.append(shading)
    shading.set(qn("w:fill"), fill)


def set_cell_margins(
    cell,
    top: int = 70,
    start: int = 80,
    bottom: int = 70,
    end: int = 80,
) -> None:
    """Apply internal table-cell padding in twentieths of a point."""
    properties = cell._tc.get_or_add_tcPr()
    margins = properties.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        properties.append(margins)
    for tag, value in (
        ("w:top", top),
        ("w:start", start),
        ("w:bottom", bottom),
        ("w:end", end),
    ):
        node = margins.find(qn(tag))
        if node is None:
            node = OxmlElement(tag)
            margins.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table) -> None:
    """Apply light gray outer and internal borders."""
    properties = table._tbl.tblPr
    borders = properties.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        properties.append(borders)
    for tag in ("top", "left", "bottom", "right", "insideH", "insideV"):
        edge = borders.find(qn(f"w:{tag}"))
        if edge is None:
            edge = OxmlElement(f"w:{tag}")
            borders.append(edge)
        edge.set(qn("w:val"), "single")
        edge.set(qn("w:sz"), "4")
        edge.set(qn("w:space"), "0")
        edge.set(qn("w:color"), BORDER)


def repeat_header(row) -> None:
    """Mark a table row as a repeating header."""
    properties = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    properties.append(header)


def prevent_row_split(row) -> None:
    """Prevent a table row from splitting across pages."""
    properties = row._tr.get_or_add_trPr()
    properties.append(OxmlElement("w:cantSplit"))


def add_hyperlink(
    paragraph,
    text: str,
    url: str,
    size: float = 7.5,
) -> None:
    """Add a styled external hyperlink to a paragraph."""
    relationship_id = paragraph.part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), relationship_id)
    run = OxmlElement("w:r")
    properties = OxmlElement("w:rPr")
    fonts = OxmlElement("w:rFonts")
    fonts.set(qn("w:ascii"), FONT_NAME)
    fonts.set(qn("w:hAnsi"), FONT_NAME)
    fonts.set(qn("w:eastAsia"), FONT_NAME)
    properties.append(fonts)
    font_size = OxmlElement("w:sz")
    font_size.set(qn("w:val"), str(int(size * 2)))
    properties.append(font_size)
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    properties.append(color)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    properties.append(underline)
    run.append(properties)
    text_node = OxmlElement("w:t")
    text_node.text = text
    run.append(text_node)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def add_page_number(paragraph) -> None:
    """Add a centered PAGE field to the footer."""
    run = paragraph.add_run()
    field_begin = OxmlElement("w:fldChar")
    field_begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    field_separate = OxmlElement("w:fldChar")
    field_separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    field_end = OxmlElement("w:fldChar")
    field_end.set(qn("w:fldCharType"), "end")
    run._r.extend([field_begin, instruction, field_separate, text, field_end])
    set_font(run, 8)


def add_paragraphs(document: Document, paragraphs: Iterable[str]) -> None:
    """Add body paragraphs with consistent typography."""
    for text in paragraphs:
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(3)
        paragraph.paragraph_format.line_spacing = 1.0
        paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        set_font(paragraph.add_run(text), 9.2)


def remove_paragraph_borders(paragraph) -> None:
    """Remove any style-inherited paragraph border from a title."""
    properties = paragraph._p.get_or_add_pPr()
    for border in properties.findall(qn("w:pBdr")):
        properties.remove(border)


def remove_style_paragraph_borders(style) -> None:
    """Remove a built-in paragraph border inherited from a Word style."""
    properties = style._element.get_or_add_pPr()
    for border in properties.findall(qn("w:pBdr")):
        properties.remove(border)


def build(data_path: Path, output_path: Path) -> None:
    """Build the manuscript DOCX from the checked-in JSON source."""
    data = json.loads(data_path.read_text(encoding="utf-8"))
    document = Document()
    section = document.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(1.7)
    section.bottom_margin = Cm(1.7)
    section.left_margin = Cm(1.9)
    section.right_margin = Cm(1.9)

    normal = document.styles["Normal"]
    normal.font.name = FONT_NAME
    normal.font.size = Pt(9.2)
    normal.font.color.rgb = BLACK
    normal._element.rPr.rFonts.set(qn("w:ascii"), FONT_NAME)
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), FONT_NAME)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_NAME)

    for style_name, size in (("Title", 16), ("Heading 1", 11),
                             ("Heading 2", 10)):
        style = document.styles[style_name]
        style.font.name = FONT_NAME
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = BLACK
        style._element.rPr.rFonts.set(qn("w:ascii"), FONT_NAME)
        style._element.rPr.rFonts.set(qn("w:hAnsi"), FONT_NAME)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_NAME)
        remove_style_paragraph_borders(style)

    title = document.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    title.paragraph_format.space_after = Pt(9)
    remove_paragraph_borders(title)
    set_font(title.add_run(data["title"]), 16, bold=True)

    author_paragraph = document.add_paragraph()
    author_paragraph.paragraph_format.space_after = Pt(2)
    for index, author in enumerate(data["authors"]):
        if index:
            set_font(author_paragraph.add_run(", "), 10)
        set_font(author_paragraph.add_run(author["name"]), 10, bold=True)
        set_font(
            author_paragraph.add_run(
                ",".join(str(value) for value in author["affiliations"])
                + ("*" if author["corresponding"] else "")
            ),
            7,
        )

    email_paragraph = document.add_paragraph()
    email_paragraph.paragraph_format.space_after = Pt(4)
    emails = "; ".join(
        f"{author['name']}: {author['email']}" for author in data["authors"]
    )
    set_font(email_paragraph.add_run(emails), 8)
    set_font(
        email_paragraph.add_run("  *Corresponding author."),
        8,
    )

    for affiliation in data["affiliations"]:
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(1)
        set_font(
            paragraph.add_run(
                f"{affiliation['number']} {affiliation['text']}"
            ),
            8,
        )

    document.add_paragraph()
    abstract_heading = document.add_paragraph()
    abstract_heading.paragraph_format.space_after = Pt(2)
    set_font(abstract_heading.add_run("Abstract"), 10, bold=True)
    abstract = document.add_paragraph()
    abstract.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    abstract.paragraph_format.line_spacing = 1.0
    abstract.paragraph_format.space_after = Pt(5)
    set_font(abstract.add_run(data["abstract"]), 9)

    keyword_paragraph = document.add_paragraph()
    keyword_paragraph.paragraph_format.space_after = Pt(7)
    set_font(keyword_paragraph.add_run("Keywords: "), 9, bold=True)
    set_font(keyword_paragraph.add_run("; ".join(data["keywords"])), 9)

    body_heading = document.add_paragraph()
    body_heading.paragraph_format.space_after = Pt(4)
    set_font(body_heading.add_run("1. Code metadata"), 11, bold=True)

    table = document.add_table(rows=1, cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.columns[0].width = Cm(1.2)
    table.columns[1].width = Cm(5.4)
    table.columns[2].width = Cm(10.4)
    set_table_borders(table)
    headers = ("Nr.", "Code metadata description", "Metadata")
    for index, text in enumerate(headers):
        cell = table.rows[0].cells[index]
        cell.width = (Cm(1.2), Cm(5.4), Cm(10.4))[index]
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_shading(cell, DARK_HEADER)
        set_cell_margins(cell)
        paragraph = cell.paragraphs[0]
        paragraph.alignment = (
            WD_ALIGN_PARAGRAPH.CENTER if index == 0
            else WD_ALIGN_PARAGRAPH.LEFT
        )
        run = paragraph.add_run(text)
        set_font(run, 8, bold=True)
        run.font.color.rgb = RGBColor(255, 255, 255)
    repeat_header(table.rows[0])

    widths = (Cm(1.2), Cm(5.4), Cm(10.4))
    for row_index, item in enumerate(data["code_metadata"], start=1):
        row = table.add_row()
        prevent_row_split(row)
        values = (item["id"], item["label"], item["value"])
        for column_index, value in enumerate(values):
            cell = row.cells[column_index]
            cell.width = widths[column_index]
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            if row_index % 2 == 0:
                set_cell_shading(cell, LIGHT_ROW)
            paragraph = cell.paragraphs[0]
            paragraph.alignment = (
                WD_ALIGN_PARAGRAPH.CENTER if column_index == 0
                else WD_ALIGN_PARAGRAPH.LEFT
            )
            set_font(paragraph.add_run(value), 7.5)

    spacer = document.add_paragraph()
    spacer.paragraph_format.space_after = Pt(1)

    for section_data in data["sections"]:
        heading = document.add_paragraph(style="Heading 1")
        heading.paragraph_format.page_break_before = bool(
            section_data.get("page_break_before", False)
        )
        heading.paragraph_format.space_before = Pt(7)
        heading.paragraph_format.space_after = Pt(3)
        heading.paragraph_format.keep_with_next = True
        set_font(heading.add_run(section_data["heading"]), 11, bold=True)
        add_paragraphs(document, section_data["paragraphs"])

    references_heading = document.add_paragraph(style="Heading 1")
    references_heading.paragraph_format.space_before = Pt(5)
    references_heading.paragraph_format.space_after = Pt(3)
    references_heading.paragraph_format.keep_with_next = True
    set_font(references_heading.add_run("References"), 11, bold=True)
    for reference in data["references"]:
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.left_indent = Cm(0.5)
        paragraph.paragraph_format.first_line_indent = Cm(-0.5)
        paragraph.paragraph_format.space_after = Pt(0.8)
        paragraph.paragraph_format.line_spacing = 0.90
        text = reference
        match = re.search(r"https?://[^\s]+", text)
        if match is not None:
            url = match.group(0).rstrip(".,;")
            before = text[:match.start()]
            after = text[match.start() + len(url):]
            set_font(paragraph.add_run(before), 7.2)
            add_hyperlink(paragraph, url, url, size=7.2)
            set_font(paragraph.add_run(after), 7.2)
        else:
            set_font(paragraph.add_run(text), 7.2)

    footer = section.footer
    footer_paragraph = footer.paragraphs[0]
    footer_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_page_number(footer_paragraph)

    document.core_properties.title = data["title"]
    document.core_properties.author = ", ".join(
        author["name"] for author in data["authors"]
    )
    document.core_properties.subject = (
        "Software Impacts Original Software Publication working draft"
    )
    document.core_properties.keywords = "; ".join(data["keywords"])
    document.core_properties.comments = ""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(output_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).with_name("manuscript.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name(
            "CPO-STCM_SoftwareImpacts_Working_Draft.docx"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Build the manuscript and print its location."""
    args = parse_args()
    build(args.data, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
