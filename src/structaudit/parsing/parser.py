"""
Document parser: turns raw markdown-ish text into a ParsedDocument.

Headings, lists, and tables come from real markdown tokens.
Footnotes, notes, TOC entries, and reference mentions aren't representable
as markdown tokens in these documents (footnotes appear as plain "(1)
text" paragraphs), so those are extracted via regex over
the raw text instead.
"""

from __future__ import annotations

import re

from markdown_it import MarkdownIt
from mdit_py_plugins.gfm import gfm_plugin

from structaudit.core import (
    DocumentUnit,
    ParsedDocument,
    ReferenceKind,
    ReferenceMention,
    TableUnit,
    UnitType,
)

_MD = MarkdownIt("commonmark").use(gfm_plugin)

# h1 -> SECTION; everything deeper collapses to SUBSECTION (no finer
# per-level unit type exists).
_HEADING_TAG_TO_UNIT_TYPE = {"h1": UnitType.SECTION}

_CHAPTER_HEADING_RE = re.compile(r"^Chapter\s+([IVXLCDM]+)\b", re.IGNORECASE)
_APPENDIX_HEADING_RE = re.compile(r"^Appendix\s+([A-Za-z0-9]+)\b", re.IGNORECASE)
_SECTION_HEADING_RE = re.compile(r"^Section\s+(\d+(?:\.\d+)*)\b", re.IGNORECASE)

_REFERENCE_MENTION_RE = re.compile(
    r"\b(Table|Figure|Chart|Appendix|Section|Chapter|Note|Footnote)\s+"
    r"([A-Z]|\d+(?:\.\d+)*|[IVXLCDM]+)\b",
    re.IGNORECASE,
)

# "(1) text..." paragraphs following a Footnotes heading -- not markdown lists.
_FOOTNOTE_LINE_RE = re.compile(r"^\((\d+)\)\s+(.+)$", re.MULTILINE)

# "Note 14: ..." / "Note 14. ..."
_NOTE_LINE_RE = re.compile(r"^Note\s+(\d+)\s*[:.]?\s*(.+)$", re.MULTILINE | re.IGNORECASE)

# Dot-leader / trailing-page-number TOC line, e.g. "Introduction .... 4".
_TOC_LINE_RE = re.compile(r"^(.{3,100}?)\s*(?:\.{2,}|\s{2,})\s*(\d{1,4})\s*$", re.MULTILINE)


def _build_line_offsets(text: str) -> list[int]:
    """offsets[i] = char index where line i starts, for converting
    markdown-it-py's line-based token.map into char_start/char_end.
    """
    offsets = [0]
    for line in text.split("\n"):
        offsets.append(offsets[-1] + len(line) + 1)

    return offsets


def _char_span(token_map: list[int] | None, line_offsets: list[int]) -> tuple[int | None, int | None]:
    if token_map is None:
        return None, None

    start_line, end_line = token_map
    char_start = line_offsets[start_line]
    char_end = line_offsets[end_line] if end_line < len(line_offsets) else line_offsets[-1]
    return char_start, char_end


def _inline_text(token) -> str:
    return token.content if token else ""


def extract_sections(tokens, line_offsets: list[int]) -> list[DocumentUnit]:
    """Headings -> SECTION (h1) / SUBSECTION (h2+) units, with section_path
    tracking the ancestor heading stack.
    """
    units: list[DocumentUnit] = []
    stack: list[tuple[int, str]] = []
    order = 0

    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.type == "heading_open":
            level = int(token.tag[1:])
            inline = tokens[i + 1] if i + 1 < len(tokens) else None
            text = _inline_text(inline)
            char_start, char_end = _char_span(token.map, line_offsets)

            while stack and stack[-1][0] >= level:
                stack.pop()

            section_path = [t for _, t in stack]
            stack.append((level, text))

            unit_type = _HEADING_TAG_TO_UNIT_TYPE.get(token.tag, UnitType.SUBSECTION)

            label = None
            if match := _APPENDIX_HEADING_RE.match(text):
                unit_type = UnitType.APPENDIX
                label = f"APPENDIX:{match.group(1)}"
            elif match := _CHAPTER_HEADING_RE.match(text):
                label = f"CHAPTER:{match.group(1)}"
            elif match := _SECTION_HEADING_RE.match(text):
                label = f"SECTION:{match.group(1)}"

            units.append(
                DocumentUnit(
                    unit_id=f"heading-{order}",
                    unit_type=unit_type,
                    order=order,
                    text=text,
                    title=text,
                    label=label,
                    section_path=section_path,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
            order += 1
        i += 1
    return units


def extract_lists(tokens, line_offsets: list[int]) -> list[DocumentUnit]:
    """bullet_list/ordered_list + list_item tokens -> LIST / LIST_ITEM units."""
    units: list[DocumentUnit] = []
    list_stack: list[str] = []
    order = 0

    for i, token in enumerate(tokens):
        if token.type in ("bullet_list_open", "ordered_list_open"):
            char_start, char_end = _char_span(token.map, line_offsets)
            list_id = f"list-{order}"
            units.append(
                DocumentUnit(
                    unit_id=list_id,
                    unit_type=UnitType.LIST,
                    parent_id=list_stack[-1] if list_stack else None,
                    order=order,
                    text="",
                    char_start=char_start,
                    char_end=char_end,
                )
            )
            list_stack.append(list_id)
            order += 1
        elif token.type in ("bullet_list_close", "ordered_list_close"):
            if list_stack:
                list_stack.pop()
        elif token.type == "list_item_open":
            char_start, char_end = _char_span(token.map, line_offsets)
            depth = 0
            item_text = ""
            for j in range(i + 1, len(tokens)):
                if tokens[j].type == "list_item_open":
                    depth += 1
                elif tokens[j].type == "list_item_close":
                    if depth == 0:
                        break
                    depth -= 1
                elif tokens[j].type == "inline" and not item_text:
                    item_text = tokens[j].content

            units.append(
                DocumentUnit(
                    unit_id=f"listitem-{order}",
                    unit_type=UnitType.LIST_ITEM,
                    parent_id=list_stack[-1] if list_stack else None,
                    order=order,
                    text=item_text,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
            order += 1
    return units


_TABLE_CAPTION_RE = re.compile(r"^(Table\s+([A-Za-z0-9]+))\.?\s*(.*)$", re.IGNORECASE)


def extract_tables(tokens, line_offsets: list[int]) -> list[TableUnit]:
    """table_open...table_close -> TableUnit. column_headers comes from the
    single supported header row; any other rows (including a second
    header-like row) land in cells.

    Caption/label comes from the most recent heading that itself looked
    like a table caption ("Table X. ..."), tracked separately from
    ordinary headings so it survives unrelated headings (e.g. "Footnotes")
    interleaved between a table split across multiple GFM blocks.
    """
    tables: list[TableUnit] = []
    order = 0
    last_table_caption_match: tuple[str, str] | None = None

    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.type == "heading_open":
            inline = tokens[i + 1] if i + 1 < len(tokens) else None
            heading_text = _inline_text(inline).strip()
            match = _TABLE_CAPTION_RE.match(heading_text)
            if match:
                last_table_caption_match = (f"Table {match.group(2)}", heading_text)
        elif token.type == "table_open":
            char_start, char_end = _char_span(token.map, line_offsets)
            column_headers: list[str] = []
            cells: list[list[str]] = []
            current_row: list[str] | None = None
            in_header = False

            j = i + 1
            while j < len(tokens) and tokens[j].type != "table_close":
                t = tokens[j]
                if t.type == "thead_open":
                    in_header = True
                elif t.type == "thead_close":
                    in_header = False
                elif t.type == "tr_open":
                    current_row = []
                elif t.type == "tr_close":
                    if current_row is not None:
                        if in_header:
                            column_headers = current_row
                        else:
                            cells.append(current_row)
                    current_row = None
                elif t.type == "inline" and current_row is not None:
                    current_row.append(t.content)
                j += 1

            label, caption = last_table_caption_match if last_table_caption_match else (None, None)

            tables.append(
                TableUnit(
                    unit_id=f"table-{order}",
                    order=order,
                    label=label,
                    caption=caption,
                    column_headers=column_headers,
                    cells=cells,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
            order += 1
            i = j
        i += 1
    return tables


def extract_toc_entries(text: str) -> list[DocumentUnit]:
    """Dot-leader / trailing-page-number lines. Returns [] for documents with no TOC."""
    units = []
    for order, match in enumerate(_TOC_LINE_RE.finditer(text)):
        units.append(
            DocumentUnit(
                unit_id=f"toc-{order}",
                unit_type=UnitType.TOC_ENTRY,
                order=order,
                text=match.group(0).strip(),
                title=match.group(1).strip(),
                char_start=match.start(),
                char_end=match.end(),
            )
        )
    return units


def extract_footnotes(text: str) -> list[DocumentUnit]:
    """"(N) text" paragraph lines -> FOOTNOTE units."""
    units = []
    for order, match in enumerate(_FOOTNOTE_LINE_RE.finditer(text)):
        units.append(
            DocumentUnit(
                unit_id=f"footnote-{order}",
                unit_type=UnitType.FOOTNOTE,
                order=order,
                text=match.group(2).strip(),
                label=f"FOOTNOTE:{match.group(1)}",
                char_start=match.start(),
                char_end=match.end(),
            )
        )
    return units


def extract_notes(text: str) -> list[DocumentUnit]:
    """"Note N: ..." numbered lines -> NOTE units."""
    units = []
    for order, match in enumerate(_NOTE_LINE_RE.finditer(text)):
        units.append(
            DocumentUnit(
                unit_id=f"note-{order}",
                unit_type=UnitType.NOTE,
                order=order,
                text=match.group(2).strip(),
                label=f"NOTE:{match.group(1)}",
                char_start=match.start(),
                char_end=match.end(),
            )
        )
    return units


def extract_reference_mentions(text: str) -> list[ReferenceMention]:
    """Explicit structural pointers ("Table 4", "footnote 49", "(See table A.)")
    as raw surface evidence -- normalization and resolution happen later.
    """
    mentions = []
    for match in _REFERENCE_MENTION_RE.finditer(text):
        mentions.append(
            ReferenceMention(
                raw_text=match.group(0),
                kind=ReferenceKind[match.group(1).upper()],
                identifier_raw=match.group(2),
                char_start=match.start(),
                char_end=match.end(),
            )
        )
    return mentions


def parse_document(text: str) -> ParsedDocument:
    """Parse raw text into headings, lists, tables, TOC entries, footnotes,
    notes, and reference mentions, merged into document order.
    """
    tokens = _MD.parse(text)
    line_offsets = _build_line_offsets(text)

    sections = extract_sections(tokens, line_offsets)
    lists_ = extract_lists(tokens, line_offsets)
    tables = extract_tables(tokens, line_offsets)
    toc_entries = extract_toc_entries(text)
    footnotes = extract_footnotes(text)
    notes = extract_notes(text)
    reference_mentions = extract_reference_mentions(text)

    all_units = sections + lists_ + toc_entries + footnotes + notes
    all_units.sort(key=lambda u: u.char_start if u.char_start is not None else 0)

    return ParsedDocument(
        raw_text=text,
        units=all_units,
        tables=tables,
        reference_mentions=reference_mentions,
    )
