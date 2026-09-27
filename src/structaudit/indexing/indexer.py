"""
Structural indexes: deterministic lookup tables built from a
ParsedDocument, so a reference like "Table 7" or "footnote 049" can be
resolved to a unit_id without re-scanning the document.
"""

from __future__ import annotations

import re

from structaudit.core import ParsedDocument, ReferenceKind, StructuralIndexes, UnitType

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_ROMAN_RE = re.compile(r"^[IVXLCDM]+$", re.IGNORECASE)


def _roman_to_int(roman: str) -> int:
    roman = roman.upper()
    total = 0
    prev = 0
    for ch in reversed(roman):
        value = _ROMAN_VALUES[ch]
        total += -value if value < prev else value
        prev = max(prev, value)

    return total


def normalize_identifier(kind: ReferenceKind, raw: str) -> str:
    """Normalize a raw identifier into "KIND:VALUE" form:
        Table 03      -> TABLE:3
        chapter V     -> CHAPTER:5
        Appendix b    -> APPENDIX:B
        footnote 049  -> FOOTNOTE:49

    Dotted numeric identifiers have leading zeros stripped per segment;
    roman numerals convert to their integer value; anything else is
    uppercased.
    """
    raw = raw.strip()

    if re.fullmatch(r"\d+(\.\d+)*", raw):
        value = ".".join(str(int(seg)) for seg in raw.split("."))
    elif kind == ReferenceKind.CHAPTER and _ROMAN_RE.match(raw):
        value = str(_roman_to_int(raw))
    else:
        value = raw.upper()

    return f"{kind.value}:{value}"


_FOOTNOTE_LABEL_RE = re.compile(r"^FOOTNOTE:(\d+)$")
_NOTE_LABEL_RE = re.compile(r"^NOTE:(\d+)$")
_CHAPTER_LABEL_RE = re.compile(r"^CHAPTER:(.+)$")
_APPENDIX_LABEL_RE = re.compile(r"^APPENDIX:(.+)$")
_SECTION_LABEL_RE = re.compile(r"^SECTION:(.+)$")
_TABLE_LABEL_RE = re.compile(r"^Table\s+(.+)$", re.IGNORECASE)


def build_structural_indexes(parsed: ParsedDocument) -> StructuralIndexes:
    """Build lookup indexes from a parsed document's units and tables.

    A unit whose label doesn't parse into the expected shape is skipped
    rather than raising, so one malformed label doesn't fail the whole
    document's index construction.
    """
    table_number_to_unit: dict[str, str] = {}
    for table in parsed.tables:
        if table.label:
            match = _TABLE_LABEL_RE.match(table.label)
            if match:
                key = normalize_identifier(ReferenceKind.TABLE, match.group(1))
                # First occurrence wins: a table split across multiple
                # page breaks can share one label with no
                # structural way to tell which fragment a reference means.
                table_number_to_unit.setdefault(key, table.unit_id)

    section_number_to_unit: dict[str, str] = {}
    chapter_number_to_unit: dict[str, str] = {}
    appendix_label_to_unit: dict[str, str] = {}
    footnote_number_to_unit: dict[str, str] = {}
    note_number_to_unit: dict[str, str] = {}

    headings_by_order: list[str] = []
    toc_entries: list[str] = []
    lists_by_order: list[str] = []

    for unit in parsed.units:
        if unit.unit_type in (UnitType.SECTION, UnitType.SUBSECTION, UnitType.APPENDIX):
            headings_by_order.append(unit.unit_id)
        elif unit.unit_type == UnitType.TOC_ENTRY:
            toc_entries.append(unit.unit_id)
        elif unit.unit_type == UnitType.LIST:
            lists_by_order.append(unit.unit_id)

        if not unit.label:
            continue
        if match := _FOOTNOTE_LABEL_RE.match(unit.label):
            footnote_number_to_unit[normalize_identifier(ReferenceKind.FOOTNOTE, match.group(1))] = unit.unit_id
        elif match := _NOTE_LABEL_RE.match(unit.label):
            note_number_to_unit[normalize_identifier(ReferenceKind.NOTE, match.group(1))] = unit.unit_id
        elif match := _APPENDIX_LABEL_RE.match(unit.label):
            appendix_label_to_unit[normalize_identifier(ReferenceKind.APPENDIX, match.group(1))] = unit.unit_id
        elif match := _CHAPTER_LABEL_RE.match(unit.label):
            chapter_number_to_unit[normalize_identifier(ReferenceKind.CHAPTER, match.group(1))] = unit.unit_id
        elif match := _SECTION_LABEL_RE.match(unit.label):
            section_number_to_unit[normalize_identifier(ReferenceKind.SECTION, match.group(1))] = unit.unit_id

    return StructuralIndexes(
        table_number_to_unit=table_number_to_unit,
        figure_number_to_unit={},
        chart_number_to_unit={},
        section_number_to_unit=section_number_to_unit,
        chapter_number_to_unit=chapter_number_to_unit,
        appendix_label_to_unit=appendix_label_to_unit,
        note_number_to_unit=note_number_to_unit,
        footnote_number_to_unit=footnote_number_to_unit,
        toc_entries=toc_entries,
        headings_by_order=headings_by_order,
        tables_by_order=[t.unit_id for t in parsed.tables],
        figures_by_order=[],
        lists_by_order=lists_by_order,
    )
