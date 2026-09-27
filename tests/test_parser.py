from pathlib import Path

from structaudit.parsing.parser import (
    extract_footnotes,
    extract_lists,
    extract_reference_mentions,
    extract_sections,
    extract_tables,
    extract_toc_entries,
    parse_document,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


# extract_sections

def test_sections_levels_and_path():
    doc = parse_document("# Top\n\n## Middle\n\n### Bottom\n\nSome text.\n")
    sections = [u for u in doc.units if u.unit_type.value in ("SECTION", "SUBSECTION")]
    assert [s.text for s in sections] == ["Top", "Middle", "Bottom"]
    assert sections[0].unit_type.value == "SECTION"  # h1
    assert sections[1].unit_type.value == "SUBSECTION"  # h2
    assert sections[2].unit_type.value == "SUBSECTION"  # h3 also collapses to SUBSECTION
    assert sections[1].section_path == ["Top"]
    assert sections[2].section_path == ["Top", "Middle"]


def test_sections_sibling_headings():
    doc = parse_document("# A\n\n## A.1\n\n## A.2\n\n# B\n\n## B.1\n")
    sections = [u for u in doc.units if u.unit_type.value in ("SECTION", "SUBSECTION")]
    by_text = {s.text: s.section_path for s in sections}
    assert by_text["A.2"] == ["A"]  # sibling to A.1, not nested under it
    assert by_text["B.1"] == ["B"]  # new h1 resets the stack


def test_sections_char_offsets():
    text = "intro\n\n# Heading Text\n\nbody\n"
    doc = parse_document(text)
    heading = doc.units[0]
    assert text[heading.char_start:heading.char_end].strip("\n") == "# Heading Text"


# extract_lists

def test_lists_flat():
    doc = parse_document("- one\n- two\n- three\n")
    lists = [u for u in doc.units if u.unit_type.value == "LIST"]
    items = [u for u in doc.units if u.unit_type.value == "LIST_ITEM"]
    assert len(lists) == 1
    assert [i.text for i in items] == ["one", "two", "three"]
    assert all(i.parent_id == lists[0].unit_id for i in items)


def test_lists_nested_parent():
    md = "- outer1\n  - inner1\n  - inner2\n- outer2\n"
    from markdown_it import MarkdownIt
    from mdit_py_plugins.gfm import gfm_plugin
    from structaudit.parsing.parser import _build_line_offsets

    tokens = MarkdownIt("commonmark").use(gfm_plugin).parse(md)
    units = extract_lists(tokens, _build_line_offsets(md))
    lists = [u for u in units if u.unit_type.value == "LIST"]
    items = [u for u in units if u.unit_type.value == "LIST_ITEM"]
    assert len(lists) == 2  # outer list + nested inner list
    outer_items = [i for i in items if i.parent_id == lists[0].unit_id]
    inner_items = [i for i in items if i.parent_id == lists[1].unit_id]
    assert {i.text for i in outer_items} == {"outer1", "outer2"}
    assert {i.text for i in inner_items} == {"inner1", "inner2"}


# extract_tables

def test_tables_headers_and_cells():
    md = "| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |\n"
    doc = parse_document(md)
    assert len(doc.tables) == 1
    table = doc.tables[0]
    assert table.column_headers == ["A", "B"]
    assert table.cells == [["1", "2"], ["3", "4"]]


def test_tables_caption():
    md = "### Table 4. Some Caption Here\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n"
    doc = parse_document(md)
    assert doc.tables[0].label == "Table 4"
    assert doc.tables[0].caption == "Table 4. Some Caption Here"


def test_tables_caption_survives_interruption():
    md = (
        "### Table 2. Split Table\n\n"
        "| A | B |\n| --- | --- |\n| 1 | 2 |\n\n"
        "# Footnotes\n(1) some footnote text\n\n"
        "| A | B |\n| --- | --- |\n| 3 | 4 |\n"
    )
    doc = parse_document(md)
    assert len(doc.tables) == 2
    assert doc.tables[0].label == "Table 2"
    assert doc.tables[1].label == "Table 2"  # must survive the Footnotes heading


def test_tables_no_caption():
    md = "| A | B |\n| --- | --- |\n| 1 | 2 |\n"
    doc = parse_document(md)
    assert doc.tables[0].label is None
    assert doc.tables[0].caption is None


def test_tables_char_span():
    md = "intro\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\noutro\n"
    doc = parse_document(md)
    table = doc.tables[0]
    span_text = md[table.char_start:table.char_end]
    assert "| A | B |" in span_text
    assert "outro" not in span_text


# extract_footnotes / extract_toc_entries / extract_reference_mentions

def test_footnotes_real_shape():
    text = "# Footnotes\n(1) First footnote text.\n(2) Second footnote text.\n"
    footnotes = extract_footnotes(text)
    assert len(footnotes) == 2
    assert footnotes[0].label == "FOOTNOTE:1"
    assert footnotes[0].text == "First footnote text."
    assert footnotes[1].label == "FOOTNOTE:2"


def test_footnotes_char_span():
    text = "before\n(1) footnote body here.\nafter"
    footnotes = extract_footnotes(text)
    fn = footnotes[0]
    assert text[fn.char_start:fn.char_end] == "(1) footnote body here."


def test_toc_entries_empty_without_toc():
    text = "# Heading\n\nJust ordinary prose with no dot-leader lines.\n"
    assert extract_toc_entries(text) == []


def test_notes_numbered():
    from structaudit.parsing.parser import extract_notes

    text = "Note 14: Some annotation text here.\nNote 15. Another one.\n"
    notes = extract_notes(text)
    assert len(notes) == 2
    assert notes[0].label == "NOTE:14"
    assert notes[0].text == "Some annotation text here."
    assert notes[1].label == "NOTE:15"


def test_reference_mentions_examples():
    text = "See footnote 49. Also Table 4 and Appendix B and Chapter V."
    mentions = extract_reference_mentions(text)
    kinds_and_ids = [(m.kind.value, m.identifier_raw) for m in mentions]
    assert ("FOOTNOTE", "49") in kinds_and_ids
    assert ("TABLE", "4") in kinds_and_ids
    assert ("APPENDIX", "B") in kinds_and_ids
    assert ("CHAPTER", "V") in kinds_and_ids


def test_reference_mentions_case_insensitive():
    mentions = extract_reference_mentions("(See table A.)")
    assert len(mentions) == 1
    assert mentions[0].kind.value == "TABLE"
    assert mentions[0].identifier_raw == "A"
    assert mentions[0].raw_text == "table A"


def test_parse_document_real_excerpt():
    text = (FIXTURE_DIR / "bls_ppi_excerpt.md").read_text()
    parsed = parse_document(text)

    assert len(parsed.units) > 0
    assert len(parsed.tables) > 0
    assert len(parsed.reference_mentions) > 0

    # At least one real data table should have gotten a caption.
    assert any(t.label == "Table A" for t in parsed.tables)

    # Every unit's char span should be internally consistent and within bounds.
    for unit in parsed.units:
        if unit.char_start is not None and unit.char_end is not None:
            assert 0 <= unit.char_start <= unit.char_end <= len(text)

    for table in parsed.tables:
        if table.char_start is not None and table.char_end is not None:
            assert 0 <= table.char_start <= table.char_end <= len(text)


def test_parse_document_units_sorted():
    text = (FIXTURE_DIR / "bls_ppi_excerpt.md").read_text()
    parsed = parse_document(text)
    starts = [u.char_start for u in parsed.units if u.char_start is not None]
    assert starts == sorted(starts)
