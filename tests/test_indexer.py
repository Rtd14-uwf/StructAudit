from structaudit.core import ReferenceKind
from structaudit.indexing.indexer import build_structural_indexes, normalize_identifier
from structaudit.parsing.parser import parse_document


# normalize_identifier

def test_normalize_identifier_examples():
    assert normalize_identifier(ReferenceKind.TABLE, "03") == "TABLE:3"
    assert normalize_identifier(ReferenceKind.CHAPTER, "V") == "CHAPTER:5"
    assert normalize_identifier(ReferenceKind.APPENDIX, "b") == "APPENDIX:B"
    assert normalize_identifier(ReferenceKind.FOOTNOTE, "049") == "FOOTNOTE:49"


def test_normalize_identifier_dotted():
    assert normalize_identifier(ReferenceKind.SECTION, "03.02") == "SECTION:3.2"


def test_normalize_identifier_no_leading_zero():
    assert normalize_identifier(ReferenceKind.TABLE, "4") == "TABLE:4"


def test_normalize_identifier_roman():
    assert normalize_identifier(ReferenceKind.CHAPTER, "XIV") == "CHAPTER:14"


def test_normalize_identifier_uppercase():
    assert normalize_identifier(ReferenceKind.APPENDIX, "B") == "APPENDIX:B"


# build_structural_indexes

def test_indexes_from_real_fixture(tmp_path):
    from pathlib import Path

    fixture = Path(__file__).resolve().parent / "fixtures" / "bls_ppi_excerpt.md"
    text = fixture.read_text()
    parsed = parse_document(text)
    indexes = build_structural_indexes(parsed)

    assert "TABLE:A" in indexes.table_number_to_unit
    assert "FOOTNOTE:1" in indexes.footnote_number_to_unit

    table_unit_id = indexes.table_number_to_unit["TABLE:A"]
    resolved = next(t for t in parsed.tables if t.unit_id == table_unit_id)
    assert resolved.label == "Table A"


def test_indexes_split_table_first_wins():
    md = (
        "### Table 2. Split Table\n\n"
        "| A | B |\n| --- | --- |\n| 1 | 2 |\n\n"
        "# Footnotes\n(1) some footnote text\n\n"
        "| A | B |\n| --- | --- |\n| 3 | 4 |\n"
    )
    parsed = parse_document(md)
    indexes = build_structural_indexes(parsed)
    first_table_unit_id = parsed.tables[0].unit_id
    assert indexes.table_number_to_unit["TABLE:2"] == first_table_unit_id


def test_indexes_chapter_appendix_section():
    md = "# Chapter V: Something\n\n## Appendix b: Notes\n\n### Section 3.2 Details\n"
    parsed = parse_document(md)
    indexes = build_structural_indexes(parsed)
    assert "CHAPTER:5" in indexes.chapter_number_to_unit
    assert "APPENDIX:B" in indexes.appendix_label_to_unit
    assert "SECTION:3.2" in indexes.section_number_to_unit


def test_indexes_notes():
    md = "Note 14: Some annotation.\n"
    parsed = parse_document(md)
    indexes = build_structural_indexes(parsed)
    assert "NOTE:14" in indexes.note_number_to_unit


def test_indexes_figures_charts_empty():
    parsed = parse_document("# Heading\n\nSome text with no figures.\n")
    indexes = build_structural_indexes(parsed)
    assert indexes.figure_number_to_unit == {}
    assert indexes.chart_number_to_unit == {}
    assert indexes.figures_by_order == []


def test_indexes_ordering_lists():
    md = "# One\n\n## Two\n\n- item a\n- item b\n\n| X | Y |\n| --- | --- |\n| 1 | 2 |\n"
    parsed = parse_document(md)
    indexes = build_structural_indexes(parsed)
    assert len(indexes.headings_by_order) == 2
    assert len(indexes.lists_by_order) == 1
    assert len(indexes.tables_by_order) == 1
