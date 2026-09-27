from pathlib import Path

import yaml

from structaudit.candidates.count_rules import extract_count_candidates
from structaudit.candidates.coverage_rules import extract_coverage_candidates
from structaudit.candidates.extract import extract_candidates
from structaudit.candidates.reference_rules import extract_reference_candidates
from structaudit.candidates.schema_rules import extract_schema_candidates
from structaudit.parsing.parser import extract_reference_mentions, parse_document

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "extraction_rules.yaml"

_GROUP_TO_EXTRACTOR = {
    "reference_explicit_pointer": lambda text: extract_reference_candidates(text, extract_reference_mentions(text)),
    "reference_target_topic_match": lambda text: extract_reference_candidates(text, extract_reference_mentions(text)),
    "count_n_noun_shown_in_target": extract_count_candidates,
    "coverage_members_relation_target": extract_coverage_candidates,
    "schema_target_contains_columns": extract_schema_candidates,
}


def _load_fixtures() -> dict:
    with FIXTURE_PATH.open() as f:
        return yaml.safe_load(f)


def _run_fixture_case(group: str, case: dict):
    extractor = _GROUP_TO_EXTRACTOR[group]
    candidates = extractor(case["text"])

    if case["expected_candidate_type"] is None:
        assert candidates == [], f"[{group}] expected no candidates for {case['text']!r}, got {candidates}"
        return

    assert len(candidates) >= 1, f"[{group}] expected a candidate for {case['text']!r}, got none"
    candidate = candidates[0]
    assert candidate.candidate_type.value == case["expected_candidate_type"]

    if "expected_target" in case:
        assert candidate.target_query == case["expected_target"]
    if "expected_count" in case:
        assert candidate.expected_count == case["expected_count"]
    if "expected_members" in case:
        assert candidate.expected_members == case["expected_members"]
    if "expected_schema" in case:
        assert candidate.expected_schema == case["expected_schema"]
    if "expected_operations" in case:
        assert [op.value for op in candidate.operations] == case["expected_operations"]


def test_rule_fixtures():
    fixtures = _load_fixtures()
    total_cases = 0
    for group, cases in fixtures.items():
        for case in cases:
            _run_fixture_case(group, case)
            total_cases += 1
    assert total_cases == 25


def test_count_following_and_below_forms():
    c = extract_count_candidates("The following four risks were identified:")
    assert len(c) == 1 and c[0].expected_count == 4 and c[0].target_query == "following list"

    c = extract_count_candidates("The four risks below are significant.")
    assert len(c) == 1 and c[0].expected_count == 4


def test_coverage_both_and_following_forms():
    c = extract_coverage_candidates("Both revenues and expenses are summarized below.")
    assert len(c) == 1 and set(c[0].expected_members) == {"revenues", "expenses"}

    c = extract_coverage_candidates("The following categories are included in Table 6:\nA, B, and C.")
    assert len(c) == 1 and c[0].expected_members == ["A", "B", "C"]


def test_schema_rows_below_form():
    c = extract_schema_candidates("The rows below represent Current, Long-Term, and Total debt.")
    assert len(c) == 1
    assert c[0].expected_schema == ["Current", "Long-Term", "Total"]
    assert c[0].target_query == "following table"


def test_extract_candidates_real_excerpt():
    fixture = Path(__file__).resolve().parent / "fixtures" / "bls_ppi_excerpt.md"
    parsed = parse_document(fixture.read_text())
    candidates = extract_candidates(parsed, document_id="ppi.md")
    assert len(candidates) > 0
    assert all(c.document_id == "ppi.md" for c in candidates)
    assert all(c.candidate_id for c in candidates)
