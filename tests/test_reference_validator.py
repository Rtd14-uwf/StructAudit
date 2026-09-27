from structaudit.core import (
    CandidateType,
    OperationType,
    ParsedDocument,
    ResolutionMethod,
    Status,
    StructuralCandidate,
    TargetResolution,
)
from structaudit.models.semantic_backend import LexicalOverlapBackend
from structaudit.verifiers.reference_validator import (
    validate_structural_alignment,
    validate_target_exists,
    validate_target_label_match,
    validate_target_topic_match,
)


def _reference_candidate(target_query: str = "Table 4") -> StructuralCandidate:
    return StructuralCandidate(
        candidate_id="c1", document_id="d1", anchor_unit_id="", anchor_text=target_query,
        candidate_type=CandidateType.REFERENCE, target_query=target_query,
        operations=[OperationType.TARGET_EXISTS], rule_id="r", rule_confidence=0.9, parse_confidence=1.0,
    )


def test_target_exists_satisfied_when_resolved():
    candidate = _reference_candidate("Table 4")
    resolution = TargetResolution(unit_id="table-0", method=ResolutionMethod.NORMALIZED, confidence=1.0)
    result = validate_target_exists(candidate, resolution)
    assert result.status == Status.SATISFIED
    assert result.deterministic is True
    assert result.confidence == 1.0
    assert result.evidence_unit_ids == ["table-0"]
    assert OperationType.TARGET_EXISTS in result.operations_run


def test_target_exists_violated_when_unresolved():
    candidate = _reference_candidate("Table 99")
    resolution = TargetResolution(unit_id=None, method=ResolutionMethod.UNRESOLVED, confidence=0.0)
    result = validate_target_exists(candidate, resolution)
    assert result.status == Status.VIOLATED  # not UNRESOLVED -- see module docstring
    assert result.deterministic is True
    assert result.evidence_unit_ids == []


def test_target_exists_violated_case_carries_no_evidence_units():
    candidate = _reference_candidate("Appendix Z")
    resolution = TargetResolution(unit_id=None, method=ResolutionMethod.UNRESOLVED, confidence=0.0)
    result = validate_target_exists(candidate, resolution)
    assert result.evidence_unit_ids == []
    assert result.evidence_text == []


# STRUCTURAL_ALIGNMENT

def test_structural_alignment_high_similarity_is_deterministic_satisfied():
    backend = LexicalOverlapBackend()
    result = validate_structural_alignment("c1", "Chapter 5: Introduction", "Chapter 5: Introduction", backend)
    assert result.status == Status.SATISFIED
    assert result.deterministic is True
    assert len(backend.call_log) == 0  # no semantic call needed for a clear match


def test_structural_alignment_low_similarity_is_deterministic_violated():
    backend = LexicalOverlapBackend()
    result = validate_structural_alignment("c1", "Chapter 5: Introduction", "Chapter 9: Conclusion", backend)
    assert result.status == Status.VIOLATED
    assert result.deterministic is True
    assert len(backend.call_log) == 0


def test_structural_alignment_ambiguous_band_falls_back_to_semantic_check():
    backend = LexicalOverlapBackend()
    result = validate_structural_alignment("c1", "Chapter V: Introduction", "Chapter 5: Introduction", backend)
    assert 70 <= result.observed["similarity"] < 95
    assert result.deterministic is False
    assert len(backend.call_log) == 1


# TARGET_LABEL_MATCH


def _label_match_candidate_and_resolution(target_text, table_caption):
    from structaudit.candidates.reference_rules import extract_reference_candidates
    from structaudit.indexing.indexer import build_structural_indexes
    from structaudit.parsing.parser import extract_reference_mentions, parse_document
    from structaudit.resolution.resolver import resolve_target

    text = f"{target_text}\n\n### Table 7. {table_caption}\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n"
    parsed = parse_document(text)
    indexes = build_structural_indexes(parsed)
    mentions = extract_reference_mentions(text)
    candidate = next(c for c in extract_reference_candidates(text, mentions) if c.expected_topic)
    resolution = resolve_target(candidate, parsed, indexes)
    return candidate, resolution, parsed


def test_target_label_match_satisfied_when_caption_extends_expected_identity():
    candidate, resolution, parsed = _label_match_candidate_and_resolution(
        "Refinancing defaults are reported in Table 7.", "Refinancing Defaults by Quarter"
    )
    result = validate_target_label_match(candidate, resolution, parsed)
    assert result.status == Status.SATISFIED


def test_target_label_match_violated_when_caption_is_unrelated():
    candidate, resolution, parsed = _label_match_candidate_and_resolution(
        "Refinancing defaults are reported in Table 7.", "Employee Headcount by Region"
    )
    result = validate_target_label_match(candidate, resolution, parsed)
    assert result.status == Status.VIOLATED


def test_target_label_match_unresolved_when_no_expected_topic():
    candidate = _reference_candidate("Table 4")  # bare reference, no expected_topic
    resolution = TargetResolution(unit_id="table-0", method=ResolutionMethod.NORMALIZED, confidence=1.0)
    result = validate_target_label_match(candidate, resolution, ParsedDocument(raw_text=""))
    assert result.status == Status.UNRESOLVED


# TARGET_TOPIC_MATCH

def test_target_topic_match_satisfied_for_matching_table():
    from structaudit.candidates.reference_rules import extract_reference_candidates
    from structaudit.indexing.indexer import build_structural_indexes
    from structaudit.parsing.parser import extract_reference_mentions, parse_document
    from structaudit.resolution.resolver import resolve_target

    text = "Refinancing defaults are reported in Table 7.\n\n### Table 7. Refinancing Defaults by Quarter\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n"
    parsed = parse_document(text)
    indexes = build_structural_indexes(parsed)
    mentions = extract_reference_mentions(text)
    candidate = next(c for c in extract_reference_candidates(text, mentions) if c.expected_topic)
    resolution = resolve_target(candidate, parsed, indexes)
    result = validate_target_topic_match(candidate, resolution, parsed, LexicalOverlapBackend())
    assert result.status == Status.SATISFIED
    assert result.deterministic is False  # semantic checks are never deterministic


def test_target_topic_match_violated_for_mismatched_table():
    from structaudit.candidates.reference_rules import extract_reference_candidates
    from structaudit.indexing.indexer import build_structural_indexes
    from structaudit.parsing.parser import extract_reference_mentions, parse_document
    from structaudit.resolution.resolver import resolve_target

    text = "Refinancing defaults are reported in Table 7.\n\n### Table 7. Employee Headcount by Region\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n"
    parsed = parse_document(text)
    indexes = build_structural_indexes(parsed)
    mentions = extract_reference_mentions(text)
    candidate = next(c for c in extract_reference_candidates(text, mentions) if c.expected_topic)
    resolution = resolve_target(candidate, parsed, indexes)
    result = validate_target_topic_match(candidate, resolution, parsed, LexicalOverlapBackend())
    assert result.status == Status.VIOLATED


def test_target_topic_match_unresolved_when_target_not_found():
    candidate = _reference_candidate("Table 99")
    candidate.expected_topic = "some topic"
    resolution = TargetResolution(unit_id=None, method=ResolutionMethod.UNRESOLVED, confidence=0.0)
    result = validate_target_topic_match(candidate, resolution, ParsedDocument(raw_text=""), LexicalOverlapBackend())
    assert result.status == Status.UNRESOLVED


def test_reference_validator_runs_on_real_document_excerpt():
    from pathlib import Path

    from structaudit.candidates.extract import extract_candidates
    from structaudit.indexing.indexer import build_structural_indexes
    from structaudit.parsing.parser import parse_document
    from structaudit.resolution.resolver import resolve_target

    fixture = Path(__file__).resolve().parent / "fixtures" / "bls_ppi_excerpt.md"
    parsed = parse_document(fixture.read_text())
    indexes = build_structural_indexes(parsed)
    candidates = extract_candidates(parsed, document_id="ppi.md")
    backend = LexicalOverlapBackend()

    for candidate in candidates:
        if candidate.candidate_type != CandidateType.REFERENCE:
            continue
        resolution = resolve_target(candidate, parsed, indexes)
        validate_target_exists(candidate, resolution)
        if candidate.expected_topic:
            validate_target_label_match(candidate, resolution, parsed)
            validate_target_topic_match(candidate, resolution, parsed, backend)
