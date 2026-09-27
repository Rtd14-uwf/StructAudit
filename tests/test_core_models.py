import sys

import pytest
from pydantic import ValidationError

from structaudit.core import (
    CandidateType,
    DocumentUnit,
    OperationType,
    Status,
    StructuralCandidate,
    TableUnit,
    UnitType,
    VerificationResult,
)


def test_unit_type_values():
    expected = {
        "DOCUMENT", "TOC_ENTRY", "SECTION", "SUBSECTION", "PARAGRAPH", "SENTENCE",
        "LIST", "LIST_ITEM", "TABLE", "TABLE_ROW", "TABLE_COLUMN", "TABLE_CELL",
        "FIGURE", "CAPTION", "FOOTNOTE", "NOTE", "APPENDIX",
    }
    assert {m.value for m in UnitType} == expected


def test_operation_type_values():
    expected = {
        "TARGET_EXISTS", "STRUCTURAL_ALIGNMENT", "TARGET_LABEL_MATCH", "TARGET_TOPIC_MATCH",
        "COUNT_MATCH", "COVERAGE_SET_MATCH", "SCHEMA_HEADER_MATCH", "SCHEMA_ROW_MATCH",
    }
    assert {m.value for m in OperationType} == expected


def test_status_values():
    assert {m.value for m in Status} == {"SATISFIED", "VIOLATED", "UNCERTAIN", "UNRESOLVED"}


# DocumentUnit

def test_document_unit_minimal():
    unit = DocumentUnit(unit_id="U1", unit_type=UnitType.PARAGRAPH, order=0, text="Hello.")
    assert unit.parse_confidence == 1.0
    assert unit.section_path == []


def test_document_unit_bad_type():
    with pytest.raises(ValidationError):
        DocumentUnit(unit_id="U1", unit_type="NOT_A_REAL_TYPE", order=0, text="x")


def test_document_unit_bad_span():
    with pytest.raises(ValidationError):
        DocumentUnit(unit_id="U1", unit_type=UnitType.PARAGRAPH, order=0, text="x",
                      char_start=10, char_end=5)


def test_document_unit_self_parent():
    with pytest.raises(ValidationError):
        DocumentUnit(unit_id="U1", unit_type=UnitType.PARAGRAPH, order=0, text="x", parent_id="U1")


def test_document_unit_bad_confidence():
    with pytest.raises(ValidationError):
        DocumentUnit(unit_id="U1", unit_type=UnitType.PARAGRAPH, order=0, text="x",
                      parse_confidence=1.5)


def test_document_unit_json_roundtrip():
    unit = DocumentUnit(
        unit_id="U1", unit_type=UnitType.TABLE, order=3, text="Table 4",
        label="Table 4", section_path=["Results", "3.2"],
    )
    payload = unit.model_dump_json()
    restored = DocumentUnit.model_validate_json(payload)
    assert restored == unit
    assert '"unit_type":"TABLE"' in payload


# TableUnit

def test_table_unit_rectangular():
    table = TableUnit(
        unit_id="T4", order=1, label="Table 4",
        column_headers=["Revenue", "Expenses"],
        cells=[["100", "50"], ["200", "80"]],
    )
    assert len(table.cells) == 2


def test_table_unit_jagged():
    with pytest.raises(ValidationError):
        TableUnit(unit_id="T4", order=1, cells=[["100", "50"], ["200"]])


def test_table_unit_no_cells():
    table = TableUnit(unit_id="T4", order=1)
    assert table.cells == []


# StructuralCandidate

def _base_candidate_kwargs(**overrides):
    kwargs = dict(
        candidate_id="C1", document_id="D1", anchor_unit_id="U1",
        anchor_text="See footnote 49.", candidate_type=CandidateType.REFERENCE,
        target_query="footnote 49", operations=[OperationType.TARGET_EXISTS],
        rule_id="ref_explicit_pointer", rule_confidence=0.9, parse_confidence=1.0,
    )
    kwargs.update(overrides)
    return kwargs


def test_reference_candidate():
    candidate = StructuralCandidate(**_base_candidate_kwargs())
    assert candidate.candidate_type == CandidateType.REFERENCE


def test_reference_needs_target():
    with pytest.raises(ValidationError):
        StructuralCandidate(**_base_candidate_kwargs(target_query=None))


def test_count_needs_expected_count():
    kwargs = _base_candidate_kwargs(
        candidate_type=CandidateType.COUNT, target_query=None,
        operations=[OperationType.COUNT_MATCH],
    )
    with pytest.raises(ValidationError):
        StructuralCandidate(**kwargs)


def test_count_with_expected_count():
    kwargs = _base_candidate_kwargs(
        candidate_type=CandidateType.COUNT, target_query="following list",
        expected_count=4, operations=[OperationType.COUNT_MATCH],
    )
    candidate = StructuralCandidate(**kwargs)
    assert candidate.expected_count == 4


def test_coverage_needs_members():
    kwargs = _base_candidate_kwargs(
        candidate_type=CandidateType.COVERAGE, target_query="Table 4",
        operations=[OperationType.TARGET_EXISTS, OperationType.COVERAGE_SET_MATCH],
    )
    with pytest.raises(ValidationError):
        StructuralCandidate(**kwargs)


def test_schema_needs_schema():
    kwargs = _base_candidate_kwargs(
        candidate_type=CandidateType.SCHEMA, target_query="Table 4",
        operations=[OperationType.TARGET_EXISTS, OperationType.SCHEMA_HEADER_MATCH],
    )
    with pytest.raises(ValidationError):
        StructuralCandidate(**kwargs)


def test_count_implicit_target_ok():
    kwargs = _base_candidate_kwargs(
        candidate_type=CandidateType.COUNT, target_query="following list",
        expected_count=4, operations=[OperationType.COUNT_MATCH],
    )
    candidate = StructuralCandidate(**kwargs)
    assert OperationType.TARGET_EXISTS not in candidate.operations


def test_operations_bad_value():
    with pytest.raises(ValidationError):
        StructuralCandidate(**_base_candidate_kwargs(operations=["NOT_A_REAL_OP"]))


def test_rank_score_bounds():
    candidate = StructuralCandidate(**_base_candidate_kwargs())
    assert candidate.rank_score == 0.0
    with pytest.raises(ValidationError):
        StructuralCandidate(**_base_candidate_kwargs(rank_score=-0.1))


# VerificationResult

def test_verification_result_satisfied():
    result = VerificationResult(
        candidate_id="C1", status=Status.SATISFIED,
        operations_run=[OperationType.TARGET_EXISTS],
        expected={}, observed={}, evidence_unit_ids=["T4"], evidence_text=["Table 4"],
        deterministic=True, confidence=1.0, explanation="Target found.",
    )
    assert result.status == Status.SATISFIED


def test_unresolved_rejects_evidence():
    with pytest.raises(ValidationError):
        VerificationResult(
            candidate_id="C1", status=Status.UNRESOLVED,
            operations_run=[OperationType.TARGET_EXISTS],
            expected={}, observed={}, evidence_unit_ids=["T4"], evidence_text=["Table 4"],
            deterministic=False, confidence=0.0, explanation="Could not resolve target.",
        )


def test_verification_bad_confidence():
    with pytest.raises(ValidationError):
        VerificationResult(
            candidate_id="C1", status=Status.VIOLATED,
            operations_run=[OperationType.TARGET_EXISTS],
            expected={}, observed={}, evidence_unit_ids=[], evidence_text=[],
            deterministic=True, confidence=1.5, explanation="x",
        )


def test_verification_result_json_roundtrip():
    result = VerificationResult(
        candidate_id="C1", status=Status.VIOLATED,
        operations_run=[OperationType.COVERAGE_SET_MATCH],
        expected={"members": ["BERT", "T5", "RoBERTa"]},
        observed={"members": ["BERT", "RoBERTa"]},
        evidence_unit_ids=["T4"], evidence_text=["Table 4 shows BERT, RoBERTa."],
        deterministic=True, confidence=0.96, explanation="Table 4 omits T5.",
    )
    payload = result.model_dump_json()
    restored = VerificationResult.model_validate_json(payload)
    assert restored == result
