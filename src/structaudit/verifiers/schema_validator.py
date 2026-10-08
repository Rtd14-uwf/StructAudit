"""
SCHEMA Validator: checks that a table's column headers (SCHEMA_HEADER_MATCH)
or row labels (SCHEMA_ROW_MATCH) contain what a SCHEMA candidate's
expected_schema says they should.

Logic:
  - every expected label matched                        -> SATISFIED
  - at least one expected label definitively not found   -> VIOLATED
  - otherwise, something couldn't be decided             -> UNCERTAIN
"""

from __future__ import annotations

from structaudit.core import (
    OperationType,
    ParsedDocument,
    Status,
    StructuralCandidate,
    TableUnit,
    TargetResolution,
    VerificationResult,
)
from structaudit.models.semantic_backend import SemanticBackend
from structaudit.verifiers._matching import match_labels, token_key

_SEMANTIC_TASK = "schema_label_equivalence"


def _row_labels(table: TableUnit) -> list[str]:
    return table.row_headers or [row[0] for row in table.cells if row]


def _validate_schema(
    candidate: StructuralCandidate,
    resolution: TargetResolution,
    parsed: ParsedDocument,
    backend: SemanticBackend,
    operation: OperationType,
    observed_labels_of,
    label_kind: str,
    guard_second_header_row: bool,
) -> VerificationResult:
    expected = candidate.expected_schema

    if resolution.unit_id is None:
        return VerificationResult(
            candidate_id=candidate.candidate_id, status=Status.UNRESOLVED,
            operations_run=[operation],
            expected={"expected_schema": expected}, observed={},
            evidence_unit_ids=[], evidence_text=[],
            deterministic=True, confidence=1.0,
            explanation="No resolved target to check the schema of.",
        )

    table = next((t for t in parsed.tables if t.unit_id == resolution.unit_id), None)
    if table is None:
        return VerificationResult(
            candidate_id=candidate.candidate_id, status=Status.UNCERTAIN,
            operations_run=[operation],
            expected={"expected_schema": expected}, observed={"unit_id": resolution.unit_id},
            evidence_unit_ids=[resolution.unit_id], evidence_text=[],
            deterministic=False, confidence=0.0,
            explanation=f"Resolved unit {resolution.unit_id!r} is not a table.",
        )

    observed_labels = observed_labels_of(table)
    match = match_labels(expected, observed_labels, backend, _SEMANTIC_TASK)

    possible_second_header_row: list[str] = []
    if guard_second_header_row and table.cells:
        first_body_row_keys = {token_key(cell) for cell in table.cells[0]}
        possible_second_header_row = [label for label in match.missing if token_key(label) in first_body_row_keys]

    definite_missing = [label for label in match.missing if label not in possible_second_header_row]
    undecided = match.uncertain + possible_second_header_row

    if definite_missing:
        status = Status.VIOLATED
    elif undecided:
        status = Status.UNCERTAIN
    else:
        status = Status.SATISFIED

    deterministic = match.semantic_calls == 0 and status != Status.UNCERTAIN
    if status == Status.UNCERTAIN:
        confidence = 0.0
    else:
        confidence = min(match.semantic_confidences, default=1.0)

    if status == Status.SATISFIED:
        explanation = f"All {len(expected)} expected {label_kind} found."
    elif status == Status.VIOLATED:
        explanation = f"Missing from {label_kind}: {definite_missing}."
    elif possible_second_header_row:
        explanation = (
            f"{possible_second_header_row} not in the header row but present in the first body row, "
            "which may be a second header row."
        )
    else:
        explanation = f"Could not decide whether {undecided} match any of the {label_kind}."

    observed = {
        "matched_schema": list(match.matched),
        "missing_schema": definite_missing,
        "extra_schema": list(dict.fromkeys(match.extra)),
        "match_methods": match.methods,
    }
    if undecided:
        observed["uncertain_schema"] = undecided
    if possible_second_header_row:
        observed["possible_second_header_row"] = possible_second_header_row

    return VerificationResult(
        candidate_id=candidate.candidate_id, status=status,
        operations_run=[operation],
        expected={"expected_schema": expected},
        observed=observed,
        evidence_unit_ids=[table.unit_id], evidence_text=observed_labels,
        deterministic=deterministic, confidence=confidence,
        explanation=explanation,
    )


def validate_schema_header_match(
    candidate: StructuralCandidate, resolution: TargetResolution, parsed: ParsedDocument, backend: SemanticBackend
) -> VerificationResult:
    """SCHEMA_HEADER_MATCH: expected_schema vs the table's column headers."""
    return _validate_schema(
        candidate, resolution, parsed, backend,
        OperationType.SCHEMA_HEADER_MATCH, lambda t: t.column_headers, "column headers",
        guard_second_header_row=True,
    )


def validate_schema_row_match(
    candidate: StructuralCandidate, resolution: TargetResolution, parsed: ParsedDocument, backend: SemanticBackend
) -> VerificationResult:
    """SCHEMA_ROW_MATCH: expected_schema vs the table's row labels."""
    return _validate_schema(
        candidate, resolution, parsed, backend,
        OperationType.SCHEMA_ROW_MATCH, _row_labels, "row labels",
        guard_second_header_row=False,
    )
