"""
COUNT Validator: compares a COUNT candidate's expected_count against what's
actually at the resolved target.

The counting unit (list items, table rows, table columns, or an inline
enumerated list) is determined from the anchor's noun when explicit
("rows" -> table rows, "columns" -> table columns), falling back to
whatever the resolved target ("risks"/"items"/"cases" -> list
items if the target is a list, or table rows if the target is a table).
"""

from __future__ import annotations

import re

from structaudit.candidates._shared import NUMBER, split_members
from structaudit.core import (
    OperationType,
    ParsedDocument,
    Status,
    StructuralCandidate,
    TableUnit,
    TargetResolution,
    UnitType,
    VerificationResult,
)

_ANCHOR_NOUN_RE = re.compile(NUMBER + r"\s+(\w+)", re.IGNORECASE)

_ROW_NOUNS = {"row", "rows"}
_COLUMN_NOUNS = {"column", "columns"}


def _find_unit_by_id(unit_id: str, parsed: ParsedDocument):
    for table in parsed.tables:
        if table.unit_id == unit_id:
            return table
    for unit in parsed.units:
        if unit.unit_id == unit_id:
            return unit
    return None


def _extract_anchor_noun(anchor_text: str) -> str | None:
    """Pull the counted noun out of anchor_text, e.g. "all six regions" -> "regions"."""
    match = _ANCHOR_NOUN_RE.search(anchor_text)
    return match.group(2).lower() if match else None


def count_list_items(list_unit_id: str, parsed: ParsedDocument) -> int:
    """Count LIST_ITEM units whose parent_id is the given list's unit_id."""
    return sum(1 for u in parsed.units if u.unit_type == UnitType.LIST_ITEM and u.parent_id == list_unit_id)


def count_table_rows(table: TableUnit) -> int:
    return len(table.cells)


def count_table_columns(table: TableUnit) -> int:
    if table.column_headers:
        return len(table.column_headers)
    if table.cells:
        return len(table.cells[0])  # TableUnit's own validator guarantees cells is rectangular
    return 0


def count_enumerated_members(text: str) -> int:
    """Count members in an inline comma/"and"-coordinated list within text."""
    return len(split_members(text))


def validate_count_match(
    candidate: StructuralCandidate, resolution: TargetResolution, parsed: ParsedDocument
) -> VerificationResult:
    if resolution.unit_id is None:
        return VerificationResult(
            candidate_id=candidate.candidate_id, status=Status.UNRESOLVED,
            operations_run=[OperationType.COUNT_MATCH],
            expected={"expected_count": candidate.expected_count}, observed={},
            evidence_unit_ids=[], evidence_text=[],
            deterministic=False, confidence=0.0,
            explanation="No resolved target to count against.",
        )

    target = _find_unit_by_id(resolution.unit_id, parsed)
    if target is None:
        return VerificationResult(
            candidate_id=candidate.candidate_id, status=Status.UNCERTAIN,
            operations_run=[OperationType.COUNT_MATCH],
            expected={"expected_count": candidate.expected_count}, observed={"unit_id": resolution.unit_id},
            evidence_unit_ids=[resolution.unit_id], evidence_text=[],
            deterministic=False, confidence=0.0,
            explanation=f"Resolved unit {resolution.unit_id!r} not found in document.",
        )

    noun = _extract_anchor_noun(candidate.anchor_text)
    observed_count: int | None = None
    counting_unit = None

    if noun in _ROW_NOUNS:
        if isinstance(target, TableUnit):
            observed_count = count_table_rows(target)
            counting_unit = "table_rows"
    elif noun in _COLUMN_NOUNS:
        if isinstance(target, TableUnit):
            observed_count = count_table_columns(target)
            counting_unit = "table_columns"
    else:
        if isinstance(target, TableUnit):
            observed_count = count_table_rows(target)
            counting_unit = "table_rows"
        elif target.unit_type == UnitType.LIST:
            observed_count = count_list_items(target.unit_id, parsed)
            counting_unit = "list_items"
        elif getattr(target, "text", None):
            observed_count = count_enumerated_members(target.text)
            counting_unit = "enumerated_members"

    if observed_count is None or counting_unit is None:
        return VerificationResult(
            candidate_id=candidate.candidate_id, status=Status.UNCERTAIN,
            operations_run=[OperationType.COUNT_MATCH],
            expected={"expected_count": candidate.expected_count, "anchor_noun": noun},
            observed={"unit_id": resolution.unit_id},
            evidence_unit_ids=[resolution.unit_id], evidence_text=[],
            deterministic=False, confidence=0.0,
            explanation=f"Could not determine a counting unit for anchor noun {noun!r} against {resolution.unit_id!r}.",
        )

    status = Status.SATISFIED if observed_count == candidate.expected_count else Status.VIOLATED
    return VerificationResult(
        candidate_id=candidate.candidate_id, status=status,
        operations_run=[OperationType.COUNT_MATCH],
        expected={"expected_count": candidate.expected_count, "counting_unit": counting_unit},
        observed={"observed_count": observed_count, "unit_id": resolution.unit_id},
        evidence_unit_ids=[resolution.unit_id], evidence_text=[],
        deterministic=True, confidence=1.0,
        explanation=f"Expected {candidate.expected_count} ({counting_unit}), observed {observed_count}.",
    )
