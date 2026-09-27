"""
SCHEMA candidate rules.

Requires the anchor to name structural fields via a trigger noun
(column/row/header/field/category) AND explicitly assert them as headers
or rows. Members named without that assertion belong to COVERAGE instead
"""

from __future__ import annotations

import re

from structaudit.candidates._shared import MEMBER_LIST, TARGET_GROUP, split_members
from structaudit.core import CandidateType, OperationType, StructuralCandidate

# "Table 4 contains columns Revenue, Expenses, and Net Income."
_PATTERN_TARGET_CONTAINS_COLUMNS = re.compile(
    TARGET_GROUP + r"\s+contains?\s+columns?\s+" + MEMBER_LIST, re.IGNORECASE
)
# "The rows below represent Current, Long-Term, and Total debt." The
# trailing common noun on the last member ("debt") is stripped below.
_PATTERN_ROWS_BELOW_REPRESENT = re.compile(
    r"\bthe rows below represents?\s+" + MEMBER_LIST, re.IGNORECASE
)


def extract_schema_candidates(text: str, document_id: str = "") -> list[StructuralCandidate]:
    candidates = []
    order = 0

    for match in _PATTERN_TARGET_CONTAINS_COLUMNS.finditer(text):
        target = f"{match.group(1)} {match.group(2)}"
        members = split_members(match.group(3))
        candidates.append(
            StructuralCandidate(
                candidate_id=f"schema-{order}",
                document_id=document_id,
                anchor_unit_id="",
                anchor_text=match.group(0),
                candidate_type=CandidateType.SCHEMA,
                target_query=target,
                expected_schema=members,
                operations=[OperationType.TARGET_EXISTS, OperationType.SCHEMA_HEADER_MATCH],
                rule_id="schema_target_contains_columns",
                rule_confidence=0.85,
                parse_confidence=1.0,
            )
        )
        order += 1

    for match in _PATTERN_ROWS_BELOW_REPRESENT.finditer(text):
        raw_members = split_members(match.group(1))
        # Assume the last member's trailing word is a shared noun ("debt")
        if raw_members and len(raw_members[-1].split()) > 1:
            raw_members[-1] = " ".join(raw_members[-1].split()[:-1])

        candidates.append(
            StructuralCandidate(
                candidate_id=f"schema-{order}",
                document_id=document_id,
                anchor_unit_id="",
                anchor_text=match.group(0),
                candidate_type=CandidateType.SCHEMA,
                target_query="following table",
                expected_schema=raw_members,
                operations=[OperationType.SCHEMA_ROW_MATCH],
                rule_id="schema_rows_below_represent",
                rule_confidence=0.75,
                parse_confidence=1.0,
            )
        )
        order += 1

    return candidates
