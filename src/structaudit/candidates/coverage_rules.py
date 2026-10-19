"""
COVERAGE candidate rules.

Requires an explicit coordinated member set ("A, B, and C"), a
reporting/containment verb, and a bounded target. Each rule finds the verb
phrase first then parses the list on its side of it with parse_member_list()
"""

from __future__ import annotations

import re

from structaudit.candidates._members import list_text_after, list_text_before, parse_member_list
from structaudit.candidates._shared import TARGET_GROUP, NonOverlappingMatches
from structaudit.core import CandidateType, OperationType, StructuralCandidate

_RELATION_PASSIVE = r"(?:shown|reported|listed|included|summarized|presented|contained|provided)"
_RELATION_ACTIVE = r"(?:shows|reports|lists|includes|summarizes|presents|contains|provides)"

# "... are shown in Table 4"  (the list is before it)
_PASSIVE_ANCHOR = re.compile(r"\b(?:are|is)\s+" + _RELATION_PASSIVE + r"\s+in\s+" + TARGET_GROUP, re.IGNORECASE)
# "Table 4 reports ..."  (the list is after it)
_ACTIVE_ANCHOR = re.compile(r"\b" + TARGET_GROUP + r"\s+" + _RELATION_ACTIVE + r"\s+", re.IGNORECASE)
# "The following categories are included in Table 6:\nA, B, and C."  (the list is after it)
_FOLLOWING_ANCHOR = re.compile(
    r"\bthe following \w+\s+(?:are|is)\s+" + _RELATION_PASSIVE + r"\s+in\s+" + TARGET_GROUP, re.IGNORECASE
)
# "Both revenues and expenses are summarized below."
_BOTH_PATTERN = re.compile(
    r"\bboth\s+(\w+)\s+and\s+(\w+)\s+(?:are|is)\s+" + _RELATION_PASSIVE + r"\s+below\b", re.IGNORECASE
)


def extract_coverage_candidates(text: str, document_id: str = "") -> list[StructuralCandidate]:
    candidates = []
    order = 0
    claimed = NonOverlappingMatches()

    def _add(span: tuple[int, int], members: list[str], target_query: str, rule_id: str) -> None:
        nonlocal order
        if not claimed.try_claim(*span):
            return
        operations = [OperationType.COVERAGE_SET_MATCH]
        if target_query != "following list":
            operations.insert(0, OperationType.TARGET_EXISTS)
        candidates.append(
            StructuralCandidate(
                candidate_id=f"coverage-{order}",
                document_id=document_id,
                anchor_unit_id="",
                anchor_text=text[span[0]:span[1]],
                anchor_char_start=span[0],
                anchor_char_end=span[1],
                candidate_type=CandidateType.COVERAGE,
                target_query=target_query,
                expected_members=members,
                operations=operations,
                rule_id=rule_id,
                rule_confidence=0.8,
                parse_confidence=1.0,
            )
        )
        order += 1

    for match in _FOLLOWING_ANCHOR.finditer(text):
        list_end, list_text = list_text_after(text, match.end())
        members = parse_member_list(list_text, trim="trailing")
        if members:
            _add((match.start(), list_end), members, f"{match.group(1)} {match.group(2)}",
                 "coverage_following_relation_target_then_list")

    for match in _PASSIVE_ANCHOR.finditer(text):
        list_start, list_text = list_text_before(text, match.start())
        members = parse_member_list(list_text, trim="leading")
        if members:
            _add((list_start, match.end()), members, f"{match.group(1)} {match.group(2)}",
                 "coverage_members_relation_target")

    for match in _ACTIVE_ANCHOR.finditer(text):
        list_end, list_text = list_text_after(text, match.end())
        members = parse_member_list(list_text, trim="trailing")
        if members:
            _add((match.start(), list_end), members, f"{match.group(1)} {match.group(2)}",
                 "coverage_target_relation_members")

    for match in _BOTH_PATTERN.finditer(text):
        members = parse_member_list(f"{match.group(1)} and {match.group(2)}", trim="leading")
        if members:
            _add(match.span(), members, "following list", "coverage_both_members_relation_below")

    return candidates