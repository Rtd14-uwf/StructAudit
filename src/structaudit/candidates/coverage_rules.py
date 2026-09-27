"""
COVERAGE candidate rules.

Requires an explicit member set, a reporting/containment verb, and a
bounded target. Members are extracted via comma/"and" splitting rather
than a real dependency parse, so only flat, explicit lists are supported
("A, B, and C")
"""

from __future__ import annotations

import re

from structaudit.candidates._shared import MEMBER_LIST, TARGET_GROUP, NonOverlappingMatches, split_members
from structaudit.core import CandidateType, OperationType, StructuralCandidate

_RELATION_PASSIVE = r"(?:shown|reported|listed|included|summarized|presented|contained|provided)"
_RELATION_ACTIVE = r"(?:shows|reports|lists|includes|summarizes|presents|contains|provides)"

# "BERT, T5, and RoBERTa are shown in Table 4."
_PATTERN_MEMBERS_RELATION_TARGET = re.compile(
    MEMBER_LIST + r"\s+(?:are|is)\s+" + _RELATION_PASSIVE + r"\s+in\s+" + TARGET_GROUP, re.IGNORECASE
)
# "Table 4 reports BERT, T5, and RoBERTa."
_PATTERN_TARGET_RELATION_MEMBERS = re.compile(
    TARGET_GROUP + r"\s+" + _RELATION_ACTIVE + r"\s+" + MEMBER_LIST, re.IGNORECASE
)
# "Both revenues and expenses are summarized below."
_PATTERN_BOTH_MEMBERS_RELATION_BELOW = re.compile(
    r"\bboth\s+(\w+)\s+and\s+(\w+)\s+(?:are|is)\s+" + _RELATION_PASSIVE + r"\s+below\b", re.IGNORECASE
)
# "The following categories are included in Table 6:\nA, B, and C."
_PATTERN_FOLLOWING_RELATION_TARGET_THEN_LIST = re.compile(
    r"\bthe following \w+\s+(?:are|is)\s+" + _RELATION_PASSIVE + r"\s+in\s+" + TARGET_GROUP
    + r"\s*:?\s*\n?\s*" + MEMBER_LIST,
    re.IGNORECASE,
)


def extract_coverage_candidates(text: str, document_id: str = "") -> list[StructuralCandidate]:
    candidates = []
    order = 0
    claimed = NonOverlappingMatches()

    def _add(anchor_text, members, target_query, rule_id):
        nonlocal order
        operations = [OperationType.COVERAGE_SET_MATCH]
        if target_query and target_query != "following list":
            operations.insert(0, OperationType.TARGET_EXISTS)
        candidates.append(
            StructuralCandidate(
                candidate_id=f"coverage-{order}",
                document_id=document_id,
                anchor_unit_id="",
                anchor_text=anchor_text,
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

    for match in _PATTERN_FOLLOWING_RELATION_TARGET_THEN_LIST.finditer(text):
        claimed.claim(*match.span())
        target = f"{match.group(1)} {match.group(2)}"
        members = split_members(match.group(3))
        _add(match.group(0), members, target, "coverage_following_relation_target_then_list")

    for match in _PATTERN_MEMBERS_RELATION_TARGET.finditer(text):
        if not claimed.try_claim(*match.span()):
            continue
        members = split_members(match.group(1))
        target = f"{match.group(2)} {match.group(3)}"
        _add(match.group(0), members, target, "coverage_members_relation_target")

    for match in _PATTERN_TARGET_RELATION_MEMBERS.finditer(text):
        if not claimed.try_claim(*match.span()):
            continue
        target = f"{match.group(1)} {match.group(2)}"
        members = split_members(match.group(3))
        _add(match.group(0), members, target, "coverage_target_relation_members")

    for match in _PATTERN_BOTH_MEMBERS_RELATION_BELOW.finditer(text):
        if not claimed.try_claim(*match.span()):
            continue
        members = [match.group(1), match.group(2)]
        _add(match.group(0), members, "following list", "coverage_both_members_relation_below")

    return candidates
