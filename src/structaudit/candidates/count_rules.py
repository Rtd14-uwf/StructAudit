"""
COUNT candidate rules.

A COUNT candidate is only created when a quantity modifies a bounded set.
Every pattern below requires the bounding phrase (a following list, "... below", or an explicit target) as
part of the match itself.
"""

from __future__ import annotations

import re

from structaudit.candidates._shared import NUMBER, TARGET_GROUP, NonOverlappingMatches, to_int
from structaudit.core import CandidateType, OperationType, StructuralCandidate


# "the following four risks were identified:" -- bounded by "following ...:", no explicit target.
_PATTERN_FOLLOWING_N_NOUN = re.compile(
    r"\bthe following " + NUMBER + r"\s+(\w+)\b", re.IGNORECASE
)
# "the four risks below"
_PATTERN_N_NOUN_BELOW = re.compile(
    r"\bthe " + NUMBER + r"\s+(\w+)\s+below\b", re.IGNORECASE
)
# "four risks are listed below" (no leading "the")
_PATTERN_N_NOUN_ARE_LISTED_BELOW = re.compile(
    NUMBER + r"\s+(\w+)\s+are\s+listed\s+below\b", re.IGNORECASE
)
# "Results for all six regions are reported in Table 5." / "six regions are shown in Table 5"
_PATTERN_N_NOUN_SHOWN_IN_TARGET = re.compile(
    r"(?:all\s+)?" + NUMBER + r"\s+(\w+)\s+(?:are|is)\s+(?:shown|reported)\s+in\s+" + TARGET_GROUP,
    re.IGNORECASE,
)

# "Both revenues and expenses are summarized below" is implemented under
# COVERAGE rules instead, since that's a member-set claim, not a count.


def extract_count_candidates(text: str, document_id: str = "") -> list[StructuralCandidate]:
    candidates = []
    order = 0
    claimed = NonOverlappingMatches()

    for match in _PATTERN_N_NOUN_SHOWN_IN_TARGET.finditer(text):
        if not claimed.try_claim(*match.span()):
            continue

        count = to_int(match.group(1))
        target_kind, target_ident = match.group(3), match.group(4)
        candidates.append(
            StructuralCandidate(
                candidate_id=f"count-{order}",
                document_id=document_id,
                anchor_unit_id="",
                anchor_text=match.group(0),
                candidate_type=CandidateType.COUNT,
                target_query=f"{target_kind} {target_ident}",
                expected_count=count,
                operations=[OperationType.TARGET_EXISTS, OperationType.COUNT_MATCH],
                rule_id="count_n_noun_shown_in_target",
                rule_confidence=0.85,
                parse_confidence=1.0,
            )
        )
        order += 1

    for pattern, rule_id, target_query in (
        (_PATTERN_FOLLOWING_N_NOUN, "count_following_n_noun", "following list"),
        (_PATTERN_N_NOUN_BELOW, "count_n_noun_below", "following list"),
        (_PATTERN_N_NOUN_ARE_LISTED_BELOW, "count_n_noun_are_listed_below", "following list"),
    ):
        for match in pattern.finditer(text):
            if not claimed.try_claim(*match.span()):
                continue
            count = to_int(match.group(1))
            candidates.append(
                StructuralCandidate(
                    candidate_id=f"count-{order}",
                    document_id=document_id,
                    anchor_unit_id="",
                    anchor_text=match.group(0),
                    candidate_type=CandidateType.COUNT,
                    target_query=target_query,
                    expected_count=count,
                    operations=[OperationType.COUNT_MATCH],
                    rule_id=rule_id,
                    rule_confidence=0.8,
                    parse_confidence=1.0,
                )
            )
            order += 1

    return candidates
