"""
COUNT candidate rules.

A COUNT candidate is only created when a quantity modifies a bounded set.
Every pattern below requires the bounding phrase (a following list, "... below", or an explicit target) as
part of the match itself.
"""

from __future__ import annotations

import re

from structaudit.candidates._shared import TARGET_GROUP, NonOverlappingMatches
from structaudit.core import CandidateType, OperationType, StructuralCandidate

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_NUMBER = r"(\d+|" + "|".join(_NUMBER_WORDS) + r")"


def _to_int(number_text: str) -> int:
    number_text = number_text.lower()
    return _NUMBER_WORDS.get(number_text, None) or int(number_text)


# "the following four risks were identified:" -- bounded by "following ...:", no explicit target.
_PATTERN_FOLLOWING_N_NOUN = re.compile(
    r"\bthe following " + _NUMBER + r"\s+(\w+)\b", re.IGNORECASE
)
# "the four risks below"
_PATTERN_N_NOUN_BELOW = re.compile(
    r"\bthe " + _NUMBER + r"\s+(\w+)\s+below\b", re.IGNORECASE
)
# "four risks are listed below" (no leading "the")
_PATTERN_N_NOUN_ARE_LISTED_BELOW = re.compile(
    _NUMBER + r"\s+(\w+)\s+are\s+listed\s+below\b", re.IGNORECASE
)
# "Results for all six regions are reported in Table 5." / "six regions are shown in Table 5"
_PATTERN_N_NOUN_SHOWN_IN_TARGET = re.compile(
    r"(?:all\s+)?" + _NUMBER + r"\s+(\w+)\s+(?:are|is)\s+(?:shown|reported)\s+in\s+" + TARGET_GROUP,
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

        count = _to_int(match.group(1))
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
            count = _to_int(match.group(1))
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
