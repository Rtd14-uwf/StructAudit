"""
Shared regex fragments and helpers used across the candidate rule
modules (reference_rules, count_rules, coverage_rules, schema_rules).
"""

from __future__ import annotations

import re

# The eight structural reference kinds, matched case-insensitively everywhere below.
REFERENCE_KINDS = r"(Table|Figure|Chart|Appendix|Section|Chapter|Note|Footnote)"

# letter | dotted digits | roman numeral
IDENTIFIER = r"(?:[A-Za-z]|\d+(?:\.\d+)*|[IVXLCDM]+)"

# Two capture groups: (kind, identifier), e.g. matches "Table 4" as ("Table", "4").
TARGET_GROUP = REFERENCE_KINDS + r"\s+(" + IDENTIFIER + r")"

# A flat, explicit comma/"and" coordinated list: "A, B, and C". Not a real
# noun-phrase or dependency parse -- nested or non-flat coordination isn't handled.
#
# Must always be embedded in a larger pattern with something after it that
# forces backtracking (e.g. a required verb like "are"/"is") -- used
# standalone with nothing following, its trailing [A-Za-z0-9 \-]* run keeps
# matching through ordinary prose with no way to know where the list ends.
MEMBER_LIST = r"([A-Za-z][A-Za-z0-9 \-]*(?:,\s*[A-Za-z][A-Za-z0-9 \-]*)*(?:,?\s+and\s+[A-Za-z][A-Za-z0-9 \-]*)?)"


def split_members(member_text: str) -> list[str]:
    """Split a MEMBER_LIST match into individual member strings."""
    member_text = re.sub(r"\s+and\s+", ", ", member_text)
    return [m.strip() for m in member_text.split(",") if m.strip()]


class NonOverlappingMatches:
    """Tracks match spans across multiple regex passes over the same text,
    so later patterns can skip a span already claimed by an earlier one.

    Rule functions run several patterns over the same text in priority
    order; without this, two patterns can both fire on overlapping text
    and produce two candidates for what's really one anchor.
    """

    def __init__(self) -> None:
        self._spans: list[tuple[int, int]] = []

    def overlaps(self, start: int, end: int) -> bool:
        return any(s < end and start < e for s, e in self._spans)

    def claim(self, start: int, end: int) -> None:
        self._spans.append((start, end))

    def try_claim(self, start: int, end: int) -> bool:
        """Claim the span if free; return False if it was already claimed."""
        if self.overlaps(start, end):
            return False
        self.claim(start, end)
        return True
