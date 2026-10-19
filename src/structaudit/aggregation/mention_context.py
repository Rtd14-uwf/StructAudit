"""
Is a reference mention actually a pointer to something in this document?

The reference extractor treats every "Table 4" / "Note 3" / "Section 265" as a
pointer.

  definition  The mention *is* the target's own label: "## Note 3: Investments",
              "[Footnote 5: text...]", "### Table 4. Revenue". It starts its line
              and is followed by ":" / "." / end of line. (14 of 91)

  external    The mention points at another document: "Section 265 of the Code",
              "Wisconsin Statutes Section 66.1105", "Section 22 of the Act".
              (45 of 91)

  pointer     Anything else: "(See table 5.)", "Table 3 shows ...".
"""

from __future__ import annotations

import re
from typing import Literal

MentionContext = Literal["definition", "external", "pointer"]

_ONLY_LINE_MARKERS_BEFORE = re.compile(r"[ \t>#*_\[(]*")
_DEFINITION_FOLLOWER = re.compile(r"[*_\s]*(?:[:.\-\u2013\u2014)\]]|$)")
_EXTERNAL_FOLLOWER = re.compile(
    r"(?:\([\w.]+\))*\s*,?\s*(?:of|under)\s+(?:the|any|such|an?|its|our|that)\b", re.IGNORECASE
)
_EXTERNAL_PRECEDER = re.compile(
    r"(?:statutes?|codes?|acts?|rules?|regulations?|indenture|resolution|ordinance|agreement|u\.s\.c\.?|c\.f\.r\.?)"
    r"\s*,?\s*$",
    re.IGNORECASE,
)


def classify_mention(text: str, start: int, end: int) -> MentionContext:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    line_end = len(text) if line_end == -1 else line_end
    before_on_line = text[line_start:start]
    after_on_line = text[end:line_end]

    if _ONLY_LINE_MARKERS_BEFORE.fullmatch(before_on_line) and _DEFINITION_FOLLOWER.match(after_on_line):
        return "definition"
    if _EXTERNAL_FOLLOWER.match(after_on_line) or _EXTERNAL_PRECEDER.search(before_on_line):
        return "external"
    return "pointer"
