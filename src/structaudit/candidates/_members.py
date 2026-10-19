"""
Parse a coordinated member list ("BERT, T5, and RoBERTa") out of the text
next to a "...are shown in Table 4" / "Table 4 reports ..." phrase.

1. split on commas / "and"
2. trim lead-in prose from the first member ("Results for BERT" -> "BERT")
    or trailing prose from the last ("RoBERTa for each quarter" -> "RoBERTa")
3. require every member to be item-shaped: 1-4 tokens, no stopwords inside
    (a stopword inside means a clause, not a name)
4. require at least two members: one noun phrase is not a coordinated set

Candidates whose text fails any check are dropped rather than repaired.
"""

from __future__ import annotations

import re
from typing import Literal

MAX_MEMBER_TOKENS = 4
MAX_MEMBERS = 15
WINDOW_CHARS = 300

_DETERMINERS = frozenset(
    "a an the this that these those its their our his her all both any some each every".split()
)
_STOPWORDS = _DETERMINERS | frozenset(
    """for of in on at by with from to into onto over under about as than
    and or but nor so yet
    is are was were be been being has have had do does did will would can could may might shall should
    which who whom whose what when where while not no
    it they we you he she them us""".split()
)
# "Bank of America", "total cost of services": "of" is fine inside a name.
_ALLOWED_INSIDE = frozenset({"of"})

_SPLIT_RE = re.compile(r"\s*,\s*(?:and\s+)?|\s+and\s+")

# A hard boundary ends the text a list can occupy. A period immediately followed
# by a comma ("Jan., Feb., and Mar.") is an abbreviation inside a list, not a boundary.
_BOUNDARY_RE = re.compile(r"[.!?;:](?:\s+|$)|\n")


# Generic nouns that open a sentence about a list ("Results for BERT, T5, and X",
# "Summary of A, B, and C"). Only used to trim the first member's lead-in.
_LEAD_IN_NOUNS = frozenset(
    "results data summary details figures values breakdown comparison list information statistics totals".split()
)


def _stem(token: str) -> str:
    return re.sub(r"[^\w']", "", token.lower())


def _is_stop(token: str) -> bool:
    return _stem(token) in _STOPWORDS


def _is_label(tokens: list[str]) -> bool:
    """A lone capital letter ("A", "B") is a label, not the article "a"."""
    return len(tokens) == 1 and len(tokens[0]) == 1 and tokens[0].isupper()


def _is_boundary_word(token: str) -> bool:
    stem = _stem(token)
    return (stem in _STOPWORDS and stem not in _ALLOWED_INSIDE) or stem in _LEAD_IN_NOUNS


def _strip_determiners(tokens: list[str]) -> list[str]:
    while len(tokens) > 1 and _stem(tokens[0]) in _DETERMINERS:
        tokens = tokens[1:]
    return tokens


def _strip_edge_of(tokens: list[str]) -> list[str]:
    while tokens and _stem(tokens[0]) in _ALLOWED_INSIDE:
        tokens = tokens[1:]
    while tokens and _stem(tokens[-1]) in _ALLOWED_INSIDE:
        tokens = tokens[:-1]
    return tokens


def _is_item_shaped(tokens: list[str]) -> bool:
    if _is_label(tokens):
        return True
    if not 1 <= len(tokens) <= MAX_MEMBER_TOKENS:
        return False
    if not any(ch.isalnum() for tok in tokens for ch in tok):
        return False
    for i, tok in enumerate(tokens):
        if _is_stop(tok):
            if _stem(tok) not in _ALLOWED_INSIDE or i in (0, len(tokens) - 1):
                return False
    return True


def parse_member_list(text: str, trim: Literal["leading", "trailing"]) -> list[str] | None:
    """Members of a coordinated list, or None if `text` isn't one.

    trim="leading": the list ends the text (it sits before a verb), so any
    prose in front of the first member is a lead-in and is cut.
    trim="trailing": the list starts the text (it sits after a verb), so prose
    after the last member is cut.
    """
    raw = [item.strip() for item in _SPLIT_RE.split(text.strip()) if item.strip()]
    if not 2 <= len(raw) <= MAX_MEMBERS:
        return None

    members = []
    for i, item in enumerate(raw):
        tokens = item.split()
        if not _is_label(tokens):
            tokens = _strip_determiners(tokens)
            if trim == "leading" and i == 0:
                cut = max((j for j, tok in enumerate(tokens) if _is_boundary_word(tok)), default=-1)
                tokens = tokens[cut + 1:]
            elif trim == "trailing" and i == len(raw) - 1:
                cut = next((j for j, tok in enumerate(tokens) if _is_boundary_word(tok)), len(tokens))
                tokens = tokens[:cut]
            tokens = _strip_edge_of(_strip_determiners(tokens))
        if not _is_item_shaped(tokens):
            return None
        members.append(" ".join(tokens).rstrip("."))
    return members


def list_text_before(text: str, end: int) -> tuple[int, str]:
    """(start offset, text) of the list that ends at 'end', back to the nearest hard boundary."""
    start = max(0, end - WINDOW_CHARS)
    window = text[start:end].rstrip()
    if window.endswith("."):
        window = window[:-1]  # abbreviation period right before the verb ("... and Mar. are shown")
    boundaries = list(_BOUNDARY_RE.finditer(window))
    cut = boundaries[-1].end() if boundaries else 0
    if cut == 0 and start > 0:
        space = re.search(r"\s", window)  # the window began mid-word; drop the partial word
        cut = space.end() if space else len(window)
    return start + cut, window[cut:].strip()


def list_text_after(text: str, start: int) -> tuple[int, str]:
    """(end offset, text) of the list that begins at 'start', up to the nearest hard boundary."""
    while start < len(text) and text[start] in ": \t\n":
        start += 1
    window = text[start:start + WINDOW_CHARS]
    boundary = _BOUNDARY_RE.search(window)
    segment = window[: boundary.start()] if boundary else window
    return start + len(segment), segment.strip()
