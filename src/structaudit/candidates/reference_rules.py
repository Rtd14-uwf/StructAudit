"""
REFERENCE candidate rules.

Every explicit structural pointer found by the parser becomes a REFERENCE
candidate with TARGET_EXISTS. TARGET_TOPIC_MATCH is added when the
surrounding sentence states what the target should contain, via three
surface patterns (topic-then-target, target-then-topic, "see X for Y").
"""

from __future__ import annotations

import re

from structaudit.candidates._shared import IDENTIFIER, REFERENCE_KINDS
from structaudit.core import (
    CandidateType,
    OperationType,
    ParsedDocument,
    ReferenceMention,
    StructuralCandidate,
)

# Simple word/comma run up to the next clause boundary, not a real noun-phrase parse.
_TOPIC_CHUNK = r"([A-Za-z][A-Za-z0-9 ,\-]*?)"

_PATTERN_TOPIC_THEN_TARGET = re.compile(
    _TOPIC_CHUNK + r"\s+(?:is|are)\s+(?:shown|reported|listed|summarized|described)\s+in\s+"
    + REFERENCE_KINDS + r"\s+(" + IDENTIFIER + r")",
    re.IGNORECASE,
)
_PATTERN_TARGET_THEN_TOPIC = re.compile(
    REFERENCE_KINDS + r"\s+(" + IDENTIFIER + r")\s+"
    r"(?:shows|reports|lists|summarizes|describes)\s+" + _TOPIC_CHUNK + r"(?:\.|$)",
    re.IGNORECASE,
)
_PATTERN_SEE_TARGET_FOR_TOPIC = re.compile(
    r"(?:see|refer to)\s+" + REFERENCE_KINDS + r"\s+("
    + IDENTIFIER + r")\s+for\s+" + _TOPIC_CHUNK + r"(?:\.|$)",
    re.IGNORECASE,
)


def _find_topic_match(text: str, mention: ReferenceMention) -> str | None:
    """Check the three topic-match patterns in a window around the mention
    (cheap, and avoids matching an unrelated target elsewhere in the document).
    """
    window_start = max(0, mention.char_start - 120)
    window_end = min(len(text), mention.char_end + 120)
    window = text[window_start:window_end]

    for pattern, target_groups, topic_group in (
        (_PATTERN_TOPIC_THEN_TARGET, (2, 3), 1),
        (_PATTERN_TARGET_THEN_TOPIC, (1, 2), 3),
        (_PATTERN_SEE_TARGET_FOR_TOPIC, (1, 2), 3),
    ):
        for match in pattern.finditer(window):
            kind_text, ident_text = match.group(target_groups[0]), match.group(target_groups[1])
            if kind_text.upper() == mention.kind.value and ident_text == mention.identifier_raw:
                return match.group(topic_group).strip().rstrip(",")
    return None


def extract_reference_candidates(
    text: str, reference_mentions: list[ReferenceMention], document_id: str = ""
) -> list[StructuralCandidate]:
    """One REFERENCE candidate per explicit pointer."""
    candidates = []
    for i, mention in enumerate(reference_mentions):
        operations = [OperationType.TARGET_EXISTS]
        expected_topic = _find_topic_match(text, mention)
        if expected_topic:
            operations.append(OperationType.TARGET_TOPIC_MATCH)

        candidates.append(
            StructuralCandidate(
                candidate_id=f"ref-{i}",
                document_id=document_id,
                anchor_unit_id="",  # no sentence-level unit to anchor to yet
                anchor_text=mention.raw_text,
                candidate_type=CandidateType.REFERENCE,
                target_query=mention.raw_text,
                target_kind=mention.kind.value,
                expected_topic=expected_topic,
                operations=operations,
                rule_id="reference_explicit_pointer",
                rule_confidence=0.9,
                parse_confidence=1.0,
            )
        )
    return candidates


def extract_reference_candidates_from_document(
    parsed: ParsedDocument, document_id: str = ""
) -> list[StructuralCandidate]:
    return extract_reference_candidates(parsed.raw_text, parsed.reference_mentions, document_id)
