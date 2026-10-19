"""
Violation aggregation: turns per-candidate VerificationResults into the warnings.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from structaudit.aggregation.mention_context import classify_mention
from structaudit.core import (
    CandidateType,
    OperationType,
    ResolutionMethod,
    Status,
    StructuralCandidate,
    TargetResolution,
    VerificationResult,
)


@dataclass(frozen=True)
class WarningPolicy:
    emit_statuses: frozenset[Status] = frozenset({Status.VIOLATED})
    require_deterministic: bool = True
    # With require_deterministic on, semantic violations are held back unless their
    # operation is named here (e.g. {OperationType.TARGET_TOPIC_MATCH}).
    allow_semantic_for: frozenset[OperationType] = frozenset()
    min_confidence: float = 0.0
    require_explicit_target: bool = True  # hold back verdicts whose target was resolved only heuristically
    hold_back_definition_sites: bool = True
    hold_back_external_references: bool = True
    max_warnings: int = 5
    max_evidence_spans: int = 3
    max_evidence_chars: int = 400


@dataclass
class ValidatedCandidate:
    candidate: StructuralCandidate
    resolution: TargetResolution
    result: VerificationResult


class AnchorSpan(BaseModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str


class AuditWarning(BaseModel):
    warning_id: str
    document_id: str
    warning_type: CandidateType
    operation: OperationType
    status: Status
    emitted: bool  # False: a VIOLATED verdict the policy held back
    hold_reason: str | None = None  # why it was held back
    target_query: str | None
    resolved_unit_id: str | None
    expected: dict
    observed: dict
    anchors: list[AnchorSpan]
    evidence_text: list[str]
    candidate_ids: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    deterministic: bool
    explanation: str
    rank_score: float


# ---------------------------------------------------------------------------
# Evidence: the sentence around an anchor
# ---------------------------------------------------------------------------

_LEFT_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"\u201c\[])|\n")
_RIGHT_BOUNDARY_RE = re.compile(r"[.!?](?=\s|$)|\n")


def evidence_span(text: str, start: int, end: int, max_chars: int = 400) -> tuple[int, int]:
    """(start, end) of the sentence containing text[start:end]"""
    window_start = max(0, start - max_chars)
    boundaries = list(_LEFT_BOUNDARY_RE.finditer(text, window_start, start))
    if boundaries:
        left = boundaries[-1].end()
    elif window_start == 0:
        left = 0
    else:  # no boundary within reach: start at the next word
        space = re.search(r"\s", text[window_start:start])
        left = window_start + space.end() if space else start

    right_match = _RIGHT_BOUNDARY_RE.search(text, end, min(len(text), end + max_chars))
    if right_match is None:
        right = min(len(text), end + max_chars)
    elif text[right_match.start()] == "\n":
        right = right_match.start()
    else:
        right = right_match.end()

    if right - left > max_chars:  # a very long sentence: keep the anchor and its nearer context
        left = max(left, start - max_chars // 3)
        right = min(right, end + 2 * max_chars // 3)
        if left > 0 and not text[left - 1].isspace():  # don't cut a word in half
            word_end = re.search(r"\s", text[left:start])
            left = left + word_end.end() if word_end else start
        if right < len(text) and not text[right].isspace():
            last_space = text.rfind(" ", end, right)
            right = last_space if last_space != -1 else end

    while left < start and text[left].isspace():
        left += 1
    while right > end and text[right - 1].isspace():
        right -= 1
    return left, right


# ---------------------------------------------------------------------------
# Merging and policy
# ---------------------------------------------------------------------------


def _hold_reason(v: "ValidatedCandidate", text: str, policy: WarningPolicy) -> str | None:
    """Why this VIOLATED verdict should not become a warning, or None if it may."""
    result = v.result
    if (
        policy.require_deterministic
        and not result.deterministic
        and result.operations_run[0] not in policy.allow_semantic_for
    ):
        return "semantic_backend"
    if result.confidence < policy.min_confidence:
        return "low_confidence"
    if policy.require_explicit_target and v.resolution.method == ResolutionMethod.RELATIVE:
        return "heuristic_target"
    if result.operations_run[0] == OperationType.TARGET_EXISTS and (
        policy.hold_back_definition_sites or policy.hold_back_external_references
    ):
        span = _anchor_span_of(v, text)
        if span is not None:
            context = classify_mention(text, span[0], span[1])
            if context == "definition" and policy.hold_back_definition_sites:
                return "definition_site"
            if context == "external" and policy.hold_back_external_references:
                return "external_reference"
    return None


def _merge_key(v: ValidatedCandidate) -> tuple:
    operation = v.result.operations_run[0]
    target = (v.candidate.target_query or "").strip().lower()
    if operation == OperationType.TARGET_EXISTS:
        return (operation, target)  # the same missing target, whichever claim cited it
    c = v.candidate
    expected = json.dumps(
        {"count": c.expected_count, "members": sorted(m.lower() for m in c.expected_members),
         "schema": sorted(s.lower() for s in c.expected_schema), "topic": (c.expected_topic or "").lower()},
        sort_keys=True,
    )
    return (operation, target, expected)


def _anchor_span_of(v: ValidatedCandidate, text: str) -> tuple[int, int] | None:
    c = v.candidate
    if c.anchor_char_start is not None:
        return c.anchor_char_start, c.anchor_char_end
    found = text.find(c.anchor_text)
    return (found, found + len(c.anchor_text)) if found >= 0 else None


def _build_warnings(
    items: list[ValidatedCandidate], text: str, document_id: str, policy: WarningPolicy, emitted: bool,
    hold_reason: str | None = None,
) -> list[AuditWarning]:
    groups: dict[tuple, list[ValidatedCandidate]] = {}
    for v in items:
        groups.setdefault(_merge_key(v), []).append(v)

    built = []
    for members in groups.values():
        members.sort(key=lambda v: v.candidate.anchor_char_start if v.candidate.anchor_char_start is not None else 0)
        first = members[0]
        anchors: list[AnchorSpan] = []
        seen_spans = set()
        for v in members:
            span = _anchor_span_of(v, text)
            if span is None:
                continue
            sentence = evidence_span(text, span[0], span[1], policy.max_evidence_chars)
            if sentence not in seen_spans:
                seen_spans.add(sentence)
                anchors.append(AnchorSpan(start=sentence[0], end=sentence[1], text=text[sentence[0]:sentence[1]]))
        anchors = anchors[: policy.max_evidence_spans]

        operation = first.result.operations_run[0]
        explanation = first.result.explanation
        if len(members) > 1:
            explanation += f" ({len(members)} mentions)"
        built.append(AuditWarning(
            warning_id="",
            document_id=document_id,
            warning_type=CandidateType.REFERENCE if operation == OperationType.TARGET_EXISTS else first.candidate.candidate_type,
            operation=operation,
            status=first.result.status,
            emitted=emitted,
            hold_reason=hold_reason,
            target_query=first.candidate.target_query,
            resolved_unit_id=first.resolution.unit_id,
            expected=first.result.expected,
            observed=first.result.observed,
            anchors=anchors,
            evidence_text=[a.text for a in anchors],
            candidate_ids=[v.candidate.candidate_id for v in members],
            confidence=min(v.result.confidence for v in members),
            deterministic=all(v.result.deterministic for v in members),
            explanation=explanation,
            rank_score=max(v.candidate.rank_score for v in members),
        ))

    built.sort(key=lambda w: (-w.confidence, -w.rank_score, w.anchors[0].start if w.anchors else 0))
    for number, warning in enumerate(built):
        warning.warning_id = f"{document_id}:{'w' if emitted else 'held'}{number}"
    return built


def aggregate(
    validated: list[ValidatedCandidate], text: str, document_id: str, policy: WarningPolicy | None = None
) -> tuple[list[AuditWarning], list[AuditWarning]]:
    """(emitted warnings, held-back warnings). Only verdicts whose status is in policy.emit_statuses are considered."""
    policy = policy or WarningPolicy()
    candidates = [v for v in validated if v.result.status in policy.emit_statuses]
    passing: list[ValidatedCandidate] = []
    failing: dict[str, list[ValidatedCandidate]] = {}
    for v in candidates:
        reason = _hold_reason(v, text, policy)
        if reason is None:
            passing.append(v)
        else:
            failing.setdefault(reason, []).append(v)
    emitted = _build_warnings(passing, text, document_id, policy, emitted=True)[: policy.max_warnings]
    held_back = [
        w for reason, items in failing.items()
        for w in _build_warnings(items, text, document_id, policy, emitted=False, hold_reason=reason)
    ]
    for number, warning in enumerate(held_back):
        warning.warning_id = f"{document_id}:held{number}"
    return emitted, held_back
