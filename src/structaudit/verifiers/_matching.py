"""
Match a list of expected labels against a list of observed labels in the following order:
  1. normalized exact match
  2. token-normalized match (order-insensitive, conservative plural folding)
  3. semantic equivalence via a SemanticBackend

Each stage is run across all still-unmatched expected labels before the
next stage starts. Matching per-label through all three stages would let a
semantic guess for an early expected label consume an observed label that a
later expected label matches exactly.

Observed labels are consumed one-to-one. Whatever is left over is extra.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from structaudit.models.semantic_backend import SemanticBackend

_PUNCTUATION_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_exact(label: str) -> str:
    """Case, punctuation, and whitespace normalization."""
    label = _PUNCTUATION_RE.sub(" ", label.casefold())
    return _WHITESPACE_RE.sub(" ", label).strip()


def fold_plural(token: str) -> str:
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def token_key(label: str) -> tuple[str, ...]:
    """Order-insensitive token form: "Income, Net" and "Net Income" share a key."""
    return tuple(sorted(fold_plural(t) for t in normalize_exact(label).split()))


@dataclass
class LabelMatchResult:
    matched: dict[str, str] = field(default_factory=dict)  # expected -> observed label it matched
    methods: dict[str, str] = field(default_factory=dict)  # expected -> "normalized_exact" | "token_normalized" | "semantic"
    missing: list[str] = field(default_factory=list)  # definitively not found
    uncertain: list[str] = field(default_factory=list)  # could not decide
    extra: list[str] = field(default_factory=list)
    semantic_calls: int = 0
    semantic_confidences: list[float] = field(default_factory=list)  # confidences behind semantic outcomes


def match_labels(
    expected: list[str],
    observed: list[str],
    backend: SemanticBackend,
    semantic_task: str,
) -> LabelMatchResult:
    result = LabelMatchResult()
    pool = [label for label in observed if label.strip()]
    consumed = [False] * len(pool)

    def claim(expected_label: str, index: int, method: str) -> None:
        consumed[index] = True
        result.matched[expected_label] = pool[index]
        result.methods[expected_label] = method

    remaining = list(expected)

    for method, key in (("normalized_exact", normalize_exact), ("token_normalized", token_key)):
        still_remaining = []
        for expected_label in remaining:
            target = key(expected_label)
            index = next((i for i, label in enumerate(pool) if not consumed[i] and key(label) == target), None)
            if index is None:
                still_remaining.append(expected_label)
            else:
                claim(expected_label, index, method)
        remaining = still_remaining

    for expected_label in remaining:
        unconsumed = [label for i, label in enumerate(pool) if not consumed[i]]
        if not unconsumed:
            result.missing.append(expected_label)  # nothing left to compare against; no model call needed
            continue

        result.semantic_calls += 1
        verdict = backend.classify(semantic_task, {"expected_label": expected_label, "candidate_labels": unconsumed})

        if verdict["label"] == "SUPPORTED":
            index = next(i for i, label in enumerate(pool) if not consumed[i] and label == verdict["best_match"])
            claim(expected_label, index, "semantic")
            result.semantic_confidences.append(verdict["confidence"])
        elif verdict["label"] == "NOT_SUPPORTED":
            result.missing.append(expected_label)
            result.semantic_confidences.append(verdict["confidence"])
        else:
            result.uncertain.append(expected_label)

    result.extra = [label for i, label in enumerate(pool) if not consumed[i]]
    return result
