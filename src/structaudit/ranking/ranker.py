"""
Candidate ranking: deduplicates candidates sharing the same anchor,
target, operations, and expected state, then scores the rest with

    s(c) = 0.30*s_rule + 0.25*s_target + 0.20*s_parse + 0.15*s_scope + 0.10*s_specificity

returning them sorted descending, with s(c) written back onto each
candidate's rank_score field.

s_rule: rule_confidence, as stored at extraction time.
s_target: 1.0 for an explicit target ("Table 5"), 0.7 for a relative
one ("following list"), 0.4 if there's no target at all.
s_parse: parse_confidence, currently always 1.0.
s_scope: 1.0 if there's any target_query, 0.5 otherwise.
s_specificity: 1.0 with an explicit expected state (members/count/schema),
0.6 for topic-only, 0.3 for a bare existence check.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

from structaudit.core import StructuralCandidate

_DEFAULT_WEIGHTS = {"rule": 0.30, "target": 0.25, "parse": 0.20, "scope": 0.15, "specificity": 0.10}

_EXPLICIT_TARGET_RE = re.compile(
    r"^(Table|Figure|Chart|Appendix|Section|Chapter|Note|Footnote)\s+"
    r"(?:[A-Za-z]|\d+(?:\.\d+)*|[IVXLCDM]+)$",
    re.IGNORECASE,
)
_RELATIVE_TARGETS = {"following list", "following table"}


def _find_upward(start: Path, relative_path: str, max_levels: int = 8) -> Path | None:
    """Search upward from start for relative_path (e.g. "configs/x.yaml"), checking each ancestor directory in turn."""
    current = start
    for _ in range(max_levels):
        candidate = current / relative_path
        if candidate.exists():
            return candidate
        if current.parent == current:  # reached filesystem root
            break
        current = current.parent

    return None


@lru_cache(maxsize=8)
def _load_ranking_weights_cached(config_path: str | Path | None = None) -> dict[str, float]:
    if config_path is None:
        config_path = _find_upward(Path(__file__).resolve().parent, "configs/structaudit.yaml")
    else:
        config_path = Path(config_path)

    if config_path is None or not config_path.exists():
        return dict(_DEFAULT_WEIGHTS)

    with config_path.open() as f:
        data = yaml.safe_load(f)
    return data["ranking"]["weights"]


def load_ranking_weights(config_path: str | Path | None = None) -> dict[str, float]:
    """Load weights from configs/structaudit.yaml, falling back to defaults if not found."""
    return dict(_load_ranking_weights_cached(config_path))


def _dedup_key(candidate: StructuralCandidate) -> tuple:
    return (
        candidate.anchor_text,
        candidate.target_query,
        tuple(op.value for op in candidate.operations),
        candidate.expected_count,
        tuple(candidate.expected_members),
        tuple(candidate.expected_schema),
        candidate.expected_topic,
    )


def deduplicate_candidates(candidates: list[StructuralCandidate]) -> list[StructuralCandidate]:
    """Drop candidates sharing the same anchor/target/operations/expected state. First occurrence wins."""
    seen = set()
    deduped = []
    for c in candidates:
        key = _dedup_key(c)
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    return deduped


def _score_target(candidate: StructuralCandidate) -> float:
    if candidate.target_query is None:
        return 0.4
    if candidate.target_query in _RELATIVE_TARGETS:
        return 0.7
    if _EXPLICIT_TARGET_RE.match(candidate.target_query):
        return 1.0
    return 0.7


def _score_scope(candidate: StructuralCandidate) -> float:
    return 1.0 if candidate.target_query is not None else 0.5


def _score_specificity(candidate: StructuralCandidate) -> float:
    if candidate.expected_members or candidate.expected_count is not None or candidate.expected_schema:
        return 1.0
    if candidate.expected_topic:
        return 0.6
    return 0.3


def compute_rank_score(candidate: StructuralCandidate, weights: dict[str, float]) -> float:
    return (
        weights["rule"] * candidate.rule_confidence
        + weights["target"] * _score_target(candidate)
        + weights["parse"] * candidate.parse_confidence
        + weights["scope"] * _score_scope(candidate)
        + weights["specificity"] * _score_specificity(candidate)
    )


def rank_candidates(
    candidates: list[StructuralCandidate],
    weights: dict[str, float] | None = None,
) -> list[StructuralCandidate]:
    """Deduplicate, score, and sort descending, writing rank_score back onto each candidate."""
    weights = weights or load_ranking_weights()
    deduped = deduplicate_candidates(candidates)
    for c in deduped:
        c.rank_score = compute_rank_score(c, weights)
    return sorted(deduped, key=lambda c: c.rank_score, reverse=True)
