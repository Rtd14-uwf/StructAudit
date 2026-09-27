"""
FIND benchmark evaluation metrics.

Implements the strict evidence metric from Lovering et al. (2026),
Appendix E / Algorithm E.1: bipartite matching of predicted vs. reference
evidence spans, scored by longest-common-substring character overlap.

The description metric (BLEURT) and task metric (LLM-as-judge) are left as stubs (for now)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np
from scipy.optimize import linear_sum_assignment

# Tagged output format: <answer><evidence>e1</evidence>...
# <description>d</description></answer>, repeated per predicted
# inconsistency, or a single empty <answer></answer> for none found.
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_EVIDENCE_RE = re.compile(r"<evidence>(.*?)</evidence>", re.DOTALL)
_DESCRIPTION_RE = re.compile(r"<description>(.*?)</description>", re.DOTALL)


@dataclass
class PredictedAnswer:
    """One <answer>...</answer> block: a candidate inconsistency."""
    evidence: list[str] = field(default_factory=list)
    description: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.evidence and not self.description.strip()


def parse_answer_text(raw_output: str) -> list[PredictedAnswer]:
    """Parse a raw model completion into PredictedAnswer objects. Empty
    <answer></answer> blocks are dropped. Regex-based and intentionally
    permissive to match the format models actually produce.
    """
    answers: list[PredictedAnswer] = []
    for block in _ANSWER_RE.findall(raw_output):
        evidence = [e.strip() for e in _EVIDENCE_RE.findall(block)]
        description_match = _DESCRIPTION_RE.search(block)
        description = description_match.group(1).strip() if description_match else ""
        answer = PredictedAnswer(evidence=evidence, description=description)
        if not answer.is_empty:
            answers.append(answer)
    return answers


def _lcstr_overlap(x: str, y: str) -> int:
    """Length of the longest contiguous common substring of x and y."""
    if not x or not y:
        return 0

    matcher = SequenceMatcher(a=x, b=y, autojunk=False)
    match = matcher.find_longest_match(0, len(x), 0, len(y))
    return match.size


def _overlap_score(x: str, y: str) -> tuple[float, int]:
    """F1 of character-level LCStr overlap between spans x and y, plus the raw overlap amount."""
    overlap_amount = _lcstr_overlap(x, y)
    if overlap_amount == 0 or not x or not y:
        return 0.0, 0

    precision = overlap_amount / len(x)
    recall = overlap_amount / len(y)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, overlap_amount


@dataclass
class EvidenceMetricResult:
    precision: float
    recall: float
    f1: float
    matches: list[tuple[int, int]]  # (predicted_idx, reference_idx) pairs kept
    unmatched_predicted: list[int]  # spans not verbatim in the document
    total_char_overlap: int
    total_char_predicted: int
    total_char_reference: int


def evidence_metric_strict(
    predicted_spans: list[str],
    reference_spans: list[str],
    document_text: str,
) -> EvidenceMetricResult:
    """Strict evidence metric (Algorithm E.1, mode="strict").

    A predicted span not found verbatim in document_text is dropped
    from matching.
    """
    n_pred, n_ref = len(predicted_spans), len(reference_spans)

    if n_pred == 0 and n_ref == 0:
        return EvidenceMetricResult(1.0, 1.0, 1.0, [], [], 0, 0, 0)

    if n_pred == 0 or n_ref == 0:
        total_predicted = sum(len(p) for p in predicted_spans)
        total_reference = sum(len(r) for r in reference_spans)
        return EvidenceMetricResult(0.0, 0.0, 0.0, [], list(range(n_pred)), 0,
                                     total_predicted, total_reference)

    DISQUALIFIED = -1.0
    similarity = np.full((n_pred, n_ref), DISQUALIFIED)
    overlap_chars = np.zeros((n_pred, n_ref), dtype=int)
    unmatched_predicted: list[int] = []

    for i, pred in enumerate(predicted_spans):
        if pred not in document_text:
            unmatched_predicted.append(i)
            continue
        for j, ref in enumerate(reference_spans):
            f1, overlap = _overlap_score(pred, ref)
            similarity[i, j] = f1
            overlap_chars[i, j] = overlap

    row_ind, col_ind = linear_sum_assignment(similarity, maximize=True)
    matches = [
        (int(i), int(j))
        for i, j in zip(row_ind, col_ind)
        if similarity[i, j] > DISQUALIFIED
    ]

    total_char_overlap = sum(overlap_chars[i, j] for i, j in matches)
    total_char_predicted = sum(len(p) for p in predicted_spans)
    total_char_reference = sum(len(r) for r in reference_spans)

    precision = total_char_overlap / total_char_predicted if total_char_predicted else 0.0
    recall = total_char_overlap / total_char_reference if total_char_reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return EvidenceMetricResult(
        precision=precision,
        recall=recall,
        f1=f1,
        matches=matches,
        unmatched_predicted=unmatched_predicted,
        total_char_overlap=total_char_overlap,
        total_char_predicted=total_char_predicted,
        total_char_reference=total_char_reference,
    )


def response_level_score(per_candidate_scores: list[float]) -> float:
    """Best candidate score for a single document."""
    return max(per_candidate_scores) if per_candidate_scores else 0.0
