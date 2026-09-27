"""
Candidate extraction evaluation: Recall@k (gold inconsistencies surfaced
within the top k ranked candidates, divided by number of gold
inconsistencies) plus candidate-volume statistics.

"Surfaced" here means the gold evidence text overlaps
(either string contains the other) with some candidate's anchor_text
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass

from structaudit.core import StructuralCandidate


def candidate_surfaces_evidence(candidate: StructuralCandidate, gold_evidence: list[str]) -> bool:
    return any(ev in candidate.anchor_text or candidate.anchor_text in ev for ev in gold_evidence if ev)


@dataclass
class DocumentCandidateEval:
    doc_id: str
    n_candidates: int
    surfaced_at_1: bool
    surfaced_at_3: bool
    surfaced_at_5: bool
    surfaced_at_all: bool


def evaluate_document(
    doc_id: str, ranked_candidates: list[StructuralCandidate], gold_evidence: list[str]
) -> DocumentCandidateEval:
    """ranked_candidates must already be sorted. Recall@k means within the top k."""
    return DocumentCandidateEval(
        doc_id=doc_id,
        n_candidates=len(ranked_candidates),
        surfaced_at_1=any(candidate_surfaces_evidence(c, gold_evidence) for c in ranked_candidates[:1]),
        surfaced_at_3=any(candidate_surfaces_evidence(c, gold_evidence) for c in ranked_candidates[:3]),
        surfaced_at_5=any(candidate_surfaces_evidence(c, gold_evidence) for c in ranked_candidates[:5]),
        surfaced_at_all=any(candidate_surfaces_evidence(c, gold_evidence) for c in ranked_candidates),
    )


@dataclass
class CandidateEvalSummary:
    """Aggregate recall/volume metrics. The metric fields are None when n_documents is 0"""

    n_documents: int
    recall_at_1: float | None = None
    recall_at_3: float | None = None
    recall_at_5: float | None = None
    recall_at_all: float | None = None
    mean_candidates_per_document: float | None = None
    median_candidates_per_document: float | None = None
    min_candidates_per_document: int | None = None
    max_candidates_per_document: int | None = None

    def to_json(self) -> dict:
        return asdict(self)


def aggregate_candidate_eval(results: list[DocumentCandidateEval]) -> CandidateEvalSummary:
    n = len(results)
    if n == 0:
        return CandidateEvalSummary(n_documents=0)

    candidate_counts = [r.n_candidates for r in results]
    return CandidateEvalSummary(
        n_documents=n,
        recall_at_1=sum(r.surfaced_at_1 for r in results) / n,
        recall_at_3=sum(r.surfaced_at_3 for r in results) / n,
        recall_at_5=sum(r.surfaced_at_5 for r in results) / n,
        recall_at_all=sum(r.surfaced_at_all for r in results) / n,
        mean_candidates_per_document=statistics.mean(candidate_counts),
        median_candidates_per_document=statistics.median(candidate_counts),
        min_candidates_per_document=min(candidate_counts),
        max_candidates_per_document=max(candidate_counts),
    )
