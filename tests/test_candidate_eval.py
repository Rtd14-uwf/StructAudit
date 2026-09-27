from structaudit.candidates.reference_rules import extract_reference_candidates
from structaudit.evaluation.candidate_eval import (
    aggregate_candidate_eval,
    candidate_surfaces_evidence,
    evaluate_document,
)
from structaudit.parsing.parser import extract_reference_mentions
from structaudit.ranking.ranker import rank_candidates


def _candidates(text: str):
    return rank_candidates(extract_reference_candidates(text, extract_reference_mentions(text)))


def test_surfaces_evidence_either_direction():
    candidates = _candidates("See Table 7 for refinancing defaults.")
    # gold evidence is a superstring of the candidate's anchor_text
    assert candidate_surfaces_evidence(candidates[0], ["Please see Table 7 for refinancing defaults. Thanks."])
    # gold evidence is a substring of the candidate's anchor_text
    assert candidate_surfaces_evidence(candidates[0], ["Table 7"])


def test_surfaces_evidence_no_overlap():
    candidates = _candidates("See Table 7 for refinancing defaults.")
    assert not candidate_surfaces_evidence(candidates[0], ["completely unrelated text"])


def test_surfaces_evidence_empty_gold():
    candidates = _candidates("See Table 7 for refinancing defaults.")
    assert not candidate_surfaces_evidence(candidates[0], [])


def test_evaluate_document_recall_thresholds():
    # Three candidates; only the third (lowest-ranked) one overlaps gold evidence.
    text = "See footnote 49. See Chart 6. See Table 7 for refinancing defaults."
    candidates = _candidates(text)
    assert len(candidates) == 3
    # Force a known rank order for a deterministic test: put the matching
    # one last by giving it the lowest rank_score directly.
    matching = next(c for c in candidates if "Table 7" in c.anchor_text)
    others = [c for c in candidates if c is not matching]
    ranked = others + [matching]  # matching is now rank 3

    result = evaluate_document("doc1", ranked, ["Table 7"])
    assert result.surfaced_at_1 is False
    assert result.surfaced_at_3 is True
    assert result.surfaced_at_5 is True
    assert result.surfaced_at_all is True
    assert result.n_candidates == 3


def test_evaluate_document_no_surfacing():
    candidates = _candidates("See footnote 49.")
    result = evaluate_document("doc1", candidates, ["something never mentioned in the document"])
    assert result.surfaced_at_1 is False
    assert result.surfaced_at_all is False


def test_aggregate_eval_recall_and_volume():
    from structaudit.evaluation.candidate_eval import DocumentCandidateEval

    results = [
        DocumentCandidateEval("d1", n_candidates=2, surfaced_at_1=True, surfaced_at_3=True, surfaced_at_5=True, surfaced_at_all=True),
        DocumentCandidateEval("d2", n_candidates=4, surfaced_at_1=False, surfaced_at_3=False, surfaced_at_5=True, surfaced_at_all=True),
        DocumentCandidateEval("d3", n_candidates=6, surfaced_at_1=False, surfaced_at_3=False, surfaced_at_5=False, surfaced_at_all=False),
        DocumentCandidateEval("d4", n_candidates=0, surfaced_at_1=False, surfaced_at_3=False, surfaced_at_5=False, surfaced_at_all=False),
    ]
    metrics = aggregate_candidate_eval(results)
    assert metrics.n_documents == 4
    assert metrics.recall_at_1 == 0.25
    assert metrics.recall_at_5 == 0.5
    assert metrics.recall_at_all == 0.5
    assert metrics.mean_candidates_per_document == 3.0
    assert metrics.median_candidates_per_document == 3.0
    assert metrics.min_candidates_per_document == 0
    assert metrics.max_candidates_per_document == 6


def test_aggregate_eval_empty():
    summary = aggregate_candidate_eval([])
    assert summary.n_documents == 0
    assert summary.recall_at_1 is None
    assert summary.mean_candidates_per_document is None
