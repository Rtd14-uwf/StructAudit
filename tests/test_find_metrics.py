"""
Tests for find_metrics.py.

The main sanity check reproduces the paper's own worked example
(Appendix E, Figures E.2/E.3):

    predicted E_hat = {"this is evidence", "this is more evidence"}
    reference E     = {"this is evidence", "this is some more evidence"}
"""

from structaudit.evaluation.find_metrics import (
    evidence_metric_strict,
    parse_answer_text,
    response_level_score,
)

DOCUMENT_TEXT = (
    "Some preamble. this is evidence and also this is more evidence, "
    "found within the document body."
)


def test_paper_worked_example_matching():
    predicted = ["this is evidence", "this is more evidence"]
    reference = ["this is evidence", "this is some more evidence"]

    result = evidence_metric_strict(predicted, reference, DOCUMENT_TEXT)

    # The identical span should be matched to the identical span.
    pred_idx_for_identical = predicted.index("this is evidence")
    ref_idx_for_identical = reference.index("this is evidence")
    assert (pred_idx_for_identical, ref_idx_for_identical) in result.matches

    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0
    assert 0.0 <= result.f1 <= 1.0
    # Both predicted spans should end up matched (both appear verbatim).
    assert len(result.unmatched_predicted) == 0
    assert len(result.matches) == 2


def test_non_verbatim_span_is_disqualified():
    predicted = ["this text never appears in the document"]
    reference = ["this is evidence"]
    result = evidence_metric_strict(predicted, reference, DOCUMENT_TEXT)
    assert result.matches == []
    assert result.unmatched_predicted == [0]
    assert result.precision == 0.0


def test_both_empty_is_perfect_score():
    result = evidence_metric_strict([], [], DOCUMENT_TEXT)
    assert result.precision == result.recall == result.f1 == 1.0


def test_one_empty_is_zero_score():
    result = evidence_metric_strict(["this is evidence"], [], DOCUMENT_TEXT)
    assert result.f1 == 0.0
    result2 = evidence_metric_strict([], ["this is evidence"], DOCUMENT_TEXT)
    assert result2.f1 == 0.0


def test_parse_answer_text_basic():
    raw = (
        "<answer><evidence>Table 4</evidence><evidence>BERT, T5, RoBERTa"
        "</evidence><description>Table 4 omits T5.</description></answer>"
        "<answer></answer>"
    )
    answers = parse_answer_text(raw)
    assert len(answers) == 1  # empty <answer></answer> is dropped
    assert answers[0].evidence == ["Table 4", "BERT, T5, RoBERTa"]
    assert answers[0].description == "Table 4 omits T5."


def test_response_level_score_takes_max():
    assert response_level_score([0.1, 0.9, 0.4]) == 0.9
    assert response_level_score([]) == 0.0
