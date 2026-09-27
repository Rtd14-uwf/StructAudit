from structaudit.candidates.count_rules import extract_count_candidates
from structaudit.candidates.reference_rules import extract_reference_candidates
from structaudit.parsing.parser import extract_reference_mentions
from structaudit.ranking.ranker import (
    compute_rank_score,
    deduplicate_candidates,
    load_ranking_weights,
    rank_candidates,
)

_WEIGHTS = {"rule": 0.30, "target": 0.25, "parse": 0.20, "scope": 0.15, "specificity": 0.10}


def _reference_candidate(text: str):
    mentions = extract_reference_mentions(text)
    return extract_reference_candidates(text, mentions)[0]


def test_load_ranking_weights():
    weights = load_ranking_weights()
    assert weights == _WEIGHTS


def test_load_ranking_weights_caching_is_mutation_safe():
    w1 = load_ranking_weights()
    w1["rule"] = 999.0
    w2 = load_ranking_weights()
    assert w2["rule"] == _WEIGHTS["rule"]


def test_find_upward_robust_to_depth(tmp_path):
    from structaudit.ranking.ranker import _find_upward

    (tmp_path / "configs").mkdir()
    config_file = tmp_path / "configs" / "structaudit.yaml"
    config_file.write_text("x: 1")

    deep = tmp_path / "src" / "pkg" / "sub" / "subsub"
    deep.mkdir(parents=True)
    assert _find_upward(deep, "configs/structaudit.yaml") == config_file


def test_find_upward_returns_none_when_not_found(tmp_path):
    from structaudit.ranking.ranker import _find_upward

    assert _find_upward(tmp_path, "configs/does_not_exist.yaml") is None


def test_rank_score_topic_match_arithmetic():
    candidate = _reference_candidate("See Table 7 for refinancing defaults.")
    score = compute_rank_score(candidate, _WEIGHTS)
    assert abs(score - 0.93) < 1e-9


def test_rank_score_count_arithmetic():
    candidate = extract_count_candidates("Results for all six regions are reported in Table 5.")[0]
    score = compute_rank_score(candidate, _WEIGHTS)
    assert abs(score - 0.955) < 1e-9


def test_rank_score_bare_reference_tier():
    candidate = _reference_candidate("See footnote 49.")
    score = compute_rank_score(candidate, _WEIGHTS)
    # s_rule=0.9, s_target=1.0 (explicit "footnote 49"), s_parse=1.0,
    # s_scope=1.0, s_specificity=0.3 -> 0.3*0.9+0.25+0.2+0.15+0.1*0.3 = 0.90
    assert abs(score - 0.90) < 1e-9


def test_rank_score_relative_vs_explicit():
    explicit = extract_count_candidates("Results for all six regions are reported in Table 5.")[0]
    relative = extract_count_candidates("The following four risks were identified:")[0]
    assert compute_rank_score(explicit, _WEIGHTS) > compute_rank_score(relative, _WEIGHTS)


def test_dedup_collapses_identical():
    text = "See Table 7 for refinancing defaults."
    candidates = _reference_candidate(text)
    doubled = [candidates, candidates]  # same object twice, same dedup key
    deduped = deduplicate_candidates(doubled)
    assert len(deduped) == 1


def test_dedup_keeps_distinct():
    c1 = _reference_candidate("See Table 7 for refinancing defaults.")
    c2 = extract_count_candidates("The following four risks were identified:")[0]
    deduped = deduplicate_candidates([c1, c2])
    assert len(deduped) == 2


def test_rank_candidates_sorted_and_scored():
    text = "See Table 7 for refinancing defaults. Results for all six regions are reported in Table 5."
    mentions = extract_reference_mentions(text)
    candidates = extract_reference_candidates(text, mentions) + extract_count_candidates(text)
    ranked = rank_candidates(candidates)

    scores = [c.rank_score for c in ranked]
    assert scores == sorted(scores, reverse=True)
    assert all(s > 0 for s in scores)
