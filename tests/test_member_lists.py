import time

import pytest

from structaudit.candidates._members import list_text_after, list_text_before, parse_member_list
from structaudit.candidates.coverage_rules import extract_coverage_candidates
from structaudit.core import OperationType


def _members(text):
    return [c.expected_members for c in extract_coverage_candidates(text)]


# ---------------------------------------------------------------------------
# parse_member_list
# ---------------------------------------------------------------------------


def test_coordination_is_required():
    """One item is not a coordinated set. Uses phrases that would pass every
    other check, so this is the rule doing the rejecting.
    """
    assert parse_member_list("total revenue", trim="leading") is None
    assert parse_member_list("BERT", trim="trailing") is None
    assert parse_member_list("", trim="leading") is None


def test_comma_and_and_separators():
    assert parse_member_list("BERT, T5, and RoBERTa", trim="leading") == ["BERT", "T5", "RoBERTa"]
    assert parse_member_list("revenues and expenses", trim="leading") == ["revenues", "expenses"]


def test_leading_trim_cuts_lead_in_prose_from_the_first_member():
    assert parse_member_list("Results for BERT, T5, and RoBERTa", trim="leading") == ["BERT", "T5", "RoBERTa"]
    assert parse_member_list("Summary of BERT, T5, and RoBERTa", trim="leading") == ["BERT", "T5", "RoBERTa"]


def test_trailing_trim_cuts_prose_after_the_last_member():
    assert parse_member_list("BERT, T5, and RoBERTa for each quarter", trim="trailing") == ["BERT", "T5", "RoBERTa"]
    assert parse_member_list("revenues and expenditures for the District", trim="trailing") == ["revenues", "expenditures"]


def test_trim_applies_only_on_the_side_the_list_is_on():
    """A trailing-mode list keeps its first member intact (and so rejects a
    clause there); a leading-mode list keeps its last member intact.
    """
    assert parse_member_list("for each BERT, T5, and RoBERTa", trim="trailing") is None
    assert parse_member_list("BERT, T5, and RoBERTa for each", trim="leading") is None


def test_single_capital_letters_are_labels_not_the_article_a():
    assert parse_member_list("A, B, and C", trim="trailing") == ["A", "B", "C"]
    assert parse_member_list("A, B, and C", trim="leading") == ["A", "B", "C"]


def test_of_is_allowed_inside_a_member():
    assert parse_member_list("the total cost of services and the net cost of services", trim="trailing") == [
        "total cost of services", "net cost of services",
    ]


def test_leading_determiners_are_stripped_from_every_member():
    assert parse_member_list("the revenues and the expenses", trim="leading") == ["revenues", "expenses"]


def test_apostrophes_survive_inside_members():
    members = parse_member_list("District's revenues, expenditures, and transfers", trim="leading")
    assert members == ["District's revenues", "expenditures", "transfers"]


def test_clauses_are_rejected_by_stopwords_inside_a_member():
    # real corpus text: prose that happens to contain "and"
    assert parse_member_list(
        "the occupations with the highest speaking and listening skill scores", trim="trailing"
    ) is None
    assert parse_member_list("the State Treasurer is to reduce the amounts and the local treasurer must pay", trim="trailing") is None


def test_a_stopword_inside_a_short_member_is_rejected():
    """The length limit alone would let these through (3 tokens each); a
    stopword inside means the member is a clause fragment, not a name.
    """
    assert parse_member_list("sales with fees and costs", trim="trailing") is None
    assert parse_member_list("income is stable and expenses", trim="trailing") is None
    assert parse_member_list("costs and sales with fees", trim="leading") is None


def test_overlong_members_and_overlong_lists_are_rejected():
    assert parse_member_list("one two three four five, six", trim="leading") is None
    assert parse_member_list(", ".join(f"item{i}" for i in range(16)), trim="leading") is None


def test_members_without_any_alphanumeric_are_rejected():
    assert parse_member_list("--, ++", trim="leading") is None


# ---------------------------------------------------------------------------
# Text windows
# ---------------------------------------------------------------------------


def test_list_text_before_stops_at_the_previous_sentence():
    text = "Earlier prose here. BERT, T5, and RoBERTa are shown in Table 4."
    start, segment = list_text_before(text, text.index(" are shown"))
    assert segment == "BERT, T5, and RoBERTa"
    assert text[start:].startswith("BERT")


def test_list_text_before_keeps_abbreviation_periods_inside_a_list():
    text = "Jan., Feb., and Mar. are shown in Table 4."
    _, segment = list_text_before(text, text.index(" are shown"))
    assert segment == "Jan., Feb., and Mar"


def test_list_text_before_stops_at_a_newline_and_a_colon():
    assert list_text_before("intro\nA, B, and C are shown", len("intro\nA, B, and C"))[1] == "A, B, and C"
    assert list_text_before("Categories: A, B, and C are shown", len("Categories: A, B, and C"))[1] == "A, B, and C"


def test_list_text_after_skips_a_colon_and_newline_and_stops_at_the_sentence_end():
    text = "included in Table 6:\nA, B, and C. Next sentence."
    end, segment = list_text_after(text, text.index("Table 6") + len("Table 6"))
    assert segment == "A, B, and C"
    assert text[end:].startswith(". Next")


@pytest.mark.parametrize("text, expected", [
    # the spec's own worked examples
    ("BERT, T5, and RoBERTa are shown in Table 4.", ["BERT", "T5", "RoBERTa"]),
    ("Table 4 reports BERT, T5, and RoBERTa.", ["BERT", "T5", "RoBERTa"]),
    ("Both revenues and expenses are summarized below.", ["revenues", "expenses"]),
    ("The following categories are included in Table 6:\nA, B, and C.", ["A", "B", "C"]),
    # previously truncated or polluted
    ("The District's revenues, expenditures, and transfers are shown in Table 1.", ["District's revenues", "expenditures", "transfers"]),
    ("Jan., Feb., and Mar. are shown in Table 4.", ["Jan", "Feb", "Mar"]),
    ("Results for BERT, T5, and RoBERTa are shown in Table 4.", ["BERT", "T5", "RoBERTa"]),
    ("Table 4 reports BERT, T5, and RoBERTa for each quarter.", ["BERT", "T5", "RoBERTa"]),
    # real corpus sentences that are genuine coordinated claims
    ("Table 3 presents revenues and expenditures for the District's Building Fund for the last five Fiscal Years", ["revenues", "expenditures"]),
    ("Table 3 shows the total cost of services and the net cost of services", ["total cost of services", "net cost of services"]),
    ("Table 9 summarizes S&P Global Ratings' NCF and valuation assessment by property type", ["S&P Global Ratings' NCF", "valuation assessment"]),
])
def test_extracts_genuine_coordinated_lists(text, expected):
    assert _members(text) == [expected]


@pytest.mark.parametrize("text", [
    "The results are reported in Table 20.",
    "A summary of the District's financial position is shown in Table 1.",
    "Chart 3 shows the occupations with the highest interpersonal skill scores for those occupations projected to have many openings and fast growth.",
    "Section 20 provides that the State Treasurer is to reduce the amounts payable by the Commonwealth to affected cities and towns.",
    "Chapter 9 provides for the recognition of a security interest represented by a specifically pledged source of revenues, such as the pledged Net Revenues.",
    "Table 2 shows the change in Net Position for the fiscal years 2024 and 2023.",
    "Table 12 shows the underwriting characteristics for Tier 1 Strategic and Tier 2 Strategic loans as of fourth-quarter 2019.",
    "means the Plan Sponsor and any other Related Employer that is listed in Section 1.04.",
])
def test_rejects_prose_that_only_looks_like_a_list(text):
    assert extract_coverage_candidates(text) == []


def test_following_list_form_yields_one_candidate_without_target_exists():
    candidates = extract_coverage_candidates("The following categories are included in Table 6:\nA, B, and C.")
    assert len(candidates) == 1
    assert candidates[0].target_query == "Table 6"
    assert OperationType.TARGET_EXISTS in candidates[0].operations


def test_both_form_targets_the_following_list_without_target_exists():
    (candidate,) = extract_coverage_candidates("Both revenues and expenses are summarized below.")
    assert candidate.target_query == "following list"
    assert candidate.operations == [OperationType.COVERAGE_SET_MATCH]


def test_anchor_text_is_the_matched_sentence_fragment():
    (candidate,) = extract_coverage_candidates("Prose first. Table 4 reports BERT, T5, and RoBERTa. More prose.")
    assert candidate.anchor_text == "Table 4 reports BERT, T5, and RoBERTa"


def test_document_id_is_set_at_construction():
    (candidate,) = extract_coverage_candidates("Table 4 reports BERT, T5, and RoBERTa.", "doc1")
    assert candidate.document_id == "doc1"


def test_extraction_time_is_linear_on_long_unpunctuated_prose():
    text = ("alpha beta gamma delta epsilon " * 10_000)[:300_000]
    start = time.perf_counter()
    assert extract_coverage_candidates(text) == []
    assert time.perf_counter() - start < 2.0
