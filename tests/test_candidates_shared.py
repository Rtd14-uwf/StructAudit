import re

from structaudit.candidates._shared import (
    MEMBER_LIST,
    TARGET_GROUP,
    NonOverlappingMatches,
    split_members,
)


def test_target_group_matches_kind_and_identifier():
    match = re.search(TARGET_GROUP, "See Table 7 for details.")
    assert match.group(1) == "Table"
    assert match.group(2) == "7"


def test_target_group_matches_all_eight_kinds():
    for kind, ident in [
        ("Table", "4"), ("Figure", "2"), ("Chart", "6"), ("Appendix", "B"),
        ("Section", "3.2"), ("Chapter", "V"), ("Note", "14"), ("Footnote", "49"),
    ]:
        match = re.search(TARGET_GROUP, f"{kind} {ident}")
        assert match is not None, f"{kind} {ident} should match"
        assert match.group(1) == kind
        assert match.group(2) == ident


def test_split_members_comma_and_and():
    assert split_members("BERT, T5, and RoBERTa") == ["BERT", "T5", "RoBERTa"]


def test_split_members_two_items_no_oxford_comma():
    assert split_members("revenues and expenses") == ["revenues", "expenses"]


def test_split_members_single_item():
    assert split_members("Revenue") == ["Revenue"]


def test_member_list_pattern_matches_flat_list_when_anchored():
    anchored_pattern = MEMBER_LIST + r"\s+(?:are|is)\b"
    match = re.search(anchored_pattern, "BERT, T5, and RoBERTa are shown")
    assert match.group(1) == "BERT, T5, and RoBERTa"


def test_non_overlapping_matches_rejects_overlap():
    claimed = NonOverlappingMatches()
    assert claimed.try_claim(0, 10) is True
    assert claimed.try_claim(5, 15) is False  # overlaps [0, 10)
    assert claimed.try_claim(10, 20) is True  # adjacent, not overlapping


def test_non_overlapping_matches_claim_without_check():
    claimed = NonOverlappingMatches()
    claimed.claim(0, 10)
    assert claimed.overlaps(5, 15) is True
    assert claimed.overlaps(10, 20) is False
