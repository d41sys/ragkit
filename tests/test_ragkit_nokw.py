"""The keyword-free rule: toy corpus here, the committed question set in ``test_nokw_set``."""

from __future__ import annotations

from ragkit.nokw import MAX_OVERLAP, TOP_K, TermIndex, naming_violations, overlap, violations

DOCS = [
    (
        "reg_a_art1",
        "# Regulation A, Article 1: Gizmo review\n\nThe gizmo operator shall review the gizmo "
        "every month and notify the regulator of the gizmo outcome.",
    ),
    ("reg_a_art2", "# Regulation A, Article 2: Other\n\nThe operator shall notify the regulator."),
    ("reg_a_art3", "# Regulation A, Article 3: Third\n\nThe regulator shall publish a report."),
]


def test_constants_are_the_preregistered_ones() -> None:
    assert (TOP_K, MAX_OVERLAP) == (20, 1)


def test_top_terms_rank_by_tf_idf_and_drop_terms_present_everywhere() -> None:
    index = TermIndex(DOCS)
    top = index.top_terms("reg_a_art1")
    assert top[0] == "gizmo"  # tf 3, rare
    assert "regul" not in top or top.index("regul") > top.index("gizmo")
    assert all(index.df[t] < index.n for t in top)


def test_overlap_counts_stems_so_an_inflection_of_a_top_term_overlaps() -> None:
    index = TermIndex(DOCS)
    count, shared = overlap(index, "Who reviews gizmos monthly?", ["reg_a_art1"])
    assert count >= 1 and "gizmo" in shared
    assert overlap(index, "Who decides what happens next?", ["reg_a_art1"])[0] == 0


def test_overlap_takes_the_worst_gold_article() -> None:
    index = TermIndex(DOCS)
    count, _ = overlap(index, "gizmo report", ["reg_a_art3", "reg_a_art1"])
    assert count == 1
    count, _ = overlap(index, "gizmo review outcome", ["reg_a_art3", "reg_a_art1"])
    assert count >= 2


def test_naming_an_act_an_article_a_legal_word_or_the_heading_is_flagged() -> None:
    index = TermIndex(DOCS)
    gold = ["reg_a_art1"]
    assert "names an act number" in naming_violations(index, "What does 2017/2195 say?", gold)
    assert "names an article number" in naming_violations(index, "Under Article 5, who?", gold)
    assert naming_violations(index, "What does the directive say?", gold)
    assert naming_violations(index, "Rules on a Gizmo Review please?", gold)
    assert naming_violations(index, "Who decides what happens next?", gold) == []


def test_violations_combine_overlap_and_naming() -> None:
    index = TermIndex(DOCS)
    record = {
        "q": "Does the gizmo outcome matter under 2017/2195?",
        "relevant_doc_ids": ["reg_a_art1"],
    }
    problems = violations(index, record)
    assert any("overlap" in p for p in problems) and "names an act number" in problems
    ok = {"q": "Who decides what happens next?", "relevant_doc_ids": ["reg_a_art1"]}
    assert violations(index, ok) == []
