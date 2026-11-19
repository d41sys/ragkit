"""Long documents (papers): windows, page and citation fields, evidence recall."""

from __future__ import annotations

import json
from pathlib import Path

import ragkit
from ragkit.openpapers import CONFIGS, OpenQuestion, choose
from ragkit.openpapers import measure as open_measure
from ragkit.papersbench import Gold, PaperQuestion, measure
from ragkit.parent import CHARS_PER_TOKEN, Passage, pack_articles

FILLER = "Unrelated words about something else entirely. "


def _paper(fact: str) -> str:
    return FILLER * 400 + fact + " " + FILLER * 400


def test_windows_let_several_long_papers_share_the_budget() -> None:
    facts = {"p1": "Patching cuts attention cost.", "p2": "Inverted tokens embed each variate."}
    papers = {pid: _paper(fact) for pid, fact in facts.items()}
    pool = [Passage(pid, fact) for pid, fact in facts.items()]
    whole = pack_articles(pool, papers, budget_tokens=2000)
    windows = pack_articles(pool, papers, budget_tokens=2000, window_tokens=1000)
    assert [p.article_id for p in whole] == ["p1"]  # one paper fills the budget, packing ends
    assert [p.article_id for p in windows] == ["p1", "p2"]
    for p in windows:
        assert facts[p.article_id] in p.text
        assert len(p.text) == 1000 * CHARS_PER_TOKEN


def test_slices_keep_quotes_from_two_parts_of_one_long_paper() -> None:
    early, late = "Batch size 32 is used.", "The model fails on noisy series."
    paper = early + " " + FILLER * 800 + late + " " + FILLER * 800
    pool = [Passage("p", late), Passage("p", early)]
    one = pack_articles(pool, {"p": paper}, budget_tokens=1000)
    two = pack_articles(pool, {"p": paper}, budget_tokens=1000, slices=2)
    assert late in one[0].text and early not in one[0].text
    assert early in two[0].text and late in two[0].text
    assert two[0].text.index(early) < two[0].text.index(late)  # document order
    assert len(two[0].text) <= 1000 * CHARS_PER_TOKEN


def test_search_cites_pages_and_the_sources_citation(tmp_path: Path) -> None:
    pages = ["Title page of the study.", "Method section.", "The battery cut the peak by 31 %."]
    (tmp_path / "paper1.txt").write_text("\f".join(pages), encoding="utf-8")
    (tmp_path / "note.txt").write_text("A short note on heat pumps in schools.", encoding="utf-8")
    sources = {"documents": [{"id": "paper1", "title": "A study", "citation": "Doe, ICML 2024"}]}
    (tmp_path / "SOURCES.json").write_text(json.dumps(sources), encoding="utf-8")
    query = "How much did the battery cut the peak?"
    paper = ragkit.search(query, corpus_dir=tmp_path, retriever="bm25").articles[0]
    assert (paper.id, paper.act, paper.pages) == ("paper1", "A study", "1-3")
    assert paper.citation == "Doe, ICML 2024"
    note = ragkit.search("heat pumps in schools", corpus_dir=tmp_path, retriever="bm25").articles[0]
    assert (note.id, note.pages, note.citation) == ("note", None, None)


def test_open_benchmarks_take_the_best_annotator_and_break_ties_to_shipped_packing() -> None:
    q = OpenQuestion("q1", "q", ("p1",), (("alpha beta", "gamma"), ("delta",)))
    packed = [Passage("p1", "Alpha beta and gamma."), Passage("p2", "other")]
    assert open_measure(q, ["p2", "p1"], packed)["primary"] == 1.0  # first annotator fully packed
    flat = {c: [0.5] for c in CONFIGS}
    flat[(12_000, 4000, 3)] = [0.504]  # inside the tie margin: the shipped packing wins
    flat[(24_000, 8000, 1)] = [0.6]
    chosen = choose({"qasper": flat, "mdaqa": flat})
    assert chosen[12_000] == (12_000, None, 1)
    assert chosen[24_000] == (24_000, 8000, 1)


def test_evidence_recall_counts_each_gold_quote() -> None:
    q = PaperQuestion("c1", "cross", "q", (Gold("p1", "Alpha Beta"), Gold("p2", "gamma delta")))
    packed = [Passage("p1", "text with alpha\n beta inside"), Passage("p3", "other")]
    got = measure(q, ["p3", "p1", "p2"], packed)
    assert got["evidence"] == 0.5 and got["papers"] == 0.5
    assert got["mrr"] == 0.5  # first gold paper is second in the ranking
