"""Long documents (papers): windows, page and citation fields, evidence recall."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

import ragkit
from ragkit import paperpool as pp
from ragkit.llm import codex_transport
from ragkit.openpapers import CONFIGS, OpenQuestion, answer_scores, answer_verdicts, choose
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


def test_paged_documents_pack_into_the_paper_budget_by_default(tmp_path: Path) -> None:
    for folder, page_break in (("paged", "\f"), ("plain", " ")):
        (tmp_path / folder).mkdir()
        for name in ("a", "b"):  # about 8 200 estimated tokens each: one fits 12 000, two 36 000
            text = f"Paper {name} measures battery peak shaving.{page_break}" + FILLER * 700
            (tmp_path / folder / f"{name}.txt").write_text(text, encoding="utf-8")

    def packed(folder: str, budget_tokens: int | None = None) -> int:
        found = ragkit.search(
            "battery peak shaving",
            corpus_dir=tmp_path / folder,
            retriever="bm25",
            budget_tokens=budget_tokens,
        )
        return len(found.articles)

    assert packed("paged") == 2
    assert packed("paged", budget_tokens=12_000) == 1  # an explicit budget still wins
    assert packed("plain") == 1  # no page breaks: the article budget, as for the EU corpus


def test_paged_documents_read_a_wider_chunk_pool_by_default(tmp_path: Path) -> None:
    for folder, page_break in (("paged", "\f"), ("plain", " ")):
        (tmp_path / folder).mkdir()
        # a: more than 10 chunks that match strongly; b: one weak match, ranked below them
        strong = page_break.join(["battery peak shaving " * 20] * 16)
        (tmp_path / folder / "a.txt").write_text(strong, encoding="utf-8")
        (tmp_path / folder / "b.txt").write_text("One battery. " + FILLER * 20, encoding="utf-8")

    def papers(folder: str, pack_chunks: int | None = None) -> set[str]:
        found = ragkit.search(
            "battery peak shaving",
            corpus_dir=tmp_path / folder,
            retriever="bm25",
            pack_chunks=pack_chunks,
        )
        return {a.id for a in found.articles}

    assert papers("paged") == {"a", "b"}
    assert papers("paged", pack_chunks=10) == {"a"}  # an explicit pool still wins
    assert papers("plain") == {"a"}  # no page breaks: the top 10 chunks, as for the EU corpus


def test_search_sees_a_document_added_after_the_first_search(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("Batteries shave the evening peak.", encoding="utf-8")
    first = ragkit.search("heat pump defrost peak", corpus_dir=tmp_path, retriever="bm25")
    assert [a.id for a in first.articles] == ["a"]
    (tmp_path / "b.txt").write_text("The heat pump runs a defrost cycle.", encoding="utf-8")
    second = ragkit.search("heat pump defrost peak", corpus_dir=tmp_path, retriever="bm25")
    assert second.articles[0].id == "b"
    (tmp_path / "b.txt").unlink()
    third = ragkit.search("heat pump defrost peak", corpus_dir=tmp_path, retriever="bm25")
    assert [a.id for a in third.articles] == ["a"]


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


def test_codex_transport_returns_the_last_message_and_the_turn_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen["command"], seen["prompt"] = command, kwargs["input"]
        Path(command[command.index("-o") + 1]).write_text('{"answer": "42"}', encoding="utf-8")
        events = [
            {"type": "thread.started", "thread_id": "t"},
            {"type": "turn.completed", "usage": {"input_tokens": 900, "output_tokens": 7}},
        ]
        return subprocess.CompletedProcess(command, 0, "\n".join(map(json.dumps, events)), "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    payload = {
        "model": "m",
        "system": [{"type": "text", "text": "RULES"}],
        "messages": [{"role": "user", "content": "QUESTION"}],
    }
    reply = codex_transport()(payload, "codex")
    assert reply["content"][0]["text"] == '{"answer": "42"}'
    assert reply["usage"] == {"input_tokens": 900, "output_tokens": 7}
    assert seen["prompt"] == "RULES\n\nQUESTION"
    assert seen["command"][seen["command"].index("-m") + 1] == "m"


def _answers(bench: str, a12: list[str], a36: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "bench": bench,
            "arm": arm,
            "verdict": verdict,
            "supports": True,
            "citations": ["p"],
            "papers": ["p"],
            "input_tokens": 1,
        }
        for a12_verdict, a36_verdict in zip(a12, a36, strict=True)
        for arm, verdict in (("A12", a12_verdict), ("A36", a36_verdict))
    ]


def test_answer_rule_reverts_only_on_shown_harm_and_counts_partial_as_not_correct() -> None:
    same = _answers("qasper", ["yes", "no"] * 20, ["yes", "no"] * 20)
    gain = _answers("mdaqa", ["partial"] * 30 + ["yes"] * 10, ["yes"] * 40)
    kept = answer_verdicts(answer_scores(same + gain))
    assert list(kept.values()) == ["kept", "confirmed", "shown"]
    harm = _answers("qasper", ["yes"] * 40, ["partial"] * 30 + ["yes"] * 10)
    reverted = answer_verdicts(answer_scores(harm + gain))
    assert list(reverted.values()) == ["reverted to 12 000", "confirmed", "not shown"]


def test_pool_choice_prefers_the_simplest_configuration_within_the_tie() -> None:
    scores = dict.fromkeys(pp.CONFIGS, 0.5)
    scores[(50, 12_000, 3)] = 0.604
    scores[(20, None, 1)] = 0.600  # one setting changed, within 0.005 of the best
    scores[(10, 18_000, 1)] = 0.600  # also one setting, but the smaller pool wins
    assert pp.choose(scores) == (10, 18_000, 1)
    scores[(50, 12_000, 3)] = 0.700
    assert pp.choose(scores) == (50, 12_000, 3)
    assert pp.choose(dict.fromkeys(pp.CONFIGS, 0.5)) == pp.B36


def _pool_rows(gain: dict[str, float]) -> dict[str, dict[str, dict[str, list[float]]]]:
    """40 questions per benchmark; C is right on ``gain`` more of them than B36 (fewer if < 0)."""
    rows = {}
    for bench in pp.BENCHMARKS:
        right = 20 + round(gain.get(bench, 0.0) * 40)
        rows[bench] = {
            "B36": {"primary": [1.0] * 20 + [0.0] * 20},
            "C": {"primary": [1.0] * right + [0.0] * (40 - right)},
        }
    return rows


def test_pool_rule_needs_a_gain_on_a_target_and_no_loss_anywhere() -> None:
    assert pp.verdict(_pool_rows({"mdaqa": 0.25})) == "default for paged corpora"
    assert pp.verdict(_pool_rows({"long": 0.25})) == "default for paged corpora"
    assert pp.verdict(_pool_rows({"qasper": 0.25})) == "not adopted"  # QASPER is not a target
    assert pp.verdict(_pool_rows({"mdaqa": 0.25, "qasper": -0.25})) == "not adopted"
    assert pp.verdict(_pool_rows({})) == "not adopted"
