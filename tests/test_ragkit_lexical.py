"""Tests for BM25F and the model-free lexical experiment (offline, no models)."""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from ragkit.bm25 import BM25Index
from ragkit.bm25f import BM25FIndex
from ragkit.chunking import Chunk
from ragkit.lexical import (
    CONFIGS,
    Question,
    cross_validate,
    evaluate,
    grid_points,
    questions_from_records,
    rank_documents,
    render_report,
    score_question,
    stratified_folds,
)

TITLED = [
    Chunk("a", "a", "frequency reserve activation rules", 0, 10, title="Balancing"),
    Chunk("b", "b", "reserve prices and congestion revenue", 0, 10, title="Congestion income"),
    Chunk("c", "c", "balancing of the system frequency reserve", 0, 10, title="Other article"),
]


def test_bm25f_with_zero_title_weight_reproduces_bm25() -> None:
    chunks = [
        Chunk(f"d{i}", f"d{i}", text, 0, len(text), title="ignored title words")
        for i, text in enumerate(
            ["cat dog", "cat mouse mouse", "dog dog dog bird", "bird fish cat cat dog"]
        )
    ]
    plain = BM25Index(k1=1.2, b=0.6)
    plain.add(chunks)
    fielded = BM25FIndex(k1=1.2, b=0.6, title_weight=0.0)
    fielded.add(chunks)

    for query in ("cat dog", "dog dog bird", "ignored title", "fish"):
        expected = plain.search(query, k=10)
        got = fielded.search(query, k=10)
        assert [c.id for c, _ in got] == [c.id for c, _ in expected]
        for (_, s_got), (_, s_exp) in zip(got, expected, strict=True):
            assert s_got == pytest.approx(s_exp, rel=1e-9)
    assert fielded.idf("title") == 0.0


def test_bm25f_hand_computed_title_boost() -> None:
    chunks = [
        Chunk("t", "t", "alpha beta gamma delta", 0, 5, title="zeta"),
        Chunk("u", "u", "alpha beta gamma delta", 0, 5, title="eta theta"),
    ]
    index = BM25FIndex(k1=1.2, b=0.5, title_weight=3.0)
    index.add(chunks)
    # N = 2, df(zeta) = 1: idf = ln((2 - 1 + .5) / (1 + .5) + 1) = ln 2.
    # avg title len = 1.5, avg body len = 4. Chunk t: title norm = .5 + .5 * 1 / 1.5 = 5/6,
    # tf~ = 3 * 1 / (5/6) = 3.6; score = ln 2 * 2.2 * 3.6 / (1.2 + 3.6).
    expected = math.log(2.0) * 2.2 * 3.6 / 4.8
    ranked = index.search("zeta")
    assert [c.id for c, _ in ranked] == ["t"]
    assert ranked[0][1] == pytest.approx(expected, rel=1e-9)


def test_bm25f_title_weight_lifts_the_heading_match() -> None:
    plain_body = BM25FIndex(title_weight=0.0)
    plain_body.add(TITLED)
    boosted = BM25FIndex(title_weight=4.0)
    boosted.add(TITLED)
    # "balancing" is in a's title and in c's body; the heading field should put a first.
    assert boosted.search("balancing")[0][0].id == "a"
    assert plain_body.search("balancing")[0][0].id == "c"


def test_bm25f_weighted_terms_and_validation() -> None:
    index = BM25FIndex()
    index.add(TITLED)
    full = dict((c.id, s) for c, s in index.search([("congestion", 1.0)]))
    half = dict((c.id, s) for c, s in index.search([("congestion", 0.5)]))
    assert half["b"] == pytest.approx(full["b"] / 2)
    assert index.search([("congestion", 0.0)]) == []
    assert index.search("absent words only") == []
    assert BM25FIndex().search("x") == []
    # Repeated terms add up, as in BM25Index.
    doubled = dict((c.id, s) for c, s in index.search("congestion congestion"))
    assert doubled["b"] == pytest.approx(2 * full["b"])
    with pytest.raises(ValueError):
        BM25FIndex(title_weight=-1.0)


def test_rank_documents_dedupes_by_first_occurrence_and_limits() -> None:
    chunks = [(Chunk(f"{d}#{i}", d, "x", 0, 1), 1.0) for i, d in enumerate("aabcab")]
    assert rank_documents(chunks) == ["a", "b", "c"]
    assert rank_documents(chunks, limit=2) == ["a", "b"]
    assert rank_documents(chunks, limit=None) == ["a", "b", "c"]


def test_score_question_metrics() -> None:
    scores = score_question(["x", "a", "b"], ["a", "b"])
    assert scores["r5"] == 1.0
    assert scores["mrr"] == 0.5
    assert 0.0 < scores["ndcg"] < 1.0


def test_stratified_folds_balance_each_stratum_and_are_seeded() -> None:
    strata = ["a"] * 5 + ["b"] * 4 + ["c"] * 3
    folds = stratified_folds(strata, 2, seed=1)
    assert folds == stratified_folds(strata, 2, seed=1)
    for s in "abc":
        counts = [
            sum(1 for f, t in zip(folds, strata, strict=True) if t == s and f == k) for k in (0, 1)
        ]
        assert abs(counts[0] - counts[1]) <= 1
    assert folds != stratified_folds(strata, 2, seed=2)


def test_cross_validate_picks_on_train_folds_and_keeps_default_on_ties() -> None:
    strata = ["t"] * 6
    points = grid_points()
    default, other = points[0], points[1]
    flat = {m: [0.5] * 6 for m in ("r5", "mrr", "ndcg")}
    better = {m: [0.9] * 6 for m in ("r5", "mrr", "ndcg")}

    scores, chosen = cross_validate({default: flat, other: dict(flat)}, strata, seeds=(0, 1))
    assert set(chosen) == {default}
    assert scores["mrr"] == pytest.approx([0.5] * 6)

    scores, chosen = cross_validate({default: flat, other: better}, strata, seeds=(0, 1))
    assert set(chosen) == {other}
    assert scores["mrr"] == pytest.approx([0.9] * 6)


def _toy_inputs() -> tuple[list[tuple[str, str]], list[Question]]:
    docs = [
        (
            "bal",
            "# Reg (EU) 1, Article 1: Balancing\n\n1. The TSO shall procure balancing energy "
            "from balancing service providers every day.\n\n2. Reserves are activated.",
        ),
        (
            "con",
            "# Reg (EU) 1, Article 2: Congestion\n\n1. Congestion income is shared between "
            "operators according to the methodology.",
        ),
        (
            "cus",
            "# Reg (EU) 1, Article 3: Customers\n\n1. Final customers may act as active "
            "customers and sell electricity.",
        ),
    ]
    qa = [
        {
            "q": "Which transmission system operator duties exist?",
            "relevant_doc_ids": ["bal"],
            "type": "paraphrase",
        },
        {"q": "Who shares congestion income?", "relevant_doc_ids": ["con"], "type": "direct"},
        {"q": "May customers sell electricity?", "relevant_doc_ids": ["cus"], "type": "direct"},
        {
            "q": "Which operators and customers?",
            "relevant_doc_ids": ["con", "cus"],
            "type": "multi-hop",
        },
    ]
    return docs, questions_from_records(qa)


def test_evaluate_runs_every_configuration_on_a_toy_corpus() -> None:
    docs, questions = _toy_inputs()
    report = evaluate(docs, questions, cv_seeds=(0,))
    assert list(report.scores) == [c.id for c in CONFIGS] + ["A5"]
    for per_config in report.scores.values():
        for values in per_config.values():
            assert len(values) == len(questions)
            assert all(0.0 <= v <= 1.0 for v in values)
    # Expansion reaches the TSO article from the spelled-out question; the baseline cannot.
    assert report.scores["A4"]["mrr"][0] > report.scores["A3"]["mrr"][0]
    text = render_report(report)
    assert "| A4 +expansion |" in text and "paraphrase (n=1)" in text


def test_lexical_command_loads_no_model(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    docs, _ = _toy_inputs()
    for doc_id, text in docs:
        (corpus / f"{doc_id}.md").write_text(text, encoding="utf-8")
    qa_path = tmp_path / "qa.jsonl"
    qa_path.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {
                    "q": "Who shares congestion income?",
                    "relevant_doc_ids": ["con"],
                    "type": "direct",
                },
                {
                    "q": "May customers sell electricity?",
                    "relevant_doc_ids": ["cus"],
                    "type": "direct",
                },
            ]
        ),
        encoding="utf-8",
    )
    script = (
        "import sys\n"
        "from ragkit.cli import main\n"
        f"args = ['lexical', '--corpus-dir', {str(corpus)!r}, '--qa-file', {str(qa_path)!r}]\n"
        "try:\n    main(args)\n"
        "except SystemExit as e:\n    assert e.code == 0\n"
        "banned = ('sentence_transformers', 'torch', 'transformers')\n"
        "loaded = [m for m in banned if m in sys.modules]\n"
        "assert not loaded, loaded\n"
        "print('OK')\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert "OK" in out.stdout
    assert "A4 +expansion" in out.stdout
