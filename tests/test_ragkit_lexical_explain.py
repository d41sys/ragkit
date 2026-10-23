"""Tests for the post-hoc lexical diagnosis (offline, no models)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ragkit.cli import BUNDLED_DATASETS, _load_documents, _read_jsonl
from ragkit.lexical import CONFIGS, Question, questions_from_records, score_retriever
from ragkit.lexical_explain import (
    Explanation,
    classify_crossing,
    explain,
    render_explanation,
    split_title,
)


@pytest.fixture(scope="module")
def eu() -> tuple[list[tuple[str, str]], list[Question], Explanation]:
    dataset = BUNDLED_DATASETS["eu"]
    docs = _load_documents(dataset.corpus_dir)
    questions = questions_from_records(_read_jsonl(dataset.qa_path))
    return docs, questions, explain(docs, questions)


def test_stage_rankings_equal_the_shipped_retrievers(
    eu: tuple[list[tuple[str, str]], list[Question], Explanation],
) -> None:
    docs, _, expl = eu
    for config in CONFIGS:
        retriever = config.build(docs)
        for run in expl.runs:
            expected = retriever(run.question.text)
            assert [h.doc_id for h in run.hits[config.id][:10]] == expected, config.id


def test_flips_account_for_the_recall_change(
    eu: tuple[list[tuple[str, str]], list[Question], Explanation],
) -> None:
    docs, questions, expl = eu
    by_id = {c.id: c for c in CONFIGS}
    r5_a0 = score_retriever(by_id["A0"].build(docs), questions)["r5"]
    r5_a4 = score_retriever(by_id["A4"].build(docs), questions)["r5"]
    delta = [0.0] * len(questions)
    for pair in expl.flips:
        n = len(questions[pair.question].relevant)
        delta[pair.question] += (1.0 if pair.net == "gained" else -1.0) / n
    for got, a0, a4 in zip(delta, r5_a0, r5_a4, strict=True):
        assert got == pytest.approx(a4 - a0)
    # The numbers quoted in the README: 3 lost, 1 gained, -0.060 Recall@5 in total.
    assert sum(p.net == "lost" for p in expl.flips) == 3
    assert sum(p.net == "gained" for p in expl.flips) == 1
    assert sum(delta) / len(questions) == pytest.approx(-0.0595, abs=1e-4)


def test_every_flip_has_a_deciding_crossing(
    eu: tuple[list[tuple[str, str]], list[Question], Explanation],
) -> None:
    _, _, expl = eu
    for pair in expl.flips:
        assert pair.crossings
        assert pair.crossings[-1].gained == (pair.net == "gained")
    # Expansion never moves a relevant article across the top-5 line on this set.
    assert not [c for p in expl.pairs for c in p.crossings if c.transition == ("A3", "A4")]


def test_classify_crossing_picks_the_side_that_moved_more() -> None:
    chunking = ("A1", "S0")
    # Loss: relevant fell 0.20, movers rose 0.05 -> the relevant article's score moved.
    assert classify_crossing(chunking, False, 0.8, 0.6, [0.7], [0.75]) == "relevant"
    # Loss: relevant fell 0.02, movers rose 0.10 -> competitors moved.
    assert classify_crossing(chunking, False, 0.8, 0.78, [0.7], [0.8]) == "competitors"
    # Gain: relevant rose 0.1 while movers fell 0.02 -> relevant; the reverse -> competitors.
    assert classify_crossing(chunking, True, 0.6, 0.7, [0.7], [0.68]) == "relevant"
    assert classify_crossing(chunking, True, 0.6, 0.61, [0.7], [0.6]) == "competitors"
    # The A0 cut has no side.
    assert classify_crossing(("A0", "A1"), True, 0.0, 0.5, [0.9], [0.9]) == ""


def test_split_title() -> None:
    title = "Regulation (EU) 2019/943 (electricity market regulation), Article 16: General rules"
    assert split_title(title) == (
        "Regulation (EU) 2019/943 (electricity market regulation)",
        "General rules",
    )
    assert split_title("Plain heading") == ("", "Plain heading")


def test_render_explanation_has_every_section(
    eu: tuple[list[tuple[str, str]], list[Question], Explanation],
) -> None:
    _, _, expl = eu
    text = render_explanation(expl)
    for needle in (
        "### Flipped pairs (A0 vs A4)",
        "### Cause table",
        "### Crossings by stage",
        "### Where the relevant articles sat under A0",
        "### How flat the scores are near the top-5 line",
        "### Does the heading field match the question?",
        "top-3 competing articles under A4:",
    ):
        assert needle in text
    assert text.count("#### Q") == len(expl.flips)


def test_explain_command_loads_no_model(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "bal.md").write_text(
        "# Reg (EU) 1, Article 1: Balancing\n\n1. The TSO shall procure balancing energy "
        "from balancing service providers every day.\n",
        encoding="utf-8",
    )
    (corpus / "con.md").write_text(
        "# Reg (EU) 1, Article 2: Congestion\n\n1. Congestion income is shared between "
        "operators according to the methodology.\n",
        encoding="utf-8",
    )
    qa_path = tmp_path / "qa.jsonl"
    qa_path.write_text(
        json.dumps({"q": "Who shares congestion income?", "relevant_doc_ids": ["con"]}),
        encoding="utf-8",
    )
    script = (
        "import sys\n"
        "from ragkit.cli import main\n"
        f"args = ['lexical', '--explain', '--corpus-dir', {str(corpus)!r}, "
        f"'--qa-file', {str(qa_path)!r}]\n"
        "try:\n    main(args)\n"
        "except SystemExit as e:\n    assert e.code == 0\n"
        "banned = ('sentence_transformers', 'torch', 'transformers')\n"
        "loaded = [m for m in banned if m in sys.modules]\n"
        "assert not loaded, loaded\n"
        "print('OK')\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert "OK" in out.stdout
    assert "### Cause table" in out.stdout
