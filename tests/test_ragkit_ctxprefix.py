"""Tests for the model-free context prefix (offline, no models)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ragkit.chunking import Chunk, recursive
from ragkit.ctxprefix import (
    BUILDERS,
    RETRIEVERS,
    SATURATION_RECALL5,
    SetResult,
    VariantResult,
    act_of,
    baseline_recall5,
    build_chunks,
    context_prefix,
    decide,
    evaluate_set,
    heading_of,
    is_saturated,
    is_wrong_act,
    names_an_act,
    render_hard_protocol,
    render_saturation,
    render_set,
    render_verdicts,
    with_context_prefix,
    without_act_named,
)
from ragkit.lexical import METRICS, Question

HEADING_A = "Regulation (EU) 2019/943 (electricity market regulation), Article 16: General rules"
HEADING_B = "Directive (EU) 2019/944 (internal market for electricity), Article 40: Revenue tasks"


def _article(heading: str, sentences: list[str]) -> str:
    return f"# {heading}\n\n" + "\n\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))


def _filler(word: str, n: int = 12) -> str:
    return " ".join(f"{word}{i}" for i in range(n)) + " must be settled promptly by operators."


def _twins() -> list[tuple[str, str]]:
    """Two articles with identical bodies, so only the heading tells them apart."""
    words = ("alpha", "beta", "gamma", "delta", "eps", "zeta", "eta", "theta")
    body = [_filler(w, 20) for w in words]
    return [
        ("reg_2019_943_art16", _article(HEADING_A, body)),
        ("dir_2019_944_art40", _article(HEADING_B, body)),
    ]


def test_heading_of_and_context_prefix() -> None:
    assert heading_of(f"# {HEADING_A}\n\nbody") == HEADING_A
    assert heading_of("no heading here") == ""
    assert (
        context_prefix(HEADING_A)
        == "Regulation (EU) 2019/943 (electricity market regulation) — Article 16: General rules"
    )
    assert context_prefix("Plain title") == "Plain title"
    assert (
        context_prefix("Act, Article 7a: Title, with comma")
        == "Act — Article 7a: Title, with comma"
    )


def test_act_of_strips_the_article_suffix() -> None:
    assert act_of("reg_2019_943_art16") == "reg_2019_943"
    assert act_of("reg_2017_2195_art2") == "reg_2017_2195"
    assert act_of("doc_01_demand_response") == "doc_01_demand_response"


def test_prefix_reaches_every_chunk_without_doubling_the_heading() -> None:
    doc_id, text = _twins()[0]
    base = recursive(text, doc_id, chunk_size=500, overlap=50)
    assert len(base) > 2 and base[0].text.startswith("# ")
    prefixed = with_context_prefix(base, text)
    expected = context_prefix(HEADING_A)
    assert len(prefixed) == len(base)
    for before, after in zip(base, prefixed, strict=True):
        assert after.text.startswith(expected + "\n")
        assert (after.id, after.doc_id, after.start, after.end) == (
            before.id,
            before.doc_id,
            before.start,
            before.end,
        )
    assert HEADING_A not in prefixed[0].text  # replaced by the em-dash form, not duplicated
    assert prefixed[0].text.count("Article 16") == 1
    body_of_first = base[0].text.split("\n", 1)[1].lstrip("\n")
    assert prefixed[0].text == f"{expected}\n{body_of_first}"
    assert prefixed[1].text == f"{expected}\n{base[1].text}"


def test_document_without_heading_is_left_unchanged() -> None:
    chunks = [Chunk("c0", "d", "plain body text", 0, 15)]
    assert with_context_prefix(chunks, "plain body text") == chunks


def test_build_chunks_base_matches_recursive_and_prefix_keeps_boundaries() -> None:
    docs = _twins()
    base = build_chunks(docs, prefix=False)
    prefixed = build_chunks(docs, prefix=True)
    expected = [
        c for doc_id, text in docs for c in recursive(text, doc_id, chunk_size=500, overlap=50)
    ]
    assert base == expected
    assert [(c.id, c.start, c.end) for c in prefixed] == [(c.id, c.start, c.end) for c in base]
    assert all(c.text != b.text for c, b in zip(prefixed, base, strict=True))


def test_wrong_act_counts_top1_outside_the_gold_acts() -> None:
    gold = ["reg_2019_943_art16"]
    assert not is_wrong_act(["reg_2019_943_art13", "dir_2019_944_art40"], gold)
    assert is_wrong_act(["dir_2019_944_art40", "reg_2019_943_art16"], gold)
    assert is_wrong_act([], gold)
    assert not is_wrong_act(["dir_2019_944_art40"], ["reg_2019_943_art16", "dir_2019_944_art36"])


def test_prefix_separates_identical_articles_that_only_the_heading_distinguishes() -> None:
    docs = _twins()
    questions = [
        Question("alpha3 beta4 directive revenue tasks", ("dir_2019_944_art40",), "near-duplicate"),
        Question("gamma2 delta7 electricity market regulation", ("reg_2019_943_art16",), "x"),
    ]
    result = evaluate_set(docs, questions, "toy")
    for name in RETRIEVERS:
        base = result.results[(name, "base")]
        pre = result.results[(name, "prefix")]
        assert pre.scores["mrr"] == [1.0, 1.0], name
        assert pre.wrong_act == [False, False], name
        assert sum(pre.scores["mrr"]) >= sum(base.scores["mrr"]), name


def test_retrievers_are_the_pinned_model_free_constructors() -> None:
    assert set(BUILDERS) == set(RETRIEVERS) == {"bm25", "hybrid"}
    chunks = build_chunks(_twins(), prefix=True)
    for build in BUILDERS.values():
        ranked = build(chunks)("alpha3 beta4")
        assert ranked and set(ranked) <= {"reg_2019_943_art16", "dir_2019_944_art40"}


def _result(mrr_base: list[float], mrr_prefix: list[float]) -> SetResult:
    """A SetResult whose two retrievers both have the given per-question MRR."""
    n = len(mrr_base)
    results: dict[tuple[str, str], VariantResult] = {}
    for name in RETRIEVERS:
        for variant, mrr in (("base", mrr_base), ("prefix", mrr_prefix)):
            scores = {m: list(mrr) for m in METRICS}
            results[(name, variant)] = VariantResult(scores, [v < 1.0 for v in mrr])
    return SetResult("toy", [Question(f"q{i}", ("d",), "t") for i in range(n)], results)


def test_decision_rule_adopts_only_with_a_separated_fresh_gain_and_no_synthetic_loss() -> None:
    n = 30
    gain = _result([0.5] * n, [1.0] * n)  # every question improves
    flat = _result([0.5] * n, [0.5] * n)
    loss = _result([1.0] * n, [0.5] * n)
    assert {v.decision for v in decide(gain, flat)} == {"ADOPTED"}
    assert {v.decision for v in decide(flat, flat)} == {"NOT ADOPTED (no evidence)"}
    assert {v.decision for v in decide(gain, loss)} == {"REJECTED"}  # synthetic regression
    assert {v.decision for v in decide(loss, flat)} == {"REJECTED"}  # fresh CI below 0
    noisy = _result([0.5] * n, [1.0] + [0.5] * (n - 1))  # one question up: CI includes 0
    assert {v.decision for v in decide(noisy, flat)} == {"NOT ADOPTED (no evidence)"}


def test_decision_rule_tolerates_a_tiny_synthetic_drop_inside_the_margin() -> None:
    n = 40
    tiny = _result([1.0] * n, [1.0] * (n - 1) + [0.5])  # -0.0125 on the mean, CI upper bound 0
    gain = _result([0.5] * 30, [1.0] * 30)
    assert {v.decision for v in decide(gain, tiny)} == {"ADOPTED"}


def test_render_set_and_verdicts_show_the_required_columns() -> None:
    n = 30
    result = _result([0.5] * n, [1.0] * n)
    text = render_set(result, ", primary evidence")
    for token in ("Recall@5", "MRR", "nDCG@10", "wrong-act", "fixed / broken", "bm25", "hybrid"):
        assert token in text
    assert "better / worse / tied" in text
    verdicts = render_verdicts(decide(result, _result([0.5] * n, [0.5] * n)))
    assert "**ADOPTED**" in verdicts


def test_ctxprefix_command_loads_no_model(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for doc_id, text in _twins():
        (corpus / f"{doc_id}.md").write_text(text, encoding="utf-8")
    qa_path = tmp_path / "qa.jsonl"
    qa_path.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"q": "alpha3 revenue tasks", "relevant_doc_ids": ["dir_2019_944_art40"]},
                {"q": "gamma2 market regulation", "relevant_doc_ids": ["reg_2019_943_art16"]},
            ]
        ),
        encoding="utf-8",
    )
    script = (
        "import sys\n"
        "from ragkit.cli import main\n"
        f"args = ['ctxprefix', '--corpus-dir', {str(corpus)!r}, '--qa-file', {str(qa_path)!r}]\n"
        "try:\n    main(args)\n"
        "except SystemExit as e:\n    assert e.code == 0\n"
        "banned = ('sentence_transformers', 'torch', 'transformers')\n"
        "loaded = [m for m in banned if m in sys.modules]\n"
        "assert not loaded, loaded\n"
        "print('OK')\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert "OK" in out.stdout
    assert "wrong-act" in out.stdout
    assert "hybrid" in out.stdout


@pytest.mark.parametrize("name", RETRIEVERS)
def test_every_retriever_row_is_scored_for_both_variants(name: str) -> None:
    result = evaluate_set(_twins(), [Question("alpha3", ("reg_2019_943_art16",), "t")], "toy")
    assert {(name, "base"), (name, "prefix")} <= set(result.results)


def test_names_an_act_matches_act_numbers_only() -> None:
    assert names_an_act("Under Regulation 2017/2195, how long must it last?")
    assert names_an_act("the capacity regulation (2015/1222)")
    assert not names_an_act("How long after 31 December 2025 may a TSO act?")
    kept = without_act_named(
        [Question("per 2015/1222 x", ("d",), "t"), Question("plain", ("d",), "t")]
    )
    assert [q.text for q in kept] == ["plain"]


def test_protocol_command_runs_the_three_sets_with_no_model_library() -> None:
    script = (
        "import sys\n"
        "from ragkit.cli import main\n"
        "try:\n    main(['ctxprefix', '--protocol'])\n"
        "except SystemExit as e:\n    assert e.code == 0\n"
        "banned = ('sentence_transformers', 'torch', 'transformers')\n"
        "loaded = [m for m in banned if m in sys.modules]\n"
        "assert not loaded, loaded\n"
        "print('OK')\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert "OK" in out.stdout
    for token in (
        "fresh EU set (qa_eu_fresh.jsonl) (29 questions",
        "TAINTED",
        "regression guard",
        "act-named questions removed",
        "Decision rule",
    ):
        assert token in out.stdout, token


def test_saturation_guard_flags_a_baseline_recall_above_the_threshold() -> None:
    n = 20
    easy = _result([1.0] * n, [1.0] * n)  # every question solved at rank 1
    assert SATURATION_RECALL5 == 0.85
    assert baseline_recall5(easy) == 1.0 and is_saturated(easy)
    assert "SATURATED" in render_saturation(easy)
    hard = _result([0.0] * 12 + [1.0] * 8, [0.0] * 12 + [1.0] * 8)  # Recall@5 = 0.4
    assert baseline_recall5(hard) == 0.4 and not is_saturated(hard)
    assert "not saturated" in render_saturation(hard)
    edge = _result([0.0] * 3 + [1.0] * 17, [1.0] * 20)  # exactly 0.85 is not above it
    assert baseline_recall5(edge) == 0.85 and not is_saturated(edge)


def test_hard_report_has_guard_synthetic_and_the_unchanged_rule() -> None:
    n = 30
    report = render_hard_protocol(_result([0.5] * n, [0.5] * n), _result([0.5] * n, [0.5] * n))
    for token in (
        "Saturation guard",
        "not saturated",
        "regression guard",
        "docs/ctxprefix_hard_protocol.md",
        "hard dMRR",
        "NOT ADOPTED (no evidence)",
    ):
        assert token in report, token
