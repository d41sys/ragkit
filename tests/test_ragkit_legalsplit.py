"""LegalBench-RAG-mini query split and document routing (docs/legalsplit_protocol.md)."""

from __future__ import annotations

import re

import numpy as np
import pytest

from ragkit.analysis import lucene_english
from ragkit.bm25 import BM25Index
from ragkit.chunking import Chunk
from ragkit.legalbench import KS, SOURCES, LegalResult, MiniBenchmark, MiniTest
from ragkit.legalsplit import (
    ARM_A,
    BASELINE,
    Cell,
    DocRouter,
    Tuning,
    arm_b,
    choose_profile,
    chosen_from_json,
    dev_test_split,
    doc_profile,
    evaluate,
    overall_verdict,
    paired_all_ci,
    routing_accuracy,
    split_query,
    verdict,
)
from ragkit.neural import DenseSide

DOC_TEXT = {
    "Acme": "Agreement with Acme. Payment due in thirty days. Governing law Delaware.",
    "Zeta": "Agreement with Zeta. Payment due in sixty days. Governing law Delaware.",
}


def _toy() -> tuple[MiniBenchmark, list[Chunk]]:
    """Four sources, two documents each; every query is about the Zeta document of its source."""
    corpus: dict[str, str] = {}
    chunks: list[Chunk] = []
    tests: list[MiniTest] = []
    for n, source in enumerate(SOURCES):
        for name, template in DOC_TEXT.items():
            path = f"{source}/{name}{n}.txt"
            text = template.replace(name, f"{name}{n}")
            corpus[path] = text
            for m in re.finditer(r"[^.]+\.", text):
                chunks.append(Chunk(f"{path}#{m.start()}", path, m.group(), m.start(), m.end()))
        text = corpus[f"{source}/Zeta{n}.txt"]
        start = text.index("Governing")
        tests.append(
            MiniTest(
                source,
                f"Consider the Agreement between Zeta{n} and Corp; Governing law Delaware",
                ((f"{source}/Zeta{n}.txt", start, start + len("Governing law Delaware.")),),
            )
        )
    return MiniBenchmark(tests=tests, corpus=dict(sorted(corpus.items()))), chunks


def _bm25_cell(chunks: list[Chunk]) -> Cell:
    index = BM25Index(k1=0.9, b=0.4, analyzer=lucene_english)
    index.add(chunks)
    return Cell("bm25", "toy", chunks, bm25=index)


def test_split_query_cuts_at_the_first_semicolon_and_drops_consider() -> None:
    assert split_query("Consider the Lease between A and B; What is the term?") == (
        "the Lease between A and B",
        "What is the term?",
    )
    assert split_query('Consider "Fiverr"\'s privacy policy; who sees tasks?') == (
        '"Fiverr"\'s privacy policy',
        "who sees tasks?",
    )
    assert split_query("Consider X; first; second") == ("X", "first; second")
    assert split_query("  no separator here ") == ("", "no separator here")


def test_dev_test_split_is_per_source_disjoint_and_seed_fixed() -> None:
    tests = [MiniTest(s, f"q{i}", (("f", 0, 1),)) for s in SOURCES for i in range(194)]
    dev, test = dev_test_split(tests)
    assert (len(dev), len(test)) == (156, 620)
    assert not set(dev) & set(test)
    assert sorted(dev + test) == list(range(len(tests)))
    for source in SOURCES:
        assert sum(1 for i in dev if tests[i].source == source) == 39
    assert dev_test_split(tests) == (dev, test)
    # not simply the first 39 of each source
    assert dev != [i for s in range(4) for i in range(s * 194, s * 194 + 39)]


def test_doc_profile_variants_and_unknown_profile() -> None:
    text = "x" * 3000
    assert doc_profile("cuad/Foo Bar.txt", text, "name") == "Foo Bar"
    assert doc_profile("cuad/Foo.txt", text, "name+open500") == "Foo\n" + "x" * 500
    assert doc_profile("cuad/Foo.txt", text, "name+open2000") == "Foo\n" + "x" * 2000
    assert doc_profile("cuad/Foo.txt", text, "open2000") == "x" * 2000
    with pytest.raises(ValueError):
        doc_profile("a.txt", text, "nope")


def test_router_ranks_by_name_and_returns_nothing_without_a_match() -> None:
    bench, _ = _toy()
    router = DocRouter(bench.corpus, "name")
    assert router.rank("the Agreement between Zeta2 and Corp")[0].endswith("/Zeta2.txt")
    assert router.rank("completely unrelated words") == []
    # the opening text matches what the file name does not say
    corpus = {"a/x1.txt": "Lease with Orion Holdings", "a/x2.txt": "Lease with Vega Holdings"}
    assert DocRouter(corpus, "name").rank("Vega") == []
    assert DocRouter(corpus, "name+open500").rank("Vega") == ["a/x2.txt"]


def test_routing_accuracy_and_profile_tie_goes_to_the_first_profile() -> None:
    bench, _ = _toy()
    dev = list(range(len(bench.tests)))
    acc = routing_accuracy(DocRouter(bench.corpus, "name"), bench.tests, dev)
    assert acc["ALL"][0] == 1.0 and acc["ALL"][1] == 1.0
    best, table = choose_profile(bench, dev)
    assert table["name"]["top1"] == 1.0
    assert best == "name"  # every profile that scores 1.0 loses the tie to the earliest


def test_routed_arm_searches_only_the_routed_document() -> None:
    bench, chunks = _toy()
    cell = _bm25_cell(chunks)
    router = DocRouter(bench.corpus, "name")
    indices = list(range(len(bench.tests)))
    base = evaluate(cell, bench.tests, indices, BASELINE, router).result
    routed = evaluate(cell, bench.tests, indices, arm_b(1, "question"), router)
    k1 = KS.index(1)
    assert all(row[k1] == 1.0 for s in SOURCES for row in routed.result.recall[s])
    assert any(row[k1] == 0.0 for s in SOURCES for row in base.recall[s])
    assert routed.fallbacks == 0


def test_routed_arm_falls_back_to_the_full_query_when_no_document_matches() -> None:
    bench, chunks = _toy()
    cell = _bm25_cell(chunks)
    router = DocRouter(bench.corpus, "name")
    test = MiniTest("cuad", "Consider the qqq between www and eee; Governing law Delaware", ())
    tests = [test]
    run = evaluate(cell, tests, [0], arm_b(1, "question"), router)
    assert run.fallbacks == 1
    # fallback searches every document with the full query: same hits as the baseline
    base = evaluate(cell, tests, [0], BASELINE, router)
    assert run.result.precision == base.result.precision
    with pytest.raises(ValueError):
        evaluate(cell, tests, [0], arm_b(1, "full"), None)


def test_arm_a_drops_the_prefix_from_the_query() -> None:
    chunks = [
        Chunk("a", "d1", "Consider the Agreement between Zeta and Corp", 0, 43),
        Chunk("b", "d2", "Governing law Delaware", 0, 22),
    ]
    cell = _bm25_cell(chunks)
    test = MiniTest(
        "cuad", "Consider the Agreement between Zeta and Corp; Governing law", (("d2", 0, 22),)
    )
    full = evaluate(cell, [test], [0], BASELINE).result.recall["cuad"][0][0]
    question = evaluate(cell, [test], [0], ARM_A).result.recall["cuad"][0][0]
    assert (full, question) == (0.0, 1.0)


def test_dense_cell_masks_to_routed_documents_with_index_tiebreak() -> None:
    class FakeEmbedder:
        def embed_queries(self, texts: list[str]) -> np.ndarray:
            return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    chunks = [Chunk(f"c{i}", f"d{i % 2}", "t", i, i + 1) for i in range(6)]
    matrix = np.asarray(
        [[1, 0], [1, 0], [0.6, 0.8], [0.9, 0.1], [0, 1], [0.6, 0.8]], dtype=np.float32
    )
    cell = Cell("dense-e5", "toy", chunks, dense=DenseSide(FakeEmbedder(), matrix))  # type: ignore[arg-type]
    everything = cell.rank("q", None)
    assert [s[1] for s in everything] == [0, 1, 3, 2, 5, 4]  # cosine desc, index asc on ties
    only_d1 = cell.rank("q", ["d1"])
    assert [s[1] for s in only_d1] == [1, 3, 5]
    assert {s[0] for s in only_d1} == {"d1"}


def _result(rows: dict[str, list[float]]) -> LegalResult:
    result = LegalResult("x")
    for source, values in rows.items():
        result.recall[source] = [[v] * len(KS) for v in values]
    return result


def test_paired_ci_is_zero_for_equal_systems_and_positive_for_a_clear_gain() -> None:
    base = _result({s: [0.1, 0.2, 0.3, 0.4] * 10 for s in SOURCES})
    mean, lo, hi = paired_all_ci(base, base)
    assert mean == lo == hi == 0.0
    better = _result({s: [v + 0.1 for v in [0.1, 0.2, 0.3, 0.4] * 10] for s in SOURCES})
    mean, lo, hi = paired_all_ci(better, base)
    assert mean == pytest.approx(10.0)
    assert lo > 0.0 and hi >= lo


def test_verdict_follows_the_preregistered_rule() -> None:
    assert verdict(0.01, 2.0) == "improvement"
    assert verdict(0.0, 2.0) == "no detectable effect"  # lower bound must be above 0
    assert verdict(-1.0, 1.0) == "no detectable effect"
    assert verdict(-2.0, -0.01) == "worse"
    both = {("bm25", "naive500"): "improvement", ("dense-e5", "recursive500"): "improvement"}
    assert overall_verdict(both) == "improvement overall (both primary cells)"
    one = {**both, ("dense-e5", "recursive500"): "no detectable effect"}
    assert overall_verdict(one) == "improvement in bm25-naive500 only"
    none = dict.fromkeys(both, "no detectable effect")
    assert overall_verdict(none) == "no detectable effect"


def test_tuning_json_round_trips_the_chosen_settings() -> None:
    tuning = Tuning(
        "name+open500",
        {"name": {"top1": 0.5, "top2": 0.6}},
        {"bm25": (1, "question"), "dense-e5": (2, "full")},
        {"bm25": {"n=1,question": 1.0}},
        156,
    )
    profile, settings = chosen_from_json(tuning.to_json())
    assert profile == "name+open500"
    assert settings == {"bm25": (1, "question"), "dense-e5": (2, "full")}
