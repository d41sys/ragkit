"""Tests for the neural engine and runners with fake models (offline, no model library)."""

import json
import subprocess
import sys
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ragkit.analysis import lucene_english
from ragkit.bm25 import BM25Index
from ragkit.chunking import Chunk
from ragkit.embed import HashingEmbedder
from ragkit.hybrid import rrf
from ragkit.lexical import Question
from ragkit.neural import (
    ALPHA_GRID,
    LEVEL_SYSTEMS,
    DenseSide,
    NeuralEngine,
    alpha_of_best,
    convex_fuse,
    embed_corpus,
    rerank_head,
    rrf_fuse,
)
from ragkit.neuralbench import (
    beir_mean_ci,
    pool,
    recommendation,
    render_beir_mean,
    render_eu_neural,
    render_neural_beir,
    run_beir_neural,
    run_eu_neural,
)

TEXTS = [
    "cats are small domestic animals that purr",
    "dogs are loyal domestic animals that bark",
    "the stock market fell sharply on monday",
    "stock markets rise and fall with interest rates",
    "grid congestion management and redispatching rules",
    "balancing energy platforms for frequency restoration",
]


class FakeEmbedder:
    """Hashing embedder behind the ``SentenceEmbedder`` surface (no model library)."""

    model_id = "fake@0"

    def __init__(self, texts: Sequence[str]) -> None:
        self.inner = HashingEmbedder(dim=128).fit(list(texts))

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self.inner.embed(list(texts))

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self.inner.embed_query(t) for t in texts])


class FakeCrossEncoder:
    """Query-term overlap as a relevance logit; counts the pairs it is asked to score."""

    scorer_id = "fake-ce"

    def __init__(self) -> None:
        self.pairs = 0

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        self.pairs += len(passages)
        terms = set(question.lower().split())
        return [float(len(terms & set(p.lower().split()))) for p in passages]


def _chunks(texts: Sequence[str]) -> list[Chunk]:
    return [
        Chunk(id=f"d{i}", doc_id=f"d{i}", text=t, start=0, end=len(t)) for i, t in enumerate(texts)
    ]


def _engine(texts: Sequence[str] = TEXTS, ce: FakeCrossEncoder | None = None) -> NeuralEngine:
    chunks = _chunks(texts)
    bm25 = BM25Index(k1=0.9, b=0.4, analyzer=lucene_english)
    bm25.add(chunks)
    emb = FakeEmbedder(texts)
    sides = {
        "bge": DenseSide(emb, emb.embed(texts)),  # type: ignore[arg-type]
        "e5": DenseSide(emb, emb.embed(texts)),  # type: ignore[arg-type]
    }
    return NeuralEngine(chunks, bm25, sides, ce)


def test_rrf_fuse_matches_hybrid_rrf() -> None:
    chunks = _chunks(TEXTS)
    a = [(0, 3.0), (1, 2.0), (2, 1.0)]
    b = [(2, 0.9), (0, 0.8), (3, 0.1)]
    expected = rrf([[(chunks[i], s) for i, s in r] for r in (a, b)], k=60)
    got = rrf_fuse([a, b], k=60)
    assert [chunks[i].id for i, _ in got] == [c.id for c, _ in expected]
    assert [s for _, s in got] == pytest.approx([s for _, s in expected])


def test_convex_fuse_endpoints_and_missing_scores() -> None:
    a = [(0, 10.0), (1, 5.0), (2, 0.0)]
    b = [(2, 0.9), (3, 0.5), (0, 0.1)]
    assert [i for i, _ in convex_fuse(a, b, 1.0)][:2] == [0, 1]
    assert [i for i, _ in convex_fuse(a, b, 0.0)][:2] == [2, 3]
    fused = dict(convex_fuse(a, b, 0.5))
    assert fused[1] == pytest.approx(0.5 * 0.5)  # missing from b: scores 0 there
    assert fused[3] == pytest.approx(0.5 * 0.5)  # (0.5 - 0.1) / 0.8 = 0.5, missing from a


def test_rerank_head_keeps_the_tail_and_the_candidate_set() -> None:
    first = [(0, 9.0), (1, 8.0), (2, 7.0), (3, 6.0)]
    out = rerank_head(first, {0: 0.1, 1: 0.9, 2: 0.5, 3: 5.0}, depth=3)
    assert [i for i, _ in out] == [1, 2, 0, 3]  # doc 3 is below the depth: not promoted
    assert {i for i, _ in out} == {0, 1, 2, 3}


def test_alpha_of_best_breaks_ties_towards_one_half_then_smaller() -> None:
    assert alpha_of_best({0.2: 0.5, 0.5: 0.5, 0.8: 0.5}) == 0.5
    assert alpha_of_best({0.3: 0.6, 0.7: 0.6, 0.0: 0.1}) == 0.3
    assert alpha_of_best({0.0: 0.2, 1.0: 0.9}) == 1.0


def test_engine_systems_and_ce_depth_reuse_scores() -> None:
    ce = FakeCrossEncoder()
    engine = _engine(ce=ce)
    engine.prepare(["domestic animals"])
    cands = engine.candidates("domestic animals")
    for name in ("bm25", "dense", "dense-e5", "hybrid"):
        assert engine.system(name, "domestic animals", cands)
    assert engine.system("hybrid-convex", "domestic animals", cands, alpha=0.5)
    with pytest.raises(ValueError, match="alpha"):
        engine.system("hybrid-convex", "domestic animals", cands)
    full = engine.system("hybrid+ce@100", "domestic animals", cands)
    scored = ce.pairs
    top2 = engine.system("hybrid+ce@2", "domestic animals", cands)
    assert ce.pairs == scored  # a shallower depth reuses the cached pair scores
    assert {i for i, _ in full} == {i for i, _ in top2}
    assert [i for i, _ in full][:2] == [0, 1]  # both animal documents lead after the rerank


def test_engine_removes_the_self_hit() -> None:
    engine = _engine()
    engine.prepare(["cats purr"])
    cands = engine.candidates("cats purr", exclude_id="d0")
    assert all(i != 0 for i, _ in cands.bm25)
    assert all(i != 0 for hits in cands.dense.values() for i, _ in hits)


def test_dense_search_is_cosine_ranked_and_complete() -> None:
    engine = _engine()
    side = engine.dense["bge"]
    hits = side.search("stock market", 3)
    assert len(hits) == 3
    assert hits[0][1] >= hits[1][1] >= hits[2][1]
    assert len(side.search("stock market", 100)) == len(TEXTS)


def test_embed_corpus_caches_by_text_digest(tmp_path: Path) -> None:
    emb = FakeEmbedder(TEXTS)
    first = embed_corpus(emb, TEXTS, tmp_path)  # type: ignore[arg-type]

    class Boom(FakeEmbedder):
        def embed(self, texts: Sequence[str]) -> np.ndarray:
            raise AssertionError("must load from the cache")

    again = embed_corpus(Boom(TEXTS), TEXTS, tmp_path)  # type: ignore[arg-type]
    assert np.array_equal(first, again)
    other = embed_corpus(emb, [*TEXTS[:-1], "different"], tmp_path)  # type: ignore[arg-type]
    assert other.shape == first.shape and len(list(tmp_path.glob("*.npy"))) == 2


def test_level_systems_are_nested() -> None:
    assert set(LEVEL_SYSTEMS["bm25"]) <= set(LEVEL_SYSTEMS["dense"]) <= set(LEVEL_SYSTEMS["hybrid"])
    assert set(LEVEL_SYSTEMS["hybrid"]) <= set(LEVEL_SYSTEMS["hybrid+ce"])
    assert len(ALPHA_GRID) == 11


def _patch_engine(monkeypatch: pytest.MonkeyPatch, ce: bool = True) -> None:
    def build(chunks: Sequence[Chunk], bm25: BM25Index, level: str, **_: object) -> NeuralEngine:
        texts = [c.text for c in chunks]
        emb = FakeEmbedder(texts)
        sides = {k: DenseSide(emb, emb.embed(texts)) for k in ("bge", "e5")}  # type: ignore[arg-type]
        return NeuralEngine(chunks, bm25, sides, FakeCrossEncoder() if ce else None)

    monkeypatch.setattr("ragkit.neuralbench.build_engine", build)


def _make_beir_zip(root: Path) -> None:
    docs = [{"_id": f"d{i}", "title": "", "text": t, "metadata": {}} for i, t in enumerate(TEXTS)]
    queries = [("q1", "domestic animals"), ("q2", "stock market"), ("q3", "grid congestion")]
    with zipfile.ZipFile(root / "scifact.zip", "w") as zf:
        zf.writestr("scifact/corpus.jsonl", "\n".join(json.dumps(d) for d in docs))
        zf.writestr(
            "scifact/queries.jsonl",
            "\n".join(json.dumps({"_id": q, "text": t}) for q, t in queries),
        )
        head = "query-id\tcorpus-id\tscore\n"
        zf.writestr("scifact/qrels/test.tsv", head + "q1\td0\t1\nq1\td1\t1\nq2\td2\t1\n")
        zf.writestr("scifact/qrels/train.tsv", head + "q3\td4\t1\n")


def test_beir_neural_run_picks_alpha_on_train_and_scores_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_beir_zip(tmp_path)
    _patch_engine(monkeypatch)
    run = run_beir_neural("scifact", tmp_path, "hybrid+ce", download=False)
    assert run.alpha is not None and run.dev_name == "scifact train"
    assert set(run.results) == set(LEVEL_SYSTEMS["hybrid+ce"])
    assert all(len(r.ndcg) == 2 for r in run.results.values())  # q1, q2 (q3 is train only)
    text = render_neural_beir(run)
    assert "Reproduction check" in text and "bge-small-en-v1.5 (MTEB)" in text
    assert "chosen on scifact train" in text


def test_beir_neural_arguana_like_dataset_has_no_convex_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_beir_zip(tmp_path)
    (tmp_path / "scifact.zip").rename(tmp_path / "arguana.zip")
    with zipfile.ZipFile(tmp_path / "arguana.zip") as zf:  # re-pack under the arguana prefix
        members = {n.replace("scifact/", "arguana/"): zf.read(n) for n in zf.namelist()}
    with zipfile.ZipFile(tmp_path / "arguana.zip", "w") as zf:
        for name, blob in members.items():
            zf.writestr(name, blob)
    _patch_engine(monkeypatch)
    run = run_beir_neural("arguana", tmp_path, "hybrid", download=False)
    assert run.alpha is None and "hybrid-convex" not in run.results
    assert "RRF only" in render_neural_beir(run)


def _eu_docs() -> list[tuple[str, str]]:
    return [
        (f"reg_2019_943_art{i}", f"# Reg, Article {i}: T\n\n{t}") for i, t in enumerate(TEXTS, 1)
    ]


def test_eu_run_scores_dev_and_held_out_sets_and_applies_the_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_engine(monkeypatch)
    dev = [Question("domestic animals", ("reg_2019_943_art1",), "direct")]
    sets = {
        "fresh": [Question("stock market", ("reg_2019_943_art3",), "direct")],
        "hard": [Question("grid congestion", ("reg_2019_943_art5",), "direct")],
    }
    run = run_eu_neural(_eu_docs(), dev, sets, "hybrid+ce")
    assert run.alpha is not None
    assert set(run.dev) == set(LEVEL_SYSTEMS["hybrid+ce"])
    pooled = pool(list(run.sets.values()))
    assert len(pooled["bm25"].scores["mrr"]) == 2 and len(pooled["bm25"].wrong_act) == 2
    ok, text = recommendation(pooled, "hybrid")
    assert isinstance(ok, bool) and "wrong-act" in text
    report = render_eu_neural(run)
    assert "pooled held-out sets" in report and "dev only, not a test" in report


def test_importing_neural_modules_loads_no_model_library() -> None:
    script = (
        "import sys\n"
        "import ragkit.neural, ragkit.neuralbench, ragkit.cli\n"
        "banned = ('sentence_transformers', 'torch', 'transformers')\n"
        "loaded = [m for m in banned if m in sys.modules]\n"
        "assert not loaded, loaded\n"
        "print('OK')\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert "OK" in out.stdout


def test_beir_mean_ci_is_the_dataset_mean_with_a_paired_interval() -> None:
    from types import SimpleNamespace

    from ragkit.beir import SystemResult

    def run(name: str, base: list[float], better: list[float]) -> Any:
        return SimpleNamespace(
            data=SimpleNamespace(name=name),
            results={"bm25": SystemResult("bm25", base), "dense": SystemResult("dense", better)},
        )

    runs = [run("a", [0.2, 0.4, 0.6, 0.8], [0.3, 0.5, 0.7, 0.9]), run("b", [0.1, 0.1], [0.4, 0.2])]
    mean, lo, hi = beir_mean_ci(runs, "dense")
    assert mean == pytest.approx((0.6 + 0.3) / 2)
    assert lo <= mean <= hi
    d, dlo, dhi = beir_mean_ci(runs, "dense", "bm25")
    assert d == pytest.approx((0.1 + 0.2) / 2)
    assert dlo <= d <= dhi and dlo > 0  # every query improves: the paired interval excludes 0
    text = render_beir_mean(runs)
    assert "mean over 2 datasets (a, b)" in text and "| dense |" in text and "baseline" in text
