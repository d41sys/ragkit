"""Tests for ``ragkit.search``: the library API, its serving guarantees and the fallbacks."""

import json
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ragkit import cli, neural
from ragkit import search as search_fn
from ragkit.abstention import best_threshold, cross_validated_accept
from ragkit.bm25 import BM25Index
from ragkit.chunking import Chunk
from ragkit.embed import HashingEmbedder
from ragkit.genbench import (
    items_from_records,
    offline_reader_scorers,
    reader_scores,
    retrieve_passages,
)
from ragkit.lexical import questions_from_records
from ragkit.neural import DenseSide, NeuralChunkRetriever, NeuralEngine, embed_corpus
from ragkit.parent import Passage
from ragkit.pipeline import RAGPipeline
from ragkit.search import (
    ANSWERABLE_THRESHOLDS,
    MAX_QUERY_CHARS,
    SearchResult,
    _entries,
    _Entry,
)

ROOT = Path(__file__).parent.parent

# Context of ``ragkit answer`` before it called ``search()``: ids of the packed articles for the
# default retriever on corpus_eu (12 000-token budget, 10 chunks, ``stop``), captured from the
# pre-refactor ``cmd_answer`` code path.
CONGESTION_Q = "What must a transmission system operator do about congestion between bidding zones?"
CONGESTION_IDS = [
    "reg_2019_943_art16",
    "reg_2017_2195_art39",
    "reg_2019_943_art2",
    "reg_2019_943_art42",
    "reg_2019_943_art15",
    "dir_2019_944_art40",
]
CAPACITY_Q = (
    "How are capacity allocation and congestion management principles stated in the "
    "electricity regulation?"
)
CAPACITY_IDS = [
    "reg_2019_943_art3",
    "reg_2019_943_art16",
    "reg_2015_1222_art73",
    "reg_2019_943_art14",
    "reg_2015_1222_art74",
    "reg_2015_1222_art72",
]


@pytest.fixture(autouse=True)
def fresh_entries() -> Any:
    saved = dict(_entries)
    _entries.clear()
    yield
    _entries.clear()
    _entries.update(saved)


def test_import_leaves_llm_modules_out() -> None:
    code = (
        "import sys, ragkit, ragkit.search\n"
        "bad = [m for m in ('ragkit.generation', 'ragkit.llm', 'ragkit.hardset_review', 'torch')"
        " if m in sys.modules]\n"
        "assert callable(ragkit.search), 'ragkit.search must stay the function'\n"
        "assert not bad, bad\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_rejects_bad_input() -> None:
    for bad in ("", "   \n\t"):
        with pytest.raises(ValueError, match="empty"):
            search_fn(bad, retriever="bm25")
    with pytest.raises(ValueError, match="limit"):
        search_fn("x" * (MAX_QUERY_CHARS + 1), retriever="bm25")
    search_fn("x" * MAX_QUERY_CHARS, retriever="bm25")  # the cap itself is allowed
    with pytest.raises(ValueError, match="retriever"):
        search_fn("congestion", retriever="magic")
    with pytest.raises(ValueError, match="pack_chunks"):
        search_fn("congestion", retriever="bm25", pack_chunks=0)


def _corpus(tmp_path: Path, consolidated: bool) -> Path:
    acts: list[dict[str, Any]] = [
        {
            "short": "reg_2019_943",
            "celex": "32019R0943",
            "title": "Regulation (EU) 2019/943 (electricity market regulation)",
            "url": "https://example.org/reg",
            "retrieved_on": "2026-10-16",
        },
        {
            "short": "dir_2019_944",
            "celex": "32019L0944",
            "title": "Directive (EU) 2019/944",
            "url": "https://example.org/dir",
            "retrieved_on": "2026-10-16",
        },
    ]
    if consolidated:
        acts[0]["consolidated_date"] = "2024-07-16"
    (tmp_path / "SOURCES.json").write_text(json.dumps({"acts": acts}))
    (tmp_path / "reg_2019_943_art16.md").write_text(
        "<!-- source: x -->\n\n# Regulation (EU) 2019/943 (electricity market regulation), "
        "Article 16: Congestion management\n\nCongestion shall be addressed by market solutions."
    )
    (tmp_path / "dir_2019_944_art3.md").write_text(
        "# Directive (EU) 2019/944, Article 3: Market rules\n\nConsumers may choose suppliers."
    )
    return tmp_path


@pytest.mark.parametrize("consolidated", [False, True])
def test_result_is_citable_and_json(tmp_path: Path, consolidated: bool) -> None:
    result = search_fn(
        "congestion market solutions", corpus_dir=_corpus(tmp_path, consolidated), retriever="bm25"
    )
    assert isinstance(result, SearchResult)
    assert result.retriever == "bm25"
    first = result.articles[0]
    assert (first.id, first.rank, first.article) == ("reg_2019_943_art16", 1, "16")
    assert first.act == "Regulation (EU) 2019/943 (electricity market regulation)"
    assert first.heading == "Congestion management"
    assert (first.celex, first.url) == ("32019R0943", "https://example.org/reg")
    assert first.version == (
        "consolidated version of 2024-07-16"
        if consolidated
        else "original Official Journal text (retrieved 2026-10-16)"
    )
    assert result.estimated_tokens == sum(
        -(-len(a.text) // 4) for a in result.articles
    )  # parent.estimate_tokens
    assert json.loads(json.dumps(result.to_dict()))["articles"][0]["id"] == first.id
    assert result.answerable_score is None


def test_corpus_without_sources_still_searches(tmp_path: Path) -> None:
    (tmp_path / "a_1.md").write_text("# Heating\n\nHeat pumps shift load.")
    only = search_fn("heat pumps", corpus_dir=tmp_path, retriever="bm25").articles[0]
    assert (only.act, only.celex, only.url, only.version) == ("a_1", None, None, None)
    assert only.heading == "Heating"


def test_search_matches_the_answer_context() -> None:
    """``search()`` returns the article ids of the pre-refactor ``ragkit answer`` context."""
    for question, expected in ((CONGESTION_Q, CONGESTION_IDS), (CAPACITY_Q, CAPACITY_IDS)):
        result = search_fn(question, corpus_dir=cli.EU_CORPUS_DIR, retriever="default")
        assert [a.id for a in result.articles] == expected
        assert result.retriever == "default"


def test_cmd_answer_context_comes_from_search(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    class Recorder:
        def __init__(self, client: Any) -> None: ...

        def generate(self, question: str, passages: Sequence[Passage]) -> Any:
            seen.append([p.article_id for p in passages])
            return type("A", (), {"answer": "a", "citations": (), "supports": True})()

    monkeypatch.setattr(cli, "LLMAnswerer", Recorder)
    monkeypatch.setattr(cli, "_llm_client", lambda model, cache: None)
    with pytest.raises(SystemExit):
        cli.main(["answer", CONGESTION_Q, "--corpus", "eu", "--retriever", "default"])
    assert seen == [CONGESTION_IDS]


def test_pipeline_is_built_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = sys.modules["ragkit.search"]  # ``ragkit.search`` itself is the function
    built: list[tuple[Path, str]] = []
    real = module._build_entry
    monkeypatch.setattr(module, "_build_entry", lambda d, r: built.append((d, r)) or real(d, r))
    corpus = _corpus(tmp_path, False)
    for _ in range(3):
        search_fn("congestion", corpus_dir=corpus, retriever="bm25")
    assert len(built) == 1


def test_falls_back_to_bm25_when_models_cannot_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    def offline(self: Any) -> Any:
        raise OSError("offline and no local copy")

    monkeypatch.setattr(neural.SentenceEmbedder, "_load", offline)
    corpus = _corpus(tmp_path, False)
    for name in ("hybrid+ce", "dense"):
        result = search_fn("congestion market solutions", corpus_dir=corpus, retriever=name)
        assert result.retriever == "bm25"
        assert result.articles[0].id == "reg_2019_943_art16"
    assert "falling back to bm25" in caplog.text


# --- serving: bounded caches, one lock, same answers under threads ------------------------------

TEXTS = [
    f"article {i} on {w} and congestion rule {i % 7}"
    for i, w in enumerate(
        ["balancing", "capacity", "storage", "demand", "redispatch", "tariffs"] * 6
    )
]


class FakeEmbedder:
    model_id = "fake@0"

    def __init__(self) -> None:
        self.inner = HashingEmbedder(dim=64).fit(TEXTS)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self.inner.embed(list(texts))

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self.inner.embed_query(t) for t in texts])


class ExclusiveScorer:
    """A cross-encoder stub that fails if two threads are inside it at once."""

    scorer_id = "stub"

    def __init__(self) -> None:
        self.inside = 0
        self.overlaps = 0

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        self.inside += 1
        if self.inside > 1:
            self.overlaps += 1
        try:
            time.sleep(0.0005)  # lets another thread in if nothing keeps it out
            terms = set(question.split())
            return [float(len(terms & set(p.split()))) for p in passages]
        finally:
            self.inside -= 1


def _serving_entry(serving: bool) -> tuple[_Entry, NeuralEngine, ExclusiveScorer]:
    chunks = [Chunk(f"a{i}_c", f"a{i}", t, 0, len(t)) for i, t in enumerate(TEXTS)]
    bm25 = BM25Index()
    bm25.add(chunks)
    emb = FakeEmbedder()
    side = DenseSide(emb, emb.embed(TEXTS), cache_queries=not serving)  # type: ignore[arg-type]
    ce = ExclusiveScorer()
    engine = NeuralEngine(chunks, bm25, {"bge": side}, ce, cache_ce=not serving)
    retriever = NeuralChunkRetriever("hybrid+ce", serving=serving)
    retriever._engine = engine
    pipeline = RAGPipeline(retriever=retriever, rerank=False)
    entry = _Entry(pipeline, {f"a{i}": t for i, t in enumerate(TEXTS)}, {}, "hybrid+ce")
    return entry, engine, ce


def _install(entry: _Entry, tmp_path: Path) -> None:
    _entries[(tmp_path.resolve(), "hybrid+ce")] = entry


def _ids(tmp_path: Path, q: str) -> list[str]:
    return [a.id for a in search_fn(q, corpus_dir=tmp_path, retriever="hybrid+ce").articles]


def test_serving_caches_stay_empty_over_many_distinct_queries(tmp_path: Path) -> None:
    entry, engine, _ = _serving_entry(serving=True)
    _install(entry, tmp_path)
    for i in range(3000):
        search_fn(f"distinct question {i} about congestion", corpus_dir=tmp_path)
    assert len(engine.ce_cache) == 0
    assert all(len(s.queries) == 0 for s in engine.dense.values())


def test_experiment_mode_still_caches(tmp_path: Path) -> None:
    entry, engine, _ = _serving_entry(serving=False)
    _install(entry, tmp_path)
    for i in range(20):
        search_fn(f"distinct question {i} about congestion", corpus_dir=tmp_path)
    assert len(engine.ce_cache) > 0
    assert len(engine.dense["bge"].queries) == 20


def test_threads_match_sequential_runs(tmp_path: Path) -> None:
    entry, _, ce = _serving_entry(serving=True)
    _install(entry, tmp_path)
    queries = [f"balancing capacity question {i} congestion rule {i % 5}" for i in range(60)]
    sequential = {q: _ids(tmp_path, q) for q in queries}
    got: dict[int, dict[str, list[str]]] = {}

    def work(n: int) -> None:
        got[n] = {q: _ids(tmp_path, q) for q in queries}

    threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(got) == 8 and all(r == sequential for r in got.values())
    assert ce.overlaps == 0  # the lock kept every call out of the shared scorer one at a time


# --- prebuilt index -----------------------------------------------------------------------------


def test_embed_corpus_reads_a_prebuilt_matrix_and_never_writes_when_serving(
    tmp_path: Path,
) -> None:
    class Counting(FakeEmbedder):
        calls = 0

        def embed(self, texts: Sequence[str]) -> np.ndarray:
            Counting.calls += 1
            return super().embed(texts)

    emb: Any = Counting()
    index = tmp_path / "index"
    first = embed_corpus(emb, TEXTS, index)  # the build step writes
    assert Counting.calls == 1 and len(list(index.glob("*.npy"))) == 1
    again = embed_corpus(emb, TEXTS, index, write=False)  # serving loads it
    assert Counting.calls == 1 and np.array_equal(first, again)
    embed_corpus(emb, TEXTS[:-1], index, write=False)  # another corpus: in memory only
    assert Counting.calls == 2 and len(list(index.glob("*.npy"))) == 1


# --- answerable score ---------------------------------------------------------------------------


class FakeReader:
    def __init__(self) -> None:
        self.seen: list[list[str]] = []

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        self.seen.append(list(passages))
        return [float(len(p)) for p in passages]


def test_answerable_score_is_the_best_reader_score_of_the_top_five(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = sys.modules["ragkit.search"]
    reader = FakeReader()
    monkeypatch.setattr(module, "_reader", reader)
    result = search_fn(CONGESTION_Q, retriever="default", answerable=True)
    assert len(reader.seen[0]) == 5
    assert result.answerable_score == max(len(p) for p in reader.seen[0])
    assert result.answerable_threshold == ANSWERABLE_THRESHOLDS["default"]
    assert search_fn(CONGESTION_Q, retriever="bm25", answerable=True).answerable_threshold is None


def _eu_reader_scores() -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    ds = cli.BUNDLED_DATASETS["eu"]
    items = items_from_records(cli._read_jsonl(ds.qa_path), cli._read_jsonl(ds.unanswerable_path))
    pipeline = cli._make_pipeline("default", 0.0)
    passages = retrieve_passages(pipeline, cli._load_documents(ds.corpus_dir), items, top_k=5)
    cache = ROOT / "data" / "llm_cache" / "reader_scores.json"
    scores = reader_scores(offline_reader_scorers(cache), items, passages)
    return scores, np.array([i.answerable for i in items])


def test_threshold_reproduces_the_published_reader_rows() -> None:
    """Published ``reader`` row of the default retriever: 13/45 answered, 5/42 refused."""
    scores, labels = _eu_reader_scores()
    cv = cross_validated_accept(scores, labels, 5, 0)
    assert int((cv.accepted & ~labels).sum()) == 13
    assert int((~cv.accepted & labels).sum()) == 5
    t = ANSWERABLE_THRESHOLDS["default"]
    assert t == pytest.approx(best_threshold(scores, labels))
    assert t == float(np.median(cv.thresholds))
    # One threshold for all 87 questions is in-sample: one fewer unanswerable answered.
    assert int(((scores >= t) & ~labels).sum()) == 12
    assert int(((scores < t) & labels).sum()) == 5


# --- with the real models (slow: needs `[st]` and the pinned models in the local HF cache) -------

HYBRID_CE_CONGESTION_IDS = ["reg_2019_943_art16", "reg_2019_943_art14", "reg_2019_943_art2"]


@pytest.mark.slow
def test_default_retriever_with_real_models_reproduces_the_packing_study() -> None:
    pytest.importorskip("sentence_transformers")
    result = search_fn(CONGESTION_Q, corpus_dir=cli.EU_CORPUS_DIR)  # hybrid+ce, the default
    assert result.retriever == "hybrid+ce"
    assert [a.id for a in result.articles] == HYBRID_CE_CONGESTION_IDS  # pre-refactor context
    held_out = (cli.EU_QA_PATH, cli.EU_HARD_QA_PATH, cli.EU_NOKW_QA_PATH)
    questions = [q for p in held_out for q in questions_from_records(cli._read_jsonl(p))]
    hits = sum(
        bool(
            set(q.relevant)
            & {a.id for a in search_fn(q.text, corpus_dir=cli.EU_CORPUS_DIR).articles}
        )
        for q in questions
    )
    assert (hits, len(questions)) == (111, 130)  # docs/packing_protocol.md, hybrid+ce pack
