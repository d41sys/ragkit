"""BEIR test-set evaluation of the model-free retrievers (protocol: ``docs/beir_protocol.md``).

Loads a BEIR v1.0.0 dataset from its original zip, runs BM25, a Lucene-matched BM25, BM25F and
the lexical reranker, and scores nDCG@10 and Recall@100 with trec_eval conventions so that the
numbers can be set beside the Pyserini BEIR 2CR table. Only numpy and the standard library are
used: no model is built or loaded.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import time
import urllib.request
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from ragkit.analysis import (
    lucene_english,
    lucene_length,
    lucene_stop,
    lucene_tokens,
)
from ragkit.bm25 import BM25Index, tokenize, top_k_indices
from ragkit.bm25f import BM25FIndex
from ragkit.chunking import Chunk
from ragkit.eval import bootstrap_ci
from ragkit.rerank import LexicalReranker

BEIR_DATASETS = ("scifact", "nfcorpus", "fiqa", "arguana", "trec-covid", "quora")
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
DEFAULT_BEIR_DIR = Path("data/raw/beir")
NDCG_CUT = 10
RECALL_CUT = 100
RERANK_DEPTH = 100

# Anserini / Pyserini BM25 parameters (the "default Lucene parameters" of the BEIR baselines).
LUCENE_K1 = 0.9
LUCENE_B = 0.4

# Pyserini BEIR 2CR table, https://castorini.github.io/pyserini/2cr/beir.html, test split,
# (nDCG@10, Recall@100), read 2026-10-26. Columns: BM25 Flat, BM25 MF, SPLADE++ ED,
# Contriever (msmarco), BGE-base-en-v1.5.
PYSERINI_COLUMNS = (
    "Pyserini BM25 flat",
    "Pyserini BM25 multifield",
    "Pyserini SPLADE++ ED",
    "Pyserini Contriever-msmarco",
    "Pyserini BGE-base-en-v1.5",
)
PYSERINI_2CR: dict[str, tuple[tuple[float, float], ...]] = {
    "scifact": ((0.679, 0.925), (0.665, 0.908), (0.704, 0.935), (0.677, 0.947), (0.741, 0.967)),
    "nfcorpus": ((0.322, 0.246), (0.325, 0.250), (0.347, 0.285), (0.328, 0.301), (0.373, 0.337)),
    "fiqa": ((0.236, 0.539), (0.236, 0.539), (0.347, 0.631), (0.329, 0.656), (0.406, 0.742)),
    "arguana": ((0.397, 0.932), (0.414, 0.943), (0.526, 0.983), (0.446, 0.977), (0.636, 0.992)),
    "trec-covid": ((0.595, 0.109), (0.656, 0.114), (0.727, 0.128), (0.596, 0.091), (0.781, 0.141)),
    "quora": ((0.789, 0.973), (0.789, 0.973), (0.834, 0.986), (0.865, 0.994), (0.889, 0.997)),
}


@dataclass(frozen=True)
class BeirData:
    """One BEIR dataset split: corpus fields, scored queries and positive judgements."""

    name: str
    doc_ids: list[str]
    titles: list[str]
    texts: list[str]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]

    @property
    def has_titles(self) -> bool:
        """True when a meaningful share of documents has a title (not FiQA or Quora)."""
        return sum(1 for t in self.titles if t.strip()) * 10 > len(self.titles)


def zip_path(name: str, root: Path) -> Path:
    """Where the zip of dataset ``name`` lives under ``root``."""
    return root / f"{name}.zip"


def download_dataset(name: str, root: Path) -> Path:
    """Download the BEIR zip of ``name`` to ``root`` (atomic: a partial file is never left)."""
    if name not in BEIR_DATASETS:
        raise ValueError(f"unknown BEIR dataset {name!r}; choose from {', '.join(BEIR_DATASETS)}")
    root.mkdir(parents=True, exist_ok=True)
    target = zip_path(name, root)
    with tempfile.NamedTemporaryFile(dir=root, suffix=".part", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with urllib.request.urlopen(BEIR_URL.format(name=name), timeout=120) as response:
                shutil.copyfileobj(response, tmp)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
    tmp_path.replace(target)
    return target


def load_beir(
    name: str,
    root: Path = DEFAULT_BEIR_DIR,
    *,
    split: str = "test",
    download: bool = True,
) -> BeirData:
    """Read a BEIR dataset from ``root/<name>.zip`` (downloading it when missing).

    Queries without any relevant document (qrels score > 0) are dropped; qrels with score 0
    are not relevant. See ``docs/beir_protocol.md``.
    """
    path = zip_path(name, root)
    if not path.is_file():
        if not download:
            raise FileNotFoundError(f"{path} not found; run `ragkit beir` without --no-download")
        path = download_dataset(name, root)
    with zipfile.ZipFile(path) as zf:
        doc_ids: list[str] = []
        titles: list[str] = []
        texts: list[str] = []
        with zf.open(f"{name}/corpus.jsonl") as handle:
            for line in handle:
                rec = json.loads(line)
                doc_ids.append(str(rec["_id"]))
                titles.append(str(rec.get("title", "")))
                texts.append(str(rec.get("text", "")))
        all_queries: dict[str, str] = {}
        with zf.open(f"{name}/queries.jsonl") as handle:
            for line in handle:
                rec = json.loads(line)
                all_queries[str(rec["_id"])] = str(rec["text"])
        qrels: dict[str, dict[str, int]] = {}
        with zf.open(f"{name}/qrels/{split}.tsv") as handle:
            for i, raw in enumerate(handle):
                line = raw.decode("utf-8").strip()  # some BEIR qrels files use CRLF
                if not line:
                    continue
                parts = line.split("\t")
                if i == 0 and parts[-1] == "score":
                    continue
                qid, did, score = parts[0], parts[1], int(parts[2])
                if score > 0:
                    qrels.setdefault(qid, {})[did] = score
    queries = {qid: all_queries[qid] for qid in qrels if qid in all_queries}
    return BeirData(
        name=name,
        doc_ids=doc_ids,
        titles=titles,
        texts=texts,
        queries=queries,
        qrels={qid: qrels[qid] for qid in queries},
    )


# ---------------------------------------------------------------------------------------------
# Metrics (trec_eval: ndcg_cut.10 with linear gain, recall.100)
# ---------------------------------------------------------------------------------------------


def ndcg_cut(ranked: Sequence[str], rels: Mapping[str, int], k: int = NDCG_CUT) -> float:
    """trec_eval ``ndcg_cut.k``: linear gain, ``1 / log2(rank + 1)``, ideal from all judged."""
    ideal = sorted((g for g in rels.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    if idcg == 0.0:
        return 0.0
    dcg = sum(rels.get(doc, 0) / math.log2(i + 2) for i, doc in enumerate(ranked[:k]))
    return dcg / idcg


def recall_at(ranked: Sequence[str], rels: Mapping[str, int], k: int = RECALL_CUT) -> float:
    """trec_eval ``recall.k``: relevant documents in the top k over all relevant documents."""
    relevant = sum(1 for g in rels.values() if g > 0)
    if relevant == 0:
        return 0.0
    return sum(1 for doc in ranked[:k] if rels.get(doc, 0) > 0) / relevant


# ---------------------------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------------------------


class BeirRetriever(Protocol):
    """Ranks documents for a query; ``exclude_id`` removes a self hit (Pyserini --remove-query)."""

    def search(self, query: str, k: int, exclude_id: str | None = None) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(doc_id, score)`` pairs, best first."""
        ...


def _flat_chunks(data: BeirData) -> list[Chunk]:
    out: list[Chunk] = []
    for doc_id, title, text in zip(data.doc_ids, data.titles, data.texts, strict=True):
        body = f"{title} {text}".strip()
        out.append(Chunk(id=doc_id, doc_id=doc_id, text=body, start=0, end=len(body)))
    return out


def _single_field(doc_id: str, text: str) -> Chunk:
    return Chunk(id=doc_id, doc_id=doc_id, text=text, start=0, end=len(text))


def _field_chunks(data: BeirData) -> list[Chunk]:
    return [
        Chunk(id=doc_id, doc_id=doc_id, text=text, start=0, end=len(text), title=title)
        for doc_id, title, text in zip(data.doc_ids, data.titles, data.texts, strict=True)
    ]


def _ranked(
    scores: np.ndarray, doc_ids: Sequence[str], k: int, exclude_id: str | None
) -> list[tuple[str, float]]:
    top = top_k_indices(scores, k + 1 if exclude_id is not None else k)
    hits = [(doc_ids[i], s) for i, s in top if doc_ids[i] != exclude_id]
    return hits[:k]


class FlatBM25:
    """``BM25Index`` over ``title + " " + text`` (the Anserini "flat" document)."""

    def __init__(
        self,
        data: BeirData,
        *,
        k1: float,
        b: float,
        analyzer: Callable[[str], list[str]] = tokenize,
        length_fn: Callable[[int], int] | None = None,
    ) -> None:
        self.doc_ids = data.doc_ids
        self.index = BM25Index(k1=k1, b=b, analyzer=analyzer, length_fn=length_fn)
        self.index.add(_flat_chunks(data))

    def search(self, query: str, k: int, exclude_id: str | None = None) -> list[tuple[str, float]]:
        """Top ``k`` documents by BM25."""
        return _ranked(self.index.score_all(query), self.doc_ids, k, exclude_id)


class MultiFieldBM25:
    """Anserini "multifield": BM25 on the text plus BM25 on the title, equal weights.

    Each field has its own idf and average length, computed over the documents that have the
    field (Lucene's per-field ``docCount``).
    """

    def __init__(
        self,
        data: BeirData,
        *,
        k1: float,
        b: float,
        analyzer: Callable[[str], list[str]],
    ) -> None:
        self.doc_ids = data.doc_ids
        n = len(data.doc_ids)
        text_rows = [i for i in range(n) if data.texts[i].strip()]
        title_rows = [i for i in range(n) if data.titles[i].strip()]
        self.text_rows = np.asarray(text_rows, dtype=np.int64)
        self.title_rows = np.asarray(title_rows, dtype=np.int64)
        self.text_index = BM25Index(k1=k1, b=b, analyzer=analyzer)
        self.text_index.add([_single_field(data.doc_ids[i], data.texts[i]) for i in text_rows])
        self.title_index = BM25Index(k1=k1, b=b, analyzer=analyzer)
        self.title_index.add([_single_field(data.doc_ids[i], data.titles[i]) for i in title_rows])

    def search(self, query: str, k: int, exclude_id: str | None = None) -> list[tuple[str, float]]:
        """Top ``k`` documents by summed field scores."""
        scores = np.zeros(len(self.doc_ids), dtype=np.float64)
        scores[self.text_rows] += self.text_index.score_all(query)
        scores[self.title_rows] += self.title_index.score_all(query)
        return _ranked(scores, self.doc_ids, k, exclude_id)


class FieldBM25F:
    """``BM25FIndex`` with the document title as the heading field."""

    def __init__(
        self,
        data: BeirData,
        *,
        k1: float,
        b: float,
        analyzer: Callable[[str], list[str]] = tokenize,
        title_weight: float = 2.0,
        body_weight: float = 1.0,
    ) -> None:
        self.doc_ids = data.doc_ids
        self.index = BM25FIndex(
            k1=k1,
            b=b,
            title_weight=title_weight,
            body_weight=body_weight,
            analyzer=analyzer,
        )
        self.index.add(_field_chunks(data))

    def search(self, query: str, k: int, exclude_id: str | None = None) -> list[tuple[str, float]]:
        """Top ``k`` documents by BM25F."""
        return _ranked(self.index.score_all(query), self.doc_ids, k, exclude_id)


class Reranked:
    """A first-stage retriever whose top ``depth`` documents are reordered by a reranker."""

    def __init__(
        self,
        first: BeirRetriever,
        chunks: Mapping[str, Chunk],
        reranker: LexicalReranker,
        depth: int = RERANK_DEPTH,
    ) -> None:
        self.first = first
        self.chunks = chunks
        self.reranker = reranker
        self.depth = depth

    def search(self, query: str, k: int, exclude_id: str | None = None) -> list[tuple[str, float]]:
        """Rerank the first stage's top ``depth`` and return the first ``k``."""
        candidates = [
            (self.chunks[doc_id], score)
            for doc_id, score in self.first.search(query, self.depth, exclude_id)
        ]
        reranked = self.reranker.rerank(query, candidates, top_k=self.depth)
        return [(chunk.id, score) for chunk, score in reranked[:k]]


# ---------------------------------------------------------------------------------------------
# Systems of the protocol
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SystemSpec:
    """A named system: a factory over the dataset and whether it needs document titles."""

    name: str
    description: str
    build: Callable[[BeirData], BeirRetriever]
    needs_titles: bool = False


def _rerank_system(first: BeirRetriever, data: BeirData, reranker: LexicalReranker) -> Reranked:
    return Reranked(first, {c.id: c for c in _flat_chunks(data)}, reranker)


def _bm25_ragkit(data: BeirData) -> BeirRetriever:
    return FlatBM25(data, k1=1.5, b=0.75)


def _bm25_lucene(data: BeirData) -> BeirRetriever:
    return FlatBM25(data, k1=LUCENE_K1, b=LUCENE_B, analyzer=lucene_english)


SYSTEMS: tuple[SystemSpec, ...] = (
    SystemSpec("bm25-ragkit", "BM25 k1=1.5 b=0.75, [a-z0-9]+ tokens", _bm25_ragkit),
    SystemSpec(
        "bm25-lucene", "BM25 k1=0.9 b=0.4, Lucene English analyzer (Anserini flat)", _bm25_lucene
    ),
    SystemSpec(
        "bm25-mf-lucene",
        "BM25 text + BM25 title, k1=0.9 b=0.4, Lucene analyzer (Anserini multifield)",
        lambda d: MultiFieldBM25(d, k1=LUCENE_K1, b=LUCENE_B, analyzer=lucene_english),
        needs_titles=True,
    ),
    SystemSpec(
        "bm25f-ragkit",
        "BM25F title x2, k1=1.5 b=0.75, [a-z0-9]+ tokens",
        lambda d: FieldBM25F(d, k1=1.5, b=0.75),
        needs_titles=True,
    ),
    SystemSpec(
        "bm25f-lucene",
        "BM25F title x2, k1=0.9 b=0.4, Lucene English analyzer",
        lambda d: FieldBM25F(d, k1=LUCENE_K1, b=LUCENE_B, analyzer=lucene_english),
        needs_titles=True,
    ),
    SystemSpec(
        "rerank-ragkit",
        "lexical reranker (default) over the top 100 of bm25-ragkit",
        lambda d: _rerank_system(_bm25_ragkit(d), d, LexicalReranker()),
    ),
    SystemSpec(
        "rerank-lucene",
        "lexical reranker (Lucene analyzer, k1=0.9 b=0.4) over the top 100 of bm25-lucene",
        lambda d: _rerank_system(
            _bm25_lucene(d),
            d,
            LexicalReranker(k1=LUCENE_K1, b=LUCENE_B, analyzer=lucene_english),
        ),
    ),
)

LADDER: tuple[SystemSpec, ...] = (
    SystemSpec("L0 default", "tokenize, k1=1.5 b=0.75", _bm25_ragkit),
    SystemSpec(
        "L1 +lucene tokens",
        "Lucene tokens + possessive filter, no stop words, no stemming, k1=1.5 b=0.75",
        lambda d: FlatBM25(d, k1=1.5, b=0.75, analyzer=lucene_tokens),
    ),
    SystemSpec(
        "L2 +stop words",
        "L1 + Lucene English stop words",
        lambda d: FlatBM25(d, k1=1.5, b=0.75, analyzer=lucene_stop),
    ),
    SystemSpec(
        "L3 +Porter",
        "L2 + Porter stemmer (= Lucene English analyzer), k1=1.5 b=0.75",
        lambda d: FlatBM25(d, k1=1.5, b=0.75, analyzer=lucene_english),
    ),
    SystemSpec("L4 k1=0.9 b=0.4", "L3 with Anserini's k1 and b (= bm25-lucene)", _bm25_lucene),
    SystemSpec(
        "L5 +Lucene length norm",
        "L4 with Lucene's one-byte document-length norm",
        lambda d: FlatBM25(
            d, k1=LUCENE_K1, b=LUCENE_B, analyzer=lucene_english, length_fn=lucene_length
        ),
    ),
)


@dataclass
class SystemResult:
    """Per-query scores of one system on one dataset."""

    name: str
    ndcg: list[float] = field(default_factory=list)
    recall: list[float] = field(default_factory=list)
    seconds: float = 0.0

    def summary(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """``(mean, lo, hi)`` of nDCG@10 and Recall@100."""
        return bootstrap_ci(self.ndcg), bootstrap_ci(self.recall)


def evaluate_system(
    spec: SystemSpec, data: BeirData, *, max_queries: int | None = None
) -> SystemResult:
    """Build ``spec`` on ``data`` and score every query (``max_queries`` limits for smoke runs)."""
    started = time.perf_counter()
    retriever = spec.build(data)
    result = SystemResult(spec.name)
    items = list(data.queries.items())[:max_queries]
    for qid, text in items:
        ranked = [doc for doc, _ in retriever.search(text, RECALL_CUT, exclude_id=qid)]
        rels = data.qrels[qid]
        result.ndcg.append(ndcg_cut(ranked, rels))
        result.recall.append(recall_at(ranked, rels))
    result.seconds = time.perf_counter() - started
    return result


def pyserini_rows(name: str) -> list[tuple[str, float, float]]:
    """The published Pyserini rows ``(system, nDCG@10, Recall@100)`` for ``name``."""
    return [
        (label, ndcg, recall)
        for label, (ndcg, recall) in zip(PYSERINI_COLUMNS, PYSERINI_2CR.get(name, ()), strict=False)
    ]


def render_beir_report(
    data: BeirData,
    results: Sequence[SystemResult],
    *,
    title: str,
) -> str:
    """Markdown table of our systems with CIs followed by the published Pyserini rows."""
    lines = [
        f"### {title}",
        "",
        f"{len(results[0].ndcg) if results else 0} test queries, {len(data.doc_ids)} documents; "
        "mean [bootstrap 95% CI over queries]",
        "",
        "| System | nDCG@10 | Recall@100 | s |",
        "|---|---|---|---|",
    ]
    for res in results:
        (n, nlo, nhi), (r, rlo, rhi) = res.summary()
        lines.append(
            f"| {res.name} | {n:.3f} [{nlo:.3f}, {nhi:.3f}] | {r:.3f} [{rlo:.3f}, {rhi:.3f}] "
            f"| {res.seconds:.0f} |"
        )
    published = pyserini_rows(data.name)
    if published:
        lines += [
            "",
            "Published, same dataset/split/metrics (Pyserini BEIR 2CR table, "
            "https://castorini.github.io/pyserini/2cr/beir.html):",
            "",
            "| System | nDCG@10 | Recall@100 |",
            "|---|---|---|",
        ]
        lines += [f"| {label} | {n:.3f} | {r:.3f} |" for label, n, r in published]
    return "\n".join(lines)
