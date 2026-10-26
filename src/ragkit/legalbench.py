"""LegalBench-RAG-mini with the authors' character-overlap metric (``docs/legalbench_protocol.md``).

Reproduces the sampling of ``legalbenchrag/benchmark.py`` (194 tests per source, sorted by a
file-path-seeded random key) and the precision / recall of ``run_benchmark.py``, then runs the
model-free retrievers of ragkit over the pooled 72-document corpus. Only numpy and the standard
library are used.
"""

from __future__ import annotations

import json
import random
import shutil
import tempfile
import time
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ragkit.analysis import lucene_english
from ragkit.bm25 import BM25Index, tokenize, top_k_indices
from ragkit.chunking import Chunk, fixed, recursive
from ragkit.rerank import LexicalReranker

LEGALBENCH_URL = (
    "https://www.dropbox.com/scl/fo/r7xfa5i3hdsbxex1w6amw/AID389Olvtm-ZLTKAPrw6k4"
    "?rlkey=5n8zrbk4c08lbit3iiexofmwg&st=0hu354cq&dl=1"
)
DEFAULT_LEGALBENCH_ZIP = Path("data/raw/legalbenchrag.zip")
SOURCES = ("privacy_qa", "contractnli", "maud", "cuad")
MAX_TESTS_PER_SOURCE = 194
KS = (1, 2, 4, 8, 16, 32, 64)
RERANK_DEPTH = 300
# Paper Table 3: the mini corpus.
EXPECTED_DOCS = {"contractnli": 18, "maud": 18, "cuad": 29, "privacy_qa": 7}
EXPECTED_CHARS = 8_682_104

Span = tuple[str, int, int]


@dataclass(frozen=True)
class MiniTest:
    """One query with its ground-truth character spans."""

    source: str
    query: str
    truth: tuple[Span, ...]


@dataclass(frozen=True)
class MiniBenchmark:
    """The sampled tests and the pooled corpus (file path to text) of LegalBench-RAG-mini."""

    tests: list[MiniTest]
    corpus: dict[str, str]

    def docs_per_source(self) -> dict[str, int]:
        """Number of corpus files referenced by each source's sampled tests."""
        files: dict[str, set[str]] = {s: set() for s in SOURCES}
        for test in self.tests:
            files[test.source].update(path for path, _, _ in test.truth)
        return {s: len(p) for s, p in files.items()}

    def chars(self) -> int:
        """Total characters of the corpus."""
        return sum(len(t) for t in self.corpus.values())


def download_legalbench(path: Path) -> Path:
    """Download the authors' archive to ``path`` (atomic)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with urllib.request.urlopen(LEGALBENCH_URL, timeout=300) as response:
                shutil.copyfileobj(response, tmp)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
    tmp_path.replace(path)
    return path


def _universal_newlines(raw: bytes) -> str:
    """Decode as Python text mode would: ``\\r\\n`` and ``\\r`` become ``\\n``."""
    return raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


def sample_tests(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The official mini sample of one source: stable sort by a path-seeded key, keep 194."""
    if len(records) <= MAX_TESTS_PER_SOURCE:
        return list(records)

    def key(record: dict[str, Any]) -> float:
        return random.Random(str(record["snippets"][0]["file_path"])).random()

    return sorted(records, key=key)[:MAX_TESTS_PER_SOURCE]


def load_mini(path: Path = DEFAULT_LEGALBENCH_ZIP, *, download: bool = True) -> MiniBenchmark:
    """Read LegalBench-RAG-mini from the authors' zip (downloading it when missing)."""
    if not path.is_file():
        if not download:
            raise FileNotFoundError(
                f"{path} not found; run `ragkit legalbench` without --no-download"
            )
        download_legalbench(path)
    tests: list[MiniTest] = []
    used: set[str] = set()
    with zipfile.ZipFile(path) as zf:
        for source in SOURCES:
            records = json.loads(zf.read(f"benchmarks/{source}.json"))["tests"]
            for record in sample_tests(records):
                truth = tuple(
                    (str(s["file_path"]), int(s["span"][0]), int(s["span"][1]))
                    for s in record["snippets"]
                )
                used.update(file for file, _, _ in truth)
                tests.append(MiniTest(source=source, query=str(record["query"]), truth=truth))
        corpus = {p: _universal_newlines(zf.read(f"corpus/{p}")) for p in sorted(used)}
    return MiniBenchmark(tests=tests, corpus=corpus)


# ---------------------------------------------------------------------------------------------
# Metric (run_benchmark.py)
# ---------------------------------------------------------------------------------------------


def _overlap(a: Span, b: Span) -> int:
    if a[0] != b[0]:
        return 0
    return max(0, min(a[2], b[2]) - max(a[1], b[1]))


def char_precision(retrieved: Sequence[Span], truth: Sequence[Span]) -> float:
    """Retrieved characters that overlap ground truth over all retrieved characters."""
    total = sum(end - start for _, start, end in retrieved)
    if total == 0:
        return 0.0
    hit = sum(_overlap(r, g) for r in retrieved for g in truth)
    return hit / total


def char_recall(retrieved: Sequence[Span], truth: Sequence[Span]) -> float:
    """Ground-truth characters overlapped by retrieved chunks over all ground-truth characters."""
    total = sum(end - start for _, start, end in truth)
    if total == 0:
        return 0.0
    hit = sum(_overlap(r, g) for g in truth for r in retrieved)
    return hit / total


# ---------------------------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------------------------


def chunk_corpus(corpus: dict[str, str], chunker: str) -> list[Chunk]:
    """Chunk every file with ``naive500`` or ``recursive500`` (see the protocol)."""
    chunks: list[Chunk] = []
    for path, text in corpus.items():
        if chunker == "naive500":
            chunks.extend(fixed(text, path, size=500, overlap=0))
        elif chunker == "recursive500":
            chunks.extend(recursive(text, path, chunk_size=500, overlap=0))
        else:
            raise ValueError(f"unknown chunker {chunker!r}")
    return chunks


@dataclass(frozen=True)
class LegalSystem:
    """A named system: chunker, BM25 settings and an optional rerank depth."""

    name: str
    chunker: str
    k1: float
    b: float
    lucene: bool
    rerank: bool


SYSTEMS: tuple[LegalSystem, ...] = (
    LegalSystem("bm25-naive500", "naive500", 1.5, 0.75, False, False),
    LegalSystem("bm25-recursive500", "recursive500", 1.5, 0.75, False, False),
    LegalSystem("bm25-lucene-naive500", "naive500", 0.9, 0.4, True, False),
    LegalSystem("bm25-lucene-recursive500", "recursive500", 0.9, 0.4, True, False),
    LegalSystem("rerank-naive500", "naive500", 1.5, 0.75, False, True),
    LegalSystem("rerank-recursive500", "recursive500", 1.5, 0.75, False, True),
)


@dataclass
class LegalResult:
    """Per-query precision and recall at each k, by source."""

    name: str
    precision: dict[str, list[list[float]]] = field(default_factory=dict)
    recall: dict[str, list[list[float]]] = field(default_factory=dict)
    short_lists: int = 0
    n_chunks: int = 0
    seconds: float = 0.0

    def source_mean(self, metric: str, source: str) -> list[float]:
        """Mean over the source's queries at each k of ``KS``."""
        rows = np.asarray((self.precision if metric == "precision" else self.recall)[source])
        return [float(v) for v in rows.mean(axis=0)]

    def all_mean(self, metric: str) -> list[float]:
        """Equal-weight mean of the four source means at each k of ``KS``."""
        per_source = np.asarray([self.source_mean(metric, s) for s in SOURCES])
        return [float(v) for v in per_source.mean(axis=0)]

    def all_ci(
        self, metric: str, *, n_resamples: int = 2000, seed: int = 0
    ) -> list[tuple[float, float]]:
        """Percentile-bootstrap 95 % CI of ``all_mean`` (queries resampled within each source)."""
        rng = np.random.default_rng(seed)
        data = self.precision if metric == "precision" else self.recall
        means = np.zeros((n_resamples, len(KS)))
        for source in SOURCES:
            rows = np.asarray(data[source])
            idx = rng.integers(0, len(rows), size=(n_resamples, len(rows)))
            means += rows[idx].mean(axis=1) / len(SOURCES)
        lo, hi = np.quantile(means, [0.025, 0.975], axis=0)
        return [(float(a), float(b)) for a, b in zip(lo, hi, strict=True)]


def evaluate_system(
    system: LegalSystem, bench: MiniBenchmark, *, max_queries: int | None = None
) -> LegalResult:
    """Chunk, index and score every sampled query with ``system``."""
    started = time.perf_counter()
    chunks = chunk_corpus(bench.corpus, system.chunker)
    analyzer: Callable[[str], list[str]] = lucene_english if system.lucene else tokenize
    index = BM25Index(k1=system.k1, b=system.b, analyzer=analyzer)
    index.add(chunks)
    reranker = LexicalReranker() if system.rerank else None
    depth = RERANK_DEPTH if system.rerank else max(KS)
    result = LegalResult(system.name, n_chunks=len(chunks))
    per_source_count: dict[str, int] = {s: 0 for s in SOURCES}
    for test in bench.tests:
        if max_queries is not None and per_source_count[test.source] >= max_queries:
            continue
        per_source_count[test.source] += 1
        scores = index.score_all(test.query)
        hits = [(index.chunks[i], s) for i, s in top_k_indices(scores, depth)]
        if reranker is not None:
            hits = reranker.rerank(test.query, hits, top_k=max(KS))
        ranked = [(c.doc_id, c.start, c.end) for c, _ in hits[: max(KS)]]
        if len(ranked) < max(KS):
            result.short_lists += 1
        result.precision.setdefault(test.source, []).append(
            [char_precision(ranked[:k], test.truth) for k in KS]
        )
        result.recall.setdefault(test.source, []).append(
            [char_recall(ranked[:k], test.truth) for k in KS]
        )
    result.seconds = time.perf_counter() - started
    return result


def render_legalbench_report(bench: MiniBenchmark, results: Sequence[LegalResult]) -> str:
    """Markdown report: acceptance counts, then recall and precision at each k per system."""
    docs = bench.docs_per_source()
    n_queries = {s: sum(1 for t in bench.tests if t.source == s) for s in SOURCES}
    ok = docs == EXPECTED_DOCS and bench.chars() == EXPECTED_CHARS
    lines = [
        "### ragkit legalbench: LegalBench-RAG-mini (model-free)",
        "",
        f"{len(bench.tests)} queries ({', '.join(f'{s} {n_queries[s]}' for s in SOURCES)}); "
        f"{len(bench.corpus)} documents ({', '.join(f'{s} {docs[s]}' for s in SOURCES)}); "
        f"{bench.chars():,} characters. Paper Table 3: 72 documents, "
        f"{EXPECTED_CHARS:,} characters: "
        + ("match." if ok else "MISMATCH, this is not the paper's sample."),
        "",
        "Character-level metric of the official `run_benchmark.py`; ALL = equal-weight mean of the "
        "four sources; ALL carries a 95% bootstrap CI. Values in %.",
    ]
    header = "| System | " + " | ".join(f"@{k}" for k in KS) + " |"
    sep = "|---|" + "---|" * len(KS)
    for metric, label in (("recall", "Recall"), ("precision", "Precision")):
        lines += ["", f"**{label}@k, ALL**", "", header, sep]
        for res in results:
            cis = res.all_ci(metric)
            cells = [
                f"{100 * m:.2f} [{100 * lo:.1f}, {100 * hi:.1f}]"
                for m, (lo, hi) in zip(res.all_mean(metric), cis, strict=True)
            ]
            lines.append(f"| {res.name} | " + " | ".join(cells) + " |")
        for source in SOURCES:
            lines += ["", f"**{label}@k, {source}**", "", header, sep]
            for res in results:
                cells = [f"{100 * m:.2f}" for m in res.source_mean(metric, source)]
                lines.append(f"| {res.name} | " + " | ".join(cells) + " |")
    lines += ["", "| System | chunks | queries with < 64 hits | s |", "|---|---|---|---|"]
    lines += [f"| {r.name} | {r.n_chunks:,} | {r.short_lists} | {r.seconds:.0f} |" for r in results]
    return "\n".join(lines)
