"""Library search over a corpus of articles: ``search(query)`` returns citable, packed context.

ragkit only retrieves; the caller's model writes the answer. ``search`` is what ``ragkit answer``
builds its context with: chunk retrieval, then whole parent articles packed in rank order into a
token budget (``docs/packing_protocol.md``). Each article comes with the act title, article
number, source URL and corpus version that a citation needs.

Serving behaviour:

- The pipeline of each ``(corpus_dir, retriever)`` is built once and reused. Its query-vector and
  cross-encoder caches are off (they would grow with every new question) and one lock per
  pipeline serialises calls, so concurrent threads are safe and return what a sequential run
  returns.
- The document matrix is read from ``INDEX_DIR`` when a file for exactly this corpus exists
  (``ragkit build-index`` writes it) and is embedded in memory otherwise.
- If the neural models of the default retriever cannot be loaded (extra not installed, no local
  copy while offline), the pipeline falls back to BM25 alone; ``SearchResult.retriever`` says
  which one ran. A retriever named explicitly never falls back: it raises ``RuntimeError``, so an
  experiment cannot record BM25 numbers under another retriever's name.

Importing this module loads no LLM or model code. The public names are re-exported by
``ragkit/__init__.py``; this module is private so that ``ragkit.search`` stays the function.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ragkit.bm25 import BM25Index
from ragkit.eurlex import strip_metadata_comment
from ragkit.parent import (
    ARTICLE_BUDGET_TOKENS,
    PACK_CHUNKS,
    PACK_OVERFLOW,
    Passage,
    estimate_tokens,
    pack_articles,
)
from ragkit.pipeline import RAGPipeline

log = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent / "data"
DEFAULT_CORPUS_DIR = _DATA_DIR / "corpus_eu_consolidated"  # serving corpus: in-force text (#4)
INDEX_DIR = _DATA_DIR / "index"  # prebuilt document matrices, shipped in the wheel
DEFAULT_RETRIEVER = "hybrid+ce"  # measured best (docs/packing_protocol.md: 111/130 gold in context)
RETRIEVERS = ("default", "bm25", "dense", "hybrid", "hybrid+ce")
MAX_QUERY_CHARS = 2000  # the query comes from a model or a user
READER_CHUNKS = 5  # top chunks the answerable score reads (docs/generation_protocol.md)
# Reader (``qa_answerable``) thresholds: answer when score >= t. Each is the threshold
# ``abstention.best_threshold`` picks on all 87 EU questions (42 answerable + 45 unanswerable),
# which is also the median of the five fold thresholds of the 5-fold CV (seed 0) behind the
# published ``reader`` rows (docs/generation_protocol.md:56,299,458). Retrievers not listed have
# no measured threshold. Measured rates: README, "Library use".
ANSWERABLE_THRESHOLDS: dict[str, float] = {
    "default": -3.405692895874381,
    "hybrid+ce": -0.7443351745605469,
}

_HEADING = re.compile(r"^#+\s*(?:.*?,\s*)?Article\s+(?P<num>\S+?):\s*(?P<title>.+)$")
_ARTICLE_ID = re.compile(r"_art(?P<num>[^_]+)$")
_WARMUP_QUERY = "warm-up"


@dataclass(frozen=True)
class Article:
    """One whole article of the packed context, with the fields a citation needs."""

    id: str
    act: str
    celex: str | None
    article: str
    heading: str
    text: str
    rank: int  # 1 = the article of the best-ranked chunk
    url: str | None
    version: str | None


@dataclass(frozen=True)
class SearchResult:
    """Packed articles for one query; ``retriever`` is the one that actually ran."""

    query: str
    retriever: str
    articles: list[Article]
    estimated_tokens: int
    answerable_score: float | None = None
    answerable_threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Plain dict (``json.dumps`` accepts it)."""
        return asdict(self)


def load_documents(corpus_dir: Path) -> list[tuple[str, str]]:
    """Load all markdown and text documents of a directory as ``(doc_id, text)``."""
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    docs: list[tuple[str, str]] = []
    for file_path in sorted(corpus_dir.glob("*")):
        if file_path.suffix.lower() in {".md", ".txt"}:
            text = strip_metadata_comment(file_path.read_text(encoding="utf-8"))
            docs.append((file_path.stem, text))
    return docs


def _load_acts(corpus_dir: Path) -> dict[str, dict[str, Any]]:
    """``acts[]`` of ``SOURCES.json`` by ``short`` (the doc-id prefix); empty if there is none."""
    path = corpus_dir / "SOURCES.json"
    if not path.is_file():
        return {}
    return {a["short"]: a for a in json.loads(path.read_text(encoding="utf-8"))["acts"]}


def _version(act: dict[str, Any]) -> str:
    if act.get("consolidated_date"):
        return f"consolidated version of {act['consolidated_date']}"
    return f"original Official Journal text (retrieved {act['retrieved_on']})"


def _article(rank: int, passage: Passage, acts: dict[str, dict[str, Any]]) -> Article:
    first = passage.text.lstrip().split("\n", 1)[0]
    match = _HEADING.match(first)
    by_id = _ARTICLE_ID.search(passage.article_id)
    number = match["num"] if match else (by_id["num"] if by_id else "")
    heading = match["title"].strip() if match else first.lstrip("# ").strip()
    short = max((s for s in acts if passage.article_id.startswith(s + "_")), key=len, default=None)
    act = acts[short] if short is not None else None
    return Article(
        id=passage.article_id,
        act=act["title"] if act else passage.article_id,
        celex=act.get("celex") if act else None,
        article=number,
        heading=heading,
        text=passage.text,
        rank=rank,
        url=act.get("url") if act else None,
        version=_version(act) if act else None,
    )


def _make_pipeline(retriever: str, *, serving: bool) -> RAGPipeline:
    """The shipped pipeline, plain BM25, or one on a neural retriever (needs the ``[st]`` extra)."""
    if retriever == "default":
        return RAGPipeline()
    if retriever == "bm25":
        return RAGPipeline(retriever=BM25Index(), rerank=False)
    from ragkit.neural import NeuralChunkRetriever

    neural = NeuralChunkRetriever(retriever, cache_dir=INDEX_DIR, serving=serving)
    return RAGPipeline(retriever=neural, rerank=False)


@dataclass
class _Entry:
    """A ready pipeline, its articles and its lock (one per corpus and retriever)."""

    pipeline: RAGPipeline
    articles: dict[str, str]
    acts: dict[str, dict[str, Any]]
    retriever: str
    load_error: str | None = None  # why the requested retriever fell back to bm25
    lock: threading.Lock = field(default_factory=threading.Lock)


_entries: dict[tuple[Path, str], _Entry] = {}
_entries_lock = threading.Lock()


def _build_entry(corpus_dir: Path, retriever: str) -> _Entry:
    documents = load_documents(corpus_dir)
    if not documents:
        raise ValueError(f"no .md or .txt documents in {corpus_dir}")
    try:
        pipeline = _make_pipeline(retriever, serving=True)
        pipeline.index_documents(documents)
        if retriever not in ("default", "bm25"):
            pipeline.retrieve(_WARMUP_QUERY, 1)  # loads every model now, not on the first query
    except Exception as exc:
        log.warning("retriever %r unavailable, falling back to bm25", retriever, exc_info=True)
        pipeline = _make_pipeline("bm25", serving=True)
        pipeline.index_documents(documents)
        return _Entry(pipeline, dict(documents), _load_acts(corpus_dir), "bm25", repr(exc))
    return _Entry(pipeline, dict(documents), _load_acts(corpus_dir), retriever)


def _entry(corpus_dir: Path, retriever: str | None) -> _Entry:
    """The pipeline for ``retriever``; ``None`` means the default, which may fall back to bm25."""
    key = (corpus_dir.resolve(), retriever or DEFAULT_RETRIEVER)
    with _entries_lock:
        if key not in _entries:
            _entries[key] = _build_entry(*key)
        entry = _entries[key]
    if retriever is not None and entry.load_error is not None:
        raise RuntimeError(f"retriever {retriever!r} is unavailable: {entry.load_error}")
    return entry


def _validated(query: str, retriever: str | None) -> str:
    if retriever is not None and retriever not in RETRIEVERS:
        raise ValueError(f"retriever must be one of {RETRIEVERS}, got {retriever!r}")
    query = query.strip()
    if not query:
        raise ValueError("query is empty")
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"query has {len(query)} characters, the limit is {MAX_QUERY_CHARS}")
    return query


_reader: Any = None
_reader_lock = threading.Lock()


def _reader_score(query: str, chunks: list[Passage]) -> float:
    """``qa_answerable``: best SQuAD2 span score minus null score over the chunk texts."""
    global _reader
    with _reader_lock:
        if _reader is None:
            from ragkit.modelsignals import SquadNullScorer

            _reader = SquadNullScorer()
        return max(_reader.score(query, [c.text for c in chunks]), default=0.0)


def retrieve_chunks(
    query: str,
    k: int,
    *,
    corpus_dir: Path | str | None = None,
    retriever: str | None = None,
) -> list[Passage]:
    """The top-``k`` chunks of the pipeline ``search`` uses (``ragkit answer`` gates on them)."""
    query = _validated(query, retriever)
    entry = _entry(Path(corpus_dir or DEFAULT_CORPUS_DIR), retriever)
    with entry.lock:
        return [Passage(c.doc_id, c.text) for c in entry.pipeline.retrieve(query, k)]


def search(
    query: str,
    *,
    corpus_dir: Path | str | None = None,
    retriever: str | None = None,
    budget_tokens: int = ARTICLE_BUDGET_TOKENS,
    pack_chunks: int = PACK_CHUNKS,
    answerable: bool = False,
    overflow: str = PACK_OVERFLOW,
) -> SearchResult:
    """Distinct parent articles of the top ``pack_chunks`` chunks, packed into ``budget_tokens``.

    ``retriever=None`` uses ``DEFAULT_RETRIEVER`` and falls back to bm25 if its models cannot
    load; a retriever named explicitly raises ``RuntimeError`` instead.
    ``answerable=True`` also fills ``answerable_score`` (the SQuAD2 reader signal on the top
    ``READER_CHUNKS`` chunks, needs the ``[st]`` extra) and ``answerable_threshold``
    (``ANSWERABLE_THRESHOLDS``; ``None`` where none was measured): refuse when the score is below
    it. Raises ``ValueError`` for an empty or over-long query, an unknown retriever or a
    non-positive ``pack_chunks`` / ``budget_tokens``.
    """
    query = _validated(query, retriever)
    if pack_chunks < 1:
        raise ValueError("pack_chunks must be positive")
    entry = _entry(Path(corpus_dir or DEFAULT_CORPUS_DIR), retriever)
    with entry.lock:
        pool = [Passage(c.doc_id, c.text) for c in entry.pipeline.retrieve(query, pack_chunks)]
        packed = pack_articles(pool, entry.articles, budget_tokens=budget_tokens, overflow=overflow)
        top = entry.pipeline.retrieve(query, READER_CHUNKS) if answerable else []
    score = _reader_score(query, [Passage(c.doc_id, c.text) for c in top]) if answerable else None
    return SearchResult(
        query=query,
        retriever=entry.retriever,
        articles=[_article(i, p, entry.acts) for i, p in enumerate(packed, 1)],
        estimated_tokens=sum(estimate_tokens(p.text) for p in packed),
        answerable_score=score,
        answerable_threshold=ANSWERABLE_THRESHOLDS.get(entry.retriever) if answerable else None,
    )


def build_index(corpus_dir: Path | str | None = None, index_dir: Path | str | None = None) -> Path:
    """Write the document matrix of ``corpus_dir`` to ``index_dir`` (default: ``INDEX_DIR``).

    The file name holds the model and a digest of every chunk text (``corpus_cache_path``), so
    ``search`` uses it only for exactly this corpus and chunking. Chunks are not stored: they are
    recomputed from the corpus in milliseconds and the digest covers them. Returns the file.
    """
    from ragkit.neural import NeuralChunkRetriever, bge_small, corpus_cache_path

    documents = load_documents(Path(corpus_dir or DEFAULT_CORPUS_DIR))
    out = Path(index_dir or INDEX_DIR)
    retriever = NeuralChunkRetriever("dense", cache_dir=out)
    pipeline = RAGPipeline(retriever=retriever, rerank=False)
    pipeline.index_documents(documents)
    return corpus_cache_path(bge_small(), [c.text for c in pipeline.indexed_chunks], out)
