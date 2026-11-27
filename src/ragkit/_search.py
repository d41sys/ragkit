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
import os
import re
import threading
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ragkit.bm25 import BM25Index
from ragkit.eurlex import strip_metadata_comment
from ragkit.parent import (
    ARTICLE_BUDGET_TOKENS,
    PACK_CHUNKS,
    PACK_OVERFLOW,
    PAPER_BUDGET_TOKENS,
    PAPER_PACK_CHUNKS,
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
RETRIEVERS = ("default", "bm25", "dense", "hybrid", "hybrid+ce", "hybrid+bge-ce")
MAX_QUERY_CHARS = 2000  # the query comes from a model or a user
READER_CHUNKS = 5  # top chunks the answerable score reads (docs/generation_protocol.md)
DOCUMENT_CHUNKS = 100  # ranking depth a search inside chosen documents reads (followup_protocol)
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
    pages: str | None = None  # "3" or "3-4" for a document with form-feed page breaks (PDFs)
    citation: str | None = None  # SOURCES.json documents[].citation: "Nie et al., ICLR 2023"


@dataclass(frozen=True)
class Match:
    """A document with a chunk in the pool whose text did not fit the budget: search inside it."""

    id: str
    act: str
    citation: str | None
    pages: str | None  # pages of its best chunk


@dataclass(frozen=True)
class SearchResult:
    """Packed articles for one query; ``retriever`` is the one that actually ran."""

    query: str
    retriever: str
    articles: list[Article]
    estimated_tokens: int
    answerable_score: float | None = None
    answerable_threshold: float | None = None
    also_matched: list[Match] = field(default_factory=list)  # in the rank order of best chunks

    def to_dict(self) -> dict[str, Any]:
        """Plain dict (``json.dumps`` accepts it)."""
        return asdict(self)


def _pdf_text(path: Path) -> str:
    """Text of a PDF, pages joined by a form feed (needs the ``[pdf]`` extra)."""
    from pypdf import PdfReader

    logging.getLogger("pypdf").setLevel(logging.ERROR)  # font-encoding warnings flood the log
    return "\f".join(page.extract_text() or "" for page in PdfReader(path).pages)


def _cached_pdf_text(path: Path, cache_dir: Path | None) -> str:
    """``_pdf_text``, kept under ``cache_dir`` by file name, size and modification time."""
    if cache_dir is None:
        return _pdf_text(path)
    stat = path.stat()
    cached = cache_dir / f"{path.stem}-{stat.st_size}-{stat.st_mtime_ns}.txt"
    if cached.is_file():
        return cached.read_bytes().decode("utf-8", "surrogatepass")
    text = _pdf_text(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # bytes: no newline translation; written aside and renamed, so no reader sees half a file
    tmp = cached.with_name(f"{cached.name}.{os.getpid()}.tmp")
    tmp.write_bytes(text.encode("utf-8", "surrogatepass"))
    os.replace(tmp, cached)
    return text


def load_documents(corpus_dir: Path, text_cache: Path | None = None) -> list[tuple[str, str]]:
    """Load all markdown, text and PDF documents of a directory as ``(doc_id, text)``.

    ``text_cache`` keeps the extracted text of each PDF (extraction takes about a second each).
    """
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    docs: list[tuple[str, str]] = []
    for file_path in sorted(corpus_dir.glob("*")):
        suffix = file_path.suffix.lower()
        if suffix in {".md", ".txt"}:
            text = strip_metadata_comment(file_path.read_text(encoding="utf-8"))
            docs.append((file_path.stem, text))
        elif suffix == ".pdf":
            try:
                docs.append((file_path.stem, _cached_pdf_text(file_path, text_cache)))
            except ImportError:  # the [pdf] extra is missing: no PDF can be read
                raise
            except Exception:  # an uploaded file may be corrupt or encrypted: keep the rest
                log.warning("skipping unreadable PDF %s", file_path.name, exc_info=True)
    return docs


def _load_acts(corpus_dir: Path) -> dict[str, dict[str, Any]]:
    """``SOURCES.json``: ``acts[]`` by ``short`` (doc-id prefix), ``documents[]`` by ``id``."""
    path = corpus_dir / "SOURCES.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {a["short"]: a for a in data.get("acts", [])} | {
        d["id"]: d for d in data.get("documents", [])
    }


def _version(act: dict[str, Any]) -> str | None:
    if act.get("consolidated_date"):
        return f"consolidated version of {act['consolidated_date']}"
    if act.get("retrieved_on"):
        return f"original Official Journal text (retrieved {act['retrieved_on']})"
    return None


def _pages(passage: Passage, document: str) -> str | None:
    """Pages a passage spans in a document with form-feed page breaks; ``None`` without them."""
    if "\f" not in document:
        return None
    start = max(document.find(passage.text), 0)
    first = document.count("\f", 0, start) + 1
    last = document.count("\f", 0, start + len(passage.text)) + 1
    return str(first) if first == last else f"{first}-{last}"


def _article(
    rank: int, passage: Passage, acts: dict[str, dict[str, Any]], document: str
) -> Article:
    first = passage.text.lstrip().split("\n", 1)[0]
    match = _HEADING.match(first)
    by_id = _ARTICLE_ID.search(passage.article_id)
    number = match["num"] if match else (by_id["num"] if by_id else "")
    heading = match["title"].strip() if match else first.lstrip("# ").strip()
    short = max((s for s in acts if passage.article_id.startswith(s + "_")), key=len, default=None)
    act = acts.get(passage.article_id) or (acts[short] if short is not None else None)
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
        pages=_pages(passage, document.strip()),
        citation=act.get("citation") if act else None,
    )


def _make_pipeline(retriever: str, *, serving: bool, index_dir: Path | None = None) -> RAGPipeline:
    """The shipped pipeline, plain BM25, or one on a neural retriever (needs the ``[st]`` extra)."""
    if retriever == "default":
        return RAGPipeline()
    if retriever == "bm25":
        return RAGPipeline(retriever=BM25Index(), rerank=False)
    from ragkit.neural import NeuralChunkRetriever

    neural = NeuralChunkRetriever(
        retriever, cache_dir=INDEX_DIR, serving=serving, doc_cache_dir=index_dir
    )
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
    budget_tokens: int = ARTICLE_BUDGET_TOKENS  # the default budget of ``search`` on this corpus
    pack_chunks: int = PACK_CHUNKS  # the default chunk pool of ``search`` on this corpus
    stamp: tuple[tuple[str, int, int], ...] = ()  # the corpus files this entry was built from


_entries: dict[tuple[Path, str], _Entry] = {}
_entries_lock = threading.Lock()  # one build at a time; a built entry is read without it


def _build_entry(corpus_dir: Path, retriever: str, index_dir: Path | None = None) -> _Entry:
    documents = load_documents(corpus_dir, index_dir)
    if not documents:
        raise ValueError(f"no .md, .txt or .pdf documents in {corpus_dir}")
    # paged text (a PDF, or form-feed page breaks) means papers: one fills the 12 000 budget
    # alone, 36 000 fits one to three (docs/open_papers_protocol.md), and the papers a question
    # needs are more often among the top 30 chunks than the top 10 (docs/paper_pool_protocol.md)
    paged = any("\f" in text for _, text in documents)
    budget = PAPER_BUDGET_TOKENS if paged else ARTICLE_BUDGET_TOKENS
    pool = PAPER_PACK_CHUNKS if paged else PACK_CHUNKS
    acts = _load_acts(corpus_dir)
    error: str | None = None  # why the requested retriever fell back to bm25
    try:
        pipeline = _make_pipeline(retriever, serving=True, index_dir=index_dir)
        pipeline.index_documents(documents)
        if retriever not in ("default", "bm25"):
            pipeline.retrieve(_WARMUP_QUERY, 1)  # loads every model now, not on the first query
    except Exception as exc:
        log.warning("retriever %r unavailable, falling back to bm25", retriever, exc_info=True)
        retriever, error = "bm25", repr(exc)
        pipeline = _make_pipeline("bm25", serving=True)
        pipeline.index_documents(documents)
    return _Entry(
        pipeline, dict(documents), acts, retriever, error, budget_tokens=budget, pack_chunks=pool
    )


def _stamp(corpus_dir: Path) -> tuple[tuple[str, int, int], ...]:
    """Name, size and modification time of every corpus file."""
    files = sorted(p for p in corpus_dir.glob("*") if p.is_file())
    return tuple((p.name, s.st_size, s.st_mtime_ns) for p in files for s in (p.stat(),))


def _entry(corpus_dir: Path, retriever: str | None, index_dir: Path | None = None) -> _Entry:
    """The pipeline for ``retriever``; ``None`` means the default, which may fall back to bm25.

    It is rebuilt when a file of the corpus was added, removed or changed (uploaded papers);
    with ``index_dir`` the rebuild reads and embeds only the new files.
    """
    key = (corpus_dir.resolve(), retriever or DEFAULT_RETRIEVER)
    stamp = _stamp(key[0])
    entry = _entries.get(key)
    if entry is None or entry.stamp != stamp:
        # indexing a corpus (minutes for a folder of PDFs) must not block searches of the others
        with _entries_lock:
            entry = _entries.get(key)  # another thread may have built it meanwhile
            if entry is None or entry.stamp != stamp:
                entry = _build_entry(*key, index_dir)
                entry.stamp = stamp
                _entries[key] = entry
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
    budget_tokens: int | None = None,
    pack_chunks: int | None = None,
    answerable: bool = False,
    overflow: str = PACK_OVERFLOW,
    window_tokens: int | None = None,
    index_dir: Path | str | None = None,
    documents: Sequence[str] | None = None,
) -> SearchResult:
    """Distinct parent articles of the top ``pack_chunks`` chunks, packed into ``budget_tokens``.

    ``budget_tokens=None`` and ``pack_chunks=None`` are ``PAPER_BUDGET_TOKENS`` and
    ``PAPER_PACK_CHUNKS`` when a document of the corpus has page breaks (a PDF of two or more
    pages, or text with form feeds), ``ARTICLE_BUDGET_TOKENS`` and ``PACK_CHUNKS`` otherwise
    (``docs/open_papers_protocol.md``, ``docs/paper_pool_protocol.md``).
    ``retriever=None`` uses ``DEFAULT_RETRIEVER`` and falls back to bm25 if its models cannot
    load; a retriever named explicitly raises ``RuntimeError`` instead.
    ``answerable=True`` also fills ``answerable_score`` (the SQuAD2 reader signal on the top
    ``READER_CHUNKS`` chunks, needs the ``[st]`` extra) and ``answerable_threshold``
    (``ANSWERABLE_THRESHOLDS``; ``None`` where none was measured): refuse when the score is below
    it. ``window_tokens`` packs a document longer than it as the window of that size around its
    best chunk (long papers; ``docs/papers_protocol.md``). ``index_dir`` is a writable folder
    that keeps the extracted PDF text and the embeddings per document, so a restart or a new
    file does not index the whole corpus again (a library of uploads). Raises ``ValueError``
    for an empty or over-long query, an unknown retriever or a non-positive ``pack_chunks`` /
    ``budget_tokens`` / ``window_tokens``. ``also_matched`` lists the documents of the pool that
    were not packed. ``documents`` packs only those ids, from the top ``DOCUMENT_CHUNKS`` chunks
    restricted to them (a document without a chunk there is packed from its start; one the budget
    has no room for is in ``also_matched``); an unknown id raises ``ValueError``
    (``docs/followup_protocol.md``).
    """
    query = _validated(query, retriever)
    if pack_chunks is not None and pack_chunks < 1:
        raise ValueError("pack_chunks must be positive")
    index = Path(index_dir) if index_dir is not None else None
    entry = _entry(Path(corpus_dir or DEFAULT_CORPUS_DIR), retriever, index)
    with entry.lock:
        if documents is not None:
            wanted = list(dict.fromkeys(documents))
            unknown = [d for d in wanted if d not in entry.articles]
            if not wanted or unknown:
                raise ValueError(f"documents must name documents of the corpus, got {unknown}")
            hits = entry.pipeline.retrieve(query, max(DOCUMENT_CHUNKS, pack_chunks or 0))
            pool = [Passage(c.doc_id, c.text) for c in hits if c.doc_id in wanted]
            pool += [Passage(d, "") for d in wanted if d not in {p.article_id for p in pool}]
        else:
            hits = entry.pipeline.retrieve(query, pack_chunks or entry.pack_chunks)
            pool = [Passage(c.doc_id, c.text) for c in hits]
        packed = pack_articles(
            pool,
            entry.articles,
            budget_tokens=entry.budget_tokens if budget_tokens is None else budget_tokens,
            overflow=overflow,
            window_tokens=window_tokens,
        )
        top = entry.pipeline.retrieve(query, READER_CHUNKS) if answerable else []
    score = _reader_score(query, [Passage(c.doc_id, c.text) for c in top]) if answerable else None
    return SearchResult(
        query=query,
        retriever=entry.retriever,
        articles=[
            _article(i, p, entry.acts, entry.articles[p.article_id])
            for i, p in enumerate(packed, 1)
        ],
        estimated_tokens=sum(estimate_tokens(p.text) for p in packed),
        answerable_score=score,
        answerable_threshold=ANSWERABLE_THRESHOLDS.get(entry.retriever) if answerable else None,
        also_matched=_unpacked(pool, packed, entry),
    )


def _unpacked(pool: list[Passage], packed: list[Passage], entry: _Entry) -> list[Match]:
    """Documents of ``pool`` not in ``packed``, each at its best (first) chunk."""
    done = {p.article_id for p in packed}
    best: dict[str, Passage] = {}
    for p in pool:
        if p.article_id not in done:
            best.setdefault(p.article_id, p)
    out = []
    for p in best.values():
        a = _article(0, p, entry.acts, entry.articles[p.article_id])
        out.append(Match(a.id, a.act, a.citation, a.pages))
    return out


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
