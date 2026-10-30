"""Neural dense, hybrid and cross-encoder retrieval (protocol: ``docs/neural_protocol.md``).

Everything that needs a model library lives behind the optional ``[st]`` extra and is imported
lazily: importing this module loads no model code, and the default BM25 path never imports it.
One engine serves every benchmark: it ranks *units* (BEIR documents, LegalBench-RAG chunks, EU
chunks) with BM25, a dense bi-encoder, RRF / convex fusion of the two and a cross-encoder rerank
of the fused or BM25 top ``depth``. Rankings are lists of ``(unit index, score)``.
"""

from __future__ import annotations

import copy
import hashlib
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ragkit.bm25 import BM25Index, top_k_indices
from ragkit.chunking import Chunk
from ragkit.modelsignals import CE_MODEL, CE_REVISION, CrossEncoderScorer, PassageScorer

BGE_MODEL = "BAAI/bge-small-en-v1.5"  # MIT
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
E5_MODEL = "intfloat/e5-small-v2"  # MIT
E5_REVISION = "ffb93f3bd4047442299a41ebb6fa998a38507c52"

DEPTH = 100
RRF_K = 60
CE_DEPTHS = (20, 50, 100)
ALPHA_GRID = tuple(i / 10 for i in range(11))
MAX_SEQ_LENGTH = 512
CE_SLICE = 50  # pairs per cross-encoder forward pass (bounds memory on long BEIR documents)

RETRIEVERS = ("bm25", "dense", "hybrid", "hybrid+ce")
# Systems reported by each ``--retriever`` level (docs/neural_protocol.md, "Systems").
LEVEL_SYSTEMS: dict[str, tuple[str, ...]] = {
    "bm25": ("bm25",),
    "dense": ("bm25", "dense", "dense-e5"),
    "hybrid": ("bm25", "dense", "dense-e5", "hybrid", "hybrid-convex"),
    "hybrid+ce": (
        "bm25",
        "dense",
        "dense-e5",
        "hybrid",
        "hybrid-convex",
        *(f"bm25+ce@{d}" for d in CE_DEPTHS),
        *(f"hybrid+ce@{d}" for d in CE_DEPTHS),
    ),
}

Hits = list[tuple[int, float]]
_CE_SYSTEM = re.compile(r"^(?P<first>bm25|hybrid)\+ce@(?P<depth>\d+)$")


def _missing_extra(err: ImportError) -> ImportError:
    return ImportError(
        "neural retrieval needs sentence-transformers (transformers + torch). "
        "Install with: pip install 'ragkit[st]'"
    ).with_traceback(err.__traceback__)


def default_device() -> str:
    """``mps`` when Apple's Metal backend is available, else ``cpu``."""
    try:
        import torch
    except ImportError as err:
        raise _missing_extra(err) from err
    return "mps" if torch.backends.mps.is_available() else "cpu"


class SentenceEmbedder:
    """A pinned sentence-transformers bi-encoder with query / document prefixes."""

    def __init__(
        self,
        name: str,
        revision: str,
        *,
        query_prefix: str = "",
        doc_prefix: str = "",
        device: str | None = None,
        max_seq_length: int = MAX_SEQ_LENGTH,
        batch_size: int = 64,
    ) -> None:
        self.name, self.revision = name, revision
        self.query_prefix, self.doc_prefix = query_prefix, doc_prefix
        self.device = device
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.model_id = f"{name}@{revision}"
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as err:
                raise _missing_extra(err) from err
            device = self.device or default_device()
            model = SentenceTransformer(self.name, revision=self.revision, device=device)
            model.max_seq_length = self.max_seq_length
            self._model = model
        return self._model

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        out = self._load().encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(out, dtype=np.float32)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Unit-norm document vectors, shape ``(N, D)``."""
        return self._encode([self.doc_prefix + t for t in texts])

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Unit-norm query vectors, shape ``(N, D)``."""
        return self._encode([self.query_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        """One unit-norm query vector, shape ``(D,)``."""
        vec: np.ndarray = self.embed_queries([text])[0]
        return vec

    def with_query_prefix(self, prefix: str) -> SentenceEmbedder:
        """A copy that shares this embedder's loaded model but uses another query prefix."""
        clone = copy.copy(self)
        clone._model = self._load()
        clone.query_prefix = prefix
        return clone


def bge_small(device: str | None = None) -> SentenceEmbedder:
    """``BAAI/bge-small-en-v1.5`` with its retrieval query instruction."""
    return SentenceEmbedder(
        BGE_MODEL, BGE_REVISION, query_prefix=BGE_QUERY_INSTRUCTION, device=device
    )


def e5_small(device: str | None = None) -> SentenceEmbedder:
    """``intfloat/e5-small-v2`` with its ``query: `` / ``passage: `` prefixes."""
    return SentenceEmbedder(
        E5_MODEL, E5_REVISION, query_prefix="query: ", doc_prefix="passage: ", device=device
    )


def embed_corpus(
    embedder: SentenceEmbedder, texts: Sequence[str], cache_dir: Path | None
) -> np.ndarray:
    """Document matrix of ``texts``; cached under ``cache_dir`` by model and text digest."""
    path: Path | None = None
    if cache_dir is not None:
        digest = hashlib.sha1(usedforsecurity=False)
        for text in texts:
            digest.update(text.encode("utf-8"))
            digest.update(b"\x00")
        tag = embedder.model_id.replace("/", "_").replace("@", "-")
        path = cache_dir / f"{tag}-{len(texts)}-{digest.hexdigest()[:16]}.npy"
        if path.is_file():
            cached: np.ndarray = np.load(path)
            return cached
    matrix = embedder.embed(texts)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, matrix)
    return matrix


# ---------------------------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------------------------


def rrf_fuse(rankings: Sequence[Hits], k: int = RRF_K) -> Hits:
    """Reciprocal Rank Fusion: ``sum 1 / (k + rank + 1)``, as ``ragkit.hybrid.rrf``."""
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (idx, _) in enumerate(ranking):
            scores[idx] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def _min_max(hits: Hits) -> dict[int, float]:
    if not hits:
        return {}
    values = [s for _, s in hits]
    low, span = min(values), max(values) - min(values)
    return {i: (1.0 if span == 0.0 else (s - low) / span) for i, s in hits}


def convex_fuse(first: Hits, second: Hits, alpha: float) -> Hits:
    """``alpha * minmax(first) + (1 - alpha) * minmax(second)``; a missing hit scores 0."""
    a, b = _min_max(first), _min_max(second)
    fused = {i: alpha * a.get(i, 0.0) + (1.0 - alpha) * b.get(i, 0.0) for i in a.keys() | b.keys()}
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)


def rerank_head(first: Hits, scores: Mapping[int, float], depth: int) -> Hits:
    """Reorder the first ``depth`` hits by ``scores``; the tail keeps its first-stage order."""
    head = sorted(first[:depth], key=lambda hit: scores[hit[0]], reverse=True)
    return [(i, scores[i]) for i, _ in head] + first[depth:]


def alpha_of_best(dev_scores: Mapping[float, float]) -> float:
    """Best dev score; ties go to the alpha closest to 0.5, then the smaller one."""
    return min(dev_scores, key=lambda a: (-dev_scores[a], round(abs(a - 0.5), 9), a))


# ---------------------------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------------------------


@dataclass
class DenseSide:
    """A document matrix with its embedder and a query-vector cache."""

    embedder: SentenceEmbedder
    matrix: np.ndarray
    queries: dict[str, np.ndarray] = field(default_factory=dict)

    def prepare(self, queries: Sequence[str]) -> None:
        """Embed every not yet seen query in one batched call."""
        todo = [q for q in dict.fromkeys(queries) if q not in self.queries]
        if todo:
            for q, vec in zip(todo, self.embedder.embed_queries(todo), strict=True):
                self.queries[q] = vec

    def unit_scores(self, query: str) -> np.ndarray:
        """Score of every unit for an already prepared query (cosine; overridden for pooling)."""
        scores: np.ndarray = self.matrix @ self.queries[query]
        return scores

    def search(self, query: str, k: int, exclude: int | None = None) -> Hits:
        """Top ``k`` units by cosine (units are unit-norm), best first."""
        self.prepare([query])
        scores = self.unit_scores(query)
        want = k + 1 if exclude is not None else k
        want = min(want, len(scores))
        top = np.argpartition(-scores, want - 1)[:want] if want < len(scores) else np.arange(want)
        order = sorted(top.tolist(), key=lambda i: (-float(scores[i]), i))
        return [(i, float(scores[i])) for i in order if i != exclude][:k]


@dataclass(frozen=True)
class Candidates:
    """First-stage lists of one query, from which every system is derived."""

    bm25: Hits
    dense: dict[str, Hits]


class NeuralEngine:
    """BM25 + dense sides + optional cross-encoder over one list of units."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        bm25: BM25Index,
        dense: Mapping[str, DenseSide],
        ce: PassageScorer | None = None,
        *,
        depth: int = DEPTH,
        rrf_k: int = RRF_K,
    ) -> None:
        if len(bm25.chunks) != len(chunks):
            raise ValueError("the BM25 index must hold exactly the engine's units")
        self.chunks = list(chunks)
        self.bm25 = bm25
        self.dense = dict(dense)
        self.ce = ce
        self.depth = depth
        self.rrf_k = rrf_k
        self.ce_cache: dict[tuple[str, int], float] = {}
        self._index_of = {c.id: i for i, c in enumerate(self.chunks)}

    def prepare(self, queries: Sequence[str]) -> None:
        """Batch-embed the queries on every dense side."""
        for side in self.dense.values():
            side.prepare(queries)

    def candidates(self, query: str, exclude_id: str | None = None) -> Candidates:
        """BM25 and dense top ``depth`` lists of one query (a self hit is removed)."""
        exclude = self._index_of.get(exclude_id) if exclude_id is not None else None
        scores = self.bm25.score_all(query)
        if exclude is not None:
            scores[exclude] = 0.0
        return Candidates(
            bm25=top_k_indices(scores, self.depth),
            dense={n: s.search(query, self.depth, exclude) for n, s in self.dense.items()},
        )

    def ce_scores(self, query: str, indices: Sequence[int]) -> dict[int, float]:
        """Cross-encoder logits of ``(query, unit text)``; scored once per pair."""
        if self.ce is None:
            raise RuntimeError("this engine was built without a cross-encoder")
        todo = [i for i in indices if (query, i) not in self.ce_cache]
        for start in range(0, len(todo), CE_SLICE):
            part = todo[start : start + CE_SLICE]
            for i, score in zip(
                part, self.ce.score(query, [self.chunks[i].text for i in part]), strict=True
            ):
                self.ce_cache[(query, i)] = score
        return {i: self.ce_cache[(query, i)] for i in indices}

    def system(self, name: str, query: str, cands: Candidates, alpha: float | None = None) -> Hits:
        """Ranking of one protocol system (``bm25``, ``dense``, ``hybrid+ce@50`` ...)."""
        match = _CE_SYSTEM.match(name)
        if match:
            first = self.system(match.group("first"), query, cands, alpha)[: self.depth]
            scores = self.ce_scores(query, [i for i, _ in first])
            return rerank_head(first, scores, int(match.group("depth")))
        if name == "bm25":
            return cands.bm25
        if name == "dense":
            return cands.dense["bge"]
        if name == "dense-e5":
            return cands.dense["e5"]
        if name == "hybrid":
            return rrf_fuse([cands.bm25, cands.dense["bge"]], self.rrf_k)[: self.depth]
        if name == "hybrid-convex":
            if alpha is None:
                raise ValueError("hybrid-convex needs an alpha chosen on a dev split")
            return convex_fuse(cands.bm25, cands.dense["bge"], alpha)[: self.depth]
        raise ValueError(f"unknown system {name!r}")


def build_engine(
    chunks: Sequence[Chunk],
    bm25: BM25Index,
    level: str,
    *,
    cache_dir: Path | None = None,
    device: str | None = None,
    dense_e5: bool = True,
) -> NeuralEngine:
    """Encode the units and build an engine for the models ``level`` needs."""
    if level not in RETRIEVERS or level == "bm25":
        raise ValueError(f"level must be one of {RETRIEVERS[1:]}, got {level!r}")
    texts = [c.text for c in chunks]
    sides: dict[str, DenseSide] = {}
    wanted = [("bge", bge_small(device))]
    if dense_e5:
        wanted.append(("e5", e5_small(device)))
    for key, embedder in wanted:
        sides[key] = DenseSide(embedder, embed_corpus(embedder, texts, cache_dir))
    ce: PassageScorer | None = None
    if level == "hybrid+ce":
        ce = CrossEncoderScorer(CE_MODEL, CE_REVISION, device or default_device())
    return NeuralEngine(chunks, bm25, sides, ce)


# ---------------------------------------------------------------------------------------------
# A drop-in ``Retriever`` for ``RAGPipeline`` (ragkit ask / answer --retriever)
# ---------------------------------------------------------------------------------------------


class NeuralChunkRetriever:
    """``add`` / ``search`` over chunks with the ``dense``, ``hybrid`` or ``hybrid+ce`` system."""

    def __init__(self, level: str, *, device: str | None = None) -> None:
        if level not in ("dense", "hybrid", "hybrid+ce"):
            raise ValueError(f"unsupported retriever {level!r}")
        self.level = level
        self.device = device
        self._engine: NeuralEngine | None = None

    def add(self, chunks: Sequence[Chunk]) -> None:
        """Index the chunks (replaces any earlier index)."""
        bm25 = BM25Index()
        bm25.add(chunks)
        self._engine = build_engine(chunks, bm25, self.level, device=self.device, dense_e5=False)

    def search(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        """Top ``k`` chunks of the selected system (the cross-encoder reranks the top 100)."""
        if self._engine is None:
            return []
        engine = self._engine
        cands = engine.candidates(query)
        system = {"dense": "dense", "hybrid": "hybrid", "hybrid+ce": "hybrid+ce@100"}[self.level]
        hits = engine.system(system, query, cands)[:k]
        return [(engine.chunks[i], score) for i, score in hits]
