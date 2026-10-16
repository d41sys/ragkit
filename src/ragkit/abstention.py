"""Abstention signals and their evaluation.

Every signal is oriented so that a *higher* value means "more likely answerable"; the sign is
fixed in advance and never flipped to flatter a result, so an AUROC below 0.5 is reported as is.
Evaluation uses only numpy: AUROC with (paired) stratified bootstrap, risk-coverage, stratified
k-fold cross-validation of a threshold chosen on the training folds.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ragkit.bm25 import BM25Index, tokenize
from ragkit.chunking import Chunk
from ragkit.hybrid import HybridRetriever
from ragkit.pipeline import STOPWORDS, RAGPipeline, query_support

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.intp]
Scores = Sequence[float] | FloatArray
Flags = Sequence[bool] | BoolArray

SIGNALS: dict[str, str] = {
    "query_support": "share of query content terms in the best of the top-5 chunks (current)",
    "idf_support": "same, weighted by BM25 IDF (unseen terms get the maximum IDF)",
    "bm25_top": "top raw BM25 score over the whole corpus",
    "bm25_norm": "top BM25 score / its upper bound for this query (sum of idf * (k1 + 1))",
    "bm25_margin": "BM25 top1 - top2 score",
    "dense_top": "top hashing-embedder cosine",
    "dense_margin": "dense top1 - top2 cosine",
    "rerank_top": "top LexicalReranker score over the hybrid candidates",
    "rerank_margin": "LexicalReranker top1 - top2 score",
}


@dataclass(frozen=True)
class SignalRow:
    """Abstention signals for one question plus the documents it retrieved."""

    values: dict[str, float]
    top_doc_ids: list[str]


def _top2(scores: Sequence[float]) -> tuple[float, float]:
    """Best and second-best score; a missing entry counts as 0.0."""
    top = scores[0] if len(scores) > 0 else 0.0
    second = scores[1] if len(scores) > 1 else 0.0
    return top, second


def compute_signals(pipeline: RAGPipeline, query: str, top_k: int = 5) -> SignalRow:
    """All abstention signals for one query from the pipeline's own components.

    Needs the default hybrid retriever (BM25 + dense) and a reranker. ``query_support`` equals
    ``RAGPipeline.answer(query).support`` by construction (same retrieval, same top chunks).
    """
    retriever = pipeline.retriever
    if not isinstance(retriever, HybridRetriever) or pipeline.reranker is None:
        raise TypeError("compute_signals needs a HybridRetriever and a reranker")
    bm25, dense = retriever.bm25, retriever.dense

    q_terms = tokenize(query)
    bm_top, bm_second = _top2([s for _, s in bm25.search(query, k=2)])
    upper = sum(bm25.idf(t) * (bm25.k1 + 1.0) for t in q_terms)
    dn_top, dn_second = _top2([s for _, s in dense.search(query, k=2)])

    scored = pipeline.retrieve_scored(query, top_k=top_k)
    chunks = [c for c, _ in scored]
    rr_top, rr_second = _top2([s for _, s in scored])

    return SignalRow(
        values={
            "query_support": max((query_support(query, c) for c in chunks), default=0.0),
            "idf_support": _idf_support(bm25, q_terms, chunks),
            "bm25_top": bm_top,
            "bm25_norm": bm_top / upper if upper > 0.0 else 0.0,
            "bm25_margin": bm_top - bm_second,
            "dense_top": dn_top,
            "dense_margin": dn_top - dn_second,
            "rerank_top": rr_top,
            "rerank_margin": rr_top - rr_second,
        },
        top_doc_ids=list(dict.fromkeys(c.doc_id for c in chunks)),
    )


def _idf_support(bm25: BM25Index, q_terms: Sequence[str], chunks: Sequence[Chunk]) -> float:
    """IDF-weighted share of the query's content terms found in the best retrieved chunk."""
    terms = {t for t in q_terms if t not in STOPWORDS}
    if not terms or not chunks:
        return 0.0
    unseen_idf = math.log((len(bm25.chunks) + 0.5) / 0.5 + 1.0)
    weight = {t: bm25.idf(t) if bm25.df.get(t, 0) > 0 else unseen_idf for t in terms}
    total = sum(weight.values())
    best = max(sum(weight[t] for t in terms & set(tokenize(c.text))) for c in chunks)
    return best / total


@dataclass(frozen=True)
class SignalTable:
    """Per-question signals; answerable questions first, then the unanswerable ones."""

    scores: dict[str, FloatArray]
    answerable: BoolArray
    retrieval_miss: BoolArray  # answerable question with no relevant article in the top 5

    @property
    def n_answerable(self) -> int:
        return int(self.answerable.sum())

    @property
    def n_unanswerable(self) -> int:
        return int((~self.answerable).sum())


def collect_signals(
    documents: Sequence[tuple[str, str]],
    qa_items: Sequence[dict[str, Any]],
    unanswerable: Sequence[str],
) -> SignalTable:
    """Index ``documents`` with the default pipeline and score every question."""
    pipeline = RAGPipeline()
    pipeline.index_documents(documents)
    rows: dict[str, list[float]] = {name: [] for name in SIGNALS}
    miss: list[bool] = []
    for item in qa_items:
        row = compute_signals(pipeline, str(item["q"]))
        relevant = {str(d) for d in item["relevant_doc_ids"]}
        miss.append(not relevant & set(row.top_doc_ids))
        for name, value in row.values.items():
            rows[name].append(value)
    for q in unanswerable:
        for name, value in compute_signals(pipeline, q).values.items():
            rows[name].append(value)
        miss.append(False)
    n_ans = len(qa_items)
    labels = np.array([True] * n_ans + [False] * len(unanswerable), dtype=np.bool_)
    return SignalTable(
        scores={name: np.asarray(v, dtype=np.float64) for name, v in rows.items()},
        answerable=labels,
        retrieval_miss=np.array(miss, dtype=np.bool_),
    )


def _split(scores: Scores, labels: Flags) -> tuple[FloatArray, FloatArray]:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.bool_)
    if s.shape != y.shape or s.ndim != 1:
        raise ValueError("scores and labels must be 1-D and of the same length")
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("need at least one answerable and one unanswerable question")
    return pos, neg


def _auroc_rows(pos: FloatArray, neg: FloatArray) -> FloatArray:
    """AUROC along the last axis: P(pos > neg) + 0.5 P(pos == neg), over all pairs."""
    p = pos[..., :, None]
    n = neg[..., None, :]
    return np.asarray(((p > n) + 0.5 * (p == n)).mean(axis=(-2, -1)), dtype=np.float64)


def auroc(scores: Scores, labels: Flags) -> float:
    """Area under the ROC curve for ``labels`` True (answerable) vs False; ties count 1/2."""
    pos, neg = _split(scores, labels)
    return float(_auroc_rows(pos, neg))


def _resample(n_pos: int, n_neg: int, n_resamples: int, seed: int) -> tuple[IntArray, IntArray]:
    """Stratified bootstrap indices: answerable and unanswerable are resampled separately."""
    rng = np.random.default_rng(seed)
    return (
        rng.integers(0, n_pos, size=(n_resamples, n_pos)),
        rng.integers(0, n_neg, size=(n_resamples, n_neg)),
    )


def _percentile(values: FloatArray, confidence: float) -> tuple[float, float]:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    tail = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(values, [tail, 1.0 - tail])
    return float(lo), float(hi)


def auroc_ci(
    scores: Scores,
    labels: Flags,
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """AUROC with a percentile-bootstrap CI (classes resampled separately, seeded)."""
    pos, neg = _split(scores, labels)
    ip, ineg = _resample(len(pos), len(neg), n_resamples, seed)
    boot = _auroc_rows(pos[ip], neg[ineg])
    lo, hi = _percentile(boot, confidence)
    return float(_auroc_rows(pos, neg)), lo, hi


def paired_auroc_ci(
    scores_a: Scores,
    scores_b: Scores,
    labels: Flags,
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI of ``AUROC(a) - AUROC(b)``; both signals use the same resampled questions."""
    pos_a, neg_a = _split(scores_a, labels)
    pos_b, neg_b = _split(scores_b, labels)
    ip, ineg = _resample(len(pos_a), len(neg_a), n_resamples, seed)
    boot = _auroc_rows(pos_a[ip], neg_a[ineg]) - _auroc_rows(pos_b[ip], neg_b[ineg])
    lo, hi = _percentile(boot, confidence)
    return float(_auroc_rows(pos_a, neg_a) - _auroc_rows(pos_b, neg_b)), lo, hi


def risk_at_coverage(scores: Scores, bad: Flags, coverage: float) -> float:
    """Share of ``bad`` items among the ``round(coverage * n)`` highest-scoring questions.

    Questions tied at the cut-off are split fractionally (their mean badness times the number of
    slots left), so the result does not depend on input order.
    """
    s = np.asarray(scores, dtype=np.float64)
    b = np.asarray(bad, dtype=np.bool_)
    if s.shape != b.shape or s.ndim != 1 or len(s) == 0:
        raise ValueError("scores and bad must be non-empty 1-D arrays of the same length")
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    m = max(1, math.floor(coverage * len(s) + 0.5))
    cutoff = np.sort(s)[::-1][m - 1]
    above = s > cutoff
    tied = s == cutoff
    slots = m - int(above.sum())
    expected_bad = float(b[above].sum()) + slots * float(b[tied].mean())
    return expected_bad / m


def stratified_folds(labels: Flags, k: int = 5, seed: int = 0) -> list[tuple[IntArray, IntArray]]:
    """Deterministic stratified k-fold split by question: ``(train_idx, test_idx)`` per fold."""
    y = np.asarray(labels, dtype=np.bool_)
    if k < 2:
        raise ValueError("k must be at least 2")
    if min(int(y.sum()), int((~y).sum())) < k:
        raise ValueError("each class needs at least k questions")
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), dtype=np.intp)
    for cls in (True, False):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        fold_of[idx] = np.arange(len(idx)) % k
    return [(np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)) for f in range(k)]


def best_threshold(scores: Scores, labels: Flags) -> float:
    """Threshold ``t`` (answer when ``score >= t``) maximising balanced accuracy.

    Balanced accuracy is the mean of the answer rate on answerable and the abstain rate on
    unanswerable questions. Candidates are the midpoints between distinct scores plus
    ``-inf`` (answer everything) and ``+inf`` (abstain on everything); ties go to the lowest.
    """
    pos, neg = _split(scores, labels)
    vals = np.unique(np.concatenate([pos, neg]))
    cands = np.concatenate([[-np.inf], (vals[:-1] + vals[1:]) / 2.0, [np.inf]])
    answered = (pos[None, :] >= cands[:, None]).mean(axis=1)
    abstained = (neg[None, :] < cands[:, None]).mean(axis=1)
    return float(cands[int(np.argmax((answered + abstained) / 2.0))])


@dataclass(frozen=True)
class CVResult:
    """Held-out decisions from cross-validated thresholds."""

    thresholds: list[float]  # one per fold, chosen on that fold's training questions only
    accepted: BoolArray  # held-out decision for every question (True = answer)


def cross_validated_accept(scores: Scores, labels: Flags, k: int = 5, seed: int = 0) -> CVResult:
    """Choose a threshold on each training split and apply it to the held-out questions."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.bool_)
    accepted = np.zeros(len(s), dtype=np.bool_)
    thresholds: list[float] = []
    for train, test in stratified_folds(y, k=k, seed=seed):
        t = best_threshold(s[train], y[train])
        thresholds.append(t)
        accepted[test] = s[test] >= t
    return CVResult(thresholds, accepted)


def calibrate_threshold(answerable_scores: Scores, quantile: float = 0.05) -> float:
    """Answerable-only calibration: the ``quantile`` of the scores (no unanswerable data)."""
    return float(np.quantile(np.asarray(answerable_scores, dtype=np.float64), quantile))


def accept_counts(accepted: Flags, labels: Flags) -> tuple[int, int, int, int]:
    """``(answered_answerable, n_answerable, abstained_unanswerable, n_unanswerable)``."""
    a = np.asarray(accepted, dtype=np.bool_)
    y = np.asarray(labels, dtype=np.bool_)
    return int((a & y).sum()), int(y.sum()), int((~a & ~y).sum()), int((~y).sum())
