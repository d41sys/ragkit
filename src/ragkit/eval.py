"""Retrieval and generation metrics: P@k, R@k, MRR, nDCG, faithfulness, answer relevance."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ragkit.bm25 import tokenize

if TYPE_CHECKING:
    from ragkit.embed import Embedder

_STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "of",
    "to",
    "in",
    "on",
    "at",
    "for",
    "with",
    "and",
    "or",
    "as",
    "by",
    "that",
    "this",
    "it",
    "its",
    "from",
    "what",
    "how",
    "which",
    "who",
    "why",
    "when",
    "where",
}

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def precision_at_k(
    retrieved: Sequence[str],
    relevant: Sequence[str] | set[str],
    k: int,
) -> float:
    """Fraction of top-k retrieved items that are relevant."""
    if k <= 0 or not retrieved:
        return 0.0
    top = retrieved[:k]
    rel_set = set(relevant)
    if not rel_set:
        return 0.0
    hits = sum(1 for item in top if item in rel_set)
    return hits / k


def recall_at_k(
    retrieved: Sequence[str],
    relevant: Sequence[str] | set[str],
    k: int,
) -> float:
    """Fraction of relevant items found in the top-k retrieved items."""
    if k <= 0 or not retrieved:
        return 0.0
    rel_set = set(relevant)
    if not rel_set:
        return 0.0
    top_set = set(retrieved[:k])
    return len(top_set & rel_set) / len(rel_set)


def mrr(
    retrieved: Sequence[str],
    relevant: Sequence[str] | set[str],
) -> float:
    """Mean Reciprocal Rank for a single query (1 / rank of first relevant item)."""
    rel_set = set(relevant)
    for rank, item in enumerate(retrieved):
        if item in rel_set:
            return 1.0 / (rank + 1)
    return 0.0


def dcg_at_k(
    retrieved: Sequence[str],
    graded_or_relevant: Sequence[str] | set[str] | dict[str, int],
    k: int,
) -> float:
    """Discounted Cumulative Gain at k using logarithmic discounting."""
    if k <= 0 or not retrieved:
        return 0.0

    if isinstance(graded_or_relevant, dict):
        graded = graded_or_relevant
    else:
        rel_set = set(graded_or_relevant)
        graded = {item: 1 for item in rel_set}

    dcg = 0.0
    for i, item in enumerate(retrieved[:k]):
        rel = graded.get(item, 0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(i + 2)
    return dcg


def ndcg_at_k(
    retrieved: Sequence[str],
    graded_or_relevant: Sequence[str] | set[str] | dict[str, int],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain at rank k."""
    if k <= 0 or not retrieved:
        return 0.0

    if isinstance(graded_or_relevant, dict):
        graded = graded_or_relevant
    else:
        rel_set = set(graded_or_relevant)
        graded = {item: 1 for item in rel_set}

    if not graded:
        return 0.0

    actual_dcg = dcg_at_k(retrieved, graded, k)
    if actual_dcg == 0.0:
        return 0.0

    # Ideal ranking: items with highest relevance first
    ideal_items = sorted(graded.keys(), key=lambda x: graded[x], reverse=True)
    ideal_dcg = dcg_at_k(ideal_items, graded, k)
    if ideal_dcg == 0.0:
        return 0.0

    return actual_dcg / ideal_dcg


def _content_tokens(text: str) -> set[str]:
    """Tokenize and remove small stopwords."""
    tokens = tokenize(text)
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


def faithfulness(
    answer: str,
    context_chunks: Sequence[str],
    threshold: float = 0.5,
) -> float:
    """Fraction of answer sentences supported by retrieved chunks via token overlap >= threshold.

    Args:
        answer: Generated answer text.
        context_chunks: List of retrieved context passages.
        threshold: Minimum content token overlap fraction to consider a sentence supported.

    Returns:
        Faithfulness score in [0.0, 1.0].
    """
    raw_sentences = _SENTENCE_BOUNDARY.split(answer.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    if not sentences:
        return 0.0

    if not context_chunks:
        return 0.0

    chunk_token_sets = [_content_tokens(c) for c in context_chunks if c.strip()]
    if not chunk_token_sets:
        return 0.0

    supported_count = 0
    for sent in sentences:
        sent_tokens = _content_tokens(sent)
        if not sent_tokens:
            # If a sentence has no content tokens, consider it neutral/supported
            supported_count += 1
            continue

        is_supported = False
        for c_tokens in chunk_token_sets:
            overlap = len(sent_tokens & c_tokens) / len(sent_tokens)
            if overlap >= threshold:
                is_supported = True
                break

        if is_supported:
            supported_count += 1

    return supported_count / len(sentences)


def answer_relevance(
    question: str,
    answer: str,
    embedder: Embedder | None = None,
) -> float:
    """Compute cosine similarity between question and answer embeddings."""
    if not question.strip() or not answer.strip():
        return 0.0

    if embedder is None:
        from ragkit.embed import HashingEmbedder

        embedder = HashingEmbedder(dim=512)

    qv = embedder.embed_query(question)
    av = embedder.embed_query(answer)
    cos = float(qv @ av)
    return max(0.0, min(1.0, cos))
