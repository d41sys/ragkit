"""Chunking strategies for document indexing.

Provides fixed-window, recursive-character, sentence-based, semantic and structure-aware
chunking. Every strategy returns a list of Chunk objects with exact character spans.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragkit.embed import Embedder


@dataclass(frozen=True)
class Chunk:
    """A bounded passage extracted from a document with exact character offsets.

    ``title`` is the heading of the enclosing document or article when the chunker knows it
    (``structured``); it is empty otherwise. ``start``/``end`` always locate the body text.
    """

    id: str
    doc_id: str
    text: str
    start: int
    end: int
    title: str = ""

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"start cannot be negative: {self.start}")
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) must be >= start ({self.start})")


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def fixed(text: str, doc_id: str, size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Split text into fixed-length character windows with overlap.

    Args:
        text: Source text.
        doc_id: Document identifier.
        size: Target window character size.
        overlap: Character overlap between consecutive windows.

    Returns:
        List of Chunks with exact start and end offsets.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and smaller than size")

    stripped = text
    n = len(stripped)
    if n == 0:
        return []

    chunks: list[Chunk] = []
    step = size - overlap
    i = 0
    idx = 0
    while i < n:
        end = min(i + size, n)
        chunk_text = stripped[i:end]
        chunks.append(
            Chunk(
                id=f"{doc_id}#fixed_{idx}",
                doc_id=doc_id,
                text=chunk_text,
                start=i,
                end=end,
            )
        )
        idx += 1
        if end >= n:
            break
        i += step

    return chunks


def sentence(
    text: str,
    doc_id: str,
    max_tokens: int = 100,
    overlap: int = 0,
) -> list[Chunk]:
    """Split text into sentence-bounded chunks of at most max_tokens words.

    Args:
        text: Source text.
        doc_id: Document identifier.
        max_tokens: Approximate token (word) limit per chunk.
        overlap: Sentence overlap count between consecutive chunks.

    Returns:
        List of sentence-packed Chunks.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if not text.strip():
        return []

    # Find sentences with spans
    sentence_spans: list[tuple[int, int, str]] = []
    cursor = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        end = match.start()
        seg = text[cursor:end]
        if seg.strip():
            sentence_spans.append((cursor, end, seg))
        cursor = match.end()

    if cursor < len(text):
        tail = text[cursor:]
        if tail.strip():
            sentence_spans.append((cursor, len(text), tail))

    if not sentence_spans:
        return []

    chunks: list[Chunk] = []
    idx = 0
    i = 0
    num_sents = len(sentence_spans)

    while i < num_sents:
        group: list[tuple[int, int, str]] = []
        token_count = 0
        j = i
        while j < num_sents:
            s_start, s_end, s_text = sentence_spans[j]
            words = len(s_text.split())
            if group and (token_count + words > max_tokens):
                break
            group.append((s_start, s_end, s_text))
            token_count += words
            j += 1

        first_span = group[0]
        last_span = group[-1]
        c_start = first_span[0]
        c_end = last_span[1]
        chunk_text = text[c_start:c_end]

        chunks.append(
            Chunk(
                id=f"{doc_id}#sent_{idx}",
                doc_id=doc_id,
                text=chunk_text,
                start=c_start,
                end=c_end,
            )
        )
        idx += 1

        if j >= num_sents:
            break

        # Advance with overlap
        advance = max(1, len(group) - overlap) if overlap > 0 else len(group)
        i += advance

    return chunks


def recursive(
    text: str,
    doc_id: str,
    separators: Sequence[str] | None = None,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """Recursively split text by structural separators while keeping exact spans.

    Args:
        text: Source text.
        doc_id: Document identifier.
        separators: Priority list of separators (coarse to fine).
        chunk_size: Maximum character length for merged chunks.
        overlap: Character overlap between consecutive chunks.

    Returns:
        List of Chunks with character offsets.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    if not text.strip():
        return []

    seps = ["\n\n", "\n", ". ", " ", ""] if separators is None else list(separators)

    # Recursive split helper returning spans (start, end)
    def _split_spans(start_pos: int, seg: str, sep_idx: int) -> list[tuple[int, int]]:
        if len(seg) <= chunk_size or sep_idx >= len(seps):
            return [(start_pos, start_pos + len(seg))]

        sep = seps[sep_idx]
        if sep == "":
            # Character-level fallback
            spans: list[tuple[int, int]] = []
            curr = 0
            while curr < len(seg):
                nxt = min(curr + chunk_size, len(seg))
                spans.append((start_pos + curr, start_pos + nxt))
                curr = nxt
            return spans

        parts = seg.split(sep)
        if len(parts) <= 1:
            return _split_spans(start_pos, seg, sep_idx + 1)

        result_spans: list[tuple[int, int]] = []
        offset = 0
        for p in parts:
            p_len = len(p)
            if p_len > 0:
                p_start = start_pos + offset
                sub_spans = _split_spans(p_start, p, sep_idx + 1)
                result_spans.extend(sub_spans)
            offset += p_len + len(sep)

        return result_spans

    raw_spans = _split_spans(0, text, 0)
    if not raw_spans:
        return []

    # Merge spans up to chunk_size
    merged_spans: list[tuple[int, int]] = []
    current_start = raw_spans[0][0]
    current_end = raw_spans[0][1]

    for s_start, s_end in raw_spans[1:]:
        # If adding this piece stays within chunk_size
        if (s_end - current_start) <= chunk_size:
            current_end = s_end
        else:
            merged_spans.append((current_start, current_end))
            # Start new segment, optionally with overlap
            if overlap > 0:
                target_back = current_end - overlap
                current_start = max(0, min(target_back, s_start))
            else:
                current_start = s_start
            current_end = max(s_end, current_start)

    merged_spans.append((current_start, current_end))

    chunks: list[Chunk] = []
    for idx, (sp_start, sp_end) in enumerate(merged_spans):
        c_text = text[sp_start:sp_end]
        if c_text.strip():
            chunks.append(
                Chunk(
                    id=f"{doc_id}#rec_{idx}",
                    doc_id=doc_id,
                    text=c_text,
                    start=sp_start,
                    end=sp_end,
                )
            )

    return chunks


_HEADING = re.compile(r"[ \t\r\n]*#[ \t]+([^\n]*?)[ \t]*(?:\n|$)")
_LINE = r"[^\n]*\S[^\n]*"
_BLOCK = re.compile(rf"{_LINE}(?:\n{_LINE})*")
_PARAGRAPH_START = re.compile(r"(?:\d+\.|\(\d+\))\s")


def structured(
    text: str,
    doc_id: str,
    min_chars: int = 250,
    max_chars: int = 1000,
    prepend_heading: bool = True,
) -> list[Chunk]:
    """Split an article along its paragraph/point structure and keep its heading.

    The leading ``# `` line is the heading. The body is cut into blank-line separated blocks;
    a block starting with ``N.`` or ``(N)`` opens a new paragraph unit and every other block
    (``(a)`` points, continuation lines) stays with the unit it follows. Units shorter than
    ``min_chars`` are merged with the following ones, a chunk closes once it reaches
    ``min_chars`` or the next unit would push it past ``max_chars``. A unit longer than
    ``max_chars`` is split at block (point) boundaries, and a single block longer than that
    falls back to ``recursive``.

    ``start``/``end`` are exact spans of the body in ``text`` (the heading is outside them).
    ``title`` is the heading; with ``prepend_heading`` the chunk text is the heading, a blank
    line and the body, otherwise just the body.
    """
    if min_chars <= 0 or max_chars < min_chars:
        raise ValueError("require 0 < min_chars <= max_chars")

    heading = _HEADING.match(text)
    title = heading.group(1) if heading else ""
    body_start = heading.end() if heading else 0
    blocks = [(m.start(), m.end()) for m in _BLOCK.finditer(text, body_start)]

    units: list[list[tuple[int, int]]] = []
    for span in blocks:
        if not units or _PARAGRAPH_START.match(text, span[0]):
            units.append([span])
        else:
            units[-1].append(span)

    pieces: list[tuple[int, int]] = []
    for unit in units:
        if unit[-1][1] - unit[0][0] <= max_chars:
            pieces.append((unit[0][0], unit[-1][1]))
            continue
        start = end = -1
        for b_start, b_end in unit:
            if b_end - b_start > max_chars:
                if start >= 0:
                    pieces.append((start, end))
                    start = end = -1
                pieces.extend(
                    (b_start + c.start, b_start + c.end)
                    for c in recursive(text[b_start:b_end], doc_id, chunk_size=max_chars, overlap=0)
                )
            elif start < 0:
                start, end = b_start, b_end
            elif b_end - start <= max_chars:
                end = b_end
            else:
                pieces.append((start, end))
                start, end = b_start, b_end
        if start >= 0:
            pieces.append((start, end))

    spans: list[tuple[int, int]] = []
    current: tuple[int, int] | None = None
    for p_start, p_end in pieces:
        if current is None:
            current = (p_start, p_end)
        elif p_end - current[0] > max_chars:
            spans.append(current)
            current = (p_start, p_end)
        else:
            current = (current[0], p_end)
        if current[1] - current[0] >= min_chars:
            spans.append(current)
            current = None
    if current is not None:
        spans.append(current)

    chunks: list[Chunk] = []
    for idx, (s, e) in enumerate(spans):
        body = text[s:e]
        chunk_text = f"{title}\n\n{body}" if prepend_heading and title else body
        chunks.append(
            Chunk(
                id=f"{doc_id}#struct_{idx}",
                doc_id=doc_id,
                text=chunk_text,
                start=s,
                end=e,
                title=title,
            )
        )
    return chunks


def semantic(
    text: str,
    doc_id: str,
    threshold: float = 0.5,
    embedder: Embedder | None = None,
) -> list[Chunk]:
    """Split text at semantic shift boundaries based on adjacent sentence cosine similarity.

    Args:
        text: Source text.
        doc_id: Document identifier.
        threshold: Cosine similarity threshold below which a boundary is created.
        embedder: Embedder Protocol instance. Defaults to HashingEmbedder if None.

    Returns:
        List of Chunks split at topic/semantic shifts.
    """
    if not text.strip():
        return []

    # Local import to avoid circular dependency
    if embedder is None:
        from ragkit.embed import HashingEmbedder

        embedder = HashingEmbedder(dim=256)

    # Collect sentence spans
    sentence_spans: list[tuple[int, int, str]] = []
    cursor = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        end = match.start()
        seg = text[cursor:end]
        if seg.strip():
            sentence_spans.append((cursor, end, seg))
        cursor = match.end()

    if cursor < len(text):
        tail = text[cursor:]
        if tail.strip():
            sentence_spans.append((cursor, len(text), tail))

    if not sentence_spans:
        return []

    if len(sentence_spans) == 1:
        s0, e0, t0 = sentence_spans[0]
        return [Chunk(id=f"{doc_id}#sem_0", doc_id=doc_id, text=t0, start=s0, end=e0)]

    # Compute sentence embeddings
    sent_texts = [s[2] for s in sentence_spans]
    embeddings = embedder.embed(sent_texts)

    # Group sentences into semantic clusters
    chunks: list[Chunk] = []
    group_start_idx = 0
    chunk_idx = 0

    for i in range(len(sentence_spans) - 1):
        # Cosine similarity between sentence i and i+1
        vec_a = embeddings[i]
        vec_b = embeddings[i + 1]
        sim = float(vec_a @ vec_b)  # normalized embeddings

        if sim < threshold:
            # Cut boundary
            c_start = sentence_spans[group_start_idx][0]
            c_end = sentence_spans[i][1]
            chunk_text = text[c_start:c_end]
            chunks.append(
                Chunk(
                    id=f"{doc_id}#sem_{chunk_idx}",
                    doc_id=doc_id,
                    text=chunk_text,
                    start=c_start,
                    end=c_end,
                )
            )
            chunk_idx += 1
            group_start_idx = i + 1

    # Final group
    c_start = sentence_spans[group_start_idx][0]
    c_end = sentence_spans[-1][1]
    chunk_text = text[c_start:c_end]
    chunks.append(
        Chunk(
            id=f"{doc_id}#sem_{chunk_idx}",
            doc_id=doc_id,
            text=chunk_text,
            start=c_start,
            end=c_end,
        )
    )

    return chunks
