"""Parent-document (small-to-big) context: retrieve on chunks, answer from whole articles.

Protocol and frozen parameters: ``docs/generation_protocol.md`` (Phase L). The chunks that rank
best are only used to pick articles; ``GEN`` then sees the full text of the distinct parent
articles, in rank order, within a token budget. ``pack_articles`` fills the budget instead of
stopping at ``k`` articles (opt-in, ``docs/packing_protocol.md``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

ARTICLE_K = 3
ARTICLE_BUDGET_TOKENS = 12_000
PAPER_BUDGET_TOKENS = 36_000  # search() default for paged corpora (open_papers_protocol.md)
PACK_CHUNKS = 10  # chunks read by budget packing (dev-chosen, docs/packing_protocol.md)
PACK_OVERFLOW = "stop"  # dev-chosen overflow policy of budget packing
CHARS_PER_TOKEN = 4  # 337 kB of corpus = about 85 k tokens, the K-R2 full-context figure


@dataclass(frozen=True)
class Passage:
    """A retrieved chunk (or a whole article) labelled with the article it comes from."""

    article_id: str
    text: str


def estimate_tokens(text: str) -> int:
    """Token estimate used for the budget: characters / 4, rounded up."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


OVERFLOW_POLICIES = ("stop", "skip")


def _window(text: str, bests: Sequence[str], width: int) -> str:
    """Windows centred on each of ``bests`` sharing ``width`` characters, merged, in text order.

    One chunk gives the ``width``-character window around it (from the start if not found);
    overlapping windows merge and the rest are joined by a blank line, still within ``width``.
    """
    each = (width - 2 * (len(bests) - 1)) // len(bests)
    spans = []
    for best in bests:
        hit = text.find(best)
        start = max(0, min(hit + len(best) // 2 - each // 2, len(text) - each)) if hit > 0 else 0
        spans.append((start, start + each))
    spans.sort()
    merged = [list(spans[0])]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return "\n\n".join(text[start:end] for start, end in merged)


def parent_articles(
    chunks: Sequence[Passage],
    articles: Mapping[str, str],
    *,
    k: int | None = ARTICLE_K,
    budget_tokens: int = ARTICLE_BUDGET_TOKENS,
    overflow: str = "stop",
    window_tokens: int | None = None,
    slices: int = 1,
) -> list[Passage]:
    """The full text of the top-``k`` distinct parent articles of ranked ``chunks``.

    Articles keep the rank of their best chunk and appear once. They are added whole until ``k``
    are taken (``k=None``: no limit, budget packing) or the next one would exceed
    ``budget_tokens``. ``overflow`` is what happens to that article: ``stop`` drops it and all
    later ones, ``skip`` drops only it and keeps trying the later ones. ``window_tokens`` (long
    documents such as papers) replaces an article longer than it by the window of that size
    around its best chunk before packing. The first article is always kept; if it alone exceeds
    the budget, the budget-sized window around its best chunk is kept and packing ends. A window
    starts at the beginning when the chunk is not found in the text. ``slices`` > 1 splits a
    window into up to that many smaller ones around the article's first chunks in ``chunks``. The
    summed ``estimate_tokens`` of the result never exceeds ``budget_tokens``.
    """
    if (k is not None and k < 1) or budget_tokens < 1:
        raise ValueError("k and budget_tokens must be positive")
    if (window_tokens is not None and window_tokens < 1) or slices < 1:
        raise ValueError("window_tokens and slices must be positive")
    if overflow not in OVERFLOW_POLICIES:
        raise ValueError(f"overflow must be one of {OVERFLOW_POLICIES}, got {overflow!r}")
    out: list[Passage] = []
    used = 0
    for article_id in dict.fromkeys(c.article_id for c in chunks):
        if len(out) == k:
            break
        text = articles[article_id].strip()
        bests = [c.text.strip() for c in chunks if c.article_id == article_id][:slices]
        if window_tokens is not None and estimate_tokens(text) > window_tokens:
            text = _window(text, bests, window_tokens * CHARS_PER_TOKEN)
        cost = estimate_tokens(text)
        if not out and cost > budget_tokens:
            # long documents (papers, reports): keep the part that matched, not the first pages
            out.append(Passage(article_id, _window(text, bests, budget_tokens * CHARS_PER_TOKEN)))
            break
        if used + cost > budget_tokens:
            if overflow == "stop":
                break
            continue
        out.append(Passage(article_id, text))
        used += cost
    return out


def pack_articles(
    chunks: Sequence[Passage],
    articles: Mapping[str, str],
    *,
    budget_tokens: int = ARTICLE_BUDGET_TOKENS,
    overflow: str = "stop",
    window_tokens: int | None = None,
    slices: int = 1,
) -> list[Passage]:
    """Budget packing: distinct parent articles in rank order until the budget is full."""
    return parent_articles(
        chunks,
        articles,
        k=None,
        budget_tokens=budget_tokens,
        overflow=overflow,
        window_tokens=window_tokens,
        slices=slices,
    )
