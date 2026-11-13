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


def parent_articles(
    chunks: Sequence[Passage],
    articles: Mapping[str, str],
    *,
    k: int | None = ARTICLE_K,
    budget_tokens: int = ARTICLE_BUDGET_TOKENS,
    overflow: str = "stop",
) -> list[Passage]:
    """The full text of the top-``k`` distinct parent articles of ranked ``chunks``.

    Articles keep the rank of their best chunk and appear once. They are added whole until ``k``
    are taken (``k=None``: no limit, budget packing) or the next one would exceed
    ``budget_tokens``. ``overflow`` is what happens to that article: ``stop`` drops it and all
    later ones, ``skip`` drops only it and keeps trying the later ones. The first article is
    always kept, cut at the budget only if it alone exceeds it (packing then ends). The summed
    ``estimate_tokens`` of the result never exceeds ``budget_tokens``.
    """
    if (k is not None and k < 1) or budget_tokens < 1:
        raise ValueError("k and budget_tokens must be positive")
    if overflow not in OVERFLOW_POLICIES:
        raise ValueError(f"overflow must be one of {OVERFLOW_POLICIES}, got {overflow!r}")
    out: list[Passage] = []
    used = 0
    for article_id in dict.fromkeys(c.article_id for c in chunks):
        if len(out) == k:
            break
        text = articles[article_id].strip()
        cost = estimate_tokens(text)
        if not out and cost > budget_tokens:
            out.append(Passage(article_id, text[: budget_tokens * CHARS_PER_TOKEN]))
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
) -> list[Passage]:
    """Budget packing: distinct parent articles in rank order until the budget is full."""
    return parent_articles(chunks, articles, k=None, budget_tokens=budget_tokens, overflow=overflow)
