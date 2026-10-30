"""Parent-document (small-to-big) context: retrieve on chunks, answer from whole articles.

Protocol and frozen parameters: ``docs/generation_protocol.md`` (Phase L). The chunks that rank
best are only used to pick articles; ``GEN`` then sees the full text of the distinct parent
articles, in rank order, within a token budget.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from ragkit.generation import Passage

ARTICLE_K = 3
ARTICLE_BUDGET_TOKENS = 12_000
CHARS_PER_TOKEN = 4  # 337 kB of corpus = about 85 k tokens, the K-R2 full-context figure


def estimate_tokens(text: str) -> int:
    """Token estimate used for the budget: characters / 4, rounded up."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def parent_articles(
    chunks: Sequence[Passage],
    articles: Mapping[str, str],
    *,
    k: int = ARTICLE_K,
    budget_tokens: int = ARTICLE_BUDGET_TOKENS,
) -> list[Passage]:
    """The full text of the top-``k`` distinct parent articles of ranked ``chunks``.

    Articles keep the rank of their best chunk and appear once. They are added whole until ``k``
    are taken or the next one would exceed ``budget_tokens`` (it and all later ones are then
    dropped). The first article is always kept, cut at the budget only if it alone exceeds it.
    """
    if k < 1 or budget_tokens < 1:
        raise ValueError("k and budget_tokens must be positive")
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
            break
        out.append(Passage(article_id, text))
        used += cost
    return out
