"""Mechanical keyword-free rule of the no-keyword EU question set.

Protocol: ``docs/ctxprefix_nokw_protocol.md``.

A question is *keyword-free* when it shares at most ``MAX_OVERLAP`` content word with the
discriminating vocabulary of its gold article. That vocabulary is the article's ``TOP_K``
highest TF-IDF terms, computed over the 94-article EU corpus with ragkit's own analyzer
(``ragkit.analysis.lucene_english``: lowercase alphanumeric tokens, Lucene stop words removed,
Porter stems, so an inflected form of a top term counts as an overlap too). It also may not name
an act, an article number or the gold article's heading.

No retriever is involved: the check reads one gold article's term statistics and the question text.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ragkit.analysis import lucene_english
from ragkit.bm25 import tokenize
from ragkit.ctxprefix import heading_of
from ragkit.eurlex import strip_metadata_comment

# Frozen by the protocol, committed before any question was written.
TOP_K = 20
MAX_OVERLAP = 1

_ACT_NUMBER = re.compile(r"\d{4}/\d{1,4}")
_ARTICLE_NUMBER = re.compile(r"\b(?:article|articles|art)\b\.?\s*\d", re.IGNORECASE)
_LEGAL_NAME_WORDS = frozenset({"regulation", "regulations", "directive", "directives", "annex"})
_HEADING_TITLE = re.compile(r"Article \d+[A-Za-z]?: (?P<title>.+)$")


class TermIndex:
    """Per-article stemmed term counts and corpus document frequencies (articles are documents)."""

    def __init__(self, documents: Sequence[tuple[str, str]]) -> None:
        self.texts = dict(documents)
        self.counts = {doc_id: Counter(lucene_english(text)) for doc_id, text in documents}
        self.df: Counter[str] = Counter(t for c in self.counts.values() for t in c)
        self.n = len(self.counts)

    def top_terms(self, doc_id: str, k: int = TOP_K) -> list[str]:
        """The ``k`` highest ``tf * ln(N / df)`` stems of one article (ties: alphabetical)."""
        weights = {t: n * math.log(self.n / self.df[t]) for t, n in self.counts[doc_id].items()}
        ranked = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
        return [t for t, w in ranked[:k] if w > 0.0]

    def heading_title(self, doc_id: str) -> str:
        """The article's heading without the act title and ``Article N:``; ``""`` when absent."""
        match = _HEADING_TITLE.search(heading_of(self.texts[doc_id]))
        return match.group("title") if match else ""


def overlap(index: TermIndex, question: str, gold: Sequence[str]) -> tuple[int, list[str]]:
    """Largest overlap with any gold article's top terms and the shared stems of that article."""
    stems = set(lucene_english(question))
    best: list[str] = []
    for doc_id in gold:
        shared = sorted(stems & set(index.top_terms(doc_id)))
        if len(shared) > len(best):
            best = shared
    return len(best), best


def naming_violations(index: TermIndex, question: str, gold: Sequence[str]) -> list[str]:
    """Reasons the question names an act, an article number or a gold article's heading."""
    problems: list[str] = []
    if _ACT_NUMBER.search(question):
        problems.append("names an act number")
    if _ARTICLE_NUMBER.search(question):
        problems.append("names an article number")
    named = _LEGAL_NAME_WORDS & set(tokenize(question))
    if named:
        problems.append(f"uses {sorted(named)}")
    padded = " " + " ".join(tokenize(question)) + " "
    for doc_id in gold:
        title = " ".join(tokenize(index.heading_title(doc_id)))
        if title and f" {title} " in padded:
            problems.append(f"contains the heading of {doc_id}: {title!r}")
    return problems


def violations(index: TermIndex, record: Mapping[str, Any]) -> list[str]:
    """Every way one question record breaks the rule; an empty list means it passes."""
    gold = list(record["relevant_doc_ids"])
    count, shared = overlap(index, str(record["q"]), gold)
    problems = naming_violations(index, str(record["q"]), gold)
    if count > MAX_OVERLAP:
        problems.append(f"overlap {count} > {MAX_OVERLAP} with the top-{TOP_K} terms: {shared}")
    return problems


def load_index(corpus_dir: Path) -> TermIndex:
    """The term index of a corpus directory (metadata comment stripped, as in retrieval)."""
    return TermIndex(
        [
            (p.stem, strip_metadata_comment(p.read_text(encoding="utf-8")))
            for p in sorted(corpus_dir.glob("*.md"))
        ]
    )


def render_rule_status(index: TermIndex, records: Sequence[Mapping[str, Any]]) -> str:
    """One line for the run report: questions passing the frozen rule, and the worst overlap."""
    worst = max(overlap(index, str(r["q"]), list(r["relevant_doc_ids"]))[0] for r in records)
    passing = sum(1 for r in records if not violations(index, r))
    return (
        f"Keyword-free rule (docs/ctxprefix_nokw_protocol.md): top-{TOP_K} TF-IDF stems of the "
        f"gold article, overlap <= {MAX_OVERLAP}: {passing} / {len(records)} questions pass, "
        f"largest overlap {worst}.\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Print the overlap of every question of a jsonl file; exit 1 when any breaks the rule."""
    parser = argparse.ArgumentParser(description="Keyword-free rule check of a QA jsonl file.")
    parser.add_argument("qa_file", type=Path)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    index = load_index(args.corpus_dir)
    failed = 0
    for i, line in enumerate(args.qa_file.read_text(encoding="utf-8").splitlines()):
        if not line:
            continue
        record = json.loads(line)
        count, shared = overlap(index, record["q"], record["relevant_doc_ids"])
        problems = violations(index, record)
        failed += bool(problems)
        status = "FAIL" if problems else "ok  "
        print(f"{status} {i:2d} overlap={count} {shared} {problems if problems else ''}")
    print(f"\n{failed} question(s) break the rule (top-{TOP_K}, overlap <= {MAX_OVERLAP})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
