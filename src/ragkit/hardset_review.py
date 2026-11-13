"""Independent LLM review of a question set against the whole EU corpus (stdlib only).

For every question the reviewer model sees the entire corpus (cached prompt prefix) and the
question with its claimed gold article(s) and answer. It reports whether the gold article
supports the answer, which other articles also answer, and whether the question is ambiguous.
The reviewer is never given the retrievers' output or the drafter's list of decoys.

Every response is cached on disk under the SHA-256 of the request, so a re-run costs nothing
and the tests never need a key. ``MAX_CALLS`` bounds the live calls ever made into one cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ragkit.eurlex import strip_metadata_comment
from ragkit.llm import post_message

MODEL = "claude-sonnet-5-5"
MAX_CALLS = 150
MAX_TOKENS = 1500

INSTRUCTIONS = """You are an independent reviewer of a retrieval benchmark over a corpus of EU \
electricity-law articles. The complete corpus follows, one article per block headed by its id.

For each item you get a question, the article id(s) claimed as gold, and a reference answer. \
Judge using ONLY the corpus text. Reply with one JSON object and nothing else:
{"gold_answers": "yes" | "partial" | "no",
 "gold_reason": "one or two sentences naming the paragraph that supports the answer or what is \
missing or wrong in the reference answer",
 "other_articles": [{"doc_id": "<id>", "answers": "fully" | "partially", "why": "short"}],
 "ambiguity": "none" | "minor" | "major",
 "ambiguity_reason": "short",
 "suggested_fix": "how to reword the question or change the gold ids, or empty"}

Rules: "gold_answers" is "yes" only if the gold article(s) contain everything the reference \
answer states. "other_articles" lists every article OTHER than the gold ones that independently \
answers the question as asked (a reader who found only that article could give the answer); \
ignore articles that merely mention the topic. Search the whole corpus before answering. \
"ambiguity" is "major" if a reasonable reader could read the question so that a different \
answer or a different article is correct."""


def corpus_block(documents: Sequence[tuple[str, str]]) -> str:
    """The corpus as one text: ``=== <id> ===`` followed by the article, in the given order."""
    return "\n\n".join(f"=== {doc_id} ===\n{text.strip()}" for doc_id, text in documents)


def build_request(corpus: str, record: Mapping[str, Any]) -> dict[str, Any]:
    """The Messages API payload for one question; the corpus block is the cached prefix."""
    item = (
        f"Question: {record['q']}\n"
        f"Claimed gold article id(s): {', '.join(record['relevant_doc_ids'])}\n"
        f"Reference answer: {record['answer']}"
    )
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": [
            {"type": "text", "text": INSTRUCTIONS},
            {"type": "text", "text": corpus, "cache_control": {"type": "ephemeral"}},
        ],
        "messages": [{"role": "user", "content": item}],
    }


def request_key(payload: Mapping[str, Any]) -> str:
    """Stable cache key of a request payload."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def parse_verdict(text: str) -> dict[str, Any]:
    """The JSON object in a model reply (tolerates a fenced or prefixed reply)."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    verdict = json.loads(text[start : end + 1])
    if not isinstance(verdict, dict) or "gold_answers" not in verdict:
        raise ValueError(f"reply lacks gold_answers: {text[:200]!r}")
    return verdict


Transport = Callable[[dict[str, Any], str], dict[str, Any]]


def review_question(
    corpus: str,
    record: Mapping[str, Any],
    cache_dir: Path,
    transport: Transport | None = None,
    api_key: str = "",
) -> tuple[dict[str, Any], bool]:
    """Verdict for one question and whether a live call was made (False = cache hit)."""
    payload = build_request(corpus, record)
    path = cache_dir / f"{request_key(payload)}.json"
    if path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
        return dict(cached["verdict"]), False
    live_calls = len(list(cache_dir.glob("*.json")))
    if live_calls >= MAX_CALLS:
        raise RuntimeError(f"call budget of {MAX_CALLS} reached; cache holds {live_calls}")
    send = transport or post_message
    response = send(payload, api_key)
    text = "".join(
        str(block.get("text", "")) for block in response.get("content", []) if block.get("text")
    )
    verdict = parse_verdict(text)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stored = {
        "model": MODEL,
        "question": record["q"],
        "gold": list(record["relevant_doc_ids"]),
        "usage": response.get("usage", {}),
        "reply": text,
        "verdict": verdict,
    }
    path.write_text(json.dumps(stored, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return verdict, True


def summarise(verdicts: Sequence[Mapping[str, Any]]) -> str:
    """One markdown row per question: gold support, other answering articles, ambiguity."""
    lines = [
        "| # | gold answers | other articles that also answer | ambiguity | note |",
        "|---|---|---|---|---|",
    ]
    for i, v in enumerate(verdicts):
        others = ", ".join(f"{o['doc_id']} ({o['answers']})" for o in v.get("other_articles", []))
        note = str(v.get("suggested_fix") or v.get("gold_reason", "")).replace("|", "/")
        lines.append(
            f"| {i} | {v['gold_answers']} | {others or '-'} | {v.get('ambiguity', '?')} | "
            f"{note[:160]} |"
        )
    return "\n".join(lines)


def total_usage(cache_dir: Path) -> dict[str, int]:
    """Summed token usage over every cached response."""
    totals: dict[str, int] = {}
    for path in cache_dir.glob("*.json"):
        usage = json.loads(path.read_text(encoding="utf-8")).get("usage", {})
        for key, value in usage.items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent LLM review of a QA jsonl file.")
    parser.add_argument("qa_file", type=Path)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Write verdicts as jsonl here.")
    args = parser.parse_args(argv)

    corpus = corpus_block(
        [
            (p.stem, strip_metadata_comment(p.read_text(encoding="utf-8")))
            for p in sorted(args.corpus_dir.glob("*.md"))
        ]
    )
    records = [
        json.loads(line) for line in args.qa_file.read_text(encoding="utf-8").splitlines() if line
    ]
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    verdicts: list[dict[str, Any]] = []
    live = 0
    for record in records:
        verdict, was_live = review_question(corpus, record, args.cache_dir, api_key=api_key)
        live += was_live
        verdicts.append(verdict)
    with args.out.open("w", encoding="utf-8") as fh:
        for record, verdict in zip(records, verdicts, strict=True):
            fh.write(json.dumps({"q": record["q"], "verdict": verdict}, ensure_ascii=False) + "\n")
    print(summarise(verdicts))
    cached = len(list(args.cache_dir.glob("*.json")))
    print(f"\nlive calls this run: {live}; cached responses: {cached}")
    print(f"token usage over the cache: {total_usage(args.cache_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
