"""LLM answers with citations and a sufficient-context check (``docs/generation_protocol.md``).

Five kinds of call, all through :class:`ragkit.llm.CachedMessages`:

- ``GEN``: answer from retrieved chunks with article-id citations and a ``context_supports`` flag;
- ``SUFF``: an autorater in the style of Joren et al. 2024 (arXiv:2411.06037): is the context
  sufficient to answer the question? Sees only the question and the chunks;
- ``FC``: the same answer schema with the whole corpus as a cached system block;
- ``CORRECT`` and ``CITE``: LLM judges of answer correctness against a reference and of whether a
  cited article supports the answer.

The prompts are frozen with the protocol and must not be edited after the scored run.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ragkit.llm import CachedMessages

GEN_MAX_TOKENS = 1200
JUDGE_MAX_TOKENS = 800

GEN_INSTRUCTIONS = """You answer questions about EU electricity law using ONLY the context \
passages provided. Each passage is headed by the id of the article it comes from, in square \
brackets.

Reply with one JSON object and nothing else:
{"answer": "<answer>", "citations": ["<article id>", ...], "context_supports": true | false}

Rules:
- "answer": your best answer to the question as asked, in at most 120 words, using only the \
passages. Do not use outside knowledge. If the passages do not contain what is needed, say what \
they do state and that they do not answer the question.
- "citations": the ids of the articles whose passages you used. Use only ids that appear in the \
context.
- "context_supports": true only if the passages contain what is needed to answer the question \
definitively, false otherwise (also false if you could only guess, or if the passages merely \
concern the same topic)."""

SUFF_INSTRUCTIONS = """You are an expert evaluator. You get a QUESTION and a CONTEXT made of \
passages from EU electricity-law articles. Decide whether the CONTEXT is SUFFICIENT: whether the \
question can be answered definitively using only the CONTEXT, without outside knowledge.

Rules:
- Sufficient means the CONTEXT states what the question asks for. Passages that only concern the \
same topic, or from which the answer could merely be guessed, are NOT sufficient.
- If the question asks for a specific value, date, name or rule, the CONTEXT must state it.
- If the question refers to a specific act or article that the CONTEXT does not contain, it is \
NOT sufficient.
- Work step by step: list what the question needs, check each need against the CONTEXT, then decide.

Reply with one JSON object and nothing else:
{"reasoning": "<short step-by-step check>", "sufficient": true | false, \
"confidence": <integer 0-100, your probability that the CONTEXT is sufficient>}"""

FC_INSTRUCTIONS = """You answer questions about EU electricity law using ONLY the corpus below, \
which contains the complete text of 94 articles, each headed by "=== <article id> ===".

Reply with one JSON object and nothing else:
{"answer": "<answer>", "citations": ["<article id>", ...], "context_supports": true | false}

Rules:
- "answer": your best answer to the question as asked, in at most 120 words, using only the \
corpus. Do not use outside knowledge. If the corpus does not contain what is needed, say so and \
state what it does contain on the topic.
- "citations": the ids of the articles whose text you used. Use only ids that appear in the corpus.
- "context_supports": true only if the corpus contains what is needed to answer the question \
definitively, false otherwise (also false if you could only guess, or if the corpus merely \
concerns the same topic)."""

CORRECT_INSTRUCTIONS = """You grade a candidate answer against a reference answer for the same \
question. Reply with one JSON object and nothing else:
{"verdict": "yes" | "partial" | "no", "reason": "<one sentence>"}

- "yes": the candidate states the key facts of the reference answer, with nothing that \
contradicts it. Extra correct detail is fine; different wording is fine.
- "partial": the candidate states some but not all key facts, or is correct but too vague.
- "no": the candidate misses the key facts, contradicts the reference, or says it cannot answer."""

CITE_INSTRUCTIONS = """You check a citation. You get a question, an answer and the full text of \
one article that the answer cites. Reply with one JSON object and nothing else:
{"verdict": "yes" | "partial" | "no", "reason": "<one sentence>"}

- "yes": the article states (or directly entails) a substantive claim the answer makes, and \
nothing in the article contradicts the answer's claims about it.
- "partial": the article supports only a minor part of the answer or only loosely.
- "no": the article does not contain support for the answer's claims, or contradicts them."""


@dataclass(frozen=True)
class Passage:
    """A retrieved chunk labelled with the article it comes from."""

    article_id: str
    text: str


@dataclass(frozen=True)
class GenAnswer:
    """An answer with its citations and the model's own ``context_supports`` flag."""

    answer: str
    citations: tuple[str, ...]
    supports: bool


@dataclass(frozen=True)
class Sufficiency:
    """The autorater's verdict and its stated probability that the context is sufficient."""

    sufficient: bool
    confidence: float
    reasoning: str


@dataclass(frozen=True)
class Verdict:
    """A judge's ``yes`` / ``partial`` / ``no`` with its reason."""

    verdict: str
    reason: str


def context_block(passages: Sequence[Passage]) -> str:
    """Passages as ``[article id]`` followed by the chunk text, in retrieval order."""
    return "\n\n".join(f"[{p.article_id}]\n{p.text.strip()}" for p in passages)


def corpus_block(documents: Sequence[tuple[str, str]]) -> str:
    """The whole corpus as ``=== <id> ===`` blocks (the cached full-context prefix)."""
    return "\n\n".join(f"=== {doc_id} ===\n{text.strip()}" for doc_id, text in documents)


def parse_json_object(text: str) -> dict[str, Any]:
    """The first JSON object in a reply (tolerates a fenced or prefixed reply)."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError(f"reply is not a JSON object: {text[:200]!r}")
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


class LLMAnswerer:
    """The calls of the protocol over one cached client."""

    def __init__(self, client: CachedMessages) -> None:
        self.client = client

    def _json(
        self,
        kind: str,
        system: Sequence[dict[str, Any]],
        user: str,
        max_tokens: int,
        required: Sequence[str],
    ) -> dict[str, Any]:
        """One call; a reply that is not the expected JSON is retried with a reminder.

        Attempt 2 repeats the request with a JSON reminder; attempt 3 also doubles ``max_tokens``
        (a reply cut off at the limit is the usual cause). Earlier attempts keep their requests,
        and so their cache keys, when a later one is added.
        """
        reminder = "\n\nReply with only the JSON object, nothing else."
        attempts = (
            (user, max_tokens),
            (user + reminder, max_tokens),
            (user + reminder, 2 * max_tokens),
        )
        for attempt, limit in attempts:
            reply = self.client.complete(kind, system, attempt, limit)
            try:
                parsed = parse_json_object(reply.text)
            except ValueError:
                continue
            if all(key in parsed for key in required):
                return parsed
        raise ValueError(f"{kind}: no valid JSON reply after {len(attempts)} attempts")

    @staticmethod
    def _gen(parsed: dict[str, Any]) -> GenAnswer:
        cites = parsed.get("citations") or []
        return GenAnswer(
            answer=str(parsed.get("answer", "")).strip(),
            citations=tuple(dict.fromkeys(str(c) for c in cites)),
            supports=_as_bool(parsed.get("context_supports")),
        )

    def generate(self, question: str, passages: Sequence[Passage]) -> GenAnswer:
        """``GEN``: best-effort answer from the retrieved chunks."""
        user = f"Context passages:\n\n{context_block(passages)}\n\nQuestion: {question}"
        parsed = self._json(
            "GEN",
            [{"type": "text", "text": GEN_INSTRUCTIONS}],
            user,
            GEN_MAX_TOKENS,
            ("answer", "citations", "context_supports"),
        )
        return self._gen(parsed)

    def sufficiency(self, question: str, passages: Sequence[Passage]) -> Sufficiency:
        """``SUFF``: is the retrieved context sufficient (question and chunks only)?"""
        user = f"QUESTION: {question}\n\nCONTEXT:\n\n{context_block(passages)}"
        parsed = self._json(
            "SUFF",
            [{"type": "text", "text": SUFF_INSTRUCTIONS}],
            user,
            JUDGE_MAX_TOKENS,
            ("sufficient", "confidence"),
        )
        confidence = float(parsed["confidence"])
        return Sufficiency(
            sufficient=_as_bool(parsed["sufficient"]),
            confidence=max(0.0, min(100.0, confidence)),
            reasoning=str(parsed.get("reasoning", "")),
        )

    def full_context(self, question: str, corpus: str) -> GenAnswer:
        """``FC``: the same schema with the whole corpus as a cached system block."""
        system: list[dict[str, Any]] = [
            {"type": "text", "text": FC_INSTRUCTIONS},
            {"type": "text", "text": corpus, "cache_control": {"type": "ephemeral"}},
        ]
        parsed = self._json(
            "FC",
            system,
            f"Question: {question}",
            GEN_MAX_TOKENS,
            ("answer", "citations", "context_supports"),
        )
        return self._gen(parsed)

    def judge_correct(self, question: str, reference: str, candidate: str) -> Verdict:
        """``CORRECT``: LLM-judged correctness of a candidate against the reference answer."""
        user = f"Question: {question}\n\nReference answer: {reference}\n\nCandidate: {candidate}"
        parsed = self._json(
            "CORRECT",
            [{"type": "text", "text": CORRECT_INSTRUCTIONS}],
            user,
            JUDGE_MAX_TOKENS,
            ("verdict",),
        )
        return Verdict(str(parsed["verdict"]).strip().lower(), str(parsed.get("reason", "")))

    def judge_citation(self, question: str, answer: str, article: str) -> Verdict:
        """``CITE``: LLM-judged support of the answer by one cited article's full text."""
        user = f"Question: {question}\n\nAnswer: {answer}\n\nCited article:\n\n{article.strip()}"
        parsed = self._json(
            "CITE",
            [{"type": "text", "text": CITE_INSTRUCTIONS}],
            user,
            JUDGE_MAX_TOKENS,
            ("verdict",),
        )
        return Verdict(str(parsed["verdict"]).strip().lower(), str(parsed.get("reason", "")))
