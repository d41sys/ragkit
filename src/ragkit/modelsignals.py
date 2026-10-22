"""Model-based abstention signals: read the retrieved passages against the question.

Two signals, fixed in ``docs/model_signals_protocol.md`` before any score was computed. Both look
at the same top chunks as ``query_support`` and aggregate by the maximum; higher = more likely
answerable.

- ``ce_top``: cross-encoder relevance logit of ``(question, chunk)``.
- ``qa_answerable``: SQuAD2 reader, ``best non-null span score - null score`` per chunk.

The model boundary is the ``PassageScorer`` protocol, so offline tests plug in a fake. The real
scorers need the ``[st]`` extra (sentence-transformers, which brings transformers and torch) and
are imported and loaded lazily (a fully cached run needs neither). Scores are cached by
(scorer id, question, passage) in a JSON file.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

MODEL_SIGNALS: dict[str, str] = {
    "ce_top": "max cross-encoder (ms-marco-MiniLM-L-6-v2) relevance logit over the top-5 chunks",
    "qa_answerable": "max SQuAD2 reader (roberta-base-squad2) best-span minus no-answer score",
}

# Pinned Hugging Face commits (read from the Hub API on 2026-10-23) and licences.
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Apache-2.0
CE_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
QA_MODEL = "deepset/roberta-base-squad2"  # CC-BY-4.0
QA_REVISION = "adc3b06f79f797d1c575d5479d6f5efe54a9e3b4"

MAX_ANSWER_TOKENS = 30
DEFAULT_CACHE = Path(".ragkit") / "model_scores.json"


class PassageScorer(Protocol):
    """Scores each passage against a question; higher = more relevant / answerable."""

    scorer_id: str

    def score(self, question: str, passages: Sequence[str]) -> list[float]: ...


@dataclass
class ScoreCache:
    """JSON-backed ``key -> score`` store; ``path=None`` keeps it in memory only."""

    path: Path | None = None
    data: dict[str, float] = field(default_factory=dict)
    dirty: bool = False

    def __post_init__(self) -> None:
        if self.path is not None and self.path.exists():
            self.data = {str(k): float(v) for k, v in json.loads(self.path.read_text()).items()}

    def save(self) -> None:
        if self.path is None or not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, sort_keys=True))
        self.dirty = False


def _key(scorer_id: str, question: str, passage: str) -> str:
    digest = hashlib.sha1(f"{question}\x00{passage}".encode(), usedforsecurity=False)
    return f"{scorer_id}|{digest.hexdigest()}"


class CachedScorer:
    """Wraps a scorer: only uncached (question, passage) pairs reach the model."""

    def __init__(self, inner: PassageScorer, cache: ScoreCache) -> None:
        self.inner = inner
        self.cache = cache
        self.scorer_id = inner.scorer_id
        self.hits = 0
        self.misses = 0
        self.seconds = 0.0  # wall time spent inside the model, cache lookups excluded

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        keys = [_key(self.scorer_id, question, p) for p in passages]
        todo = [i for i, k in enumerate(keys) if k not in self.cache.data]
        self.hits += len(keys) - len(todo)
        if todo:
            start = time.perf_counter()
            fresh = self.inner.score(question, [passages[i] for i in todo])
            self.seconds += time.perf_counter() - start
            if len(fresh) != len(todo):
                raise ValueError(f"scorer returned {len(fresh)} scores for {len(todo)} passages")
            for i, value in zip(todo, fresh, strict=True):
                self.cache.data[keys[i]] = float(value)
            self.misses += len(todo)
            self.cache.dirty = True
        return [self.cache.data[k] for k in keys]


@dataclass
class ModelScorers:
    """The two scorers behind ``ce_top`` and ``qa_answerable``."""

    relevance: PassageScorer
    answerability: PassageScorer
    cache: ScoreCache | None = None

    def signals(self, question: str, passages: Sequence[str]) -> dict[str, float]:
        """Both model signals for one question and its retrieved chunk texts."""
        return {
            "ce_top": max(self.relevance.score(question, passages), default=0.0),
            "qa_answerable": max(self.answerability.score(question, passages), default=0.0),
        }

    def stats(self) -> str:
        """One line of pair counts and model wall time, for the run log."""
        parts = []
        for label, s in (("ce", self.relevance), ("qa", self.answerability)):
            if isinstance(s, CachedScorer):
                parts.append(f"{label}: {s.misses} scored in {s.seconds:.1f}s, {s.hits} cached")
        return "; ".join(parts) if parts else "no cache statistics"

    def save(self) -> None:
        if self.cache is not None:
            self.cache.save()


def _missing_extra(err: ImportError) -> ImportError:
    return ImportError(
        "model signals need sentence-transformers (transformers + torch). "
        "Install with: pip install 'ragkit[st]'"
    ).with_traceback(err.__traceback__)


class CrossEncoderScorer:
    """Raw (pre-activation) logit of a cross-encoder relevance model; loaded on first use."""

    def __init__(
        self, name: str = CE_MODEL, revision: str = CE_REVISION, device: str = "cpu"
    ) -> None:
        self.name, self.revision, self.device = name, revision, device
        self.scorer_id = f"{name}@{revision}"
        self._model: Any = None
        self._identity: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                import torch
                from sentence_transformers import CrossEncoder
            except ImportError as err:
                raise _missing_extra(err) from err
            self._identity = torch.nn.Identity()
            self._model = CrossEncoder(self.name, revision=self.revision, device=self.device)
        return self._model

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        model = self._load()
        pairs = [(question, p) for p in passages]
        out = model.predict(
            pairs, activation_fn=self._identity, show_progress_bar=False, batch_size=len(pairs)
        )
        return [float(x) for x in np.asarray(out, dtype=np.float64).reshape(-1)]


def span_minus_null(
    start: np.ndarray[Any, Any],
    end: np.ndarray[Any, Any],
    context: np.ndarray[Any, Any],
    max_answer_tokens: int = MAX_ANSWER_TOKENS,
) -> float:
    """Best non-null span score minus the null score for one (question, passage) example.

    ``start`` / ``end`` are the reader's logits per token, ``context`` the boolean mask of the
    passage tokens, whose position 0 (the ``<s>`` token) carries the null score. A span is
    ``start[i] + end[j]`` for context tokens ``i <= j < i + max_answer_tokens``.
    """
    idx = np.flatnonzero(context)
    if idx.size == 0:
        return 0.0
    null = float(start[0] + end[0])
    best = -np.inf
    for i in idx:
        stop = idx[(idx >= i) & (idx < i + max_answer_tokens)]
        best = max(best, float(start[i] + end[stop].max()))
    return best - null


class SquadNullScorer:
    """SQuAD2 extractive reader: how much does a real answer span beat "no answer"."""

    def __init__(
        self, name: str = QA_MODEL, revision: str = QA_REVISION, device: str = "cpu"
    ) -> None:
        self.name, self.revision, self.device = name, revision, device
        self.scorer_id = f"{name}@{revision}"
        self._loaded: tuple[Any, Any, Any] | None = None

    def _load(self) -> tuple[Any, Any, Any]:
        if self._loaded is None:
            try:
                import torch
                from transformers import AutoModelForQuestionAnswering, AutoTokenizer
            except ImportError as err:
                raise _missing_extra(err) from err
            tokenizer = AutoTokenizer.from_pretrained(self.name, revision=self.revision)
            model = AutoModelForQuestionAnswering.from_pretrained(self.name, revision=self.revision)
            self._loaded = (torch, tokenizer, model.to(self.device).eval())
        return self._loaded

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        torch, tokenizer, model = self._load()
        enc = tokenizer(
            [question] * len(passages),
            list(passages),
            truncation="only_second",
            max_length=512,
            padding=True,
            return_tensors="pt",
        )
        batch = {
            "input_ids": enc["input_ids"].to(self.device),
            "attention_mask": enc["attention_mask"].to(self.device),
        }
        with torch.no_grad():
            out = model(**batch)
        starts = out.start_logits.float().cpu().numpy().astype(np.float64)
        ends = out.end_logits.float().cpu().numpy().astype(np.float64)
        scores = []
        for i in range(len(passages)):
            context = np.array([s == 1 for s in enc.sequence_ids(i)], dtype=np.bool_)
            scores.append(span_minus_null(starts[i], ends[i], context))
        return scores


def load_model_scorers(
    cache_path: Path | None = DEFAULT_CACHE, device: str = "cpu"
) -> ModelScorers:
    """The two real scorers, behind a persistent score cache."""
    cache = ScoreCache(cache_path)
    return ModelScorers(
        relevance=CachedScorer(CrossEncoderScorer(device=device), cache),
        answerability=CachedScorer(SquadNullScorer(device=device), cache),
        cache=cache,
    )
