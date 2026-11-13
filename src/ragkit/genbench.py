"""The K-R2 experiment: LLM answers, sufficient-context gates and a full-context baseline.

Protocol, decision rule and predictions: ``docs/generation_protocol.md``. Calls go through
``ragkit.generation.LLMAnswerer`` (cached JSONL), the existing SQuAD2 reader scores come from a
committed score cache, and every table is computed offline from those two caches.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ragkit.abstention import (
    FloatArray,
    accept_counts,
    auroc,
    auroc_ci,
    cross_validated_accept,
    nested_cv_logistic,
)
from ragkit.eval import paired_bootstrap_ci, wilson_interval
from ragkit.generation import (
    GenAnswer,
    LLMAnswerer,
    Sufficiency,
    Verdict,
    corpus_block,
)
from ragkit.llm import CachedMessages, CacheMiss, usage_by_kind, usage_totals
from ragkit.modelsignals import (
    CE_MODEL,
    CE_REVISION,
    QA_MODEL,
    QA_REVISION,
    CachedScorer,
    ModelScorers,
    ScoreCache,
    _key,
)
from ragkit.parent import Passage
from ragkit.pipeline import RAGPipeline

CV_FOLDS = 5
CV_SEED = 0
SYSTEMS_WITH_READER = ("reader", "self", "suff", "strict", "lenient", "stack", "fullctx")
SYSTEMS_NO_READER = ("self", "suff", "fullctx")


@dataclass(frozen=True)
class Item:
    """One benchmark question: answerable (with reference and gold articles) or not."""

    question: str
    answerable: bool
    reference: str = ""
    relevant: tuple[str, ...] = ()
    kind: str = ""  # miss_type of an unanswerable question


@dataclass
class Record:
    """Everything the API returned for one question."""

    item: Item
    passages: list[Passage]
    gen: GenAnswer
    suff: Sufficiency
    fc: GenAnswer | None = None
    correct_gen: Verdict | None = None
    correct_fc: Verdict | None = None
    cites_gen: dict[str, str] | None = None
    cites_fc: dict[str, str] | None = None

    @property
    def retrieval_miss(self) -> bool:
        """An answerable question whose gold article is not among the retrieved ones."""
        return self.item.answerable and not set(self.item.relevant) & {
            p.article_id for p in self.passages
        }


def items_from_records(
    answerable: Sequence[Mapping[str, Any]], unanswerable: Sequence[Mapping[str, Any]]
) -> list[Item]:
    """Answerable questions first, then the unanswerable ones (the abstention-table order)."""
    items = [
        Item(
            str(r["q"]),
            True,
            reference=str(r.get("answer", "")),
            relevant=tuple(str(d) for d in r["relevant_doc_ids"]),
        )
        for r in answerable
    ]
    items += [
        Item(str(r["q"]), False, kind=str(r.get("miss_type", "original"))) for r in unanswerable
    ]
    return items


def retrieve_passages(
    pipeline: RAGPipeline, documents: Sequence[tuple[str, str]], items: Sequence[Item], top_k: int
) -> list[list[Passage]]:
    """The pipeline's top-``top_k`` chunks of every question, labelled with their article id."""
    pipeline.index_documents(documents)
    return [
        [Passage(c.doc_id, c.text) for c in pipeline.retrieve(item.question, top_k=top_k)]
        for item in items
    ]


# ---------------------------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------------------------


def judge_citations(
    answerer: LLMAnswerer,
    item: Item,
    answer: GenAnswer,
    allowed: set[str],
    articles: Mapping[str, str],
) -> dict[str, str]:
    """``article id -> yes / partial / no / invalid`` (invalid: not retrieved or not in corpus)."""
    out: dict[str, str] = {}
    for cited in answer.citations:
        if cited not in allowed or cited not in articles:
            out[cited] = "invalid"
        else:
            out[cited] = answerer.judge_citation(
                item.question, answer.answer, articles[cited]
            ).verdict
    return out


def run_generation(
    items: Sequence[Item],
    passages: Sequence[Sequence[Passage]],
    documents: Sequence[tuple[str, str]],
    answerer: LLMAnswerer,
    *,
    full_context: bool = True,
    workers: int = 6,
) -> list[Record]:
    """GEN, SUFF, FC and the judges for every question (cached; threads over questions)."""
    articles = dict(documents)
    corpus = corpus_block(documents)
    # The first full-context call alone: it writes the prompt cache the others then read.
    fc: list[GenAnswer | None] = [None] * len(items)
    if full_context and items:
        fc[0] = answerer.full_context(items[0].question, corpus)

    def first_stage(i: int) -> tuple[GenAnswer, Sufficiency, GenAnswer | None]:
        item, psg = items[i], list(passages[i])
        gen = answerer.generate(item.question, psg)
        suff = answerer.sufficiency(item.question, psg)
        full = fc[i]
        if full_context and full is None:
            full = answerer.full_context(item.question, corpus)
        return gen, suff, full

    def second_stage(i: int, stage: tuple[GenAnswer, Sufficiency, GenAnswer | None]) -> Record:
        gen, suff, full = stage
        item = items[i]
        rec = Record(item, list(passages[i]), gen, suff, full)
        if item.answerable:
            rec.correct_gen = answerer.judge_correct(item.question, item.reference, gen.answer)
            if full is not None:
                rec.correct_fc = answerer.judge_correct(item.question, item.reference, full.answer)
        rec.cites_gen = judge_citations(
            answerer, item, gen, {p.article_id for p in rec.passages}, articles
        )
        if full is not None:
            rec.cites_fc = judge_citations(answerer, item, full, set(articles), articles)
        return rec

    with ThreadPoolExecutor(max_workers=workers) as pool:
        stages = list(pool.map(first_stage, range(len(items))))
        return list(pool.map(lambda args: second_stage(*args), enumerate(stages)))


# ---------------------------------------------------------------------------------------------
# Reader scores from the committed cache (no model load)
# ---------------------------------------------------------------------------------------------


class _NoModel:
    """A scorer id with no model behind it: a cache miss is an error, not a model load."""

    def __init__(self, scorer_id: str) -> None:
        self.scorer_id = scorer_id

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        raise CacheMiss(f"{self.scorer_id}: score of {question[:60]!r} not in the committed cache")


def offline_reader_scorers(path: Path) -> ModelScorers:
    """The cross-encoder and SQuAD2 reader scorers backed only by a score-cache file."""
    cache = ScoreCache(path)
    return ModelScorers(
        relevance=CachedScorer(_NoModel(f"{CE_MODEL}@{CE_REVISION}"), cache),
        answerability=CachedScorer(_NoModel(f"{QA_MODEL}@{QA_REVISION}"), cache),
        cache=cache,
    )


def subset_score_cache(
    source: Path, destination: Path, items: Sequence[Item], passages: Sequence[Sequence[Passage]]
) -> int:
    """Copy the reader entries of ``source`` that these questions need into ``destination``."""
    cache = ScoreCache(source)
    wanted: dict[str, float] = {}
    for item, psg in zip(items, passages, strict=True):
        for scorer_id in (f"{CE_MODEL}@{CE_REVISION}", f"{QA_MODEL}@{QA_REVISION}"):
            for p in psg:
                key = _key(scorer_id, item.question, p.text)
                wanted[key] = cache.data[key]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(wanted, sort_keys=True))
    return len(wanted)


def reader_scores(
    scorers: ModelScorers, items: Sequence[Item], passages: Sequence[Sequence[Passage]]
) -> FloatArray:
    """``qa_answerable`` (max SQuAD2 reader score) of every question on its chunk texts."""
    return np.asarray(
        [
            max(scorers.answerability.score(item.question, [p.text for p in psg]), default=0.0)
            for item, psg in zip(items, passages, strict=True)
        ],
        dtype=np.float64,
    )


def build_reader_subset(
    destination: Path,
    items: Sequence[Item],
    passages: Sequence[Sequence[Passage]],
    scorers: ModelScorers,
) -> int:
    """Run the real reader on these chunks and write only their ``qa_answerable`` scores.

    ``scorers`` come from ``load_model_scorers`` (its cache file keeps every score); the
    destination holds just the entries these questions need, so it replays offline.
    """
    if scorers.cache is None:
        raise ValueError("scorers need a score cache")
    qa_id = f"{QA_MODEL}@{QA_REVISION}"
    wanted: dict[str, float] = {}
    for item, psg in zip(items, passages, strict=True):
        texts = [p.text for p in psg]
        scorers.answerability.score(item.question, texts)
        for text in texts:
            key = _key(qa_id, item.question, text)
            wanted[key] = scorers.cache.data[key]
    scorers.save()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(wanted, sort_keys=True))
    return len(wanted)


# ---------------------------------------------------------------------------------------------
# Decisions and metrics
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Decisions:
    """Answer / abstain per system and the continuous scores behind the AUROC rows."""

    labels: np.ndarray[Any, np.dtype[np.bool_]]
    answered: dict[str, np.ndarray[Any, np.dtype[np.bool_]]]
    scores: dict[str, FloatArray]


def decide(records: Sequence[Record], reader: FloatArray | None) -> Decisions:
    """The decision systems of the protocol on one set of records."""
    labels = np.array([r.item.answerable for r in records], dtype=np.bool_)
    suff_conf = np.array([r.suff.confidence for r in records], dtype=np.float64)
    answered: dict[str, np.ndarray[Any, np.dtype[np.bool_]]] = {
        "self": np.array([r.gen.supports for r in records], dtype=np.bool_),
        "suff": np.array([r.suff.sufficient for r in records], dtype=np.bool_),
    }
    scores: dict[str, FloatArray] = {"suff_confidence": suff_conf}
    if all(r.fc is not None for r in records):
        answered["fullctx"] = np.array([bool(r.fc and r.fc.supports) for r in records])
    if reader is not None:
        reader_ok = cross_validated_accept(reader, labels, CV_FOLDS, CV_SEED).accepted
        stack = nested_cv_logistic(np.column_stack([reader, suff_conf]), labels, CV_FOLDS, CV_SEED)
        answered["reader"] = reader_ok
        answered["strict"] = answered["suff"] & reader_ok
        answered["lenient"] = answered["suff"] | reader_ok
        answered["stack"] = stack.accepted
        scores["qa_answerable"] = reader
        scores["stack"] = stack.scores
    order = SYSTEMS_WITH_READER if reader is not None else SYSTEMS_NO_READER
    return Decisions(labels, {s: answered[s] for s in order if s in answered}, scores)


def balanced_accuracy(answered: Sequence[bool] | np.ndarray[Any, Any], labels: Any) -> float:
    """(answerable answered rate + unanswerable abstained rate) / 2."""
    ya, na, yu, nu = accept_counts(answered, labels)
    return (ya / na + yu / nu) / 2.0


def _bootstrap_balacc(
    systems: Sequence[Any], labels: Any, n_resamples: int, seed: int
) -> list[np.ndarray[Any, Any]]:
    """Balanced accuracy of each system on shared stratified bootstrap resamples."""
    y = np.asarray(labels, dtype=np.bool_)
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    rng = np.random.default_rng(seed)
    ip = pos[rng.integers(0, len(pos), size=(n_resamples, len(pos)))]
    ineg = neg[rng.integers(0, len(neg), size=(n_resamples, len(neg)))]
    out = []
    for a in systems:
        arr = np.asarray(a, dtype=np.float64)
        out.append((arr[ip].mean(axis=1) + (1.0 - arr[ineg]).mean(axis=1)) / 2.0)
    return out


def balacc_ci(
    answered: Any, labels: Any, *, n_resamples: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Balanced accuracy with a stratified bootstrap 95 % CI."""
    (boot,) = _bootstrap_balacc([answered], labels, n_resamples, seed)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return balanced_accuracy(answered, labels), float(lo), float(hi)


def paired_balacc_ci(
    a: Any, b: Any, labels: Any, *, n_resamples: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Bootstrap CI of the balanced-accuracy difference ``a - b`` on the same resamples."""
    boot_a, boot_b = _bootstrap_balacc([a, b], labels, n_resamples, seed)
    lo, hi = np.quantile(boot_a - boot_b, [0.025, 0.975])
    return balanced_accuracy(a, labels) - balanced_accuracy(b, labels), float(lo), float(hi)


def verdicts_of(records: Sequence[Record], source: str) -> list[str | None]:
    out: list[str | None] = []
    for r in records:
        v = r.correct_fc if source == "fc" else r.correct_gen
        out.append(v.verdict if v is not None else None)
    return out


def correct_flags(records: Sequence[Record], source: str) -> np.ndarray[Any, Any]:
    """Per question: judged ``yes`` (answerable only; False elsewhere)."""
    return np.array([v == "yes" for v in verdicts_of(records, source)], dtype=np.bool_)


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------


def format_rate(hit: int, total: int) -> str:
    lo, hi = wilson_interval(hit, total)
    return f"{hit}/{total} ({hit / max(1, total):.2f}) [{lo:.2f}, {hi:.2f}]"


def format_signed(t: tuple[float, float, float]) -> str:
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def _source_of(system: str) -> str:
    return "fc" if system == "fullctx" else "gen"


def _citation_table(records: Sequence[Record], decisions: Decisions) -> list[str]:
    lines = [
        "| System (answer source) | answers given | citations | yes | partial | no | invalid "
        "| yes rate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for system, answered in decisions.answered.items():
        counts = dict.fromkeys(("yes", "partial", "no", "invalid"), 0)
        n_answers = 0
        for rec, ok in zip(records, answered, strict=True):
            cites = rec.cites_fc if _source_of(system) == "fc" else rec.cites_gen
            if not ok or cites is None:
                continue
            n_answers += 1
            for verdict in cites.values():
                counts[verdict if verdict in counts else "no"] += 1
        total = sum(counts.values())
        lines.append(
            f"| {system} ({_source_of(system).upper()}) | {n_answers} | {total} | "
            f"{counts['yes']} | {counts['partial']} | {counts['no']} | {counts['invalid']} | "
            f"{counts['yes'] / max(1, total):.3f} |"
        )
    return lines


def _citation_yes_rate(records: Sequence[Record], source: str) -> float:
    verdicts = [
        v for r in records for v in ((r.cites_fc if source == "fc" else r.cites_gen) or {}).values()
    ]
    return sum(v == "yes" for v in verdicts) / max(1, len(verdicts))


def render_generation(
    records: Sequence[Record],
    decisions: Decisions,
    client: CachedMessages,
    *,
    title: str,
) -> str:
    """All tables of the protocol, the pre-registered predictions and the decision rule."""
    labels = decisions.labels
    n_a, n_u = int(labels.sum()), int((~labels).sum())
    ans_idx = np.flatnonzero(labels)
    lines = [
        f"### {title}",
        "",
        f"{n_a} answerable and {n_u} unanswerable EU questions; model `{client.model}`; "
        "LLM-judged rows are labelled as such (no human check). CIs: Wilson for counts, "
        "stratified bootstrap (2000 resamples, seed 0) for differences.",
        "",
        "**Abstention AUROC (answerable vs unanswerable)**",
        "",
        "| Score | AUROC [95% CI] |",
        "|---|---|",
    ]
    for name, scores in decisions.scores.items():
        a, lo, hi = auroc_ci(scores, labels)
        lines.append(f"| `{name}` | {a:.3f} [{lo:.3f}, {hi:.3f}] |")
    for name, flags in decisions.answered.items():
        if name in ("self", "suff", "fullctx"):
            lines.append(f"| `{name}` (binary flag) | {auroc(flags.astype(float), labels):.3f} |")
    lines += ["", "**Decisions**", ""]
    base = decisions.answered.get("reader")
    head = (
        "| System | unanswerable answered | answerable wrongly abstained | balanced acc. "
        "| paired d vs reader | correct answers (LLM-judged, of "
        f"{n_a}) | partial | correct when answered |"
    )
    lines += [head, "|---|---|---|---|---|---|---|---|"]
    for system, answered in decisions.answered.items():
        ya, _, yu, _ = accept_counts(answered, labels)
        verdicts = verdicts_of(records, _source_of(system))
        yes = sum(1 for i in ans_idx if answered[i] and verdicts[i] == "yes")
        partial = sum(1 for i in ans_idx if answered[i] and verdicts[i] == "partial")
        bal = balacc_ci(answered, labels)
        if base is None or system == "reader":
            delta = "baseline" if system == "reader" else "-"
        else:
            delta = format_signed(paired_balacc_ci(answered, base, labels))
        lines.append(
            f"| `{system}` | {format_rate(n_u - yu, n_u)} | {format_rate(n_a - ya, n_a)} "
            f"| {bal[0]:.3f} [{bal[1]:.3f}, {bal[2]:.3f}] | {delta} | {format_rate(yes, n_a)} "
            f"| {partial} | {yes}/{max(1, ya)} |"
        )
    lines += [
        "",
        "**Citation support (LLM-judged: does the cited article support the answer?)**",
        "",
    ]
    lines += _citation_table(records, decisions)
    lines += ["", "**Retrieval misses and unanswerable types**", ""]
    miss = np.array([r.retrieval_miss for r in records], dtype=np.bool_)
    lines.append(
        f"{int(miss.sum())} of {n_a} answerable questions have no gold article in the retrieved "
        "chunks."
    )
    for system in ("suff", "self", "reader"):
        if system in decisions.answered:
            abst = ~decisions.answered[system] & labels
            lines.append(
                f"- `{system}` wrongly abstains on {int(abst.sum())} answerable questions, "
                f"{int((abst & miss).sum())} of them retrieval misses."
            )
    kinds = sorted({r.item.kind for r in records if not r.item.answerable})
    lines += [
        "",
        "| unanswerable type | n | "
        + " | ".join(f"`{s}` answered" for s in decisions.answered)
        + " |",
    ]
    lines.append("|---|---|" + "---|" * len(decisions.answered))
    for kind in kinds:
        idx = [i for i, r in enumerate(records) if not r.item.answerable and r.item.kind == kind]
        cells = [str(int(sum(decisions.answered[s][i] for i in idx))) for s in decisions.answered]
        lines.append(f"| {kind} | {len(idx)} | " + " | ".join(cells) + " |")
    lines += ["", "**Full context vs RAG (accuracy on the answerable questions, LLM-judged)**", ""]
    lines += _fullctx_lines(records, decisions)
    lines += ["", "**Calls and cost (assumed list prices)**", ""]
    lines += cost_lines(client)
    lines += ["", "**Pre-registered predictions and decision rule**", ""]
    lines += _prediction_lines(records, decisions)
    return "\n".join(lines)


def _fullctx_lines(records: Sequence[Record], decisions: Decisions) -> list[str]:
    if "fullctx" not in decisions.answered:
        return ["No full-context run in these records."]
    labels = decisions.labels
    idx = np.flatnonzero(labels)
    fc_ok = correct_flags(records, "fc")
    gen_ok = correct_flags(records, "gen")
    rows = [
        ("RAG `self` (GEN answers when `context_supports`)", "self", gen_ok),
        ("RAG `suff` (GEN answers when sufficient)", "suff", gen_ok),
        ("RAG, always answer (GEN, no gate)", None, gen_ok),
        ("full context `fullctx`", "fullctx", fc_ok),
        ("full context, always answer (FC, no gate)", None, fc_ok),
    ]
    lines = [
        "| System | correct (of answerable) | paired d vs full context always-answer |",
        "|---|---|---|",
    ]
    ref = [bool(fc_ok[i]) for i in idx]
    for label, system, ok in rows:
        flags = [bool(ok[i] and (system is None or decisions.answered[system][i])) for i in idx]
        d = paired_bootstrap_ci([float(f) for f in flags], [float(r) for r in ref])
        lines.append(f"| {label} | {format_rate(sum(flags), len(idx))} | {format_signed(d)} |")
    return lines


def cost_lines(client: CachedMessages) -> list[str]:
    records = list(client.records.values())
    totals = usage_totals(records)
    lines = [
        f"{totals['calls']} cached calls ({client.live_calls} live in this run); tokens: "
        f"input {totals.get('input_tokens', 0):,}, cache write "
        f"{totals.get('cache_creation_input_tokens', 0):,}, cache read "
        f"{totals.get('cache_read_input_tokens', 0):,}, output {totals.get('output_tokens', 0):,}; "
        f"cost at assumed list prices (3 / 15 USD per M input / output, cache write 1.25x, read "
        f"0.1x): {client.spent():.2f} USD.",
        "",
        "| call | calls | USD |",
        "|---|---|---|",
    ]
    for kind, (n, cost) in sorted(usage_by_kind(records).items()):
        lines.append(f"| {kind} | {n} | {cost:.2f} |")
    return lines


def _prediction_lines(records: Sequence[Record], decisions: Decisions) -> list[str]:
    labels = decisions.labels
    n_a, n_u = int(labels.sum()), int((~labels).sum())
    d = decisions.answered
    out: list[str] = []

    def verdict(ok: bool) -> str:
        return "holds" if ok else "does not hold"

    conf = auroc(decisions.scores["suff_confidence"], labels)
    reader_auc = (
        auroc(decisions.scores["qa_answerable"], labels)
        if "qa_answerable" in decisions.scores
        else None
    )
    ref = f" (reader {reader_auc:.3f})" if reader_auc is not None else ""
    out.append(
        f"- A1 `SUFF.confidence` AUROC {conf:.3f}{ref} above the reader and at least 0.90: "
        + verdict(conf >= 0.90 and (reader_auc is None or conf > reader_auc))
    )
    ya, _, yu, _ = accept_counts(d["suff"], labels)
    out.append(
        f"- A2 `suff` answers {n_u - yu}/{n_u} unanswerable (<= 6) and wrongly abstains on "
        f"{n_a - ya}/{n_a} answerable (<= 10): " + verdict(n_u - yu <= 6 and n_a - ya <= 10)
    )
    _, _, yu_self, _ = accept_counts(d["self"], labels)
    out.append(
        f"- A3 `self` answers {n_u - yu_self} unanswerable vs `suff` {n_u - yu} (self more): "
        + verdict(n_u - yu_self > n_u - yu)
    )
    gen_rate = _citation_yes_rate(records, "gen")
    fc_rate = _citation_yes_rate(records, "fc") if "fullctx" in d else None
    fc_text = f", FC {fc_rate:.3f} (>= 0.90)" if fc_rate is not None else ""
    out.append(
        f"- A4 citation `yes` rate GEN {gen_rate:.3f} (>= 0.85){fc_text}: "
        + verdict(gen_rate >= 0.85 and (fc_rate is None or fc_rate >= 0.90))
    )
    if "fullctx" in d:
        idx = np.flatnonzero(labels)
        fc_acc = float(np.mean([correct_flags(records, "fc")[i] and d["fullctx"][i] for i in idx]))
        self_acc = float(np.mean([correct_flags(records, "gen")[i] and d["self"][i] for i in idx]))
        _, _, yu_fc, _ = accept_counts(d["fullctx"], labels)
        out.append(
            f"- A5 `fullctx` accuracy {fc_acc:.3f} vs RAG `self` {self_acc:.3f} (>= +0.05) and "
            f"answers {n_u - yu_fc} unanswerable (<= `self` {n_u - yu_self}): "
            + verdict(fc_acc - self_acc >= 0.05 and n_u - yu_fc <= n_u - yu_self)
        )
    if "strict" in d:
        gates = {s: d[s] for s in ("reader", "self", "suff", "strict", "lenient", "stack")}
        wrong_u = {s: int((flags & ~labels).sum()) for s, flags in gates.items()}
        wrong_a = {s: int((~flags & labels).sum()) for s, flags in gates.items()}
        out.append(
            f"- A6 `strict` has the fewest unanswerable answered ({wrong_u['strict']}; others "
            f"{ {k: v for k, v in wrong_u.items() if k != 'strict'} }) and the most wrong "
            f"abstentions ({wrong_a['strict']}; others "
            f"{ {k: v for k, v in wrong_a.items() if k != 'strict'} }): "
            + verdict(
                wrong_u["strict"] <= min(wrong_u.values())
                and wrong_a["strict"] >= max(wrong_a.values())
            )
        )
        diff = paired_balacc_ci(d["suff"], d["reader"], labels)
        out.append(
            f"- Decision rule: balanced accuracy `suff` minus `reader` {format_signed(diff)}; "
            + (
                "CI lower bound above 0: the sufficiency check is adopted as the `--llm` gate."
                if diff[1] > 0
                else "CI lower bound not above 0: no adoption (no detectable effect"
                + (" or a loss)." if diff[2] < 0 else ").")
            )
        )
        for system in ("stack", "strict", "lenient"):
            sec = paired_balacc_ci(d[system], d["reader"], labels)
            out.append(
                f"- secondary: `{system}` minus `reader` balanced accuracy {format_signed(sec)}."
            )
    return out
