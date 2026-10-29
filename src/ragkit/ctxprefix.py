"""Model-free "contextual retrieval lite" (protocol: ``docs/ctxprefix_protocol.md``).

Every chunk gets the article's own context, ``<act title> — Article N: <article heading>``,
prepended to its text, so the BM25 text and the hashing-dense text of a chunk both carry the
article it came from. No model is involved: the context is the heading line the corpus already
has. Only numpy, the standard library and the existing ``BM25Index`` / ``HashingEmbedder`` /
``HybridRetriever`` are used; the evaluation is pinned to those constructors.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from ragkit.bm25 import BM25Index
from ragkit.chunking import Chunk, recursive
from ragkit.dense import DenseIndex
from ragkit.embed import HashingEmbedder
from ragkit.eval import bootstrap_ci, paired_bootstrap_ci
from ragkit.hybrid import HybridRetriever
from ragkit.lexical import METRIC_LABELS, METRICS, Question, rank_documents, score_question

# Frozen by the protocol, committed before any scoring.
CHUNK_SIZE = 500
OVERLAP = 50
DENSE_DIM = 512
SEARCH_CHUNKS = 10
MIN_MRR_GAIN_LOWER = 0.0  # fresh set: paired MRR CI lower bound must exceed this
SYNTHETIC_MAX_DROP = 0.02  # synthetic set: paired MRR may not fall by more than this
SATURATION_RECALL5 = 0.85  # hard set: baseline BM25 Recall@5 above this means "saturated"
FLOOR_RECALL5 = 0.15  # no-keyword set: baseline BM25 Recall@5 below this means "too hard"

RETRIEVERS = ("bm25", "hybrid")
VARIANTS = ("base", "prefix")

Retriever = Callable[[str], list[str]]

_HEADING_LINE = re.compile(r"^[ \t]*#[ \t]+(?P<heading>\S.*?)[ \t]*$", re.MULTILINE)
_ARTICLE_HEADING = re.compile(r"^(?P<act>.+), Article (?P<num>\d+[A-Za-z]?): (?P<title>.+)$")
_ACT_OF_DOC_ID = re.compile(r"_art\d+[a-z]*$")
_ACT_NUMBER = re.compile(r"\d{4}/\d{1,4}")


def heading_of(text: str) -> str:
    """The first ``# `` heading line of a document without its marker; ``""`` when absent."""
    match = _HEADING_LINE.search(text)
    return match.group("heading") if match else ""


def context_prefix(heading: str) -> str:
    """``<act title> — Article N: <article heading>`` from an article heading.

    Any heading that does not have the ``<act>, Article N: <title>`` shape is returned as is.
    """
    match = _ARTICLE_HEADING.match(heading)
    if match is None:
        return heading
    return f"{match.group('act')} — Article {match.group('num')}: {match.group('title')}"


def with_context_prefix(chunks: Sequence[Chunk], document: str) -> list[Chunk]:
    """Chunks of one ``document`` with its context prefix prepended to every chunk's text.

    A leading ``# heading`` line (the first recursive chunk starts with it) is replaced by the
    prefix, so no chunk carries the heading twice. Spans, ids and titles are unchanged.
    A document without a heading is returned unchanged.
    """
    heading = heading_of(document)
    if not heading:
        return list(chunks)
    prefix = context_prefix(heading)
    out: list[Chunk] = []
    for chunk in chunks:
        body = chunk.text
        first = _HEADING_LINE.match(body.lstrip())
        if first is not None and first.group("heading") == heading:
            body = body.lstrip()[first.end() :].lstrip("\n")
        out.append(replace(chunk, text=f"{prefix}\n{body}"))
    return out


def build_chunks(documents: Sequence[tuple[str, str]], *, prefix: bool) -> list[Chunk]:
    """The ``recursive`` 500/50 chunks of every document, optionally with the context prefix."""
    chunks: list[Chunk] = []
    for doc_id, text in documents:
        doc_chunks = recursive(text, doc_id, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
        chunks.extend(with_context_prefix(doc_chunks, text) if prefix else doc_chunks)
    return chunks


def bm25_retriever(chunks: Sequence[Chunk]) -> Retriever:
    """Pinned BM25 (k1 1.5, b 0.75): top chunks, deduplicated to documents."""
    index = BM25Index()
    index.add(chunks)
    return lambda q: rank_documents(index.search(q, k=SEARCH_CHUNKS))


def hybrid_retriever(chunks: Sequence[Chunk]) -> Retriever:
    """Pinned hybrid: BM25 + hashing-embedder dense index fused with RRF, as in ``ragkit eval``."""
    embedder = HashingEmbedder(dim=DENSE_DIM)
    embedder.fit([c.text for c in chunks])
    hybrid = HybridRetriever(bm25=BM25Index(), dense=DenseIndex(embedder=embedder))
    hybrid.add(chunks)
    return lambda q: rank_documents(hybrid.search(q, k=SEARCH_CHUNKS))


BUILDERS: dict[str, Callable[[Sequence[Chunk]], Retriever]] = {
    "bm25": bm25_retriever,
    "hybrid": hybrid_retriever,
}


def act_of(doc_id: str) -> str:
    """Act of an article id: ``reg_2019_943_art16`` -> ``reg_2019_943`` (other ids: unchanged)."""
    return _ACT_OF_DOC_ID.sub("", doc_id)


def is_wrong_act(retrieved: Sequence[str], relevant: Sequence[str]) -> bool:
    """True when the top-1 document is from an act none of the relevant documents belong to."""
    if not retrieved:
        return True
    return act_of(retrieved[0]) not in {act_of(d) for d in relevant}


@dataclass
class VariantResult:
    """Per-question outcome of one retriever with one chunk-text variant."""

    scores: dict[str, list[float]]
    wrong_act: list[bool]


@dataclass
class SetResult:
    """Every (retriever, variant) outcome on one question set."""

    label: str
    questions: list[Question]
    results: dict[tuple[str, str], VariantResult]


def evaluate_set(
    documents: Sequence[tuple[str, str]], questions: Sequence[Question], label: str
) -> SetResult:
    """Score both retrievers with and without the prefix on one question set."""
    results: dict[tuple[str, str], VariantResult] = {}
    for variant in VARIANTS:
        chunks = build_chunks(documents, prefix=variant == "prefix")
        for name in RETRIEVERS:
            retriever = BUILDERS[name](chunks)
            scores: dict[str, list[float]] = {m: [] for m in METRICS}
            wrong: list[bool] = []
            for question in questions:
                retrieved = retriever(question.text)
                for metric, value in score_question(retrieved, question.relevant).items():
                    scores[metric].append(value)
                wrong.append(is_wrong_act(retrieved, question.relevant))
            results[(name, variant)] = VariantResult(scores, wrong)
    return SetResult(label, list(questions), results)


def paired_mrr(result: SetResult, retriever: str) -> tuple[float, float, float]:
    """Paired ``prefix`` minus ``base`` MRR difference (mean, lo, hi) of one retriever."""
    return paired_bootstrap_ci(
        result.results[(retriever, "prefix")].scores["mrr"],
        result.results[(retriever, "base")].scores["mrr"],
    )


@dataclass(frozen=True)
class Verdict:
    """Outcome of the pre-registered decision rule for one retriever."""

    retriever: str
    decision: str
    fresh: tuple[float, float, float]
    synthetic: tuple[float, float, float]
    gain_on_fresh: bool
    synthetic_ok: bool


def decide(fresh: SetResult, synthetic: SetResult) -> list[Verdict]:
    """Apply the protocol's rule: adopt iff the fresh MRR gain CI is above 0 and synthetic holds."""
    verdicts: list[Verdict] = []
    for name in RETRIEVERS:
        f_mean, f_lo, f_hi = paired_mrr(fresh, name)
        s_mean, s_lo, s_hi = paired_mrr(synthetic, name)
        gain = f_lo > MIN_MRR_GAIN_LOWER
        synthetic_ok = s_mean >= -SYNTHETIC_MAX_DROP and s_hi >= 0.0
        if gain and synthetic_ok:
            decision = "ADOPTED"
        elif f_hi < 0.0 or not synthetic_ok:
            decision = "REJECTED"
        else:
            decision = "NOT ADOPTED (no evidence)"
        verdicts.append(
            Verdict(name, decision, (f_mean, f_lo, f_hi), (s_mean, s_lo, s_hi), gain, synthetic_ok)
        )
    return verdicts


def _cell(values: Sequence[float]) -> str:
    mean, lo, hi = bootstrap_ci(values)
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def _delta(a: Sequence[float], b: Sequence[float]) -> str:
    mean, lo, hi = paired_bootstrap_ci(a, b)
    return f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def _better_worse_tied(a: Sequence[float], b: Sequence[float]) -> str:
    better = sum(1 for x, y in zip(a, b, strict=True) if x > y + 1e-9)
    worse = sum(1 for x, y in zip(a, b, strict=True) if x < y - 1e-9)
    return f"{better} / {worse} / {len(a) - better - worse}"


def render_set(result: SetResult, note: str = "") -> str:
    """Markdown tables of one question set: scores, paired differences and wrong-act counts."""
    n = len(result.questions)
    heading = f"#### {result.label} ({n} questions{note})"
    lines = [heading, "", "mean [bootstrap 95% CI over questions, 2000 resamples, seed 0]", ""]
    lines += ["| Retriever | Chunk text | Recall@5 | MRR | nDCG@10 | wrong-act top-1 |"]
    lines += ["|---|---|---|---|---|---|"]
    for name in RETRIEVERS:
        for variant in VARIANTS:
            res = result.results[(name, variant)]
            cells = " | ".join(_cell(res.scores[m]) for m in METRICS)
            lines.append(f"| {name} | {variant} | {cells} | {sum(res.wrong_act)} / {n} |")
    lines += ["", "Paired difference `prefix` minus `base` (same questions):", ""]
    lines += ["| Retriever | dRecall@5 | dMRR | dnDCG@10 | wrong-act: fixed / broken |"]
    lines += ["|---|---|---|---|---|"]
    for name in RETRIEVERS:
        base, pre = result.results[(name, "base")], result.results[(name, "prefix")]
        cells = " | ".join(_delta(pre.scores[m], base.scores[m]) for m in METRICS)
        fixed = sum(1 for b, p in zip(base.wrong_act, pre.wrong_act, strict=True) if b and not p)
        broken = sum(1 for b, p in zip(base.wrong_act, pre.wrong_act, strict=True) if p and not b)
        lines.append(f"| {name} | {cells} | {fixed} / {broken} |")
    lines += ["", "Questions better / worse / tied with the prefix:", ""]
    lines += ["| Retriever | " + " | ".join(METRIC_LABELS[m] for m in METRICS) + " |"]
    lines += ["|---|---|---|---|"]
    for name in RETRIEVERS:
        base, pre = result.results[(name, "base")], result.results[(name, "prefix")]
        cells = " | ".join(_better_worse_tied(pre.scores[m], base.scores[m]) for m in METRICS)
        lines.append(f"| {name} | {cells} |")
    types = sorted({q.qtype for q in result.questions if q.qtype})
    if len(types) > 1:
        lines += ["", "Paired dMRR and wrong-act count (base -> prefix) by question type:", ""]
        lines += ["| Retriever | " + " | ".join(types) + " |", "|---|" + "---|" * len(types)]
        for name in RETRIEVERS:
            base, pre = result.results[(name, "base")], result.results[(name, "prefix")]
            type_cells: list[str] = []
            for t in types:
                idx = [i for i, q in enumerate(result.questions) if q.qtype == t]
                diff = _delta(
                    [pre.scores["mrr"][i] for i in idx], [base.scores["mrr"][i] for i in idx]
                )
                wb = sum(base.wrong_act[i] for i in idx)
                wp = sum(pre.wrong_act[i] for i in idx)
                type_cells.append(f"n={len(idx)}: {diff}; wrong-act {wb} -> {wp}")
            lines.append(f"| {name} | " + " | ".join(type_cells) + " |")
    lines.append("")
    return "\n".join(lines)


def render_verdicts(
    verdicts: Sequence[Verdict], primary: str = "fresh", protocol: str = "ctxprefix_protocol.md"
) -> str:
    """The decision rule's inputs and outcome per retriever (``primary`` names the gating set)."""
    lines = [
        f"Decision rule (docs/{protocol}): adopt iff the {primary}-set paired dMRR CI lower",
        "bound is above 0 and the synthetic dMRR is >= -0.02 with CI upper bound >= 0.",
        "",
        f"| Retriever | {primary} dMRR | gain on {primary} | synthetic dMRR | synthetic ok "
        "| verdict |",
        "|---|---|---|---|---|---|",
    ]
    for v in verdicts:
        fresh = f"{v.fresh[0]:+.3f} [{v.fresh[1]:+.3f}, {v.fresh[2]:+.3f}]"
        synth = f"{v.synthetic[0]:+.3f} [{v.synthetic[1]:+.3f}, {v.synthetic[2]:+.3f}]"
        gain, ok = ("yes" if v.gain_on_fresh else "no"), ("yes" if v.synthetic_ok else "no")
        lines.append(f"| {v.retriever} | {fresh} | {gain} | {synth} | {ok} | **{v.decision}** |")
    lines.append("")
    return "\n".join(lines)


def baseline_recall5(result: SetResult, retriever: str = "bm25") -> float:
    """Mean Recall@5 of the ``base`` chunk text of one retriever on a question set."""
    values = result.results[(retriever, "base")].scores["r5"]
    return sum(values) / len(values)


def is_saturated(result: SetResult) -> bool:
    """Pre-registered saturation guard: baseline BM25 Recall@5 above ``SATURATION_RECALL5``."""
    return baseline_recall5(result) > SATURATION_RECALL5


def render_saturation(result: SetResult, protocol: str = "ctxprefix_hard_protocol.md") -> str:
    """The saturation guard's input and outcome, in one line for the report."""
    recall = baseline_recall5(result)
    outcome = "SATURATED" if is_saturated(result) else "not saturated"
    return (
        f"Saturation guard (docs/{protocol}): baseline BM25 Recall@5 = "
        f"{recall:.3f}, threshold {SATURATION_RECALL5:.2f} -> **{outcome}**.\n"
    )


def is_too_hard(result: SetResult) -> bool:
    """Pre-registered floor: baseline BM25 Recall@5 below ``FLOOR_RECALL5`` (no gold retrieved)."""
    return baseline_recall5(result) < FLOOR_RECALL5


def render_floor(result: SetResult, protocol: str = "ctxprefix_nokw_protocol.md") -> str:
    """The floor guard's input and outcome; below the floor the set cannot test the prefix."""
    recall = baseline_recall5(result)
    if is_too_hard(result):
        outcome = (
            "**TOO HARD FOR LEXICAL RETRIEVAL**: the prefix cannot help when the gold article "
            "is not in the candidate set"
        )
    else:
        outcome = "**not too hard**"
    return (
        f"Floor guard (docs/{protocol}): baseline BM25 Recall@5 = {recall:.3f}, "
        f"floor {FLOOR_RECALL5:.2f} -> {outcome}.\n"
    )


def names_an_act(question: str) -> bool:
    """True when the question text contains an act number such as ``2015/1222``."""
    return _ACT_NUMBER.search(question) is not None


def without_act_named(questions: Sequence[Question]) -> list[Question]:
    """The questions that do not name their act (sensitivity set of the protocol)."""
    return [q for q in questions if not names_an_act(q.text)]


def render_protocol(
    fresh: SetResult, old: SetResult, synthetic: SetResult, sensitivity: SetResult
) -> str:
    """Full protocol report: fresh (primary), old (tainted), synthetic (guard), sensitivity."""
    return "\n".join(
        [
            render_set(fresh, ", primary evidence"),
            render_set(sensitivity, ", sensitivity: act-named questions removed, never gates"),
            render_set(old, ", TAINTED by earlier diagnosis, secondary, never gates"),
            render_set(synthetic, ", regression guard"),
            render_verdicts(decide(fresh, synthetic)),
        ]
    )


def render_hard_protocol(hard: SetResult, synthetic: SetResult) -> str:
    """Hard-set report: scores, saturation guard, synthetic guard, the unchanged decision rule."""
    return "\n".join(
        [
            render_set(hard, ", primary evidence"),
            render_saturation(hard),
            render_set(synthetic, ", regression guard"),
            render_verdicts(
                decide(hard, synthetic), primary="hard", protocol="ctxprefix_hard_protocol.md"
            ),
        ]
    )


def render_nokw_protocol(nokw: SetResult, synthetic: SetResult) -> str:
    """No-keyword-set report: scores, both guards, synthetic guard, the unchanged decision rule."""
    protocol = "ctxprefix_nokw_protocol.md"
    return "\n".join(
        [
            render_set(nokw, ", primary evidence"),
            render_saturation(nokw, protocol),
            render_floor(nokw, protocol),
            render_set(synthetic, ", regression guard"),
            render_verdicts(decide(nokw, synthetic), primary="no-keyword", protocol=protocol),
        ]
    )
