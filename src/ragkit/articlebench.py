"""Phase L: parent-document (article) context against chunk context and full context.

Protocol, predictions and decision rule: ``docs/generation_protocol.md`` (Phase L). The chunk
records of K-R2 (``genbench.run_generation``, replayed from the cache) supply the retrieved
chunks, the ``SUFF`` confidences, the full-context answers and their judges. Only ``GEN``,
``CORRECT`` and ``CITE`` are new, with the full text of the top-k distinct parent articles as the
context (``ragkit.parent``).
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from ragkit.abstention import accept_counts
from ragkit.eval import paired_bootstrap_ci
from ragkit.genbench import (
    Decisions,
    Record,
    balacc_ci,
    cost_lines,
    format_rate,
    format_signed,
    judge_citations,
    verdicts_of,
)
from ragkit.generation import LLMAnswerer
from ragkit.llm import CachedMessages
from ragkit.parent import ARTICLE_BUDGET_TOKENS, ARTICLE_K, parent_articles

# Fixed before the scored run (docs/generation_protocol.md, Phase L): a text match, no judge.
CONTEXT_GAP = re.compile(
    r"cut off|cuts off|cut short|truncat|incomplete|not provided|not included"
    r"|cannot (?:confirm|determine)"
    r"|\b(?:passages?|context|text|excerpts?|fragments?)\b[^.]{0,40}"
    r"\b(?:do|does|did)(?:n't| not)\b",
    re.IGNORECASE,
)
LEVELS: dict[str, tuple[str, ...]] = {"strict": ("yes",), "lenient": ("yes", "partial")}
FC_SAMPLE_SIZE = 10
FC_SAMPLE_SEED = 0


def says_context_gap(answer: str) -> bool:
    """True if the answer says the passages are cut off, not provided or do not state the fact."""
    return CONTEXT_GAP.search(answer) is not None


def cite_mask(decisions: Decisions) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Questions whose article-context ``CITE`` calls are made: answerable, or answered by a gate.

    The reported rows count the answers of ``reader`` and ``stack`` only, so an unanswerable
    question that neither answers needs no citation judgement.
    """
    both = decisions.answered["reader"] | decisions.answered["stack"]
    return np.asarray(decisions.labels | both, dtype=np.bool_)


def run_article_generation(
    chunk_records: Sequence[Record],
    documents: Sequence[tuple[str, str]],
    answerer: LLMAnswerer,
    mask: Sequence[bool],
    *,
    k: int = ARTICLE_K,
    budget_tokens: int = ARTICLE_BUDGET_TOKENS,
    workers: int = 6,
) -> list[Record]:
    """``GEN`` on the parent articles of each question's chunks, plus ``CORRECT`` and ``CITE``.

    The returned records keep the chunk records' ``SUFF`` and full-context results; ``passages``
    is the article context ``GEN`` saw, and ``cites_gen`` is ``None`` outside ``mask``.
    """
    articles = dict(documents)

    def one(i: int) -> Record:
        base = chunk_records[i]
        item = base.item
        context = parent_articles(base.passages, articles, k=k, budget_tokens=budget_tokens)
        gen = answerer.generate(item.question, context)
        rec = Record(
            item, context, gen, base.suff, base.fc, None, base.correct_fc, None, base.cites_fc
        )
        if item.answerable:
            rec.correct_gen = answerer.judge_correct(item.question, item.reference, gen.answer)
        if mask[i]:
            rec.cites_gen = judge_citations(
                answerer, item, gen, {p.article_id for p in context}, articles
            )
        return rec

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, range(len(chunk_records))))


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------


def level_flags(
    records: Sequence[Record], source: str, level: str
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Per question: the ``CORRECT`` verdict is in the level's set (False if not judged)."""
    wanted = LEVELS[level]
    return np.array([v in wanted for v in verdicts_of(records, source)], dtype=np.bool_)


def _paired(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any], idx: np.ndarray[Any, Any]) -> str:
    return format_signed(
        paired_bootstrap_ci([float(a[i]) for i in idx], [float(b[i]) for i in idx])
    )


def _cite_counts(
    records: Sequence[Record], answered: Sequence[bool] | np.ndarray[Any, Any], source: str
) -> dict[str, int]:
    counts = dict.fromkeys(("answers", "yes", "partial", "no", "invalid"), 0)
    for rec, ok in zip(records, answered, strict=True):
        cites = rec.cites_fc if source == "fc" else rec.cites_gen
        if not ok or cites is None:
            continue
        counts["answers"] += 1
        for verdict in cites.values():
            counts[verdict if verdict in counts else "no"] += 1
    return counts


def _cite_row(label: str, counts: Mapping[str, int]) -> str:
    total = sum(counts[v] for v in ("yes", "partial", "no", "invalid"))
    return (
        f"| {label} | {counts['answers']} | {total} | {counts['yes']} | {counts['partial']} "
        f"| {counts['no']} | {counts['invalid']} | {counts['yes'] / max(1, total):.3f} |"
    )


def _cite_rate(counts: Mapping[str, int]) -> float:
    total = sum(counts[v] for v in ("yes", "partial", "no", "invalid"))
    return counts["yes"] / max(1, total)


def _gap_row(
    label: str, records: Sequence[Record], source: str, labels: Any
) -> tuple[str, int, int]:
    verdicts = verdicts_of(records, source)
    nonyes = [int(i) for i in np.flatnonzero(labels) if verdicts[i] != "yes"]
    gap = [i for i in nonyes if says_context_gap(_answer(records[i], source))]
    by_verdict = {v: sum(1 for i in nonyes if verdicts[i] == v) for v in ("partial", "no")}
    share = len(gap) / max(1, len(nonyes))
    return (
        f"| {label} | {len(nonyes)} ({by_verdict['no']} no, {by_verdict['partial']} partial) "
        f"| {len(gap)} | {share:.2f} |",
        len(nonyes),
        len(gap),
    )


def _answer(rec: Record, source: str) -> str:
    answer = rec.fc if source == "fc" else rec.gen
    return answer.answer if answer is not None else ""


def fc_citation_audit(records: Sequence[Record]) -> list[str]:
    """FC citations by verdict against whether the cited id is a gold article (cache only)."""
    rows: dict[tuple[str, bool], int] = {}
    by_count = {"one": [0, 0], "several": [0, 0]}  # [yes, total] by citations in the answer
    gold_cited = answerable = 0
    for rec in records:
        if rec.fc is None or rec.cites_fc is None or not rec.item.answerable:
            continue
        answerable += 1
        gold = set(rec.item.relevant)
        gold_cited += bool(gold & set(rec.cites_fc))
        bucket = by_count["one" if len(rec.cites_fc) == 1 else "several"]
        for cited, verdict in rec.cites_fc.items():
            rows[(verdict, cited in gold)] = rows.get((verdict, cited in gold), 0) + 1
            bucket[0] += verdict == "yes"
            bucket[1] += 1
    lines = [
        f"FC answers to the {answerable} answerable questions cite a gold article in {gold_cited}.",
        "",
        "| verdict | cited id is a gold article | cited id is not a gold article |",
        "|---|---|---|",
    ]
    for verdict in ("yes", "partial", "no", "invalid"):
        lines.append(
            f"| {verdict} | {rows.get((verdict, True), 0)} | {rows.get((verdict, False), 0)} |"
        )
    for name, (yes, total) in by_count.items():
        lines.append(
            f"\nAnswers citing {name} article(s): {yes} of {total} citations `yes` "
            f"({yes / max(1, total):.2f})."
        )
    return lines


def _gold_rows(label: str, records: Sequence[Record], idx: np.ndarray[Any, Any]) -> list[str]:
    """Rows of the post-hoc table: ``CORRECT`` verdicts by gold article in/out of the context."""
    verdicts = verdicts_of(records, "gen")
    rows = []
    for present in (True, False):
        chosen = [i for i in idx if (not records[i].retrieval_miss) == present]
        counts = [sum(1 for i in chosen if verdicts[i] == v) for v in ("yes", "partial", "no")]
        rows.append(
            f"| {label} | {'yes' if present else 'no'} | {len(chosen)} | {counts[0]} "
            f"| {counts[1]} | {counts[2]} |"
        )
    return rows


def fc_citation_sample(
    records: Sequence[Record],
    articles: Mapping[str, str],
    answerer: LLMAnswerer,
    *,
    n: int = FC_SAMPLE_SIZE,
    seed: int = FC_SAMPLE_SEED,
    answerable_only: bool = False,
) -> list[str]:
    """``n`` random non-``yes`` FC ``CITE`` records with the judge's reason, for a manual read."""
    pool = [
        (rec, cited, verdict)
        for rec in records
        if rec.cites_fc is not None and rec.fc is not None
        if rec.item.answerable or not answerable_only
        for cited, verdict in rec.cites_fc.items()
        if verdict != "yes"
    ]
    picked = random.Random(seed).sample(pool, min(n, len(pool)))
    lines = []
    for number, (rec, cited, verdict) in enumerate(picked, 1):
        assert rec.fc is not None and rec.cites_fc is not None
        reason = (
            answerer.judge_citation(rec.item.question, rec.fc.answer, articles[cited]).reason
            if cited in articles
            else "id not in the corpus"
        )
        gold = ", ".join(rec.item.relevant) if rec.item.answerable else "(unanswerable)"
        lines.append(
            f"{number}. `{cited}` judged **{verdict}**; gold: {gold}; the answer cites "
            f"{len(rec.cites_fc)}: {', '.join(rec.cites_fc)}.\n"
            f"   Q: {rec.item.question}\n   A: {rec.fc.answer}\n   Judge: {reason}"
        )
    return lines


def render_article_report(
    chunk_records: Sequence[Record],
    article_records: Sequence[Record],
    decisions: Decisions,
    client: CachedMessages,
    answerer: LLMAnswerer,
    documents: Sequence[tuple[str, str]],
    *,
    title: str,
    k: int,
    budget_tokens: int,
    l1_threshold: int,
) -> str:
    """The Phase L tables, the pre-registered predictions and the decision rule."""
    labels = decisions.labels
    idx = np.flatnonzero(labels)
    n_a, n_u = len(idx), int((~labels).sum())
    lines = [
        f"### {title}",
        "",
        f"{n_a} answerable and {n_u} unanswerable EU questions; model `{client.model}`; context "
        f"for article rows: top-{k} distinct parent articles of the top-5 chunks, full text, "
        f"budget {budget_tokens} estimated tokens. All correctness and citation rows are "
        "LLM-judged (no human check). CIs: Wilson for counts, paired bootstrap over questions "
        "(2000 resamples, seed 0) for differences.",
        "",
        "**1. Always-answer correctness on the answerable questions (strict = `yes`; lenient = "
        "`yes` + `partial`)**",
        "",
        f"| Context | strict (of {n_a}) [Wilson 95%] | lenient (of {n_a}) [Wilson 95%] | `no` |",
        "|---|---|---|---|",
    ]
    contexts = {
        "chunk RAG (top-5 chunks)": (chunk_records, "gen"),
        "article RAG": (article_records, "gen"),
        "full context (FC)": (chunk_records, "fc"),
    }
    flags = {
        (name, level): level_flags(recs, source, level)
        for name, (recs, source) in contexts.items()
        for level in LEVELS
    }
    for name, (recs, source) in contexts.items():
        verdicts = verdicts_of(recs, source)
        no = sum(1 for i in idx if verdicts[i] == "no")
        strict = int(flags[name, "strict"][idx].sum())
        lenient = int(flags[name, "lenient"][idx].sum())
        lines.append(
            f"| {name} | {format_rate(strict, n_a)} | {format_rate(lenient, n_a)} | {no} |"
        )
    chunk, art, fc = (f"{n}" for n in contexts)
    lines += [
        "",
        "| Paired difference (mean of per-question 0/1 differences) | strict [95% CI] | lenient "
        "[95% CI] |",
        "|---|---|---|",
    ]
    for label, a, b in (
        ("article RAG - chunk RAG", art, chunk),
        ("article RAG - full context", art, fc),
        ("chunk RAG - full context", chunk, fc),
    ):
        cells = [_paired(flags[a, level], flags[b, level], idx) for level in LEVELS]
        lines.append(f"| {label} | {cells[0]} | {cells[1]} |")

    lines += ["", "**2. Gates with article-context answers (decisions as in K-R2)**", ""]
    lines += [
        "| System | context | unanswerable answered | answerable wrongly abstained | balanced acc. "
        f"| correct strict (of {n_a}) | correct lenient (of {n_a}) | paired d strict vs chunk |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for system in ("reader", "stack", "fullctx"):
        answered = decisions.answered[system]
        ya, _, yu, _ = accept_counts(answered, labels)
        bal = balacc_ci(answered, labels)
        shown = (
            [("FC", "fc", chunk_records)]
            if system == "fullctx"
            else [("chunk", "gen", chunk_records), ("article", "gen", article_records)]
        )
        base = None
        for label, source, recs in shown:
            got = level_flags(recs, source, "strict") & answered
            got_lenient = level_flags(recs, source, "lenient") & answered
            delta = "-"
            if label == "article" and base is not None:
                delta = _paired(got, base, idx)
            if label == "chunk":
                base = got
            lines.append(
                f"| `{system}` | {label} | {format_rate(n_u - yu, n_u)} "
                f"| {format_rate(n_a - ya, n_a)} "
                f"| {bal[0]:.3f} [{bal[1]:.3f}, {bal[2]:.3f}] "
                f"| {format_rate(int(got[idx].sum()), n_a)} "
                f"| {format_rate(int(got_lenient[idx].sum()), n_a)} | {delta} |"
            )

    lines += [
        "",
        "**3. Citation support (LLM-judged, per cited article; article-context `CITE` calls only "
        "on the questions behind the rows below)**",
        "",
        "| System (context) | answers given | citations | yes | partial | no | invalid "
        "| yes rate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    cite_rows: dict[str, dict[str, int]] = {}
    for system in ("reader", "stack"):
        for label, recs in (("chunk", chunk_records), ("article", article_records)):
            cite_rows[f"{system} ({label})"] = _cite_counts(recs, decisions.answered[system], "gen")
    for label, recs in (("chunk", chunk_records), ("article", article_records)):
        cite_rows[f"all answerable, always answer ({label})"] = _cite_counts(recs, labels, "gen")
    cite_rows["all answerable, always answer (FC)"] = _cite_counts(chunk_records, labels, "fc")
    cite_rows["fullctx (FC)"] = _cite_counts(chunk_records, decisions.answered["fullctx"], "fc")
    lines += [_cite_row(label, counts) for label, counts in cite_rows.items()]

    mask = [r.cites_gen is not None for r in article_records]
    fc_all = _cite_counts(chunk_records, [True] * len(chunk_records), "fc")
    chunk_same = _cite_counts(chunk_records, mask, "gen")
    article_same = _cite_counts(article_records, mask, "gen")
    lines += [
        "",
        f"On the same {sum(mask)} questions: chunk-context `GEN` citation yes rate "
        f"{_cite_rate(chunk_same):.3f}, article-context {_cite_rate(article_same):.3f}, "
        f"FC (all {len(chunk_records)} answers) {_cite_rate(fc_all):.3f}.",
    ]

    lines += [
        "",
        "**4. Context-gap answers among the non-`yes` verdicts (always answer, answerable "
        "questions; text match fixed in the protocol)**",
        "",
        "| Context | non-`yes` verdicts | answer says the context is cut off / not provided / "
        "does not state it | share |",
        "|---|---|---|---|",
    ]
    gap_shares: dict[str, float] = {}
    for name, (recs, source) in contexts.items():
        row, total, gap = _gap_row(name, recs, source, labels)
        lines.append(row)
        gap_shares[name] = gap / max(1, total)

    chunk_miss = int(sum(r.retrieval_miss for r in chunk_records))
    article_miss = int(sum(r.retrieval_miss for r in article_records))
    lines += [
        "",
        "**5. Retrieval misses (gold article absent from the context `GEN` sees)**",
        "",
        f"Top-5 chunks: {chunk_miss} of {n_a}. Top-{k} articles: {article_miss} of {n_a}.",
        "",
        "Post-hoc (not pre-registered): `CORRECT` verdicts of the always-answer rows by whether "
        "the gold article is in the context.",
        "",
        "| Context | gold article in context | questions | yes | partial | no |",
        "|---|---|---|---|---|---|",
        *_gold_rows("chunk RAG", chunk_records, idx),
        *_gold_rows("article RAG", article_records, idx),
        "",
        "**6. FC citation audit (cache only)**",
        "",
    ]
    lines += fc_citation_audit(chunk_records)
    lines += [
        "",
        f"Random sample (seed {FC_SAMPLE_SEED}) of non-`yes` FC citations (all 87 answers):",
        "",
    ]
    lines += fc_citation_sample(chunk_records, dict(documents), answerer)
    lines += [
        "",
        f"Post-hoc (not pre-registered): the same sample restricted to the {n_a} answerable "
        "questions, the population of the FC citation yes rate quoted in the README:",
        "",
    ]
    lines += fc_citation_sample(chunk_records, dict(documents), answerer, answerable_only=True)
    lines += ["", "**7. Calls and cost (assumed list prices)**", ""]
    lines += cost_lines(client)

    lines += ["", "**8. Pre-registered predictions and decision rule**", ""]
    strict_art = int(flags[art, "strict"][idx].sum())
    strict_chunk = int(flags[chunk, "strict"][idx].sum())
    d_chunk = paired_bootstrap_ci(
        [float(flags[art, "strict"][i]) for i in idx],
        [float(flags[chunk, "strict"][i]) for i in idx],
    )
    d_fc = paired_bootstrap_ci(
        [float(flags[art, "strict"][i]) for i in idx],
        [float(flags[fc, "strict"][i]) for i in idx],
    )
    verdict = {True: "holds", False: "does not hold"}
    fc_cites = [
        (v, c in rec.item.relevant)
        for rec in chunk_records
        if rec.item.answerable and rec.cites_fc
        for c, v in rec.cites_fc.items()
        if v != "yes"
    ]
    fc_nonyes = len(fc_cites)
    fc_partial_gold = sum(1 for v, gold in fc_cites if v == "partial" and gold)
    fc_partial_other = sum(1 for v, gold in fc_cites if v == "partial" and not gold)
    fc_no = sum(1 for v, _ in fc_cites if v == "no")
    lines += [
        f"- L1 article RAG strict {strict_art}/{n_a} (>= {l1_threshold}; chunk {strict_chunk}): "
        + verdict[strict_art >= l1_threshold],
        f"- L2 paired d article - chunk {format_signed(d_chunk)} (lower bound > 0): "
        + verdict[d_chunk[1] > 0],
        f"- L3 paired d article - FC {format_signed(d_fc)} (point in [-0.25, 0)): "
        + verdict[-0.25 <= d_fc[0] < 0],
        f"- L4 context-gap share of non-`yes`: chunk {gap_shares[chunk]:.2f}, article "
        f"{gap_shares[art]:.2f} (<= 0.30): " + verdict[gap_shares[art] <= 0.30],
        f"- L5 article citation yes rate {_cite_rate(article_same):.3f} (> chunk "
        f"{_cite_rate(chunk_same):.3f} on the same questions, >= 0.70, < 0.85): "
        + verdict[
            _cite_rate(article_same) > _cite_rate(chunk_same)
            and 0.70 <= _cite_rate(article_same) < 0.85
        ],
        "- L6 is read by hand (section 6; docs/generation_protocol.md). Automated part: among "
        f"the {fc_nonyes} non-`yes` FC citations on answerable questions, {fc_partial_gold} are "
        f"`partial` on a gold article, {fc_partial_other} `partial` on a non-gold article and "
        f"{fc_no} `no`.",
        "- Decision rule (this retriever): strict paired d article - chunk "
        f"{format_signed(d_chunk)}; "
        + (
            "lower bound above 0: article context passes for this retriever."
            if d_chunk[1] > 0
            else "lower bound not above 0: article context does not pass for this retriever."
        ),
    ]
    return "\n".join(lines)
