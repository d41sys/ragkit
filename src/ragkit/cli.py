"""Command-line interface for RagKit: indexing, question answering, and strategy evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from ragkit.abstention import (
    SIGNALS,
    SignalTable,
    accept_counts,
    auroc,
    auroc_ci,
    calibrate_threshold,
    collect_signals,
    cross_validated_accept,
    fit_logistic,
    nested_cv_logistic,
    paired_auroc_ci,
    risk_at_coverage,
)
from ragkit.articlebench import cite_mask, render_article_report, run_article_generation
from ragkit.beir import (
    BEIR_DATASETS,
    DEFAULT_BEIR_DIR,
    LADDER,
    SYSTEMS,
    SystemSpec,
    evaluate_system,
    load_beir,
    render_beir_report,
)
from ragkit.bm25 import BM25Index
from ragkit.chunking import Chunk, fixed, recursive, semantic, sentence
from ragkit.ctxprefix import evaluate_set as evaluate_ctxprefix
from ragkit.ctxprefix import render_hard_protocol as render_ctxprefix_hard_protocol
from ragkit.ctxprefix import render_nokw_protocol as render_ctxprefix_nokw_protocol
from ragkit.ctxprefix import render_protocol as render_ctxprefix_protocol
from ragkit.ctxprefix import render_set as render_ctxprefix_set
from ragkit.ctxprefix import without_act_named
from ragkit.dense import DenseIndex
from ragkit.embed import HashingEmbedder
from ragkit.eurlex import EU_ACTS, build_corpus, strip_metadata_comment
from ragkit.eval import (
    answer_f1,
    bootstrap_ci,
    faithfulness,
    mrr,
    ndcg_at_k,
    paired_bootstrap_ci,
    recall_at_k,
    wilson_interval,
)
from ragkit.genbench import (
    build_reader_subset,
    decide,
    items_from_records,
    offline_reader_scorers,
    reader_scores,
    render_generation,
    retrieve_passages,
    run_generation,
    subset_score_cache,
)
from ragkit.generation import LLMAnswerer, Passage
from ragkit.hybrid import HybridRetriever
from ragkit.legalbench import DEFAULT_LEGALBENCH_ZIP, load_mini, render_legalbench_report
from ragkit.legalbench import SYSTEMS as LEGAL_SYSTEMS
from ragkit.legalbench import evaluate_system as evaluate_legal
from ragkit.lexical import Question, questions_from_records, render_decision_inputs, render_report
from ragkit.lexical import evaluate as evaluate_lexical
from ragkit.lexical_explain import explain as explain_lexical
from ragkit.lexical_explain import render_explanation
from ragkit.llm import MODEL, CachedMessages
from ragkit.modelsignals import DEFAULT_CACHE, MODEL_SIGNALS, ModelScorers, load_model_scorers
from ragkit.nokw import load_index, render_rule_status
from ragkit.parent import ARTICLE_BUDGET_TOKENS, ARTICLE_K, parent_articles
from ragkit.pipeline import ABSTAIN_TEXT, DEFAULT_MIN_SUPPORT, RAGPipeline, query_support
from ragkit.rerank import LexicalReranker
from ragkit.rewrite import HyDERewriter, TemplateLLM

DEFAULT_DATA_DIR = Path(__file__).parent / "data"
DEFAULT_CORPUS_DIR = DEFAULT_DATA_DIR / "corpus"
DEFAULT_QA_PATH = DEFAULT_DATA_DIR / "qa.jsonl"
DEFAULT_UNANSWERABLE_PATH = DEFAULT_DATA_DIR / "unanswerable.jsonl"
EU_CORPUS_DIR = DEFAULT_DATA_DIR / "corpus_eu"
EU_QA_PATH = DEFAULT_DATA_DIR / "qa_eu.jsonl"
EU_FRESH_QA_PATH = DEFAULT_DATA_DIR / "qa_eu_fresh.jsonl"
EU_HARD_QA_PATH = DEFAULT_DATA_DIR / "qa_eu_hard.jsonl"
EU_NOKW_QA_PATH = DEFAULT_DATA_DIR / "qa_eu_nokw.jsonl"
EU_UNANSWERABLE_PATH = DEFAULT_DATA_DIR / "unanswerable_eu.jsonl"
COHORTS = ("all", "original-12", "expanded")


@dataclass(frozen=True)
class Dataset:
    """A corpus directory with its answerable and unanswerable question files."""

    corpus_dir: Path
    qa_path: Path
    unanswerable_path: Path


BUNDLED_DATASETS = {
    "synthetic": Dataset(DEFAULT_CORPUS_DIR, DEFAULT_QA_PATH, DEFAULT_UNANSWERABLE_PATH),
    "eu": Dataset(EU_CORPUS_DIR, EU_QA_PATH, EU_UNANSWERABLE_PATH),
}


def _add_dataset_args(parser: argparse.ArgumentParser, *, questions: bool = True) -> None:
    """Shared dataset selection: a bundled preset, each part overridable by path."""
    parser.add_argument(
        "--corpus",
        choices=sorted(BUNDLED_DATASETS),
        default="synthetic",
        help="Bundled dataset: synthetic energy-flexibility docs, or eu (EUR-Lex articles).",
    )
    parser.add_argument("--corpus-dir", default=None, help="Override the corpus directory.")
    if questions:
        parser.add_argument("--qa-file", default=None, help="Override the answerable QA jsonl.")
        parser.add_argument(
            "--unanswerable-file", default=None, help="Override the unanswerable jsonl."
        )
        parser.add_argument(
            "--cohort",
            choices=COHORTS,
            default="all",
            help="Keep only unanswerable questions of this cohort (original-12 or expanded).",
        )


def _resolve_dataset(args: argparse.Namespace) -> Dataset:
    """Bundled preset from ``--corpus``, with any explicit path flag taking precedence."""
    base = BUNDLED_DATASETS[args.corpus]
    qa_file = getattr(args, "qa_file", None)
    unanswerable_file = getattr(args, "unanswerable_file", None)
    return Dataset(
        Path(args.corpus_dir) if args.corpus_dir else base.corpus_dir,
        Path(qa_file) if qa_file else base.qa_path,
        Path(unanswerable_file) if unanswerable_file else base.unanswerable_path,
    )


def _ci(values: Sequence[float]) -> str:
    """``mean [lo, hi]`` with a 95% percentile-bootstrap interval over questions."""
    mean, lo, hi = bootstrap_ci(values)
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def _load_documents(corpus_dir: Path) -> list[tuple[str, str]]:
    """Load all markdown and text documents from a directory."""
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    docs: list[tuple[str, str]] = []
    for file_path in sorted(corpus_dir.glob("*")):
        if file_path.suffix.lower() in {".md", ".txt"}:
            doc_id = file_path.stem
            text = strip_metadata_comment(file_path.read_text(encoding="utf-8"))
            docs.append((doc_id, text))
    return docs


def _chunk_corpus(
    documents: Sequence[tuple[str, str]],
    strategy: str = "recursive",
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """Chunk documents using specified strategy."""
    all_chunks: list[Chunk] = []
    for doc_id, text in documents:
        if strategy == "fixed":
            chunks = fixed(text, doc_id, size=chunk_size, overlap=overlap)
        elif strategy == "sentence":
            chunks = sentence(text, doc_id, max_tokens=max(1, chunk_size // 5))
        elif strategy == "semantic":
            chunks = semantic(text, doc_id, threshold=0.5)
        else:
            chunks = recursive(text, doc_id, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(chunks)
    return all_chunks


def _save_index(path: Path, chunks: Sequence[Chunk], strategy: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "strategy": strategy,
        "chunks": [
            {"id": c.id, "doc_id": c.doc_id, "text": c.text, "start": c.start, "end": c.end}
            for c in chunks
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _load_index(path: Path) -> list[Chunk]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk(**c) for c in payload["chunks"]]


def cmd_index(args: argparse.Namespace) -> int:
    """Chunk a directory of documents and persist the chunks as a JSON index.

    BM25 statistics and hashing embeddings are rebuilt from the chunks at load time
    (milliseconds for this corpus), so the chunk list is the only persisted state.
    """
    corpus_dir = Path(args.dir)
    docs = _load_documents(corpus_dir)
    print(f"Loaded {len(docs)} documents from {corpus_dir}")

    chunks = _chunk_corpus(
        docs,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(f"Generated {len(chunks)} chunks using strategy '{args.strategy}'")

    out = Path(args.out)
    _save_index(out, chunks, args.strategy)
    print(f"Wrote index to {out}")
    return 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_unanswerable(ds: Dataset, cohort: str = "all") -> list[dict[str, Any]]:
    """Unanswerable records of a dataset, optionally restricted to one ``cohort`` tag."""
    records = _read_jsonl(ds.unanswerable_path)
    if cohort == "all":
        return records
    kept = [r for r in records if r.get("cohort") == cohort]
    if not kept:
        raise SystemExit(f"Error: no unanswerable records with cohort {cohort!r}")
    return kept


def _make_pipeline(retriever: str, min_support: float) -> RAGPipeline:
    """The shipped pipeline, or one on a neural retriever (needs the ``[st]`` extra)."""
    if retriever == "default":
        return RAGPipeline(min_support=min_support)
    from ragkit.neural import NeuralChunkRetriever

    return RAGPipeline(
        retriever=NeuralChunkRetriever(retriever), rerank=False, min_support=min_support
    )


def cmd_ask(args: argparse.Namespace) -> int:
    """Answer a user query with chunk citations, or abstain when support is low."""
    pipeline = _make_pipeline(args.retriever, args.min_support)
    if args.index:
        pipeline.index_chunks(_load_index(Path(args.index)))
    else:
        corpus_dir = Path(args.corpus_dir) if args.corpus_dir else DEFAULT_CORPUS_DIR
        pipeline.index_documents(_load_documents(corpus_dir))

    answer = pipeline.answer(args.query, top_k=args.top_k)
    print(f"Question: {args.query}\n")
    print(f"Answer:\n{answer.text}\n")
    if answer.citations:
        print(f"Citations: {', '.join(answer.citations)}")
    print(f"Support: {answer.support:.2f} (threshold {args.min_support:.2f})")
    return 0


def cmd_abstention(args: argparse.Namespace) -> int:
    """Report answer rate on answerable and abstain rate on unanswerable questions."""
    if args.signals is not None:
        return _cmd_abstention_signals(args)
    ds = _resolve_dataset(args)
    pipeline = RAGPipeline(min_support=args.min_support)
    pipeline.index_documents(_load_documents(ds.corpus_dir))

    qa_items = _read_jsonl(ds.qa_path)
    answerable = [str(item["q"]) for item in qa_items]
    unanswerable = _read_unanswerable(ds, args.cohort)
    answers = [pipeline.answer(q) for q in answerable]
    answered = sum(not a.abstained for a in answers)
    rows = [("answerable (answered)", answered, len(answerable))]
    for qtype in sorted({str(item["type"]) for item in qa_items if "type" in item}):
        flags = [
            not a.abstained
            for a, item in zip(answers, qa_items, strict=True)
            if item.get("type") == qtype
        ]
        rows.append((f"answerable {qtype} (answered)", sum(flags), len(flags)))
    for kind in sorted({str(u["kind"]) for u in unanswerable}):
        qs = [str(u["q"]) for u in unanswerable if u["kind"] == kind]
        abstained = sum(pipeline.answer(q).abstained for q in qs)
        rows.append((f"unanswerable {kind} (abstained)", abstained, len(qs)))
    for miss in sorted({str(u["miss_type"]) for u in unanswerable if "miss_type" in u}):
        qs = [str(u["q"]) for u in unanswerable if u.get("miss_type") == miss]
        abstained = sum(pipeline.answer(q).abstained for q in qs)
        rows.append((f"unanswerable {miss} (abstained)", abstained, len(qs)))

    print(f"\n### Refuse-on-low-confidence (min_support = {args.min_support:.2f})\n")
    print("| Set | Correct | Rate | Wilson 95% CI |")
    print("|---|---|---|---|")
    for name, hit, total in rows:
        lo, hi = wilson_interval(hit, total)
        print(f"| {name} | {hit}/{total} | {hit / max(1, total):.2f} | [{lo:.2f}, {hi:.2f}] |")
    p5 = float(np.quantile([a.support for a in answers], 0.05)) if answers else 0.0
    print(f"\n5th percentile of best-chunk support on the answerable set: {p5:.3f}\n")
    return 0


def _collect_table(
    ds: Dataset,
    cohort: str = "all",
    scorers: ModelScorers | None = None,
    mismatched_context: bool = False,
) -> SignalTable:
    """Score every question of a dataset with all abstention signals."""
    unanswerable = [str(u["q"]) for u in _read_unanswerable(ds, cohort)]
    return collect_signals(
        _load_documents(ds.corpus_dir),
        _read_jsonl(ds.qa_path),
        unanswerable,
        scorers,
        mismatched_context,
    )


def _signed_ci(t: tuple[float, float, float]) -> str:
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def _rate_cell(hit: int, total: int) -> str:
    lo, hi = wilson_interval(hit, total)
    return f"{hit}/{total} ({hit / max(1, total):.2f}) [{lo:.2f}, {hi:.2f}]"


_COVERAGES = (1.0, 0.9, 0.8, 0.7)
ALL_SIGNALS = {**SIGNALS, **MODEL_SIGNALS}


def _combined_groups(names: Sequence[str]) -> list[tuple[str, list[str]]]:
    """Feature sets of the combined logistic models: lexical, model-based, and both."""
    lexical = [n for n in names if n in SIGNALS]
    model = [n for n in names if n in MODEL_SIGNALS]
    groups = []
    if lexical:
        groups.append(("combined", lexical))
    if model:
        groups.append(("combined_model", model))
    if lexical and model:
        groups.append(("combined_all", list(names)))
    return groups


_BREAKDOWN_SIGNALS = ("query_support", "dense_top", "ce_top", "qa_answerable")


def _print_breakdown(ds: Dataset, cohort: str, table: SignalTable, names: Sequence[str]) -> None:
    """AUROC per answerable ``type`` and per unanswerable ``miss_type`` (diagnostic only).

    Each answerable type is scored against all unanswerable questions, each ``miss_type``
    against all answerable ones. ``question_words`` (more words = more answerable) is a
    surface-feature control, not a candidate signal.
    """
    qa = _read_jsonl(ds.qa_path)
    unans = _read_unanswerable(ds, cohort)
    labels = table.answerable
    cols: list[tuple[str, np.ndarray[Any, Any]]] = [
        (n, table.scores[n]) for n in _BREAKDOWN_SIGNALS if n in names
    ]
    words = [len(str(r["q"]).split()) for r in [*qa, *unans]]
    cols.append(("question_words", np.asarray(words, dtype=np.float64)))
    groups: list[tuple[str, np.ndarray[Any, Any]]] = [("all questions", np.ones_like(labels))]
    for kind, tag, records, offset in (
        ("answerable type", "type", qa, 0),
        ("unanswerable miss_type", "miss_type", unans, len(qa)),
    ):
        for value in sorted({str(r[tag]) for r in records if tag in r}):
            mask = np.zeros(len(labels), dtype=np.bool_)
            for i, r in enumerate(records):
                mask[offset + i] = str(r.get(tag)) == value
            groups.append((f"{kind} = {value}", mask | (~labels if offset == 0 else labels)))
    print("Diagnostic AUROC by question group (each group against the whole other class;")
    print("bootstrap 95% CI; groups hold 7 to 20 questions, so these are descriptive):\n")
    header = " | ".join(f"`{n}`" for n, _ in cols)
    print(f"| Group | n (answerable / unanswerable) | {header} |")
    print("|---|---|" + "---|" * len(cols))
    for title, keep in groups:
        n_pos, n_neg = int((keep & labels).sum()), int((keep & ~labels).sum())
        cells = []
        for _, scores in cols:
            a, lo, hi = auroc_ci(scores[keep], labels[keep])
            cells.append(f"{a:.3f} [{lo:.2f}, {hi:.2f}]")
        print(f"| {title} | {n_pos} / {n_neg} | " + " | ".join(cells) + " |")
    print()


def _cmd_abstention_signals(args: argparse.Namespace) -> int:
    """Compare abstention signals: AUROC, risk-coverage, CV and zero-shot thresholds."""
    ds = _resolve_dataset(args)
    if args.signals == "all":
        names = list(SIGNALS) + (list(MODEL_SIGNALS) if args.model_signals else [])
    else:
        names = [s.strip() for s in args.signals.split(",") if s.strip()]
        unknown = [s for s in names if s not in ALL_SIGNALS]
        if unknown:
            print(
                f"Error: unknown signals {unknown}; choose from {list(ALL_SIGNALS)}",
                file=sys.stderr,
            )
            return 2
    use_model = any(n in MODEL_SIGNALS for n in names)
    scorers = (
        load_model_scorers(Path(args.model_cache) if args.model_cache else None, args.model_device)
        if use_model
        else None
    )
    synthetic = BUNDLED_DATASETS["synthetic"]
    control = args.model_control == "mismatched"
    table = _collect_table(ds, args.cohort, scorers, control)
    cal = _collect_table(synthetic, "all", scorers, control) if ds != synthetic else None
    if scorers is not None:
        scorers.save()
    labels = table.answerable
    n_a, n_u = table.n_answerable, table.n_unanswerable
    baseline = table.scores["query_support"]
    combos = {
        label: nested_cv_logistic(
            np.column_stack([table.scores[n] for n in feats]), labels, args.cv_folds, args.cv_seed
        )
        for label, feats in _combined_groups(names)
    }
    columns = [(name, table.scores[name]) for name in names] + [
        (label, res.scores) for label, res in combos.items()
    ]

    print(f"\n### Abstention signals ({n_a} answerable vs {n_u} unanswerable questions)\n")
    for name in names:
        print(f"- `{name}`: {ALL_SIGNALS[name]}")
    print("\nHigher = more likely answerable for every signal; the sign is fixed in advance.\n")
    for label, feats in _combined_groups(names):
        print(f"`{label}` = L2 logistic regression on {', '.join(feats)} (numpy only).")
    print("Combined models are scored on held-out folds only:")
    print(f"{args.cv_folds}-fold stratified nested CV by question (seed {args.cv_seed}); the")
    print("ridge strength and threshold are chosen inside each training fold, so the held-out")
    print("fold never enters a fit.\n")
    if scorers is not None:
        print(f"Model scoring (device {args.model_device}): {scorers.stats()}.\n")
    if control:
        print("CONTROL `--model-control mismatched`: the model signals read another question's")
        print(
            "chunks (the next question of the same class), so only the question is informative.\n"
        )

    print("AUROC, answerable vs unanswerable (stratified bootstrap 95% CI, 2000 resamples).")
    print("`vs chance`: CI above 0.5 = separates the classes. The paired difference vs the")
    print("`query_support` baseline resamples the same questions for both signals.\n")
    print("| Signal | AUROC | vs chance | AUROC - baseline | separated from baseline |")
    print("|---|---|---|---|---|")
    above = 0
    for name, scores in columns:
        a, lo, hi = auroc_ci(scores, labels)
        chance = "above" if lo > 0.5 else "below" if hi < 0.5 else "no"
        above += chance == "above" and name in names
        if name == "query_support":
            diff, sep = "baseline", "-"
        else:
            d = paired_auroc_ci(scores, baseline, labels)
            diff = _signed_ci(d)
            sep = "yes (better)" if d[1] > 0 else "yes (worse)" if d[2] < 0 else "no"
        print(f"| `{name}` | {a:.3f} [{lo:.3f}, {hi:.3f}] | {chance} | {diff} | {sep} |")
    print(f"\n{above} of {len(names)} signals have an AUROC CI entirely above 0.5.\n")
    if control:
        matched = _collect_table(ds, args.cohort, scorers)
        print("Matched minus mismatched context, paired AUROC difference (same questions):\n")
        print("| Signal | matched AUROC | mismatched AUROC | difference |")
        print("|---|---|---|---|")
        for name in (n for n in names if n in MODEL_SIGNALS):
            d = paired_auroc_ci(matched.scores[name], table.scores[name], labels)
            a, c = auroc(matched.scores[name], labels), auroc(table.scores[name], labels)
            print(f"| `{name}` | {a:.3f} | {c:.3f} | {_signed_ci(d)} |")
        print()
    if args.breakdown:
        _print_breakdown(ds, args.cohort, table, names)

    n = len(labels)
    n_miss = int(table.retrieval_miss.sum())
    print(f"Risk-coverage: answer the top share of all {n} questions by each signal.")
    print("Rows are the share of answered questions that are unanswerable (ties at the cut-off")
    print("are split fractionally). 'Bad' = unanswerable, or answerable with no relevant article")
    print(f"in the retrieved top 5 ({n_miss} of {n_a} answerable questions miss).\n")
    for title, bad in (
        ("unanswerable share of answered", ~labels),
        ("bad share of answered (unanswerable or retrieval miss)", ~labels | table.retrieval_miss),
    ):
        print(f"{title}:\n")
        print("| Signal | " + " | ".join(f"{int(c * 100)}%" for c in _COVERAGES) + " |")
        print("|---|" + "---|" * len(_COVERAGES))
        for name, scores in columns:
            cells = [f"{risk_at_coverage(scores, bad, c):.3f}" for c in _COVERAGES]
            print(f"| `{name}` | " + " | ".join(cells) + " |")
        print()

    print(f"Threshold chosen on training folds only ({args.cv_folds}-fold stratified CV by")
    print(f"question, seed {args.cv_seed}; maximises balanced accuracy on the training folds).")
    print("Held-out decisions are pooled over folds. 'Always answer' scores 0.500.\n")
    print(
        "| Signal | answered (answerable) | abstained (unanswerable) | balanced acc. | median t |"
    )
    print("|---|---|---|---|---|")
    for name in names:
        cv = cross_validated_accept(table.scores[name], labels, args.cv_folds, args.cv_seed)
        ya, na, yu, nu = accept_counts(cv.accepted, labels)
        bal = (ya / na + yu / nu) / 2.0
        t = float(np.median(cv.thresholds))
        print(f"| `{name}` | {_rate_cell(ya, na)} | {_rate_cell(yu, nu)} | {bal:.3f} | {t:.3g} |")
    for label, res in combos.items():
        ya, na, yu, nu = accept_counts(res.accepted, labels)
        bal = (ya / na + yu / nu) / 2.0
        t = float(np.median(res.thresholds))
        row = f"{_rate_cell(ya, na)} | {_rate_cell(yu, nu)} | {bal:.3f} | {t:.3g} (logit)"
        print(f"| `{label}` | {row} |")
    print()
    for label, feats in _combined_groups(names):
        res = combos[label]
        print(f"`{label}`: ridge strength chosen per outer fold = {res.l2}.")
        chosen = float(np.median(res.l2))
        full = fit_logistic(np.column_stack([table.scores[n] for n in feats]), labels, l2=chosen)
        coefs = sorted(zip(feats, full.weights, strict=True), key=lambda c: -abs(float(c[1])))
        print(f"In-sample fit on all questions, description only, never scored (l2 = {chosen:g}):")
        print("standardised coefficients " + ", ".join(f"{n} {w:+.2f}" for n, w in coefs) + ".\n")

    if cal is None:
        print("Zero-shot transfer: not applicable (this is the calibration set).\n")
        return 0
    print("Zero-shot transfer: threshold = 5th percentile of each signal on the synthetic")
    print(
        "answerable set (as for `--min-support`; no unanswerable data used), applied unchanged.\n"
    )
    print("| Signal | synthetic t | answered (answerable) | abstained (unanswerable) | bal. acc. |")
    print("|---|---|---|---|---|")
    for name in names:
        t = calibrate_threshold(cal.scores[name][cal.answerable])
        ya, na, yu, nu = accept_counts(table.scores[name] >= t, labels)
        bal = (ya / na + yu / nu) / 2.0
        print(f"| `{name}` | {t:.3f} | {_rate_cell(ya, na)} | {_rate_cell(yu, nu)} | {bal:.3f} |")
    print()
    return 0


def _ablation_stages(min_support: float) -> list[tuple[str, RAGPipeline]]:
    """Cumulative stages: each row adds one component to the previous row."""
    return [
        ("BM25", RAGPipeline(retriever=BM25Index(), rerank=False)),
        ("+ dense (hybrid RRF)", RAGPipeline(retriever=HybridRetriever(), rerank=False)),
        ("+ lexical rerank", RAGPipeline(retriever=HybridRetriever())),
        (
            f"+ refuse (support >= {min_support:.2f})",
            RAGPipeline(retriever=HybridRetriever(), min_support=min_support),
        ),
    ]


def cmd_ablation(args: argparse.Namespace) -> int:
    """Run the full pipeline stage by stage on the same questions and print one table."""
    ds = _resolve_dataset(args)
    docs = _load_documents(ds.corpus_dir)
    answerable = _read_jsonl(ds.qa_path)
    unanswerable = [str(u["q"]) for u in _read_unanswerable(ds, args.cohort)]

    print("\n### Staged ablation (end-to-end, top_k = 5; mean [bootstrap 95% CI over questions])\n")
    print("| Stage | Recall@5 | MRR | Answer F1 | Faithfulness | Answered | Abstain (unans.) |")
    print("|---|---|---|---|---|---|---|")
    for name, pipeline in _ablation_stages(args.min_support):
        pipeline.index_documents(docs)
        r5: list[float] = []
        rr: list[float] = []
        f1: list[float] = []
        faith: list[float] = []
        answered = 0
        for item in answerable:
            q = str(item["q"])
            chunks = pipeline.retrieve(q, top_k=5)
            doc_ids = list(dict.fromkeys(c.doc_id for c in chunks))
            relevant = [str(d) for d in item["relevant_doc_ids"]]
            r5.append(recall_at_k(doc_ids, relevant, k=5))
            rr.append(mrr(doc_ids, relevant))
            ans = pipeline.answer(q, top_k=5)
            f1.append(0.0 if ans.abstained else answer_f1(ans.text, str(item["answer"])))
            if not ans.abstained:
                answered += 1
                faith.append(faithfulness(ans.text, [c.text for c in chunks]))
        abstained = sum(pipeline.answer(q).abstained for q in unanswerable)
        n = len(answerable)
        print(
            f"| {name} | {_ci(r5)} | {_ci(rr)} | {sum(f1) / n:.3f} | "
            f"{sum(faith) / max(1, len(faith)):.3f} | {answered}/{n} | "
            f"{abstained}/{len(unanswerable)} |"
        )
    print()
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate retrieval and reranking strategies on the QA benchmark dataset."""
    ds = _resolve_dataset(args)
    corpus_dir, qa_path = ds.corpus_dir, ds.qa_path

    if not qa_path.is_file():
        print(f"Error: QA benchmark file not found at {qa_path}", file=sys.stderr)
        return 1

    # Load corpus
    docs = _load_documents(corpus_dir)
    chunks = _chunk_corpus(docs, strategy="recursive", chunk_size=500, overlap=50)

    # Build indices
    bm25 = BM25Index()
    bm25.add(chunks)

    embedder = HashingEmbedder(dim=512)
    texts = [c.text for c in chunks]
    embedder.fit(texts)

    dense = DenseIndex(embedder=embedder)
    dense.add(chunks)

    hybrid = HybridRetriever(bm25=bm25, dense=dense)
    reranker = LexicalReranker()
    llm = TemplateLLM()
    hyde = HyDERewriter(llm=llm)

    qa_pairs = _read_jsonl(qa_path)

    names = {
        "bm25": "BM25",
        "dense": "Dense",
        "hybrid": "Hybrid (RRF)",
        "rerank": "Hybrid + Rerank",
    }
    if args.strategies == "all":
        selected = list(names)
    else:
        selected = [s.strip().lower() for s in args.strategies.split(",") if s.strip()]
        unknown = [s for s in selected if s not in names]
        if unknown:
            print(
                f"Error: unknown strategies {unknown}; choose from {list(names)}", file=sys.stderr
            )
            return 2
    strategies = [(names[s], rw) for s in selected for rw in (False, True)]

    # Per-question scores for every (strategy, rewrite) row, in qa_pairs order.
    scores: dict[tuple[str, bool], dict[str, list[float]]] = {}

    for name, use_rewrite in strategies:
        per_q: dict[str, list[float]] = {"r5": [], "mrr": [], "ndcg": []}

        for item in qa_pairs:
            q = str(item["q"])
            relevant_docs = [str(d) for d in item["relevant_doc_ids"]]

            # Query preprocessing with rewrite if enabled
            effective_q = q
            if use_rewrite:
                hypo = hyde.rewrite(q)
                effective_q = f"{q} {hypo}"

            # Retrieve top candidates
            if name == "BM25":
                candidates = bm25.search(effective_q, k=10)
            elif name == "Dense":
                candidates = dense.search(effective_q, k=10)
            elif name == "Hybrid (RRF)":
                candidates = hybrid.search(effective_q, k=10, method="rrf")
            elif name == "Hybrid + Rerank":
                initial = hybrid.search(effective_q, k=20, method="rrf")
                candidates = reranker.rerank(effective_q, initial, top_k=10)
            else:
                candidates = []

            # Map chunk IDs to doc IDs
            retrieved_doc_ids: list[str] = []
            for chunk, _ in candidates:
                if chunk.doc_id not in retrieved_doc_ids:
                    retrieved_doc_ids.append(chunk.doc_id)

            per_q["r5"].append(recall_at_k(retrieved_doc_ids, relevant_docs, k=5))
            per_q["mrr"].append(mrr(retrieved_doc_ids, relevant_docs))
            per_q["ndcg"].append(ndcg_at_k(retrieved_doc_ids, relevant_docs, k=10))

        scores[(name, use_rewrite)] = per_q

    n = len(qa_pairs)
    print(f"\n### RagKit Strategy Benchmark Results ({n} questions; mean [bootstrap 95% CI])\n")
    print("| Strategy | Rewrite | Recall@5 | MRR | nDCG@10 |")
    print("|---|---|---|---|---|")
    for (name, use_rewrite), per_q in scores.items():
        rewrite = "HyDE" if use_rewrite else "None"
        nd = sum(per_q["ndcg"]) / max(1, n)
        print(f"| {name} | {rewrite} | {_ci(per_q['r5'])} | {_ci(per_q['mrr'])} | {nd:.3f} |")
    print()

    base_key = ("BM25", False)
    if base_key in scores and len(scores) > 1:
        print("Paired difference vs BM25 / None (mean [bootstrap 95% CI]); a CI that excludes")
        print("0 means the two methods are separated on this question set.\n")
        print("| Strategy | Rewrite | dRecall@5 | dMRR |")
        print("|---|---|---|---|")
        for (name, use_rewrite), per_q in scores.items():
            if (name, use_rewrite) == base_key:
                continue
            rewrite = "HyDE" if use_rewrite else "None"
            d5 = paired_bootstrap_ci(per_q["r5"], scores[base_key]["r5"])
            dm = paired_bootstrap_ci(per_q["mrr"], scores[base_key]["mrr"])
            print(
                f"| {name} | {rewrite} | {d5[0]:+.3f} [{d5[1]:+.3f}, {d5[2]:+.3f}] | "
                f"{dm[0]:+.3f} [{dm[1]:+.3f}, {dm[2]:+.3f}] |"
            )
        print()

    for use_rewrite in (False, True):
        a, b = ("Hybrid + Rerank", use_rewrite), ("Hybrid (RRF)", use_rewrite)
        if a not in scores or b not in scores:
            continue
        rewrite = "HyDE" if use_rewrite else "None"
        d5 = paired_bootstrap_ci(scores[a]["r5"], scores[b]["r5"])
        dm = paired_bootstrap_ci(scores[a]["mrr"], scores[b]["mrr"])
        print(f"Paired difference, Hybrid + Rerank minus Hybrid (RRF), rewrite {rewrite}:\n")
        print("| dRecall@5 | dMRR |")
        print("|---|---|")
        print(f"| {_signed_ci(d5)} | {_signed_ci(dm)} |")
        print()

    types = sorted({str(item["type"]) for item in qa_pairs if "type" in item})
    if types:
        print("Breakdown by question type (no rewrite): mean Recall@5 / MRR\n")
        print("| Strategy | " + " | ".join(f"{t} (n={_count(qa_pairs, t)})" for t in types) + " |")
        print("|---|" + "---|" * len(types))
        for (name, use_rewrite), per_q in scores.items():
            if use_rewrite:
                continue
            cells = []
            for t in types:
                idx = [i for i, item in enumerate(qa_pairs) if item.get("type") == t]
                r5 = sum(per_q["r5"][i] for i in idx) / len(idx)
                rr = sum(per_q["mrr"][i] for i in idx) / len(idx)
                cells.append(f"{r5:.3f} / {rr:.3f}")
            print(f"| {name} | " + " | ".join(cells) + " |")
        # How much of each question's wording the best relevant chunk already contains.
        overlap = [
            max(
                (
                    query_support(str(item["q"]), c)
                    for c in chunks
                    if c.doc_id in item["relevant_doc_ids"]
                ),
                default=0.0,
            )
            for item in qa_pairs
        ]
        cells = []
        for t in types:
            idx = [i for i, item in enumerate(qa_pairs) if item.get("type") == t]
            cells.append(f"{sum(overlap[i] for i in idx) / len(idx):.3f}")
        print("| *question terms in best relevant chunk* | " + " | ".join(cells) + " |")
        print()
    return 0


def _count(items: Sequence[dict[str, Any]], qtype: str) -> int:
    return sum(1 for item in items if item.get("type") == qtype)


def cmd_lexical(args: argparse.Namespace) -> int:
    """Model-free lexical retrieval comparison (docs/lexical_protocol.md).

    Builds BM25 / BM25F indexes only: no embedder, no cross-encoder, no dense or hybrid index.
    """
    ds = _resolve_dataset(args)
    if not ds.qa_path.is_file():
        print(f"Error: QA benchmark file not found at {ds.qa_path}", file=sys.stderr)
        return 1
    questions = questions_from_records(_read_jsonl(ds.qa_path))
    documents = _load_documents(ds.corpus_dir)
    if args.explain:
        print("\n### RagKit lexical diagnosis (model-free, post hoc; A0 vs A4 top-5 changes)\n")
        print(render_explanation(explain_lexical(documents, questions)))
        return 0
    report = evaluate_lexical(documents, questions)
    print("\n### RagKit lexical retrieval (model-free; dense and hybrid are not run)\n")
    print(render_report(report))
    print(render_decision_inputs(report))
    return 0


def cmd_ctxprefix(args: argparse.Namespace) -> int:
    """Model-free context-prefix experiment (docs/ctxprefix_protocol.md).

    Builds BM25Index, HashingEmbedder, DenseIndex and HybridRetriever only: no sentence-
    transformers embedder, cross-encoder or reranker. ``--protocol`` runs the three bundled
    question sets of the protocol; ``--hard`` runs the hard held-out set against the synthetic
    guard (docs/ctxprefix_hard_protocol.md); ``--nokw`` runs the keyword-free EU set with the
    saturation and floor guards (docs/ctxprefix_nokw_protocol.md); otherwise one set is scored.
    """
    if args.nokw:
        nokw_records = _read_jsonl(EU_NOKW_QA_PATH)
        nokw = questions_from_records(nokw_records)
        synthetic = questions_from_records(_read_jsonl(DEFAULT_QA_PATH))
        eu_docs = _load_documents(EU_CORPUS_DIR)
        print(
            "\n### RagKit context prefix on the keyword-free EU set (model-free; BM25 and hybrid)\n"
        )
        print(render_rule_status(load_index(EU_CORPUS_DIR), nokw_records))
        print(
            render_ctxprefix_nokw_protocol(
                evaluate_ctxprefix(eu_docs, nokw, "keyword-free EU set (qa_eu_nokw.jsonl)"),
                evaluate_ctxprefix(
                    _load_documents(DEFAULT_CORPUS_DIR), synthetic, "synthetic set (qa.jsonl)"
                ),
            )
        )
        return 0
    if args.hard:
        hard = questions_from_records(_read_jsonl(EU_HARD_QA_PATH))
        synthetic = questions_from_records(_read_jsonl(DEFAULT_QA_PATH))
        print("\n### RagKit context prefix on the hard EU set (model-free; BM25 and hybrid)\n")
        print(
            render_ctxprefix_hard_protocol(
                evaluate_ctxprefix(
                    _load_documents(EU_CORPUS_DIR), hard, "hard EU set (qa_eu_hard.jsonl)"
                ),
                evaluate_ctxprefix(
                    _load_documents(DEFAULT_CORPUS_DIR), synthetic, "synthetic set (qa.jsonl)"
                ),
            )
        )
        return 0
    if args.protocol:
        eu_docs = _load_documents(EU_CORPUS_DIR)
        fresh = questions_from_records(_read_jsonl(EU_FRESH_QA_PATH))
        old = questions_from_records(_read_jsonl(EU_QA_PATH))
        synthetic = questions_from_records(_read_jsonl(DEFAULT_QA_PATH))
        print("\n### RagKit context prefix (model-free; BM25 and hashing hybrid only)\n")
        print(
            render_ctxprefix_protocol(
                evaluate_ctxprefix(eu_docs, fresh, "fresh EU set (qa_eu_fresh.jsonl)"),
                evaluate_ctxprefix(eu_docs, old, "old EU set (qa_eu.jsonl)"),
                evaluate_ctxprefix(
                    _load_documents(DEFAULT_CORPUS_DIR), synthetic, "synthetic set (qa.jsonl)"
                ),
                evaluate_ctxprefix(
                    eu_docs, without_act_named(fresh), "fresh EU set without act-named questions"
                ),
            )
        )
        return 0
    ds = _resolve_dataset(args)
    if not ds.qa_path.is_file():
        print(f"Error: QA benchmark file not found at {ds.qa_path}", file=sys.stderr)
        return 1
    questions = questions_from_records(_read_jsonl(ds.qa_path))
    result = evaluate_ctxprefix(_load_documents(ds.corpus_dir), questions, ds.qa_path.name)
    print("\n### RagKit context prefix (model-free; BM25 and hashing hybrid only)\n")
    print(render_ctxprefix_set(result))
    return 0


DEFAULT_EMBED_DIR = Path("data/raw/emb")


def _add_neural_args(parser: argparse.ArgumentParser) -> None:
    """``--retriever`` and its model options (docs/neural_protocol.md)."""
    parser.add_argument(
        "--retriever",
        choices=("bm25", "dense", "hybrid", "hybrid+ce"),
        default="bm25",
        help="bm25 = the model-free systems; dense, hybrid, hybrid+ce need `[st]` (neural).",
    )
    parser.add_argument(
        "--cache-dir", default=str(DEFAULT_EMBED_DIR), help="Gitignored document-matrix cache."
    )
    parser.add_argument("--device", default=None, help="cpu or mps (default: mps if available).")


def _cmd_beir_neural(args: argparse.Namespace) -> int:
    """Neural systems on the BEIR sets of docs/neural_protocol.md."""
    from ragkit.neuralbench import (
        NEURAL_BEIR,
        render_beir_mean,
        render_neural_beir,
        run_beir_neural,
    )

    names = list(NEURAL_BEIR) if args.dataset == "all" else [args.dataset]
    runs = []
    for name in names:
        if name not in NEURAL_BEIR:
            print(f"Error: neural protocol covers {list(NEURAL_BEIR)}, not {name}", file=sys.stderr)
            return 2
        run = run_beir_neural(
            name,
            Path(args.data_dir),
            args.retriever,
            cache_dir=Path(args.cache_dir),
            device=args.device,
            max_queries=args.max_queries,
            download=not args.no_download,
        )
        runs.append(run)
        print()
        print(render_neural_beir(run))
    if len(runs) > 1:
        print()
        print(render_beir_mean(runs))
    return 0


def _cmd_legalbench_neural(args: argparse.Namespace) -> int:
    """Neural systems on LegalBench-RAG-mini (RRF only: no dev split)."""
    from ragkit.neuralbench import run_legal_neural

    bench = load_mini(Path(args.data_zip), download=not args.no_download)
    results = run_legal_neural(
        bench,
        args.retriever,
        cache_dir=Path(args.cache_dir),
        device=args.device,
        max_queries=args.max_queries,
    )
    print()
    title = "### ragkit legalbench: LegalBench-RAG-mini (neural: bge-small, e5-small-v2, MiniLM CE)"
    print(render_legalbench_report(bench, results, title))
    return 0


def cmd_neural(args: argparse.Namespace) -> int:
    """Neural systems on the original-42 (dev) and the fresh, hard and keyword-free EU sets."""
    from ragkit.neuralbench import render_eu_neural, run_eu_neural

    def load(path: Path) -> list[Question]:
        return questions_from_records(_read_jsonl(path))

    run = run_eu_neural(
        _load_documents(EU_CORPUS_DIR),
        load(EU_QA_PATH),
        {
            "fresh 29": load(EU_FRESH_QA_PATH),
            "hard 39": load(EU_HARD_QA_PATH),
            "keyword-free 49": load(EU_NOKW_QA_PATH),
        },
        args.retriever,
        cache_dir=Path(args.cache_dir),
        device=args.device,
    )
    print()
    print(render_eu_neural(run))
    return 0


def cmd_beir(args: argparse.Namespace) -> int:
    """Retrievers on a BEIR test set (docs/beir_protocol.md); nDCG@10 and Recall@100."""
    if args.retriever != "bm25":
        return _cmd_beir_neural(args)
    names = list(BEIR_DATASETS) if args.dataset == "all" else [args.dataset]
    by_name = {spec.name: spec for spec in SYSTEMS}
    if args.systems == "all":
        specs: list[SystemSpec] = list(SYSTEMS)
    else:
        unknown = [n for n in args.systems.split(",") if n not in by_name]
        if unknown:
            print(
                f"Error: unknown systems {unknown}; choose from {sorted(by_name)}", file=sys.stderr
            )
            return 2
        specs = [by_name[n] for n in args.systems.split(",")]
    if args.ladder:
        specs = list(LADDER)
    for name in names:
        data = load_beir(name, Path(args.data_dir), split="test", download=not args.no_download)
        runnable = [s for s in specs if data.has_titles or not s.needs_titles]
        skipped = [s.name for s in specs if s not in runnable]
        results = [evaluate_system(s, data, max_queries=args.max_queries) for s in runnable]
        title = f"ragkit beir: {name} (test, model-free)"
        print()
        print(render_beir_report(data, results, title=title))
        if skipped:
            print(f"\n(no document titles in {name}; skipped: {', '.join(skipped)})")
    return 0


def cmd_legalbench(args: argparse.Namespace) -> int:
    """Retrievers on LegalBench-RAG-mini (docs/legalbench_protocol.md)."""
    if args.retriever != "bm25":
        return _cmd_legalbench_neural(args)
    by_name = {s.name: s for s in LEGAL_SYSTEMS}
    if args.systems == "all":
        specs = list(LEGAL_SYSTEMS)
    else:
        unknown = [n for n in args.systems.split(",") if n not in by_name]
        if unknown:
            print(
                f"Error: unknown systems {unknown}; choose from {sorted(by_name)}", file=sys.stderr
            )
            return 2
        specs = [by_name[n] for n in args.systems.split(",")]
    bench = load_mini(Path(args.data_zip), download=not args.no_download)
    results = [evaluate_legal(s, bench, max_queries=args.max_queries) for s in specs]
    print()
    print(render_legalbench_report(bench, results))
    return 0


DEFAULT_LLM_CACHE = Path(".ragkit") / "llm_cache.jsonl"
GENERATION_CACHE = Path("data/llm_cache/generation.jsonl")
READER_CACHE = Path("data/llm_cache/reader_scores.json")
READER_CACHE_HCE = Path("data/llm_cache/reader_scores_hybrid_ce.json")


def _llm_client(model: str, cache: str) -> CachedMessages:
    """A cached client; live calls need ``ANTHROPIC_API_KEY``, replays need nothing."""
    return CachedMessages(Path(cache), api_key=os.environ.get("ANTHROPIC_API_KEY", ""), model=model)


def _add_context_args(parser: argparse.ArgumentParser, default: str) -> None:
    """``--context`` and the frozen parent-document parameters (docs/generation_protocol.md)."""
    parser.add_argument(
        "--context",
        choices=("chunk", "article"),
        default=default,
        help="chunk: the retrieved chunks; article: the full text of their distinct parent "
        f"articles in rank order (parent-document / small-to-big, Phase L). Default: {default}.",
    )
    parser.add_argument(
        "--articles",
        type=int,
        default=ARTICLE_K,
        help="--context article: number of distinct parent articles (frozen: 3).",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=ARTICLE_BUDGET_TOKENS,
        help="--context article: token budget of the articles, chars/4 (frozen: 12000).",
    )


def _reader_cache_path(args: argparse.Namespace) -> Path | None:
    """The reader-score file of a generation run (hybrid+ce has one only for Phase L)."""
    if args.reader_cache:
        return Path(args.reader_cache)
    if args.retriever == "default":
        return READER_CACHE
    if args.context == "article" or args.build_reader_cache:
        return READER_CACHE_HCE
    return None


def cmd_answer(args: argparse.Namespace) -> int:
    """Retrieve, optionally gate on sufficient context, then answer with article citations."""
    ds = _resolve_dataset(args)
    pipeline = _make_pipeline(args.retriever, 0.0)
    documents = _load_documents(ds.corpus_dir)
    pipeline.index_documents(documents)
    passages = [Passage(c.doc_id, c.text) for c in pipeline.retrieve(args.query, args.top_k)]
    context = (
        parent_articles(passages, dict(documents), k=args.articles, budget_tokens=args.budget)
        if args.context == "article"
        else passages
    )
    answerer = LLMAnswerer(_llm_client(args.llm, args.cache))
    print(f"Question: {args.query}\n")
    # The sufficiency check keeps seeing the retrieved chunks, as in the K-R2 experiment.
    suff = answerer.sufficiency(args.query, passages) if args.gate == "suff" else None
    if suff is not None:
        print(f"Sufficient context: {suff.sufficient} (confidence {suff.confidence:.0f}/100)")
        print(f"  {suff.reasoning}\n")
    answer = answerer.generate(args.query, context)
    abstain = (suff is not None and not suff.sufficient) or (
        args.gate == "self" and not answer.supports
    )
    if abstain:
        print(f"Answer:\n{ABSTAIN_TEXT}")
        return 0
    print(f"Answer:\n{answer.answer}\n")
    if answer.citations:
        print(f"Citations: {', '.join(answer.citations)}")
    print(f"Model says the context supports the answer: {answer.supports}")
    return 0


def cmd_generation(args: argparse.Namespace) -> int:
    """K-R2 on the 42 + 45 EU questions (docs/generation_protocol.md); replays the cache.

    ``--context article`` is Phase L: the same questions, answered from whole parent articles.
    """
    ds = BUNDLED_DATASETS["eu"]
    documents = _load_documents(ds.corpus_dir)
    items = items_from_records(_read_jsonl(ds.qa_path), _read_jsonl(ds.unanswerable_path))
    pipeline = _make_pipeline(args.retriever, 0.0)
    passages = retrieve_passages(pipeline, documents, items, top_k=5)
    reader_path = _reader_cache_path(args)
    if args.build_reader_cache:
        assert reader_path is not None
        if args.compute_reader:
            scorers = load_model_scorers(Path(args.build_reader_cache))
            n = build_reader_subset(reader_path, items, passages, scorers)
        else:
            n = subset_score_cache(Path(args.build_reader_cache), reader_path, items, passages)
        print(f"Wrote {n} reader scores to {reader_path}")
        return 0
    article = args.context == "article"
    if article and args.no_fullcontext:
        print("Error: --context article needs the full-context records", file=sys.stderr)
        return 2
    client = CachedMessages(
        Path(args.cache),
        api_key=os.environ.get("ANTHROPIC_API_KEY", "") if args.live else "",
    )
    answerer = LLMAnswerer(client)
    records = run_generation(
        items,
        passages,
        documents,
        answerer,
        full_context=not args.no_fullcontext,
        workers=args.workers,
    )
    reader = (
        reader_scores(offline_reader_scorers(reader_path), items, passages)
        if reader_path is not None
        else None
    )
    decisions = decide(records, reader)
    if article:
        if reader is None:
            print("Error: --context article needs reader scores (--reader-cache)", file=sys.stderr)
            return 2
        article_records = run_article_generation(
            records,
            documents,
            answerer,
            cite_mask(decisions).tolist(),
            k=args.articles,
            budget_tokens=args.budget,
            workers=args.workers,
        )
        title = f"ragkit generation --context article: parent articles ({args.retriever})"
        print()
        print(
            render_article_report(
                records,
                article_records,
                decisions,
                client,
                answerer,
                documents,
                title=title,
                k=args.articles,
                budget_tokens=args.budget,
                l1_threshold=30 if args.retriever == "default" else 32,
            )
        )
        return 0
    title = f"ragkit generation: EU answers, sufficient-context gates (retriever {args.retriever})"
    print()
    print(render_generation(records, decisions, client, title=title))
    return 0


def cmd_fetch_eu(args: argparse.Namespace) -> int:
    """Download the EU legal acts and rebuild the article-level corpus (needs network)."""
    doc_ids = build_corpus(
        EU_ACTS,
        Path(args.out),
        cache_dir=Path(args.cache),
        retrieved_on=args.retrieved_on,
    )
    print(f"Wrote {len(doc_ids)} article documents to {args.out}")
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="ragkit",
        description="RagKit: Modular RAG pipeline from scratch with hybrid search and reranking.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Index command
    p_index = subparsers.add_parser("index", help="Index a folder of documents.")
    p_index.add_argument("dir", help="Path to documents directory.")
    p_index.add_argument(
        "--strategy",
        choices=["fixed", "recursive", "sentence", "semantic"],
        default="recursive",
        help="Chunking strategy.",
    )
    p_index.add_argument("--chunk-size", type=int, default=500, help="Chunk character size.")
    p_index.add_argument("--overlap", type=int, default=50, help="Chunk overlap in characters.")
    p_index.add_argument(
        "--out", default=".ragkit/index.json", help="Where to write the chunk index."
    )

    # Ask command
    p_ask = subparsers.add_parser("ask", help="Query the RAG pipeline.")
    p_ask.add_argument("query", help="Question text.")
    _add_dataset_args(p_ask, questions=False)
    p_ask.add_argument(
        "--index", default=None, help="Chunk index written by `ragkit index` (overrides corpus)."
    )
    p_ask.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks.")
    p_ask.add_argument(
        "--retriever",
        choices=("default", "dense", "hybrid", "hybrid+ce"),
        default="default",
        help="default = shipped hashing hybrid + lexical rerank; the others need `[st]`.",
    )
    p_ask.add_argument(
        "--min-support",
        type=float,
        default=DEFAULT_MIN_SUPPORT,
        help="Abstain when no retrieved chunk covers this share of the query's content terms.",
    )

    # Abstention command
    p_abs = subparsers.add_parser(
        "abstention", help="Measure refusal on unanswerable vs answerable questions."
    )
    _add_dataset_args(p_abs)
    p_abs.add_argument("--min-support", type=float, default=DEFAULT_MIN_SUPPORT)
    p_abs.add_argument(
        "--signals",
        default=None,
        help="all, or comma-separated abstention signals: compare them by AUROC / risk-coverage.",
    )
    p_abs.add_argument(
        "--model-signals",
        action="store_true",
        help="with --signals all: add the model-based signals ce_top and qa_answerable "
        "(needs the [st] extra; names given explicitly to --signals also enable them).",
    )
    p_abs.add_argument(
        "--model-cache",
        default=str(DEFAULT_CACHE),
        help="JSON cache of model scores (gitignored); pass '' to disable.",
    )
    p_abs.add_argument(
        "--model-control",
        choices=("none", "mismatched"),
        default="none",
        help="mismatched: score each question against another question's chunks (control).",
    )
    p_abs.add_argument(
        "--breakdown",
        action="store_true",
        help="also print the AUROC per answerable type / unanswerable miss_type (diagnostic).",
    )
    p_abs.add_argument(
        "--model-device", default="cpu", help="torch device of the model signals (cpu, mps)."
    )
    p_abs.add_argument("--cv-folds", type=int, default=5, help="Folds for threshold CV.")
    p_abs.add_argument("--cv-seed", type=int, default=0, help="Seed of the CV fold assignment.")

    # Ablation command
    p_abl = subparsers.add_parser("ablation", help="Staged end-to-end ablation table.")
    _add_dataset_args(p_abl)
    p_abl.add_argument("--min-support", type=float, default=DEFAULT_MIN_SUPPORT)

    # Eval command
    p_eval = subparsers.add_parser("eval", help="Evaluate retrieval and reranking strategies.")
    p_eval.add_argument(
        "--strategies",
        default="all",
        help="all, or comma-separated subset of bm25,dense,hybrid,rerank.",
    )
    _add_dataset_args(p_eval)

    # Lexical command
    p_lex = subparsers.add_parser(
        "lexical", help="Model-free lexical retrieval comparison (structure, BM25F, expansion)."
    )
    _add_dataset_args(p_lex, questions=False)
    p_lex.add_argument("--qa-file", default=None, help="Override the answerable QA jsonl.")
    p_lex.add_argument(
        "--explain",
        action="store_true",
        help="Print the post-hoc diagnosis of the A0 vs A4 Recall@5 changes instead of the report.",
    )

    # Context prefix command
    p_ctx = subparsers.add_parser(
        "ctxprefix",
        help="Model-free context prefix (act title and article heading on every chunk).",
    )
    _add_dataset_args(p_ctx, questions=False)
    p_ctx.add_argument("--qa-file", default=None, help="Override the answerable QA jsonl.")
    p_ctx.add_argument(
        "--hard",
        action="store_true",
        help="Run the hard held-out EU set with the synthetic guard, saturation guard and rule.",
    )
    p_ctx.add_argument(
        "--nokw",
        action="store_true",
        help="Run the keyword-free EU set with the synthetic guard, saturation and floor guards.",
    )
    p_ctx.add_argument(
        "--protocol",
        action="store_true",
        help="Run the protocol's three bundled question sets and print the decision rule.",
    )

    # BEIR command
    p_beir = subparsers.add_parser(
        "beir",
        help="Model-free retrievers on a BEIR test set: nDCG@10 and Recall@100 vs Pyserini.",
    )
    p_beir.add_argument(
        "--dataset",
        default="scifact",
        choices=[*BEIR_DATASETS, "all"],
        help="BEIR dataset (test split); the zip is downloaded to --data-dir when missing.",
    )
    p_beir.add_argument(
        "--systems",
        default="all",
        help="all, or comma-separated system ids from docs/beir_protocol.md.",
    )
    p_beir.add_argument(
        "--ladder",
        action="store_true",
        help="run the analyzer / k1,b diagnostic ladder L0-L5 instead of the systems.",
    )
    p_beir.add_argument("--data-dir", default=str(DEFAULT_BEIR_DIR), help="Gitignored zip cache.")
    p_beir.add_argument("--no-download", action="store_true", help="Fail if the zip is missing.")
    _add_neural_args(p_beir)
    p_beir.add_argument(
        "--max-queries", type=int, default=None, help="Score only the first N queries (smoke test)."
    )

    # LegalBench-RAG command
    p_lbr = subparsers.add_parser(
        "legalbench",
        help="Model-free retrievers on LegalBench-RAG-mini: character precision/recall at k.",
    )
    p_lbr.add_argument(
        "--systems",
        default="all",
        help="all, or comma-separated system ids from docs/legalbench_protocol.md.",
    )
    p_lbr.add_argument(
        "--data-zip",
        default=str(DEFAULT_LEGALBENCH_ZIP),
        help="The authors' Dropbox archive (downloaded when missing, gitignored).",
    )
    p_lbr.add_argument("--no-download", action="store_true", help="Fail if the zip is missing.")
    _add_neural_args(p_lbr)
    p_lbr.add_argument(
        "--max-queries", type=int, default=None, help="Score only N queries per source (smoke)."
    )

    # Neural EU command
    p_neural = subparsers.add_parser(
        "neural",
        help="Neural dense / hybrid / cross-encoder retrieval on the four EU question sets.",
    )
    p_neural.add_argument(
        "--retriever",
        choices=("dense", "hybrid", "hybrid+ce"),
        default="hybrid+ce",
        help="Systems reported: dense = bm25, dense; hybrid adds fusion; hybrid+ce adds rerank.",
    )
    p_neural.add_argument("--cache-dir", default=str(DEFAULT_EMBED_DIR))
    p_neural.add_argument("--device", default=None, help="cpu or mps (default: mps if available).")

    # LLM answer command
    p_ans = subparsers.add_parser(
        "answer",
        help="LLM answer with article citations and a sufficient-context gate (needs an API key).",
    )
    p_ans.add_argument("query", help="Question text.")
    _add_dataset_args(p_ans, questions=False)
    p_ans.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks.")
    _add_context_args(p_ans, "article")  # adopted by the Phase L decision rule
    p_ans.add_argument(
        "--retriever",
        choices=("default", "dense", "hybrid", "hybrid+ce"),
        default="default",
        help="default = shipped hashing hybrid + lexical rerank; the others need `[st]`.",
    )
    p_ans.add_argument("--llm", default=MODEL, help="Anthropic model id (ANTHROPIC_API_KEY).")
    p_ans.add_argument(
        "--gate",
        choices=("suff", "self", "none"),
        default="none",
        help="none (default: the sufficiency gate did not beat the reader gate, see README): "
        "always answer; suff: abstain unless the sufficient-context check passes; self: abstain "
        "when the answer says the context does not support it.",
    )
    p_ans.add_argument("--cache", default=str(DEFAULT_LLM_CACHE), help="JSONL call cache.")

    # Generation experiment command
    p_gen = subparsers.add_parser(
        "generation",
        help="K-R2: LLM answers, sufficient-context gates and full-context baseline on the EU set.",
    )
    p_gen.add_argument(
        "--retriever",
        choices=("default", "hybrid+ce"),
        default="default",
        help="default = shipped pipeline (reader signal available); hybrid+ce = secondary run.",
    )
    p_gen.add_argument("--cache", default=str(GENERATION_CACHE), help="JSONL call cache.")
    _add_context_args(p_gen, "chunk")  # the K-R2 replay; `--context article` is Phase L
    p_gen.add_argument(
        "--compute-reader",
        action="store_true",
        help="With --build-reader-cache: run the SQuAD2 reader on the retrieved chunks (needs "
        "`[st]`) and write only the qa_answerable scores, instead of copying from a score cache.",
    )
    p_gen.add_argument(
        "--reader-cache",
        default=None,
        help="Reader score cache (default: data/llm_cache/reader_scores.json for the default "
        "retriever; the hybrid+ce file with --context article).",
    )
    p_gen.add_argument(
        "--live", action="store_true", help="Allow API calls (needs ANTHROPIC_API_KEY)."
    )
    p_gen.add_argument("--no-fullcontext", action="store_true", help="Skip the full-context run.")
    p_gen.add_argument("--workers", type=int, default=6, help="Concurrent API calls.")
    p_gen.add_argument(
        "--build-reader-cache",
        default=None,
        help="Copy the needed reader scores from this score cache into --reader-cache and exit "
        "(with --compute-reader, this file is the score cache the reader fills).",
    )

    # Fetch-EU command
    p_fetch = subparsers.add_parser(
        "fetch-eu", help="Download EUR-Lex acts and rebuild the article-level EU corpus."
    )
    p_fetch.add_argument("--out", default=str(EU_CORPUS_DIR), help="Output corpus directory.")
    p_fetch.add_argument(
        "--cache", default="data/raw/eurlex", help="Gitignored cache for the downloaded XHTML."
    )
    p_fetch.add_argument(
        "--retrieved-on",
        default=date.today().isoformat(),
        help="Retrieval date recorded in SOURCES.json (default: today).",
    )

    parsed = parser.parse_args(argv)

    if parsed.subcommand == "index":
        sys.exit(cmd_index(parsed))
    elif parsed.subcommand == "ask":
        sys.exit(cmd_ask(parsed))
    elif parsed.subcommand == "eval":
        sys.exit(cmd_eval(parsed))
    elif parsed.subcommand == "lexical":
        sys.exit(cmd_lexical(parsed))
    elif parsed.subcommand == "ctxprefix":
        sys.exit(cmd_ctxprefix(parsed))
    elif parsed.subcommand == "beir":
        sys.exit(cmd_beir(parsed))
    elif parsed.subcommand == "legalbench":
        sys.exit(cmd_legalbench(parsed))
    elif parsed.subcommand == "abstention":
        sys.exit(cmd_abstention(parsed))
    elif parsed.subcommand == "ablation":
        sys.exit(cmd_ablation(parsed))
    elif parsed.subcommand == "fetch-eu":
        sys.exit(cmd_fetch_eu(parsed))
    elif parsed.subcommand == "neural":
        sys.exit(cmd_neural(parsed))
    elif parsed.subcommand == "answer":
        sys.exit(cmd_answer(parsed))
    elif parsed.subcommand == "generation":
        sys.exit(cmd_generation(parsed))


if __name__ == "__main__":
    main()
