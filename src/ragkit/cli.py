"""Command-line interface for RagKit: indexing, question answering, and strategy evaluation."""

from __future__ import annotations

import argparse
import json
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
    auroc_ci,
    calibrate_threshold,
    collect_signals,
    cross_validated_accept,
    fit_logistic,
    nested_cv_logistic,
    paired_auroc_ci,
    risk_at_coverage,
)
from ragkit.bm25 import BM25Index
from ragkit.chunking import Chunk, fixed, recursive, semantic, sentence
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
from ragkit.hybrid import HybridRetriever
from ragkit.modelsignals import DEFAULT_CACHE, MODEL_SIGNALS, ModelScorers, load_model_scorers
from ragkit.pipeline import DEFAULT_MIN_SUPPORT, RAGPipeline, query_support
from ragkit.rerank import LexicalReranker
from ragkit.rewrite import HyDERewriter, TemplateLLM

DEFAULT_DATA_DIR = Path(__file__).parent / "data"
DEFAULT_CORPUS_DIR = DEFAULT_DATA_DIR / "corpus"
DEFAULT_QA_PATH = DEFAULT_DATA_DIR / "qa.jsonl"
DEFAULT_UNANSWERABLE_PATH = DEFAULT_DATA_DIR / "unanswerable.jsonl"
EU_CORPUS_DIR = DEFAULT_DATA_DIR / "corpus_eu"
EU_QA_PATH = DEFAULT_DATA_DIR / "qa_eu.jsonl"
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


def cmd_ask(args: argparse.Namespace) -> int:
    """Answer a user query with chunk citations, or abstain when support is low."""
    pipeline = RAGPipeline(min_support=args.min_support)
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
    groups: list[tuple[str, np.ndarray[Any, Any]]] = []
    for kind, tag, records, offset in (
        ("answerable type", "type", qa, 0),
        ("unanswerable miss_type", "miss_type", unans, len(qa)),
    ):
        for value in sorted({str(r[tag]) for r in records if tag in r}):
            mask = np.zeros(len(labels), dtype=np.bool_)
            for i, r in enumerate(records):
                mask[offset + i] = str(r.get(tag)) == value
            other = ~labels if offset == 0 else labels
            groups.append((f"{kind} = {value}", mask | other))
    print("Diagnostic AUROC by question group (each group against the whole other class;")
    print("bootstrap 95% CI; groups hold 7 to 20 questions, so these are descriptive):\n")
    print("| Group | n (group / other class) | " + " | ".join(f"`{n}`" for n, _ in cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for title, keep in groups:
        n_group = int((keep & (labels == (title.startswith("answerable")))).sum())
        n_other = int(keep.sum()) - n_group
        cells = []
        for _, scores in cols:
            a, lo, hi = auroc_ci(scores[keep], labels[keep])
            cells.append(f"{a:.3f} [{lo:.2f}, {hi:.2f}]")
        print(f"| {title} | {n_group} / {n_other} | " + " | ".join(cells) + " |")
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
    elif parsed.subcommand == "abstention":
        sys.exit(cmd_abstention(parsed))
    elif parsed.subcommand == "ablation":
        sys.exit(cmd_ablation(parsed))
    elif parsed.subcommand == "fetch-eu":
        sys.exit(cmd_fetch_eu(parsed))


if __name__ == "__main__":
    main()
