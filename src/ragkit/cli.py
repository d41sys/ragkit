"""Command-line interface for RagKit: indexing, question answering, and strategy evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ragkit.bm25 import BM25Index
from ragkit.chunking import Chunk, fixed, recursive, semantic, sentence
from ragkit.dense import DenseIndex
from ragkit.embed import HashingEmbedder
from ragkit.eval import mrr, ndcg_at_k, recall_at_k
from ragkit.hybrid import HybridRetriever
from ragkit.pipeline import RAGPipeline
from ragkit.rerank import LexicalReranker
from ragkit.rewrite import HyDERewriter, TemplateLLM

DEFAULT_DATA_DIR = Path(__file__).parent / "data"
DEFAULT_CORPUS_DIR = DEFAULT_DATA_DIR / "corpus"
DEFAULT_QA_PATH = DEFAULT_DATA_DIR / "qa.jsonl"


def _load_documents(corpus_dir: Path) -> list[tuple[str, str]]:
    """Load all markdown and text documents from a directory."""
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    docs: list[tuple[str, str]] = []
    for file_path in sorted(corpus_dir.glob("*")):
        if file_path.suffix.lower() in {".md", ".txt"}:
            doc_id = file_path.stem
            text = file_path.read_text(encoding="utf-8")
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


def cmd_index(args: argparse.Namespace) -> int:
    """Index a directory of documents and report metrics."""
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

    bm25 = BM25Index()
    bm25.add(chunks)

    dense = DenseIndex()
    dense.add(chunks)

    print(f"Successfully indexed {len(chunks)} chunks into BM25 and Dense indices.")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """Answer a user query with chunk citations."""
    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else DEFAULT_CORPUS_DIR
    docs = _load_documents(corpus_dir)

    pipeline = RAGPipeline()
    pipeline.index_documents(docs)

    answer = pipeline.answer(args.query, top_k=args.top_k)
    print(f"Question: {args.query}\n")
    print(f"Answer:\n{answer.text}\n")
    if answer.citations:
        print(f"Citations: {', '.join(answer.citations)}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate retrieval and reranking strategies on the QA benchmark dataset."""
    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else DEFAULT_CORPUS_DIR
    qa_path = Path(args.qa_file) if args.qa_file else DEFAULT_QA_PATH

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

    # Load QA pairs
    qa_pairs: list[dict[str, Any]] = []
    with qa_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qa_pairs.append(json.loads(line))

    strategies = [
        ("BM25", False),
        ("BM25", True),
        ("Dense", False),
        ("Dense", True),
        ("Hybrid (RRF)", False),
        ("Hybrid (RRF)", True),
        ("Hybrid + Rerank", False),
        ("Hybrid + Rerank", True),
    ]

    results: list[tuple[str, str, float, float, float]] = []

    for name, use_rewrite in strategies:
        recall_list: list[float] = []
        mrr_list: list[float] = []
        ndcg_list: list[float] = []

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

            r5 = recall_at_k(retrieved_doc_ids, relevant_docs, k=5)
            rr = mrr(retrieved_doc_ids, relevant_docs)
            n10 = ndcg_at_k(retrieved_doc_ids, relevant_docs, k=10)

            recall_list.append(r5)
            mrr_list.append(rr)
            ndcg_list.append(n10)

        mean_recall = sum(recall_list) / max(1, len(recall_list))
        mean_mrr = sum(mrr_list) / max(1, len(mrr_list))
        mean_ndcg = sum(ndcg_list) / max(1, len(ndcg_list))

        results.append((name, "HyDE" if use_rewrite else "None", mean_recall, mean_mrr, mean_ndcg))

    # Print markdown table
    print("\n### RagKit Strategy Benchmark Results\n")
    print(
        f"| {'Strategy':<18} | {'Rewrite':<8} | {'Recall@5':<10} | {'MRR':<10} | {'nDCG@10':<10} |"
    )
    print(f"|{'-' * 20}|{'-' * 10}|{'-' * 12}|{'-' * 12}|{'-' * 12}|")
    for strategy, rewrite, r5, rr, n10 in results:
        print(f"| {strategy:<18} | {rewrite:<8} | {r5:<10.4f} | {rr:<10.4f} | {n10:<10.4f} |")
    print()
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

    # Ask command
    p_ask = subparsers.add_parser("ask", help="Query the RAG pipeline.")
    p_ask.add_argument("query", help="Question text.")
    p_ask.add_argument(
        "--corpus-dir",
        default=None,
        help="Custom corpus directory (defaults to bundled energy corpus).",
    )
    p_ask.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks.")

    # Eval command
    p_eval = subparsers.add_parser("eval", help="Evaluate retrieval and reranking strategies.")
    p_eval.add_argument(
        "--strategies",
        default="all",
        help="Strategies to evaluate ('all' or comma-separated list).",
    )
    p_eval.add_argument("--corpus-dir", default=None, help="Path to corpus directory.")
    p_eval.add_argument("--qa-file", default=None, help="Path to qa.jsonl benchmark file.")

    parsed = parser.parse_args(argv)

    if parsed.subcommand == "index":
        sys.exit(cmd_index(parsed))
    elif parsed.subcommand == "ask":
        sys.exit(cmd_ask(parsed))
    elif parsed.subcommand == "eval":
        sys.exit(cmd_eval(parsed))


if __name__ == "__main__":
    main()
