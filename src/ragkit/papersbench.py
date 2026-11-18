"""Research papers as a corpus (P-1): questions, scoring and verdicts (docs/papers_protocol.md).

B0 packs whole documents (the first one cut to the budget around its best chunk), W packs windows of
at most ``WINDOW_TOKENS`` around each paper's best chunk, WC is W with a Qwen context prepended to
every chunk for retrieval only. The primary metric is evidence recall: the share of a question's
gold quotes that appear in the packed context.
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragkit.eval import bootstrap_ci, paired_bootstrap_ci
from ragkit.parent import (
    ARTICLE_BUDGET_TOKENS,
    PACK_CHUNKS,
    PACK_OVERFLOW,
    Passage,
    estimate_tokens,
    pack_articles,
)

DATA_DIR = Path(__file__).parent / "data"
SOURCES_PATH = DATA_DIR / "papers_forecasting.json"
QA_PATH = DATA_DIR / "qa_papers_forecasting.jsonl"
CORPUS_DIR = Path("data/raw/papers/forecasting")  # gitignored: arXiv PDFs are not redistributable
CONTEXTS_PATH = Path("docs/sota-runs/papers_contexts.json")
WINDOW_TOKENS = 4000  # above the longest EU article (3 314 tokens): W equals B0 on the EU corpus
CONTEXT_MODEL = "qwen3:8b"  # amendment 2: the 27B model did not fit the shared GPUs at 32k context
CONTEXT_DOC_CHARS = 96_000  # the document the model sees, about 24 000 estimated tokens
CONTEXT_NUM_CTX = 32_768
TYPES = ("named", "described", "cross")
EVIDENCE_WORDS = (10, 60)
CROSS_PAPERS = (2, 4)
N_RESAMPLES = 10_000
METRICS = ("evidence", "papers", "mrr", "tokens", "packed")
LABELS = {
    "evidence": "evidence recall",
    "papers": "paper recall",
    "mrr": "MRR (paper)",
    "tokens": "tokens",
    "packed": "papers packed",
}


@dataclass(frozen=True)
class Gold:
    paper: str
    evidence: str


@dataclass(frozen=True)
class PaperQuestion:
    id: str
    type: str
    text: str
    gold: tuple[Gold, ...]


def normalize(text: str) -> str:
    """Lowercase with every whitespace run (page breaks included) collapsed to one space."""
    return " ".join(text.lower().split())


def load_questions(path: Path = QA_PATH) -> list[PaperQuestion]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            gold = tuple(Gold(g["paper"], g["evidence"]) for g in r["gold"])
            out.append(PaperQuestion(r["id"], r["type"], r["question"], gold))
    return out


def validate(questions: Sequence[PaperQuestion], documents: Mapping[str, str]) -> list[str]:
    """Every violation of the protocol's question rules; empty when the set is valid."""
    errors = []
    texts = {d: normalize(t) for d, t in documents.items()}
    if len({q.id for q in questions}) != len(questions):
        errors.append("question ids are not unique")
    for q in questions:
        if q.type not in TYPES:
            errors.append(f"{q.id}: unknown type {q.type!r}")
        lo, hi = CROSS_PAPERS if q.type == "cross" else (1, 1)
        papers = {g.paper for g in q.gold}
        if not lo <= len(papers) <= hi or len(q.gold) != len(papers):
            errors.append(f"{q.id}: {len(q.gold)} gold items, expected {lo}-{hi} distinct papers")
        for g in q.gold:
            words = len(g.evidence.split())
            if not EVIDENCE_WORDS[0] <= words <= EVIDENCE_WORDS[1]:
                errors.append(f"{q.id}: evidence for {g.paper} has {words} words")
            if g.paper not in texts:
                errors.append(f"{q.id}: unknown paper {g.paper}")
            elif normalize(g.evidence) not in texts[g.paper]:
                errors.append(f"{q.id}: evidence not found in {g.paper}: {g.evidence[:60]!r}")
    for paper in documents:
        kinds = sorted(q.type for q in questions if q.type != "cross" and q.gold[0].paper == paper)
        if kinds != ["described", "named"]:
            errors.append(f"{paper}: single questions {kinds}, expected one named, one described")
    if sum(q.type == "cross" for q in questions) != 10:
        errors.append("expected 10 cross questions")
    return errors


def fetch(dest: Path = CORPUS_DIR, sources: Path = SOURCES_PATH) -> int:
    """Download the missing PDFs from arXiv and write ``SOURCES.json``; returns the new count."""
    dest.mkdir(parents=True, exist_ok=True)
    data = json.loads(sources.read_text(encoding="utf-8"))
    (dest / "SOURCES.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    new = 0
    for doc in data["documents"]:
        path = dest / f"{doc['id']}.pdf"
        if path.is_file() and path.stat().st_size:
            continue
        request = urllib.request.Request(
            f"https://arxiv.org/pdf/{doc['id']}", headers={"User-Agent": "ragkit-research"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            path.write_bytes(response.read())
        new += 1
        time.sleep(3)  # arXiv asks for at most one request every 3 seconds
    return new


def measure(
    q: PaperQuestion, ranked_papers: Sequence[str], packed: Sequence[Passage]
) -> dict[str, float]:
    """Metric values of one question: ``ranked_papers`` is the chunk ranking's paper order."""
    context = normalize(" ".join(p.text for p in packed))
    ids = {p.article_id for p in packed}
    gold = {g.paper for g in q.gold}
    order = list(dict.fromkeys(ranked_papers))
    first = min((order.index(p) + 1 for p in gold if p in order), default=0)
    return {
        "evidence": sum(normalize(g.evidence) in context for g in q.gold) / len(q.gold),
        "papers": len(gold & ids) / len(gold),
        "mrr": 1 / first if first else 0.0,
        "tokens": float(sum(estimate_tokens(p.text) for p in packed)),
        "packed": float(len(packed)),
    }


def pack(
    pool: Sequence[Passage], documents: Mapping[str, str], window: int | None
) -> list[Passage]:
    """The shipped ``search()`` packing of a top-``PACK_CHUNKS`` pool, with or without windows."""
    return pack_articles(
        pool[:PACK_CHUNKS],
        documents,
        budget_tokens=ARTICLE_BUDGET_TOKENS,
        overflow=PACK_OVERFLOW,
        window_tokens=window,
    )


Scores = dict[str, list[float]]


def _diff(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return paired_bootstrap_ci(a, b, n_resamples=N_RESAMPLES)


def _fmt_diff(a: Sequence[float], b: Sequence[float]) -> str:
    m, lo, hi = _diff(a, b)
    better = sum(x > y for x, y in zip(a, b, strict=True))
    worse = sum(x < y for x, y in zip(a, b, strict=True))
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}] ({better}/{worse}/{len(a) - better - worse})"


def _mean(values: Sequence[float]) -> str:
    if not values:
        return ""
    m, lo, hi = bootstrap_ci(values, n_resamples=N_RESAMPLES)
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def verdicts(rows: Mapping[str, Scores], eu_identical: bool) -> dict[str, str]:
    """The protocol's decisions: W against B0 with the EU guard, then WC against W."""
    _, lo_w, _ = _diff(rows["W"]["evidence"], rows["B0"]["evidence"])
    w = lo_w > 0 and eu_identical
    out = {"W": "adopted" if w else "not adopted"}
    if w:
        _, lo_c, _ = _diff(rows["WC"]["evidence"], rows["W"]["evidence"])
        out["WC"] = "adopted" if lo_c > 0 else "not adopted"
    else:
        out["WC"] = "not judged (W not adopted); WC - B0 reported only"
    return out


def render(
    rows: Mapping[str, Scores],
    types: Sequence[str],
    eu_identical: bool,
    notes: Sequence[str],
) -> str:
    """Tables per metric and type, paired differences, verdicts and the non-gating notes."""
    lines = ["### ragkit papers score: P-1 (docs/papers_protocol.md)", ""]
    header = " | ".join(LABELS[m] for m in METRICS)
    for kind in ("all", *TYPES):
        keep = [i for i, t in enumerate(types) if kind == "all" or t == kind]
        lines += [f"#### {kind} ({len(keep)} questions)", "", f"| system | {header} |"]
        lines.append("|---|" + "---|" * len(METRICS))
        for name, scores in rows.items():
            cells = " | ".join(_mean([scores[m][i] for i in keep]) for m in METRICS)
            lines.append(f"| {name} | {cells} |")
        lines.append("")
        for a, b in (("W", "B0"), ("WC", "W"), ("WC", "B0")):
            ev = [rows[a]["evidence"][i] for i in keep], [rows[b]["evidence"][i] for i in keep]
            lines.append(f"- evidence recall {a} - {b}: {_fmt_diff(*ev)}")
        lines.append("")
    lines += ["#### Verdicts", ""]
    lines.append(f"- EU guard (W packs what B0 packs on all 190 EU questions): {eu_identical}")
    lines += [f"- {name}: **{status}**" for name, status in verdicts(rows, eu_identical).items()]
    lines += ["", "#### Not gating", ""]
    lines += [f"- {n}" for n in notes]
    return "\n".join(lines)


def load_contexts(path: Path = CONTEXTS_PATH) -> dict[str, str]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in data["contexts"].items()}
