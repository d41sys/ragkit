"""Paper packing on open benchmarks (P-2): QASPER and MDA-QA (docs/open_papers_protocol.md).

A packing configuration is (budget, window, slices). The dev splits run every configuration and
choose S_b per budget; the test splits are scored once for B0 and the chosen S_b, and on MDA-QA
also with Qwen contexts (WC_b). P-3 (docs/paper_answers_protocol.md) answers fresh questions from
the contexts packed at 12 000 and 36 000 tokens and has the answers judged.
"""

from __future__ import annotations

import json
import random
import re
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragkit.eval import bootstrap_ci, paired_bootstrap_ci
from ragkit.papersbench import normalize
from ragkit.parent import PACK_CHUNKS, PACK_OVERFLOW, Passage, estimate_tokens, pack_articles

DATA_DIR = Path("data/raw/open_papers")  # gitignored downloads (CC BY 4.0)
QASPER_FILES = {
    "dev": "qasper-dev-v0.3.json",
    "test": "qasper-test-v0.3.json",
    "train": "qasper-train-v0.3.json",  # P-3 only
}
MDA_FILE = "MDA-QA.json"
SPIQA_ZIP = "spiqa_paragraphs.zip"
MDA_SPLITS = {"dev": (0, 100), "test": (100, 200), "answers": (200, 300)}  # equal pools
BENCHMARKS = ("qasper", "mdaqa")
BUDGETS = (12_000, 24_000, 36_000)
WINDOWS = (None, 4000, 8000)
SLICES = (1, 3)
TIE = 0.005
QASPER_MARGIN = -0.02  # rule 1: S_12000 may not lose more than this on QASPER
N_RESAMPLES = 10_000
DEV_PATH = Path("docs/sota-runs/open_papers_dev.json")
CONTEXTS_PATH = Path("docs/sota-runs/open_papers_contexts.json")
PROBE_LIMIT = 32_568  # num_ctx 32 768 minus the 200-token reply

Config = tuple[int, int | None, int]
B0: Config = (12_000, None, 1)
CONFIGS: tuple[Config, ...] = tuple((b, w, s) for b in BUDGETS for w in WINDOWS for s in SLICES)
METRICS = ("primary", "all", "mrr", "tokens", "packed")
LABELS = {
    "primary": "primary",
    "all": "all gold papers packed",
    "mrr": "MRR (paper)",
    "tokens": "tokens",
    "packed": "papers packed",
}


@dataclass(frozen=True)
class OpenQuestion:
    id: str
    text: str
    papers: tuple[str, ...]
    evidence: tuple[tuple[str, ...], ...]  # QASPER: text evidence per qualifying annotator
    answer: str = ""  # reference answer (P-3)


def label(config: Config) -> str:
    budget, window, slices = config
    return f"{budget}/{window or 'none'}/{slices}"


Docs = list[tuple[str, str]]


def qasper(split: str, data_dir: Path = DATA_DIR) -> tuple[Docs, list[OpenQuestion]]:
    """The split's papers and its questions with at least one answer with text evidence."""
    raw: dict[str, Any] = json.loads((data_dir / QASPER_FILES[split]).read_text(encoding="utf-8"))
    docs, questions = [], []
    for pid, paper in raw.items():
        sections = [
            "\n\n".join([s["section_name"] or "", *s["paragraphs"]]) for s in paper["full_text"]
        ]
        docs.append((pid, "\n\n".join(sections)))
        for qa in paper["qas"]:
            qualifying = [
                (a["answer"], ev)
                for a in qa["answers"]
                if not a["answer"]["unanswerable"]
                and (ev := tuple(e for e in a["answer"]["evidence"] if not _float(e)))
            ]
            if qualifying:
                first = qualifying[0][0]
                reference = (
                    first["free_form_answer"]
                    or "; ".join(first["extractive_spans"])
                    or ("Yes" if first["yes_no"] else "No")
                )
                text = f'In the paper "{paper["title"]}": {qa["question"].strip()}'
                evidence = tuple(ev for _, ev in qualifying)
                question = OpenQuestion(qa["question_id"], text, (pid,), evidence, reference)
                questions.append(question)
    return docs, questions


def _float(evidence: str) -> bool:
    return evidence.startswith("FLOAT SELECTED")  # tables and figures: not in the text


def mdaqa(split: str, data_dir: Path = DATA_DIR) -> tuple[Docs, list[OpenQuestion]]:
    """The split's questions (ids shuffled with seed 0) and the union of their support papers."""
    raw: list[dict[str, Any]] = json.loads((data_dir / MDA_FILE).read_text(encoding="utf-8"))
    ids = [q["id"] for q in raw]
    random.Random(0).shuffle(ids)
    lo, hi = MDA_SPLITS[split]
    keep = set(ids[lo:hi])
    questions = [
        OpenQuestion(str(q["id"]), q["question"], tuple(q["support"]), (), q["answer"])
        for q in raw
        if q["id"] in keep
    ]
    wanted = {p for q in questions for p in q.papers}
    latest: dict[str, tuple[int, str]] = {}
    with zipfile.ZipFile(data_dir / SPIQA_ZIP) as z:
        for name in z.namelist():
            m = re.search(r"([^/]+)v(\d+)\.txt$", name)
            if m and m.group(1) in wanted and int(m.group(2)) > latest.get(m.group(1), (0, ""))[0]:
                latest[m.group(1)] = (int(m.group(2)), name)
        docs = [(pid, z.read(latest[pid][1]).decode("utf-8")) for pid in sorted(wanted)]
    return docs, questions


def pack(pool: Sequence[Passage], documents: Mapping[str, str], config: Config) -> list[Passage]:
    budget, window, slices = config
    return pack_articles(
        pool[:PACK_CHUNKS],
        documents,
        budget_tokens=budget,
        overflow=PACK_OVERFLOW,
        window_tokens=window,
        slices=slices,
    )


def measure(
    q: OpenQuestion, ranked_papers: Sequence[str], packed: Sequence[Passage]
) -> dict[str, float]:
    """QASPER: best annotator's share of evidence packed; MDA-QA: share of support papers packed."""
    ids = {p.article_id for p in packed}
    support = sum(p in ids for p in q.papers) / len(q.papers)
    if q.evidence:
        context = normalize(" ".join(p.text for p in packed))
        primary = max(sum(normalize(e) in context for e in ev) / len(ev) for ev in q.evidence)
    else:
        primary = support
    order = list(dict.fromkeys(ranked_papers))
    first = min((order.index(p) + 1 for p in q.papers if p in order), default=0)
    return {
        "primary": primary,
        "all": float(support == 1.0),
        "mrr": 1 / first if first else 0.0,
        "tokens": float(sum(estimate_tokens(p.text) for p in packed)),
        "packed": float(len(packed)),
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def choose(dev: Mapping[str, Mapping[Config, Sequence[float]]]) -> dict[int, Config]:
    """S_b per budget: best mean dev primary; ties within TIE go to window none, then slices 1."""
    out = {}
    for budget in BUDGETS:
        cands = [(budget, w, s) for w in WINDOWS for s in SLICES]  # preference order
        score = {c: _mean([_mean(dev[b][c]) for b in BENCHMARKS]) for c in cands}
        best = max(score.values())
        out[budget] = next(c for c in cands if score[c] >= best - TIE)
    return out


def render_dev(
    dev: Mapping[str, Mapping[Config, Sequence[float]]], chosen: Mapping[int, Config]
) -> str:
    lines = ["### ragkit openpapers dev: P-2 (docs/open_papers_protocol.md)", ""]
    lines += ["| budget/window/slices | QASPER evidence | MDA-QA support | mean | chosen |"]
    lines.append("|---|---|---|---|---|")
    for c in CONFIGS:
        q, m = _mean(dev["qasper"][c]), _mean(dev["mdaqa"][c])
        mark = f"S_{c[0]}" if chosen[c[0]] == c else ""
        lines.append(f"| {label(c)} | {q:.3f} | {m:.3f} | {(q + m) / 2:.3f} | {mark} |")
    return "\n".join(lines)


def _fmt_mean(values: Sequence[float]) -> str:
    m, lo, hi = bootstrap_ci(values, n_resamples=N_RESAMPLES)
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def _fmt_diff(a: Sequence[float], b: Sequence[float]) -> str:
    m, lo, hi = paired_bootstrap_ci(a, b, n_resamples=N_RESAMPLES)
    better = sum(x > y for x, y in zip(a, b, strict=True))
    worse = sum(x < y for x, y in zip(a, b, strict=True))
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}] ({better}/{worse}/{len(a) - better - worse})"


Rows = Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]]  # bench -> system -> metric


def _lower(a: Sequence[float], b: Sequence[float]) -> float:
    return paired_bootstrap_ci(a, b, n_resamples=N_RESAMPLES)[1]


def verdicts(rows: Rows, chosen: Mapping[int, Config], eu_identical: bool) -> dict[str, str]:
    """Rule 1 (packing at the shipped budget) and rule 3 (contexts)."""
    q, m = rows["qasper"], rows["mdaqa"]
    out = {}
    if chosen[12_000] == B0:
        out["S_12000"] = "equals B0, nothing to judge"
    else:
        ok = (
            _lower(m["S_12000"]["primary"], m["B0"]["primary"]) > 0
            and _lower(q["S_12000"]["primary"], q["B0"]["primary"]) > QASPER_MARGIN
            and eu_identical
        )
        out["S_12000"] = "adopted" if ok else "not adopted"
    lo_c = _lower(m["WC_12000"]["primary"], m["S_12000"]["primary"])
    out["contexts"] = "recommended" if lo_c > 0 else "not recommended"
    return out


def render_test(
    rows: Rows, chosen: Mapping[int, Config], eu_identical: bool, notes: Sequence[str]
) -> str:
    lines = ["### ragkit openpapers score: P-2 (docs/open_papers_protocol.md)", ""]
    lines += [f"- S_{b} = {label(c)}" for b, c in chosen.items()]
    header = " | ".join(LABELS[m] for m in METRICS)
    for bench, systems in rows.items():
        n = len(next(iter(systems.values()))["primary"])
        primary = "evidence recall" if bench == "qasper" else "support recall"
        lines += ["", f"#### {bench} test ({n} questions; primary = {primary})", ""]
        lines += [f"| system | {header} |", "|---|" + "---|" * len(METRICS)]
        for name, scores in systems.items():
            cells = " | ".join(_fmt_mean(scores[m]) for m in METRICS)
            lines.append(f"| {name} | {cells} |")
        lines.append("")
        pairs = [(f"S_{b}", "B0") for b in BUDGETS]
        if "WC_12000" in systems:
            pairs += [(f"WC_{b}", f"S_{b}") for b in BUDGETS]
        for a, b in pairs:
            diff = _fmt_diff(systems[a]["primary"], systems[b]["primary"])
            lines.append(f"- {a} - {b}: {diff}")
    lines += ["", "#### Verdicts", ""]
    lines.append(f"- EU guard (S_12000 packs as B0 on all 190 EU questions): {eu_identical}")
    lines += [f"- {k}: **{v}**" for k, v in verdicts(rows, chosen, eu_identical).items()]
    lines += ["- budget: reported only (rule 2)", "", "#### Not gating", ""]
    lines += [f"- {n}" for n in notes]
    return "\n".join(lines)


def save_chosen(chosen: Mapping[int, Config], path: Path = DEV_PATH) -> None:
    data = {str(b): list(c) for b, c in chosen.items()}
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def load_chosen(path: Path = DEV_PATH) -> dict[int, Config]:
    data: dict[str, list[Any]] = json.loads(path.read_text(encoding="utf-8"))
    return {int(b): (c[0], c[1], c[2]) for b, c in data.items()}


def prompt_tokens(base_url: str, model: str, prompt: str, num_ctx: int) -> int:
    """Ollama's prompt token count for ``prompt`` (``num_ctx`` when Ollama had to cut it)."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"num_ctx": num_ctx, "num_predict": 1, "temperature": 0},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        json.dumps(body).encode("utf-8"),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return int(json.load(response)["prompt_eval_count"])


# P-3: answers from the packed contexts (docs/paper_answers_protocol.md)

ANSWERS_PATH = Path("docs/sota-runs/paper_answers.json")
READER_MODEL = "gpt-6-sol"  # through the Codex CLI, reasoning effort low
ARMS = {"A12": 12_000, "A36": 36_000}
ANSWER_PAPERS = 416  # QASPER corpus size, as P-2's test pool
ANSWER_QASPER = 300  # amendment 1 cut it to 60 (the interim look), amendment 2 restored it
NO_HARM_MARGIN = -0.05
PAPER_INSTRUCTIONS = """You answer questions about research papers using ONLY the context \
passages provided. Each passage is headed by the id of the paper it comes from, in square \
brackets.

Reply with one JSON object and nothing else:
{"answer": "<answer>", "citations": ["<paper id>", ...], "context_supports": true | false}

Rules:
- "answer": your best answer to the question as asked, in at most 120 words, using only the \
passages. Do not use outside knowledge. If the passages do not contain what is needed, say what \
they do state and that they do not answer the question.
- "citations": the ids of the papers whose passages you used. Use only ids that appear in the \
context.
- "context_supports": true only if the passages contain what is needed to answer the question \
definitively, false otherwise (also false if you could only guess, or if the passages merely \
concern the same topic)."""
ANSWER_METRICS = {
    "correct": "correct",
    "partial": "yes or partial",
    "supports": "context_supports",
    "cited": "gold papers cited",
    "input_tokens": "Codex input tokens",
}


def answer_sets(data_dir: Path = DATA_DIR) -> dict[str, tuple[Docs, list[OpenQuestion]]]:
    """The fresh P-3 corpora and questions: 416 QASPER train papers (300 questions), MDA-QA 100."""
    docs, questions = qasper("train", data_dir)
    ids = sorted(pid for pid, _ in docs)
    random.Random(0).shuffle(ids)
    keep = set(ids[:ANSWER_PAPERS])
    pool = [q for q in questions if q.papers[0] in keep]
    random.Random(0).shuffle(pool)
    kept_docs = [d for d in docs if d[0] in keep]
    # MDA-QA first: a run stopped by the cost guard has finished the benchmark of rule 2
    return {"mdaqa": mdaqa("answers", data_dir), "qasper": (kept_docs, pool[:ANSWER_QASPER])}


AnswerScores = dict[str, dict[str, dict[str, list[float]]]]  # bench -> arm -> metric


def answer_scores(records: Sequence[Mapping[str, Any]]) -> AnswerScores:
    """Per-question values; both arms of a benchmark list its questions in the same order."""
    out: AnswerScores = {}
    for r in records:
        row = out.setdefault(r["bench"], {}).setdefault(r["arm"], {m: [] for m in ANSWER_METRICS})
        row["correct"].append(float(r["verdict"] == "yes"))
        row["partial"].append(float(r["verdict"] in ("yes", "partial")))
        row["supports"].append(float(r["supports"]))
        row["cited"].append(len(set(r["citations"]) & set(r["papers"])) / len(r["papers"]))
        row["input_tokens"].append(float(r["input_tokens"]))
    return out


def answer_verdicts(scores: Rows) -> dict[str, str]:
    """Rule 1 (harm on either benchmark), 2 (MDA-QA gain), 3 (no harm on QASPER)."""
    ci = {
        b: paired_bootstrap_ci(
            scores[b]["A36"]["correct"], scores[b]["A12"]["correct"], n_resamples=N_RESAMPLES
        )
        for b in BENCHMARKS
    }
    harm = any(hi < 0 for _, _, hi in ci.values())
    return {
        "36 000 default for paged corpora": "reverted to 12 000" if harm else "kept",
        "answer gain on MDA-QA": "confirmed" if ci["mdaqa"][1] > 0 else "not confirmed",
        "no harm on QASPER": "shown" if ci["qasper"][1] > NO_HARM_MARGIN else "not shown",
    }


def render_answers(records: Sequence[Mapping[str, Any]], notes: Sequence[str]) -> str:
    scores = answer_scores(records)
    lines = ["### ragkit openpapers answers-score: P-3 (docs/paper_answers_protocol.md)"]
    for bench, arms in scores.items():
        n = len(arms["A12"]["correct"])
        lines += ["", f"#### {bench} ({n} questions)", ""]
        lines.append("| arm | " + " | ".join(ANSWER_METRICS.values()) + " |")
        lines.append("|---|" + "---|" * len(ANSWER_METRICS))
        for arm, row in arms.items():
            cells = " | ".join(_fmt_mean(row[m]) for m in ANSWER_METRICS)
            lines.append(f"| {arm} | {cells} |")
        lines.append("")
        for m in ("correct", "partial", "supports"):
            diff = _fmt_diff(arms["A36"][m], arms["A12"][m])
            lines.append(f"- {ANSWER_METRICS[m]}, A36 - A12: {diff}")
    lines += ["", "#### Verdicts", ""]
    lines += [f"- {k}: **{v}**" for k, v in answer_verdicts(scores).items()]
    lines += ["", "#### Not gating", ""]
    lines += [f"- {n}" for n in notes]
    return "\n".join(lines)
