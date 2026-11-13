"""End-to-end EU-law answering through the ezcross-ems agent (``docs/agent_eval_protocol.md``).

Four setups, one question set (``eu-consolidated``: 43 answerable + 44 unanswerable):

- ``A``: ``ragkit.search`` defaults (budget packing, 12k tokens, 10 chunks);
- ``B``: the three best articles only (``ragkit answer --no-pack``, k = 3);
- ``C``: the packed articles of ``A`` in reversed order (worst first);
- ``D``: the whole corpus in the prompt, no tool.

A to C run through the real agent loop (``ezc-cli`` session, model ``local/<id>``) and the real
``search_eu_law`` tool; ``serve`` stands in for the backend's ``POST /regulation/search`` (it calls
the same ``ragkit.search`` and returns the same JSON, shaped per setup). ``ask-full`` plays D. The
agent writes free text, so the answers are judged with ``CORRECT``, ``CITE`` and ``REFUSE``
(``generation.py``) by a second local model through a cached Ollama transport; ``report`` replays
that cache and prints the tables and the decision.

    python -m ragkit.agentbench questions | serve --setup A | ask-full | report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import threading
import time
import urllib.request
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from ragkit.eval import bootstrap_ci, paired_bootstrap_ci
from ragkit.generation import LLMAnswerer, corpus_block
from ragkit.llm import CachedMessages, ollama_transport
from ragkit.parent import estimate_tokens

DATA = Path(__file__).parent / "data"
CORPUS_DIR = DATA / "corpus_eu_consolidated"
QA_FILE = DATA / "qa_eu_consolidated.jsonl"
UNANSWERABLE_FILE = DATA / "unanswerable_eu_consolidated.jsonl"
RUNS_DIR = Path("runs/agent_eval")
JUDGE_CACHE = Path("data/llm_cache/agent_eval_judge.jsonl")
SETUPS = ("A", "B", "C", "D")
CANDIDATES = ("B", "C")  # D is a reference: the corpus does not go into every request
K3_CHUNKS = 5  # the chunks ``ragkit answer --no-pack`` reads (``--top-k``)
K3_ARTICLES = 3  # ``--articles``
JUDGE_MODEL = "gemma3:27b"
JUDGE_NUM_CTX = 32768
D_NUM_CTX = 131072  # the corpus is about 68 k tokens (353 kB at 5.2 characters per token)
D_MAX_TOKENS = 8192  # as the agent's local model (``createLocalModel`` fallback)
MAX_FAILED = 3  # failed runs a setup may have and still be eligible
REFUSAL_MARGIN = 0.10  # candidate may refuse this much less of the unanswerable questions than A
RESAMPLES = 2000
SEED = 0

D_PREAMBLE = (
    "Reference corpus: the complete text of 94 EU-law articles (electricity market, energy), each "
    'headed by "=== <article id> ===". There is no search tool in this conversation: answer from '
    "these articles. Cite every claim by article id. If no article supports an answer, say so "
    "plainly; do not answer EU-law questions from memory."
)


# --- questions -------------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_questions() -> list[dict[str, Any]]:
    """Answerable ``a01..`` then unanswerable ``u01..``, in file order."""
    rows = [
        {
            "id": f"a{i:02d}",
            "q": r["q"],
            "kind": "answerable",
            "reference": r["answer"],
            "gold": r["relevant_doc_ids"],
        }
        for i, r in enumerate(_read_jsonl(QA_FILE), 1)
    ]
    rows += [
        {"id": f"u{i:02d}", "q": r["q"], "kind": "unanswerable", "reference": None, "gold": []}
        for i, r in enumerate(_read_jsonl(UNANSWERABLE_FILE), 1)
    ]
    return rows


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- the backend stand-in --------------------------------------------------------------------


def search_for_setup(setup: str, query: str, max_tokens: int | None) -> dict[str, Any]:
    """The JSON ``POST /regulation/search`` returns, shaped for setup A, B or C."""
    import ragkit

    kwargs: dict[str, Any] = {}
    if max_tokens is not None:
        kwargs["budget_tokens"] = max_tokens
    if setup == "B":
        kwargs["pack_chunks"] = K3_CHUNKS
    body = ragkit.search(query, **kwargs).to_dict()
    if (
        setup == "B"
    ):  # the first k articles of the packed list are what ``parent_articles(k)`` takes
        body["articles"] = body["articles"][:K3_ARTICLES]
    elif setup == "C":
        body["articles"] = body["articles"][::-1]
    body["estimated_tokens"] = sum(estimate_tokens(a["text"]) for a in body["articles"])
    return dict(body)


def _handler(setup: str, log_path: Path) -> type[BaseHTTPRequestHandler]:
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, body: Mapping[str, Any]) -> None:
            raw = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self) -> None:
            if self.path != "/api/v1/regulation/search":
                self._reply(404, {"detail": "Not Found"})
                return
            if not self.headers.get("authorization", "").startswith("Bearer "):
                self._reply(401, {"detail": "Not authenticated"})
                return
            data = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
            query, max_tokens = data.get("query"), data.get("max_tokens")
            if not isinstance(query, str) or not 1 <= len(query) <= 2000:
                self._reply(422, {"detail": "query must be a string of 1 to 2000 characters"})
                return
            if max_tokens is not None and not (
                isinstance(max_tokens, int) and 1000 <= max_tokens <= 32000
            ):
                self._reply(422, {"detail": "max_tokens must be an integer from 1000 to 32000"})
                return
            started = time.monotonic()
            try:
                body = search_for_setup(setup, query, max_tokens)
            except ValueError as exc:
                self._reply(422, {"detail": str(exc)})
                return
            entry = {
                "setup": setup,
                "query": query,
                "max_tokens": max_tokens,
                "retriever": body["retriever"],
                "ids": [a["id"] for a in body["articles"]],
                "estimated_tokens": body["estimated_tokens"],
                "ms": round((time.monotonic() - started) * 1000),
            }
            with lock, log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
            self._reply(200, body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    return Handler


def serve(setup: str, port: int, log_path: Path) -> None:
    """Serve ``POST /api/v1/regulation/search`` for ``setup`` until interrupted."""
    search_for_setup(setup, "warm-up", None)  # builds the pipeline now, not on the first question
    log_path.parent.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _handler(setup, log_path))
    print(f"setup {setup} on http://127.0.0.1:{port}/api/v1", flush=True)
    httpd.serve_forever()


# --- setup D ---------------------------------------------------------------------------------

CHARS_PER_TOKEN = (
    5.2  # what the qwen tokenizer gives on the corpus (10 649 tokens for 55 444 chars)
)


def truncation_problem(
    prompt_tokens: int, completion_tokens: int, chars: int, num_ctx: int
) -> str | None:
    """Why a model call looks truncated (Ollama cuts a prompt over ``num_ctx`` silently), else None.

    The prompt filled the context, prompt plus completion did, or the prompt has less than 0.8 of
    the tokens its size implies at ``CHARS_PER_TOKEN``. The same rule runs in ``eu-law-run.ts``.
    """
    if prompt_tokens >= num_ctx - 8:
        return f"prompt of {prompt_tokens} tokens fills num_ctx {num_ctx}"
    if prompt_tokens + completion_tokens >= num_ctx:
        return f"prompt + completion ({prompt_tokens + completion_tokens}) fill num_ctx {num_ctx}"
    if prompt_tokens < 0.8 * chars / CHARS_PER_TOKEN:
        return (
            f"prompt of {prompt_tokens} tokens is below 0.8 x {round(chars / CHARS_PER_TOKEN)} "
            f"expected from {chars} characters"
        )
    return None


def ask_full(
    questions: Sequence[Mapping[str, Any]], out: Path, base_url: str, model: str, system: str
) -> None:
    """Setup D: the whole corpus in the system prompt, no tool, one native ``/api/chat`` call each.

    The native API takes ``options.num_ctx`` (the ``/v1`` one ignores it). Every call is checked
    with ``truncation_problem``; a call that fails it is recorded as failed, never scored.
    """
    from ragkit._search import load_documents

    system_prompt = f"{system}\n\n{D_PREAMBLE}\n\n{corpus_block(load_documents(CORPUS_DIR))}"
    done = {r["id"] for r in _load_records(out)} if out.is_file() else set()
    for question in questions:
        if question["id"] in done:
            continue
        chars = len(system_prompt) + len(question["q"])
        body = json.dumps(
            {
                "model": model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question["q"]},
                ],
                "options": {
                    "num_ctx": D_NUM_CTX,
                    "temperature": 0,
                    "seed": SEED,
                    "num_predict": D_MAX_TOKENS,
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/chat",
            data=body,
            headers={"content-type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        record: dict[str, Any] = {"id": question["id"], "q": question["q"], "model": model}
        try:
            with urllib.request.urlopen(request, timeout=3600) as response:
                decoded = json.loads(response.read().decode("utf-8"))
            message = decoded["message"]
            call = {
                "prompt_tokens": decoded["prompt_eval_count"],
                "completion_tokens": decoded["eval_count"],
                "stop_reason": decoded.get("done_reason"),
                "request_chars": chars,
            }
            record |= {
                "final_text": message.get("content") or "",
                "thinking": message.get("thinking") or "",
                "model_calls": [call],
                "tool_calls": [],
            }
            problem = truncation_problem(
                call["prompt_tokens"], call["completion_tokens"], chars, D_NUM_CTX
            )
            if problem:
                record["error"] = f"truncation guard: {problem}"
        except Exception as exc:  # a failed run is recorded and counted, not hidden
            record |= {"final_text": "", "error": repr(exc), "model_calls": [], "tool_calls": []}
        record["latency_ms"] = round((time.monotonic() - started) * 1000)
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            question["id"], record["latency_ms"] // 1000, "s", record.get("error", ""), flush=True
        )


# --- citations -------------------------------------------------------------------------------

_ID = re.compile(r"\b(?:reg|dir)_\d{4}_\d+_art[0-9a-z]+\b")
_ARTICLE = re.compile(r"Articles?\s+(\d+[a-z]?)\b")
_ACT = re.compile(r"\b(\d{4})/(\d+)\b")
_REACH = 80  # characters around ``Article N`` searched for the act number


def cited_articles(text: str, corpus_ids: set[str]) -> list[str]:
    """Article ids an answer cites: ids verbatim, else ``Article N ... YYYY/NNN`` resolved to an id.

    Ids not in the corpus are kept (they count as unsupported). The natural-language form takes the
    first ``YYYY/NNN`` within 80 characters after ``Article N`` (``Articles 16 and 17`` gives 16),
    else the last one within 80 characters before it (``Regulation 2019/943, Article 16``).
    """
    cited = list(dict.fromkeys(_ID.findall(text)))
    if cited:
        return cited
    for match in _ARTICLE.finditer(text):
        act = _ACT.search(text, match.end(), match.end() + _REACH)
        if not act:
            before = list(_ACT.finditer(text, max(0, match.start() - _REACH), match.start()))
            act = before[-1] if before else None
        if act:
            for prefix in ("reg", "dir"):
                candidate = f"{prefix}_{act[1]}_{act[2]}_art{match[1]}"
                if candidate in corpus_ids:
                    cited.append(candidate)
    return list(dict.fromkeys(cited))


def returned_ids(record: Mapping[str, Any]) -> set[str]:
    """Article ids the ``search_eu_law`` results of an agent transcript returned."""
    ids: set[str] = set()
    for message in record.get("messages", []):
        if message.get("role") == "toolResult" and message.get("toolName") == "search_eu_law":
            data = (message.get("details") or {}).get("data") or {}
            ids |= {a["id"] for a in data.get("articles", [])}
    return ids


# --- scoring ---------------------------------------------------------------------------------


def _load_records(path: Path) -> list[dict[str, Any]]:
    """Records of a run file; the last line of an id wins (a re-run replaces a failed one)."""
    latest: dict[str, dict[str, Any]] = {}
    for record in _read_jsonl(path):
        latest[record["id"]] = record
    return list(latest.values())


@dataclass
class Scored:
    """Judged outcome of one setup, per question in ``questions`` order (``nan`` = undefined)."""

    setup: str
    failed: int
    correct: np.ndarray  # answerable: 1 iff CORRECT = yes and the answer is not a refusal
    partial: np.ndarray  # answerable: 1 iff CORRECT = partial
    refused_answerable: np.ndarray  # answerable: REFUSE = yes
    refused_unanswerable: np.ndarray  # unanswerable: REFUSE = yes
    cite_yes: np.ndarray  # answerable, not refused: share of cited articles with CITE = yes
    cited: np.ndarray  # answerable, not refused: number of cited articles
    no_citation: np.ndarray  # answerable, not refused: 1 iff the answer cites nothing
    context_tokens: np.ndarray  # prompt tokens of the last model call (all questions)
    latency_s: np.ndarray
    tool_called: np.ndarray  # 1 iff the agent called search_eu_law at least once (A to C)
    searches: np.ndarray  # search_eu_law calls in the run
    max_prompt: np.ndarray  # largest prompt (tokens) of any model call of the run
    runaways: np.ndarray  # 1 iff a model call used the whole 8192-token limit
    verdicts: list[dict[str, Any]]


def judge_client(base_url: str, *, live: bool, cache: Path = JUDGE_CACHE) -> CachedMessages:
    """The cached judge client: temperature 0, no cost cap (local model), live calls on request."""
    return CachedMessages(
        cache,
        api_key="local" if live else "",
        model=JUDGE_MODEL,
        temperature=0.0,
        max_calls=100_000,
        max_cost_usd=math.inf,
        transport=ollama_transport(base_url, num_ctx=JUDGE_NUM_CTX, seed=SEED),
    )


def score_setup(
    setup: str,
    records: Sequence[Mapping[str, Any]],
    questions: Sequence[Mapping[str, Any]],
    answerer: LLMAnswerer,
    corpus: Mapping[str, str],
    *,
    workers: int = 2,
) -> Scored:
    """Judge every answer of one setup (cached calls replay offline)."""
    by_id = {r["id"]: r for r in records}
    corpus_ids = set(corpus)

    def one(question: Mapping[str, Any]) -> dict[str, Any]:
        record = by_id.get(question["id"])
        # a run the harness flagged (timeout, truncation guard, crash) is a failed run, never scored
        answer = "" if not record or record.get("error") else str(record["final_text"]).strip()
        out: dict[str, Any] = {"id": question["id"], "failed": not answer}
        if not answer:
            return out
        out["refused"] = answerer.judge_refusal(question["q"], answer).verdict == "yes"
        if question["kind"] != "answerable" or out["refused"]:
            return out
        out["correct"] = answerer.judge_correct(
            question["q"], question["reference"], answer
        ).verdict
        assert record is not None  # an answer exists only for a record
        retrieved = corpus_ids if setup == "D" else returned_ids(record)
        cites = []
        for article_id in cited_articles(answer, corpus_ids):
            if article_id in corpus and article_id in retrieved:
                verdict = answerer.judge_citation(question["q"], answer, corpus[article_id]).verdict
            else:
                verdict = "invalid"  # not in the corpus, or never returned by the tool this run
            cites.append({"id": article_id, "verdict": verdict})
        out["cites"] = cites
        return out

    with ThreadPoolExecutor(workers) as pool:
        verdicts = list(pool.map(one, questions))
    nan = float("nan")
    answerable = [q["kind"] == "answerable" for q in questions]
    n = len(questions)
    correct, partial, ref_a, ref_u = (np.full(n, nan) for _ in range(4))
    cite_yes, cited, no_cite = (np.full(n, nan) for _ in range(3))
    for i, v in enumerate(verdicts):
        if answerable[i]:
            correct[i] = float(v.get("correct") == "yes")
            partial[i] = float(v.get("correct") == "partial")
            ref_a[i] = float(v.get("refused", False))
            if "cites" in v:
                cites = v["cites"]
                cited[i] = len(cites)
                no_cite[i] = float(not cites)
                cite_yes[i] = (
                    sum(c["verdict"] == "yes" for c in cites) / len(cites) if cites else 0.0
                )
        else:
            ref_u[i] = float(v.get("refused", False))
    calls = [by_id[q["id"]]["model_calls"] if q["id"] in by_id else [] for q in questions]
    return Scored(
        setup=setup,
        failed=sum(v["failed"] for v in verdicts),
        correct=correct,
        partial=partial,
        refused_answerable=ref_a,
        refused_unanswerable=ref_u,
        cite_yes=cite_yes,
        cited=cited,
        no_citation=no_cite,
        context_tokens=np.array([c[-1]["prompt_tokens"] if c else nan for c in calls], dtype=float),
        latency_s=np.array(
            [by_id[q["id"]]["latency_ms"] / 1000 if q["id"] in by_id else nan for q in questions]
        ),
        tool_called=np.array(
            [
                float(bool(by_id[q["id"]]["tool_calls"])) if q["id"] in by_id else nan
                for q in questions
            ]
        ),
        searches=np.array(
            [len(by_id[q["id"]]["tool_calls"]) if q["id"] in by_id else nan for q in questions],
            dtype=float,
        ),
        max_prompt=np.array(
            [max((c["prompt_tokens"] for c in k), default=nan) for k in calls], dtype=float
        ),
        runaways=np.array(
            [float(any(c["completion_tokens"] >= D_MAX_TOKENS for c in k)) for k in calls]
        ),
        verdicts=verdicts,
    )


def _valid(x: np.ndarray) -> np.ndarray:
    return x[~np.isnan(x)]


def ci(x: np.ndarray) -> tuple[float, float, float]:
    """Mean and percentile-bootstrap 95 % CI over the questions where ``x`` is defined."""
    return bootstrap_ci(list(_valid(x)), n_resamples=RESAMPLES, seed=SEED)


def paired(x: np.ndarray, base: np.ndarray) -> tuple[float, float, float, int]:
    """Paired bootstrap of ``mean(x - base)`` over the questions defined in both; plus their n."""
    keep = ~np.isnan(x) & ~np.isnan(base)
    m, lo, hi = paired_bootstrap_ci(
        list(x[keep]), list(base[keep]), n_resamples=RESAMPLES, seed=SEED
    )
    return m, lo, hi, int(keep.sum())


def decide(results: Mapping[str, Scored]) -> tuple[str, list[str]]:
    """The pre-written rule of ``docs/agent_eval_protocol.md``: the default setup and why."""
    base = results["A"]
    lines: list[str] = []
    passing: list[tuple[float, str]] = []
    for name in CANDIDATES:
        if name not in results:
            continue
        r = results[name]
        d_correct = paired(r.correct, base.correct)
        ref_x, ref_a = (
            float(np.nanmean(r.refused_unanswerable)),
            float(np.nanmean(base.refused_unanswerable)),
        )
        r1 = d_correct[1] > 0
        r2 = ref_x >= ref_a - REFUSAL_MARGIN
        r3 = r.failed <= MAX_FAILED
        lines.append(
            f"{name}: R1 correct diff {d_correct[0]:+.3f} "
            f"[{d_correct[1]:+.3f}, {d_correct[2]:+.3f}] "
            f"lower bound > 0: {r1}; R2 refusal on unanswerable {ref_x:.3f} vs {ref_a:.3f} "
            f"(margin {REFUSAL_MARGIN:.2f}): {r2}; R3 failed runs {r.failed} <= {MAX_FAILED}: {r3}"
        )
        if r1 and r2 and r3:
            passing.append((d_correct[0], name))
    if base.failed > MAX_FAILED:
        lines.append(f"A has {base.failed} failed runs (> {MAX_FAILED}): the baseline is not valid")
    default = max(passing)[1] if passing else "A"
    lines.append(f"default: {default}")
    return default, lines


def _fmt(t: tuple[float, float, float], scale: float = 1.0, digits: int = 3) -> str:
    return f"{t[0] * scale:.{digits}f} [{t[1] * scale:.{digits}f}, {t[2] * scale:.{digits}f}]"


def render(results: Mapping[str, Scored]) -> str:
    """Markdown tables of the study: per setup with CIs, then paired differences against A."""
    names = [s for s in SETUPS if s in results]
    rows = [
        ("correct (answerable)", "correct", 1.0, 3),
        ("partial (answerable)", "partial", 1.0, 3),
        ("wrongly refused (answerable)", "refused_answerable", 1.0, 3),
        ("refused (unanswerable)", "refused_unanswerable", 1.0, 3),
        ("cited articles judged `yes` (answerable, answered)", "cite_yes", 1.0, 3),
        ("articles cited per answer", "cited", 1.0, 2),
        ("answers without citation", "no_citation", 1.0, 3),
        ("agent called the tool (all)", "tool_called", 1.0, 3),
        ("prompt tokens of the last call (all)", "context_tokens", 1.0, 0),
        ("latency, s per question (all)", "latency_s", 1.0, 0),
    ]
    out = ["| metric (mean [95 % CI]) | " + " | ".join(names) + " |"]
    out.append("|---|" + "---|" * len(names))
    for label, attr, scale, digits in rows:
        cells = [_fmt(ci(getattr(results[s], attr)), scale, digits) for s in names]
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    out.append("| failed runs | " + " | ".join(str(results[s].failed) for s in names) + " |")
    out += ["", "Paired differences against A (X - A; paired bootstrap over questions):", ""]
    out.append("| metric | " + " | ".join(f"{s} - A" for s in names if s != "A") + " |")
    out.append("|---|" + "---|" * (len(names) - 1))
    for label, attr, _, digits in rows[:5]:
        cells = []
        for s in (n for n in names if n != "A"):
            m, lo, hi, n = paired(getattr(results[s], attr), getattr(results["A"], attr))
            cells.append(f"{m:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}] (n={n})")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    out += [
        "",
        "Agent loop by question kind (mean / max per question; runaway = a call hit 8192 tokens):",
        "",
    ]
    out.append("| setup | kind | searches | max prompt tokens | runaways | n |")
    out.append("|---|---|---|---|---|---|")
    for s in names:
        r = results[s]
        for kind, prefix in (("answerable", "a"), ("unanswerable", "u")):
            keep = np.array([v["id"].startswith(prefix) for v in r.verdicts]) & ~np.isnan(
                r.searches
            )
            if keep.any():
                out.append(
                    f"| {s} | {kind} "
                    f"| {r.searches[keep].mean():.2f} / {r.searches[keep].max():.0f} "
                    f"| {r.max_prompt[keep].mean():.0f} / {r.max_prompt[keep].max():.0f} "
                    f"| {int(r.runaways[keep].sum())} | {int(keep.sum())} |"
                )
    _, rule = decide(results)
    return "\n".join(out + ["", "Decision rule:", ""] + [f"- {line}" for line in rule])


# --- command line ----------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m ragkit.agentbench", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("questions", help="write runs/agent_eval/questions.jsonl and print file hashes")
    s = sub.add_parser("serve", help="backend stand-in for setup A, B or C")
    s.add_argument("--setup", required=True, choices=("A", "B", "C"))
    s.add_argument("--port", type=int, required=True)
    s.add_argument("--log", default=None)
    d = sub.add_parser("ask-full", help="setup D: the whole corpus in the prompt")
    d.add_argument("--base-url", default="http://localhost:21434")
    d.add_argument("--model", required=True)
    d.add_argument("--system-prompt-from", default=str(RUNS_DIR / "A.jsonl.config.json"))
    d.add_argument("--only", default=None, help="comma-separated question ids")
    r = sub.add_parser("report", help="judge (cached) and print the tables and the decision")
    r.add_argument("--base-url", default="http://localhost:21434")
    r.add_argument(
        "--live", action="store_true", help="make missing judge calls (needs the server)"
    )
    r.add_argument("--setups", default=",".join(SETUPS))
    r.add_argument("--out", default=None)
    r.add_argument("--verdicts-out", default=None, help="write every per-question verdict as JSON")
    r.add_argument("--workers", type=int, default=2)
    args = p.parse_args(argv)

    if args.cmd == "questions":
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        rows = load_questions()
        path = RUNS_DIR / "questions.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        for f in (QA_FILE, UNANSWERABLE_FILE):
            print(f.name, sha256_file(f))
        print(path, len(rows), "questions")
    elif args.cmd == "serve":
        log = Path(args.log) if args.log else RUNS_DIR / f"{args.setup}.server.jsonl"
        serve(args.setup, args.port, log)
    elif args.cmd == "ask-full":
        system = json.loads(Path(args.system_prompt_from).read_text(encoding="utf-8"))[
            "systemPrompt"
        ]
        wanted = args.only.split(",") if args.only else None
        rows = [q for q in load_questions() if wanted is None or q["id"] in wanted]
        ask_full(rows, RUNS_DIR / "D.jsonl", args.base_url, args.model, system)
    else:
        from ragkit._search import load_documents

        corpus = dict(load_documents(CORPUS_DIR))
        answerer = LLMAnswerer(judge_client(args.base_url, live=args.live))
        questions = load_questions()
        results = {
            s: score_setup(
                s,
                _load_records(RUNS_DIR / f"{s}.jsonl"),
                questions,
                answerer,
                corpus,
                workers=args.workers,
            )
            for s in args.setups.split(",")
        }
        text = render(results)
        print(text)
        if args.out:
            Path(args.out).write_text(text + "\n", encoding="utf-8")
        if args.verdicts_out:
            dump = {s: r.verdicts for s, r in results.items()}
            Path(args.verdicts_out).write_text(json.dumps(dump, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
