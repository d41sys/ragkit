"""Tests for the cached Messages client, the LLM calls and the K-R2 report (no key, no network)."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ragkit.cli import _load_documents, main
from ragkit.genbench import (
    Item,
    balacc_ci,
    balanced_accuracy,
    decide,
    items_from_records,
    offline_reader_scorers,
    paired_balacc_ci,
    reader_scores,
    render_generation,
    retrieve_passages,
    run_generation,
    subset_score_cache,
)
from ragkit.generation import (
    LLMAnswerer,
    Passage,
    context_block,
    corpus_block,
    parse_json_object,
)
from ragkit.llm import CachedMessages, CacheMiss, request_key, usage_cost
from ragkit.modelsignals import CE_MODEL, CE_REVISION, QA_MODEL, QA_REVISION, ScoreCache, _key
from ragkit.pipeline import RAGPipeline

USAGE = {"input_tokens": 1000, "output_tokens": 100}


class FakeApi:
    """A transport that answers by instruction text and counts calls."""

    def __init__(self, replies: dict[str, str] | None = None) -> None:
        self.calls = 0
        self.payloads: list[dict[str, Any]] = []
        self.replies = replies or {}

    def __call__(self, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        self.calls += 1
        self.payloads.append(payload)
        system = payload["system"][0]["text"]
        user = payload["messages"][0]["content"]
        if system.startswith(
            "You answer questions about EU electricity law using ONLY the context"
        ):
            text = json.dumps(
                {"answer": "a", "citations": ["art1", "nope"], "context_supports": "art1" in user}
            )
        elif system.startswith("You are an expert evaluator"):
            ok = "[art1]" in user and "wanted" in user
            text = json.dumps({"reasoning": "r", "sufficient": ok, "confidence": 90 if ok else 10})
        elif system.startswith(
            "You answer questions about EU electricity law using ONLY the corpus"
        ):
            text = json.dumps({"answer": "fc", "citations": ["art1"], "context_supports": True})
        elif system.startswith("You grade"):
            text = json.dumps({"verdict": "yes", "reason": "ok"})
        else:
            text = json.dumps({"verdict": "partial", "reason": "ok"})
        return {"content": [{"type": "text", "text": text}], "usage": USAGE}


def _client(tmp_path: Path, api: FakeApi | None = None, key: str = "k") -> CachedMessages:
    return CachedMessages(tmp_path / "c.jsonl", api_key=key, transport=api or FakeApi())


SYSTEM = [{"type": "text", "text": "sys"}]


def test_client_caches_replays_offline_and_counts_one_live_call(tmp_path: Path) -> None:
    api = FakeApi()
    client = _client(tmp_path, api)
    first = client.complete("GEN", SYSTEM, "q", 10)
    second = client.complete("GEN", SYSTEM, "q", 10)
    assert (api.calls, first.cached, second.cached) == (1, False, True)
    offline = CachedMessages(tmp_path / "c.jsonl")  # no key
    assert offline.complete("GEN", SYSTEM, "q", 10).text == first.text
    with pytest.raises(CacheMiss):
        offline.complete("GEN", SYSTEM, "another", 10)


def test_client_key_depends_on_every_payload_field(tmp_path: Path) -> None:
    client = _client(tmp_path)
    base = client.payload(SYSTEM, "q", 10)
    assert request_key(base) != request_key(client.payload(SYSTEM, "q2", 10))
    assert request_key(base) != request_key(client.payload(SYSTEM, "q", 11))
    assert request_key(base) == request_key(client.payload(SYSTEM, "q", 10))


def test_client_drops_temperature_when_the_api_rejects_it(tmp_path: Path) -> None:
    sent: list[dict[str, Any]] = []

    def transport(payload: dict[str, Any], key: str) -> dict[str, Any]:
        sent.append(payload)
        if "temperature" in payload:
            raise RuntimeError("API error 400: temperature is not supported")
        return {"content": [{"type": "text", "text": "{}"}], "usage": USAGE}

    client = CachedMessages(tmp_path / "c.jsonl", api_key="k", transport=transport, temperature=0.0)
    client.complete("GEN", SYSTEM, "q", 10)
    assert "temperature" in sent[0] and "temperature" not in sent[1]
    record = next(iter(client.records.values()))
    assert record["temperature_dropped"] is True
    # the same request replays from the key of the payload that was asked for
    replay = CachedMessages(tmp_path / "c.jsonl", temperature=0.0)
    assert replay.complete("GEN", SYSTEM, "q", 10).cached


def test_client_enforces_call_and_cost_budgets(tmp_path: Path) -> None:
    capped = CachedMessages(tmp_path / "a.jsonl", api_key="k", transport=FakeApi(), max_calls=1)
    capped.complete("GEN", SYSTEM, "1", 10)
    with pytest.raises(RuntimeError, match="call budget"):
        capped.complete("GEN", SYSTEM, "2", 10)
    poor = CachedMessages(tmp_path / "b.jsonl", api_key="k", transport=FakeApi(), max_cost_usd=0.0)
    with pytest.raises(RuntimeError, match="cost budget"):
        poor.complete("GEN", SYSTEM, "1", 10)


def test_long_system_blocks_are_stored_as_hash_and_length(tmp_path: Path) -> None:
    client = _client(tmp_path)
    big = [{"type": "text", "text": "x" * 5000, "cache_control": {"type": "ephemeral"}}]
    client.complete("FC", big, "q", 10)
    block = next(iter(client.records.values()))["request"]["system"][0]
    assert block["chars"] == 5000 and "text" not in block and "cache_control" in block


def test_usage_cost_uses_cache_factors() -> None:
    usage = {
        "input_tokens": 1_000_000,
        "cache_creation_input_tokens": 1_000_000,
        "cache_read_input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
    }
    assert usage_cost(usage) == pytest.approx(3.0 + 3.75 + 0.3 + 15.0)


def test_parse_json_object_tolerates_fences_and_rejects_prose() -> None:
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json_object("no json here")


def test_answerer_retries_once_on_malformed_json(tmp_path: Path) -> None:
    replies = iter(["sorry, here is prose", '{"verdict": "yes", "reason": "r"}'])

    def transport(payload: dict[str, Any], key: str) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": next(replies)}], "usage": USAGE}

    client = CachedMessages(tmp_path / "c.jsonl", api_key="k", transport=transport)
    assert LLMAnswerer(client).judge_correct("q", "ref", "cand").verdict == "yes"
    assert len(client.records) == 2
    broken = CachedMessages(
        tmp_path / "d.jsonl",
        api_key="k",
        transport=lambda p, k: {"content": [{"type": "text", "text": "prose"}], "usage": USAGE},
    )
    with pytest.raises(ValueError, match="no valid JSON"):
        LLMAnswerer(broken).judge_correct("q", "ref", "cand")


def test_context_and_corpus_blocks() -> None:
    assert context_block([Passage("a1", " text "), Passage("a2", "more")]) == (
        "[a1]\ntext\n\n[a2]\nmore"
    )
    assert corpus_block([("a1", "x\n"), ("a2", "y")]) == "=== a1 ===\nx\n\n=== a2 ===\ny"


ITEMS = [
    Item("wanted one", True, "ref one", ("art1",)),
    Item("wanted two", True, "ref two", ("art2",)),
    Item("nothing here", False, kind="national_rule"),
    Item("also nothing", False, kind="missing_number"),
]
PASSAGES = [
    [Passage("art1", "p1")],
    [Passage("art9", "p9")],  # retrieval miss for the second question
    [Passage("art1", "p3")],
    [Passage("art8", "p4")],
]
DOCS = [("art1", "Article one text"), ("art2", "Article two text")]


def _records(tmp_path: Path) -> tuple[list[Any], CachedMessages]:
    client = _client(tmp_path)
    return run_generation(ITEMS, PASSAGES, DOCS, LLMAnswerer(client), workers=2), client


def test_run_generation_collects_every_call_and_flags_invalid_citations(tmp_path: Path) -> None:
    records, client = _records(tmp_path)
    assert [r.retrieval_miss for r in records] == [False, True, False, False]
    first = records[0]
    assert first.gen.supports and first.suff.sufficient and first.fc and first.fc.supports
    assert first.correct_gen and first.correct_gen.verdict == "yes"
    assert first.cites_gen == {"art1": "partial", "nope": "invalid"}
    assert records[2].correct_gen is None  # unanswerable: no correctness judge
    kinds = {r["kind"] for r in client.records.values()}
    assert kinds == {"GEN", "SUFF", "FC", "CORRECT", "CITE"}


def test_decide_and_report_without_reader(tmp_path: Path) -> None:
    records, client = _records(tmp_path)
    decisions = decide(records, None)
    assert list(decisions.answered) == ["self", "suff", "fullctx"]
    assert decisions.answered["suff"].tolist() == [True, False, False, False]
    assert decisions.answered["self"].tolist() == [True, False, True, False]
    text = render_generation(records, decisions, client, title="T")
    for needle in ("Abstention AUROC", "Citation support", "Full context vs RAG", "A2 ", "A6"):
        assert needle in text or needle == "A6"  # A6 needs the reader systems
    assert "retrieval misses" in text and "national_rule" in text


def test_decide_with_reader_adds_the_combined_systems(tmp_path: Path) -> None:
    items = [Item(f"wanted {i}", True, f"ref {i}", ("art1",)) for i in range(6)]
    items += [Item(f"nothing {i}", False, kind="national_rule") for i in range(6)]
    passages = [[Passage("art1" if i % 2 == 0 else "art9", f"p{i}")] for i in range(12)]
    client = _client(tmp_path)
    records = run_generation(items, passages, DOCS, LLMAnswerer(client), workers=2)
    reader = np.array([3.0, 2.5, 2.0, -1.0, 3.5, 1.0, -2.0, -3.0, 0.5, -4.0, -2.5, -3.5])
    decisions = decide(records, reader)
    assert list(decisions.answered) == [
        "reader",
        "self",
        "suff",
        "strict",
        "lenient",
        "stack",
        "fullctx",
    ]
    strict, lenient = decisions.answered["strict"], decisions.answered["lenient"]
    assert (strict <= decisions.answered["suff"]).all() and (lenient >= strict).all()
    text = render_generation(records, decisions, client, title="T")
    assert "Decision rule" in text and "A6" in text


def test_balanced_accuracy_and_paired_ci() -> None:
    labels = np.array([True, True, False, False])
    perfect, always = np.array([True, True, False, False]), np.ones(4, dtype=bool)
    assert balanced_accuracy(perfect, labels) == 1.0
    assert balanced_accuracy(always, labels) == 0.5
    mean, lo, hi = paired_balacc_ci(perfect, always, labels)
    assert mean == pytest.approx(0.5) and lo == pytest.approx(0.5) and hi == pytest.approx(0.5)
    m, _, _ = balacc_ci(perfect, labels)
    assert m == 1.0


def test_items_follow_the_abstention_table_order() -> None:
    items = items_from_records(
        [{"q": "a", "relevant_doc_ids": ["x"], "answer": "r"}], [{"q": "u", "miss_type": "m"}]
    )
    assert [i.answerable for i in items] == [True, False] and items[1].kind == "m"


def test_offline_reader_scores_come_from_a_subset_cache_and_miss_loudly(tmp_path: Path) -> None:
    source = tmp_path / "src.json"
    data = {}
    for item, psg in zip(ITEMS, PASSAGES, strict=True):
        for p in psg:
            data[_key(f"{CE_MODEL}@{CE_REVISION}", item.question, p.text)] = 0.5
            data[_key(f"{QA_MODEL}@{QA_REVISION}", item.question, p.text)] = 1.5
    data["unrelated"] = 9.9
    source.write_text(json.dumps(data))
    out = tmp_path / "sub.json"
    assert subset_score_cache(source, out, ITEMS, PASSAGES) == 8
    scorers = offline_reader_scorers(out)
    assert reader_scores(scorers, ITEMS, PASSAGES).tolist() == [1.5] * 4
    assert "unrelated" not in ScoreCache(out).data
    with pytest.raises(CacheMiss):
        reader_scores(scorers, [Item("unseen", True)], [[Passage("a", "zz")]])


def test_answer_cli_replays_a_prefilled_cache_without_a_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from ragkit.cli import DEFAULT_CORPUS_DIR

    query = "What is imbalance settlement?"
    passages = retrieve_passages(
        RAGPipeline(), _load_documents(DEFAULT_CORPUS_DIR), [Item(query, True)], top_k=5
    )[0]
    cache = tmp_path / "llm.jsonl"
    live = CachedMessages(cache, api_key="k", transport=FakeApi())
    answerer = LLMAnswerer(live)
    answerer.sufficiency(query, passages)
    answerer.generate(query, passages)
    with pytest.raises(SystemExit) as exc:
        main(["answer", query, "--cache", str(cache)])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Sufficient context:" in out and "Answer:" in out
