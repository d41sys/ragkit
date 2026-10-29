"""Offline tests of the independent review tool (no key, no network)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ragkit import hardset_review as hr

RECORD = {
    "q": "How long must X last?",
    "relevant_doc_ids": ["reg_a_art1"],
    "answer": "Two years.",
}
REPLY = (
    '```json\n{"gold_answers": "yes", "gold_reason": "para 2", "other_articles": '
    '[{"doc_id": "reg_b_art9", "answers": "partially", "why": "similar"}], '
    '"ambiguity": "none", "ambiguity_reason": "", "suggested_fix": ""}\n```'
)


def _fake(calls: list[dict[str, Any]]) -> hr.Transport:
    def send(payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        calls.append(payload)
        return {"content": [{"type": "text", "text": REPLY}], "usage": {"input_tokens": 7}}

    return send


def test_corpus_block_and_request_cache_the_corpus_and_hide_nothing_else() -> None:
    corpus = hr.corpus_block([("reg_a_art1", "# A\n\ntext"), ("reg_b_art9", "# B\n\nother")])
    assert corpus.startswith("=== reg_a_art1 ===\n# A")
    payload = hr.build_request(corpus, RECORD)
    assert payload["model"] == "claude-sonnet-5-5"
    assert payload["system"][1]["cache_control"] == {"type": "ephemeral"}
    user = payload["messages"][0]["content"]
    assert "How long must X last?" in user and "reg_a_art1" in user and "Two years." in user
    assert "decoy" not in json.dumps(payload).lower()


def test_parse_verdict_accepts_fenced_json_and_rejects_garbage() -> None:
    verdict = hr.parse_verdict(REPLY)
    assert verdict["gold_answers"] == "yes"
    assert verdict["other_articles"][0]["doc_id"] == "reg_b_art9"
    with pytest.raises(ValueError, match="no JSON"):
        hr.parse_verdict("no object here")
    with pytest.raises(ValueError, match="gold_answers"):
        hr.parse_verdict('{"x": 1}')


def test_review_is_cached_by_request_hash_and_second_call_is_free(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    corpus = hr.corpus_block([("reg_a_art1", "text")])
    first, live1 = hr.review_question(corpus, RECORD, tmp_path, _fake(calls))
    second, live2 = hr.review_question(corpus, RECORD, tmp_path, _fake(calls))
    assert (live1, live2) == (True, False)
    assert first == second and len(calls) == 1
    stored = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert stored["usage"] == {"input_tokens": 7} and stored["question"] == RECORD["q"]
    changed = dict(RECORD, answer="Three years.")
    _, live3 = hr.review_question(corpus, changed, tmp_path, _fake(calls))
    assert live3 and len(calls) == 2
    assert hr.total_usage(tmp_path) == {"input_tokens": 14}


def test_call_budget_is_enforced_over_the_whole_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hr, "MAX_CALLS", 1)
    corpus = hr.corpus_block([("reg_a_art1", "text")])
    hr.review_question(corpus, RECORD, tmp_path, _fake([]))
    with pytest.raises(RuntimeError, match="budget"):
        hr.review_question(corpus, dict(RECORD, answer="Other."), tmp_path, _fake([]))
    verdict, live = hr.review_question(corpus, RECORD, tmp_path, _fake([]))  # cache hit still ok
    assert verdict["gold_answers"] == "yes" and not live


def test_summary_table_lists_other_answering_articles() -> None:
    table = hr.summarise([hr.parse_verdict(REPLY)])
    assert "reg_b_art9 (partially)" in table and "| 0 | yes |" in table
