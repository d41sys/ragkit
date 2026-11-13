"""Tests for ``ragkit.agentbench``: setups, citation extraction, truncation guard, decision rule."""

import io
import json
from typing import Any

import numpy as np
import pytest

import ragkit
from ragkit import agentbench, llm
from ragkit.agentbench import Scored


def _article(i: int) -> dict[str, Any]:
    return {"id": f"reg_2019_943_art{i}", "text": "x" * 400, "rank": i}


@pytest.fixture
def fake_search(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    class Result:
        def to_dict(self) -> dict[str, Any]:
            return {"retriever": "hybrid+ce", "articles": [_article(i) for i in range(1, 8)]}

    def search(query: str, **kwargs: Any) -> Result:
        calls.append(kwargs)
        return Result()

    monkeypatch.setattr(ragkit, "search", search)
    return calls


def test_setups_shape_the_packed_list(fake_search: list[dict[str, Any]]) -> None:
    a = agentbench.search_for_setup("A", "q", None)
    b = agentbench.search_for_setup("B", "q", 4000)
    c = agentbench.search_for_setup("C", "q", None)
    assert [x["rank"] for x in a["articles"]] == [1, 2, 3, 4, 5, 6, 7]
    assert [x["rank"] for x in b["articles"]] == [1, 2, 3]
    assert [x["rank"] for x in c["articles"]] == [7, 6, 5, 4, 3, 2, 1]
    assert b["estimated_tokens"] == 300
    assert fake_search == [{}, {"pack_chunks": 5, "budget_tokens": 4000}, {}]


def test_cited_articles_ids_then_natural_references() -> None:
    corpus = {"reg_2019_943_art16", "dir_2019_944_art15", "reg_2019_943_art17"}
    assert agentbench.cited_articles("see reg_2019_943_art16 and dir_2019_999_art1", corpus) == [
        "reg_2019_943_art16",
        "dir_2019_999_art1",
    ]
    assert agentbench.cited_articles(
        "**Article 16(8)** of **Regulation (EU) 2019/943**", corpus
    ) == ["reg_2019_943_art16"]
    assert agentbench.cited_articles("Directive (EU) 2019/944, Article 15", corpus) == [
        "dir_2019_944_art15"
    ]
    assert agentbench.cited_articles("Article 99 of Regulation (EU) 2019/943", corpus) == []


def test_truncation_guard() -> None:
    ok = agentbench.truncation_problem(10_000, 500, 50_000, 65_536)
    assert ok is None
    assert "fills num_ctx" in str(agentbench.truncation_problem(4096, 10, 60_000, 4096))
    assert "fill num_ctx" in str(agentbench.truncation_problem(30_000, 3_000, 150_000, 32_768))
    assert "below 0.8" in str(agentbench.truncation_problem(3_000, 10, 60_000, 65_536))


def test_ollama_transport_sends_options_and_maps_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict[str, Any] = {}

    class Response(io.BytesIO):
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def urlopen(request: Any, timeout: int) -> Response:
        sent.update(json.loads(request.data), url=request.full_url)
        reply = {"message": {"content": "hi"}, "prompt_eval_count": 7, "eval_count": 2}
        return Response(json.dumps(reply).encode())

    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    send = llm.ollama_transport("http://h:1/", num_ctx=8192, seed=3)
    payload = {
        "model": "m",
        "max_tokens": 100,
        "temperature": 0.0,
        "system": [{"text": "sys"}],
        "messages": [{"role": "user", "content": "u"}],
    }
    reply = send(payload, "k")
    assert sent["url"] == "http://h:1/api/chat"
    assert sent["options"] == {"num_ctx": 8192, "seed": 3, "num_predict": 100, "temperature": 0.0}
    assert sent["messages"][0] == {"role": "system", "content": "sys"}
    assert reply["content"] == [{"type": "text", "text": "hi"}]
    assert reply["usage"] == {"input_tokens": 7, "output_tokens": 2}
    with pytest.raises(RuntimeError, match="does not fit"):
        send({**payload, "messages": [{"role": "user", "content": "u" * 30_000}]}, "k")


def _scored(setup: str, correct: list[float], refused: list[float], failed: int = 0) -> Scored:
    """Scores of ``len(correct)`` answerable then ``len(refused)`` unanswerable questions."""
    n = len(correct) + len(refused)
    nan = np.full(n, np.nan)
    return Scored(
        setup=setup,
        failed=failed,
        correct=np.array(correct + [np.nan] * len(refused)),
        partial=nan,
        refused_answerable=nan,
        refused_unanswerable=np.array([np.nan] * len(correct) + refused),
        cite_yes=nan,
        cited=nan,
        no_citation=nan,
        context_tokens=nan,
        latency_s=nan,
        tool_called=nan,
        searches=nan,
        max_prompt=nan,
        runaways=nan,
        verdicts=[],
    )


def test_decision_rule() -> None:
    a = _scored("A", [0.0] * 30 + [1.0] * 10, [1.0] * 30 + [0.0] * 10)
    better = _scored("B", [0.0] * 20 + [1.0] * 20, [1.0] * 30 + [0.0] * 10)
    lossy = _scored("C", [0.0] * 20 + [1.0] * 20, [1.0] * 20 + [0.0] * 20)
    default, lines = agentbench.decide({"A": a, "B": better, "C": lossy})
    assert default == "B"
    assert any(line.startswith("C:") and "refusal" in line for line in lines)
    same = _scored("B", [0.0] * 30 + [1.0] * 10, [1.0] * 30 + [0.0] * 10)
    assert agentbench.decide({"A": a, "B": same})[0] == "A"
    flaky = _scored("B", [0.0] * 20 + [1.0] * 20, [1.0] * 30 + [0.0] * 10, failed=4)
    assert agentbench.decide({"A": a, "B": flaky})[0] == "A"
