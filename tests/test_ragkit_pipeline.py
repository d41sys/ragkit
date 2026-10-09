"""Tests for pipeline answering, citation generation, and query rewriters."""

import pytest
from ragkit.chunking import Chunk
from ragkit.pipeline import ExtractiveGenerator, RAGPipeline
from ragkit.rewrite import (
    AnthropicLLM,
    DecompositionRewriter,
    HyDERewriter,
    MultiQueryRewriter,
    TemplateLLM,
)


def test_extractive_generator_citations() -> None:
    c1 = Chunk(
        id="doc_a#0",
        doc_id="doc_a",
        text="Imbalance settlement is the commercial accounting framework for grid deviations.",
        start=0,
        end=81,
    )
    c2 = Chunk(
        id="doc_b#0",
        doc_id="doc_b",
        text="Balance Responsible Parties face dual pricing penalties during system short events.",
        start=0,
        end=83,
    )

    gen = ExtractiveGenerator(max_sentences=2)
    ans = gen.generate("What is imbalance settlement?", [c1, c2])

    assert "[doc_a#0]" in ans.text
    assert "doc_a#0" in ans.citations


def test_rag_pipeline_end_to_end() -> None:
    pipeline = RAGPipeline()
    docs = [
        (
            "doc_bess",
            "Battery Energy Storage Systems provide primary frequency containment reserves. "
            "SoC restoration algorithms maintain cell state near fifty percent.",
        ),
        (
            "doc_pv",
            "Photovoltaic plants manage cloud ramp rate fluctuations using battery buffers. "
            "Inverters can maintain synthetic curtailment headroom for upward reserves.",
        ),
    ]
    pipeline.index_documents(docs)

    answer = pipeline.answer("How do solar inverters manage cloud ramps?")
    assert len(answer.citations) > 0
    assert any("doc_pv" in cid for cid in answer.citations)
    assert "[doc_pv" in answer.text


def test_template_llm_rewriters() -> None:
    llm = TemplateLLM()
    hyde = HyDERewriter(llm=llm)
    mq = MultiQueryRewriter(llm=llm, num_queries=3)
    decomp = DecompositionRewriter(llm=llm)

    q = "What is demand response?"
    hypo = hyde.rewrite(q)
    assert "demand response" in hypo.lower() or "flexibility" in hypo.lower()

    queries = mq.rewrite(q)
    assert len(queries) >= 2
    assert queries[0] == q

    sub_q = decomp.rewrite(q)
    assert len(sub_q) >= 2


def test_anthropic_llm_missing_key_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is not set"):
        AnthropicLLM()
