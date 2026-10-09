"""Query rewriting strategies: HyDE, multi-query, and decomposition."""

from __future__ import annotations

import os
import re
from typing import Protocol, runtime_checkable

from ragkit.bm25 import tokenize


@runtime_checkable
class LLM(Protocol):
    """Protocol for language model generation."""

    def generate(self, prompt: str) -> str:
        """Generate response text for a given prompt."""
        ...


class TemplateLLM:
    """Offline deterministic LLM using structured templates."""

    def __init__(self, domain: str = "energy") -> None:
        self.domain = domain

    def _extract_subject(self, prompt: str) -> str:
        # Extract query or topic from prompt
        match = re.search(r"['\"]([^'\"]+)['\"]", prompt)
        if match:
            return match.group(1).strip()
        lines = prompt.strip().splitlines()
        last_line = lines[-1].strip() if lines else prompt
        words = tokenize(last_line)
        return " ".join(words[-4:]) if words else "system flexibility"

    def generate(self, prompt: str) -> str:
        subject = self._extract_subject(prompt)
        p_lower = prompt.lower()

        if "hypothetical" in p_lower or "passage" in p_lower or "hyde" in p_lower:
            return (
                f"In energy networks and flexibility markets, {subject} provides "
                "dispatchable capacity and balancing services. Operators use telemetry, "
                "day-ahead schedules, and automated control signals to resolve grid "
                "congestion, balance supply and demand, and minimize imbalance settlement "
                "penalties across transmission and distribution interfaces."
            )

        if "sub-questions" in p_lower or "decompose" in p_lower:
            return (
                "1. What are the operational requirements and telemetry standards for "
                f"{subject}?\n"
                "2. How does the market clear and settle financial imbalances related to "
                f"{subject}?\n"
                "3. What asset types and control algorithms are deployed to optimize "
                f"{subject}?"
            )

        if "alternative" in p_lower or "multi-query" in p_lower or "variations" in p_lower:
            return (
                f"1. How does {subject} operate in electricity balancing and ancillary "
                "service markets?\n"
                "2. Technical specification, dispatch constraints, and economic incentives "
                f"for {subject}.\n"
                f"3. Grid code compliance, settlement mechanisms, and scheduling for {subject}."
            )

        return f"Operational analysis and settlement standards governing {subject}."


class AnthropicLLM:
    """Anthropic Claude LLM provider for online query rewriting and answering."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-latest",
        api_key: str | None = None,
    ) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it or use TemplateLLM for offline runs."
            )

        try:
            import anthropic
        except ImportError as err:
            raise ImportError(
                "anthropic package not installed. Install with: pip install 'ragkit[llm]'"
            ) from err

        self.client = anthropic.Anthropic(api_key=key)
        self.model = model

    def generate(self, prompt: str) -> str:
        """Call Anthropic Claude API."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        content_block = response.content[0]
        text_val = getattr(content_block, "text", "")
        return str(text_val)


class HyDERewriter:
    """Hypothetical Document Embeddings: rewrites query into an idealized answer."""

    def __init__(self, llm: LLM | None = None) -> None:
        self.llm: LLM = llm if llm is not None else TemplateLLM()

    def rewrite(self, query: str) -> str:
        prompt = (
            f"Write a comprehensive technical passage answering the question: '{query}'. "
            f"Include exact domain terminology, operational details, and market mechanisms."
        )
        return self.llm.generate(prompt)


class MultiQueryRewriter:
    """Multi-Query expansion: generates diverse formulations of the original query."""

    def __init__(self, llm: LLM | None = None, num_queries: int = 3) -> None:
        self.llm: LLM = llm if llm is not None else TemplateLLM()
        self.num_queries = num_queries

    def rewrite(self, query: str) -> list[str]:
        prompt = (
            f"Generate {self.num_queries} alternative search queries from different perspectives "
            f"for the user question: '{query}'. Return one query per line."
        )
        raw = self.llm.generate(prompt)
        lines = [re.sub(r"^\d+[\.\)]\s*", "", line).strip() for line in raw.splitlines()]
        cleaned = [ln for ln in lines if ln]
        # Always include original query first
        results = [query]
        for q in cleaned:
            if q.lower() != query.lower() and q not in results:
                results.append(q)
        return results[: self.num_queries + 1]


class DecompositionRewriter:
    """Decomposes a complex query into independent sub-questions."""

    def __init__(self, llm: LLM | None = None) -> None:
        self.llm: LLM = llm if llm is not None else TemplateLLM()

    def rewrite(self, query: str) -> list[str]:
        prompt = (
            "Decompose the following complex query into 2 to 3 distinct sub-questions: "
            f"'{query}'. "
            f"Return one sub-question per line."
        )
        raw = self.llm.generate(prompt)
        lines = [re.sub(r"^\d+[\.\)]\s*", "", line).strip() for line in raw.splitlines()]
        cleaned = [ln for ln in lines if ln]
        return cleaned if cleaned else [query]
