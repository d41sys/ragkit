"""Offline checks on the bundled EUR-Lex corpus and its agent-drafted QA labels."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ragkit.cli import (
    BUNDLED_DATASETS,
    EU_CORPUS_DIR,
    EU_QA_PATH,
    EU_UNANSWERABLE_PATH,
    _load_documents,
)

_QUOTE = re.compile(r"“([^”]+)”")
_QUESTION_TYPES = {"direct", "paraphrase", "multi-hop"}


def _norm(text: str) -> str:
    """Whitespace- and case-insensitive form: a quote may start mid-sentence in lower case."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture(scope="module")
def docs() -> dict[str, str]:
    return dict(_load_documents(EU_CORPUS_DIR))


def test_eu_dataset_is_registered() -> None:
    ds = BUNDLED_DATASETS["eu"]
    assert (ds.corpus_dir, ds.qa_path, ds.unanswerable_path) == (
        EU_CORPUS_DIR,
        EU_QA_PATH,
        EU_UNANSWERABLE_PATH,
    )


def test_eu_corpus_size_and_provenance_manifest(docs: dict[str, str]) -> None:
    assert 40 <= len(docs) <= 120
    manifest = json.loads((EU_CORPUS_DIR / "SOURCES.json").read_text(encoding="utf-8"))
    assert "2011/833/EU" in manifest["licence"]
    kept = sum(len(act["articles_kept"]) for act in manifest["acts"])
    assert kept == len(docs)
    for act in manifest["acts"]:
        assert act["url"].startswith("https://publications.europa.eu/resource/celex/")
        assert len(act["sha256"]) == 64


def test_eu_documents_have_no_footnote_residue_or_metadata(docs: dict[str, str]) -> None:
    for doc_id, text in docs.items():
        assert not text.lstrip().startswith("<!--"), doc_id
        assert "()" not in text, doc_id
        assert text.lstrip().startswith("# "), doc_id


def test_eu_questions_are_wellformed_and_labels_resolve(docs: dict[str, str]) -> None:
    qa = _jsonl(EU_QA_PATH)
    assert len(qa) >= 40
    assert sum(item["type"] == "multi-hop" for item in qa) >= 8
    assert {item["type"] for item in qa} == _QUESTION_TYPES
    questions = [item["q"] for item in qa]
    assert len(set(questions)) == len(questions)
    for item in qa:
        assert item["status"] == "draft-agent"
        relevant = item["relevant_doc_ids"]
        assert isinstance(relevant, list)
        assert relevant and set(relevant) <= docs.keys(), item["q"]
        if item["type"] == "multi-hop":
            assert len(relevant) >= 2
        provenance = item["provenance"]
        assert isinstance(provenance, list)
        assert {p.split("#")[0] for p in provenance} == set(relevant), item["q"]


def test_eu_reference_answers_quote_the_cited_articles(docs: dict[str, str]) -> None:
    for item in _jsonl(EU_QA_PATH):
        relevant = item["relevant_doc_ids"]
        assert isinstance(relevant, list)
        source = _norm(" ".join(docs[d] for d in relevant))
        quotes = _QUOTE.findall(str(item["answer"]))
        assert quotes, item["q"]
        for quote in quotes:
            assert _norm(quote) in source, (item["q"], quote)


def test_eu_unanswerable_questions_are_on_topic_drafts(docs: dict[str, str]) -> None:
    unanswerable = _jsonl(EU_UNANSWERABLE_PATH)
    assert len(unanswerable) >= 12
    answerable = {str(item["q"]) for item in _jsonl(EU_QA_PATH)}
    for item in unanswerable:
        assert item["kind"] == "near_domain"
        assert item["status"] == "draft-agent"
        assert item["q"] not in answerable
        provenance = item["provenance"]
        assert isinstance(provenance, list) and provenance
        assert {p.split("#")[0] for p in provenance} <= docs.keys()
