"""Tests for the LegalBench-RAG-mini sampler, the character-overlap metric and the systems."""

import json
import random
import zipfile
from pathlib import Path

import pytest

from ragkit.cli import main
from ragkit.legalbench import (
    KS,
    SOURCES,
    SYSTEMS,
    char_precision,
    char_recall,
    evaluate_system,
    load_mini,
    render_legalbench_report,
    sample_tests,
)


def test_char_precision_and_recall_follow_the_official_definition() -> None:
    truth = [("a.txt", 10, 20), ("a.txt", 30, 40)]
    retrieved = [("a.txt", 0, 15), ("b.txt", 10, 20), ("a.txt", 35, 60)]
    # overlaps: 5 (a 0-15 vs 10-20) + 5 (a 35-60 vs 30-40); b.txt never matches
    assert char_precision(retrieved, truth) == pytest.approx(10 / (15 + 10 + 25))
    assert char_recall(retrieved, truth) == pytest.approx(10 / 20)
    assert char_precision([], truth) == 0.0
    assert char_recall(retrieved, []) == 0.0


def test_overlapping_retrieved_chunks_are_double_counted_like_the_official_code() -> None:
    truth = [("a.txt", 0, 10)]
    retrieved = [("a.txt", 0, 10), ("a.txt", 0, 10)]
    assert char_recall(retrieved, truth) == 2.0
    assert char_precision(retrieved, truth) == 1.0


def test_sample_tests_keeps_everything_when_under_the_cap() -> None:
    records = [{"query": str(i), "snippets": [{"file_path": f"f{i}"}]} for i in range(5)]
    assert sample_tests(records) == records


def test_sample_tests_sorts_by_path_seeded_key_and_clusters_by_document() -> None:
    records = [
        {"query": f"q{i}", "snippets": [{"file_path": f"doc{i % 300}.txt"}]} for i in range(600)
    ]
    kept = sample_tests(records)
    assert len(kept) == 194
    keys = [random.Random(r["snippets"][0]["file_path"]).random() for r in kept]
    assert keys == sorted(keys)
    # same document -> same key -> original order is kept (stable sort)
    same = [
        r["query"]
        for r in kept
        if r["snippets"][0]["file_path"] == kept[0]["snippets"][0]["file_path"]
    ]
    assert same == sorted(same, key=lambda q: int(q[1:]))


def _make_zip(path: Path) -> None:
    corpus = {
        "privacy_qa/a.txt": "We collect your email address and share it with partners.\r\n" * 3,
        "contractnli/b.txt": "The receiving party shall keep all confidential information secret.\n"
        * 4,
        "maud/c.txt": "The merger consideration is paid in cash at closing.\n" * 4,
        "cuad/d.txt": "This agreement is governed by the laws of the State of Delaware.\n" * 4,
    }
    benchmarks = {}
    for source, file in zip(SOURCES, corpus, strict=True):
        text = corpus[file].replace("\r\n", "\n")
        benchmarks[source] = {
            "tests": [
                {
                    "query": f"Consider the document; {text.split('.')[0]}?",
                    "snippets": [{"file_path": file, "span": [0, 40], "answer": text[:40]}],
                }
            ]
        }
    with zipfile.ZipFile(path, "w") as zf:
        for file, text in corpus.items():
            zf.writestr(f"corpus/{file}", text.encode())
        for source, payload in benchmarks.items():
            zf.writestr(f"benchmarks/{source}.json", json.dumps(payload))


def test_load_mini_reads_universal_newlines_and_pools_the_corpus(tmp_path: Path) -> None:
    zip_path = tmp_path / "lbr.zip"
    _make_zip(zip_path)
    bench = load_mini(zip_path, download=False)
    assert len(bench.tests) == 4
    assert set(bench.corpus) == {
        "privacy_qa/a.txt",
        "contractnli/b.txt",
        "maud/c.txt",
        "cuad/d.txt",
    }
    assert "\r" not in bench.corpus["privacy_qa/a.txt"]
    assert bench.docs_per_source() == {s: 1 for s in SOURCES}


def test_every_system_scores_and_finds_the_obvious_snippet(tmp_path: Path) -> None:
    zip_path = tmp_path / "lbr.zip"
    _make_zip(zip_path)
    bench = load_mini(zip_path, download=False)
    results = [evaluate_system(system, bench) for system in SYSTEMS]
    for res in results:
        assert res.all_mean("recall")[KS.index(64)] == pytest.approx(1.0), res.name
        assert all(0.0 <= v <= 1.0 for v in res.all_mean("precision")), res.name
        lo_hi = res.all_ci("recall")
        assert len(lo_hi) == len(KS)
    report = render_legalbench_report(bench, results)
    assert "MISMATCH" in report  # four tiny documents are not the paper's sample
    assert "bm25-naive500" in report


def test_cli_legalbench_runs_and_rejects_unknown_systems(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    zip_path = tmp_path / "lbr.zip"
    _make_zip(zip_path)
    with pytest.raises(SystemExit) as ok:
        main(
            [
                "legalbench",
                "--data-zip",
                str(zip_path),
                "--no-download",
                "--systems",
                "bm25-naive500",
            ]
        )
    assert ok.value.code == 0
    assert "LegalBench-RAG-mini" in capsys.readouterr().out
    with pytest.raises(SystemExit) as bad:
        main(["legalbench", "--data-zip", str(zip_path), "--no-download", "--systems", "x"])
    assert bad.value.code == 2
