"""Tests for the BEIR loader, trec_eval-style metrics and the model-free systems."""

import json
import math
import zipfile
from pathlib import Path

import pytest

from ragkit.analysis import lucene_english
from ragkit.beir import (
    LADDER,
    SYSTEMS,
    BeirData,
    FlatBM25,
    MultiFieldBM25,
    evaluate_system,
    load_beir,
    ndcg_cut,
    pyserini_rows,
    recall_at,
    render_beir_report,
)
from ragkit.cli import main

DOCS = [
    ("d1", "Cats", "Cats are small domestic animals that purr"),
    ("d2", "Dogs", "Dogs are loyal domestic animals that bark"),
    ("d3", "", "The stock market fell sharply on Monday"),
    ("d4", "Markets", "Stock markets rise and fall with interest rates"),
]


def _make_zip(root: Path, name: str = "tiny") -> None:
    corpus = "\n".join(
        json.dumps({"_id": i, "title": t, "text": x, "metadata": {}}) for i, t, x in DOCS
    )
    queries = "\n".join(
        json.dumps({"_id": q, "text": t})
        for q, t in [("q1", "domestic animals"), ("q2", "stock market"), ("q3", "nothing")]
    )
    rows = ["q1\td1\t1", "q1\td2\t1", "q2\td3\t2", "q2\td4\t1", "q2\td1\t0", "q3\td1\t0"]
    qrels = "\n".join(["query-id\tcorpus-id\tscore", *rows]) + "\n"
    with zipfile.ZipFile(root / f"{name}.zip", "w") as zf:
        zf.writestr(f"{name}/corpus.jsonl", corpus)
        zf.writestr(f"{name}/queries.jsonl", queries)
        zf.writestr(f"{name}/qrels/test.tsv", qrels)


def test_ndcg_cut_uses_linear_gain_like_trec_eval() -> None:
    rels = {"a": 2, "b": 1}
    dcg = 1 / math.log2(2) + 2 / math.log2(4)
    ideal = 2 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_cut(["b", "x", "a"], rels) == pytest.approx(dcg / ideal)
    assert ndcg_cut(["a", "b"], rels) == pytest.approx(1.0)
    assert ndcg_cut(["x", "y"], rels) == 0.0
    assert ndcg_cut(["a"], {}) == 0.0


def test_ndcg_cut_cut_off_and_ideal_depth() -> None:
    rels = {f"d{i}": 1 for i in range(20)}
    ranked = [f"d{i}" for i in range(20)]
    assert ndcg_cut(ranked, rels, k=10) == pytest.approx(1.0)
    assert ndcg_cut([f"x{i}" for i in range(10)] + ranked, rels, k=10) == 0.0


def test_recall_at_counts_all_relevant_in_denominator() -> None:
    rels = {"a": 1, "b": 2, "c": 1}
    assert recall_at(["a", "x", "b"], rels, k=3) == pytest.approx(2 / 3)
    assert recall_at(["a", "x", "b"], rels, k=1) == pytest.approx(1 / 3)
    assert recall_at(["a"], {}, k=3) == 0.0


def test_load_beir_keeps_positive_qrels_and_drops_empty_queries(tmp_path: Path) -> None:
    _make_zip(tmp_path)
    data = load_beir("tiny", tmp_path, download=False)
    assert data.doc_ids == ["d1", "d2", "d3", "d4"]
    assert set(data.queries) == {"q1", "q2"}
    assert data.qrels == {"q1": {"d1": 1, "d2": 1}, "q2": {"d3": 2, "d4": 1}}
    assert data.has_titles


def test_load_beir_missing_zip_without_download_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_beir("tiny", tmp_path, download=False)


def _data(tmp_path: Path) -> BeirData:
    _make_zip(tmp_path)
    return load_beir("tiny", tmp_path, download=False)


def test_every_system_and_ladder_row_runs_and_scores_the_tiny_set(tmp_path: Path) -> None:
    data = _data(tmp_path)
    for spec in (*SYSTEMS, *LADDER):
        result = evaluate_system(spec, data)
        assert len(result.ndcg) == 2
        assert all(0.0 <= v <= 1.0 for v in result.ndcg + result.recall), spec.name
        assert result.recall == [1.0, 1.0], spec.name


def test_self_hit_is_excluded_from_the_ranking(tmp_path: Path) -> None:
    data = _data(tmp_path)
    retriever = FlatBM25(data, k1=1.5, b=0.75)
    assert "d1" in [d for d, _ in retriever.search("cats animals", 10)]
    assert "d1" not in [d for d, _ in retriever.search("cats animals", 10, exclude_id="d1")]


def test_multifield_sums_title_and_text_scores_and_skips_empty_titles(tmp_path: Path) -> None:
    data = _data(tmp_path)
    mf = MultiFieldBM25(data, k1=0.9, b=0.4, analyzer=lucene_english)
    assert len(mf.title_rows) == 3  # d3 has no title
    # "dogs" is in the title and in the text of d2 only: both fields add to d2's score
    hits = dict(mf.search("dogs", 10))
    assert set(hits) == {"d2"}
    text_score = mf.text_index.score_all("dogs")[1]  # text field rows: d1, d2, d3, d4
    title_score = mf.title_index.score_all("dogs")[1]  # title rows: d1, d2, d4
    assert text_score > 0.0 and title_score > 0.0
    assert hits["d2"] == pytest.approx(text_score + title_score)


def test_report_and_published_rows(tmp_path: Path) -> None:
    data = _data(tmp_path)
    result = evaluate_system(SYSTEMS[0], data)
    text = render_beir_report(data, [result], title="t")
    assert "bm25-ragkit" in text and "nDCG@10" in text
    assert pyserini_rows("scifact")[0] == ("Pyserini BM25 flat", 0.679, 0.925)
    assert pyserini_rows("tiny") == []


def test_cli_beir_runs_on_a_local_zip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _make_zip(tmp_path, "scifact")
    with pytest.raises(SystemExit) as exc:
        main(["beir", "--dataset", "scifact", "--data-dir", str(tmp_path), "--no-download"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "ragkit beir: scifact" in out
    assert "bm25-lucene" in out and "Pyserini BM25 flat" in out


def test_cli_beir_rejects_unknown_system(tmp_path: Path) -> None:
    _make_zip(tmp_path, "scifact")
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "beir",
                "--data-dir",
                str(tmp_path),
                "--no-download",
                "--systems",
                "nope",
            ]
        )
    assert exc.value.code == 2
