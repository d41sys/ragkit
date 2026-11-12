from __future__ import annotations

from pathlib import Path

import pytest

from ragkit.cli import main


def test_index_then_ask_from_persisted_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "bess.md").write_text(
        "Battery dispatch schedules charging when prices are low. "
        "Batteries discharge during evening price peaks.",
        encoding="utf-8",
    )
    (corpus / "hp.md").write_text(
        "Heat pumps shift load by preheating buildings before tariff peaks.", encoding="utf-8"
    )
    index = tmp_path / "idx.json"
    with pytest.raises(SystemExit) as ex:
        main(["index", str(corpus), "--out", str(index)])
    assert ex.value.code == 0
    # Delete the corpus: ask must answer from the persisted index alone.
    for f in corpus.iterdir():
        f.unlink()
    with pytest.raises(SystemExit) as ex:
        main(["ask", "When do heat pumps preheat buildings?", "--index", str(index)])
    assert ex.value.code == 0
    out = capsys.readouterr().out
    assert "[hp#" in out


def test_ask_uses_the_corpus_preset(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as ex:
        main(["ask", "What is the minimum cross-zonal capacity for trade?", "--corpus", "eu"])
    assert ex.value.code == 0
    assert "Citations: reg_" in capsys.readouterr().out


def test_dataset_flags_select_bundled_preset_and_explicit_overrides(tmp_path: Path) -> None:
    from ragkit.cli import (
        DEFAULT_CORPUS_DIR,
        DEFAULT_QA_PATH,
        EU_CORPUS_DIR,
        EU_QA_PATH,
        _resolve_dataset,
    )

    def resolve(*argv: str) -> tuple[Path, Path]:
        import argparse

        from ragkit.cli import _add_dataset_args

        parser = argparse.ArgumentParser()
        _add_dataset_args(parser)
        ds = _resolve_dataset(parser.parse_args(argv))
        return ds.corpus_dir, ds.qa_path

    assert resolve() == (DEFAULT_CORPUS_DIR, DEFAULT_QA_PATH)
    assert resolve("--corpus", "eu") == (EU_CORPUS_DIR, EU_QA_PATH)
    custom = str(tmp_path / "docs")
    assert resolve("--corpus", "eu", "--corpus-dir", custom) == (Path(custom), EU_QA_PATH)


def test_eval_ablation_and_abstention_on_custom_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text(
        "<!-- source: https://example.org/a -->\n\n# A\n\n"
        "Congestion is relieved by redispatching generation units near the constraint.",
        encoding="utf-8",
    )
    (corpus / "b.md").write_text(
        "# B\n\nImbalance settlement charges parties for deviations from their schedules.",
        encoding="utf-8",
    )
    qa = tmp_path / "qa.jsonl"
    qa.write_text(
        "\n".join(
            json.dumps(item)
            for item in [
                {
                    "q": "How is congestion relieved?",
                    "relevant_doc_ids": ["a"],
                    "answer": "redispatching",
                    "type": "direct",
                },
                {
                    "q": "What does imbalance settlement charge?",
                    "relevant_doc_ids": ["b"],
                    "answer": "deviations",
                    "type": "paraphrase",
                },
            ]
        ),
        encoding="utf-8",
    )
    unans = tmp_path / "unans.jsonl"
    unans.write_text(
        json.dumps({"q": "Who won the football final?", "kind": "off_domain"}), encoding="utf-8"
    )
    flags = ["--corpus-dir", str(corpus), "--qa-file", str(qa), "--unanswerable-file", str(unans)]

    for argv in (
        ["eval", "--strategies", "bm25,hybrid", *flags],
        ["ablation", *flags],
        ["abstention", *flags],
    ):
        with pytest.raises(SystemExit) as ex:
            main(argv)
        assert ex.value.code == 0

    out = capsys.readouterr().out
    assert "2 questions" in out
    assert "Paired difference vs BM25" in out
    assert "direct (n=1)" in out and "paraphrase (n=1)" in out
    assert "question terms in best relevant chunk" in out
    assert "Wilson 95% CI" in out
    assert "answerable paraphrase (answered)" in out


def test_eval_on_bundled_eu_corpus_runs_offline(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as ex:
        main(["eval", "--corpus", "eu", "--strategies", "bm25"])
    assert ex.value.code == 0
    out = capsys.readouterr().out
    assert "(42 questions" in out
    assert "multi-hop (n=8)" in out
