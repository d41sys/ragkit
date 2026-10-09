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
