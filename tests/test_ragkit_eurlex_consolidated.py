"""Offline tests for the consolidated-text parser and the bundled consolidated corpus (issue #4)."""

from __future__ import annotations

import json
from pathlib import Path

from ragkit.cli import BUNDLED_DATASETS, _load_documents
from ragkit.eurlex import Act, Amendment, build_corpus, parse_consolidated_articles

DATA = Path(__file__).parent.parent / "src" / "ragkit" / "data"

# Mimics the EUR-Lex consolidated XHTML: numbered paragraphs in div.norm, points in grid-list
# divs (nested), amendment markers in p.modref, footnote references as superscripts.
CONSOLIDATED = """<html><body>
<p class="arrow"><a>►M1</a></p>
<div class="eli-subdivision" id="art_1">
  <p class="title-article-norm">Article 1</p>
  <div class="eli-title" id="art_1.tit_1"><p class="stitle-article-norm">Subject matter</p></div>
  <div class="norm"><span class="no-parag">1.  </span>
    <div class="norm inline-element">This Regulation applies<a>
      <span class="superscript">1</span></a> here.</div></div>
  <p class="modref"><a>▼M2</a></p>
  <div class="norm"><span class="no-parag">1a.  </span>
    <div class="norm inline-element">
      <p class="norm inline-element">It aims to:</p>
      <div class="grid-container grid-list">
        <div class="list grid-list-column-1"><span>(a) </span></div>
        <div class="grid-list-column-2"><p class="norm">be fast;</p>
          <div class="grid-container grid-list">
            <div class="list grid-list-column-1"><span>(i) </span></div>
            <div class="grid-list-column-2"><p class="norm">and cheap ►M2 today◄;</p></div>
          </div></div>
      </div>
    </div></div>
  <p class="norm">Unnumbered closing paragraph.</p>
</div>
<div class="eli-subdivision" id="art_7a">
  <p class="title-article-norm">Article 7a</p>
  <div class="eli-title"><p class="stitle-article-norm">Inserted</p></div>
  <table><tbody><tr>
    <td><p class="tbl-norm">Row</p></td><td><p class="tbl-norm">cells</p></td>
  </tr></tbody></table>
</div>
<p class="footnote">(<a><span class="superscript">1</span></a>) OJ L 1, p. 1.</p>
</body></html>"""

ACT = Act(
    "reg_2099_1",
    "32099R0001",
    "Regulation (EU) 2099/1 (test)",
    frozenset({"1"}),
    "02099R0001-20260102",
    (Amendment("32099R0002", "Regulation (EU) 2099/2", "https://example.org/amending"),),
)


def test_parse_consolidated_flattens_paragraphs_points_and_strips_markers() -> None:
    arts = parse_consolidated_articles(CONSOLIDATED)
    assert [(a.number, a.title) for a in arts] == [("1", "Subject matter"), ("7a", "Inserted")]
    assert arts[0].lines == (
        "1. This Regulation applies here.",
        "1a. It aims to:",
        "(a) be fast;\n  (i) and cheap today;",
        "Unnumbered closing paragraph.",
    )
    assert arts[1].lines == ("Row cells",)


def test_build_corpus_consolidated_records_version_and_amendments(tmp_path: Path) -> None:
    fetched: list[str] = []

    def fake_fetch(celex: str) -> bytes:
        fetched.append(celex)
        return CONSOLIDATED.encode("utf-8")

    out = tmp_path / "out"
    build_corpus(
        [ACT],
        out,
        cache_dir=tmp_path / "cache",
        fetch=fake_fetch,
        retrieved_on="2026-11-13",
        consolidated=True,
    )
    assert fetched == ["02099R0001-20260102"]
    entry = json.loads((out / "SOURCES.json").read_text(encoding="utf-8"))["acts"][0]
    assert entry["celex"] == "32099R0001"
    assert entry["consolidated_celex"] == "02099R0001-20260102"
    assert entry["consolidated_date"] == "2026-01-02"
    assert entry["url"].endswith("/02099R0001-20260102")
    assert entry["amended_by"] == [
        {
            "celex": "32099R0002",
            "title": "Regulation (EU) 2099/2",
            "url": "https://example.org/amending",
        }
    ]
    assert "02099R0001-20260102" in (out / "reg_2099_1_art1.md").read_text(encoding="utf-8")


def test_bundled_consolidated_corpus_matches_the_original_article_for_article() -> None:
    original = json.loads((DATA / "corpus_eu" / "SOURCES.json").read_text(encoding="utf-8"))
    cons = json.loads(
        (DATA / "corpus_eu_consolidated" / "SOURCES.json").read_text(encoding="utf-8")
    )
    assert [a["short"] for a in original["acts"]] == [a["short"] for a in cons["acts"]]
    for o, c in zip(original["acts"], cons["acts"], strict=True):
        assert o["articles_kept"] == c["articles_kept"]
        assert o["amended_by"] == c["amended_by"] and o["amended_by"]
        assert c["consolidated_celex"].endswith(c["consolidated_date"].replace("-", ""))
        assert "consolidated_celex" not in o
    docs = dict(_load_documents(DATA / "corpus_eu_consolidated"))
    assert set(docs) == {d for d, _ in _load_documents(DATA / "corpus_eu")}
    assert not any(ch in text for text in docs.values() for ch in "▼►◄")


def test_consolidated_preset_files_exist_and_are_consistent() -> None:
    ds = BUNDLED_DATASETS["eu-consolidated"]
    docs = {d for d, _ in _load_documents(ds.corpus_dir)}
    for path in (ds.qa_path, ds.unanswerable_path):
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            assert set(rec.get("relevant_doc_ids", [])) <= docs
