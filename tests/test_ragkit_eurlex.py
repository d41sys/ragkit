"""Offline tests for the EUR-Lex article extractor; fixtures mimic the Cellar XHTML structure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragkit.eurlex import (
    Act,
    Article,
    article_doc_id,
    build_corpus,
    parse_articles,
    render_article,
    strip_metadata_comment,
)

XHTML = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><body>
<div class="eli-subdivision" id="rct_1">
 <p class="oj-normal">(1) A recital, not an article.</p>
</div>
<div id="tis_I">
 <div class="eli-subdivision" id="art_1">
  <p class="oj-ti-art">Article 1</p>
  <div class="eli-title" id="art_1.tit_1"><p class="oj-sti-art">Subject matter</p></div>
  <div id="001.001"><p class="oj-normal">1.   This   Regulation lays down rules
   <a href="#n1">(<span class="oj-super oj-note-tag">1</span>)</a>.</p></div>
  <div id="001.002"><p class="oj-normal">2.   Second paragraph.</p></div>
 </div>
 <div class="eli-subdivision" id="art_2">
  <p class="oj-ti-art">Article 2</p>
  <div class="eli-title" id="art_2.tit_1"><p class="oj-sti-art">Definitions</p></div>
  <p class="oj-normal">The following definitions apply:</p>
  <table><tbody><tr>
    <td><p class="oj-normal">(1)</p></td>
    <td><p class="oj-normal">'congestion' means a situation where requests fail;</p></td>
  </tr></tbody></table>
  <table><tbody><tr>
    <td><p class="oj-normal">(2)</p></td>
    <td>
      <p class="oj-normal">'remedial action' means:</p>
      <table><tbody>
        <tr><td><p class="oj-normal">(a)</p></td>
            <td><p class="oj-normal">redispatching;</p></td></tr>
        <tr><td><p class="oj-normal">(b)</p></td>
            <td><p class="oj-normal">countertrading.</p></td></tr>
      </tbody></table>
    </td>
  </tr></tbody></table>
  <p class="oj-note">Footnote text that must be dropped.</p>
 </div>
 <div class="eli-subdivision" id="art_3a">
  <p class="oj-ti-art">Article 3a</p>
  <div class="eli-title" id="art_3a.tit_1"><p class="oj-sti-art">Inserted article</p></div>
  <p class="oj-normal">Inserted text.</p>
 </div>
</div>
</body></html>
"""

ACT = Act("reg_2099_1", "32099R0001", "Regulation (EU) 2099/1 (test)")


def test_parse_articles_extracts_numbers_titles_and_text() -> None:
    arts = parse_articles(XHTML)
    assert [a.number for a in arts] == ["1", "2", "3a"]
    assert [a.title for a in arts] == ["Subject matter", "Definitions", "Inserted article"]
    # Whitespace collapsed, footnote marker and its empty brackets removed, recital ignored.
    assert arts[0].lines == ("1. This Regulation lays down rules.", "2. Second paragraph.")


def test_parse_articles_flattens_tables_and_nested_points() -> None:
    lines = parse_articles(XHTML)[1].lines
    assert lines[0] == "The following definitions apply:"
    assert lines[1] == "(1) 'congestion' means a situation where requests fail;"
    assert lines[2] == "(2) 'remedial action' means:\n  (a) redispatching;\n  (b) countertrading."
    assert all("Footnote" not in line for line in lines)


def test_render_article_has_comment_heading_and_body() -> None:
    article = Article("16", "Congestion management", ("1. First.", "2. Second."))
    text = render_article(ACT, article)
    assert text.startswith("<!-- source: https://publications.europa.eu/resource/celex/32099R0001")
    assert "id: reg_2099_1_art16" in text.splitlines()[0]
    body = strip_metadata_comment(text)
    assert body == (
        "# Regulation (EU) 2099/1 (test), Article 16: Congestion management\n\n"
        "1. First.\n\n2. Second.\n"
    )
    assert article_doc_id(ACT, article) == "reg_2099_1_art16"


def test_strip_metadata_comment_only_touches_a_leading_comment() -> None:
    assert strip_metadata_comment("<!-- a -->\n\n# T") == "# T"
    assert strip_metadata_comment("# T\n<!-- keep -->") == "# T\n<!-- keep -->"


def test_build_corpus_writes_selected_articles_and_manifest(tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_fetch(celex: str) -> bytes:
        calls.append(celex)
        return XHTML.encode("utf-8")

    act = Act("reg_2099_1", "32099R0001", "Regulation (EU) 2099/1 (test)", frozenset({"1", "3a"}))
    out, cache = tmp_path / "out", tmp_path / "cache"
    ids = build_corpus([act], out, cache_dir=cache, fetch=fake_fetch, retrieved_on="2026-10-16")
    assert ids == ["reg_2099_1_art1", "reg_2099_1_art3a"]
    assert sorted(p.name for p in out.glob("*.md")) == ["reg_2099_1_art1.md", "reg_2099_1_art3a.md"]

    manifest = json.loads((out / "SOURCES.json").read_text(encoding="utf-8"))
    assert manifest["acts"][0]["retrieved_on"] == "2026-10-16"
    assert manifest["acts"][0]["articles_kept"] == ["1", "3a"]
    assert manifest["acts"][0]["articles_in_act"] == 3
    assert "2011/833/EU" in manifest["licence"]

    # Second run reuses the cached download instead of hitting the network again.
    build_corpus([act], out, cache_dir=cache, fetch=fake_fetch, retrieved_on="2026-10-16")
    assert calls == ["32099R0001"]


def test_build_corpus_rejects_articles_missing_from_the_source(tmp_path: Path) -> None:
    act = Act("reg_2099_1", "32099R0001", "t", frozenset({"1", "99"}))
    with pytest.raises(ValueError, match="99"):
        build_corpus(
            [act],
            tmp_path / "out",
            cache_dir=tmp_path / "cache",
            fetch=lambda _: XHTML.encode("utf-8"),
            retrieved_on="2026-10-16",
        )
