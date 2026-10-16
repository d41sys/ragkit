"""Build an article-level markdown corpus from EU legal acts published by the Publications Office.

The XHTML is fetched from the Cellar endpoint (``publications.europa.eu/resource/celex/<CELEX>``)
because ``eur-lex.europa.eu`` sits behind a bot challenge that answers HTTP 202 with an empty body.
Only the standard library is used: parsing is a small :class:`html.parser.HTMLParser` state machine.

Reuse basis: the EUR-Lex legal notice states that, unless otherwise specified, the legal documents
published in EUR-Lex may be re-used for commercial and non-commercial purposes under Commission
Decision 2011/833/EU (source acknowledged, meaning not distorted). See ``corpus_eu/SOURCES.json``.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

CELLAR_URL = "https://publications.europa.eu/resource/celex/{celex}"
USER_AGENT = "ragkit-research (+https://github.com/d41sys/ragkit)"

LEGAL_NOTICE_URL = "https://eur-lex.europa.eu/content/legal-notice/legal-notice.html"
REUSE_DECISION_URL = "https://eur-lex.europa.eu/eli/dec/2011/833/oj/eng"
LICENCE_STATEMENT = (
    "Unless otherwise specified, you can re-use the legal documents published in EUR-Lex for "
    "commercial or non-commercial purposes. The Commission's document reuse policy is based on "
    "Decision 2011/833/EU."
)

_WHITESPACE = re.compile(r"\s+")
_METADATA_COMMENT = re.compile(r"\A\s*<!--.*?-->\s*", re.DOTALL)
# What is left of a footnote tag "(1)" once its number is dropped, with the space before it.
_EMPTY_NOTE_MARK = re.compile(r"\s*\(\s*\)")
_ARTICLE_DIV = re.compile(r"art_(\d+[a-z]?)")


@dataclass(frozen=True)
class Act:
    """One EU legal act and the articles of it that enter the corpus."""

    short: str  # file-name prefix and doc-id prefix, e.g. "reg_2019_943"
    celex: str
    title: str
    articles: frozenset[str] | None = None  # None keeps every article


def _arts(spec: str) -> frozenset[str]:
    return frozenset(spec.split())


# Articles chosen for relevance to congestion management and balancing, plus neighbours that
# share vocabulary (capacity mechanisms, definitions, cost recovery) so retrieval has real
# hard negatives. Original Official Journal texts, not consolidated versions.
EU_ACTS: tuple[Act, ...] = (
    Act(
        "reg_2019_943",
        "32019R0943",
        "Regulation (EU) 2019/943 (electricity market regulation)",
        _arts("2 3 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 21 22 34 35 37 42 49"),
    ),
    Act(
        "reg_2017_2195",
        "32017R2195",
        "Commission Regulation (EU) 2017/2195 (electricity balancing guideline)",
        _arts("2 3 4 16 17 18 19 20 21 22 23 24 25 29 30 32 33 34 35 38 39 40 41 42 52 53 54 55"),
    ),
    Act(
        "reg_2015_1222",
        "32015R1222",
        "Commission Regulation (EU) 2015/1222 (capacity allocation and congestion management)",
        _arts("14 15 20 21 22 25 26 29 32 33 35 70 72 73 74"),
    ),
    Act(
        "dir_2019_944",
        "32019L0944",
        "Directive (EU) 2019/944 (internal market for electricity)",
        _arts("15 16 17 31 32 33 36 40 54"),
    ),
    Act(
        "reg_2017_1485",
        "32017R1485",
        "Commission Regulation (EU) 2017/1485 (system operation guideline)",
        _arts("20 21 22 23 143 145 146 147 153 156 157 158 166 176"),
    ),
    Act(
        "reg_2016_1719",
        "32016R1719",
        "Commission Regulation (EU) 2016/1719 (forward capacity allocation)",
        _arts("25 53 56 57"),
    ),
)


@dataclass(frozen=True)
class Article:
    """One article: number as printed ("16"), heading, and paragraph lines."""

    number: str
    title: str
    lines: tuple[str, ...]


def _tidy(text: str) -> str:
    return _WHITESPACE.sub(" ", _EMPTY_NOTE_MARK.sub("", text)).strip()


class _ArticleParser(HTMLParser):
    """Collect `<div id="art_N">` blocks, flattening paragraphs, point tables and sub-points."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.articles: list[Article] = []
        self._div_stack: list[str | None] = []
        self._art_depth: int | None = None
        self._number = ""
        self._title_parts: list[str] = []
        # Containers collect finished lines: the article body, then one per open table cell.
        self._containers: list[list[str]] = []
        self._rows: list[list[str]] = []
        self._para: list[str] | None = None
        self._p_class = ""
        self._skip_depth = 0
        self._span_stack: list[bool] = []

    # -- helpers ---------------------------------------------------------------------------
    @property
    def _in_article(self) -> bool:
        return self._art_depth is not None

    def _emit(self, line: str) -> None:
        if line:
            self._containers[-1].append(line)

    # -- HTMLParser hooks ------------------------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: v or "" for k, v in attrs}
        if tag == "div":
            div_id = a.get("id")
            self._div_stack.append(div_id)
            m = _ARTICLE_DIV.fullmatch(div_id or "")
            if m and not self._in_article:
                self._art_depth = len(self._div_stack)
                self._number = m.group(1)
                self._title_parts = []
                self._containers = [[]]
                self._rows = []
            return
        if not self._in_article:
            return
        if tag == "p":
            self._para = []
            self._p_class = a.get("class", "")
        elif tag == "span":
            hidden = "oj-super" in a.get("class", "")
            self._span_stack.append(hidden)
            if hidden:
                self._skip_depth += 1
        elif tag == "table":
            self._rows.append([])
        elif tag == "td":
            self._containers.append([])

    def handle_endtag(self, tag: str) -> None:
        if tag == "div":
            if self._art_depth is not None and len(self._div_stack) == self._art_depth:
                self.articles.append(
                    Article(
                        self._number,
                        _tidy(" ".join(self._title_parts)),
                        tuple(self._containers[0]) if self._containers else (),
                    )
                )
                self._art_depth = None
            if self._div_stack:
                self._div_stack.pop()
            return
        if not self._in_article:
            return
        if tag == "span":
            if self._span_stack and self._span_stack.pop():
                self._skip_depth -= 1
        elif tag == "p":
            self._finish_paragraph()
        elif tag == "td":
            cell = self._containers.pop()
            if self._rows:
                self._rows[-1].append("\n".join(cell))
        elif tag == "tr":
            row = self._rows[-1] if self._rows else []
            if row:
                first, *rest = " ".join(row).split("\n")
                self._emit("".join([first, *("\n  " + ln for ln in rest if ln.strip())]))
            if self._rows:
                self._rows[-1] = []
        elif tag == "table" and self._rows:
            self._rows.pop()

    def handle_data(self, data: str) -> None:
        if self._para is not None and not self._skip_depth:
            self._para.append(data)

    # -- internals -------------------------------------------------------------------------
    def _finish_paragraph(self) -> None:
        text = _tidy("".join(self._para or []))
        self._para = None
        cls = self._p_class
        if not text or "oj-note" in cls or "oj-ti-art" in cls:
            return
        if "oj-sti-art" in cls:
            self._title_parts.append(text)
            return
        self._emit(text)


def parse_articles(xhtml: str) -> list[Article]:
    """Extract every article from an Official Journal XHTML document, in document order."""
    parser = _ArticleParser()
    parser.feed(xhtml)
    parser.close()
    return parser.articles


def article_doc_id(act: Act, article: Article) -> str:
    """Stable document id such as ``reg_2019_943_art16``."""
    return f"{act.short}_art{article.number}"


def source_url(act: Act) -> str:
    return CELLAR_URL.format(celex=act.celex)


def render_article(act: Act, article: Article) -> str:
    """Markdown file body: a metadata comment (stripped at load time), a heading, the text."""
    meta = (
        f"<!-- source: {source_url(act)} | act: {act.title} | article: {article.number} | "
        f"id: {article_doc_id(act, article)} -->"
    )
    heading = f"# {act.title}, Article {article.number}: {article.title}"
    return "\n\n".join([meta, heading, *article.lines]) + "\n"


def strip_metadata_comment(text: str) -> str:
    """Drop the leading ``<!-- ... -->`` provenance line so URLs do not enter the index."""
    return _METADATA_COMMENT.sub("", text, count=1)


Fetcher = Callable[[str], bytes]


def fetch_cellar(celex: str) -> bytes:
    """Download the English XHTML of an act from the Publications Office Cellar endpoint."""
    req = urllib.request.Request(
        CELLAR_URL.format(celex=celex),
        headers={
            "Accept": "application/xhtml+xml",
            "Accept-Language": "eng",
            "User-Agent": USER_AGENT,
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data: bytes = resp.read()
    return data


def build_corpus(
    acts: Iterable[Act],
    out_dir: Path,
    *,
    cache_dir: Path,
    fetch: Fetcher = fetch_cellar,
    retrieved_on: str,
) -> list[str]:
    """Fetch (or reuse cached) acts, write one markdown file per selected article.

    Returns the written doc ids. Also writes ``SOURCES.json`` with URL, retrieval date, sha256 of
    the downloaded XHTML, selected articles and the licence statement.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    doc_ids: list[str] = []
    sources: list[dict[str, object]] = []
    for act in acts:
        cached = cache_dir / f"{act.celex}.xhtml"
        if not cached.exists():
            cached.write_bytes(fetch(act.celex))
        raw = cached.read_bytes()
        articles = parse_articles(raw.decode("utf-8"))
        kept = [a for a in articles if act.articles is None or a.number in act.articles]
        missing = (act.articles or frozenset()) - {a.number for a in kept}
        if missing:
            raise ValueError(f"{act.short}: articles not found in source: {sorted(missing)}")
        for art in kept:
            doc_id = article_doc_id(act, art)
            (out_dir / f"{doc_id}.md").write_text(render_article(act, art), encoding="utf-8")
            doc_ids.append(doc_id)
        sources.append(
            {
                "short": act.short,
                "celex": act.celex,
                "title": act.title,
                "url": source_url(act),
                "retrieved_on": retrieved_on,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "articles_in_act": len(articles),
                "articles_kept": [a.number for a in kept],
            }
        )
    manifest = {
        "licence": LICENCE_STATEMENT,
        "licence_source": LEGAL_NOTICE_URL,
        "reuse_decision": REUSE_DECISION_URL,
        "note": "Original Official Journal versions (not consolidated); later amendments "
        "are not reflected.",
        "conditions": "Source acknowledged in each file's metadata comment and in the README; "
        "the legal meaning is not altered (Decision 2011/833/EU, Art. 6(2)).",
        "modifications": "Split into one file per article; footnote markers and footnote text "
        "removed; point tables flattened to indented lines; whitespace normalised.",
        "acts": sources,
    }
    (out_dir / "SOURCES.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return doc_ids
