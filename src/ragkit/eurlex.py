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
from dataclasses import asdict, dataclass
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
class Amendment:
    """An act that amends one of the corpus acts (EUR-Lex "Modified by" relation)."""

    celex: str
    title: str
    url: str


@dataclass(frozen=True)
class Act:
    """One EU legal act and the articles of it that enter the corpus."""

    short: str  # file-name prefix and doc-id prefix, e.g. "reg_2019_943"
    celex: str
    title: str
    articles: frozenset[str] | None = None  # None keeps every article
    consolidated_celex: str = ""  # latest consolidated text, e.g. "02019R0943-20240716"
    amended_by: tuple[Amendment, ...] = ()


def _arts(spec: str) -> frozenset[str]:
    return frozenset(spec.split())


# Articles chosen for relevance to congestion management and balancing, plus neighbours that
# share vocabulary (capacity mechanisms, definitions, cost recovery) so retrieval has real
# hard negatives. Amendment history and consolidated versions: EUR-Lex document information
# pages https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:<CELEX>, read 2026-11-13
# ("Modified by" relations; "Corrected by" corrigenda and "Derogated in" decisions are not listed).
_EUR_LEX_ELI = "https://eur-lex.europa.eu/eli/{}/oj/eng"
REG_2024_1747 = Amendment(
    "32024R1747",
    "Regulation (EU) 2024/1747 amending Regulations (EU) 2019/942 and (EU) 2019/943 as regards "
    "improving the Union's electricity market design",
    _EUR_LEX_ELI.format("reg/2024/1747"),
)
DIR_2024_1711 = Amendment(
    "32024L1711",
    "Directive (EU) 2024/1711 amending Directives (EU) 2018/2001 and (EU) 2019/944 as regards "
    "improving the Union's electricity market design",
    _EUR_LEX_ELI.format("dir/2024/1711"),
)
REG_2022_869 = Amendment(
    "32022R0869",
    "Regulation (EU) 2022/869 on guidelines for trans-European energy infrastructure, amending "
    "Regulations (EC) No 715/2009, (EU) 2019/942 and (EU) 2019/943 and Directives 2009/73/EC and "
    "(EU) 2019/944, and repealing Regulation (EU) No 347/2013",
    _EUR_LEX_ELI.format("reg/2022/869"),
)
IMPL_2021_280 = Amendment(
    "32021R0280",
    "Commission Implementing Regulation (EU) 2021/280 amending Regulations (EU) 2015/1222, "
    "(EU) 2016/1719, (EU) 2017/2195 and (EU) 2017/1485 in order to align them with Regulation "
    "(EU) 2019/943",
    _EUR_LEX_ELI.format("reg_impl/2021/280"),
)
IMPL_2022_828 = Amendment(
    "32022R0828",
    "Commission Implementing Regulation (EU) 2022/828 correcting the Polish language version of "
    "Regulation (EU) 2017/2195 establishing a guideline on electricity balancing",
    _EUR_LEX_ELI.format("reg_impl/2022/828"),
)
EU_ACTS: tuple[Act, ...] = (
    Act(
        "reg_2019_943",
        "32019R0943",
        "Regulation (EU) 2019/943 (electricity market regulation)",
        _arts("2 3 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 21 22 34 35 37 42 49"),
        "02019R0943-20240716",
        (REG_2022_869, REG_2024_1747),
    ),
    Act(
        "reg_2017_2195",
        "32017R2195",
        "Commission Regulation (EU) 2017/2195 (electricity balancing guideline)",
        _arts("2 3 4 16 17 18 19 20 21 22 23 24 25 29 30 32 33 34 35 38 39 40 41 42 52 53 54 55"),
        "02017R2195-20220619",
        (IMPL_2021_280, IMPL_2022_828),
    ),
    Act(
        "reg_2015_1222",
        "32015R1222",
        "Commission Regulation (EU) 2015/1222 (capacity allocation and congestion management)",
        _arts("14 15 20 21 22 25 26 29 32 33 35 70 72 73 74"),
        "02015R1222-20210315",
        (IMPL_2021_280,),
    ),
    Act(
        "dir_2019_944",
        "32019L0944",
        "Directive (EU) 2019/944 (internal market for electricity)",
        _arts("15 16 17 31 32 33 36 40 54"),
        # 2025-10-12 version: Directive (EU) 2023/1791 partially repeals Article 70 (not kept).
        "02019L0944-20251012",
        (REG_2022_869, DIR_2024_1711),
    ),
    Act(
        "reg_2017_1485",
        "32017R1485",
        "Commission Regulation (EU) 2017/1485 (system operation guideline)",
        _arts("20 21 22 23 143 145 146 147 153 156 157 158 166 176"),
        "02017R1485-20210315",
        (IMPL_2021_280,),
    ),
    Act(
        "reg_2016_1719",
        "32016R1719",
        "Commission Regulation (EU) 2016/1719 (forward capacity allocation)",
        _arts("25 53 56 57"),
        "02016R1719-20210315",
        (IMPL_2021_280,),
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


# -- consolidated texts ---------------------------------------------------------------------
# EUR-Lex consolidated versions use a different layout from the Official Journal XHTML: numbered
# paragraphs are `<div class="norm"><span class="no-parag">1.</span><div class="norm
# inline-element">text</div></div>`, points are nested `grid-list` divs, and amendment markers
# (▼M1, ►M1 ... ◄) sit in `modref` paragraphs. Paragraph text can be mixed with nested blocks,
# so this parser builds a small tree and flattens it to the same line shapes as `parse_articles`.
_AMEND_MARK = re.compile(r"[▼►][A-Z]\d*|◄")
_SKIPPED_CLASSES = frozenset({"modref", "arrow", "footnote", "title-article-norm", "eli-title"})
_VOID_TAGS = frozenset({"br", "hr", "col", "meta", "link", "img"})
_BLOCK_TAGS = frozenset({"p", "div", "table"})


class _Node:
    def __init__(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: v or "" for k, v in attrs}
        self.tag = tag
        self.id = a.get("id", "")
        self.classes = frozenset(a.get("class", "").split())
        self.kids: list[_Node | str] = []


class _TreeBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node("root", [])
        self._stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = _Node(tag, attrs)
        self._stack[-1].kids.append(node)
        if tag not in _VOID_TAGS:
            self._stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._stack[-1].kids.append(_Node(tag, attrs))

    def handle_endtag(self, tag: str) -> None:
        for i in range(len(self._stack) - 1, 0, -1):
            if self._stack[i].tag == tag:
                del self._stack[i:]
                return

    def handle_data(self, data: str) -> None:
        self._stack[-1].kids.append(data)


def _inline_text(node: _Node | str) -> str:
    """Text of an inline subtree; footnote reference numbers (superscripts) and markers dropped."""
    if isinstance(node, str):
        return _AMEND_MARK.sub("", node)
    if "superscript" in node.classes:
        return ""
    return "".join(_inline_text(k) for k in node.kids)


def _find(node: _Node, cls: str) -> _Node | None:
    for kid in node.kids:
        if isinstance(kid, _Node):
            if cls in kid.classes:
                return kid
            if found := _find(kid, cls):
                return found
    return None


def _block_lines(node: _Node) -> list[str]:
    """Flatten the children of a block element into lines; inline runs become one line."""
    lines: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if text := _tidy("".join(run)):
            lines.append(text)
        run.clear()

    for kid in node.kids:
        if isinstance(kid, _Node) and kid.tag in _BLOCK_TAGS:
            flush()
            lines.extend(_block(kid))
        else:
            run.append(_inline_text(kid))
    flush()
    return lines


def _block(el: _Node) -> list[str]:
    if el.classes & _SKIPPED_CLASSES:
        return []
    if el.tag == "p":
        return [text] if (text := _tidy(_inline_text(el))) else []
    if el.tag == "table":
        return _table_lines(el)
    kids = [k for k in el.kids if isinstance(k, _Node)]
    if "grid-container" in el.classes:  # a point: "(a)" label beside its (possibly nested) text
        label = next(
            (_tidy(_inline_text(k)) for k in kids if "grid-list-column-1" in k.classes), ""
        )
        body = next((_block_lines(k) for k in kids if "grid-list-column-2" in k.classes), [])
        first, *rest = "\n".join([label, *body]).replace("\n", " ", 1).split("\n")
        return ["".join([first, *("\n  " + ln for ln in rest if ln.strip())])]
    number = next((k for k in kids if "no-parag" in k.classes), None)
    if number is not None:  # numbered paragraph: "1." joined to the first line of its body
        body = _block_lines(_without(el, number))
        return [f"{_tidy(_inline_text(number))} {body[0]}", *body[1:]] if body else []
    return _block_lines(el)


def _without(el: _Node, drop: _Node) -> _Node:
    rest = _Node(el.tag, [])
    rest.kids = [k for k in el.kids if k is not drop]
    return rest


def _table_lines(table: _Node) -> list[str]:
    lines: list[str] = []
    sections = [k for k in table.kids if isinstance(k, _Node)]  # <tr>, or <tbody> holding them
    for tr in [
        r for k in sections for r in ([k] if k.tag == "tr" else k.kids) if isinstance(r, _Node)
    ]:
        cells = ["\n".join(_block_lines(td)) for td in tr.kids if isinstance(td, _Node)]
        if text := " ".join(cells).strip():
            first, *rest = text.split("\n")
            lines.append("".join([first, *("\n  " + ln for ln in rest if ln.strip())]))
    return lines


def parse_consolidated_articles(xhtml: str) -> list[Article]:
    """Extract every article from an EUR-Lex consolidated-text XHTML document, in order."""
    builder = _TreeBuilder()
    builder.feed(xhtml)
    builder.close()
    articles: list[Article] = []

    def walk(node: _Node) -> None:
        for kid in node.kids:
            if not isinstance(kid, _Node):
                continue
            if kid.tag == "div" and (m := _ARTICLE_DIV.fullmatch(kid.id)):
                title = _find(kid, "stitle-article-norm")
                articles.append(
                    Article(
                        m.group(1),
                        _tidy(_inline_text(title)) if title else "",
                        tuple(_block_lines(kid)),
                    )
                )
            else:
                walk(kid)

    walk(builder.root)
    return articles


def article_doc_id(act: Act, article: Article) -> str:
    """Stable document id such as ``reg_2019_943_art16``."""
    return f"{act.short}_art{article.number}"


def source_url(act: Act, *, consolidated: bool = False) -> str:
    return CELLAR_URL.format(celex=act.consolidated_celex if consolidated else act.celex)


def render_article(act: Act, article: Article, *, consolidated: bool = False) -> str:
    """Markdown file body: a metadata comment (stripped at load time), a heading, the text."""
    meta = (
        f"<!-- source: {source_url(act, consolidated=consolidated)} | act: {act.title} | "
        f"article: {article.number} | id: {article_doc_id(act, article)} -->"
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


def _consolidated_date(celex: str) -> str:
    """``02019R0943-20240716`` -> ``2024-07-16``."""
    d = celex.rsplit("-", 1)[1]
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


def build_corpus(
    acts: Iterable[Act],
    out_dir: Path,
    *,
    cache_dir: Path,
    fetch: Fetcher = fetch_cellar,
    retrieved_on: str,
    consolidated: bool = False,
) -> list[str]:
    """Fetch (or reuse cached) acts, write one markdown file per selected article.

    Returns the written doc ids. Also writes ``SOURCES.json`` with URL, retrieval date, sha256 of
    the downloaded XHTML, selected articles, amendment history and the licence statement. With
    ``consolidated`` the latest consolidated text of each act is used instead of the original.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    doc_ids: list[str] = []
    sources: list[dict[str, object]] = []
    for act in acts:
        celex = act.consolidated_celex if consolidated else act.celex
        cached = cache_dir / f"{celex}.xhtml"
        if not cached.exists():
            cached.write_bytes(fetch(celex))
        raw = cached.read_bytes()
        parse = parse_consolidated_articles if consolidated else parse_articles
        articles = parse(raw.decode("utf-8"))
        kept = [a for a in articles if act.articles is None or a.number in act.articles]
        missing = (act.articles or frozenset()) - {a.number for a in kept}
        if missing:
            raise ValueError(f"{act.short}: articles not found in source: {sorted(missing)}")
        for art in kept:
            doc_id = article_doc_id(act, art)
            text = render_article(act, art, consolidated=consolidated)
            (out_dir / f"{doc_id}.md").write_text(text, encoding="utf-8")
            doc_ids.append(doc_id)
        version: dict[str, object] = (
            {
                "consolidated_celex": celex,
                "consolidated_date": _consolidated_date(celex),
            }
            if consolidated
            else {}
        )
        sources.append(
            {
                "short": act.short,
                "celex": act.celex,
                "title": act.title,
                "url": source_url(act, consolidated=consolidated),
                **version,
                "retrieved_on": retrieved_on,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "articles_in_act": len(articles),
                "articles_kept": [a.number for a in kept],
                "amended_by": [asdict(a) for a in act.amended_by],
            }
        )
    manifest = {
        "licence": LICENCE_STATEMENT,
        "licence_source": LEGAL_NOTICE_URL,
        "reuse_decision": REUSE_DECISION_URL,
        "note": (
            "Latest EUR-Lex consolidated versions (a documentation tool without legal effect; the "
            "authentic texts are those published in the Official Journal)."
            if consolidated
            else "Original Official Journal versions (not consolidated); later amendments are not "
            "reflected, see amended_by and corpus_eu_consolidated."
        ),
        "conditions": "Source acknowledged in each file's metadata comment and in the README; "
        "the legal meaning is not altered (Decision 2011/833/EU, Art. 6(2)).",
        "modifications": "Split into one file per article; footnote markers and footnote text "
        "removed; "
        + ("amendment markers (▼ ► ◄) removed; " if consolidated else "")
        + "point tables flattened to indented lines; whitespace normalised.",
        "acts": sources,
    }
    (out_dir / "SOURCES.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return doc_ids
