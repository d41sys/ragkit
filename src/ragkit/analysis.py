"""Text analysis that approximates Lucene's ``EnglishAnalyzer`` (used by Pyserini/Anserini BM25).

The default ``ragkit.bm25.tokenize`` lowercases and keeps ``[a-z0-9]+`` runs. Anserini's "flat" and
"multifield" BM25 baselines (the reference rows in ``ragkit beir``) use Lucene's English analyzer
instead: UAX#29-style word tokens, possessive ``'s`` removal, a 33-word stop list and the Porter
stemmer. :func:`lucene_english` reproduces those steps in pure Python so that BM25 differences
caused by the analyzer can be separated from differences caused by the ranking function.

Departures from Lucene that remain (they are small, see ``docs/beir_protocol.md``): the token
regex approximates UAX#29 (no handling of scripts other than a letter/digit class), and Lucene's
lossy one-byte document-length norm is not applied unless ``SmallFloatLength`` is requested.
"""

from __future__ import annotations

import re
from collections.abc import Callable

Analyzer = Callable[[str], list[str]]

# Lucene's EnglishAnalyzer.ENGLISH_STOP_WORDS_SET (the former StopAnalyzer set).
LUCENE_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "if",
        "in",
        "into",
        "is",
        "it",
        "no",
        "not",
        "of",
        "on",
        "or",
        "such",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "will",
        "with",
    }
)

# Letters/digits, optionally joined by an apostrophe or a period (UAX#29 MidNumLet), e.g.
# "don't", "u.s", "3.5". Digit groups may also be joined by a comma ("1,000"). Underscore is a
# word character in UAX#29 (ExtendNumLet), which ``\w`` also covers.
_TOKEN_RE = re.compile(
    r"[0-9]+(?:[,.][0-9]+)+|\w+(?:['\u2019.]\w+)*",
)
_POSSESSIVE_RE = re.compile(r"['\u2019]s$")


def lucene_tokens(text: str) -> list[str]:
    """Lowercase UAX#29-like tokens with trailing possessive ``'s`` removed (no stop/stem)."""
    out: list[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        token = _POSSESSIVE_RE.sub("", token)
        if token:
            out.append(token)
    return out


def lucene_stop(text: str) -> list[str]:
    """``lucene_tokens`` followed by Lucene's English stop-word filter, without stemming."""
    return [t for t in lucene_tokens(text) if t not in LUCENE_STOPWORDS]


def lucene_english(text: str) -> list[str]:
    """Lucene ``EnglishAnalyzer``: tokens, possessive filter, stop words, Porter stemmer."""
    return [porter_stem(t) for t in lucene_tokens(text) if t not in LUCENE_STOPWORDS]


# ---------------------------------------------------------------------------------------------
# Porter stemmer (M. F. Porter, 1980) with the two departures of the reference implementation
# that Lucene's PorterStemmer also carries: "bli" -> "ble" (instead of "abli" -> "able") and
# "logi" -> "log". Written from the algorithm description.
# ---------------------------------------------------------------------------------------------

_STEP2 = (
    ("ational", "ate"),
    ("tional", "tion"),
    ("enci", "ence"),
    ("anci", "ance"),
    ("izer", "ize"),
    ("bli", "ble"),
    ("alli", "al"),
    ("entli", "ent"),
    ("eli", "e"),
    ("ousli", "ous"),
    ("ization", "ize"),
    ("ation", "ate"),
    ("ator", "ate"),
    ("alism", "al"),
    ("iveness", "ive"),
    ("fulness", "ful"),
    ("ousness", "ous"),
    ("aliti", "al"),
    ("iviti", "ive"),
    ("biliti", "ble"),
    ("logi", "log"),
)
_STEP3 = (
    ("icate", "ic"),
    ("ative", ""),
    ("alize", "al"),
    ("iciti", "ic"),
    ("ical", "ic"),
    ("ful", ""),
    ("ness", ""),
)
_STEP4 = (
    "al",
    "ance",
    "ence",
    "er",
    "ic",
    "able",
    "ible",
    "ant",
    "ement",
    "ment",
    "ent",
    "ion",
    "ou",
    "ism",
    "ate",
    "iti",
    "ous",
    "ive",
    "ize",
)


def _is_consonant(word: str, i: int) -> bool:
    ch = word[i]
    if ch in "aeiou":
        return False
    if ch == "y":
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def _measure(stem: str) -> int:
    """Porter's m: the number of vowel-consonant sequences in ``stem``."""
    count = 0
    prev_vowel = False
    for i in range(len(stem)):
        vowel = not _is_consonant(stem, i)
        if prev_vowel and not vowel:
            count += 1
        prev_vowel = vowel
    return count


def _has_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(word: str) -> bool:
    return len(word) >= 2 and word[-1] == word[-2] and _is_consonant(word, len(word) - 1)


def _ends_cvc(word: str) -> bool:
    """consonant-vowel-consonant at the end, the last consonant not w, x or y."""
    n = len(word)
    return (
        n >= 3
        and _is_consonant(word, n - 3)
        and not _is_consonant(word, n - 2)
        and _is_consonant(word, n - 1)
        and word[-1] not in "wxy"
    )


def _step1a(word: str) -> str:
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


def _step1b(word: str) -> str:
    if word.endswith("eed"):
        return word[:-1] if _measure(word[:-3]) > 0 else word
    for suffix in ("ed", "ing"):
        if word.endswith(suffix) and _has_vowel(word[: -len(suffix)]):
            word = word[: -len(suffix)]
            if word.endswith(("at", "bl", "iz")):
                return word + "e"
            if _ends_double_consonant(word) and word[-1] not in "lsz":
                return word[:-1]
            if _measure(word) == 1 and _ends_cvc(word):
                return word + "e"
            return word
    return word


def _step1c(word: str) -> str:
    if word.endswith("y") and _has_vowel(word[:-1]):
        return word[:-1] + "i"
    return word


def _replace_suffix(word: str, rules: tuple[tuple[str, str], ...], min_measure: int) -> str:
    for suffix, replacement in rules:
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            return stem + replacement if _measure(stem) > min_measure else word
    return word


def _step4(word: str) -> str:
    # Porter tries one suffix per word: the longest listed suffix that matches, no fallback.
    matches = [s for s in _STEP4 if word.endswith(s)]
    if not matches:
        return word
    suffix = max(matches, key=len)
    stem = word[: -len(suffix)]
    if suffix == "ion" and not stem.endswith(("s", "t")):
        return word
    return stem if _measure(stem) > 1 else word


def _step5(word: str) -> str:
    if word.endswith("e"):
        stem = word[:-1]
        m = _measure(stem)
        if m > 1 or (m == 1 and not _ends_cvc(stem)):
            word = stem
    if word.endswith("ll") and _measure(word[:-1]) > 1:
        word = word[:-1]
    return word


def porter_stem(word: str) -> str:
    """Stem one lowercase word. Words of one or two letters and non-ASCII words are kept."""
    if len(word) <= 2 or not word.isascii() or not word.isalpha():
        return word
    word = _step1a(word)
    word = _step1b(word)
    word = _step1c(word)
    word = _replace_suffix(word, _STEP2, 0)
    word = _replace_suffix(word, _STEP3, 0)
    word = _step4(word)
    return _step5(word)


# ---------------------------------------------------------------------------------------------
# Lucene's one-byte document-length norm (SmallFloat.intToByte4 / byte4ToInt, Lucene >= 8).
# ---------------------------------------------------------------------------------------------


def _long_to_int4(value: int) -> int:
    num_bits = value.bit_length()
    if num_bits < 4:
        return value
    shift = num_bits - 4
    return ((value >> shift) & 0x07) | ((shift + 1) << 3)


def _int4_to_long(code: int) -> int:
    bits = code & 0x07
    shift = (code >> 3) - 1
    return bits if shift == -1 else (bits | 0x08) << shift


_MAX_INT4 = _long_to_int4(2**31 - 1)
_NUM_FREE_VALUES = 255 - _MAX_INT4


def lucene_length(length: int) -> int:
    """Document length after Lucene's lossy one-byte norm encoding (exact below 24 tokens)."""
    if length < _NUM_FREE_VALUES:
        return length
    code = _NUM_FREE_VALUES + _long_to_int4(length - _NUM_FREE_VALUES)
    if code < _NUM_FREE_VALUES:
        return code
    return _NUM_FREE_VALUES + _int4_to_long(code - _NUM_FREE_VALUES)
