"""Query-side synonym and acronym expansion for EU electricity-market vocabulary.

The map below is frozen by ``docs/lexical_protocol.md`` before any EU question is scored. It was
written from the corpus vocabulary, not from the questions: the acronym census of the corpus
(``TSO``, ``FCR``, ``LFC``, ``BSP``, ...), the defined terms of the acts and well-known
plain-English renderings of legal terms. ``tests/test_ragkit_expand.py`` checks that every group
has at least one member that occurs in the bundled EU corpus, so each expansion has a target.
"""

from __future__ import annotations

from collections.abc import Sequence

from ragkit.bm25 import tokenize

# Each group lists phrases that mean the same thing in this domain. Matching is on token
# sequences (lowercase alphanumerics), so "Cross-zonal" and "cross zonal" are the same phrase.
# Members need not occur in the corpus (users write "mFRR", the acts say "manual frequency
# restoration process"), but every group has at least one member that does.
SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    # Operators and market roles
    ("TSO", "TSOs", "transmission system operator", "transmission system operators"),
    ("DSO", "DSOs", "distribution system operator", "distribution system operators"),
    (
        "NEMO",
        "NEMOs",
        "nominated electricity market operator",
        "nominated electricity market operators",
    ),
    ("BSP", "BSPs", "balancing service provider", "balancing service providers"),
    ("BRP", "BRPs", "balance responsible party", "balance responsible parties"),
    ("SGU", "SGUs", "significant grid user", "significant grid users"),
    (
        "RCC",
        "RCCs",
        "regional coordination centre",
        "regional coordination centres",
        "regional coordination center",
    ),
    ("CCC", "coordinated capacity calculator", "coordinated capacity calculators"),
    (
        "regulator",
        "regulators",
        "regulatory authority",
        "regulatory authorities",
        "national regulatory authority",
    ),
    ("customer", "customers", "consumer", "consumers"),
    # Balancing products and processes
    ("FCR", "frequency containment reserves", "frequency containment reserve"),
    ("FRR", "frequency restoration reserves", "frequency restoration reserve", "secondary reserve"),
    ("RR", "replacement reserves", "replacement reserve", "tertiary reserve"),
    ("aFRR", "aFRP", "automatic frequency restoration process"),
    ("mFRR", "mFRP", "manual frequency restoration process"),
    ("LFC", "load frequency control"),
    # Congestion management and capacity calculation
    ("redispatch", "redispatching"),
    ("redispatching", "countertrading", "remedial actions"),
    ("cross zonal", "cross border"),
    ("NTC", "net transmission capacity"),
    ("LTTR", "LTTRs", "long term transmission rights"),
    ("SDAC", "single day ahead coupling"),
    ("SIDC", "single intraday coupling"),
    ("CACM", "capacity allocation and congestion management"),
    ("VoLL", "value of lost load"),
    ("UIOSI", "use it or sell it"),
    # Plain-English renderings of legal terms
    ("grid", "network"),
    ("demand response", "demand side response", "demand side flexibility"),
    ("derogation", "derogations", "exemption", "exemptions"),
    ("network charges", "network tariffs"),
    ("battery", "batteries", "storage", "energy storage"),
    ("outage", "outages", "blackout"),
    ("generator", "generating facility", "power generating module"),
)

DEFAULT_EXPANSION_WEIGHT = 0.4


def _contains(tokens: Sequence[str], phrase: Sequence[str]) -> bool:
    """True when ``phrase`` occurs as a contiguous run in ``tokens``."""
    n = len(phrase)
    if n == 0:
        return False
    return any(tokens[i : i + n] == list(phrase) for i in range(len(tokens) - n + 1))


class SynonymExpander:
    """Adds down-weighted synonym tokens to a query.

    Original query tokens keep weight 1.0 (with multiplicity, as in ``BM25Index``). When a
    group member occurs in the query, the tokens of every other member are added once each at
    ``weight``, unless the query already contains them.
    """

    def __init__(
        self,
        groups: Sequence[Sequence[str]] = SYNONYM_GROUPS,
        weight: float = DEFAULT_EXPANSION_WEIGHT,
    ) -> None:
        if weight < 0.0:
            raise ValueError("weight must be non-negative")
        self.weight = weight
        self.groups: list[list[tuple[str, ...]]] = [
            [tuple(tokenize(member)) for member in group] for group in groups
        ]

    def expand(self, query: str) -> list[tuple[str, float]]:
        """Weighted query terms: the original tokens, then the synonym tokens."""
        tokens = tokenize(query)
        terms: list[tuple[str, float]] = [(t, 1.0) for t in tokens]
        present = set(tokens)
        added: set[str] = set()
        for group in self.groups:
            if not any(_contains(tokens, member) for member in group):
                continue
            for member in group:
                if _contains(tokens, member):
                    continue
                for tok in member:
                    if tok not in present and tok not in added:
                        added.add(tok)
                        terms.append((tok, self.weight))
        return terms
