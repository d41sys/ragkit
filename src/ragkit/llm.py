"""Messages API client with a JSONL call cache (stdlib only; ``docs/generation_protocol.md``).

Every request is stored once in a JSONL file under the SHA-256 of its payload, so a re-run replays
offline: scoring and the tests need neither the key nor the network. ``max_calls`` and
``max_cost_usd`` bound what a cache file may ever hold. The corpus block of a full-context request
is stored as its hash and length, not as text. List prices below are assumptions, not read from
the API.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragkit.hardset_review import post_message

MODEL = "claude-sonnet-5-5"
MAX_CALLS = 1832  # cached records: K-R2 and Phase L (1432) plus at most 400 for the k = 5 analysis
MAX_COST_USD = 18.10  # cumulative at the assumed prices: 14.25 after Phase L plus 3.85 (cap 4)
# Assumed list prices in USD per million tokens (not read from the API).
INPUT_PER_MTOK = 3.0
OUTPUT_PER_MTOK = 15.0
CACHE_WRITE_FACTOR = 1.25
CACHE_READ_FACTOR = 0.1
_LONG_BLOCK = 2000  # system blocks longer than this are stored as hash + length

Transport = Callable[[dict[str, Any], str], dict[str, Any]]


class CacheMiss(RuntimeError):
    """An offline run asked for a request that is not in the cache."""


@dataclass(frozen=True)
class Reply:
    """A model reply: its text, the API usage block and whether it came from the cache."""

    text: str
    usage: Mapping[str, int]
    cached: bool


def request_key(payload: Mapping[str, Any]) -> str:
    """Stable SHA-256 of a request payload."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def usage_cost(usage: Mapping[str, int]) -> float:
    """Cost in USD of one usage block at the assumed list prices."""
    return (
        usage.get("input_tokens", 0) * INPUT_PER_MTOK
        + usage.get("cache_creation_input_tokens", 0) * INPUT_PER_MTOK * CACHE_WRITE_FACTOR
        + usage.get("cache_read_input_tokens", 0) * INPUT_PER_MTOK * CACHE_READ_FACTOR
        + usage.get("output_tokens", 0) * OUTPUT_PER_MTOK
    ) / 1_000_000


def _summarise_system(system: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for block in system:
        text = str(block.get("text", ""))
        if len(text) > _LONG_BLOCK:
            entry: dict[str, Any] = {
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "chars": len(text),
            }
        else:
            entry = {"text": text}
        if "cache_control" in block:
            entry["cache_control"] = block["cache_control"]
        out.append(entry)
    return out


class CachedMessages:
    """Messages API calls memoised in a JSONL file; thread-safe."""

    def __init__(
        self,
        path: Path,
        *,
        api_key: str = "",
        model: str = MODEL,
        temperature: float | None = None,
        max_calls: int = MAX_CALLS,
        max_cost_usd: float = MAX_COST_USD,
        transport: Transport | None = None,
    ) -> None:
        self.path = path
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_calls = max_calls
        self.max_cost_usd = max_cost_usd
        self.transport: Transport = transport or post_message
        self.live_calls = 0
        self.touched: set[str] = set()  # keys of every request made through this client
        self._lock = threading.Lock()
        self._key_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
        self.records: dict[str, dict[str, Any]] = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    self.records[str(record["key"])] = record

    def payload(
        self, system: Sequence[Mapping[str, Any]], user: str, max_tokens: int
    ) -> dict[str, Any]:
        """The request payload (and so the cache key) of one call."""
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": [dict(block) for block in system],
            "messages": [{"role": "user", "content": user}],
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        return body

    def spent(self) -> float:
        """Cost in USD of every cached response at the assumed prices."""
        return sum(usage_cost(r.get("usage", {})) for r in self.records.values())

    def complete(
        self, kind: str, system: Sequence[Mapping[str, Any]], user: str, max_tokens: int
    ) -> Reply:
        """The reply to one request: from the cache, else one live call that is then stored."""
        payload = self.payload(system, user, max_tokens)
        key = request_key(payload)
        self.touched.add(key)
        with self._key_locks[key]:
            record = self.records.get(key)
            if record is not None:
                return Reply(str(record["reply"]), record.get("usage", {}), cached=True)
            if not self.api_key:
                raise CacheMiss(f"{kind}: not in {self.path} and no API key (offline replay)")
            with self._lock:
                if len(self.records) >= self.max_calls:
                    raise RuntimeError(f"call budget of {self.max_calls} reached")
                if self.spent() >= self.max_cost_usd:
                    raise RuntimeError(f"cost budget of {self.max_cost_usd} USD reached")
            dropped = False
            try:
                response = self.transport(payload, self.api_key)
            except RuntimeError as err:
                if "temperature" not in str(err) or "temperature" not in payload:
                    raise
                payload = {k: v for k, v in payload.items() if k != "temperature"}
                dropped = True
                response = self.transport(payload, self.api_key)
            text = "".join(
                str(b.get("text", "")) for b in response.get("content", []) if b.get("text")
            )
            usage = {k: int(v) for k, v in response.get("usage", {}).items() if isinstance(v, int)}
            record = {
                "key": key,
                "kind": kind,
                "model": self.model,
                "temperature_dropped": dropped,
                "request": {"system": _summarise_system(system), "user": user},
                "usage": usage,
                "stop_reason": response.get("stop_reason"),
                "reply": text,
            }
            with self._lock:
                self.records[key] = record
                self.live_calls += 1
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            return Reply(text, usage, cached=False)


def usage_totals(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Summed token counts over cached records, plus the call count."""
    totals: dict[str, int] = {"calls": len(records)}
    for record in records:
        for name, value in record.get("usage", {}).items():
            totals[name] = totals.get(name, 0) + int(value)
    return totals


def usage_by_kind(records: Sequence[Mapping[str, Any]]) -> dict[str, tuple[int, float]]:
    """``kind -> (calls, cost in USD)`` at the assumed prices."""
    out: dict[str, tuple[int, float]] = {}
    for record in records:
        calls, cost = out.get(str(record["kind"]), (0, 0.0))
        out[str(record["kind"])] = (calls + 1, cost + usage_cost(record.get("usage", {})))
    return out
