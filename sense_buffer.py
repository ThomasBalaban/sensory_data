"""
SenseBuffer for the Context Service.
A typed, timestamped rolling buffer for one sense stream.
All entries carry an ISO wall-clock timestamp so the classifier
and downstream AI can correlate events across senses.
"""

import time
from collections import deque
from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _now_ts() -> float:
    return time.time()


class SenseEntry:
    __slots__ = ("text", "iso_ts", "unix_ts", "source")

    def __init__(self, text: str, source: str):
        self.text    = text.strip()
        self.iso_ts  = _now_iso()
        self.unix_ts = _now_ts()
        self.source  = source   # "vision" | "audio" | "mic"

    def age_s(self) -> float:
        return _now_ts() - self.unix_ts

    def formatted(self) -> str:
        """Returns a single line ready to drop into a classifier prompt."""
        # e.g.  [14:03:22.410] vision: Character takes a critical hit, health bar flashes red
        hms = datetime.fromtimestamp(self.unix_ts).strftime("%H:%M:%S.") + \
              f"{int((self.unix_ts % 1) * 1000):03d}"
        return f"[{hms}] {self.source}: {self.text}"


class SenseBuffer:
    """
    Thread-safe rolling buffer for one sense.
    Automatically marks entries as stale after `stale_after_s` seconds.
    """

    def __init__(self, name: str, maxlen: int, stale_after_s: float):
        self.name         = name
        self.stale_after_s = stale_after_s
        self._buf: deque[SenseEntry] = deque(maxlen=maxlen)

        # Deduplication: skip entries that are nearly identical to the last one
        self._last_text = ""

    def add(self, text: str):
        text = text.strip()
        if not text:
            return

        # Light deduplication — skip if >85% overlap with last entry
        if self._last_text and self._similarity(text, self._last_text) > 0.85:
            return

        self._last_text = text
        self._buf.appendleft(SenseEntry(text, self.name))   # newest first

    def recent(self, max_age_s: float | None = None) -> list[SenseEntry]:
        """Return entries that are not stale, newest first."""
        cutoff = max_age_s if max_age_s is not None else self.stale_after_s
        return [e for e in self._buf if e.age_s() <= cutoff]

    def formatted_lines(self, max_age_s: float | None = None) -> list[str]:
        return [e.formatted() for e in self.recent(max_age_s)]

    def latest(self) -> SenseEntry | None:
        return self._buf[0] if self._buf else None

    def is_stale(self) -> bool:
        entry = self.latest()
        return entry is None or entry.age_s() > self.stale_after_s

    def clear(self):
        self._buf.clear()
        self._last_text = ""

    # ── Internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """Rough character-level Jaccard similarity. Fast, no imports."""
        if not a or not b:
            return 0.0
        set_a = set(a.lower().split())
        set_b = set(b.lower().split())
        if not set_a or not set_b:
            return 0.0
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        return inter / union if union else 0.0