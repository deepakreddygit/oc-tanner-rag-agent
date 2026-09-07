"""Per-client rate limiting for the public /chat endpoint.

This is now a publicly reachable demo backed by a paid LLM API, not an internal tool
behind auth -- so a single caller (accidental retry loop, a scraper, or someone just
hammering the Swagger UI) can run up real cost with no one else in the loop. A fixed
per-key window is the simplest thing that actually stops that, without adding an
external dependency (Redis, etc.) that this single-instance free-tier deployment has no
use for. If this ever ran behind multiple instances, the counters would need to move to
a shared store -- called out in the README rather than solved speculatively here.

Deliberately in-memory and process-local, same trade-off already made by
app/memory.py's SessionStore, and for the same reason: one process, one free-tier
instance, no shared state needed yet.
"""

from __future__ import annotations

import threading
import time
from collections import deque


class RateLimiter:
    """Fixed-window-ish limiter: allows at most `max_requests` calls per key in any
    trailing `window_seconds` window. Deque-per-key so old timestamps are cheap to
    evict and memory doesn't grow unbounded for a key that goes quiet."""

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, *, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            cutoff = now - self.window_seconds
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.max_requests:
                return False
            hits.append(now)
            return True


# 20 requests/minute/IP: generous enough for a real conversation (each turn is one
# request) or a reviewer trying all three required behaviors, low enough that a runaway
# loop against a metered API gets stopped within seconds, not dollars.
chat_rate_limiter = RateLimiter(max_requests=20, window_seconds=60.0)
