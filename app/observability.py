"""Minimal structured-event logging.

A production agentic system needs traces of model calls, retrieved context, latency,
and errors (see the job's emphasis on observability). A full tracing stack (Langfuse,
OpenTelemetry) is out of scope for this assignment, but the *shape* of what should be
captured is not: every node emits one structured JSON log line with a request id, a
node name, timing, and node-specific fields. Swapping this for a real exporter later is
a one-line change in `log_event`, not a redesign.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger("agent")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def log_event(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, default=str))


@contextmanager
def timed_event(event: str, **fields: Any) -> Iterator[dict]:
    """Context manager that logs `event` with a duration_ms field, plus whatever the
    caller adds to the yielded dict before the block exits."""
    start = time.perf_counter()
    extra: dict = {}
    try:
        yield extra
    finally:
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        log_event(event, duration_ms=duration_ms, **fields, **extra)
