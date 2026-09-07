"""In-memory conversation history, keyed by session_id.

The assignment explicitly states in-memory history is sufficient, so this intentionally
does not reach for Redis/Postgres. The one production concern worth handling even at
this scope is unbounded growth: a long-running session would otherwise grow its prompt
(and cost/latency) forever, so history is capped to the most recent `max_turns` turns.
A `threading.Lock` guards the dict since FastAPI can serve requests from multiple
threads (sync route handlers run in a thread pool).
"""

from __future__ import annotations

import threading

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class SessionStore:
    def __init__(self, max_turns: int = 20):
        self._sessions: dict[str, list[BaseMessage]] = {}
        self._lock = threading.Lock()
        self.max_turns = max_turns

    def get_history(self, session_id: str) -> list[BaseMessage]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def append_turn(self, session_id: str, user_message: str, answer: str) -> None:
        with self._lock:
            history = self._sessions.setdefault(session_id, [])
            history.append(HumanMessage(content=user_message))
            history.append(AIMessage(content=answer))
            overflow = len(history) - self.max_turns * 2
            if overflow > 0:
                del history[:overflow]

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


session_store = SessionStore()
