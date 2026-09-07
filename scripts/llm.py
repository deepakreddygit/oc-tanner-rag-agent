"""Chat model factory and retry policy.

The factory is isolated in its own module so the provider can be swapped (or replaced
with a fake in tests) without touching agent logic. Only OpenAI is wired up per this
assignment's choice, but nothing in scripts/agent.py depends on that concretely -- it only
depends on langchain_core.language_models.BaseChatModel.

`invoke_with_retry` wraps every model call the agent makes. An agentic system that
calls an external LLM API on every step needs to expect that API to be occasionally
and transiently unavailable -- a rate limit, a timeout, a 5xx -- and recover instead of
failing the whole request outright. This retries only the error classes where retrying
can actually help (rate limits, timeouts, connection errors, server errors); it does
not retry on bad input or auth failures, where retrying would just waste time before
failing anyway. Three attempts with a short exponential backoff rides out a brief blip
without making a user wait long against a genuinely down provider.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from tenacity import RetryCallState, retry, retry_if_exception, stop_after_attempt, wait_exponential

from scripts.config import settings
from scripts.observability import log_event

if TYPE_CHECKING:
    from langchain_core.messages import AIMessage


def build_llm() -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return ChatOpenAI(
        model=settings.chat_model,
        temperature=settings.chat_temperature,
        api_key=settings.openai_api_key,
    )


def _is_transient_llm_error(exc: BaseException) -> bool:
    try:
        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
    except ImportError:
        return False  # a fake/test model is in use; nothing OpenAI-specific to retry
    return isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError))


def _log_retry(retry_state: RetryCallState) -> None:
    """Fires only when about to retry (not on the final failure or on success), so a
    quiet request that never needed a retry produces no extra log noise. `retry_state`
    is local to this specific call's attempt loop, not shared across concurrent
    requests, so this is safe under FastAPI's threaded request handling."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    log_event(
        "llm_retry",
        node=retry_state.kwargs.get("node", "unknown"),
        session_id=retry_state.kwargs.get("session_id", "unknown"),
        attempt=retry_state.attempt_number,
        error=str(exc) if exc else None,
    )


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=4),
    retry=retry_if_exception(_is_transient_llm_error),
    before_sleep=_log_retry,
)
def invoke_with_retry(llm: BaseChatModel, messages: list[BaseMessage], *, node: str, session_id: str) -> "AIMessage":
    """`node` and `session_id` are for logging only (see `_log_retry`) -- they don't
    change what gets sent to the model."""
    return llm.invoke(messages)
