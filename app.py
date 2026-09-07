"""FastAPI service exposing POST /chat.

Single entrypoint at the repo root -- run it directly (`python app.py`) to start the
server, then hit `/chat` (or open `/docs` in a browser) to get an answer. All the
actual logic (agent graph, vector store, config, etc.) lives in scripts/, this file
just wires it together into an API and starts it.

The agent and vector store are constructed once at startup (not per-request) and handed
to route handlers via FastAPI's dependency system, which also makes them trivial to
override with fakes in tests (see tests/test_api.py).
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Request

from scripts.agent import Agent
from scripts.llm import build_llm
from scripts.memory import session_store
from scripts.observability import log_event
from scripts.rate_limit import chat_rate_limiter
from scripts.schemas import ChatRequest, ChatResponse
from scripts.vectorstore import build_hybrid_retriever, load_vectorstore, vectorstore_exists

app = FastAPI(
    title="Acme Corp RAG Agent",
    description="Retrieval-augmented Q&A over Acme Corp's customer policies.",
    version="0.1.0",
)

_agent: Agent | None = None


def get_agent() -> Agent:
    """Lazily build the singleton Agent on first request. Lazy (not at import time) so
    the app can start and serve /health even before ingestion has run; the clear error
    below only surfaces when a chat is actually attempted."""
    global _agent
    if _agent is None:
        if not vectorstore_exists():
            raise HTTPException(
                status_code=503,
                detail="Vector store not found. Run `python scripts/ingest.py` first.",
            )
        retriever = build_hybrid_retriever(load_vectorstore())
        _agent = Agent(llm=build_llm(), retriever=retriever)
    return _agent


def get_client_ip(request: Request) -> str:
    """Render (like most PaaS hosts) puts the app behind a reverse proxy, so
    `request.client.host` is the proxy's own address, not the caller's -- every request
    would share one rate-limit bucket. `X-Forwarded-For` carries the real chain; its
    first entry is the original client. Trusted here because the only way to reach this
    app is through Render's proxy, which sets this header itself -- a caller can't
    reach this process directly to forge it."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(request: Request) -> None:
    client_ip = get_client_ip(request)
    if not chat_rate_limiter.allow(client_ip):
        log_event("rate_limited", client_ip=client_ip)
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait a moment before trying again.",
        )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "vectorstore_ready": vectorstore_exists()}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(enforce_rate_limit)])
def chat(req: ChatRequest, agent: Agent = Depends(get_agent)) -> ChatResponse:
    history = session_store.get_history(req.session_id)
    try:
        answer = agent.run(req.session_id, req.message, history)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - convert any agent failure into a 500
        log_event("chat_error", session_id=req.session_id, error=str(exc))
        raise HTTPException(
            status_code=500, detail="The agent failed to produce a response."
        ) from exc
    session_store.append_turn(req.session_id, req.message, answer)
    return ChatResponse(session_id=req.session_id, response=answer)


if __name__ == "__main__":
    # Lets you just run `python app.py` instead of remembering the uvicorn invocation.
    # No --reload here since this is a direct run, not the CLI -- use
    # `uvicorn app:app --reload` instead if you want auto-reload while editing.
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
