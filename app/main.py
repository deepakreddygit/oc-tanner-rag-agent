"""FastAPI service exposing POST /chat.

The agent and vector store are constructed once at startup (not per-request) and handed
to route handlers via FastAPI's dependency system, which also makes them trivial to
override with fakes in tests (see tests/test_api.py).
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from app.agent import Agent
from app.llm import build_llm
from app.memory import session_store
from app.observability import log_event
from app.schemas import ChatRequest, ChatResponse
from app.vectorstore import load_vectorstore, vectorstore_exists

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
        _agent = Agent(llm=build_llm(), vectorstore=load_vectorstore())
    return _agent


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "vectorstore_ready": vectorstore_exists()}


@app.post("/chat", response_model=ChatResponse)
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
