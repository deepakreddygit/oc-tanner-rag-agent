"""Test doubles.

Both fakes exist so the test suite can verify the agent's control flow and prompt
construction without a network call to an embedding provider or an LLM provider. They
are never imported by application code -- production always uses the real
HuggingFace embeddings and ChatOpenAI (see scripts/vectorstore.py and scripts/llm.py).
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

_DIM = 256


def _hash_vector(text: str) -> list[float]:
    """Deterministic bag-of-words hashing embedding. Good enough to make FAISS
    similarity search return keyword-overlapping chunks first, which is all these
    tests need -- it is not meant to approximate real embedding quality."""
    vec = [0.0] * _DIM
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % _DIM
        vec[idx] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec] if norm else vec


class FakeHashEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_hash_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return _hash_vector(text)


class RecordingFakeChatModel(BaseChatModel):
    """Returns canned responses in order and records every message list it was
    invoked with, so tests can assert on what the agent actually sent the model
    (e.g. that retrieved context made it into the system prompt)."""

    responses: list[str]
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(messages)
        idx = len(self.calls) - 1
        content = self.responses[idx] if idx < len(self.responses) else self.responses[-1]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self) -> str:
        return "recording-fake"
