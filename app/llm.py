"""Chat model factory.

Isolated in its own module so the provider can be swapped (or replaced with a fake in
tests) without touching agent logic. Only OpenAI is wired up per this assignment's
choice, but nothing in app/agent.py depends on that concretely -- it only depends on
langchain_core.language_models.BaseChatModel.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from app.config import settings


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
