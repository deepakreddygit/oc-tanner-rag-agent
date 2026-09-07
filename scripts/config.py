"""Centralized application settings.

All configuration is sourced from environment variables (loaded from a .env file in
local development). Keeping every tunable in one place makes the agent's behavior
auditable at a glance -- there is no hidden config scattered across modules.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM provider (generation) ---
    openai_api_key: str = ""
    chat_model: str = "gpt-4o-mini"
    chat_temperature: float = 0.0

    # --- Embeddings (retrieval) ---
    # "huggingface" (default): local sentence-transformers model, offline, no API key
    # or per-call cost. "openai": hosted embeddings -- used for the memory-capped
    # free-tier deployment where loading torch locally risks OOM. See
    # scripts/vectorstore.py and README "Deployment".
    embedding_provider: str = "huggingface"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"

    # --- Vector store ---
    vectorstore_dir: Path = BASE_DIR / "vectorstore"
    source_document: Path = BASE_DIR / "data" / "acme_corp_customer_policies.md"

    # --- Chunking ---
    chunk_size: int = 800
    chunk_overlap: int = 120

    # --- Retrieval ---
    retrieval_k: int = 4

    # --- App ---
    log_level: str = "INFO"


settings = Settings()
