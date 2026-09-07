"""Vector store construction and loading.

FAISS was chosen over pgvector/Chroma because this assignment ingests a single small
document and runs as a single process -- FAISS needs no running database service, which
keeps "setup instructions from zero" to a `pip install` and a script run. See README
"Trade-offs" for when this choice would change (multi-tenant documents, concurrent
writers, or a need for metadata filtering at scale would push toward pgvector).

The embedding *provider* behind FAISS is swappable via `EMBEDDING_PROVIDER`:

- `huggingface` (default, for local dev): a local sentence-transformers model. No API
  key or per-call cost, at the price of pulling in torch -- several hundred MB of RAM
  once loaded.
- `openai`: OpenAI's hosted embeddings API. No local model to load, which matters on a
  memory-capped host (e.g. a 512MB free-tier web service) where torch alone can be the
  difference between fitting and OOM-crashing. See README "Deployment" for why the
  hosted demo uses this and local dev doesn't.

Both branches build a FAISS index the same way; nothing downstream (agent.py) knows or
cares which one produced it, as long as ingestion and serving agree on the same
provider for a given deployment -- swapping providers on an existing index would break
retrieval (the stored vectors' dimensionality wouldn't match new queries), so this is a
per-environment setting, not a per-request one.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Cached so the embedding model/client is created once per process, not once per
    request (the HuggingFace branch in particular is expensive to load repeatedly)."""
    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.openai_embedding_model, api_key=settings.openai_api_key)

    # Imported lazily so a deployment running the "openai" branch never pays for
    # importing torch/sentence-transformers at all.
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def load_and_chunk_document(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [
        Document(page_content=chunk, metadata={"source": path.name, "chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]


def build_vectorstore(source_path: Path | None = None) -> FAISS:
    """Ingest `source_path` (default: the configured Acme Corp policy doc) into a fresh
    FAISS index and persist it to disk."""
    source_path = source_path or settings.source_document
    if not source_path.exists():
        raise FileNotFoundError(f"Source document not found: {source_path}")

    documents = load_and_chunk_document(source_path)
    if not documents:
        raise ValueError(f"No content extracted from {source_path}")

    store = FAISS.from_documents(documents, get_embeddings())
    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(settings.vectorstore_dir))
    return store


def vectorstore_exists() -> bool:
    index_file = settings.vectorstore_dir / "index.faiss"
    return index_file.exists()


@lru_cache(maxsize=1)
def load_vectorstore() -> FAISS:
    """Load the persisted FAISS index. Raises a clear error if ingestion has not run
    yet, instead of silently returning an empty/broken retriever."""
    if not vectorstore_exists():
        raise RuntimeError(
            "No vector store found. Run `python scripts/ingest.py` before starting the "
            "API or calling /chat."
        )
    return FAISS.load_local(
        str(settings.vectorstore_dir),
        get_embeddings(),
        allow_dangerous_deserialization=True,  # local file we generated ourselves
    )
