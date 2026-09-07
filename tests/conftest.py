import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from langchain_community.vectorstores import FAISS

from scripts.config import settings
from scripts.vectorstore import load_and_chunk_document
from tests.fakes import FakeHashEmbeddings


@pytest.fixture(scope="session")
def policy_vectorstore() -> FAISS:
    """A real FAISS index over the real sample document, embedded with the fake
    hashing embedder so the test suite never needs network access."""
    docs = load_and_chunk_document(settings.source_document)
    return FAISS.from_documents(docs, FakeHashEmbeddings())
