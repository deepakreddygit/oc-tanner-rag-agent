from scripts.config import settings
from scripts.vectorstore import load_and_chunk_document


def test_chunking_produces_multiple_chunks_with_metadata():
    docs = load_and_chunk_document(settings.source_document)
    assert len(docs) > 3
    assert all(d.metadata["source"] == settings.source_document.name for d in docs)
    assert all("chunk_index" in d.metadata for d in docs)
    # chunk_index should be sequential and unique
    indices = [d.metadata["chunk_index"] for d in docs]
    assert indices == list(range(len(docs)))


def test_retrieval_finds_relevant_chunk_for_refund_query(policy_vectorstore):
    results = policy_vectorstore.similarity_search("restocking fee refund window", k=2)
    assert results
    assert any("restocking" in r.page_content.lower() for r in results)


def test_retrieval_finds_shipping_chunk_for_shipping_query(policy_vectorstore):
    results = policy_vectorstore.similarity_search("how long does standard shipping take", k=2)
    assert results
    assert any("shipping" in r.page_content.lower() for r in results)
