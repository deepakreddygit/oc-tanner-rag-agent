from scripts.config import settings
from scripts.vectorstore import build_bm25_retriever, load_and_chunk_document


def test_chunking_produces_multiple_chunks_with_metadata():
    docs = load_and_chunk_document(settings.source_document)
    assert len(docs) > 3
    assert all(d.metadata["source"] == settings.source_document.name for d in docs)
    assert all("chunk_index" in d.metadata for d in docs)
    # chunk_index should be sequential and unique
    indices = [d.metadata["chunk_index"] for d in docs]
    assert indices == list(range(len(docs)))


def test_dense_retrieval_finds_relevant_chunk_for_refund_query(policy_vectorstore):
    results = policy_vectorstore.similarity_search("restocking fee refund window", k=2)
    assert results
    assert any("restocking" in r.page_content.lower() for r in results)


def test_dense_retrieval_finds_shipping_chunk_for_shipping_query(policy_vectorstore):
    results = policy_vectorstore.similarity_search("how long does standard shipping take", k=2)
    assert results
    assert any("shipping" in r.page_content.lower() for r in results)


def test_hybrid_retrieval_finds_relevant_chunk_for_refund_query(policy_hybrid_retriever):
    """Exercises the actual retriever the agent uses in production (BM25 + dense,
    fused) rather than the dense side alone -- catches a regression in the fusion
    itself (weights, wiring) that the dense-only tests above wouldn't."""
    results = policy_hybrid_retriever.invoke("restocking fee refund window")
    assert results
    assert any("restocking" in r.page_content.lower() for r in results)


def test_bm25_alone_finds_exact_keyword_match_dense_could_blur(policy_vectorstore):
    """BM25's whole reason for being here: it should surface a chunk containing a
    query term verbatim even in a corpus small enough that dense embeddings alone
    might not clearly separate it from neighboring chunks."""
    bm25 = build_bm25_retriever(k=2)
    results = bm25.invoke("Acme Plus membership")
    assert results
    assert any("acme plus" in r.page_content.lower() for r in results)
