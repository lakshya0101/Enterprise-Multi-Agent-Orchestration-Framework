"""Unit tests for FAISS vector store indexing, similarity search, and persistence."""

import pytest

from enterprise_orchestrator.errors.exceptions import RetrievalError
from enterprise_orchestrator.retrieval.chunking import DocumentChunk
from enterprise_orchestrator.retrieval.embeddings import FakeEmbeddingProvider
from enterprise_orchestrator.retrieval.vector_store import FAISSVectorStore


@pytest.mark.asyncio
async def test_faiss_vector_store_indexing_and_search():
    """Verify FAISS vector store indexes chunks and returns ranked matches."""
    dim = 32
    store = FAISSVectorStore(dimension=dim)
    embedder = FakeEmbeddingProvider(dimension=dim)

    chunks = [
        DocumentChunk(
            id="chunk_1",
            document_id="doc_1",
            chunk_index=0,
            content="Alpha quarterly revenue report.",
            metadata={"domain": "finance"},
        ),
        DocumentChunk(
            id="chunk_2",
            document_id="doc_2",
            chunk_index=0,
            content="Beta engineering deployment manual.",
            metadata={"domain": "devops"},
        ),
    ]

    embeddings = await embedder.embed_documents([c.content for c in chunks])
    await store.add_chunks(chunks=chunks, embeddings=embeddings)

    assert await store.count() == 2

    # Query matching chunk 1
    query_vec = await embedder.embed_text("Alpha quarterly revenue report.")
    results = await store.similarity_search(query_embedding=query_vec, top_k=2)

    assert len(results) == 2
    assert results[0].document.id == "chunk_1"
    assert results[0].score > results[1].score
    assert results[0].document.metadata["domain"] == "finance"
    assert results[0].document.metadata["parent_doc_id"] == "doc_1"


@pytest.mark.asyncio
async def test_faiss_vector_store_metadata_filtering():
    """Verify similarity search respects metadata filters."""
    dim = 16
    store = FAISSVectorStore(dimension=dim)
    embedder = FakeEmbeddingProvider(dimension=dim)

    chunks = [
        DocumentChunk(id="c1", document_id="d1", chunk_index=0, content="Document A", metadata={"tag": "public"}),
        DocumentChunk(id="c2", document_id="d2", chunk_index=0, content="Document B", metadata={"tag": "private"}),
    ]
    embeddings = await embedder.embed_documents([c.content for c in chunks])
    await store.add_chunks(chunks=chunks, embeddings=embeddings)

    query_vec = await embedder.embed_text("Document query")
    filtered_results = await store.similarity_search(
        query_embedding=query_vec,
        top_k=5,
        filters={"tag": "private"},
    )

    assert len(filtered_results) == 1
    assert filtered_results[0].document.id == "c2"


@pytest.mark.asyncio
async def test_faiss_vector_store_empty_index_safety():
    """Verify searching an empty FAISS vector store returns empty list gracefully."""
    store = FAISSVectorStore(dimension=16)
    results = await store.similarity_search(query_embedding=[0.1] * 16, top_k=3)
    assert results == []
    assert await store.count() == 0


@pytest.mark.asyncio
async def test_faiss_vector_store_persistence_lifecycle(tmp_path):
    """Verify saving and loading FAISS vector store to/from disk."""
    save_dir = str(tmp_path / "faiss_test_index")
    dim = 24
    store = FAISSVectorStore(dimension=dim)
    embedder = FakeEmbeddingProvider(dimension=dim)

    chunks = [
        DocumentChunk(id="c1", document_id="d1", chunk_index=0, content="Data chunk 1", metadata={"status": "active"}),
        DocumentChunk(id="c2", document_id="d2", chunk_index=0, content="Data chunk 2", metadata={"status": "archived"}),
    ]
    embeddings = await embedder.embed_documents([c.content for c in chunks])
    await store.add_chunks(chunks, embeddings)

    # Save to disk
    await store.save(save_dir)

    # Load in new store instance
    loaded_store = FAISSVectorStore(dimension=dim)
    await loaded_store.load(save_dir)

    assert await loaded_store.count() == 2
    query_vec = await embedder.embed_text("Data chunk 1")
    matches = await loaded_store.similarity_search(query_vec, top_k=1)
    assert len(matches) == 1
    assert matches[0].document.id == "c1"


@pytest.mark.asyncio
async def test_faiss_vector_store_corrupted_load_error(tmp_path):
    """Verify error is raised when index files are missing or mismatched."""
    store = FAISSVectorStore(dimension=16)
    with pytest.raises(RetrievalError):
        await store.load(str(tmp_path / "non_existent_folder"))
