"""Unit tests for document chunking."""

import pytest

from enterprise_orchestrator.agents.retrieval import Document
from enterprise_orchestrator.retrieval.chunking import SimpleTextChunker


def test_simple_chunker_validation():
    """Verify parameter validations on chunk_size and chunk_overlap."""
    with pytest.raises(ValueError):
        SimpleTextChunker(chunk_size=0)
    with pytest.raises(ValueError):
        SimpleTextChunker(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        SimpleTextChunker(chunk_size=100, chunk_overlap=-5)


@pytest.mark.asyncio
async def test_simple_chunker_short_and_empty_document():
    """Verify behavior on short and empty documents."""
    chunker = SimpleTextChunker(chunk_size=100, chunk_overlap=20)
    
    # Empty document
    empty_doc = Document(id="doc_empty", content="   ")
    chunks = await chunker.chunk_document(empty_doc)
    assert chunks == []

    # Short document
    short_doc = Document(id="doc_short", content="Brief sentence.", metadata={"source": "faq"})
    chunks = await chunker.chunk_document(short_doc)
    assert len(chunks) == 1
    assert chunks[0].id == "doc_short_chunk_0"
    assert chunks[0].content == "Brief sentence."
    assert chunks[0].metadata["source"] == "faq"
    assert chunks[0].metadata["source_doc_id"] == "doc_short"


@pytest.mark.asyncio
async def test_simple_chunker_sliding_window_overlap():
    """Verify chunker creates multiple overlapping chunks with sequential indices."""
    chunker = SimpleTextChunker(chunk_size=50, chunk_overlap=10)
    
    content = "The quick brown fox jumps over the lazy dog. " * 3  # ~135 chars
    doc = Document(id="doc_long", content=content, metadata={"category": "animals"})
    
    chunks = await chunker.chunk_document(doc)
    assert len(chunks) >= 3
    
    for i, c in enumerate(chunks):
        assert c.id == f"doc_long_chunk_{i}"
        assert c.chunk_index == i
        assert c.document_id == "doc_long"
        assert c.metadata["category"] == "animals"
        assert len(c.content) <= 50

    # Test batch chunking
    docs = [
        Document(id="doc_1", content="First doc text."),
        Document(id="doc_2", content="Second doc text."),
    ]
    batch_chunks = await chunker.chunk_documents(docs)
    assert len(batch_chunks) == 2
    assert batch_chunks[0].document_id == "doc_1"
    assert batch_chunks[1].document_id == "doc_2"
