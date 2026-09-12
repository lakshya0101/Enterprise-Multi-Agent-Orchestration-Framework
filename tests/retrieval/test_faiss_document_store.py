"""Unit and integration tests for FAISSDocumentStore with RetrievalAgent."""

import pytest

from enterprise_orchestrator.agents.retrieval import (
    Document,
    InMemoryDocumentStore,
    RetrievalAgent,
    RetrievalQuery,
)
from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.core.task import Task
from enterprise_orchestrator.core.types import AgentRole, TaskStatus, TaskType
from enterprise_orchestrator.retrieval.chunking import SimpleTextChunker
from enterprise_orchestrator.retrieval.embeddings import FakeEmbeddingProvider
from enterprise_orchestrator.retrieval.faiss_store import FAISSDocumentStore


@pytest.mark.asyncio
async def test_faiss_document_store_end_to_end_retrieval():
    """Scenario A: Document -> Chunk -> Fake Embedding -> FAISS -> RetrievalAgent -> Context."""
    dim = 32
    embedder = FakeEmbeddingProvider(dimension=dim)
    chunker = SimpleTextChunker(chunk_size=100, chunk_overlap=20)
    store = FAISSDocumentStore(embedding_provider=embedder, chunker=chunker)

    # Ingest source document
    doc = Document(
        id="doc_enterprise_policy",
        content="Enterprise AI Guidelines: All model executions must pass through the validation critic before deployment.",
        metadata={"category": "governance", "version": "1.0"},
    )
    await store.add_documents([doc])
    assert await store.count() >= 1

    # Execute retrieval through RetrievalAgent
    agent = RetrievalAgent(document_store=store)
    state = OrchestrationState.create_initial(request="What are the enterprise validation rules?")
    task = Task(
        id="task_retrieval_1",
        title="Retrieve AI Policy",
        task_type=TaskType.RETRIEVAL,
        assigned_agent=AgentRole.RETRIEVER,
        status=TaskStatus.PENDING,
        input_data={"query": "validation critic before deployment"},
    )

    result = await agent.execute(state, task)
    assert result.status.value == "success"
    assert result.output is not None
    assert result.output["total_found"] >= 1
    
    top_match = result.output["matches"][0]
    assert "validation critic" in top_match["content"]
    assert top_match["metadata"]["category"] == "governance"
    assert top_match["metadata"]["parent_doc_id"] == "doc_enterprise_policy"


@pytest.mark.asyncio
async def test_in_memory_and_faiss_store_contract_parity():
    """Verify InMemoryDocumentStore and FAISSDocumentStore adhere to identical BaseDocumentStore contract."""
    docs = [
        Document(id="d1", content="Financial Q3 report revenue up 20%"),
        Document(id="d2", content="HR recruitment process update"),
    ]

    mem_store = InMemoryDocumentStore()
    faiss_store = FAISSDocumentStore()

    await mem_store.add_documents(docs)
    await faiss_store.add_documents(docs)

    query = RetrievalQuery(query="revenue", top_k=1)
    mem_res = await mem_store.retrieve(query)
    faiss_res = await faiss_store.retrieve(query)

    assert mem_res.total_found >= 1
    assert faiss_res.total_found >= 1
    assert "revenue" in mem_res.matches[0].document.content.lower()
    assert "revenue" in faiss_res.matches[0].document.content.lower()
