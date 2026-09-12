"""Unit tests for RetrievalAgent and InMemoryDocumentStore."""

import pytest

from enterprise_orchestrator.agents.result import AgentStatus
from enterprise_orchestrator.agents.retrieval import (
    Document,
    InMemoryDocumentStore,
    RetrievalAgent,
    RetrievalQuery,
)
from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.core.task import Task


def make_sample_docs():
    return [
        Document(
            id="doc_cloud_1",
            content="Kubernetes cluster autoscaling and pod disruption budgets.",
            metadata={"domain": "infrastructure", "tier": "enterprise"},
        ),
        Document(
            id="doc_fin_1",
            content="Q3 Revenue increased by 14% with strong recurring subscription growth.",
            metadata={"domain": "finance", "quarter": "Q3"},
        ),
        Document(
            id="doc_sec_1",
            content="Authentication protocol enforces OAuth2 and JWT validation.",
            metadata={"domain": "security"},
        ),
    ]


@pytest.fixture
def populated_store():
    store = InMemoryDocumentStore()
    docs = make_sample_docs()
    for doc in docs:
        store._documents[doc.id] = doc
    return store


@pytest.mark.asyncio
async def test_document_store_insertion_and_retrieval(populated_store):
    """Verify document search, ranking, and count."""
    assert await populated_store.count() == 3

    query = RetrievalQuery(query="revenue subscription growth", top_k=2)
    result = await populated_store.retrieve(query)

    assert result.total_found == 1
    assert len(result.matches) == 1
    assert result.matches[0].document.id == "doc_fin_1"
    assert result.matches[0].score > 0.5


@pytest.mark.asyncio
async def test_document_store_metadata_filtering(populated_store):
    """Verify filtering by metadata tags."""
    query = RetrievalQuery(
        query="autoscaling validation",
        filters={"domain": "infrastructure"},
    )
    result = await populated_store.retrieve(query)
    assert result.total_found == 1
    assert result.matches[0].document.id == "doc_cloud_1"


@pytest.mark.asyncio
async def test_document_store_no_results(populated_store):
    """Verify clean empty match handling for non-matching queries."""
    query = RetrievalQuery(query="banana apple pineapple")
    result = await populated_store.retrieve(query)
    assert result.total_found == 0
    assert len(result.matches) == 0


@pytest.mark.asyncio
async def test_retrieval_agent_execution_with_task_input(populated_store):
    """Verify RetrievalAgent resolves query from Task input and populates state."""
    agent = RetrievalAgent(document_store=populated_store)
    state = OrchestrationState.create_initial(request="Main request")
    task = Task(
        title="Search Auth Docs",
        input_data={"query": "OAuth2 authentication", "top_k": 1},
    )

    result = await agent.execute(state, task)

    assert result.status == AgentStatus.SUCCESS
    assert result.agent_name == "retrieval_agent"
    matches = result.output["matches"]
    assert len(matches) == 1
    assert matches[0]["id"] == "doc_sec_1"
    assert len(state.retrieved_context) == 1
    assert state.retrieved_context[0]["id"] == "doc_sec_1"


@pytest.mark.asyncio
async def test_retrieval_agent_fallback_to_state_request(populated_store):
    """Verify RetrievalAgent uses state.request if task input has no explicit query."""
    agent = RetrievalAgent(document_store=populated_store)
    state = OrchestrationState.create_initial(request="Kubernetes autoscaling")

    result = await agent.execute(state)

    assert result.status == AgentStatus.SUCCESS
    assert result.output["total_found"] == 1
    assert result.output["matches"][0]["id"] == "doc_cloud_1"


@pytest.mark.asyncio
async def test_retrieval_agent_empty_query_failure(populated_store):
    """Verify agent failure if query is empty."""
    agent = RetrievalAgent(document_store=populated_store)
    state = OrchestrationState(request=" ")

    result = await agent.execute(state)
    assert result.status == AgentStatus.FAILURE
    assert "No valid query" in result.error
