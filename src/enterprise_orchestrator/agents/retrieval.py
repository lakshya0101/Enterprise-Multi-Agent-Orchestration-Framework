"""Retrieval agent and local document store abstractions."""

import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field

from enterprise_orchestrator.agents.base import AgentMetadata, BaseAgent
from enterprise_orchestrator.agents.result import AgentResult
from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.core.task import Task
from enterprise_orchestrator.core.types import AgentRole
from enterprise_orchestrator.errors.exceptions import RetrievalError


class Document(BaseModel):
    """Normalized document representation for search and context injection."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique document ID")
    content: str = Field(..., min_length=1, description="Textual body content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata attributes (e.g. source, author, category)")


class ScoredDocument(BaseModel):
    """Document paired with relevance score."""

    document: Document = Field(..., description="Matched document")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized relevance score between 0.0 and 1.0")


class RetrievalQuery(BaseModel):
    """Structured search query envelope."""

    query: str = Field(..., min_length=1, description="Search terms or semantic query text")
    top_k: int = Field(default=3, ge=1, description="Maximum number of relevant documents to return")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filter criteria")


class RetrievalResult(BaseModel):
    """Standardized retrieval output payload."""

    query: str = Field(..., description="Query executed")
    matches: List[ScoredDocument] = Field(default_factory=list, description="Ranked matching documents")
    total_found: int = Field(default=0, ge=0, description="Total matching documents found")


class BaseDocumentStore(ABC):
    """Abstract interface for document ingestion and retrieval backends."""

    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> None:
        """Ingest documents into the store."""
        pass

    @abstractmethod
    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Search and rank documents matching the query."""
        pass

    @abstractmethod
    async def count(self) -> int:
        """Return total document count."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Purge all indexed documents."""
        pass


class InMemoryDocumentStore(BaseDocumentStore):
    """Deterministic in-memory document store using lexical token-overlap scoring for testing."""

    def __init__(self) -> None:
        self._documents: Dict[str, Document] = {}

    def _tokenize(self, text: str) -> Set[str]:
        """Simple lowercase word tokenizer."""
        return set(re.findall(r"\b\w+\b", text.lower()))

    async def add_documents(self, documents: List[Document]) -> None:
        """Store documents in memory."""
        for doc in documents:
            self._documents[doc.id] = doc.model_copy(deep=True)

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Rank documents by token overlap with deterministic sorting."""
        query_tokens = self._tokenize(query.query)
        if not query_tokens or not self._documents:
            return RetrievalResult(query=query.query, matches=[], total_found=0)

        scored_matches: List[ScoredDocument] = []

        for doc in self._documents.values():
            # Apply metadata filters if specified
            if query.filters:
                match_filters = all(doc.metadata.get(k) == v for k, v in query.filters.items())
                if not match_filters:
                    continue

            doc_tokens = self._tokenize(doc.content)
            overlap = len(query_tokens.intersection(doc_tokens))
            if overlap > 0:
                score = min(1.0, overlap / len(query_tokens))
                scored_matches.append(ScoredDocument(document=doc, score=score))

        # Deterministic sorting: highest score first, tie-break alphabetically by document ID
        scored_matches.sort(key=lambda x: (-x.score, x.document.id))
        top_matches = scored_matches[: query.top_k]

        return RetrievalResult(
            query=query.query,
            matches=top_matches,
            total_found=len(scored_matches),
        )

    async def count(self) -> int:
        """Return total stored documents."""
        return len(self._documents)

    async def clear(self) -> None:
        """Clear document store."""
        self._documents.clear()


class RetrievalAgent(BaseAgent):
    """Agent that resolves knowledge and context queries against a document store."""

    def __init__(self, document_store: BaseDocumentStore) -> None:
        metadata = AgentMetadata(
            name="retrieval_agent",
            role=AgentRole.RETRIEVER,
            description="Fetches, filters, and ranks contextual documents from document stores",
            capabilities=["document_retrieval", "context_search", "metadata_filtering"],
            input_schema_desc="Task input containing 'query' string, or state.request",
            output_schema_desc="RetrievalResult with ranked ScoredDocuments",
        )
        super().__init__(metadata=metadata)
        self.document_store = document_store

    def validate_input(self, state: OrchestrationState, task: Optional[Task] = None) -> bool:
        """Check if query is present in task input or state request."""
        query = (task.input_data.get("query") if task and task.input_data else None) or state.request
        return bool(query and query.strip())

    async def execute(self, state: OrchestrationState, task: Optional[Task] = None) -> AgentResult:
        """Execute document retrieval query and return structured context."""
        start_time = time.perf_counter()

        if not self.validate_input(state, task):
            return AgentResult.failure(
                agent_name=self.name,
                error="No valid query string provided for retrieval.",
                retryable=False,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        query_str = (task.input_data.get("query") if task and task.input_data else None) or state.request
        top_k = task.input_data.get("top_k", 3) if task and task.input_data else 3
        filters = task.input_data.get("filters") if task and task.input_data else None

        query = RetrievalQuery(query=query_str, top_k=top_k, filters=filters)

        try:
            result = await self.document_store.retrieve(query)
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Convert result to serializable dict
            serialized_matches = [
                {
                    "id": m.document.id,
                    "content": m.document.content,
                    "metadata": m.document.metadata,
                    "score": m.score,
                }
                for m in result.matches
            ]

            # Optionally append to state retrieved_context for downstream audit
            for match_dict in serialized_matches:
                state.add_retrieved_context(match_dict)

            return AgentResult.success(
                agent_name=self.name,
                output={
                    "query": query_str,
                    "matches": serialized_matches,
                    "total_found": result.total_found,
                },
                execution_time_ms=latency_ms,
                metadata={"total_found": result.total_found, "top_k": top_k},
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return AgentResult.failure(
                agent_name=self.name,
                error=f"Retrieval error occurred: {e}",
                retryable=True,
                execution_time_ms=latency_ms,
            )
