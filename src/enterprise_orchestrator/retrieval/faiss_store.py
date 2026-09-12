"""FAISSDocumentStore adapter implementing BaseDocumentStore for semantic RAG."""

from typing import List, Optional

from enterprise_orchestrator.agents.retrieval import (
    BaseDocumentStore,
    Document,
    RetrievalQuery,
    RetrievalResult,
)
from enterprise_orchestrator.retrieval.chunking import BaseDocumentChunker, SimpleTextChunker
from enterprise_orchestrator.retrieval.embeddings import (
    BaseEmbeddingProvider,
    FakeEmbeddingProvider,
)
from enterprise_orchestrator.retrieval.vector_store import BaseVectorStore, FAISSVectorStore


class FAISSDocumentStore(BaseDocumentStore):
    """Semantic document store adapter unifying chunking, embeddings, and FAISS indexing."""

    def __init__(
        self,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        chunker: Optional[BaseDocumentChunker] = None,
        vector_store: Optional[BaseVectorStore] = None,
    ) -> None:
        self.embedding_provider = embedding_provider or FakeEmbeddingProvider()
        self.chunker = chunker or SimpleTextChunker()
        self.vector_store = vector_store or FAISSVectorStore(
            dimension=self.embedding_provider.dimension
        )

    async def add_documents(self, documents: List[Document]) -> None:
        """Ingest documents by chunking, generating embeddings, and storing in FAISS."""
        if not documents:
            return

        chunks = await self.chunker.chunk_documents(documents)
        if not chunks:
            return

        texts = [c.content for c in chunks]
        embeddings = await self.embedding_provider.embed_documents(texts)
        await self.vector_store.add_chunks(chunks=chunks, embeddings=embeddings)

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Embed search query and retrieve top_k scored semantic matches."""
        if not query.query or not query.query.strip():
            return RetrievalResult(query=query.query, matches=[], total_found=0)

        query_embedding = await self.embedding_provider.embed_text(query.query)
        matches = await self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=query.top_k,
            filters=query.filters,
        )

        return RetrievalResult(
            query=query.query,
            matches=matches,
            total_found=len(matches),
        )

    async def count(self) -> int:
        """Return total indexed chunks."""
        return await self.vector_store.count()

    async def clear(self) -> None:
        """Purge all indexed vectors and chunks."""
        await self.vector_store.clear()

    async def save(self, directory: str) -> None:
        """Persist vector index and metadata."""
        await self.vector_store.save(directory)

    async def load(self, directory: str) -> None:
        """Load vector index and metadata."""
        await self.vector_store.load(directory)
