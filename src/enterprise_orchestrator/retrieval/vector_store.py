"""Vector store interface and local FAISS implementation."""

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from enterprise_orchestrator.agents.retrieval import Document, ScoredDocument
from enterprise_orchestrator.errors.exceptions import ConfigurationError, RetrievalError
from enterprise_orchestrator.retrieval.chunking import DocumentChunk


class BaseVectorStore(ABC):
    """Abstract interface for dense vector indices."""

    @abstractmethod
    async def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        """Index a batch of document chunks with corresponding embedding vectors."""
        pass

    @abstractmethod
    async def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[ScoredDocument]:
        """Perform nearest-neighbor search returning top_k scored documents."""
        pass

    @abstractmethod
    async def save(self, directory: str) -> None:
        """Persist index and chunk metadata to disk."""
        pass

    @abstractmethod
    async def load(self, directory: str) -> None:
        """Load index and chunk metadata from disk."""
        pass

    @abstractmethod
    async def count(self) -> int:
        """Return total indexed chunks."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all indexed vectors and metadata."""
        pass


class FAISSVectorStore(BaseVectorStore):
    """Local FAISS vector store with cosine similarity and disk persistence."""

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0")
        self.dimension = dimension
        self._faiss = self._load_faiss()
        # Inner-product index on normalized vectors equals cosine similarity
        self._index = self._faiss.IndexFlatIP(dimension)
        self._chunks: List[DocumentChunk] = []

    def _load_faiss(self) -> Any:
        """Lazy load FAISS module."""
        try:
            import faiss
            return faiss
        except ImportError as e:
            raise ConfigurationError(
                message="faiss-cpu is not installed. Please run 'pip install faiss-cpu'.",
                code="DEPENDENCY_MISSING",
                retryable=False,
            ) from e

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalize float32 vectors for cosine similarity computation."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (vectors / norms).astype(np.float32)

    async def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        """Add chunks and dense embeddings into FAISS index."""
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        arr = np.array(embeddings, dtype=np.float32)
        if arr.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {arr.shape[1]} does not match index dimension {self.dimension}")

        norm_arr = self._normalize(arr)
        self._index.add(norm_arr)
        self._chunks.extend(chunks)

    async def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[ScoredDocument]:
        """Search top_k nearest chunks matching optional metadata filters."""
        if self._index.ntotal == 0 or not self._chunks or top_k <= 0:
            return []

        q_arr = np.array([query_embedding], dtype=np.float32)
        if q_arr.shape[1] != self.dimension:
            raise ValueError(f"Query embedding dimension {q_arr.shape[1]} does not match index dimension {self.dimension}")

        norm_q = self._normalize(q_arr)

        # Retrieve extra candidates if filtering is applied
        search_k = min(self._index.ntotal, max(top_k * 5, 20) if filters else top_k)
        scores, indices = self._index.search(norm_q, search_k)

        results: List[ScoredDocument] = []
        for raw_score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue

            chunk = self._chunks[idx]

            # Evaluate metadata filters
            if filters:
                match = all(chunk.metadata.get(k) == v for k, v in filters.items())
                if not match:
                    continue

            # Normalize cosine score from [-1.0, 1.0] to [0.0, 1.0]
            norm_score = max(0.0, min(1.0, float((raw_score + 1.0) / 2.0)))

            doc_repr = Document(
                id=chunk.id,
                content=chunk.content,
                metadata={
                    **chunk.metadata,
                    "parent_doc_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                },
            )
            results.append(ScoredDocument(document=doc_repr, score=norm_score))

            if len(results) >= top_k:
                break

        return results

    async def save(self, directory: str) -> None:
        """Persist FAISS index and chunk metadata to directory."""
        os.makedirs(directory, exist_ok=True)
        index_file = os.path.join(directory, "index.faiss")
        metadata_file = os.path.join(directory, "metadata.json")

        self._faiss.write_index(self._index, index_file)

        metadata_payload = {
            "dimension": self.dimension,
            "count": len(self._chunks),
            "chunks": [c.model_dump() for c in self._chunks],
        }
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=2)

    async def load(self, directory: str) -> None:
        """Load FAISS index and chunk metadata from directory."""
        index_file = os.path.join(directory, "index.faiss")
        metadata_file = os.path.join(directory, "metadata.json")

        if not os.path.exists(index_file) or not os.path.exists(metadata_file):
            raise RetrievalError(
                message=f"Missing index or metadata files in directory: {directory}",
                code="VECTOR_STORE_LOAD_FAILED",
                retryable=False,
            )

        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            loaded_dimension = metadata.get("dimension")
            if loaded_dimension != self.dimension:
                raise RetrievalError(
                    message=f"Loaded index dimension {loaded_dimension} does not match configured dimension {self.dimension}",
                    code="VECTOR_STORE_DIMENSION_MISMATCH",
                    retryable=False,
                )

            loaded_index = self._faiss.read_index(index_file)
            loaded_chunks = [DocumentChunk.model_validate(c) for c in metadata.get("chunks", [])]

            if loaded_index.ntotal != len(loaded_chunks):
                raise RetrievalError(
                    message=f"Corrupted vector store: index size ({loaded_index.ntotal}) does not match metadata count ({len(loaded_chunks)})",
                    code="VECTOR_STORE_CORRUPTED",
                    retryable=False,
                )

            self._index = loaded_index
            self._chunks = loaded_chunks

        except Exception as e:
            if isinstance(e, RetrievalError):
                raise e
            raise RetrievalError(
                message=f"Failed to load vector store from {directory}: {e}",
                code="VECTOR_STORE_LOAD_ERROR",
                retryable=False,
            ) from e

    async def count(self) -> int:
        return len(self._chunks)

    async def clear(self) -> None:
        self._index = self._faiss.IndexFlatIP(self.dimension)
        self._chunks.clear()
