"""Embedding provider abstractions and local/fake implementations for semantic retrieval."""

import hashlib
import math
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from enterprise_orchestrator.errors.exceptions import ConfigurationError


class BaseEmbeddingProvider(ABC):
    """Abstract interface for text embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""
        pass

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Compute embedding vector for a single text query."""
        pass

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute embedding vectors for a batch of documents."""
        pass


class FakeEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic hash-based embedding provider for offline testing and verification.

    NOTE: This is strictly a testing utility. It generates deterministic pseudo-vectors
    and does NOT capture true semantic meaning.
    """

    def __init__(self, dimension: int = 64) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _hash_vector(self, text: str) -> List[float]:
        """Generate a deterministic unit-normalized pseudo-vector for a string."""
        raw_values: List[float] = []
        for i in range(self._dimension):
            seed_str = f"{text}_{i}"
            h = int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest(), 16)
            # Map hash to [-1.0, 1.0]
            val = ((h % 10000) / 5000.0) - 1.0
            raw_values.append(val)

        # L2-normalize vector
        norm = math.sqrt(sum(x * x for x in raw_values))
        if norm > 0:
            return [x / norm for x in raw_values]
        return [0.0] * self._dimension

    async def embed_text(self, text: str) -> List[float]:
        return self._hash_vector(text)

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_vector(t) for t in texts]


class LocalSentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    """Local SentenceTransformer embedding provider with lazy-loading and offline support."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Optional[Any] = None
        self._dimension: Optional[int] = None

    def _load_model(self) -> Any:
        """Lazy load SentenceTransformer model on demand."""
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ConfigurationError(
                message="sentence-transformers is not installed. Please run 'pip install sentence-transformers'.",
                code="DEPENDENCY_MISSING",
                retryable=False,
            ) from e

        try:
            kwargs = {}
            if self.device:
                kwargs["device"] = self.device
            self._model = SentenceTransformer(self.model_name, **kwargs)
            self._dimension = self._model.get_sentence_embedding_dimension()
            return self._model
        except Exception as e:
            raise ConfigurationError(
                message=f"Failed to load local embedding model '{self.model_name}': {e}",
                code="MODEL_LOAD_FAILED",
                retryable=False,
            ) from e

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            model = self._load_model()
            self._dimension = model.get_sentence_embedding_dimension()
        return self._dimension

    async def embed_text(self, text: str) -> List[float]:
        """Generate normalized embedding vector for query text."""
        model = self._load_model()
        vector = model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate normalized embedding vectors for batch of document chunks."""
        if not texts:
            return []
        model = self._load_model()
        vectors = model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()
