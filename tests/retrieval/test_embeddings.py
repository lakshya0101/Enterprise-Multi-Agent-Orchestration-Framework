"""Unit tests for embedding providers."""

import math
import pytest

from enterprise_orchestrator.errors.exceptions import ConfigurationError
from enterprise_orchestrator.retrieval.embeddings import (
    BaseEmbeddingProvider,
    FakeEmbeddingProvider,
    LocalSentenceTransformerEmbeddingProvider,
)


def test_fake_embedding_provider_determinism_and_dimensions():
    """Verify FakeEmbeddingProvider produces deterministic, normalized vectors."""
    provider = FakeEmbeddingProvider(dimension=32)
    assert provider.dimension == 32
    assert isinstance(provider, BaseEmbeddingProvider)


@pytest.mark.asyncio
async def test_fake_embedding_provider_embed_text_and_docs():
    """Verify embed_text and embed_documents produce unit-length vectors."""
    provider = FakeEmbeddingProvider(dimension=48)
    
    vec1 = await provider.embed_text("Financial revenue growth in Q3")
    vec2 = await provider.embed_text("Financial revenue growth in Q3")
    vec3 = await provider.embed_text("Completely unrelated query about biology")

    # Determinism
    assert vec1 == vec2
    assert vec1 != vec3
    assert len(vec1) == 48

    # L2 norm check (close to 1.0)
    norm = math.sqrt(sum(x * x for x in vec1))
    assert pytest.approx(norm, rel=1e-3) == 1.0

    # Batch embedding
    batch_vecs = await provider.embed_documents(["Text A", "Text B"])
    assert len(batch_vecs) == 2
    assert len(batch_vecs[0]) == 48
    assert len(batch_vecs[1]) == 48


def test_fake_embedding_provider_invalid_dimension():
    """Verify invalid dimension raises ValueError."""
    with pytest.raises(ValueError):
        FakeEmbeddingProvider(dimension=0)


def test_local_sentence_transformer_lazy_loading():
    """Verify LocalSentenceTransformerEmbeddingProvider does not load model on initialization."""
    provider = LocalSentenceTransformerEmbeddingProvider(model_name="test-model")
    assert provider._model is None
    assert provider.model_name == "test-model"


@pytest.mark.asyncio
async def test_local_sentence_transformer_error_handling(monkeypatch):
    """Verify clear ConfigurationError is raised if sentence_transformers is not available."""
    provider = LocalSentenceTransformerEmbeddingProvider(model_name="nonexistent-dummy-model")
    
    # Mocking the loader to simulate model load failure without downloading
    def mock_load(self):
        raise ConfigurationError("Failed to load local model", code="MODEL_LOAD_FAILED")
        
    monkeypatch.setattr(LocalSentenceTransformerEmbeddingProvider, "_load_model", mock_load)

    with pytest.raises(ConfigurationError) as exc_info:
        await provider.embed_text("Test query")
    assert "Failed to load" in exc_info.value.message
