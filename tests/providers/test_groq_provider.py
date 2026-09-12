"""Unit tests for GroqProvider with dependency injection and zero network calls."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, Field

from enterprise_orchestrator.core.types import MessageRole
from enterprise_orchestrator.errors.exceptions import (
    ConfigurationError,
    ProviderError,
    ProviderUnavailableError,
)
from enterprise_orchestrator.providers.groq import GroqProvider
from enterprise_orchestrator.providers.models import LLMRequest, Message


class TaskEvaluation(BaseModel):
    decision: str = Field(..., description="Decision outcome")
    score: int = Field(..., ge=0, le=100)


def test_groq_initialization_missing_key():
    """Verify ConfigurationError if Groq API key is absent."""
    provider = GroqProvider(api_key="")
    provider._api_key = None
    with pytest.raises(ConfigurationError) as exc_info:
        provider._get_client()
    assert "GROQ_API_KEY" in str(exc_info.value)
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_groq_text_generation_success():
    """Verify text generation and TokenUsage mapping via mock AsyncGroq client."""
    mock_choice = MagicMock()
    mock_choice.message.content = "Groq fast inference output."

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 25
    mock_usage.completion_tokens = 15
    mock_usage.total_tokens = 40

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    provider = GroqProvider(api_key="fake-groq-key", client=mock_client)
    req = LLMRequest(
        messages=[Message(role=MessageRole.USER, content="Explain quantum computing in 1 sentence")],
        model="llama-3.3-70b-versatile",
    )

    resp = await provider.generate(req)

    assert resp.provider == "groq"
    assert resp.model == "llama-3.3-70b-versatile"
    assert resp.content == "Groq fast inference output."
    assert resp.usage.prompt_tokens == 25
    assert resp.usage.completion_tokens == 15
    assert resp.usage.total_tokens == 40


@pytest.mark.asyncio
async def test_groq_structured_output_success():
    """Verify structured output parsing with GroqProvider."""
    mock_choice = MagicMock()
    mock_choice.message.content = '{"decision": "APPROVED", "score": 92}'

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    provider = GroqProvider(api_key="fake-groq-key", client=mock_client)
    req = LLMRequest(messages=[Message(role=MessageRole.USER, content="Evaluate task")])

    parsed, response = await provider.generate_structured(req, TaskEvaluation)

    assert isinstance(parsed, TaskEvaluation)
    assert parsed.decision == "APPROVED"
    assert parsed.score == 92


@pytest.mark.asyncio
async def test_groq_exception_mapping():
    """Verify rate-limit (429) and auth (401) exception mapping."""
    mock_client_429 = MagicMock()
    mock_client_429.chat.completions.create = AsyncMock(
        side_effect=Exception("Error code: 429 - Rate limit reached for model")
    )
    provider_429 = GroqProvider(api_key="fake-key", client=mock_client_429)
    req = LLMRequest(messages=[Message(role=MessageRole.USER, content="Test")])

    with pytest.raises(ProviderUnavailableError) as exc_429:
        await provider_429.generate(req)
    assert exc_429.value.status_code == 429
    assert exc_429.value.retryable is True

    mock_client_401 = MagicMock()
    mock_client_401.chat.completions.create = AsyncMock(
        side_effect=Exception("Error code: 401 - Invalid API Key provided")
    )
    provider_401 = GroqProvider(api_key="fake-key", client=mock_client_401)

    with pytest.raises(ProviderError) as exc_401:
        await provider_401.generate(req)
    assert exc_401.value.status_code == 401
    assert exc_401.value.retryable is False
