"""Unit tests for GeminiProvider with dependency injection and zero network calls."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, Field

from enterprise_orchestrator.core.types import MessageRole
from enterprise_orchestrator.errors.exceptions import (
    ConfigurationError,
    ProviderError,
    ProviderUnavailableError,
    ValidationError,
)
from enterprise_orchestrator.providers.gemini import GeminiProvider
from enterprise_orchestrator.providers.models import LLMRequest, Message


class SummarySchema(BaseModel):
    summary: str = Field(..., description="Summary content")
    confidence: float = Field(..., ge=0.0, le=1.0)


def test_gemini_initialization_missing_key():
    """Verify ConfigurationError if API key is absent."""
    provider = GeminiProvider(api_key="")
    provider._api_key = None
    with pytest.raises(ConfigurationError) as exc_info:
        provider._get_client()
    assert "GEMINI_API_KEY" in str(exc_info.value)
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_gemini_text_generation_success():
    """Verify text generation and TokenUsage mapping via injected mock client."""
    mock_response = MagicMock()
    mock_response.text = "Analysis completed successfully."
    mock_response.usage_metadata = MagicMock(
        prompt_token_count=10,
        candidates_token_count=20,
        total_token_count=30,
    )

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    provider = GeminiProvider(api_key="fake-test-key", client=mock_client)
    req = LLMRequest(
        messages=[
            Message(role=MessageRole.SYSTEM, content="You are a helper."),
            Message(role=MessageRole.USER, content="Hello"),
        ],
        model="gemini-2.5-flash",
    )

    response = await provider.generate(req)

    assert response.provider == "gemini"
    assert response.model == "gemini-2.5-flash"
    assert response.content == "Analysis completed successfully."
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 20
    assert response.usage.total_tokens == 30
    assert response.latency_ms > 0


@pytest.mark.asyncio
async def test_gemini_structured_output_success():
    """Verify structured output parsing with Pydantic schema."""
    mock_response = MagicMock()
    mock_response.text = '{"summary": "Market expanded by 5%", "confidence": 0.95}'
    mock_response.usage_metadata = MagicMock(prompt_token_count=15, candidates_token_count=15, total_token_count=30)

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    provider = GeminiProvider(api_key="fake-test-key", client=mock_client)
    req = LLMRequest(messages=[Message(role=MessageRole.USER, content="Extract summary")])

    parsed, response = await provider.generate_structured(req, SummarySchema)

    assert isinstance(parsed, SummarySchema)
    assert parsed.summary == "Market expanded by 5%"
    assert parsed.confidence == 0.95
    assert response.structured_data == {"summary": "Market expanded by 5%", "confidence": 0.95}


@pytest.mark.asyncio
async def test_gemini_structured_output_corrective_retry():
    """Verify bounded retry when initial output is malformed JSON."""
    invalid_response = MagicMock()
    invalid_response.text = "This is not JSON at all."
    invalid_response.usage_metadata = None

    valid_response = MagicMock()
    valid_response.text = '{"summary": "Fixed summary", "confidence": 0.99}'
    valid_response.usage_metadata = None

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=[invalid_response, valid_response])

    provider = GeminiProvider(api_key="fake-test-key", client=mock_client)
    req = LLMRequest(messages=[Message(role=MessageRole.USER, content="Extract summary")])

    parsed, response = await provider.generate_structured(req, SummarySchema)
    assert parsed.summary == "Fixed summary"
    assert mock_client.aio.models.generate_content.call_count == 2


@pytest.mark.asyncio
async def test_gemini_exception_mapping():
    """Verify mapping of 429 and 401 errors into framework taxonomy."""
    # Rate limit test (429)
    mock_client_429 = MagicMock()
    mock_client_429.aio.models.generate_content = AsyncMock(
        side_effect=Exception("429 Resource exhausted: quota exceeded")
    )

    provider_429 = GeminiProvider(api_key="fake-key", client=mock_client_429)
    req = LLMRequest(messages=[Message(role=MessageRole.USER, content="Hi")])

    with pytest.raises(ProviderUnavailableError) as exc_429:
        await provider_429.generate(req)
    assert exc_429.value.status_code == 429
    assert exc_429.value.retryable is True

    # Auth failure test (401)
    mock_client_401 = MagicMock()
    mock_client_401.aio.models.generate_content = AsyncMock(
        side_effect=Exception("401 API_KEY_INVALID: User not authenticated")
    )
    provider_401 = GeminiProvider(api_key="fake-key", client=mock_client_401)

    with pytest.raises(ProviderError) as exc_401:
        await provider_401.generate(req)
    assert exc_401.value.status_code == 401
    assert exc_401.value.retryable is False
