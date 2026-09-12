"""LLM request and response models for provider-agnostic invocation."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from enterprise_orchestrator.core.types import MessageRole


class Message(BaseModel):
    """Standardized conversational message representation."""

    role: MessageRole = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Text content of the message")
    name: Optional[str] = Field(default=None, description="Optional name of author / tool")
    tool_call_id: Optional[str] = Field(default=None, description="Identifier for tool call resolution")


class TokenUsage(BaseModel):
    """Token consumption metrics for cost accounting and quota tracking."""

    prompt_tokens: int = Field(default=0, ge=0, description="Tokens used in prompt")
    completion_tokens: int = Field(default=0, ge=0, description="Tokens generated in completion")
    total_tokens: int = Field(default=0, ge=0, description="Cumulative tokens consumed")


class LLMRequest(BaseModel):
    """Provider-agnostic prompt generation request envelope."""

    messages: List[Message] = Field(..., min_length=1, description="Ordered conversation history or prompts")
    model: Optional[str] = Field(default=None, description="Model identifier override (uses provider default if None)")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum tokens to generate")
    structured_schema: Optional[Dict[str, Any]] = Field(default=None, description="JSON schema for enforced structured output")
    stop_sequences: List[str] = Field(default_factory=list, description="Tokens that trigger generation termination")
    timeout_seconds: float = Field(default=60.0, gt=0.0, description="Timeout deadline for provider response")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary tracking metadata")


class LLMResponse(BaseModel):
    """Standardized response envelope returned by all LLM providers."""

    content: str = Field(default="", description="Generated textual content")
    structured_data: Optional[Dict[str, Any]] = Field(default=None, description="Parsed structured data if schema was requested")
    model: str = Field(..., description="Exact model name used for generation")
    provider: str = Field(..., description="Programmatic identifier of provider (e.g. gemini, groq, fake)")
    usage: TokenUsage = Field(default_factory=TokenUsage, description="Token consumption metadata")
    finish_reason: Optional[str] = Field(default=None, description="Reason generation terminated (e.g. stop, length)")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Roundtrip invocation duration in milliseconds")
    raw_response: Optional[Dict[str, Any]] = Field(default=None, description="Provider-specific raw response envelope for debugging")
