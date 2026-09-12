"""Groq LLM provider implementation for high-speed fallback and primary inference."""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

from enterprise_orchestrator.config.settings import FrameworkSettings
from enterprise_orchestrator.core.types import MessageRole
from enterprise_orchestrator.errors.exceptions import (
    ConfigurationError,
    ProviderError,
    ProviderUnavailableError,
    TimeoutError as OrchestratorTimeoutError,
    ValidationError,
    sanitize_message,
)
from enterprise_orchestrator.providers.base import BaseLLMProvider
from enterprise_orchestrator.providers.models import (
    LLMRequest,
    LLMResponse,
    Message,
    TokenUsage,
)

T = TypeVar("T", bound=BaseModel)


class GroqProvider(BaseLLMProvider):
    """Groq Cloud LLM Provider integration."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        client: Optional[Any] = None,
    ) -> None:
        settings = FrameworkSettings()
        self._api_key = api_key or settings.groq_api_key
        model = default_model or settings.groq_model or "llama-3.3-70b-versatile"
        super().__init__(provider_name="groq", default_model=model)
        self.timeout_seconds = timeout_seconds or settings.llm_timeout_seconds
        self.max_retries = settings.llm_max_retries
        self._client = client

    def _get_client(self) -> Any:
        """Lazy-initialize or return the Groq client."""
        if self._client is not None:
            return self._client

        if not self._api_key:
            raise ConfigurationError(
                message="GROQ_API_KEY is not configured. Set it in environment or .env file.",
                code="MISSING_GROQ_API_KEY",
                retryable=False,
            )

        try:
            from groq import AsyncGroq, Groq
            self._client = AsyncGroq(api_key=self._api_key)
            return self._client
        except ImportError:
            try:
                import groq
                self._client = groq.Groq(api_key=self._api_key)
                return self._client
            except ImportError as e:
                raise ConfigurationError(
                    message="Groq SDK is not installed. Please install 'groq'.",
                    code="MISSING_GROQ_SDK",
                    retryable=False,
                ) from e

    def _format_messages_for_groq(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Convert standard framework messages to Groq/OpenAI compatible format."""
        formatted = []
        for msg in messages:
            role_str = "user"
            if msg.role == MessageRole.SYSTEM:
                role_str = "system"
            elif msg.role == MessageRole.ASSISTANT:
                role_str = "assistant"
            elif msg.role == MessageRole.TOOL:
                role_str = "tool"

            entry = {"role": role_str, "content": msg.content}
            if msg.name:
                entry["name"] = msg.name
            formatted.append(entry)
        return formatted

    def _extract_usage(self, response: Any) -> TokenUsage:
        """Extract TokenUsage from Groq chat completion response."""
        usage = TokenUsage()
        try:
            raw_usage = getattr(response, "usage", None)
            if raw_usage:
                usage.prompt_tokens = getattr(raw_usage, "prompt_tokens", 0) or 0
                usage.completion_tokens = getattr(raw_usage, "completion_tokens", 0) or 0
                usage.total_tokens = getattr(raw_usage, "total_tokens", 0) or (
                    usage.prompt_tokens + usage.completion_tokens
                )
        except Exception:
            pass
        return usage

    def _map_exception(self, exc: Exception) -> Exception:
        """Map Groq SDK exceptions to framework error taxonomy."""
        msg = str(exc)
        sanitized_msg = sanitize_message(msg, sensitive_patterns=[self._api_key] if self._api_key else None)
        lower_msg = sanitized_msg.lower()

        # Check for rate limit
        if "429" in lower_msg or "rate_limit" in lower_msg or "rate limit" in lower_msg:
            return ProviderUnavailableError(
                message=f"Groq API rate limit exceeded: {sanitized_msg}",
                provider_name=self.provider_name,
                status_code=429,
                details={"raw_error": sanitized_msg},
            )

        # Check for service unavailability
        if "503" in lower_msg or "service_unavailable" in lower_msg or "overloaded" in lower_msg:
            return ProviderUnavailableError(
                message=f"Groq service unavailable: {sanitized_msg}",
                provider_name=self.provider_name,
                status_code=503,
                details={"raw_error": sanitized_msg},
            )

        # Check for auth errors
        if "401" in lower_msg or "unauthorized" in lower_msg or "invalid api key" in lower_msg:
            return ProviderError(
                message=f"Groq authentication error: {sanitized_msg}",
                provider_name=self.provider_name,
                code="GROQ_AUTH_ERROR",
                status_code=401,
                retryable=False,
                details={"raw_error": sanitized_msg},
            )

        retryable = any(term in lower_msg for term in ["connection", "timeout", "reset", "socket", "network"])
        return ProviderError(
            message=f"Groq invocation error: {sanitized_msg}",
            provider_name=self.provider_name,
            code="GROQ_GENERATE_ERROR",
            retryable=retryable,
            details={"raw_error": sanitized_msg},
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Groq."""
        client = self._get_client()
        model_name = request.model or self.default_model
        timeout = request.timeout_seconds or self.timeout_seconds

        messages = self._format_messages_for_groq(request.messages)
        start_time = time.perf_counter()

        try:
            kwargs: Dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "temperature": request.temperature,
            }
            if request.max_tokens:
                kwargs["max_tokens"] = request.max_tokens
            if request.stop_sequences:
                kwargs["stop"] = request.stop_sequences

            # If client has chat.completions.create (AsyncGroq / Groq)
            if hasattr(client, "chat") and hasattr(client.chat, "completions"):
                create_fn = client.chat.completions.create
                # Check if create is async
                if asyncio.iscoroutinefunction(create_fn):
                    response = await asyncio.wait_for(create_fn(**kwargs), timeout=timeout)
                else:
                    loop = asyncio.get_event_loop()
                    response = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: create_fn(**kwargs)),
                        timeout=timeout,
                    )

                text = ""
                if hasattr(response, "choices") and response.choices:
                    choice = response.choices[0]
                    text = getattr(getattr(choice, "message", None), "content", "") or ""

                usage = self._extract_usage(response)

            elif callable(client):
                # Callable mock harness
                res = client(request)
                if asyncio.iscoroutine(res):
                    res = await res
                return res
            else:
                raise ProviderError(
                    message="Unsupported Groq client interface.",
                    provider_name=self.provider_name,
                    retryable=False,
                )

            latency_ms = (time.perf_counter() - start_time) * 1000
            return LLMResponse(
                content=text,
                model=model_name,
                provider=self.provider_name,
                usage=usage,
                latency_ms=latency_ms,
            )

        except asyncio.TimeoutError as e:
            raise OrchestratorTimeoutError(
                message=f"Groq generation timed out after {timeout} seconds.",
                timeout_seconds=timeout,
                details={"provider": self.provider_name, "model": model_name},
            ) from e
        except Exception as e:
            if isinstance(e, (ConfigurationError, OrchestratorTimeoutError, ProviderError)):
                raise
            raise self._map_exception(e) from e

    async def generate_structured(self, request: LLMRequest, schema: Type[T]) -> tuple[T, LLMResponse]:
        """Generate structured output with schema enforcement, Pydantic validation, and corrective retry."""
        schema_json = json.dumps(schema.model_json_schema())
        directive = (
            f"\n\nYou MUST respond ONLY with a valid JSON object matching the following JSON Schema:\n"
            f"{schema_json}\n"
            f"Do not include markdown fences or any conversational filler. Return raw JSON only."
        )

        messages = [msg.model_copy() for msg in request.messages]
        messages[-1].content = messages[-1].content + directive

        current_request = request.model_copy(update={"messages": messages, "structured_schema": schema.model_json_schema()})
        last_error_msg = ""

        for attempt in range(self.max_retries + 1):
            response = await self.generate(current_request)
            raw_text = response.content.strip()

            # Clean markdown formatting if present
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()

            try:
                parsed_json = json.loads(raw_text)
                validated_model = schema.model_validate(parsed_json)
                response.structured_data = parsed_json
                return validated_model, response
            except Exception as e:
                last_error_msg = str(e)
                if attempt < self.max_retries:
                    corrective_msg = Message(
                        role=MessageRole.USER,
                        content=(
                            f"Your response failed validation with error: {last_error_msg}\n"
                            f"Previous response was:\n{raw_text}\n\n"
                            f"Please provide valid JSON matching the schema:\n{schema_json}"
                        ),
                    )
                    current_request.messages.append(Message(role=MessageRole.ASSISTANT, content=raw_text))
                    current_request.messages.append(corrective_msg)
                else:
                    break

        raise ValidationError(
            message=f"Failed to generate valid structured output for schema '{schema.__name__}' after {self.max_retries + 1} attempts. Last error: {last_error_msg}",
            details={"schema": schema.__name__, "last_error": last_error_msg},
        )

    async def health_check(self) -> bool:
        """Lightweight health check verifying configuration presence."""
        return bool(self._api_key and len(self._api_key) > 5)
