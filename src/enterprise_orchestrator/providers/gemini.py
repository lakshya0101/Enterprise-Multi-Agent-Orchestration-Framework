"""Google Gemini LLM provider implementation."""

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


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM Provider integration."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        client: Optional[Any] = None,
    ) -> None:
        settings = FrameworkSettings()
        self._api_key = api_key or settings.gemini_api_key
        model = default_model or settings.gemini_model or "gemini-2.5-flash"
        super().__init__(provider_name="gemini", default_model=model)
        self.timeout_seconds = timeout_seconds or settings.llm_timeout_seconds
        self.max_retries = settings.llm_max_retries
        self._client = client

    def _get_client(self) -> Any:
        """Lazy-initialize or return the Gemini client."""
        if self._client is not None:
            return self._client

        if not self._api_key:
            raise ConfigurationError(
                message="GEMINI_API_KEY is not configured. Set it in environment or .env file.",
                code="MISSING_GEMINI_API_KEY",
                retryable=False,
            )

        try:
            # Attempt to use google-genai official modern SDK
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
            return self._client
        except ImportError:
            try:
                # Fallback to google-generativeai if installed
                import google.generativeai as genai
                genai.configure(api_key=self._api_key)
                self._client = genai
                return self._client
            except ImportError as e:
                raise ConfigurationError(
                    message="Google GenAI SDK is not installed. Please install 'google-genai' or 'google-generativeai'.",
                    code="MISSING_GEMINI_SDK",
                    retryable=False,
                ) from e

    def _format_messages_for_gemini(self, messages: List[Message]) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """Convert standard framework messages to Gemini contents and system prompt."""
        system_instruction = None
        contents = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                if system_instruction is None:
                    system_instruction = msg.content
                else:
                    system_instruction += f"\n{msg.content}"
            elif msg.role in (MessageRole.USER, MessageRole.TOOL):
                contents.append({"role": "user", "parts": [{"text": msg.content}]})
            elif msg.role == MessageRole.ASSISTANT:
                contents.append({"role": "model", "parts": [{"text": msg.content}]})

        return system_instruction, contents

    def _extract_usage(self, response: Any) -> TokenUsage:
        """Extract standardized TokenUsage from Gemini response metadata."""
        usage = TokenUsage()
        try:
            usage_meta = getattr(response, "usage_metadata", None)
            if usage_meta:
                usage.prompt_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
                usage.completion_tokens = getattr(usage_meta, "candidates_token_count", 0) or 0
                usage.total_tokens = getattr(usage_meta, "total_token_count", 0) or (
                    usage.prompt_tokens + usage.completion_tokens
                )
        except Exception:
            pass
        return usage

    def _map_exception(self, exc: Exception) -> Exception:
        """Map provider-specific SDK exceptions to framework error taxonomy."""
        msg = str(exc)
        sanitized_msg = sanitize_message(msg, sensitive_patterns=[self._api_key] if self._api_key else None)
        lower_msg = sanitized_msg.lower()

        # Check for rate limits or quotas
        if "429" in lower_msg or "resource_exhausted" in lower_msg or "quota" in lower_msg or "rate limit" in lower_msg:
            return ProviderUnavailableError(
                message=f"Gemini API rate limit exceeded: {sanitized_msg}",
                provider_name=self.provider_name,
                status_code=429,
                details={"raw_error": sanitized_msg},
            )

        # Check for service unavailability
        if "503" in lower_msg or "unavailable" in lower_msg or "overloaded" in lower_msg or "service unavailable" in lower_msg:
            return ProviderUnavailableError(
                message=f"Gemini service unavailable: {sanitized_msg}",
                provider_name=self.provider_name,
                status_code=503,
                details={"raw_error": sanitized_msg},
            )

        # Check for authentication / key issues
        if "401" in lower_msg or "403" in lower_msg or "api_key" in lower_msg or "unauthenticated" in lower_msg or "permission_denied" in lower_msg:
            return ProviderError(
                message=f"Gemini authentication / permission error: {sanitized_msg}",
                provider_name=self.provider_name,
                code="GEMINI_AUTH_ERROR",
                status_code=401,
                retryable=False,
                details={"raw_error": sanitized_msg},
            )

        # Default provider error (retryable if network/connection related)
        retryable = any(term in lower_msg for term in ["connection", "timeout", "reset", "socket", "network"])
        return ProviderError(
            message=f"Gemini invocation error: {sanitized_msg}",
            provider_name=self.provider_name,
            code="GEMINI_GENERATE_ERROR",
            retryable=retryable,
            details={"raw_error": sanitized_msg},
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Google Gemini."""
        client = self._get_client()
        model_name = request.model or self.default_model
        timeout = request.timeout_seconds or self.timeout_seconds

        system_instruction, contents = self._format_messages_for_gemini(request.messages)

        start_time = time.perf_counter()
        try:
            # If client provides an async interface (or custom mock client)
            if hasattr(client, "aio") and hasattr(client.aio, "models"):
                # google-genai Client async interface
                config = {
                    "temperature": request.temperature,
                }
                if system_instruction:
                    config["system_instruction"] = system_instruction
                if request.max_tokens:
                    config["max_output_tokens"] = request.max_tokens
                if request.stop_sequences:
                    config["stop_sequences"] = request.stop_sequences

                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=config,
                    ),
                    timeout=timeout,
                )
                text = response.text or ""
                usage = self._extract_usage(response)

            elif hasattr(client, "models") and hasattr(client.models, "generate_content"):
                # google-genai synchronous client called in thread
                config = {
                    "temperature": request.temperature,
                }
                if system_instruction:
                    config["system_instruction"] = system_instruction
                if request.max_tokens:
                    config["max_output_tokens"] = request.max_tokens
                if request.stop_sequences:
                    config["stop_sequences"] = request.stop_sequences

                loop = asyncio.get_event_loop()
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: client.models.generate_content(
                            model=model_name,
                            contents=contents,
                            config=config,
                        ),
                    ),
                    timeout=timeout,
                )
                text = response.text or ""
                usage = self._extract_usage(response)

            elif hasattr(client, "GenerativeModel"):
                # google-generativeai SDK legacy interface
                gen_model = client.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_instruction,
                )
                generation_config = {
                    "temperature": request.temperature,
                }
                if request.max_tokens:
                    generation_config["max_output_tokens"] = request.max_tokens
                if request.stop_sequences:
                    generation_config["stop_sequences"] = request.stop_sequences

                loop = asyncio.get_event_loop()
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: gen_model.generate_content(
                            contents,
                            generation_config=generation_config,
                        ),
                    ),
                    timeout=timeout,
                )
                text = response.text or ""
                usage = self._extract_usage(response)
            elif callable(client):
                # Callable mock / test harness
                res = client(request)
                if asyncio.iscoroutine(res):
                    res = await res
                return res
            else:
                raise ProviderError(
                    message="Unsupported Gemini client interface.",
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
                message=f"Gemini generation timed out after {timeout} seconds.",
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
            f"\n\nYou MUST respond ONLY with a valid JSON object conforming strictly to the following JSON Schema:\n"
            f"{schema_json}\n"
            f"Do not include code markdown formatting (```json) or introductory text. Output pure JSON only."
        )

        # Clone request messages and inject directive into user prompt or system prompt
        messages = [msg.model_copy() for msg in request.messages]
        messages[-1].content = messages[-1].content + directive

        current_request = request.model_copy(update={"messages": messages, "structured_schema": schema.model_json_schema()})
        last_error_msg = ""

        for attempt in range(self.max_retries + 1):
            response = await self.generate(current_request)
            raw_text = response.content.strip()

            # Strip markdown code blocks if present
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
                    # Append corrective prompt for retry
                    corrective_msg = Message(
                        role=MessageRole.USER,
                        content=(
                            f"Your previous response failed validation with error: {last_error_msg}\n"
                            f"Previous response was:\n{raw_text}\n\n"
                            f"Please correct the output to strictly conform to schema:\n{schema_json}"
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
