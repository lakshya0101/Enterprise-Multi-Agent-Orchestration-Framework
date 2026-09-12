"""Framework configuration and environment settings."""

import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env safely without raising exceptions if absent
load_dotenv()


class FrameworkSettings(BaseModel):
    """Central configuration schema loaded from environment variables."""

    gemini_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY"),
        description="Google Gemini API Key",
    )
    groq_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GROQ_API_KEY"),
        description="Groq Cloud API Key",
    )
    llm_provider: str = Field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini"),
        description="Primary default LLM provider identifier",
    )

    @property
    def default_provider(self) -> str:
        """Alias for llm_provider."""
        return self.llm_provider
    fallback_provider: str = Field(
        default_factory=lambda: os.getenv("FALLBACK_PROVIDER", "groq"),
        description="Secondary fallback LLM provider identifier",
    )
    gemini_model: str = Field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        description="Default Google Gemini model name",
    )
    groq_model: str = Field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        description="Default Groq model name",
    )
    llm_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("LLM_TIMEOUT_SECONDS", "60.0")),
        gt=0.0,
        description="Timeout deadline in seconds for LLM calls",
    )
    llm_max_retries: int = Field(
        default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "2")),
        ge=0,
        description="Maximum retry attempts for transient provider failures",
    )
    llm_enable_fallback: bool = Field(
        default_factory=lambda: os.getenv("LLM_ENABLE_FALLBACK", "true").lower() in ("true", "1", "yes"),
        description="Whether the LLMRouter automatically routes to fallback provider on failure",
    )
    langchain_tracing_v2: bool = Field(
        default_factory=lambda: os.getenv("LANGCHAIN_TRACING_V2", "false").lower() in ("true", "1", "yes"),
        description="Enable LangSmith tracing v2",
    )
    langchain_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("LANGCHAIN_API_KEY"),
        description="LangSmith API Key",
    )
    langchain_project: str = Field(
        default_factory=lambda: os.getenv("LANGCHAIN_PROJECT", "enterprise-multi-agent-framework"),
        description="LangSmith project namespace",
    )
    vector_store: str = Field(
        default_factory=lambda: os.getenv("VECTOR_STORE", "faiss"),
        description="Selected vector database backend",
    )
    embedding_provider: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "local"),
        description="Embedding provider type (e.g. 'local', 'fake')",
    )
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        description="SentenceTransformer model name or local path",
    )
    vector_store_type: str = Field(
        default_factory=lambda: os.getenv("VECTOR_STORE_TYPE", "faiss"),
        description="Vector store backend type (e.g. 'faiss')",
    )
    vector_store_path: str = Field(
        default_factory=lambda: os.getenv("VECTOR_STORE_PATH", "data/vector_store"),
        description="Filesystem directory for persisted vector indices",
    )
    state_store_type: str = Field(
        default_factory=lambda: os.getenv("STATE_STORE_TYPE", "memory"),
        description="State persistence store type ('memory', 'sqlite', 'postgres')",
    )
    sqlite_database_path: str = Field(
        default_factory=lambda: os.getenv("SQLITE_DATABASE_PATH", "data/orchestrator.db"),
        description="Filesystem path to SQLite state database",
    )
    postgres_dsn: Optional[str] = Field(
        default_factory=lambda: os.getenv("POSTGRES_DSN"),
        description="Full PostgreSQL DSN connection string (e.g. postgresql://user:pass@host:5432/dbname)",
    )
    postgres_host: str = Field(
        default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"),
        description="PostgreSQL server host",
    )
    postgres_port: int = Field(
        default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")),
        gt=0,
        le=65535,
        description="PostgreSQL server port",
    )
    postgres_db: str = Field(
        default_factory=lambda: os.getenv("POSTGRES_DB", "orchestrator"),
        description="PostgreSQL database name",
    )
    postgres_user: str = Field(
        default_factory=lambda: os.getenv("POSTGRES_USER", "postgres"),
        description="PostgreSQL user name",
    )
    postgres_password: Optional[str] = Field(
        default_factory=lambda: os.getenv("POSTGRES_PASSWORD"),
        description="PostgreSQL user password",
    )
    postgres_min_connections: int = Field(
        default_factory=lambda: int(os.getenv("POSTGRES_MIN_CONNECTIONS", "1")),
        ge=1,
        description="PostgreSQL pool minimum connections",
    )
    postgres_max_connections: int = Field(
        default_factory=lambda: int(os.getenv("POSTGRES_MAX_CONNECTIONS", "10")),
        ge=1,
        description="PostgreSQL pool maximum connections",
    )
    chunk_size: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "500")),
        gt=0,
        description="Default document chunk size in characters",
    )
    chunk_overlap: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50")),
        ge=0,
        description="Default document chunk overlap in characters",
    )
    database_url: str = Field(
        default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./data/app.db"),
        description="Persistence database connection string",
    )
    default_timeout_seconds: float = Field(
        default=60.0,
        gt=0.0,
        description="Default timeout for network and internal calls",
    )
    max_global_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum global retries per orchestration run",
    )
    api_host: str = Field(
        default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"),
        description="FastAPI service host address",
    )
    api_port: int = Field(
        default_factory=lambda: int(os.getenv("API_PORT", "8000")),
        gt=0,
        le=65535,
        description="FastAPI service port",
    )
    api_debug: bool = Field(
        default_factory=lambda: os.getenv("API_DEBUG", "false").lower() in ("true", "1", "yes"),
        description="Enable API debug mode",
    )
    environment: str = Field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development").lower(),
        description="Deployment tier: 'production', 'development', or 'test'",
    )
    api_auth_enabled: bool = Field(
        default_factory=lambda: (
            os.getenv("API_AUTH_ENABLED", "true" if os.getenv("ENVIRONMENT") == "production" else "false")
            .lower() in ("true", "1", "yes")
        ),
        description="Enforce authentication on protected API endpoints",
    )
    api_key_hashes: dict = Field(
        default_factory=lambda: (
            __import__("json").loads(os.getenv("API_KEY_HASHES", "{}"))
            if os.getenv("API_KEY_HASHES")
            else {}
        ),
        description="Dictionary mapping SHA-256 hex digests to assigned role names",
    )
    jwt_secret_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("JWT_SECRET_KEY"),
        description="Secret key for verifying Bearer JWT signatures",
    )
    jwt_algorithm: str = Field(
        default_factory=lambda: os.getenv("JWT_ALGORITHM", "HS256"),
        description="JWT signature verification algorithm",
    )
    cors_allowed_origins: list[str] = Field(
        default_factory=lambda: (
            [o.strip() for o in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",") if o.strip()]
            if os.getenv("CORS_ALLOWED_ORIGINS") is not None
            else (
                ["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:3000", "http://127.0.0.1:8000"]
                if os.getenv("ENVIRONMENT", "development").lower() != "production"
                else []
            )
        ),
        description="Explicit allowed CORS origins",
    )
    cors_allow_credentials: bool = Field(
        default_factory=lambda: os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() in ("true", "1", "yes"),
        description="Allow credentials in CORS preflight requests",
    )
    hsts_enabled: bool = Field(
        default_factory=lambda: os.getenv("HSTS_ENABLED", "true" if os.getenv("ENVIRONMENT") == "production" else "false").lower() in ("true", "1", "yes"),
        description="Inject Strict-Transport-Security header in responses",
    )
    max_request_body_bytes: int = Field(
        default_factory=lambda: int(os.getenv("MAX_REQUEST_BODY_BYTES", "1048576")),
        gt=0,
        description="Maximum allowed request payload size in bytes (default 1 MB)",
    )
    api_request_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("API_REQUEST_TIMEOUT_SECONDS", "120.0")),
        gt=0.0,
        description="Maximum request execution deadline in seconds",
    )
    rate_limit_enabled: bool = Field(
        default_factory=lambda: os.getenv("RATE_LIMIT_ENABLED", "true").lower() in ("true", "1", "yes"),
        description="Enable in-memory rate limiting",
    )
    rate_limit_public_per_minute: int = Field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_PUBLIC_PER_MINUTE", "120")),
        gt=0,
        description="Rate limit for unauthenticated public endpoints per minute per IP",
    )
    rate_limit_authenticated_per_minute: int = Field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_AUTHENTICATED_PER_MINUTE", "60")),
        gt=0,
        description="Rate limit for authenticated endpoints per minute per principal",
    )
    max_sse_subscribers_per_run: int = Field(
        default_factory=lambda: int(os.getenv("MAX_SSE_SUBSCRIBERS_PER_RUN", "10")),
        gt=0,
        description="Maximum concurrent SSE listeners allowed per run ID",
    )
    trusted_proxies: list[str] = Field(
        default_factory=lambda: (
            [p.strip() for p in os.getenv("TRUSTED_PROXIES", "").split(",") if p.strip()]
            if os.getenv("TRUSTED_PROXIES")
            else []
        ),
        description="Trusted reverse proxy IP list for X-Forwarded-For resolution",
    )
    execution_backend: str = Field(
        default_factory=lambda: os.getenv("EXECUTION_BACKEND", "local_async"),
        description="Execution backend engine ('local_async')",
    )
    execution_max_concurrency: int = Field(
        default_factory=lambda: int(os.getenv("EXECUTION_MAX_CONCURRENCY", "10")),
        gt=0,
        description="Maximum concurrent active workflow executions",
    )
    execution_max_queue_size: int = Field(
        default_factory=lambda: int(os.getenv("EXECUTION_MAX_QUEUE_SIZE", "100")),
        gt=0,
        description="Maximum pending queued workflow submissions before backpressure rejection",
    )
    execution_shutdown_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("EXECUTION_SHUTDOWN_TIMEOUT_SECONDS", "30.0")),
        gt=0.0,
        description="Maximum graceful shutdown timeout in seconds",
    )
    prometheus_metrics_enabled: bool = Field(
        default_factory=lambda: os.getenv("PROMETHEUS_METRICS_ENABLED", "true").lower() in ("true", "1", "yes"),
        description="Enable the /metrics Prometheus text exposition endpoint",
    )
    prometheus_metrics_path: str = Field(
        default_factory=lambda: os.getenv("PROMETHEUS_METRICS_PATH", "/metrics"),
        description="Path for Prometheus scraping endpoint",
    )
    prometheus_metrics_require_auth: bool = Field(
        default_factory=lambda: (
            os.getenv("PROMETHEUS_METRICS_REQUIRE_AUTH", "true" if os.getenv("ENVIRONMENT") == "production" else "false")
            .lower() in ("true", "1", "yes")
        ),
        description="Enforce Track B authentication on Prometheus /metrics endpoint",
    )
    otel_enabled: bool = Field(
        default_factory=lambda: os.getenv("OTEL_ENABLED", "false").lower() in ("true", "1", "yes"),
        description="Enable OpenTelemetry tracing adapter and exporter",
    )
    otel_service_name: str = Field(
        default_factory=lambda: os.getenv("OTEL_SERVICE_NAME", "enterprise-orchestrator"),
        description="Service name reported to OpenTelemetry collector",
    )
    otel_exporter_otlp_endpoint: Optional[str] = Field(
        default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        description="OTLP collector endpoint URL",
    )
    otel_exporter_otlp_headers: Optional[str] = Field(
        default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_HEADERS"),
        description="Custom headers for OTLP export",
    )
    otel_exporter_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("OTEL_EXPORTER_TIMEOUT_SECONDS", "5.0")),
        gt=0.0,
        description="Maximum timeout in seconds for OTel export and shutdown flush",
    )

    def model_dump_safe(self) -> dict:
        """Dump settings with all sensitive keys redacted."""
        data = self.model_dump()
        sensitive_keys = {
            "gemini_api_key",
            "groq_api_key",
            "langchain_api_key",
            "postgres_password",
            "jwt_secret_key",
            "otel_exporter_otlp_headers",
        }
        for key in sensitive_keys:
            if key in data and data[key]:
                data[key] = "[REDACTED]"
        return data

    def __repr__(self) -> str:
        """Safe string representation without secrets."""
        safe_data = self.model_dump_safe()
        return f"FrameworkSettings({safe_data})"

    def __str__(self) -> str:
        """Safe string representation without secrets."""
        return self.__repr__()
