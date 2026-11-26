"""Application settings from environment variables."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


class Settings:
    """Central configuration — never log secrets from this object."""

    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8787"))

    jwt_secret: str = os.getenv("JWT_SECRET", "")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    jwt_issuer: str = os.getenv("JWT_ISSUER", "agentic-ai-api")
    jwt_audience: str = os.getenv("JWT_AUDIENCE", "agentic-ai-cli")
    access_token_ttl_minutes: int = int(os.getenv("ACCESS_TOKEN_TTL_MINUTES", "15"))
    refresh_token_ttl_days: int = int(os.getenv("REFRESH_TOKEN_TTL_DAYS", "14"))

    cookie_secure: bool = os.getenv("AUTH_COOKIE_SECURE", "true").lower() in {
        "1",
        "true",
        "yes",
    }
    cookie_samesite: str = os.getenv("AUTH_COOKIE_SAMESITE", "lax")
    refresh_cookie_name: str = os.getenv("REFRESH_COOKIE_NAME", "refresh_token")

    cors_origins: list[str] = _split_csv(os.getenv("CORS_ORIGINS"))
    csrf_cookie_name: str = os.getenv("CSRF_COOKIE_NAME", "csrf_token")

    auth_users_collection: str = os.getenv("AUTH_USERS_COLLECTION", "auth_users")
    auth_sessions_collection: str = os.getenv(
        "AUTH_SESSIONS_COLLECTION", "auth_sessions"
    )
    auth_password_reset_collection: str = os.getenv(
        "AUTH_PASSWORD_RESET_COLLECTION", "auth_password_resets"
    )

    # Legal policy versions recorded at registration (immutable audit fields).
    terms_version: str = os.getenv("TERMS_VERSION", "2026-11-01")
    privacy_version: str = os.getenv("PRIVACY_VERSION", "2026-11-01")

    # Remember-me extends refresh-session lifetime (days).
    remember_me_refresh_ttl_days: int = int(
        os.getenv("REMEMBER_ME_REFRESH_TTL_DAYS", "30")
    )

    # Password reset tokens (single-use, hashed at rest).
    password_reset_ttl_minutes: int = int(
        os.getenv("PASSWORD_RESET_TTL_MINUTES", "30")
    )
    password_reset_public_base_url: str = os.getenv(
        "PASSWORD_RESET_PUBLIC_BASE_URL",
        "http://localhost:8081/reset-password",
    )
    # When SMTP is unset, optionally log the full reset URL for local QA only.
    password_reset_log_url_in_dev: bool = os.getenv(
        "PASSWORD_RESET_LOG_URL_IN_DEV", "false"
    ).lower() in {"1", "true", "yes"}

    # Optional SMTP for password-reset delivery.
    smtp_host: str = os.getenv("SMTP_HOST", "")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_username: str = os.getenv("SMTP_USERNAME", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    smtp_from: str = os.getenv("SMTP_FROM", "")
    smtp_use_tls: bool = os.getenv("SMTP_USE_TLS", "true").lower() in {
        "1",
        "true",
        "yes",
    }

    rate_limit_register: str = os.getenv("RATE_LIMIT_REGISTER", "5/minute")
    rate_limit_login: str = os.getenv("RATE_LIMIT_LOGIN", "10/minute")
    rate_limit_refresh: str = os.getenv("RATE_LIMIT_REFRESH", "20/minute")
    rate_limit_forgot_password: str = os.getenv(
        "RATE_LIMIT_FORGOT_PASSWORD", "5/minute"
    )
    rate_limit_reset_password: str = os.getenv(
        "RATE_LIMIT_RESET_PASSWORD", "10/minute"
    )

    # Authenticated /api/chat: Redis-backed per JWT user id (not SlowAPI memory).
    # Default 20 requests / 60s — each chat may run a multi-step agent loop.
    chat_rate_limit_count: int = int(os.getenv("CHAT_RATE_LIMIT_COUNT", "20"))
    chat_rate_limit_window_seconds: int = int(
        os.getenv("CHAT_RATE_LIMIT_WINDOW_SECONDS", "60")
    )

    # Groq SDK transport: explicit finite timeout; SDK retries default to 0 so
    # application-level decide_llm_retry remains the sole retry authority.
    groq_timeout_seconds: float = float(os.getenv("GROQ_TIMEOUT_SECONDS", "60"))
    groq_max_retries: int = int(os.getenv("GROQ_MAX_RETRIES", "0"))

    # Topic 42b — provider selection (openai reserved; adapter not implemented).
    llm_provider: str = (os.getenv("LLM_PROVIDER") or "groq").strip().lower()
    groq_model: str = os.getenv("GROQ_MODEL") or "openai/gpt-oss-20b"
    # Future OpenAI contract (unused until adapter ships):
    # OPENAI_API_KEY, OPENAI_MODEL, optional OPENAI_BASE_URL

    require_auth_for_chat: bool = os.getenv("REQUIRE_AUTH_FOR_CHAT", "true").lower() in {
        "1",
        "true",
        "yes",
    }

    def validate_startup(self) -> None:
        if not self.jwt_secret or len(self.jwt_secret) < 32:
            raise ValueError(
                "JWT_SECRET must be set in environment (min 32 characters)"
            )
        if self.jwt_algorithm != "HS256":
            raise ValueError("JWT_ALGORITHM must be HS256")
        if not self.jwt_issuer or not self.jwt_audience:
            raise ValueError("JWT_ISSUER and JWT_AUDIENCE must be configured")
        if self.chat_rate_limit_count < 1:
            raise ValueError("CHAT_RATE_LIMIT_COUNT must be >= 1")
        if self.chat_rate_limit_window_seconds < 1:
            raise ValueError("CHAT_RATE_LIMIT_WINDOW_SECONDS must be >= 1")
        if self.groq_timeout_seconds <= 0:
            raise ValueError("GROQ_TIMEOUT_SECONDS must be > 0")
        if self.groq_max_retries < 0:
            raise ValueError("GROQ_MAX_RETRIES must be >= 0")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build settings; re-read Phase B env knobs so tests can override safely."""
    settings = Settings()
    settings.chat_rate_limit_count = int(os.getenv("CHAT_RATE_LIMIT_COUNT", "20"))
    settings.chat_rate_limit_window_seconds = int(
        os.getenv("CHAT_RATE_LIMIT_WINDOW_SECONDS", "60")
    )
    settings.groq_timeout_seconds = float(os.getenv("GROQ_TIMEOUT_SECONDS", "60"))
    settings.groq_max_retries = int(os.getenv("GROQ_MAX_RETRIES", "0"))
    return settings
