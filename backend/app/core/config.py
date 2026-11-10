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
    access_token_ttl_minutes: int = int(os.getenv("ACCESS_TOKEN_TTL_MINUTES", "15"))
    refresh_token_ttl_days: int = int(os.getenv("REFRESH_TOKEN_TTL_DAYS", "14"))

    cookie_secure: bool = os.getenv("AUTH_COOKIE_SECURE", "false").lower() in {
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

    rate_limit_register: str = os.getenv("RATE_LIMIT_REGISTER", "5/minute")
    rate_limit_login: str = os.getenv("RATE_LIMIT_LOGIN", "10/minute")
    rate_limit_refresh: str = os.getenv("RATE_LIMIT_REFRESH", "20/minute")

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
