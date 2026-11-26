"""Redis-backed per-user chat rate limiting.

Uses the same Upstash Redis client as upload rate limiting so limits are
enforced across workers. Identity must be the authenticated JWT user id —
never a body-supplied field.

Configuration (via ``app.core.config.Settings`` / env):
  CHAT_RATE_LIMIT_COUNT          default 20
  CHAT_RATE_LIMIT_WINDOW_SECONDS default 60

Rationale: each chat may trigger a multi-step agent loop with several Groq
calls; 20 requests/minute per authenticated user is a conservative starting
point and is fully configurable.
"""

from __future__ import annotations

from app.core.config import get_settings
from app.services.temporary_memory_service import redis


class ChatRateLimitExceeded(Exception):
    """Raised when an authenticated user exceeds the chat request limit."""

    def __init__(self, retry_after_seconds: int) -> None:
        self.retry_after_seconds = max(1, int(retry_after_seconds))
        super().__init__("Chat rate limit exceeded")


_CHAT_RATE_LIMIT_SCRIPT = """
local count = redis.call('INCR', KEYS[1])
local ttl = redis.call('TTL', KEYS[1])

if ttl == -1 then
    local expiry_set = redis.call('EXPIRE', KEYS[1], ARGV[1])
    if expiry_set ~= 1 then
        return redis.error_reply('Failed to set chat rate-limit expiry')
    end
    ttl = redis.call('TTL', KEYS[1])
end

if ttl < 0 then
    return redis.error_reply('Chat rate-limit key has no valid expiry')
end

return {count, ttl}
"""


def chat_rate_limit_key(user_id: str) -> str:
    """Stable Redis key for an authenticated user."""
    return f"agent:chat:rate:{str(user_id).strip()}"


def enforce_chat_rate_limit(user_id: str) -> None:
    """Allow at most the configured number of chat requests per user per window.

    ``user_id`` must come from verified JWT identity (``AuthenticatedUser.id``).
    """
    normalized_user_id = str(user_id).strip()
    if not normalized_user_id:
        raise ValueError("A valid authenticated user ID is required")

    settings = get_settings()
    key = chat_rate_limit_key(normalized_user_id)
    result = redis.eval(
        _CHAT_RATE_LIMIT_SCRIPT,
        keys=[key],
        args=[str(int(settings.chat_rate_limit_window_seconds))],
    )

    if not isinstance(result, (list, tuple)) or len(result) != 2:
        raise RuntimeError("Unexpected Redis chat rate-limit response")

    count = int(result[0])
    ttl = max(1, int(result[1]))

    if count > int(settings.chat_rate_limit_count):
        raise ChatRateLimitExceeded(ttl)
