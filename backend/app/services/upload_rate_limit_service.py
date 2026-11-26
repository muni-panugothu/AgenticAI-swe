"""Redis-backed per-user upload rate limiting."""

from __future__ import annotations

from app.services.rag_config import settings
from app.services.temporary_memory_service import redis


class UploadRateLimitExceeded(Exception):
    """Raised when an authenticated user exceeds the upload limit."""

    def __init__(self, retry_after_seconds: int) -> None:
        self.retry_after_seconds = max(1, retry_after_seconds)
        super().__init__("Upload rate limit exceeded")



_UPLOAD_RATE_LIMIT_SCRIPT = """
local count = redis.call('INCR', KEYS[1])
local ttl = redis.call('TTL', KEYS[1])

if ttl == -1 then
    local expiry_set = redis.call('EXPIRE', KEYS[1], ARGV[1])

    if expiry_set ~= 1 then
        return redis.error_reply('Failed to set upload rate-limit expiry')
    end

    ttl = redis.call('TTL', KEYS[1])
end

if ttl < 0 then
    return redis.error_reply('Upload rate-limit key has no valid expiry')
end

return {count, ttl}
"""



def enforce_upload_rate_limit(user_id: str) -> None:
    """Allow at most the configured number of uploads per user per window."""

    normalized_user_id = str(user_id).strip()

    if not normalized_user_id:
        raise ValueError("A valid authenticated user ID is required")

    key = f"rag:upload:rate:{normalized_user_id}"

    result = redis.eval(
        _UPLOAD_RATE_LIMIT_SCRIPT,
        keys=[key],
        args=[str(settings.upload_rate_limit_window_seconds)],
    )

    if not isinstance(result, (list, tuple)) or len(result) != 2:
        raise RuntimeError("Unexpected Redis upload rate-limit response")

    count = int(result[0])
    ttl = max(1, int(result[1]))

    if count > settings.upload_rate_limit_count:
        raise UploadRateLimitExceeded(ttl)