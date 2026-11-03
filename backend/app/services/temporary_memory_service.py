
import os
import json
from uuid import uuid4

from dotenv import load_dotenv
from upstash_redis import Redis

load_dotenv()

redis = Redis.from_env()

TTL_SECONDS = {
    "temporary": 300,       # 5 minutes
    "short_term": 1800,     # 30 minutes
    "session": 86400,       # 24 hours
}


def save_temporary_memory(
    user_id: str,
    memory: str,
    memory_type: str = "temporary",
) -> dict:
    if memory_type not in TTL_SECONDS:
        raise ValueError(f"Unsupported memory type: {memory_type}")

    memory_id = str(uuid4())
    key = f"agent:memory:{user_id}:{memory_id}"

    payload = {
        "memory_id": memory_id,
        "user_id": user_id,
        "memory": memory,
        "memory_type": memory_type,
    }

    redis.set(
        key,
        json.dumps(payload),
        ex=TTL_SECONDS[memory_type],
    )

    return {
        "success": True,
        "memory_id": memory_id,
        "ttl_seconds": TTL_SECONDS[memory_type],
    }


def get_temporary_memory(user_id: str, memory_id: str):
    key = f"agent:memory:{user_id}:{memory_id}"
    value = redis.get(key)

    if value is None:
        return None

    if isinstance(value, bytes):
        value = value.decode("utf-8")

    if isinstance(value, str):
        return json.loads(value)

    return value


def delete_temporary_memory(user_id: str, memory_id: str) -> bool:
    key = f"agent:memory:{user_id}:{memory_id}"
    return bool(redis.delete(key))





# =======================================================
# TEMPORARY TASK / CONVERSATION CONTEXT
# =======================================================

TASK_CONTEXT_TTL = 1800 # 30 minutes
MAX_TASK_MESSAGES = 10


class TaskContextScriptError(RuntimeError):
    """A task-context script reported a failed Redis command.

    The stage names the command that failed. Earlier commands in
    the same script are already applied. Redis does not roll them back.
    """

    def __init__(self, stage: str, detail: str):
        self.stage = stage
        self.detail = detail
        super().__init__(
            f"Task context script failed at {stage}: {detail}"
        )


# One EVAL so a crash between network calls cannot separate the
# dedupe claim from the append. The script is atomic against other
# clients. It does not roll back: if RPUSH fails, the script itself
# deletes the dedupe key. If LTRIM or EXPIRE fails, the message and
# the dedupe key both remain.
_ADD_TASK_CONTEXT_ONCE_SCRIPT = """
local ttl = tonumber(ARGV[1])
local max_messages = tonumber(ARGV[2])

local claimed = redis.call('SET', KEYS[1], '1', 'EX', ttl, 'NX')
if not claimed then
    return {'duplicate'}
end

local pushed = redis.pcall('RPUSH', KEYS[2], ARGV[3])
if type(pushed) == 'table' and pushed.err then
    local removed = redis.pcall('DEL', KEYS[1])
    if type(removed) == 'table' and removed.err then
        return {'error', 'rpush_failed_dedupe_remains', pushed.err}
    end
    return {'error', 'rpush_failed_dedupe_cleared', pushed.err}
end

local trimmed = redis.pcall('LTRIM', KEYS[2], -max_messages, -1)
if type(trimmed) == 'table' and trimmed.err then
    return {'error', 'ltrim_failed_message_kept', trimmed.err}
end

local expired = redis.pcall('EXPIRE', KEYS[2], ttl)
if type(expired) == 'table' and expired.err then
    return {'error', 'expire_failed_message_kept', expired.err}
end
if expired == 0 then
    return {'error', 'expire_failed_message_kept', 'EXPIRE did not set a timeout'}
end

return {'saved'}
"""


def _decode_script_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _interpret_add_task_context_result(result, role: str, request_id: str) -> dict:
    """Map the script reply to the existing save result.

    A client exception from EVAL is not handled here. The script may
    already have finished, so the caller must not delete the dedupe key.
    """
    if isinstance(result, (str, bytes)):
        parts = [_decode_script_value(result)]
    elif isinstance(result, (list, tuple)):
        parts = [_decode_script_value(part) for part in result]
    else:
        raise TaskContextScriptError("unexpected_reply", repr(result))

    status = parts[0] if parts else None

    if status == "saved":
        return {
            "success": True,
            "duplicate": False,
            "role": role,
            "request_id": request_id,
        }

    if status == "duplicate":
        return {
            "success": True,
            "duplicate": True,
            "message": "Message was already saved for this request.",
        }

    if status == "error":
        stage = str(parts[1]) if len(parts) > 1 else "unknown"
        detail = str(parts[2]) if len(parts) > 2 else ""
        raise TaskContextScriptError(stage, detail)

    raise TaskContextScriptError("unexpected_reply", repr(result))


def add_task_context_once(
    user_id: str,
    request_id: str,
    role: str,
    content: str,
) -> dict:
    """
    Append a task-context message once per request ID and role.

    Repeated calls with the same user_id, request_id, and role
    will not append another copy during the deduplication TTL.

    SET NX, RPUSH, LTRIM, and EXPIRE run in one Lua script.
    Other clients cannot interleave those commands. If a later
    command fails, Redis leaves the earlier writes in place.
    """
    if role not in ("user", "assistant"):
        raise ValueError("Role must be 'user' or 'assistant'.")

    if not request_id or not request_id.strip():
        raise ValueError("request_id cannot be empty.")

    if not content or not content.strip():
        raise ValueError("Content cannot be empty.")

    dedupe_key = _dedupe_key(user_id, request_id, role)
    list_key = f"agent:task_context:{user_id}"
    message = json.dumps({
        "role": role,
        "content": content,
        "request_id": request_id,
    })

    # No delete here. An exception can mean the reply was lost
    # after the script completed. Deleting the dedupe key would
    # let a retry append a second copy.
    result = redis.eval(
        _ADD_TASK_CONTEXT_ONCE_SCRIPT,
        keys=[dedupe_key, list_key],
        args=[
            str(TASK_CONTEXT_TTL),
            str(MAX_TASK_MESSAGES),
            message,
        ],
    )

    return _interpret_add_task_context_result(
        result,
        role,
        request_id,
    )


def add_task_context(
    user_id:str,
    role:str,
    content:str,
) -> dict:
    """Store one user/assistant message in temporary Reis context."""

    if role not in ("user", "assistant"):
        raise ValueError("Role must be 'user' or 'assistant'.")

    if not content or not content.strip():
        raise ValueError("Content cannot be empty.")


    key = f"agent:task_context:{user_id}"

    message = json.dumps({
        "role":role,
        "content":content,
    })

    # Add only the most recent messages
    redis.rpush(key, message)
    redis.ltrim(key, -MAX_TASK_MESSAGES, -1)

    # Refresh the inactivity timeout on every new message
    redis.expire(key, TASK_CONTEXT_TTL)

    return {
        "success":True,
        "role":role,
        "ttl_seconds":TASK_CONTEXT_TTL,
    }


def get_task_context(user_id:str) -> list:
    """Retrieve recent temporary conversation messages."""

    key = f"agent:task_context:{user_id}"

    values = redis.lrange(key, 0, -1)
    messages = []

    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8")

        if isinstance(value, str):
            messages.append(json.loads(value))
        else:
            messages.append(value)

    return messages


def _dedupe_key(user_id: str, request_id: str, role: str) -> str:
    return f"agent:task_context:dedupe:{user_id}:{request_id}:{role}"


def _escape_redis_glob(value: str) -> str:
    """Escape glob characters so a user ID cannot widen a SCAN match."""
    escaped = []
    for char in value:
        if char in {"\\", "*", "?", "[", "]"}:
            escaped.append("\\")
        escaped.append(char)
    return "".join(escaped)


def _list_dedupe_keys(user_id: str) -> list[str]:
    """Return deduplication keys that belong only to this user."""
    prefix = f"agent:task_context:dedupe:{user_id}:"
    pattern = f"{_escape_redis_glob(prefix)}*"
    cursor = 0
    found = []
    seen = set()

    while True:
        cursor, keys = redis.scan(cursor, match=pattern, count=100)

        if isinstance(keys, (str, bytes)):
            keys = [keys]

        for key in keys or []:
            if isinstance(key, bytes):
                key = key.decode("utf-8")

            if not isinstance(key, str) or not key.startswith(prefix):
                continue

            request_id, separator, role = key[len(prefix):].rpartition(":")
            if (
                separator != ":"
                or not request_id
                or role not in ("user", "assistant")
                or key in seen
            ):
                continue

            seen.add(key)
            found.append(key)

        cursor = int(cursor)
        if cursor == 0:
            break

    return found


def clear_task_context(user_id: str) -> bool:
    """Clear temporary conversation context and its dedupe keys.

    The message list and keys such as
    agent:task_context:dedupe:{user_id}:{request_id}:{role}
    are removed together. Otherwise a retry with the same request
    ID can be rejected as a duplicate for the rest of the 30-minute
    dedupe TTL, even though the message itself is gone.
    """
    key = f"agent:task_context:{user_id}"
    context_existed = bool(redis.exists(key))
    dedupe_keys = _list_dedupe_keys(user_id)

    redis.delete(key, *dedupe_keys)

    return context_existed

