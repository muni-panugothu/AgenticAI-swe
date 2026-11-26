"""Redis vector-only embedding cache for RAG.

Caches float vectors keyed by model/dimensions/task/content hash/version.
Never stores document text, user IDs, or ACL metadata. Cache failures fall
back to live embedding. Cached vectors never bypass authorization.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from typing import Any

from app.services.rag_config import settings

logger = logging.getLogger("agentic.rag")

_ALLOWED_PAYLOAD_KEYS = frozenset({"vector", "dimensions"})


def _content_hash(text: str) -> str:
    normalized = " ".join(str(text).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def cache_key(
    *,
    text: str,
    model: str,
    dimensions: int,
    task_type: str,
    embedding_version: str,
    chunking_version: str | None = None,
) -> str:
    payload = "|".join(
        [
            str(model),
            str(dimensions),
            str(task_type),
            str(embedding_version),
            str(chunking_version or ""),
            _content_hash(text),
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"rag:emb:v1:{digest}"


def _validate_vector(values: Any, dimensions: int) -> list[float] | None:
    if not isinstance(values, list) or len(values) != dimensions:
        return None
    try:
        vector = [float(item) for item in values]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in vector):
        return None
    return vector


def _parse_cache_payload(raw: Any, dimensions: int) -> list[float] | None:
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
    elif isinstance(raw, dict):
        payload = raw
    else:
        return None
    if not isinstance(payload, dict):
        return None
    # Reject tainted entries that include text/user/ACL fields.
    keys = set(payload.keys())
    if not keys or not keys.issubset(_ALLOWED_PAYLOAD_KEYS):
        return None
    if "text" in payload or "user_id" in payload or "content" in payload:
        return None
    stored_dims = payload.get("dimensions")
    if stored_dims is not None:
        try:
            if int(stored_dims) != dimensions:
                return None
        except (TypeError, ValueError):
            return None
    return _validate_vector(payload.get("vector"), dimensions)


def get_cached_embedding(
    *,
    text: str,
    model: str,
    dimensions: int,
    task_type: str,
    embedding_version: str,
    chunking_version: str | None = None,
) -> list[float] | None:
    if not settings.embedding_cache:
        return None
    try:
        from app.services.temporary_memory_service import redis

        key = cache_key(
            text=text,
            model=model,
            dimensions=dimensions,
            task_type=task_type,
            embedding_version=embedding_version,
            chunking_version=chunking_version,
        )
        return _parse_cache_payload(redis.get(key), dimensions)
    except Exception as exc:
        logger.warning("RAG embedding cache read failed (%s)", type(exc).__name__)
        return None


def set_cached_embedding(
    *,
    text: str,
    model: str,
    dimensions: int,
    task_type: str,
    embedding_version: str,
    vector: list[float],
    chunking_version: str | None = None,
) -> None:
    if not settings.embedding_cache:
        return
    validated = _validate_vector(vector, dimensions)
    if validated is None:
        return
    try:
        from app.services.temporary_memory_service import redis

        key = cache_key(
            text=text,
            model=model,
            dimensions=dimensions,
            task_type=task_type,
            embedding_version=embedding_version,
            chunking_version=chunking_version,
        )
        # Vector + dims only — never store source text or identity fields.
        payload = json.dumps(
            {"vector": validated, "dimensions": int(dimensions)},
            separators=(",", ":"),
        )
        # Overwrite is safe: identical keys imply identical model/dims/content.
        redis.set(key, payload, ex=int(settings.embedding_cache_ttl_seconds))
    except Exception as exc:
        logger.warning("RAG embedding cache write failed (%s)", type(exc).__name__)
