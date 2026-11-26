"""Append-only security audit events for Topics 48–53.

Never log secrets, tokens, or full sensitive argument payloads.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from app.validation.agent_security_models import SecurityEventType

logger = logging.getLogger("agentic.security.audit")

_COLLECTION_NAME = os.getenv(
    "AGENT_SECURITY_AUDIT_COLLECTION",
    "agent_security_audit",
)

_collection = None


def configure_security_audit_collection(collection) -> None:
    global _collection
    _collection = collection


def _get_collection():
    global _collection
    if _collection is not None:
        return _collection
    try:
        from app.services.memory_service import database

        _collection = database[_COLLECTION_NAME]
        return _collection
    except Exception:
        return None


def record_security_event(
    event_type: SecurityEventType | str,
    *,
    user_id: str | None = None,
    project_id: str | None = None,
    tenant_id: str | None = None,
    tool_name: str | None = None,
    request_id: str | None = None,
    approval_id: str | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    """Best-effort durable audit; never raises into the tool path."""
    safe_detail: dict[str, Any] = {}
    if isinstance(detail, dict):
        for key, value in detail.items():
            if key.lower() in {
                "password",
                "token",
                "secret",
                "api_key",
                "authorization",
                "arguments",
                "content",
                "memory",
            }:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                safe_detail[key] = value
            else:
                safe_detail[key] = str(value)[:200]

    doc = {
        "event_type": str(event_type),
        "user_id": user_id,
        "project_id": project_id,
        "tenant_id": tenant_id,
        "tool_name": tool_name,
        "request_id": request_id,
        "approval_id": approval_id,
        "detail": safe_detail,
        "created_at": datetime.now(timezone.utc),
    }
    try:
        col = _get_collection()
        if col is not None:
            col.insert_one(doc)
    except Exception as exc:
        logger.warning("security audit persist failed: %s", exc)
    logger.info(
        "security_event type=%s tool=%s user=%s approval=%s",
        event_type,
        tool_name,
        user_id,
        approval_id,
    )
