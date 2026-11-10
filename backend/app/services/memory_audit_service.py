"""Stage 19: Traceable memory audit events persisted in MongoDB Atlas.

Supported event types:
- memory_saved
- memory_reinforced
- memory_superseded
- memory_forgotten
- memory_deleted
- memory_retrieved
- memory_deduplicated
- duplicate_detected
- conflict_detected
- conflict_confirmation_requested
- conflict_resolved
- unauthorized_operation_rejected
- operation_failed
- operation_retried
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.schemas.memory_schema import MemoryAuditEvent
from app.utils.memory_logging import log_memory_operation


# Bound at import time by memory_service after the Mongo client is ready.
audit_collection = None

# Stage 19: Event types that are recognized for observability.
KNOWN_EVENT_TYPES = frozenset({
    "memory_saved",
    "memory_reinforced",
    "memory_superseded",
    "memory_forgotten",
    "memory_deleted",
    "memory_retrieved",
    "memory_deduplicated",
    "duplicate_detected",
    "conflict_detected",
    "conflict_confirmation_requested",
    "conflict_resolved",
    "unauthorized_operation_rejected",
    "operation_failed",
    "operation_retried",
    "memory_metadata_backfilled",
    "memory_summary_generated",
    "memory_summary_invalidated",
    # Topic 21 — permissions
    "permission_denied",
    "permission_granted",
    "permission_revoked",
    # Idempotent replays
    "save_memory_idempotent",
    "update_memory_rejected",
})


def configure_audit_collection(collection) -> None:
    global audit_collection
    audit_collection = collection


def _sanitize_details(details: dict | None) -> dict:
    """Stage 19: Strip sensitive fields from audit event metadata."""
    if not details:
        return {}
    safe = dict(details)
    sensitive_keys = {
        "api_key", "password", "secret", "token", "credential",
        "mongo_uri", "connection_string", "MONGO_URI", "GROQ_API_KEY",
        "GEMINI_API_KEY", "TAVILY_API_KEY",
    }
    for key in list(safe.keys()):
        if key.lower() in {k.lower() for k in sensitive_keys}:
            safe[key] = "[REDACTED]"
    return safe


def record_audit_event(
    user_id: str,
    event_type: str,
    *,
    memory_id: str | None = None,
    operation_id: str | None = None,
    details: dict | None = None,
) -> dict:
    safe_details = _sanitize_details(details)

    payload = MemoryAuditEvent(
        user_id=user_id,
        memory_id=memory_id,
        event_type=event_type,
        operation_id=operation_id,
        details=safe_details,
        created_at=datetime.now(timezone.utc),
    ).model_dump()

    try:
        if audit_collection is None:
            log_memory_operation(
                event_type,
                user_id,
                success=False,
                memory_id=memory_id,
                operation_id=operation_id,
                details={"error": "audit_collection_not_configured"},
            )
            return {"success": False, "error": "Audit collection not configured"}

        result = audit_collection.insert_one(payload)
        log_memory_operation(
            event_type,
            user_id,
            success=True,
            memory_id=memory_id,
            operation_id=operation_id,
            details=safe_details,
        )
        return {"success": True, "audit_id": str(result.inserted_id)}
    except Exception as exc:
        log_memory_operation(
            event_type,
            user_id,
            success=False,
            memory_id=memory_id,
            operation_id=operation_id,
            details={"error": str(exc)},
        )
        return {"success": False, "error": str(exc)}
