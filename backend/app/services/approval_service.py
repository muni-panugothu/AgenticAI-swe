"""Human approval workflow with atomic one-time consumption (Topics 48/50)."""

from __future__ import annotations

import os
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from pymongo import ReturnDocument

from app.services.agent_security_audit import record_security_event
from app.services.agent_security_policy import (
    POLICY_VERSION,
    arguments_digest,
    get_tool_risk_meta,
    normalize_tool_arguments,
    target_resource_for_tool,
    tool_requires_approval,
)
from app.validation.agent_security_models import (
    APPROVAL_TRANSITIONS,
    ApprovalRecord,
    ApprovalStatus,
)

_COLLECTION_NAME = os.getenv(
    "AGENT_APPROVALS_COLLECTION",
    "agent_approvals",
)

DEFAULT_TTL_SECONDS = int(os.getenv("AGENT_APPROVAL_TTL_SECONDS", "900"))
MAX_TTL_SECONDS = int(os.getenv("AGENT_APPROVAL_MAX_TTL_SECONDS", "86400"))

_collection = None
# Process-local lock supplements Mongo atomic ops for in-memory fakes / races.
_consume_lock = threading.Lock()


def configure_approval_collection(collection) -> None:
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


def ensure_approval_indexes() -> None:
    col = _get_collection()
    if col is None:
        return
    col.create_index("approval_id", unique=True, name="approval_id_unique")
    col.create_index(
        [("user_id", 1), ("status", 1), ("expires_at", 1)],
        name="approval_user_status",
    )
    col.create_index(
        [("status", 1), ("expires_at", 1)],
        name="approval_expiry_scan",
    )


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _transition_allowed(from_status: ApprovalStatus, to_status: ApprovalStatus) -> bool:
    allowed = APPROVAL_TRANSITIONS.get(to_status)
    if not allowed:
        return False
    return from_status in allowed


def create_approval_request(
    *,
    user_id: str,
    tool_name: str,
    arguments: dict[str, Any] | None,
    project_id: str | None = None,
    tenant_id: str | None = None,
    request_id: str | None = None,
    task_id: str | None = None,
    reason: str | None = None,
    ttl_seconds: int | None = None,
) -> dict[str, Any]:
    col = _get_collection()
    if col is None:
        return {
            "success": False,
            "error": "approval_store_unavailable",
            "error_code": "APPROVAL_STORE_UNAVAILABLE",
        }
    if not user_id:
        return {
            "success": False,
            "error": "authenticated_user_required",
            "error_code": "UNAUTHORIZED",
        }

    ttl = ttl_seconds if ttl_seconds is not None else DEFAULT_TTL_SECONDS
    ttl = max(30, min(int(ttl), MAX_TTL_SECONDS))
    meta = get_tool_risk_meta(tool_name)
    normalized = normalize_tool_arguments(arguments)
    digest = arguments_digest(arguments)
    target = target_resource_for_tool(tool_name, arguments)
    now = _now()
    approval_id = str(uuid.uuid4())
    doc = {
        "approval_id": approval_id,
        "status": ApprovalStatus.PENDING.value,
        "user_id": user_id,
        "project_id": project_id,
        "tenant_id": tenant_id,
        "request_id": request_id,
        "task_id": task_id,
        "tool_name": tool_name,
        "action": "invoke",
        "arguments_digest": digest,
        "normalized_arguments": normalized,
        "target_resource": target,
        "risk_class": meta.risk_class.value,
        "reason": reason
        or f"Tool '{tool_name}' requires human approval ({meta.risk_class.value})",
        "policy_version": POLICY_VERSION,
        "created_at": now,
        "expires_at": now + timedelta(seconds=ttl),
        "decided_at": None,
        "decided_by": None,
        "consumed_at": None,
        "revoked_at": None,
        "rejection_reason": None,
    }
    col.insert_one(doc)
    record_security_event(
        "approval_created",
        user_id=user_id,
        project_id=project_id,
        tenant_id=tenant_id,
        tool_name=tool_name,
        request_id=request_id,
        approval_id=approval_id,
        detail={"risk": meta.risk_class.value},
    )
    record_security_event(
        "approval_required",
        user_id=user_id,
        project_id=project_id,
        tenant_id=tenant_id,
        tool_name=tool_name,
        request_id=request_id,
        approval_id=approval_id,
    )
    public = ApprovalRecord(**{k: v for k, v in doc.items() if k != "_id"})
    return {
        "success": True,
        "approval_required": True,
        "approval": public.model_dump(mode="json"),
        "status": ApprovalStatus.PENDING.value,
    }


def get_approval(approval_id: str) -> dict[str, Any] | None:
    col = _get_collection()
    if col is None:
        return None
    doc = col.find_one({"approval_id": approval_id})
    if not doc:
        return None
    doc.pop("_id", None)
    return doc


def _expire_if_needed(doc: dict[str, Any]) -> dict[str, Any]:
    status = ApprovalStatus(doc["status"])
    expires = _as_aware(doc.get("expires_at"))
    now = _now()
    if (
        status in {ApprovalStatus.PENDING, ApprovalStatus.APPROVED}
        and expires is not None
        and expires <= now
    ):
        col = _get_collection()
        if col is not None:
            updated = col.find_one_and_update(
                {
                    "approval_id": doc["approval_id"],
                    "status": {"$in": [status.value]},
                },
                {
                    "$set": {
                        "status": ApprovalStatus.EXPIRED.value,
                    }
                },
                return_document=ReturnDocument.AFTER,
            )
            if updated:
                updated.pop("_id", None)
                record_security_event(
                    "approval_expired",
                    user_id=updated.get("user_id"),
                    project_id=updated.get("project_id"),
                    tenant_id=updated.get("tenant_id"),
                    tool_name=updated.get("tool_name"),
                    approval_id=updated.get("approval_id"),
                )
                return updated
        doc = dict(doc)
        doc["status"] = ApprovalStatus.EXPIRED.value
    return doc


def decide_approval(
    *,
    approval_id: str,
    reviewer_user_id: str,
    approve: bool,
    rejection_reason: str | None = None,
) -> dict[str, Any]:
    """Approve or reject. Reviewer must be the same authenticated user (owner).

    Cross-user approval is rejected unless AGENT_APPROVAL_REVIEWER_USER_IDS
    includes the reviewer.
    """
    col = _get_collection()
    if col is None:
        return {"success": False, "error": "approval_store_unavailable"}

    doc = get_approval(approval_id)
    if not doc:
        return {"success": False, "error": "approval_not_found"}
    doc = _expire_if_needed(doc)
    if doc["status"] != ApprovalStatus.PENDING.value:
        return {
            "success": False,
            "error": "invalid_status_for_decision",
            "status": doc["status"],
        }

    extra_reviewers = {
        x.strip()
        for x in os.getenv("AGENT_APPROVAL_REVIEWER_USER_IDS", "").split(",")
        if x.strip()
    }
    if reviewer_user_id != doc["user_id"] and reviewer_user_id not in extra_reviewers:
        record_security_event(
            "approval_rejected",
            user_id=doc.get("user_id"),
            approval_id=approval_id,
            detail={"reason": "unauthorized_reviewer", "reviewer": reviewer_user_id},
        )
        return {
            "success": False,
            "error": "unauthorized_reviewer",
            "error_code": "APPROVAL_UNAUTHORIZED_REVIEWER",
        }

    new_status = (
        ApprovalStatus.APPROVED if approve else ApprovalStatus.REJECTED
    )
    if not _transition_allowed(ApprovalStatus.PENDING, new_status):
        return {"success": False, "error": "transition_not_allowed"}

    update: dict[str, Any] = {
        "status": new_status.value,
        "decided_at": _now(),
        "decided_by": reviewer_user_id,
    }
    if not approve:
        update["rejection_reason"] = (rejection_reason or "rejected")[:500]

    updated = col.find_one_and_update(
        {
            "approval_id": approval_id,
            "status": ApprovalStatus.PENDING.value,
        },
        {"$set": update},
        return_document=ReturnDocument.AFTER,
    )
    if not updated:
        return {"success": False, "error": "concurrent_decision_or_missing"}
    updated.pop("_id", None)
    record_security_event(
        "approval_approved" if approve else "approval_rejected",
        user_id=updated.get("user_id"),
        project_id=updated.get("project_id"),
        tenant_id=updated.get("tenant_id"),
        tool_name=updated.get("tool_name"),
        approval_id=approval_id,
        detail={"reviewer": reviewer_user_id},
    )
    return {"success": True, "approval": updated}


def revoke_approval(
    *,
    approval_id: str,
    actor_user_id: str,
) -> dict[str, Any]:
    col = _get_collection()
    if col is None:
        return {"success": False, "error": "approval_store_unavailable"}
    doc = get_approval(approval_id)
    if not doc:
        return {"success": False, "error": "approval_not_found"}
    if actor_user_id != doc["user_id"]:
        return {"success": False, "error": "unauthorized_revoker"}
    status = ApprovalStatus(doc["status"])
    if not _transition_allowed(status, ApprovalStatus.REVOKED):
        return {"success": False, "error": "cannot_revoke_from_status", "status": status.value}

    updated = col.find_one_and_update(
        {
            "approval_id": approval_id,
            "status": {"$in": [ApprovalStatus.PENDING.value, ApprovalStatus.APPROVED.value]},
        },
        {
            "$set": {
                "status": ApprovalStatus.REVOKED.value,
                "revoked_at": _now(),
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    if not updated:
        return {"success": False, "error": "revoke_race_or_invalid"}
    updated.pop("_id", None)
    record_security_event(
        "approval_revoked",
        user_id=updated.get("user_id"),
        approval_id=approval_id,
        tool_name=updated.get("tool_name"),
    )
    return {"success": True, "approval": updated}


def cancel_approval(
    *,
    approval_id: str,
    actor_user_id: str,
) -> dict[str, Any]:
    col = _get_collection()
    if col is None:
        return {"success": False, "error": "approval_store_unavailable"}
    doc = get_approval(approval_id)
    if not doc:
        return {"success": False, "error": "approval_not_found"}
    if actor_user_id != doc["user_id"]:
        return {"success": False, "error": "unauthorized"}
    status = ApprovalStatus(doc["status"])
    if not _transition_allowed(status, ApprovalStatus.CANCELLED):
        return {"success": False, "error": "cannot_cancel", "status": status.value}
    updated = col.find_one_and_update(
        {
            "approval_id": approval_id,
            "status": {"$in": [ApprovalStatus.PENDING.value, ApprovalStatus.APPROVED.value]},
        },
        {"$set": {"status": ApprovalStatus.CANCELLED.value}},
        return_document=ReturnDocument.AFTER,
    )
    if not updated:
        return {"success": False, "error": "cancel_race"}
    updated.pop("_id", None)
    return {"success": True, "approval": updated}


def consume_approval_for_execution(
    *,
    approval_id: str,
    user_id: str,
    tool_name: str,
    arguments: dict[str, Any] | None,
    project_id: str | None = None,
    tenant_id: str | None = None,
    expected_policy_version: str | None = None,
) -> dict[str, Any]:
    """Atomically consume an APPROVED approval bound to exact invocation.

    Returns ``{success: True, approval: ...}`` or a structured failure.
    Concurrent callers: at most one succeeds.
    """
    record_security_event(
        "approval_consume_attempt",
        user_id=user_id,
        project_id=project_id,
        tenant_id=tenant_id,
        tool_name=tool_name,
        approval_id=approval_id,
    )
    col = _get_collection()
    if col is None:
        record_security_event(
            "approval_consume_failed",
            user_id=user_id,
            approval_id=approval_id,
            tool_name=tool_name,
            detail={"reason": "store_unavailable"},
        )
        return {
            "success": False,
            "error": "approval_store_unavailable",
            "error_code": "APPROVAL_STORE_UNAVAILABLE",
        }

    doc = get_approval(approval_id)
    if not doc:
        record_security_event(
            "approval_consume_failed",
            user_id=user_id,
            approval_id=approval_id,
            tool_name=tool_name,
            detail={"reason": "not_found"},
        )
        return {"success": False, "error": "approval_not_found"}

    doc = _expire_if_needed(doc)
    if doc["status"] == ApprovalStatus.CONSUMED.value:
        record_security_event(
            "approval_replay",
            user_id=user_id,
            approval_id=approval_id,
            tool_name=tool_name,
        )
        return {
            "success": False,
            "error": "approval_already_consumed",
            "error_code": "APPROVAL_REPLAY",
        }
    if doc["status"] != ApprovalStatus.APPROVED.value:
        record_security_event(
            "approval_consume_failed",
            user_id=user_id,
            approval_id=approval_id,
            tool_name=tool_name,
            detail={"reason": "bad_status", "status": doc["status"]},
        )
        return {
            "success": False,
            "error": "approval_not_approved",
            "status": doc["status"],
            "error_code": "APPROVAL_INVALID_STATUS",
        }

    # Binding checks before atomic consume (re-checked via filter).
    digest = arguments_digest(arguments)
    policy_version = expected_policy_version or POLICY_VERSION
    if doc.get("user_id") != user_id:
        return {"success": False, "error": "approval_user_mismatch", "error_code": "SCOPE_MISMATCH"}
    if (doc.get("project_id") or None) != (project_id or None):
        record_security_event(
            "scope_mismatch",
            user_id=user_id,
            approval_id=approval_id,
            detail={"field": "project_id"},
        )
        return {"success": False, "error": "approval_project_mismatch", "error_code": "SCOPE_MISMATCH"}
    if (doc.get("tenant_id") or None) != (tenant_id or None):
        record_security_event(
            "scope_mismatch",
            user_id=user_id,
            approval_id=approval_id,
            detail={"field": "tenant_id"},
        )
        return {"success": False, "error": "approval_tenant_mismatch", "error_code": "SCOPE_MISMATCH"}
    if doc.get("tool_name") != tool_name:
        return {"success": False, "error": "approval_tool_mismatch"}
    if doc.get("arguments_digest") != digest:
        return {"success": False, "error": "approval_arguments_mismatch"}
    if doc.get("policy_version") != policy_version:
        return {
            "success": False,
            "error": "approval_policy_version_mismatch",
            "error_code": "POLICY_CHANGED",
        }

    now = _now()
    with _consume_lock:
        updated = col.find_one_and_update(
            {
                "approval_id": approval_id,
                "status": ApprovalStatus.APPROVED.value,
                "user_id": user_id,
                "tool_name": tool_name,
                "arguments_digest": digest,
                "policy_version": policy_version,
                "expires_at": {"$gt": now},
                # Re-bind scope atomically so a mutated approval record cannot
                # satisfy an earlier non-atomic project/tenant check.
                "project_id": project_id,
                "tenant_id": tenant_id,
            },
            {
                "$set": {
                    "status": ApprovalStatus.CONSUMED.value,
                    "consumed_at": now,
                }
            },
            return_document=ReturnDocument.AFTER,
        )

    if not updated:
        # Distinguish replay vs race.
        again = get_approval(approval_id) or {}
        if again.get("status") == ApprovalStatus.CONSUMED.value:
            record_security_event(
                "approval_replay",
                user_id=user_id,
                approval_id=approval_id,
                tool_name=tool_name,
            )
            return {
                "success": False,
                "error": "approval_already_consumed",
                "error_code": "APPROVAL_REPLAY",
            }
        record_security_event(
            "approval_consume_failed",
            user_id=user_id,
            approval_id=approval_id,
            tool_name=tool_name,
            detail={"reason": "atomic_consume_failed", "status": again.get("status")},
        )
        return {
            "success": False,
            "error": "approval_consume_failed",
            "error_code": "APPROVAL_CONSUME_FAILED",
            "status": again.get("status"),
        }

    updated.pop("_id", None)
    record_security_event(
        "approval_consume_success",
        user_id=user_id,
        project_id=project_id,
        tenant_id=tenant_id,
        tool_name=tool_name,
        approval_id=approval_id,
    )
    return {"success": True, "approval": updated}


def mark_execution_failed(approval_id: str) -> None:
    col = _get_collection()
    if col is None:
        return
    col.find_one_and_update(
        {
            "approval_id": approval_id,
            "status": ApprovalStatus.CONSUMED.value,
        },
        {"$set": {"status": ApprovalStatus.EXECUTION_FAILED.value}},
    )
    record_security_event(
        "execution_failed",
        approval_id=approval_id,
    )


def approval_required_for_tool(tool_name: str) -> bool:
    return tool_requires_approval(tool_name)


def public_pending_result(create_result: dict[str, Any]) -> dict[str, Any]:
    """Structured tool envelope for pending human review (never success)."""
    approval = create_result.get("approval") or {}
    return {
        "success": False,
        "error": "Human approval required before this tool can execute",
        "error_code": "APPROVAL_REQUIRED",
        "approval_required": True,
        "approval_id": approval.get("approval_id"),
        "approval_status": ApprovalStatus.PENDING.value,
        "risk_class": approval.get("risk_class"),
        "reason": approval.get("reason"),
        "expires_at": approval.get("expires_at"),
        "mutation_committed": False,
        # Explicit: agent must not treat this as completed work.
        "execution_completed": False,
        "pending_human_review": True,
    }
