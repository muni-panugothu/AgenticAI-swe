"""Topic 21 — Memory ownership and permission checks.

Enforced at the data-access layer. Missing permission fields default to
private ownership by ``user_id`` / ``owner_id`` so legacy documents keep
the same isolation behavior.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from app.services.memory_audit_service import record_audit_event
from app.services.memory_scope_service import _clean_id

PermissionAction = Literal["read", "write"]
VALID_PERMISSION_LEVELS = frozenset({
    "private",
    "project_read",
    "project_write",
    "public_read",
})


def memory_owner_id(doc: dict | None) -> str | None:
    if not isinstance(doc, dict):
        return None
    return _clean_id(doc.get("owner_id")) or _clean_id(doc.get("user_id"))


def normalize_permission_level(value: str | None) -> str:
    level = str(value or "private").strip().lower()
    if level not in VALID_PERMISSION_LEVELS:
        return "private"
    return level


def default_permission_fields(
    *,
    user_id: str,
    owner_id: str | None = None,
    shared_with: list[str] | None = None,
    permission_level: str | None = None,
) -> dict:
    """Permission fields for new memories (backward-compatible defaults)."""
    owner = _clean_id(owner_id) or str(user_id).strip()
    shared = sorted({
        str(u).strip()
        for u in (shared_with or [])
        if str(u).strip() and str(u).strip() != owner
    })
    return {
        "owner_id": owner,
        "shared_with": shared,
        "permission_level": normalize_permission_level(permission_level),
    }


def _same_project(doc: dict, *, project_id: str | None) -> bool:
    doc_project = _clean_id(doc.get("project_id"))
    req_project = _clean_id(project_id)
    if req_project is None or doc_project is None:
        return False
    return doc_project == req_project


def _same_tenant(doc: dict, *, tenant_id: str | None) -> bool:
    doc_tenant = _clean_id(doc.get("tenant_id"))
    req_tenant = _clean_id(tenant_id)
    if req_tenant is None and doc_tenant is None:
        return True
    if req_tenant is None or doc_tenant is None:
        return False
    return doc_tenant == req_tenant


def check_read_permission(
    user_id: str,
    memory_doc: dict | None,
    *,
    tenant_id: str | None = None,
    project_id: str | None = None,
    conversation_id: str | None = None,
    audit: bool = False,
    operation_id: str | None = None,
) -> dict:
    """Return whether ``user_id`` may read ``memory_doc``."""
    return _check_permission(
        user_id,
        memory_doc,
        action="read",
        tenant_id=tenant_id,
        project_id=project_id,
        conversation_id=conversation_id,
        audit=audit,
        operation_id=operation_id,
    )


def check_write_permission(
    user_id: str,
    memory_doc: dict | None,
    *,
    tenant_id: str | None = None,
    project_id: str | None = None,
    conversation_id: str | None = None,
    audit: bool = False,
    operation_id: str | None = None,
) -> dict:
    """Return whether ``user_id`` may mutate ``memory_doc``."""
    return _check_permission(
        user_id,
        memory_doc,
        action="write",
        tenant_id=tenant_id,
        project_id=project_id,
        conversation_id=conversation_id,
        audit=audit,
        operation_id=operation_id,
    )


def _check_permission(
    user_id: str,
    memory_doc: dict | None,
    *,
    action: PermissionAction,
    tenant_id: str | None = None,
    project_id: str | None = None,
    conversation_id: str | None = None,
    audit: bool = False,
    operation_id: str | None = None,
) -> dict:
    uid = str(user_id or "").strip()
    if not uid:
        return _deny(
            user_id="",
            memory_doc=memory_doc,
            action=action,
            reason="user_id is required",
            audit=audit,
            operation_id=operation_id,
        )
    if not isinstance(memory_doc, dict):
        return _deny(
            user_id=uid,
            memory_doc=memory_doc,
            action=action,
            reason="Memory not found",
            audit=audit,
            operation_id=operation_id,
        )

    doc_conv = _clean_id(memory_doc.get("conversation_id"))
    if doc_conv is not None:
        req_conv = _clean_id(conversation_id)
        if req_conv is None or doc_conv != req_conv:
            return _deny(
                user_id=uid,
                memory_doc=memory_doc,
                action=action,
                reason="Access denied",
                audit=audit,
                operation_id=operation_id,
            )

    # Missing ownership never grants access — including public_read.
    owner = memory_owner_id(memory_doc)
    if owner is None:
        return _deny(
            user_id=uid,
            memory_doc=memory_doc,
            action=action,
            reason="missing ownership",
            audit=audit,
            operation_id=operation_id,
        )

    # Tenant boundary — missing/mismatched tenant never grants cross-tenant access.
    if not _same_tenant(memory_doc, tenant_id=tenant_id):
        return _deny(
            user_id=uid,
            memory_doc=memory_doc,
            action=action,
            reason="Access denied",
            audit=audit,
            operation_id=operation_id,
        )

    doc_project = _clean_id(memory_doc.get("project_id"))
    req_project = _clean_id(project_id)
    # Project-tagged docs are invisible outside that project context.
    if doc_project is not None and doc_project != req_project:
        return _deny(
            user_id=uid,
            memory_doc=memory_doc,
            action=action,
            reason="Access denied",
            audit=audit,
            operation_id=operation_id,
        )
    # Unscoped query must not surface project-scoped docs.
    if req_project is None and doc_project is not None:
        return _deny(
            user_id=uid,
            memory_doc=memory_doc,
            action=action,
            reason="Access denied",
            audit=audit,
            operation_id=operation_id,
        )

    shared = [str(u) for u in (memory_doc.get("shared_with") or []) if u]
    level = normalize_permission_level(memory_doc.get("permission_level"))

    if owner == uid:
        # Owner can always access their docs in matching tenant; project
        # mismatch already rejected above.
        if req_project is not None and doc_project is None:
            # Owner viewing a project context still sees only that project's
            # docs via scope filters; unscoped owner docs stay user-level.
            return _deny(
                user_id=uid,
                memory_doc=memory_doc,
                action=action,
                reason="Access denied",
                audit=audit,
                operation_id=operation_id,
            )
        return {"allowed": True, "reason": "owner", "action": action}

    if action == "read" and uid in shared:
        return {"allowed": True, "reason": "shared_with", "action": action}

    if action == "read" and level == "public_read":
        # public_read requires an explicit matching tenant on both sides.
        # Missing tenant is never treated as "everyone".
        doc_tenant = _clean_id(memory_doc.get("tenant_id"))
        req_tenant = _clean_id(tenant_id)
        if (
            doc_tenant is None
            or req_tenant is None
            or doc_tenant != req_tenant
        ):
            return _deny(
                user_id=uid,
                memory_doc=memory_doc,
                action=action,
                reason="Access denied",
                audit=audit,
                operation_id=operation_id,
            )
        return {"allowed": True, "reason": "public_read", "action": action}

    if level in {"project_read", "project_write"}:
        if _same_project(memory_doc, project_id=project_id):
            if action == "read":
                return {"allowed": True, "reason": level, "action": action}
            if action == "write" and level == "project_write":
                return {"allowed": True, "reason": "project_write", "action": action}

    return _deny(
        user_id=uid,
        memory_doc=memory_doc,
        action=action,
        reason="Access denied",
        audit=audit,
        operation_id=operation_id,
    )


def _deny(
    *,
    user_id: str,
    memory_doc: dict | None,
    action: PermissionAction,
    reason: str,
    audit: bool,
    operation_id: str | None,
) -> dict:
    memory_id = None
    if isinstance(memory_doc, dict):
        mid = memory_doc.get("memory_id") or memory_doc.get("_id")
        memory_id = str(mid) if mid is not None else None
    if audit and user_id:
        # Never log memory content — only ids and action metadata.
        record_audit_event(
            user_id=user_id,
            event_type="permission_denied",
            memory_id=memory_id,
            operation_id=operation_id,
            details={
                "action": action,
                "reason": reason,
            },
        )
    return {
        "allowed": False,
        "reason": reason,
        "action": action,
        "error": reason,
    }


def filter_readable_documents(
    docs: list[dict],
    *,
    user_id: str,
    tenant_id: str | None = None,
    project_id: str | None = None,
) -> list[dict]:
    readable: list[dict] = []
    for doc in docs:
        result = check_read_permission(
            user_id,
            doc,
            tenant_id=tenant_id,
            project_id=project_id,
            audit=False,
        )
        if result.get("allowed"):
            readable.append(doc)
    return readable


def grant_access(
    *,
    owner_id: str,
    memory_doc: dict,
    target_user_id: str,
    memory_collection,
    operation_id: str | None = None,
) -> dict:
    """Add ``target_user_id`` to ``shared_with`` (owner-only)."""
    owner = str(owner_id or "").strip()
    target = str(target_user_id or "").strip()
    if not owner or not target:
        return {"success": False, "error": "owner_id and target_user_id are required"}
    if owner == target:
        return {"success": False, "error": "Cannot share with the owner"}
    if memory_owner_id(memory_doc) != owner:
        check_write_permission(owner, memory_doc, audit=True, operation_id=operation_id)
        return {"success": False, "error": "Access denied"}

    from bson import ObjectId

    mid = memory_doc.get("_id") or memory_doc.get("memory_id")
    if mid is None:
        return {"success": False, "error": "Memory not found"}
    result = memory_collection.update_one(
        {"_id": ObjectId(str(mid)), "user_id": memory_doc.get("user_id")},
        {
            "$addToSet": {"shared_with": target},
            "$set": {"updated_at": datetime.now(timezone.utc)},
        },
    )
    if result.matched_count == 0:
        return {"success": False, "error": "Memory not found"}
    record_audit_event(
        user_id=owner,
        event_type="permission_granted",
        memory_id=str(mid),
        operation_id=operation_id,
        details={"target_user_id": target, "access": "read"},
    )
    return {
        "success": True,
        "memory_id": str(mid),
        "shared_with_added": target,
    }


def revoke_access(
    *,
    owner_id: str,
    memory_doc: dict,
    target_user_id: str,
    memory_collection,
    operation_id: str | None = None,
) -> dict:
    """Remove ``target_user_id`` from ``shared_with`` (owner-only)."""
    owner = str(owner_id or "").strip()
    target = str(target_user_id or "").strip()
    if not owner or not target:
        return {"success": False, "error": "owner_id and target_user_id are required"}
    if memory_owner_id(memory_doc) != owner:
        return {"success": False, "error": "Access denied"}

    from bson import ObjectId

    mid = memory_doc.get("_id") or memory_doc.get("memory_id")
    if mid is None:
        return {"success": False, "error": "Memory not found"}
    result = memory_collection.update_one(
        {"_id": ObjectId(str(mid)), "user_id": memory_doc.get("user_id")},
        {
            "$pull": {"shared_with": target},
            "$set": {"updated_at": datetime.now(timezone.utc)},
        },
    )
    if result.matched_count == 0:
        return {"success": False, "error": "Memory not found"}
    record_audit_event(
        user_id=owner,
        event_type="permission_revoked",
        memory_id=str(mid),
        operation_id=operation_id,
        details={"target_user_id": target, "access": "read"},
    )
    return {
        "success": True,
        "memory_id": str(mid),
        "shared_with_removed": target,
    }


def require_write_or_error(
    user_id: str,
    memory_doc: dict | None,
    *,
    tenant_id: str | None = None,
    project_id: str | None = None,
    operation_id: str | None = None,
) -> dict | None:
    """Return an error dict when write is denied; otherwise None."""
    result = check_write_permission(
        user_id,
        memory_doc,
        tenant_id=tenant_id,
        project_id=project_id,
        audit=True,
        operation_id=operation_id,
    )
    if result.get("allowed"):
        return None
    # Generic message — do not leak private memory content.
    return {
        "success": False,
        "error": "Memory not found for this user",
        "error_code": "permission_denied",
    }
