"""Server-enforced tool permissions (Topic 49).

Deny by default. Explicit deny overrides allow. LLM never grants access.
"""

from __future__ import annotations

import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from app.services.agent_security_audit import record_security_event
from app.services.agent_security_policy import (
    BASELINE_AUTHENTICATED_TOOLS,
    get_tool_risk_meta,
)
from app.validation.agent_security_models import PermissionEffect, PermissionGrant

_COLLECTION_NAME = os.getenv(
    "AGENT_TOOL_PERMISSIONS_COLLECTION",
    "agent_tool_permissions",
)

# When false, only unknown tools / explicit denies are blocked; baseline still
# applies. Default true for security; tests can disable narrowly.
def permissions_enforce_enabled() -> bool:
    return os.getenv("AGENT_TOOL_PERMISSIONS_ENFORCE", "true").lower() in {
        "1",
        "true",
        "yes",
    }


_collection = None
_cache_lock = threading.Lock()
# Short-lived negative/positive cache busted on revoke (process-local only).
_grant_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_CACHE_TTL_SECONDS = float(os.getenv("AGENT_TOOL_PERMISSION_CACHE_TTL", "2"))


def configure_permission_collection(collection) -> None:
    global _collection
    _collection = collection
    invalidate_permission_cache()


def invalidate_permission_cache() -> None:
    with _cache_lock:
        _grant_cache.clear()


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


def ensure_permission_indexes() -> None:
    col = _get_collection()
    if col is None:
        return
    col.create_index(
        [
            ("user_id", 1),
            ("tool_name", 1),
            ("project_id", 1),
            ("tenant_id", 1),
            ("revoked_at", 1),
        ],
        name="tool_perm_lookup",
    )
    col.create_index("grant_id", unique=True, name="tool_perm_grant_id")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _scope_match(
    grant: dict[str, Any],
    *,
    project_id: str | None,
    tenant_id: str | None,
) -> bool:
    g_tenant = grant.get("tenant_id")
    g_project = grant.get("project_id")
    if g_tenant and tenant_id and g_tenant != tenant_id:
        return False
    if g_tenant and not tenant_id:
        return False
    if g_project and project_id and g_project != project_id:
        return False
    if g_project and not project_id:
        return False
    return True


def _resource_match(grant: dict[str, Any], resource: str | None) -> bool:
    scope = grant.get("resource_scope") or "*"
    if scope == "*":
        return True
    if not resource:
        return False
    return resource == scope or resource.startswith(str(scope))


def _load_grants(user_id: str) -> list[dict[str, Any]]:
    now_ts = _now().timestamp()
    with _cache_lock:
        cached = _grant_cache.get(user_id)
        if cached and cached[0] > now_ts:
            return list(cached[1])

    col = _get_collection()
    grants: list[dict[str, Any]] = []
    if col is not None:
        try:
            grants = list(
                col.find(
                    {
                        "user_id": user_id,
                        "revoked_at": None,
                    }
                )
            )
        except Exception:
            grants = []

    with _cache_lock:
        _grant_cache[user_id] = (now_ts + _CACHE_TTL_SECONDS, list(grants))
    return grants


def check_tool_permission(
    *,
    user_id: str | None,
    tool_name: str,
    project_id: str | None = None,
    tenant_id: str | None = None,
    resource: str | None = None,
    action: str = "invoke",
) -> dict[str, Any]:
    """Return ``{allowed: bool, reason: str, effect: ...}``.

    Rules:
    1. No authenticated user → deny
    2. Explicit durable DENY matching scope → deny
    3. Explicit durable ALLOW matching scope → allow
    4. Baseline authenticated catalog → allow (when enforce on)
    5. Otherwise → deny (fail closed)
    """
    if not permissions_enforce_enabled():
        # Soft mode: still deny unknown tools outside baseline.
        if tool_name in BASELINE_AUTHENTICATED_TOOLS:
            return {"allowed": True, "reason": "baseline_soft", "effect": "allow"}
        return {
            "allowed": False,
            "reason": "unknown_tool_denied",
            "effect": "deny",
        }

    meta = get_tool_risk_meta(tool_name)

    # Preserve pre-Topic-49 CLI/unit behavior for non-memory baseline tools.
    # Memory / mutating / unknown tools still require authenticated identity.
    _ANON_SAFE = frozenset({
        "calculator",
        "get_current_time",
        "read_file",
        "search_code",
        "web_search",
    })
    if not user_id:
        if tool_name in _ANON_SAFE and tool_name in BASELINE_AUTHENTICATED_TOOLS:
            return {
                "allowed": True,
                "reason": "baseline_anonymous_safe",
                "effect": "allow",
            }
        record_security_event(
            "permission_denied",
            tool_name=tool_name,
            detail={"reason": "missing_user"},
        )
        return {
            "allowed": False,
            "reason": "missing_authenticated_user",
            "effect": "deny",
        }
    grants = _load_grants(user_id)
    now = _now()

    matching = []
    for g in grants:
        if g.get("tool_name") not in {tool_name, "*"}:
            continue
        if g.get("action", "invoke") not in {action, "*"}:
            continue
        if not _scope_match(g, project_id=project_id, tenant_id=tenant_id):
            continue
        if not _resource_match(g, resource):
            continue
        expires = g.get("expires_at")
        if expires is not None:
            if getattr(expires, "tzinfo", None) is None:
                expires = expires.replace(tzinfo=timezone.utc)
            if expires <= now:
                continue
        matching.append(g)

    # Explicit deny wins.
    for g in matching:
        if g.get("effect") == PermissionEffect.DENY.value:
            record_security_event(
                "permission_denied",
                user_id=user_id,
                project_id=project_id,
                tenant_id=tenant_id,
                tool_name=tool_name,
                detail={"reason": "explicit_deny", "grant_id": g.get("grant_id")},
            )
            return {
                "allowed": False,
                "reason": "explicit_deny",
                "effect": "deny",
                "grant_id": g.get("grant_id"),
            }

    for g in matching:
        if g.get("effect") == PermissionEffect.ALLOW.value:
            record_security_event(
                "permission_allowed",
                user_id=user_id,
                project_id=project_id,
                tenant_id=tenant_id,
                tool_name=tool_name,
                detail={"reason": "explicit_allow", "grant_id": g.get("grant_id")},
            )
            return {
                "allowed": True,
                "reason": "explicit_allow",
                "effect": "allow",
                "grant_id": g.get("grant_id"),
            }

    if tool_name in BASELINE_AUTHENTICATED_TOOLS:
        record_security_event(
            "permission_allowed",
            user_id=user_id,
            project_id=project_id,
            tenant_id=tenant_id,
            tool_name=tool_name,
            detail={"reason": "baseline_authenticated", "risk": meta.risk_class.value},
        )
        return {
            "allowed": True,
            "reason": "baseline_authenticated",
            "effect": "allow",
        }

    record_security_event(
        "permission_denied",
        user_id=user_id,
        project_id=project_id,
        tenant_id=tenant_id,
        tool_name=tool_name,
        detail={"reason": "deny_by_default"},
    )
    return {"allowed": False, "reason": "deny_by_default", "effect": "deny"}


def grant_tool_permission(
    *,
    actor_user_id: str,
    target_user_id: str,
    tool_name: str,
    effect: PermissionEffect | str = PermissionEffect.ALLOW,
    project_id: str | None = None,
    tenant_id: str | None = None,
    resource_scope: str = "*",
    action: str = "invoke",
    expires_at: datetime | None = None,
    admin_user_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Admin-only grant/deny. Users cannot self-grant arbitrary tools."""
    admins = admin_user_ids or {
        x.strip()
        for x in os.getenv("AGENT_PERMISSION_ADMIN_USER_IDS", "").split(",")
        if x.strip()
    }
    if actor_user_id not in admins:
        return {
            "success": False,
            "error": "not_authorized_to_grant_permissions",
            "error_code": "PERMISSION_ADMIN_REQUIRED",
        }
    if target_user_id == actor_user_id and tool_name not in BASELINE_AUTHENTICATED_TOOLS:
        # Still require admin for non-baseline; admins may grant themselves.
        pass

    effect_val = (
        effect.value if isinstance(effect, PermissionEffect) else str(effect)
    )
    if effect_val not in {PermissionEffect.ALLOW.value, PermissionEffect.DENY.value}:
        return {"success": False, "error": "invalid_effect"}

    grant_id = str(uuid.uuid4())
    now = _now()
    doc = {
        "grant_id": grant_id,
        "effect": effect_val,
        "user_id": target_user_id,
        "project_id": project_id,
        "tenant_id": tenant_id,
        "tool_name": tool_name,
        "action": action,
        "resource_scope": resource_scope,
        "expires_at": expires_at,
        "source": f"admin:{actor_user_id}",
        "created_at": now,
        "revoked_at": None,
    }
    col = _get_collection()
    if col is None:
        return {"success": False, "error": "permission_store_unavailable"}
    col.insert_one(doc)
    invalidate_permission_cache()
    record_security_event(
        "permission_allowed" if effect_val == "allow" else "permission_denied",
        user_id=target_user_id,
        project_id=project_id,
        tenant_id=tenant_id,
        tool_name=tool_name,
        detail={"admin": actor_user_id, "grant_id": grant_id, "op": "grant"},
    )
    return {"success": True, "grant": PermissionGrant(**doc).model_dump()}


def revoke_tool_permission(
    *,
    actor_user_id: str,
    grant_id: str,
    admin_user_ids: set[str] | None = None,
) -> dict[str, Any]:
    admins = admin_user_ids or {
        x.strip()
        for x in os.getenv("AGENT_PERMISSION_ADMIN_USER_IDS", "").split(",")
        if x.strip()
    }
    if actor_user_id not in admins:
        return {
            "success": False,
            "error": "not_authorized_to_revoke_permissions",
            "error_code": "PERMISSION_ADMIN_REQUIRED",
        }
    col = _get_collection()
    if col is None:
        return {"success": False, "error": "permission_store_unavailable"}
    result = col.find_one_and_update(
        {"grant_id": grant_id, "revoked_at": None},
        {"$set": {"revoked_at": _now()}},
        return_document=True,
    )
    invalidate_permission_cache()
    if not result:
        return {"success": False, "error": "grant_not_found_or_already_revoked"}
    record_security_event(
        "permission_denied",
        user_id=result.get("user_id"),
        project_id=result.get("project_id"),
        tenant_id=result.get("tenant_id"),
        tool_name=result.get("tool_name"),
        detail={"admin": actor_user_id, "grant_id": grant_id, "op": "revoke"},
    )
    return {"success": True, "grant_id": grant_id}
