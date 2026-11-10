"""Topic 20 — Memory scoping (user / project / tenant / session).

Missing scope fields on legacy documents are treated as user-scoped defaults.
A missing ownership or tenant value never grants access to another user's data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

# Optional Redis for active project/session scope. Failures fall back to
# in-process defaults so unit tests without Redis still work.
try:
    from app.services.temporary_memory_service import redis as _redis
except Exception:  # pragma: no cover
    _redis = None


VALID_SCOPES = frozenset({"user", "project", "shared"})


@dataclass(frozen=True)
class MemoryScopeContext:
    """Authenticated request scope for memory operations."""

    user_id: str
    tenant_id: str | None = None
    project_id: str | None = None

    def normalized(self) -> "MemoryScopeContext":
        uid = str(self.user_id or "").strip()
        if not uid:
            raise ValueError("user_id is required for memory scope")
        tenant = _clean_id(self.tenant_id)
        project = _clean_id(self.project_id)
        return MemoryScopeContext(
            user_id=uid,
            tenant_id=tenant,
            project_id=project,
        )


def _clean_id(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def validate_scope_access(
    *,
    user_id: str,
    tenant_id: str | None = None,
    project_id: str | None = None,
) -> dict:
    """Validate that a scope context is well-formed.

    Does not grant cross-user access. Returns a normalized context or an error.
    """
    try:
        ctx = MemoryScopeContext(
            user_id=user_id,
            tenant_id=tenant_id,
            project_id=project_id,
        ).normalized()
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    if project_id is not None and ctx.project_id is None:
        return {"success": False, "error": "Invalid project_id"}
    if tenant_id is not None and ctx.tenant_id is None:
        return {"success": False, "error": "Invalid tenant_id"}

    return {
        "success": True,
        "user_id": ctx.user_id,
        "tenant_id": ctx.tenant_id,
        "project_id": ctx.project_id,
        "scope": "project" if ctx.project_id else "user",
    }


def resolve_memory_scope(
    *,
    project_id: str | None = None,
    scope: str | None = None,
) -> str:
    """Choose a scope label for a new memory document."""
    if scope in VALID_SCOPES:
        if scope == "project" and not project_id:
            return "user"
        return scope
    if project_id:
        return "project"
    return "user"


def _scope_field_clause(field: str, value: str | None) -> dict:
    """Match either an exact scope id or legacy missing/null for unset queries.

    When ``value`` is None the caller asked for unscoped/user-level records only.
    Documents tagged with a different non-null id must never match.
    """
    if value is None:
        return {
            "$or": [
                {field: None},
                {field: {"$exists": False}},
                {field: ""},
            ]
        }
    return {field: value}


def build_scope_filter(
    *,
    user_id: str,
    tenant_id: str | None = None,
    project_id: str | None = None,
    include_shared_with_user: bool = False,
) -> dict:
    """Mongo filter enforcing user + tenant + project isolation.

    By default only the caller's own documents match (``user_id`` equality).
    When ``include_shared_with_user`` is True, also include documents that list
    this user in ``shared_with`` (still tenant/project scoped).
    """
    ctx = MemoryScopeContext(
        user_id=user_id,
        tenant_id=tenant_id,
        project_id=project_id,
    ).normalized()

    ownership: dict[str, Any]
    if include_shared_with_user:
        ownership = {
            "$or": [
                {"user_id": ctx.user_id},
                {"shared_with": ctx.user_id},
                {"owner_id": ctx.user_id},
            ]
        }
    else:
        ownership = {"user_id": ctx.user_id}

    clauses = [
        ownership,
        _scope_field_clause("tenant_id", ctx.tenant_id),
        _scope_field_clause("project_id", ctx.project_id),
    ]
    query: dict[str, Any] = {"$and": clauses}
    # Keep a top-level user_id for owned-only queries so existing callers /
    # tests that assert query["user_id"] continue to work.
    if not include_shared_with_user:
        query["user_id"] = ctx.user_id
    return query


def build_vector_search_filter(
    *,
    user_id: str,
    tenant_id: str | None = None,
    project_id: str | None = None,
) -> dict:
    """Atlas `$vectorSearch` pre-filter.

    Atlas filter expressions are more limited than full Mongo queries, so this
    uses equality on ``user_id`` plus optional equality on tenant/project.
    Legacy documents without tenant/project fields are handled by a post-filter
    when those query params are None (see ``document_matches_scope``).
    """
    ctx = MemoryScopeContext(
        user_id=user_id,
        tenant_id=tenant_id,
        project_id=project_id,
    ).normalized()
    filt: dict[str, Any] = {"user_id": ctx.user_id}
    if ctx.tenant_id is not None:
        filt["tenant_id"] = ctx.tenant_id
    if ctx.project_id is not None:
        filt["project_id"] = ctx.project_id
    return filt


def document_matches_scope(
    doc: dict | None,
    *,
    user_id: str,
    tenant_id: str | None = None,
    project_id: str | None = None,
    conversation_id: str | None = None,
    allow_shared: bool = True,
) -> bool:
    """True when a stored document belongs in the caller's scope view."""
    if not isinstance(doc, dict):
        return False
    ctx = MemoryScopeContext(
        user_id=user_id,
        tenant_id=tenant_id,
        project_id=project_id,
    ).normalized()

    doc_user = _clean_id(doc.get("user_id"))
    doc_owner = _clean_id(doc.get("owner_id")) or doc_user
    shared = [str(u) for u in (doc.get("shared_with") or []) if u]
    # Missing ownership never matches — do not treat blank ids as public.
    if doc_user is None and doc_owner is None and ctx.user_id not in shared:
        return False
    owns = (doc_user == ctx.user_id) or (doc_owner == ctx.user_id)
    shared_ok = allow_shared and ctx.user_id in shared
    if not (owns or shared_ok):
        return False

    doc_tenant = _clean_id(doc.get("tenant_id"))
    if ctx.tenant_id is None:
        if doc_tenant is not None:
            return False
    elif doc_tenant != ctx.tenant_id:
        return False

    doc_project = _clean_id(doc.get("project_id"))
    if ctx.project_id is None:
        if doc_project is not None:
            return False
    elif doc_project != ctx.project_id:
        return False

    doc_conv = _clean_id(doc.get("conversation_id"))
    if doc_conv is not None:
        ctx_conv = _clean_id(conversation_id)
        if ctx_conv is None or doc_conv != ctx_conv:
            return False

    return True


def filter_documents_by_scope(
    docs: list[dict],
    *,
    user_id: str,
    tenant_id: str | None = None,
    project_id: str | None = None,
    allow_shared: bool = True,
) -> list[dict]:
    return [
        d
        for d in docs
        if document_matches_scope(
            d,
            user_id=user_id,
            tenant_id=tenant_id,
            project_id=project_id,
            allow_shared=allow_shared,
        )
    ]


def _active_scope_key(user_id: str) -> str:
    return f"agent:memory_scope:{user_id}"


def get_active_project_scope(user_id: str) -> dict:
    """Return the session-active project/tenant scope for a user."""
    uid = str(user_id or "").strip()
    if not uid:
        return {"success": False, "error": "user_id is required"}
    payload = {"project_id": None, "tenant_id": None}
    if _redis is not None:
        try:
            raw = _redis.get(_active_scope_key(uid))
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if isinstance(raw, str) and raw.strip():
                data = json.loads(raw)
                if isinstance(data, dict):
                    payload["project_id"] = _clean_id(data.get("project_id"))
                    payload["tenant_id"] = _clean_id(data.get("tenant_id"))
        except Exception:
            pass
    return {"success": True, "user_id": uid, **payload}


def set_active_project_scope(
    user_id: str,
    *,
    project_id: str | None = None,
    tenant_id: str | None = None,
    ttl_seconds: int = 1800,
) -> dict:
    """Persist active project/tenant scope for the user's session."""
    validated = validate_scope_access(
        user_id=user_id,
        project_id=project_id,
        tenant_id=tenant_id,
    )
    if not validated.get("success"):
        return validated
    payload = {
        "project_id": validated.get("project_id"),
        "tenant_id": validated.get("tenant_id"),
    }
    if _redis is not None:
        try:
            _redis.set(
                _active_scope_key(validated["user_id"]),
                json.dumps(payload),
                ex=max(60, int(ttl_seconds)),
            )
        except Exception as exc:
            return {
                "success": False,
                "error": f"Could not persist scope: {exc}",
            }
    return {"success": True, "user_id": validated["user_id"], **payload}


def clear_active_project_scope(user_id: str) -> dict:
    uid = str(user_id or "").strip()
    if not uid:
        return {"success": False, "error": "user_id is required"}
    if _redis is not None:
        try:
            _redis.delete(_active_scope_key(uid))
        except Exception:
            pass
    return {"success": True, "user_id": uid}


def attach_scope_fields(
    *,
    user_id: str,
    tenant_id: str | None = None,
    project_id: str | None = None,
    scope: str | None = None,
) -> dict:
    """Fields to store on a new/updated memory for Topic 20."""
    validated = validate_scope_access(
        user_id=user_id,
        tenant_id=tenant_id,
        project_id=project_id,
    )
    if not validated.get("success"):
        raise ValueError(validated.get("error") or "Invalid scope")
    resolved_scope = resolve_memory_scope(
        project_id=validated.get("project_id"),
        scope=scope,
    )
    return {
        "user_id": validated["user_id"],
        "tenant_id": validated.get("tenant_id"),
        "project_id": validated.get("project_id"),
        "scope": resolved_scope,
    }
