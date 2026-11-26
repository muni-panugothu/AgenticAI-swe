"""Post-retrieval ACL helpers shared across vector, lexical, hybrid, and rerank paths.

Authorization identity always comes from the authenticated server-side caller.
LLM-supplied user/tenant/document IDs are never trusted as ACL decisions.
"""

from __future__ import annotations

from typing import Any, Callable


SameScopeFn = Callable[..., bool]


def authorize_candidates(
    candidates: list[dict[str, Any]],
    *,
    user_id: str,
    tenant_id: str | None,
    project_id: str | None,
    allowed_docs: set[str],
    same_scope: SameScopeFn,
) -> list[dict[str, Any]]:
    """Drop candidates that fail document allow-list or ``_same_scope``.

    Ready-status is enforced at query time; rows without status are rejected here
    so fused/reranked/rewritten paths cannot broaden access.
    """
    if not candidates or not allowed_docs:
        return []
    authorized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in candidates:
        cid = str(row.get("chunk_id") or "")
        if not cid or cid in seen:
            continue
        doc_id = str(row.get("document_id") or "")
        if not doc_id or doc_id not in allowed_docs:
            continue
        status = row.get("status")
        # Fail closed: missing/unknown status must not broaden access after
        # fusion/rerank/rewrite paths that may omit readiness metadata.
        if status != "ready":
            continue
        if not same_scope(row, user_id=user_id, tenant_id=tenant_id, project_id=project_id):
            continue
        seen.add(cid)
        authorized.append(row)
    return authorized
