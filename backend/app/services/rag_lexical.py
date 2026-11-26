"""Lexical retrieval backends: Atlas Search BM25 and token-overlap fallback.

BM25 is never silently claimed when the overlap fallback is used. Callers must
surface ``lexical_backend`` / ``lexical_fallback`` diagnostics.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Callable

from app.services.rag_config import settings

logger = logging.getLogger("agentic.rag")
_TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)


class LexicalSearchError(RuntimeError):
    """Atlas Search BM25 unavailable or misconfigured."""


def tokens(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(text) if len(token) > 1]


def verify_search_index(collection: Any) -> dict[str, Any]:
    """Fail-closed readiness check for the Atlas Search (BM25) index."""
    try:
        rows = list(collection.aggregate([{"$listSearchIndexes": {}}]))
    except Exception as exc:
        raise LexicalSearchError(f"Atlas Search index inspection failed ({type(exc).__name__})") from exc
    index = next(
        (
            row for row in rows
            if isinstance(row, dict) and row.get("name") == settings.search_index
        ),
        None,
    )
    if not index:
        raise LexicalSearchError("Configured RAG search index is missing")
    index_type = str(index.get("type") or "").lower()
    if index_type == "vectorsearch":
        raise LexicalSearchError("Configured RAG search index is a vector index, not a text search index")
    status = str(index.get("status") or "").upper()
    queryable = index.get("queryable")
    ready = status == "READY" or queryable is True
    if not ready:
        raise LexicalSearchError("Configured RAG search index is not READY/queryable")
    return {"name": settings.search_index, "status": status or "READY", "backend": "bm25"}


def _acl_filter_clauses(
    *,
    tenant_id: str | None,
    project_id: str | None,
    allowed_docs: set[str],
    allowed_owners: list[str] | None,
) -> list[dict[str, Any]]:
    clauses: list[dict[str, Any]] = [
        {"equals": {"path": "status", "value": "ready"}},
        {"in": {"path": "document_id", "value": sorted(allowed_docs)}},
    ]
    if allowed_owners:
        clauses.append({"in": {"path": "user_id", "value": list(allowed_owners)}})
    if tenant_id is not None:
        clauses.append({"equals": {"path": "tenant_id", "value": tenant_id}})
    # project_id may be null; Atlas equals supports null.
    clauses.append({"equals": {"path": "project_id", "value": project_id}})
    return clauses


def bm25_candidates(
    query: str,
    *,
    collection: Any,
    user_id: str,
    tenant_id: str | None,
    project_id: str | None,
    allowed_docs: set[str],
    allowed_owners: list[str] | None,
    same_scope: Callable[..., bool],
    candidate_fn: Callable[..., dict[str, Any]],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Atlas Search text retrieval with ACL filters; raises on index failure."""
    if not query.strip() or not allowed_docs:
        return []
    verify_search_index(collection)
    limit = limit or settings.candidate_limit
    search_stage = {
        "$search": {
            "index": settings.search_index,
            "compound": {
                "must": [{"text": {"query": query, "path": "text"}}],
                "filter": _acl_filter_clauses(
                    tenant_id=tenant_id,
                    project_id=project_id,
                    allowed_docs=allowed_docs,
                    allowed_owners=allowed_owners,
                ),
            },
        }
    }
    pipeline = [
        search_stage,
        {"$limit": limit},
        {
            "$project": {
                "embedding": 0,
                "score": {"$meta": "searchScore"},
            }
        },
    ]
    try:
        rows = list(collection.aggregate(pipeline, maxTimeMS=5000))
    except Exception as exc:
        raise LexicalSearchError(f"Atlas Search BM25 query failed ({type(exc).__name__})") from exc
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        doc_id = str(row.get("document_id") or "")
        if doc_id not in allowed_docs:
            continue
        if not same_scope(row, user_id=user_id, tenant_id=tenant_id, project_id=project_id):
            continue
        score = float(row.get("score") or 0.0)
        cand = candidate_fn(row, score, "bm25")
        cid = cand["chunk_id"]
        if cid in seen:
            continue
        seen.add(cid)
        results.append(cand)
    return results


def overlap_candidates(
    query: str,
    *,
    collection: Any,
    user_id: str,
    tenant_id: str | None,
    project_id: str | None,
    allowed_docs: set[str],
    same_scope: Callable[..., bool],
    candidate_fn: Callable[..., dict[str, Any]],
) -> list[dict[str, Any]]:
    """Token-overlap lexical fallback (not BM25)."""
    terms = set(tokens(query))
    if not terms or not allowed_docs:
        return []
    mongo_query: dict[str, Any] = {
        "status": "ready",
        "document_id": {"$in": list(allowed_docs)},
        "tenant_id": tenant_id,
        "project_id": project_id,
    }
    rows = collection.find(mongo_query, {"embedding": 0}).limit(settings.lexical_scan_limit)
    scored: list[tuple[dict[str, Any], float]] = []
    for row in rows:
        if not same_scope(row, user_id=user_id, tenant_id=tenant_id, project_id=project_id):
            continue
        words = tokens(row.get("text", ""))
        overlap = sum(1 for word in words if word in terms)
        if overlap:
            score = overlap / max(1.0, math.sqrt(len(words)) * math.sqrt(len(terms)))
            scored.append((candidate_fn(row, score, "lexical"), score))
    scored.sort(key=lambda pair: (-pair[1], pair[0]["chunk_id"]))
    return [row for row, _score in scored[: settings.candidate_limit]]
