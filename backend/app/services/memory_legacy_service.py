"""Read-only helpers for legacy / ownerless memory documents (no auto-mutation)."""

from __future__ import annotations

from typing import Any

from app.services.memory_permission_service import memory_owner_id


def audit_orphan_memory_counts(
    memory_collection,
    *,
    sample_limit: int = 50,
) -> dict[str, Any]:
    """Count documents missing ownership or scope without logging content."""
    missing_owner_query = {
        "$or": [
            {"user_id": {"$exists": False}},
            {"user_id": None},
            {"user_id": ""},
        ],
        "$and": [
            {
                "$or": [
                    {"owner_id": {"$exists": False}},
                    {"owner_id": None},
                    {"owner_id": ""},
                ]
            }
        ],
    }
    missing_tenant = {
        "tenant_id": {"$exists": False},
        "user_id": {"$ne": None},
    }
    missing_project_tagged = {
        "project_id": {"$exists": True, "$nin": [None, ""]},
        "scope": {"$ne": "project"},
    }

    return {
        "success": True,
        "counts": {
            "missing_owner": memory_collection.count_documents(missing_owner_query),
            "missing_tenant_field": memory_collection.count_documents(missing_tenant),
            "inconsistent_project_scope": memory_collection.count_documents(
                missing_project_tagged
            ),
        },
        "sample_ids_missing_owner": [
            str(d["_id"])
            for d in memory_collection.find(
                missing_owner_query, {"_id": 1}
            ).limit(sample_limit)
        ],
        "policy": (
            "Do not assign orphans to arbitrary users. Deny access via "
            "memory_permission_service; quarantine manually if needed."
        ),
    }


def is_quarantined_candidate(doc: dict | None) -> bool:
    """True when the document must not be served until manually repaired."""
    if not isinstance(doc, dict):
        return True
    return memory_owner_id(doc) is None
