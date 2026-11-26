import atexit
import os
import time
from datetime import datetime, timezone
from uuid import uuid4

from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import (
    ConnectionFailure,
    OperationFailure,
    PyMongoError,
)

from app.schemas.memory_schema import MemoryDocument
from app.services.embedding_service import create_embedding
from app.services.memory_audit_service import (
    configure_audit_collection,
    record_audit_event,
)
from app.services.memory_lifecycle_service import (
    annotate_memories_with_lifecycle,
    calculate_memory_freshness_score,
    calculate_memory_priority_score,
)
from app.services.memory_permission_service import (
    check_read_permission,
    default_permission_fields,
    filter_readable_documents,
    memory_owner_id,
    require_write_or_error,
)
from app.services.memory_scope_service import (
    attach_scope_fields,
    build_scope_filter,
    build_vector_search_filter,
    document_matches_scope,
    filter_documents_by_scope,
)
from app.utils.memory_logging import log_memory_operation


def _begin_durable_commit():
    """Acquire late-worker commit lease immediately before an irreversible write.

    Returns an error dict when the execution was invalidated by timeout/cancel.
    When no execution is bound (direct unit tests), commits are allowed.
    """
    from app.services.execution_lifecycle import (
        commit_rejected_result,
        try_begin_commit,
    )

    if try_begin_commit():
        return None
    return commit_rejected_result()


def _finish_durable_commit(*, success: bool) -> None:
    from app.services.execution_lifecycle import (
        finish_commit_abort,
        finish_commit_success,
    )

    if success:
        finish_commit_success()
    else:
        finish_commit_abort()


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# MONGODB CONFIGURATION
# ============================================================

MONGO_URI = os.getenv("MONGO_URI")

MEMORY_DATABASE = os.getenv(
    "MEMORY_DATABASE",
    "agentic_ai_memory",
)

MEMORY_COLLECTION = os.getenv(
    "MEMORY_COLLECTION",
    "memories_history",
)

MEMORY_AUDIT_COLLECTION = os.getenv(
    "MEMORY_AUDIT_COLLECTION",
    "memory_audit_events",
)

MEMORY_SUMMARY_COLLECTION = os.getenv(
    "MEMORY_SUMMARY_COLLECTION",
    "memory_summaries",
)

TRANSACTION_MAX_RETRIES = int(os.getenv("MEMORY_TRANSACTION_MAX_RETRIES", "3"))


# ============================================================
# VALIDATE MONGO URI
# ============================================================

if not MONGO_URI:
    raise ValueError("MONGO_URI is not configured in .env")


# ============================================================
# MONGODB CLIENT / DATABASE / COLLECTIONS
# ============================================================

client = MongoClient(MONGO_URI)
database = client[MEMORY_DATABASE]
memory_collection = database[MEMORY_COLLECTION]
audit_collection = database[MEMORY_AUDIT_COLLECTION]
summary_collection = database[MEMORY_SUMMARY_COLLECTION]
configure_audit_collection(audit_collection)

# Wire summarization service to the same Mongo client/collections.
from app.services.memory_summary_service import (  # noqa: E402
    configure_summary_collections,
    ensure_summary_indexes,
    schedule_summary_refresh,
    search_memory_summaries,
)

configure_summary_collections(summary_collection, memory_collection)


def close_mongo_client() -> None:
    """Close the shared MongoClient (FastAPI shutdown / process exit)."""
    global client
    try:
        if client is not None:
            client.close()
    except Exception:
        pass


atexit.register(close_mongo_client)


def _memory_log_id(mem: dict) -> str:
    mid = mem.get("memory_id") or mem.get("_id")
    return str(mid) if mid is not None else "unknown"


def _authorize_memory_candidates(
    candidates: list[dict],
    *,
    user_id: str,
    tenant_id: str | None = None,
    project_id: str | None = None,
) -> list[dict]:
    """Drop unauthorized rows before ranking, logging, or LLM context."""
    if not candidates:
        return []
    normalized: list[dict] = []
    for mem in candidates:
        if not isinstance(mem, dict):
            continue
        row = dict(mem)
        _ensure_identity_fields(row, fallback_user_id=None)
        normalized.append(row)
    scoped = filter_documents_by_scope(
        normalized,
        user_id=user_id,
        tenant_id=tenant_id,
        project_id=project_id,
        allow_shared=True,
    )
    return filter_readable_documents(
        scoped,
        user_id=user_id,
        tenant_id=tenant_id,
        project_id=project_id,
    )


def ensure_memory_indexes() -> dict:
    """Create production indexes for user isolation and lifecycle queries."""
    try:
        memory_collection.create_index(
            [("user_id", ASCENDING), ("status", ASCENDING), ("updated_at", DESCENDING)],
            name="user_status_updated_idx",
        )
        memory_collection.create_index(
            [("user_id", ASCENDING), ("fact_key", ASCENDING), ("status", ASCENDING)],
            name="user_fact_status_idx",
        )
        memory_collection.create_index(
            [("user_id", ASCENDING), ("operation_id", ASCENDING)],
            name="user_operation_id_idx",
            unique=True,
            partialFilterExpression={"operation_id": {"$type": "string"}},
        )
        # Topic 20 — tenant / user / project isolation.
        memory_collection.create_index(
            [
                ("tenant_id", ASCENDING),
                ("user_id", ASCENDING),
                ("project_id", ASCENDING),
                ("status", ASCENDING),
            ],
            name="tenant_user_project_status_idx",
        )
        memory_collection.create_index(
            [("owner_id", ASCENDING), ("status", ASCENDING)],
            name="owner_status_idx",
        )
        memory_collection.create_index(
            [("shared_with", ASCENDING), ("status", ASCENDING)],
            name="shared_with_status_idx",
        )
        audit_collection.create_index(
            [("user_id", ASCENDING), ("created_at", DESCENDING)],
            name="audit_user_created_idx",
        )
        audit_collection.create_index(
            [("operation_id", ASCENDING)],
            name="audit_operation_id_idx",
        )
        ensure_summary_indexes()
        return {"success": True}
    except Exception as exc:
        print(f"[Memory index creation failed]: {exc}")
        return {"success": False, "error": str(exc)}


# Best-effort index bootstrap at import time.
ensure_memory_indexes()


# ============================================================
# CHECK DATABASE CONNECTION
# ============================================================

def check_memory_database():
    try:
        client.admin.command("ping")
        print("[MongoDB] Connection successful")
        print(f"[Database] {MEMORY_DATABASE}")
        print(f"[Collection] {MEMORY_COLLECTION}")
        return True
    except Exception as e:
        print(f"[MongoDB] Connection failed: {e}")
        return False


def _active_status_filter():
    return {
        "$or": [
            {"status": "active"},
            {"status": {"$exists": False}},
        ]
    }


def _reviewable_status_filter():
    """Active + needs_review — used for logical duplicate detection."""
    return {
        "$or": [
            {"status": "active"},
            {"status": "needs_review"},
            {"status": {"$exists": False}},
        ]
    }


def _find_by_operation_id(user_id: str, operation_id: str | None):
    if not operation_id:
        return None
    return memory_collection.find_one(
        {"user_id": user_id, "operation_id": operation_id}
    )


def _ensure_identity_fields(doc: dict, *, fallback_user_id: str | None = None) -> dict:
    """Normalize permission defaults without inventing ownership.

    Missing ``user_id`` / ``owner_id`` stay missing so permission checks deny
    access. Never assign the authenticated caller as owner of an orphan doc.
    ``fallback_user_id`` is accepted for call-site compatibility only.
    """
    if not isinstance(doc, dict):
        return doc
    _ = fallback_user_id  # intentionally unused — do not invent ownership
    stored_user = doc.get("user_id")
    if stored_user is not None and str(stored_user).strip() == "":
        doc["user_id"] = None
    if not doc.get("owner_id"):
        clean_user = str(doc.get("user_id") or "").strip()
        if clean_user:
            doc["owner_id"] = clean_user
    if not doc.get("permission_level"):
        doc["permission_level"] = "private"
    if doc.get("shared_with") is None:
        doc["shared_with"] = []
    return doc


def _doc_to_memory_candidate(
    item: dict,
    score: float = 1.0,
    *,
    fallback_user_id: str | None = None,
) -> dict:
    _ensure_identity_fields(item, fallback_user_id=fallback_user_id)
    return {
        "memory_id": str(item["_id"]),
        "memory": item.get("memory", ""),
        "user_id": item.get("user_id"),
        "importance": item.get("importance", 3),
        "confidence": item.get("confidence", 0.75),
        "status": item.get("status", "active"),
        "fact_key": item.get("fact_key"),
        "normalized_value": item.get("normalized_value"),
        "conflict_group_id": item.get("conflict_group_id"),
        "version": item.get("version", 1),
        "score": score,
        "reinforcement_count": item.get("reinforcement_count", 0),
        "last_reinforced_at": item.get("last_reinforced_at"),
        "project_id": item.get("project_id"),
        "tenant_id": item.get("tenant_id"),
        "scope": item.get("scope", "user"),
        "owner_id": item.get("owner_id") or item.get("user_id"),
        "shared_with": list(item.get("shared_with") or []),
        "permission_level": item.get("permission_level") or "private",
    }


# ============================================================
# SAVE MEMORY
# ============================================================

def save_memory(
    user_id,
    memory,
    embedding=None,
    importance=3,
    confidence=0.75,
    status="active",
    fact_key=None,
    normalized_value=None,
    conflict_type="none",
    conflict_confidence=None,
    conflict_group_id=None,
    conflict_with_memory_id=None,
    supersedes_memory_id=None,
    superseded_by_memory_id=None,
    version=1,
    last_accessed_at=None,
    access_count=0,
    last_reinforced_at=None,
    reinforcement_count=None,
    operation_id=None,
    is_stale=False,
    stale_reason=None,
    *,
    project_id=None,
    tenant_id=None,
    scope=None,
    owner_id=None,
    shared_with=None,
    permission_level=None,
):
    try:
        existing = _find_by_operation_id(user_id, operation_id)
        if existing:
            log_memory_operation(
                "save_memory_idempotent",
                user_id,
                success=True,
                memory_id=str(existing["_id"]),
                operation_id=operation_id,
            )
            return {
                "success": True,
                "memory_id": str(existing["_id"]),
                "memory": existing.get("memory", memory),
                "importance": existing.get("importance", importance),
                "status": existing.get("status", status),
                "version": existing.get("version", version),
                "idempotent": True,
            }

        current_time = datetime.now(timezone.utc)
        if reinforcement_count is None:
            reinforcement_count = 0

        scope_fields = attach_scope_fields(
            user_id=user_id,
            tenant_id=tenant_id,
            project_id=project_id,
            scope=scope,
        )
        requested_owner = str(owner_id).strip() if owner_id else None
        uid = str(user_id).strip()
        if requested_owner and requested_owner != uid:
            record_audit_event(
                user_id=uid,
                event_type="ownership_impersonation_blocked",
                operation_id=operation_id,
                details={"requested_owner_id": requested_owner},
            )
        permission_fields = default_permission_fields(
            user_id=uid,
            owner_id=None,
            shared_with=shared_with,
            permission_level=permission_level,
        )

        validated_memory = MemoryDocument(
            user_id=user_id,
            memory=memory,
            embedding=embedding,
            importance=importance,
            confidence=confidence,
            created_at=current_time,
            updated_at=current_time,
            status=status,
            fact_key=fact_key,
            normalized_value=normalized_value,
            conflict_type=conflict_type,
            conflict_confidence=conflict_confidence,
            conflict_group_id=conflict_group_id,
            conflict_with_memory_id=conflict_with_memory_id,
            supersedes_memory_id=supersedes_memory_id,
            superseded_by_memory_id=superseded_by_memory_id,
            version=version,
            last_accessed_at=last_accessed_at,
            access_count=access_count,
            last_reinforced_at=last_reinforced_at,
            reinforcement_count=reinforcement_count,
            is_stale=is_stale,
            stale_reason=stale_reason,
            operation_id=operation_id,
            project_id=scope_fields.get("project_id"),
            tenant_id=scope_fields.get("tenant_id"),
            scope=scope_fields.get("scope") or "user",
            owner_id=permission_fields["owner_id"],
            shared_with=permission_fields["shared_with"],
            permission_level=permission_fields["permission_level"],
        )

        memory_document = validated_memory.model_dump()
        blocked = _begin_durable_commit()
        if blocked is not None:
            return blocked
        try:
            result = memory_collection.insert_one(memory_document)
            memory_id = str(result.inserted_id)
            _finish_durable_commit(success=True)
        except Exception:
            _finish_durable_commit(success=False)
            raise

        record_audit_event(
            user_id=user_id,
            event_type="memory_saved",
            memory_id=memory_id,
            operation_id=operation_id,
            details={
                "status": status,
                "fact_key": fact_key,
                "conflict_type": conflict_type,
                "version": version,
            },
        )

        print(f"[Memory saved] {memory_id}")
        print(f"[Memory status]: {status}")
        print(f"[Memory importance]: {importance}/5")

        schedule_summary_refresh(
            user_id,
            reason="memory_saved",
            memory=memory,
            fact_key=fact_key,
        )

        return {
            "success": True,
            "memory_id": memory_id,
            "memory": memory,
            "importance": importance,
            "status": status,
            "version": version,
            "idempotent": False,
        }

    except Exception as e:
        print(f"[Memory save error]: {e}")
        log_memory_operation(
            "save_memory",
            user_id,
            success=False,
            operation_id=operation_id,
            details={"error": str(e)},
        )
        return {"success": False, "error": str(e)}


# ============================================================
# GET MEMORIES
# ============================================================

def get_memories(
    user_id,
    limit=10,
    *,
    project_id=None,
    tenant_id=None,
):
    try:
        query = {
            "$and": [
                build_scope_filter(
                    user_id=user_id,
                    tenant_id=tenant_id,
                    project_id=project_id,
                    include_shared_with_user=True,
                ),
                _active_status_filter(),
            ]
        }
        memories = memory_collection.find(
            query,
            {
                "_id": 1,
                "memory": 1,
                "user_id": 1,
                "created_at": 1,
                "updated_at": 1,
                "status": 1,
                "version": 1,
                "fact_key": 1,
                "normalized_value": 1,
                "importance": 1,
                "confidence": 1,
                "last_accessed_at": 1,
                "access_count": 1,
                "last_reinforced_at": 1,
                "reinforcement_count": 1,
                "is_stale": 1,
                "stale_reason": 1,
                "project_id": 1,
                "tenant_id": 1,
                "scope": 1,
                "owner_id": 1,
                "shared_with": 1,
                "permission_level": 1,
            },
        ).sort("updated_at", -1).limit(max(limit * 3, limit))

        memory_list = []
        for item in memories:
            _ensure_identity_fields(item, fallback_user_id=user_id)
            memory_list.append({
                "memory_id": str(item["_id"]),
                "memory": item["memory"],
                "user_id": item.get("user_id"),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
                "status": item.get("status", "active"),
                "version": item.get("version", 1),
                "fact_key": item.get("fact_key"),
                "normalized_value": item.get("normalized_value"),
                "importance": item.get("importance", 3),
                "confidence": item.get("confidence", 0.75),
                "last_accessed_at": item.get("last_accessed_at"),
                "access_count": item.get("access_count", 0),
                "last_reinforced_at": item.get("last_reinforced_at"),
                "reinforcement_count": item.get("reinforcement_count", 0),
                "is_stale": item.get("is_stale", False),
                "stale_reason": item.get("stale_reason"),
                "project_id": item.get("project_id"),
                "tenant_id": item.get("tenant_id"),
                "scope": item.get("scope", "user"),
                "owner_id": item.get("owner_id") or item.get("user_id"),
                "shared_with": list(item.get("shared_with") or []),
                "permission_level": item.get("permission_level") or "private",
            })

        memory_list = filter_readable_documents(
            memory_list,
            user_id=user_id,
            tenant_id=tenant_id,
            project_id=project_id,
        )[:limit]

        return {
            "success": True,
            "memories": annotate_memories_with_lifecycle(memory_list),
        }

    except Exception as e:
        print(f"[Memory retrieval error]: {e}")
        return {"success": False, "error": str(e)}


def find_active_memories_by_fact_key(user_id, fact_key, limit=20):
    """Look up active exclusive facts by key for conflict intelligence."""
    return find_memories_by_fact_key(
        user_id=user_id,
        fact_key=fact_key,
        statuses=("active",),
        limit=limit,
    )


def _fact_key_query_values(fact_key: str) -> list[str]:
    """Include legacy key aliases so old `project` rows match `primary_project`."""
    from app.services.memory_conflict_service import normalize_fact_key

    canonical = normalize_fact_key(fact_key)
    values = {canonical, fact_key}
    if canonical == "primary_project":
        values.add("project")
    if fact_key == "project":
        values.add("primary_project")
    return list(values)


def find_memories_by_fact_key(
    user_id,
    fact_key,
    *,
    statuses=("active",),
    normalized_value=None,
    limit=20,
    project_id=None,
    tenant_id=None,
):
    """Look up memories by fact_key, optionally filtered by status and value."""
    try:
        if not fact_key:
            return {"success": True, "memories": []}

        status_list = list(statuses)
        include_missing_active = "active" in status_list
        status_clauses = [{"status": status} for status in status_list]
        if include_missing_active:
            status_clauses.append({"status": {"$exists": False}})

        key_values = _fact_key_query_values(fact_key)
        query = {
            "$and": [
                build_scope_filter(
                    user_id=user_id,
                    tenant_id=tenant_id,
                    project_id=project_id,
                    include_shared_with_user=True,
                ),
                {"fact_key": {"$in": key_values}},
                {"$or": status_clauses},
            ]
        }
        if normalized_value is not None:
            query["$and"].append({"normalized_value": normalized_value})

        docs = list(
            memory_collection.find(query).sort("updated_at", -1).limit(
                max(limit * 3, limit)
            )
        )
        memories = [
            _doc_to_memory_candidate(item, fallback_user_id=user_id)
            for item in docs
        ]
        if not memories:
            memories = _legacy_compound_fact_candidates(
                user_id=user_id,
                fact_key=fact_key,
                statuses=status_list,
                include_missing_active=include_missing_active,
                limit=limit,
                project_id=project_id,
                tenant_id=tenant_id,
            )
        memories = filter_readable_documents(
            memories,
            user_id=user_id,
            tenant_id=tenant_id,
            project_id=project_id,
        )[:limit]
        return {"success": True, "memories": memories}
    except Exception as e:
        print(f"[Fact key lookup error]: {e}")
        return {"success": False, "error": str(e)}


_FACT_KEY_TEXT_PATTERNS = {
    "preferred_database": r"preferred\s+database|prefer\s+.+\s+as\s+(?:my\s+)?(?:preferred\s+)?database",
    "preferred_language": r"preferred\s+(?:programming\s+)?language|prefer\s+.+\s+(?:as\s+)?(?:my\s+)?(?:programming\s+)?language",
    "editor": r"preferred\s+code\s+editor|code\s+editor|as\s+(?:my\s+)?(?:code\s+)?editor",
    "primary_project": r"(?:primary|current|main)\s+project\s+is|my\s+project\s+is",
    "os": r"operating\s+system|primarily\s+use\s+(?:windows|linux|ubuntu|macos)",
    "company": r"current\s+company|work\s+at|employer",
    "role": r"current\s+role|job\s+title|work\s+as|i\s+am\s+an?\s+",
    "goal": r"(?:career\s+)?goal\s+is|want\s+to\s+become",
}


def _metadata_projection(doc: dict, user_id: str) -> dict:
    """Return only stored inspection fields. Missing values are JSON null."""
    stored_user = doc.get("user_id")
    # Never leak another tenant — caller must already scope the query.
    if stored_user is not None and str(stored_user) != str(user_id):
        raise PermissionError("Cross-user memory metadata access denied")

    conflict_with = doc.get("conflict_with_memory_id")
    if conflict_with is not None:
        conflict_with = str(conflict_with)

    return {
        "memory_id": str(doc.get("_id")) if doc.get("_id") is not None else None,
        "user_id": stored_user if stored_user is not None else None,
        "status": doc.get("status") if doc.get("status") is not None else None,
        "fact_key": doc.get("fact_key") if doc.get("fact_key") is not None else None,
        "normalized_value": (
            doc.get("normalized_value")
            if doc.get("normalized_value") is not None
            else None
        ),
        "conflict_group_id": (
            doc.get("conflict_group_id")
            if doc.get("conflict_group_id") is not None
            else None
        ),
        "conflict_type": (
            doc.get("conflict_type")
            if doc.get("conflict_type") is not None
            else None
        ),
        "conflict_with_memory_id": conflict_with,
        "memory": doc.get("memory") if doc.get("memory") is not None else None,
    }


def compare_conflict_group_ids(records: list[dict]) -> dict:
    """Compare stored conflict metadata. Never invent a shared group.

    ``consistent`` requires matching non-null conflict_group_id values AND
    reciprocal conflict_with_memory_id references when exactly two records
    are compared.
    """
    ids = [r.get("conflict_group_id") for r in records]
    present = [g for g in ids if g]
    unique_present = sorted(set(str(g) for g in present))

    reciprocal = None
    if len(records) == 2:
        a_id = str(records[0].get("memory_id") or "")
        b_id = str(records[1].get("memory_id") or "")
        a_with = records[0].get("conflict_with_memory_id")
        b_with = records[1].get("conflict_with_memory_id")
        a_with_s = str(a_with) if a_with is not None else None
        b_with_s = str(b_with) if b_with is not None else None
        if a_id and b_id:
            reciprocal = bool(a_with_s == b_id and b_with_s == a_id)
        # else: leave reciprocal=None — cannot evaluate without memory_ids

    if len(records) < 2:
        verdict = "insufficient_records"
    elif not present:
        verdict = "all_missing"
    elif any(g is None for g in ids) and present:
        verdict = "inconsistent_missing"
    elif len(unique_present) > 1:
        verdict = "inconsistent_mismatch"
    elif len(unique_present) == 1 and all(g is not None for g in ids):
        if len(records) == 2 and reciprocal is False:
            verdict = "asymmetric_references"
        elif len(records) == 2 and reciprocal is True:
            verdict = "consistent"
        else:
            verdict = "match"
    else:
        verdict = "inconsistent_missing"

    same = verdict in {"match", "consistent"}
    return {
        "verdict": verdict,
        "same_conflict_group": same,
        "reciprocal_references": reciprocal,
        "conflict_group_ids": ids,
        "distinct_non_null_group_ids": unique_present,
        "message": {
            "consistent": (
                "Both records share the same conflict_group_id and have "
                "reciprocal conflict_with_memory_id references."
            ),
            "match": "All inspected records share the same stored conflict_group_id.",
            "asymmetric_references": (
                "conflict_group_id values match, but conflict_with_memory_id "
                "references are missing or not reciprocal."
            ),
            "all_missing": "All inspected records have conflict_group_id = null.",
            "inconsistent_missing": (
                "Some records have a conflict_group_id while others are null. "
                "They do NOT all belong to the same stored conflict group."
            ),
            "inconsistent_mismatch": (
                "Inspected records have different non-null conflict_group_id values."
            ),
            "insufficient_records": (
                "Fewer than two records were found; cannot confirm a shared group."
            ),
        }.get(verdict, "Unable to compare conflict groups."),
    }


def inspect_memory_metadata(
    user_id: str,
    fact_key: str | None = None,
    *,
    statuses: tuple[str, ...] = ("active", "needs_review"),
    limit: int = 20,
):
    """Read-only metadata inspection for one user's memories.

    Returns actual stored fields only. Does not modify any records.
    Includes legacy rows whose text matches the fact key even when
    stored fact_key is null, so missing metadata is visible as null.
    """
    try:
        if not user_id or not str(user_id).strip():
            return {"success": False, "error": "user_id is required"}

        user_id = str(user_id).strip()
        from app.services.memory_conflict_service import normalize_fact_key

        canonical = normalize_fact_key(fact_key) if fact_key else None
        status_list = list(statuses)
        status_clauses = [{"status": status} for status in status_list]
        if "active" in status_list:
            status_clauses.append({"status": {"$exists": False}})

        or_clauses: list[dict] = []
        if canonical:
            or_clauses.append(
                {"fact_key": {"$in": _fact_key_query_values(canonical)}}
            )
            pattern = _FACT_KEY_TEXT_PATTERNS.get(canonical)
            if pattern:
                or_clauses.append(
                    {"memory": {"$regex": pattern, "$options": "i"}}
                )

        if not or_clauses:
            return {
                "success": False,
                "error": "fact_key is required for metadata inspection",
            }

        query = {
            "user_id": user_id,
            "$and": [
                {"$or": status_clauses},
                {"$or": or_clauses},
            ],
        }

        docs = list(
            memory_collection.find(query).sort("updated_at", -1).limit(limit)
        )

        # Deduplicate by _id while preserving order.
        seen = set()
        records = []
        for doc in docs:
            mid = str(doc.get("_id"))
            if mid in seen:
                continue
            seen.add(mid)
            # If a text-pattern hit clearly belongs to a different exclusive
            # key, skip it when a canonical key was requested.
            if canonical:
                stored_key = doc.get("fact_key")
                if stored_key:
                    from app.services.memory_conflict_service import (
                        normalize_fact_key as _nfk,
                    )
                    if _nfk(str(stored_key).replace("_", " ")) != canonical:
                        continue
                else:
                    from app.services.memory_conflict_service import (
                        extract_exclusive_fact,
                    )
                    extracted = extract_exclusive_fact(str(doc.get("memory") or ""))
                    if extracted and extracted[0] != canonical:
                        continue
            records.append(_metadata_projection(doc, user_id))

        comparison = compare_conflict_group_ids(records)

        return {
            "success": True,
            "read_only": True,
            "user_id": user_id,
            "fact_key": canonical,
            "record_count": len(records),
            "records": records,
            "conflict_group_comparison": comparison,
        }
    except PermissionError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        print(f"[Memory metadata inspection error]: {e}")
        return {"success": False, "error": str(e)}


def format_memory_metadata_report(inspection: dict) -> str:
    """Format inspection result for explicit user-facing metadata replies."""
    import json as _json

    if not inspection.get("success"):
        return (
            "Memory metadata inspection failed: "
            f"{inspection.get('error', 'unknown error')}"
        )

    lines = [
        "Memory metadata inspection (read-only). No records were modified.",
        f"user_id: {inspection.get('user_id')}",
        f"requested_fact_key: {inspection.get('fact_key')}",
        f"record_count: {inspection.get('record_count', 0)}",
        "",
        "Records (actual stored values; missing fields shown as null):",
    ]

    for index, record in enumerate(inspection.get("records") or [], start=1):
        payload = {
            "memory_id": record.get("memory_id"),
            "user_id": record.get("user_id"),
            "status": record.get("status"),
            "fact_key": record.get("fact_key"),
            "normalized_value": record.get("normalized_value"),
            "conflict_group_id": record.get("conflict_group_id"),
            "conflict_type": record.get("conflict_type"),
            "conflict_with_memory_id": record.get("conflict_with_memory_id"),
            "memory": record.get("memory"),
        }
        lines.append(f"{index}. {_json.dumps(payload, ensure_ascii=False)}")

    comparison = inspection.get("conflict_group_comparison") or {}
    lines.extend(
        [
            "",
            "Conflict group comparison (stored conflict_group_id only):",
            f"verdict: {comparison.get('verdict')}",
            f"same_conflict_group: {comparison.get('same_conflict_group')}",
            f"reciprocal_references: {comparison.get('reciprocal_references')}",
            "conflict_group_ids: "
            + _json.dumps(comparison.get("conflict_group_ids"), ensure_ascii=False),
            "distinct_non_null_group_ids: "
            + _json.dumps(
                comparison.get("distinct_non_null_group_ids"),
                ensure_ascii=False,
            ),
            f"detail: {comparison.get('message')}",
        ]
    )
    return "\n".join(lines)


def _propose_metadata_patch(doc: dict) -> dict | None:
    """Build a sparse $set patch for missing exclusive-fact metadata only."""
    from app.services.memory_conflict_service import extract_exclusive_fact

    patch: dict = {}
    text = str(doc.get("memory") or "")
    extracted = extract_exclusive_fact(text)

    if extracted:
        key, value = extracted
        if not doc.get("fact_key"):
            patch["fact_key"] = key
        if doc.get("normalized_value") in (None, ""):
            patch["normalized_value"] = value

    return patch or None


def _conflict_side_patch(
    doc: dict,
    *,
    partner_id: str,
    fact_key: str,
    conflict_group_id: str,
    conflict_type: str = "possible_contradiction",
    normalized_value: str | None = None,
) -> dict:
    """Build field-level $set for one side of a conflict pair.

    Fills missing fact metadata, aligns conflict_group_id when missing,
    sets conflict_type, and always ensures reciprocal conflict_with_memory_id.
    Never changes status or memory text.
    """
    from app.services.memory_conflict_service import extract_exclusive_fact

    patch: dict = {}
    extracted = extract_exclusive_fact(str(doc.get("memory") or ""))
    extracted_key = extracted[0] if extracted else None
    extracted_value = extracted[1] if extracted else None

    target_key = fact_key or extracted_key
    if target_key and (
        not doc.get("fact_key")
        or str(doc.get("fact_key")) != str(target_key)
    ):
        # Only overwrite null/empty; if a different exclusive key is already
        # stored, leave it (caller must not pair unrelated facts).
        if not doc.get("fact_key"):
            patch["fact_key"] = target_key

    target_value = normalized_value or extracted_value
    if target_value is not None and doc.get("normalized_value") in (None, ""):
        patch["normalized_value"] = target_value

    existing_group = doc.get("conflict_group_id")
    if not existing_group:
        patch["conflict_group_id"] = conflict_group_id
    elif str(existing_group) != str(conflict_group_id):
        # Do not silently rewrite a different established group.
        raise ValueError(
            f"conflict_group_id mismatch on {doc.get('_id')}: "
            f"existing={existing_group} requested={conflict_group_id}"
        )

    if doc.get("conflict_type") in (None, "", "none"):
        patch["conflict_type"] = conflict_type

    current_with = doc.get("conflict_with_memory_id")
    if str(current_with or "") != str(partner_id):
        patch["conflict_with_memory_id"] = str(partner_id)

    return patch


def _verify_conflict_pair_invariant(
    left: dict,
    right: dict,
    *,
    fact_key: str,
    conflict_group_id: str,
) -> tuple[bool, list[str]]:
    """Return (ok, issues) after reading both documents."""
    issues: list[str] = []
    left_id = str(left.get("_id") or left.get("memory_id") or "")
    right_id = str(right.get("_id") or right.get("memory_id") or "")

    if left.get("user_id") != right.get("user_id"):
        issues.append("user_id mismatch between conflict partners")
    if str(left.get("fact_key") or "") != str(fact_key):
        issues.append(f"left fact_key={left.get('fact_key')!r} expected {fact_key!r}")
    if str(right.get("fact_key") or "") != str(fact_key):
        issues.append(f"right fact_key={right.get('fact_key')!r} expected {fact_key!r}")
    if str(left.get("conflict_group_id") or "") != str(conflict_group_id):
        issues.append("left conflict_group_id mismatch")
    if str(right.get("conflict_group_id") or "") != str(conflict_group_id):
        issues.append("right conflict_group_id mismatch")
    if left.get("normalized_value") in (None, ""):
        issues.append("left normalized_value missing")
    if right.get("normalized_value") in (None, ""):
        issues.append("right normalized_value missing")
    if str(left.get("conflict_with_memory_id") or "") != right_id:
        issues.append("left conflict_with_memory_id is not reciprocal")
    if str(right.get("conflict_with_memory_id") or "") != left_id:
        issues.append("right conflict_with_memory_id is not reciprocal")
    if left.get("conflict_type") in (None, "", "none"):
        issues.append("left conflict_type missing")
    if right.get("conflict_type") in (None, "", "none"):
        issues.append("right conflict_type missing")
    return (not issues), issues


def persist_memory_conflict_pair(
    user_id: str,
    left_memory_id: str,
    right_memory_id: str,
    *,
    fact_key: str,
    conflict_group_id: str | None = None,
    conflict_type: str = "possible_contradiction",
    left_normalized_value: str | None = None,
    right_normalized_value: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Persist bilateral conflict metadata and verify the invariant.

    Updates BOTH records with matching conflict_group_id, reciprocal
    conflict_with_memory_id, fact_key, normalized_value, and conflict_type.
    Preserves status and memory text. Idempotent. Never reports success
    unless a post-write read confirms the invariant (unless dry_run).
    """
    from bson import ObjectId
    from uuid import uuid4 as _uuid4

    try:
        if not user_id or not left_memory_id or not right_memory_id or not fact_key:
            return {
                "success": False,
                "error": (
                    "user_id, left_memory_id, right_memory_id, and fact_key "
                    "are required"
                ),
            }
        if str(left_memory_id) == str(right_memory_id):
            return {"success": False, "error": "Conflict partners must be distinct"}

        user_id = str(user_id).strip()
        left_oid = ObjectId(str(left_memory_id))
        right_oid = ObjectId(str(right_memory_id))

        left = memory_collection.find_one({"_id": left_oid, "user_id": user_id})
        right = memory_collection.find_one({"_id": right_oid, "user_id": user_id})
        if not left or not right:
            return {
                "success": False,
                "error": "One or both memories were not found for this user",
            }

        # Reuse an existing valid group; never invent a second group for the pair.
        existing_groups = [
            g
            for g in (left.get("conflict_group_id"), right.get("conflict_group_id"))
            if g
        ]
        unique_groups = {str(g) for g in existing_groups}
        if conflict_group_id:
            group_id = str(conflict_group_id)
            if unique_groups and group_id not in unique_groups and len(unique_groups) == 1:
                # Prefer the already-established group on the pair.
                group_id = next(iter(unique_groups))
            elif len(unique_groups) > 1:
                return {
                    "success": False,
                    "error": "Partners already belong to different conflict groups",
                    "groups": sorted(unique_groups),
                }
        elif len(unique_groups) == 1:
            group_id = next(iter(unique_groups))
        elif len(unique_groups) > 1:
            return {
                "success": False,
                "error": "Partners already belong to different conflict groups",
                "groups": sorted(unique_groups),
            }
        else:
            group_id = _uuid4().hex

        left_patch = _conflict_side_patch(
            left,
            partner_id=str(right_oid),
            fact_key=fact_key,
            conflict_group_id=group_id,
            conflict_type=conflict_type,
            normalized_value=left_normalized_value,
        )
        right_patch = _conflict_side_patch(
            right,
            partner_id=str(left_oid),
            fact_key=fact_key,
            conflict_group_id=group_id,
            conflict_type=conflict_type,
            normalized_value=right_normalized_value,
        )

        plan = {
            "left_memory_id": str(left_oid),
            "right_memory_id": str(right_oid),
            "conflict_group_id": group_id,
            "left_status_preserved": left.get("status"),
            "right_status_preserved": right.get("status"),
            "left_set": left_patch,
            "right_set": right_patch,
        }

        if dry_run:
            return {
                "success": True,
                "dry_run": True,
                "read_only": True,
                "plan": plan,
            }

        now = datetime.now(timezone.utc)
        applied_left = False
        applied_right = False

        def _apply_side(oid, patch: dict) -> bool:
            if not patch:
                return True
            payload = dict(patch)
            payload["updated_at"] = now
            result = memory_collection.update_one(
                {"_id": oid, "user_id": user_id},
                {"$set": payload},
            )
            return result.matched_count == 1

        # Prefer a transaction when the topology supports it.
        used_transaction = False
        try:
            with client.start_session() as session:
                with session.start_transaction():
                    if left_patch:
                        payload = dict(left_patch)
                        payload["updated_at"] = now
                        result = memory_collection.update_one(
                            {"_id": left_oid, "user_id": user_id},
                            {"$set": payload},
                            session=session,
                        )
                        if result.matched_count != 1:
                            raise RuntimeError("Failed to update left conflict partner")
                        applied_left = True
                    else:
                        applied_left = True
                    if right_patch:
                        payload = dict(right_patch)
                        payload["updated_at"] = now
                        result = memory_collection.update_one(
                            {"_id": right_oid, "user_id": user_id},
                            {"$set": payload},
                            session=session,
                        )
                        if result.matched_count != 1:
                            raise RuntimeError("Failed to update right conflict partner")
                        applied_right = True
                    else:
                        applied_right = True
                used_transaction = True
        except Exception as tx_error:
            print(f"[Conflict pair transaction unavailable/failed]: {tx_error}")
            # Recoverable sequential strategy with verification.
            applied_left = _apply_side(left_oid, left_patch)
            if not applied_left:
                return {
                    "success": False,
                    "error": "Failed to update left conflict partner",
                    "plan": plan,
                }
            applied_right = _apply_side(right_oid, right_patch)
            if not applied_right:
                return {
                    "success": False,
                    "error": "Failed to update right conflict partner after left succeeded",
                    "partial": True,
                    "plan": plan,
                }

        left_after = memory_collection.find_one({"_id": left_oid, "user_id": user_id})
        right_after = memory_collection.find_one({"_id": right_oid, "user_id": user_id})
        ok, issues = _verify_conflict_pair_invariant(
            left_after,
            right_after,
            fact_key=fact_key,
            conflict_group_id=group_id,
        )
        if not ok:
            return {
                "success": False,
                "error": "Post-write conflict invariant verification failed",
                "issues": issues,
                "plan": plan,
                "used_transaction": used_transaction,
            }

        record_audit_event(
            user_id=user_id,
            event_type="memory_metadata_backfilled",
            memory_id=str(left_oid),
            details={
                "source": "persist_memory_conflict_pair",
                "partner_memory_id": str(right_oid),
                "conflict_group_id": group_id,
                "left_fields": sorted(left_patch.keys()),
                "right_fields": sorted(right_patch.keys()),
            },
        )

        return {
            "success": True,
            "dry_run": False,
            "used_transaction": used_transaction,
            "conflict_group_id": group_id,
            "plan": plan,
            "verified": True,
            "left": _metadata_projection(left_after, user_id),
            "right": _metadata_projection(right_after, user_id),
        }
    except Exception as e:
        print(f"[persist_memory_conflict_pair error]: {e}")
        return {"success": False, "error": str(e)}


def ensure_conflict_partner_metadata(
    user_id: str,
    memory_id: str,
    *,
    conflict_group_id: str,
    fact_key: str | None = None,
    normalized_value: str | None = None,
    partner_memory_id: str | None = None,
    conflict_type: str = "possible_contradiction",
    dry_run: bool = False,
) -> dict:
    """Back-compat wrapper. Prefer persist_memory_conflict_pair when both IDs known."""
    if partner_memory_id and fact_key:
        return persist_memory_conflict_pair(
            user_id,
            memory_id,
            partner_memory_id,
            fact_key=fact_key,
            conflict_group_id=conflict_group_id,
            conflict_type=conflict_type,
            left_normalized_value=normalized_value,
            dry_run=dry_run,
        )

    # Single-sided fallback (incomplete without partner id for reciprocity).
    from bson import ObjectId

    try:
        if not user_id or not memory_id or not conflict_group_id:
            return {
                "success": False,
                "error": "user_id, memory_id, and conflict_group_id are required",
            }
        doc = memory_collection.find_one(
            {"_id": ObjectId(memory_id), "user_id": user_id}
        )
        if not doc:
            return {"success": False, "error": "Memory not found for this user"}

        existing_group = doc.get("conflict_group_id")
        if existing_group and str(existing_group) != str(conflict_group_id):
            return {
                "success": True,
                "memory_id": memory_id,
                "skipped": True,
                "reason": "existing_conflict_group_mismatch",
                "existing_conflict_group_id": str(existing_group),
                "requested_conflict_group_id": str(conflict_group_id),
                "dry_run": dry_run,
            }

        patch = _propose_metadata_patch(doc) or {}
        if fact_key and not doc.get("fact_key"):
            patch["fact_key"] = fact_key
        if normalized_value and doc.get("normalized_value") in (None, ""):
            patch.setdefault("normalized_value", normalized_value)
        if not existing_group:
            patch["conflict_group_id"] = conflict_group_id
        if doc.get("conflict_type") in (None, "", "none"):
            patch["conflict_type"] = conflict_type
        if partner_memory_id and str(doc.get("conflict_with_memory_id") or "") != str(
            partner_memory_id
        ):
            patch["conflict_with_memory_id"] = str(partner_memory_id)

        if not patch:
            return {
                "success": True,
                "memory_id": memory_id,
                "skipped": True,
                "reason": "already_complete",
                "dry_run": dry_run,
            }
        if dry_run:
            return {
                "success": True,
                "memory_id": memory_id,
                "dry_run": True,
                "proposed_set": patch,
            }

        patch["updated_at"] = datetime.now(timezone.utc)
        result = memory_collection.update_one(
            {"_id": ObjectId(memory_id), "user_id": user_id},
            {"$set": patch},
        )
        if result.matched_count == 0:
            return {"success": False, "error": "Memory not found for this user"}
        record_audit_event(
            user_id=user_id,
            event_type="memory_metadata_backfilled",
            memory_id=memory_id,
            details={"fields": sorted(k for k in patch if k != "updated_at"), "source": "conflict_partner"},
        )
        return {
            "success": True,
            "memory_id": memory_id,
            "dry_run": False,
            "applied_set": {k: v for k, v in patch.items() if k != "updated_at"},
        }
    except Exception as e:
        print(f"[Conflict partner metadata update error]: {e}")
        return {"success": False, "error": str(e)}


def backfill_memory_metadata(
    user_id: str,
    *,
    fact_key: str | None = None,
    dry_run: bool = True,
    limit: int = 200,
) -> dict:
    """Optional metadata backfill for one user (dry-run by default).

    Fills missing fact_key / normalized_value from deterministic extraction
    and propagates conflict_group_id across linked active↔pending pairs
    when one side already stores a group id. Never invents links from
    embeddings alone. Never changes status, memory text, or embeddings.
    """
    from bson import ObjectId
    from app.services.memory_conflict_service import normalize_fact_key

    try:
        if not user_id or not str(user_id).strip():
            return {"success": False, "error": "user_id is required"}

        user_id = str(user_id).strip()
        canonical = normalize_fact_key(fact_key) if fact_key else None

        query: dict = {
            "user_id": user_id,
            "status": {"$in": ["active", "needs_review"]},
        }
        if canonical:
            pattern = _FACT_KEY_TEXT_PATTERNS.get(canonical)
            or_clauses = [{"fact_key": {"$in": _fact_key_query_values(canonical)}}]
            if pattern:
                or_clauses.append({"memory": {"$regex": pattern, "$options": "i"}})
            query["$or"] = or_clauses

        docs = list(memory_collection.find(query).limit(limit))
        proposals: list[dict] = []
        applied: list[dict] = []

        # Pass 1: fill missing fact_key / normalized_value on each row.
        for doc in docs:
            patch = _propose_metadata_patch(doc)
            if not patch:
                continue
            item = {
                "memory_id": str(doc["_id"]),
                "status": doc.get("status"),
                "proposed_set": patch,
                "reason": "missing_fact_metadata",
            }
            proposals.append(item)
            if not dry_run:
                set_fields = dict(patch)
                set_fields["updated_at"] = datetime.now(timezone.utc)
                memory_collection.update_one(
                    {"_id": doc["_id"], "user_id": user_id},
                    {"$set": set_fields},
                )
                record_audit_event(
                    user_id=user_id,
                    event_type="memory_metadata_backfilled",
                    memory_id=str(doc["_id"]),
                    details={
                        "fields": sorted(patch.keys()),
                        "source": "fact_extraction",
                    },
                )
                applied.append(item)

        # Pass 2: bilateral conflict metadata via persist_memory_conflict_pair.
        # Reload so pass-1 fact keys are visible when filtering.
        docs = list(memory_collection.find(query).limit(limit))
        by_id = {str(doc["_id"]): doc for doc in docs}
        seen_pairs: set[tuple[str, str]] = set()

        for doc in docs:
            pending_group = doc.get("conflict_group_id")
            partner_id = doc.get("conflict_with_memory_id")
            if not pending_group or not partner_id:
                continue
            if doc.get("status") != "needs_review":
                continue

            partner = by_id.get(str(partner_id))
            if partner is None:
                try:
                    partner = memory_collection.find_one(
                        {"_id": ObjectId(str(partner_id)), "user_id": user_id}
                    )
                except Exception:
                    partner = None
            if not partner:
                continue

            left_id = str(partner["_id"])
            right_id = str(doc["_id"])
            pair_key = tuple(sorted((left_id, right_id)))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            fact = (
                doc.get("fact_key")
                or partner.get("fact_key")
                or (_propose_metadata_patch(partner) or {}).get("fact_key")
                or (_propose_metadata_patch(doc) or {}).get("fact_key")
            )
            if not fact:
                continue

            pair_result = persist_memory_conflict_pair(
                user_id,
                left_id,
                right_id,
                fact_key=str(fact),
                conflict_group_id=str(pending_group),
                conflict_type="possible_contradiction",
                dry_run=dry_run,
            )
            if not pair_result.get("success"):
                continue

            plan = pair_result.get("plan") or {}
            left_set = plan.get("left_set") or {}
            if left_set:
                item = {
                    "memory_id": left_id,
                    "status": partner.get("status"),
                    "proposed_set": left_set,
                    "reason": "persist_memory_conflict_pair",
                    "source_pending_memory_id": right_id,
                    "conflict_group_id": plan.get("conflict_group_id"),
                }
                proposals.append(item)
                if not dry_run:
                    applied.append(item)
            right_set = plan.get("right_set") or {}
            if right_set:
                item = {
                    "memory_id": right_id,
                    "status": doc.get("status"),
                    "proposed_set": right_set,
                    "reason": "persist_memory_conflict_pair",
                    "source_pending_memory_id": right_id,
                    "conflict_group_id": plan.get("conflict_group_id"),
                }
                proposals.append(item)
                if not dry_run:
                    applied.append(item)

        return {
            "success": True,
            "read_only": dry_run,
            "dry_run": dry_run,
            "user_id": user_id,
            "fact_key": canonical,
            "proposal_count": len(proposals),
            "applied_count": 0 if dry_run else len(applied),
            "proposals": proposals,
            "applied": applied if not dry_run else [],
        }
    except Exception as e:
        print(f"[Memory metadata backfill error]: {e}")
        return {"success": False, "error": str(e)}


def find_logical_duplicate(
    user_id,
    memory_text,
    *,
    fact_key=None,
    normalized_value=None,
):
    """Find an existing active/needs_review memory that is the same logical fact.

    Matches by:
    1. fact_key + normalized_value (structured)
    2. case-insensitive normalized memory text

    Returns the best match or None. Prefer active over needs_review.
    """
    from app.services.memory_conflict_service import (
        classify_duplicate,
        normalize_memory_text,
    )

    try:
        candidates = []

        # Structured fact match across active + needs_review.
        if fact_key and normalized_value is not None:
            by_fact = find_memories_by_fact_key(
                user_id=user_id,
                fact_key=fact_key,
                statuses=("active", "needs_review"),
                normalized_value=normalized_value,
                limit=20,
            )
            if by_fact.get("success"):
                candidates.extend(by_fact.get("memories", []))

        # Text-normalized scan of reviewable memories for this user.
        # Bounded scan — enough for personal memory stores.
        docs = memory_collection.find(
            {
                "user_id": user_id,
                **_reviewable_status_filter(),
            },
            {
                "_id": 1,
                "memory": 1,
                "importance": 1,
                "confidence": 1,
                "status": 1,
                "fact_key": 1,
                "normalized_value": 1,
                "conflict_group_id": 1,
                "version": 1,
                "reinforcement_count": 1,
                "last_reinforced_at": 1,
            },
        ).sort("updated_at", -1).limit(200)

        target_norm = normalize_memory_text(memory_text)
        known_ids = {item.get("memory_id") for item in candidates}
        for item in docs:
            candidate = _doc_to_memory_candidate(
                item, fallback_user_id=user_id
            )
            if candidate["memory_id"] in known_ids:
                continue
            if normalize_memory_text(candidate.get("memory", "")) == target_norm:
                candidates.append(candidate)
                continue
            classification = classify_duplicate(
                memory_text,
                candidate.get("memory", ""),
            )
            if classification["classification"] in {
                "exact_duplicate",
                "paraphrase_duplicate",
            }:
                candidates.append(candidate)

        if not candidates:
            return {"success": True, "is_duplicate": False, "match": None}

        # Prefer active matches, then most recently reinforced / highest importance.
        def _rank(item):
            status = item.get("status", "active")
            status_rank = 0 if status in ("active", None) else 1
            return (
                status_rank,
                -int(item.get("importance", 3)),
                -float(item.get("score", 0.0)),
            )

        candidates.sort(key=_rank)
        best = candidates[0]
        return {
            "success": True,
            "is_duplicate": True,
            "match": best,
            "memory_id": best["memory_id"],
            "memory": best.get("memory"),
            "importance": best.get("importance", 3),
            "status": best.get("status", "active"),
            "candidates": candidates,
        }
    except Exception as e:
        print(f"[Logical duplicate lookup error]: {e}")
        return {"success": False, "error": str(e)}


def get_memory_versions(user_id, memory_id, limit=20):
    """Return version history linked through supersede relationships."""
    from bson import ObjectId

    try:
        current = memory_collection.find_one(
            {"_id": ObjectId(memory_id), "user_id": user_id}
        )
        if not current:
            return {"success": False, "error": "Memory not found for this user"}

        group_id = current.get("conflict_group_id")
        if group_id:
            docs = list(
                memory_collection.find(
                    {"user_id": user_id, "conflict_group_id": group_id}
                ).sort("version", ASCENDING).limit(limit)
            )
        else:
            docs = [current]

        versions = []
        for item in docs:
            versions.append({
                "memory_id": str(item["_id"]),
                "memory": item.get("memory"),
                "status": item.get("status", "active"),
                "version": item.get("version", 1),
                "fact_key": item.get("fact_key"),
                "conflict_type": item.get("conflict_type", "none"),
                "supersedes_memory_id": item.get("supersedes_memory_id"),
                "superseded_by_memory_id": item.get("superseded_by_memory_id"),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
            })

        return {"success": True, "versions": versions}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _record_memory_access(user_id, memories):
    # Track retrieval usage without changing factual freshness.
    try:
        from bson import ObjectId

        memory_ids = []
        for item in memories:
            try:
                memory_ids.append(ObjectId(item["memory_id"]))
            except (KeyError, TypeError, ValueError):
                continue

        if not memory_ids:
            return

        memory_collection.update_many(
            {
                "_id": {"$in": memory_ids},
                "user_id": user_id,
                **_active_status_filter(),
            },
            {
                "$inc": {"access_count": 1},
                "$set": {"last_accessed_at": datetime.now(timezone.utc)},
            },
        )
    except Exception as e:
        # Do not fail retrieval if telemetry tracking is unavailable.
        print(f"[Memory access tracking failed]: {e}")


# ============================================================
# SEMANTIC SEARCH MEMORIES
# ============================================================

def _normalize_lexical_token(token: str) -> str:
    """Light stemming so prefer/preferred and conflict/conflicts overlap."""
    t = token.lower()
    if t.endswith("ying") and len(t) > 5:
        return t[:-4] + "y"
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("ing") and len(t) > 5:
        return t[:-3]
    if t.endswith("ed") and len(t) > 4:
        return t[:-2]
    if t.endswith("s") and len(t) > 3 and not t.endswith("ss"):
        return t[:-1]
    return t


def _lexical_relevance_boost(query: str, memory_text: str) -> float:
    """Stage 19: Compute a 0-1 lexical overlap score between query and memory.

    This penalizes memories that are semantically similar (high embedding
    score) but share no meaningful words with the query.
    """
    import re as _re

    stop_words = {
        "i", "my", "me", "the", "a", "an", "is", "are", "was", "were",
        "am", "be", "been", "to", "of", "in", "for", "on", "at", "it",
        "that", "this", "and", "or", "but", "with", "as", "by", "from",
        "what", "which", "who", "do", "does", "did", "have", "has", "had",
        "not", "so", "if", "about", "up", "out", "can", "will", "just",
        "how", "than", "them", "their", "its", "all", "each", "every",
        "some", "any", "into", "no", "also", "your", "our", "would",
        "should", "could", "may", "might", "then", "check", "saved",
        "identify", "any", "currently",
    }

    query_words = {
        _normalize_lexical_token(w)
        for w in _re.findall(r"\w+", query)
        if w.lower() not in stop_words and len(w) > 1
    }
    memory_words = {
        _normalize_lexical_token(w)
        for w in _re.findall(r"\w+", memory_text)
        if w.lower() not in stop_words and len(w) > 1
    }

    if not query_words:
        return 0.5

    overlap = query_words & memory_words
    return len(overlap) / len(query_words)


def infer_query_fact_keys(query: str) -> list[str]:
    """Infer canonical fact keys implied by a natural-language memory query."""
    import re as _re

    q = (query or "").lower()
    keys: list[str] = []
    if _re.search(r"\b(code\s+)?editor\b|\bide\b|\bvs\s*code\b|\bintellij\b", q):
        keys.append("editor")
    if _re.search(
        r"\b(preferred\s+)?database\b|\bmongodb\b|\bpostgresql\b|\bpostgres\b",
        q,
    ):
        keys.append("preferred_database")
    if _re.search(
        r"\b(programming\s+)?language\b|\btypescript\b|\bjavascript\b",
        q,
    ) and "editor" not in q and "database" not in q:
        keys.append("preferred_language")
    if _re.search(r"\b(primary|current|main)\s+project\b", q) or _re.search(
        r"\bwhat is my (?:current )?project\b", q
    ):
        keys.append("primary_project")
    if _re.search(
        r"\b(?:ai|named)\s+project\b|\bproject\s+name\b|"
        r"\b(grocery|marketplace|freshcart)\b",
        q,
    ):
        keys.append("project_name")
    if _re.search(r"\b(agentic|bug\s*resolver)\b", q):
        keys.append("project_description")
    if _re.search(r"\b(operating\s+system|\bos\b|windows|ubuntu|linux)\b", q):
        keys.append("os")
    if _re.search(r"\b(company|employer|work\s+at)\b", q):
        keys.append("company")
    if _re.search(r"\b(role|job\s+title)\b", q):
        keys.append("role")
    if _re.search(r"\b(career\s+)?goal\b", q):
        keys.append("goal")
    return keys


def _memory_fact_key(memory: dict) -> str:
    from app.services.memory_conflict_service import (
        classify_project_fact,
        extract_exclusive_fact,
        normalize_fact_key,
    )

    key = (memory.get("fact_key") or "").lower()
    if key:
        return normalize_fact_key(key.replace("_", " "))
    text = memory.get("memory") or ""
    classified = classify_project_fact(text)
    if classified:
        return classified[0]
    extracted = extract_exclusive_fact(text)
    if extracted:
        return extracted[0]
    return ""


def _project_intent_boost(query: str, memory: dict) -> float:
    """Boost / penalize memories from query intent and canonical fact keys.

    Prefer matching exclusive keys (editor vs preferred_language, primary
    project vs project_name/description) without lowering vector thresholds.
    """
    import re as _re

    q = (query or "").lower()
    fact_key = _memory_fact_key(memory)
    text = (memory.get("memory") or "").lower()
    boost = 1.0
    intent_keys = set(infer_query_fact_keys(query))

    # Strong fact-key alignment for preference / identity queries.
    if intent_keys:
        if fact_key and fact_key in intent_keys:
            boost *= 1.40
        elif fact_key and fact_key not in intent_keys:
            # Demote clearly mismatched exclusive preferences.
            if "editor" in intent_keys and fact_key == "preferred_language":
                boost *= 0.45
            elif "preferred_language" in intent_keys and fact_key == "editor":
                boost *= 0.50
            elif "preferred_database" in intent_keys and fact_key != "preferred_database":
                boost *= 0.55
            elif "primary_project" in intent_keys and fact_key in {
                "project_name",
                "project_description",
            }:
                boost *= 0.70
            elif fact_key in {
                "preferred_language",
                "preferred_database",
                "editor",
                "os",
                "company",
                "role",
                "goal",
                "primary_project",
            }:
                boost *= 0.65

    asks_primary = "primary_project" in intent_keys
    asks_named = "project_name" in intent_keys
    asks_description = "project_description" in intent_keys

    if asks_primary:
        if fact_key == "primary_project":
            boost *= 1.15
        elif fact_key in {"project_name", "project_description"}:
            boost *= 0.85
    if asks_named:
        if fact_key == "project_name" or "freshcart" in text or "marketplace" in text:
            boost *= 1.25
        elif fact_key == "primary_project" and "freshcart" not in text:
            boost *= 0.85
    if asks_description:
        if fact_key == "project_description" or "agentic" in text or "bug resolver" in text:
            boost *= 1.25
        elif fact_key == "primary_project" and "agentic" not in text:
            boost *= 0.90

    # Lexical editor cues when fact_key is missing on legacy rows.
    if "editor" in intent_keys and not fact_key:
        if _re.search(r"\b(editor|ide|vs\s*code|intellij|vscode)\b", text):
            boost *= 1.30
        elif _re.search(r"\b(python|typescript|javascript|language)\b", text):
            boost *= 0.50

    return boost


def _deduplicate_results(memories: list[dict]) -> list[dict]:
    """Stage 19: Suppress textual duplicates in retrieval results.

    When multiple results have the same normalized text, keep only the
    one with the highest priority score.
    """
    from app.services.memory_conflict_service import normalize_memory_text

    seen: dict[str, dict] = {}
    for mem in memories:
        norm = normalize_memory_text(mem.get("memory", ""))
        if norm in seen:
            existing_score = calculate_memory_priority_score(seen[norm])
            new_score = calculate_memory_priority_score(mem)
            if new_score > existing_score:
                seen[norm] = mem
        else:
            seen[norm] = mem

    return list(seen.values())


def _vector_search_memories(
    user_id,
    query_embedding,
    *,
    status_filter,
    limit,
    min_score,
    project_id=None,
    tenant_id=None,
):
    vector_filter = build_vector_search_filter(
        user_id=user_id,
        tenant_id=tenant_id,
        project_id=project_id,
    )
    pipeline = [
        {
            "$vectorSearch": {
                "index": "memory_vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": max(100, limit * 20),
                "limit": max(limit * 5, 30),
                "filter": vector_filter,
            }
        },
        {"$match": status_filter},
        {
            "$project": {
                "_id": 1,
                "memory": 1,
                "user_id": 1,
                "importance": 1,
                "confidence": 1,
                "created_at": 1,
                "updated_at": 1,
                "status": 1,
                "fact_key": 1,
                "normalized_value": 1,
                "conflict_group_id": 1,
                "version": 1,
                "last_accessed_at": 1,
                "access_count": 1,
                "last_reinforced_at": 1,
                "reinforcement_count": 1,
                "is_stale": 1,
                "stale_reason": 1,
                "project_id": 1,
                "tenant_id": 1,
                "scope": 1,
                "owner_id": 1,
                "shared_with": 1,
                "permission_level": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    memories = []
    for item in memory_collection.aggregate(pipeline):
        score = item.get("score", 0)
        mid = str(item.get("_id", "?"))
        if score < min_score:
            print(
                f"[Memory rejected] id={mid} score={score:.4f} "
                f"< threshold {min_score}"
            )
            continue

        # Derive owner from stored user_id only — never invent from caller.
        if item.get("owner_id") is None and item.get("user_id"):
            item["owner_id"] = item.get("user_id")
        if not item.get("permission_level"):
            item["permission_level"] = "private"
        # Authorization before any content-bearing logs or downstream ranking.
        if not document_matches_scope(
            item,
            user_id=user_id,
            tenant_id=tenant_id,
            project_id=project_id,
            allow_shared=True,
        ):
            print(f"[Memory rejected] id={mid} reason=scope_denied")
            continue
        if not check_read_permission(
            user_id,
            item,
            tenant_id=tenant_id,
            project_id=project_id,
            audit=False,
        ).get("allowed"):
            print(f"[Memory rejected] id={mid} reason=permission_denied")
            continue

        print(
            f"[Memory candidate] id={mid} status={item.get('status', 'active')} "
            f"score={score:.4f}"
        )

        memories.append({
            "memory_id": str(item["_id"]),
            "memory": item["memory"],
            "user_id": item.get("user_id"),
            "score": score,
            "importance": item.get("importance", 3),
            "confidence": item.get("confidence", 0.75),
            "created_at": item.get("created_at"),
            "updated_at": item.get("updated_at"),
            "status": item.get("status", "active"),
            "fact_key": item.get("fact_key"),
            "normalized_value": item.get("normalized_value"),
            "conflict_group_id": item.get("conflict_group_id"),
            "version": item.get("version", 1),
            "last_accessed_at": item.get("last_accessed_at"),
            "access_count": item.get("access_count", 0),
            "last_reinforced_at": item.get("last_reinforced_at"),
            "reinforcement_count": item.get("reinforcement_count", 0),
            "is_stale": item.get("is_stale", False),
            "stale_reason": item.get("stale_reason"),
            "project_id": item.get("project_id"),
            "tenant_id": item.get("tenant_id"),
            "scope": item.get("scope", "user"),
            "owner_id": item.get("owner_id") or item.get("user_id"),
            "shared_with": list(item.get("shared_with") or []),
            "permission_level": item.get("permission_level") or "private",
        })
    return memories


def _extract_memory_fact_pairs(item: dict) -> list[tuple[str, str]]:
    """Return all exclusive facts from a record, including legacy compounds."""
    from app.services.memory_conflict_service import (
        EXCLUSIVE_FACT_KEYS,
        classify_project_fact,
        extract_exclusive_fact,
        normalize_fact_key,
        _normalize_value,
    )
    from app.services.memory_fact_extraction import extract_facts_from_message

    if not isinstance(item, dict):
        return []

    memory_text = str(item.get("memory", ""))
    parsed = extract_facts_from_message("Remember that " + memory_text)
    parsed_pairs = [
        (
            normalize_fact_key(str(fact.fact_key)),
            _normalize_value(str(fact.normalized_value)),
        )
        for fact in parsed.facts
        if fact.fact_key
        and normalize_fact_key(str(fact.fact_key)) in EXCLUSIVE_FACT_KEYS
        and fact.normalized_value
    ]
    if parsed_pairs:
        # Older rows may store several declarations in one text field while
        # carrying only the first declaration's fact_key/normalized_value.
        return list(dict.fromkeys(parsed_pairs))

    classified = classify_project_fact(memory_text)
    if classified:
        key, value = classified
    else:
        key = item.get("fact_key")
        value = item.get("normalized_value")
        if not key or value is None:
            extracted = extract_exclusive_fact(memory_text)
            if not extracted:
                return []
            key, value = extracted
        else:
            key = normalize_fact_key(str(key))
            value = _normalize_value(str(value))

    if key not in EXCLUSIVE_FACT_KEYS:
        return []
    return [(key, value)]


def detect_conflicting_active_facts(memories: list[dict]) -> list[dict]:
    """Group active exclusive facts that contradict each other in a result set."""
    by_key: dict[str, list[dict]] = {}
    for item in memories:
        if not isinstance(item, dict):
            continue
        status = item.get("status", "active")
        # Only active / legacy-active facts can form current-fact conflicts.
        if status not in ("active", None):
            continue

        for key, value in _extract_memory_fact_pairs(item):
            entry = dict(item)
            entry["fact_key"] = key
            entry["normalized_value"] = value
            by_key.setdefault(key, []).append(entry)

    conflicts = []
    for key, items in by_key.items():
        values = {item["normalized_value"] for item in items}
        if len(values) > 1:
            conflicts.append({
                "fact_key": key,
                "values": sorted(values),
                "conflict_kind": "active_vs_active",
                "memories": items,
            })
    return conflicts


def detect_active_pending_conflicts(memories: list[dict]) -> list[dict]:
    """Detect exclusive-key conflicts between active and needs_review facts.

    Example: active MongoDB + pending PostgreSQL for preferred_database.
    These are not active-vs-active conflicts, but must surface in retrieval
    so a fresh session can ask for clarification without chat history.
    """
    actives: dict[str, list[dict]] = {}
    pending: dict[str, list[dict]] = {}

    for item in memories:
        if not isinstance(item, dict):
            continue
        status = item.get("status", "active")
        for key, value in _extract_memory_fact_pairs(item):
            entry = dict(item)
            entry["fact_key"] = key
            entry["normalized_value"] = value
            if status == "needs_review":
                pending.setdefault(key, []).append(entry)
            elif status in ("active", None):
                actives.setdefault(key, []).append(entry)

    conflicts = []
    for key, pending_items in pending.items():
        active_items = actives.get(key) or []
        if not active_items:
            continue
        active_values = {item["normalized_value"] for item in active_items}
        pending_values = {item["normalized_value"] for item in pending_items}
        if not (pending_values - active_values):
            continue
        group_ids = {
            item.get("conflict_group_id")
            for item in (active_items + pending_items)
            if item.get("conflict_group_id")
        }
        conflicts.append({
            "fact_key": key,
            "values": sorted(active_values | pending_values),
            "active_values": sorted(active_values),
            "pending_values": sorted(pending_values - active_values),
            "conflict_kind": "active_vs_pending",
            "conflict_group_ids": sorted(group_ids),
            "memories": active_items + pending_items,
        })
    return conflicts


def _merge_memory_candidates(memories: list[dict], extra: list[dict]) -> list[dict]:
    """Append candidates by memory_id without dropping richer metadata."""
    known = {item.get("memory_id") for item in memories}
    for item in extra:
        mid = item.get("memory_id")
        if mid and mid not in known:
            memories.append(item)
            known.add(mid)
    return memories


def _reserve_intent_and_pending_slots(
    memories: list[dict],
    *,
    exclusive_intents: set[str],
    limit: int,
) -> list[dict]:
    """Keep intent-matched actives and related pending rows inside the limit.

    Prevents unrelated high-scoring actives from crowding out a
    needs_review conflict partner (e.g. PostgreSQL pending for database).
    """
    if limit <= 0:
        return []

    matched_active = []
    matched_pending = []
    other = []
    for mem in memories:
        key = _memory_fact_key(mem)
        status = mem.get("status", "active")
        if exclusive_intents and key in exclusive_intents:
            if status == "needs_review":
                matched_pending.append(mem)
            else:
                matched_active.append(mem)
        else:
            other.append(mem)

    # Always leave room for at least one pending conflict partner when present.
    pending_slots = min(len(matched_pending), max(1, limit // 3)) if matched_pending else 0
    active_slots = min(len(matched_active), max(1, limit - pending_slots))
    remaining = max(0, limit - active_slots - pending_slots)
    return (
        matched_active[:active_slots]
        + matched_pending[:pending_slots]
        + other[:remaining]
    )


def semantic_search_memories(
    user_id,
    query,
    limit=5,
    min_score=0.80,
    *,
    include_needs_review=True,
    project_id=None,
    tenant_id=None,
):
    try:
        query_embedding = create_embedding(query)
        intent_keys = set(infer_query_fact_keys(query))
        exclusive_intents = intent_keys & {
            "editor",
            "preferred_language",
            "preferred_database",
            "os",
            "company",
            "role",
            "goal",
            "primary_project",
        }

        memories = _vector_search_memories(
            user_id,
            query_embedding,
            status_filter=_active_status_filter(),
            limit=limit,
            min_score=min_score,
            project_id=project_id,
            tenant_id=tenant_id,
        )
        memories = _authorize_memory_candidates(
            memories,
            user_id=user_id,
            tenant_id=tenant_id,
            project_id=project_id,
        )
        for mem in memories:
            mem["vector_score"] = mem.get("score")

        # Stage 19: Also retrieve relevant pending-review facts so specific
        # queries (e.g. Freshcart) can surface the pending memory explicitly.
        if include_needs_review:
            pending = _vector_search_memories(
                user_id,
                query_embedding,
                status_filter={"status": "needs_review"},
                limit=max(3, limit),
                # Slightly lower threshold so named pending facts are found.
                min_score=max(0.70, min_score - 0.10),
                project_id=project_id,
                tenant_id=tenant_id,
            )
            for item in pending:
                item["vector_score"] = item.get("score")
                # Require lexical overlap for pending facts to avoid noise,
                # unless the pending fact_key matches query intent.
                key = _memory_fact_key(item)
                if (
                    key in exclusive_intents
                    or _lexical_relevance_boost(query, item.get("memory", "")) > 0
                ):
                    memories = _merge_memory_candidates(memories, [item])
                    memories = _authorize_memory_candidates(
                        memories,
                        user_id=user_id,
                        tenant_id=tenant_id,
                        project_id=project_id,
                    )

        # Structured fact-key retrieval: do not rely on vector top-k alone
        # for explicitly requested facts, including non-exclusive named
        # projects recovered from legacy compound rows.
        if intent_keys:
            statuses = ("active", "needs_review") if include_needs_review else ("active",)
            for key in intent_keys:
                by_key = find_memories_by_fact_key(
                    user_id=user_id,
                    fact_key=key,
                    statuses=statuses,
                    limit=max(6, limit),
                    project_id=project_id,
                    tenant_id=tenant_id,
                )
                if not by_key.get("success"):
                    continue
                extras = []
                for item in by_key.get("memories", []):
                    # Structured hits get a neutral base score; ranking still
                    # applies intent/lexical boosts. Preserve any vector score.
                    if "vector_score" not in item:
                        item["vector_score"] = None
                    if "score" not in item or item.get("score") is None:
                        item["score"] = 0.82
                    extras.append(item)
                memories = _merge_memory_candidates(memories, extras)
                memories = _authorize_memory_candidates(
                    memories,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    project_id=project_id,
                )
                print(
                    f"[Fact-key retrieval] key={key} "
                    f"merged={len(extras)} status={list(statuses)}"
                )

        # Pull siblings that share a conflict_group_id with retrieved rows.
        if include_needs_review:
            group_ids = {
                m.get("conflict_group_id")
                for m in memories
                if m.get("conflict_group_id")
            }
            for group_id in group_ids:
                try:
                    docs = memory_collection.find(
                        {
                            "$and": [
                                build_scope_filter(
                                    user_id=user_id,
                                    tenant_id=tenant_id,
                                    project_id=project_id,
                                    include_shared_with_user=True,
                                ),
                                {"conflict_group_id": group_id},
                                {"status": {"$in": ["active", "needs_review"]}},
                            ]
                        }
                    ).limit(10)
                    extras = []
                    for doc in docs:
                        candidate = _doc_to_memory_candidate(doc, score=0.85)
                        candidate["vector_score"] = None
                        extras.append(candidate)
                    extras = filter_readable_documents(
                        extras,
                        user_id=user_id,
                        tenant_id=tenant_id,
                        project_id=project_id,
                    )
                    memories = _merge_memory_candidates(memories, extras)
                    memories = _authorize_memory_candidates(
                        memories,
                        user_id=user_id,
                        tenant_id=tenant_id,
                        project_id=project_id,
                    )
                except Exception as group_error:
                    print(f"[Conflict-group retrieval skipped]: {group_error}")

        memories = _authorize_memory_candidates(
            memories,
            user_id=user_id,
            tenant_id=tenant_id,
            project_id=project_id,
        )

        print(
            f"[Semantic search] Query: {query[:80]!r} | "
            f"authorized_results={len(memories)}"
        )

        # Stage 19: Lexical relevance reranking — penalize results
        # with high embedding similarity but zero word overlap.
        for mem in memories:
            if mem.get("vector_score") is None and mem.get("score") is not None:
                # Keep original vector score separate from adjusted ranking score.
                pass
            lexical_boost = _lexical_relevance_boost(query, mem.get("memory", ""))
            if lexical_boost == 0.0 and mem.get("status") != "needs_review":
                mem["score"] = float(mem.get("score") or 0.0) * 0.60
                print(
                    f"[Relevance penalty] id={_memory_log_id(mem)} "
                    "no_lexical_overlap"
                )
            # Project-aware intent boost (primary vs name vs description).
            intent_boost = _project_intent_boost(query, mem)
            if intent_boost != 1.0:
                mem["score"] = float(mem.get("score") or 0.0) * intent_boost
                mem["priority_score_override"] = (
                    calculate_memory_priority_score(mem) * intent_boost
                )

        memories = annotate_memories_with_lifecycle(memories)

        # Stage 19: Deduplicate textual duplicates before ranking.
        memories = _deduplicate_results(memories)

        from app.services.memory_fact_extraction import is_summary_artifact_text

        memories = [
            m
            for m in memories
            if not is_summary_artifact_text(str(m.get("memory") or ""))
        ]
        memories = _authorize_memory_candidates(
            memories,
            user_id=user_id,
            tenant_id=tenant_id,
            project_id=project_id,
        )

        def _rank_key(item):
            status_rank = 0 if item.get("status", "active") == "active" else 1
            # Intent-matched pending partners rank just after matching actives.
            key = _memory_fact_key(item)
            intent_rank = 0 if key in exclusive_intents else 1
            score = item.get("priority_score_override")
            if score is None:
                score = calculate_memory_priority_score(item)
            return (intent_rank, status_rank, -score)

        memories.sort(key=_rank_key)

        if exclusive_intents:
            memories = _reserve_intent_and_pending_slots(
                memories,
                exclusive_intents=exclusive_intents,
                limit=limit,
            )
        else:
            memories = memories[:limit]

        active_conflicts = detect_conflicting_active_facts(
            [m for m in memories if m.get("status", "active") == "active"]
        )
        pending_conflicts = (
            detect_active_pending_conflicts(memories)
            if include_needs_review
            else []
        )
        conflicts = active_conflicts + pending_conflicts

        _record_memory_access(
            user_id,
            [m for m in memories if m.get("status", "active") == "active"],
        )
        print("[Memory retrieval ranking]")
        for item in memories:
            weighted_score = calculate_memory_priority_score(item)
            freshness_score = calculate_memory_freshness_score(item)
            stale_flag = "STALE" if item.get("is_stale") else "fresh"
            vector_score = item.get("vector_score")
            vector_part = (
                f"Vector: {float(vector_score):.4f} | "
                if vector_score is not None
                else "Vector: n/a | "
            )
            print(
                f"Status: {item.get('status', 'active')} | "
                f"id={_memory_log_id(item)} | "
                f"Importance: {item.get('importance', 3)}/5 | "
                f"Confidence: {item.get('confidence', 0.75):.2f} | "
                f"{vector_part}"
                f"Adjusted: {float(item.get('score') or 0.0):.4f} | "
                f"Freshness: {freshness_score:.4f} | "
                f"Weighted: {weighted_score:.4f} | "
                f"{stale_flag}"
            )

        memories = _authorize_memory_candidates(
            memories,
            user_id=user_id,
            tenant_id=tenant_id,
            project_id=project_id,
        )

        summaries: list[dict] = []
        try:
            summary_result = search_memory_summaries(
                user_id,
                query,
                limit=min(3, max(1, limit)),
                include_outdated=False,
            )
            if summary_result.get("success"):
                summaries = [
                    s
                    for s in (summary_result.get("summaries") or [])
                    if s.get("status") == "active" and s.get("current", True)
                ]
        except Exception as summary_error:
            print(f"[Summary retrieval skipped]: {summary_error}")

        record_audit_event(
            user_id=user_id,
            event_type="memory_retrieved",
            details={
                "query": query[:100],
                "result_count": len(memories),
                "conflict_groups": len(conflicts),
                "active_conflict_groups": len(active_conflicts),
                "active_pending_conflict_groups": len(pending_conflicts),
                "summary_count": len(summaries),
            },
        )

        return {
            "success": True,
            "memories": memories,
            "conflicts": conflicts,
            "summaries": summaries,
        }

    except Exception as e:
        print(f"[Semantic search error]: {e}")
        return {"success": False, "error": str(e)}


def list_stale_memories(user_id, limit=20):
    """Identify stale facts for review without deleting them."""
    result = get_memories(user_id=user_id, limit=max(limit, 50))
    if not result.get("success"):
        return result

    stale = [item for item in result["memories"] if item.get("is_stale")]
    stale.sort(key=lambda item: item.get("freshness_score", 1.0))
    return {"success": True, "memories": stale[:limit]}


def _reinforce_idempotent_payload(user_id, memory_id, operation_id):
    """Build a success payload for an already-applied reinforce operation."""
    from bson import ObjectId

    existing = memory_collection.find_one(
        {"_id": ObjectId(memory_id), "user_id": user_id}
    )
    last_reinforced = existing.get("last_reinforced_at") if existing else None
    if isinstance(last_reinforced, datetime):
        last_reinforced = last_reinforced.isoformat()
    return {
        "success": True,
        "memory_id": memory_id,
        "last_reinforced_at": last_reinforced,
        "reinforcement_count": (
            existing.get("reinforcement_count", 0) if existing else None
        ),
        "idempotent": True,
        "operation_id": operation_id,
    }


def _load_memory_doc(memory_id: str) -> dict | None:
    from bson import ObjectId

    try:
        doc = memory_collection.find_one({"_id": ObjectId(str(memory_id))})
    except Exception:
        return None
    return doc if isinstance(doc, dict) else None


def reinforce_memory(
    user_id,
    memory_id,
    operation_id=None,
    *,
    project_id=None,
    tenant_id=None,
):
    """Reaffirm an active memory.

    Idempotency rule:
    - Same operation_id → at-most-once reinforcement effect (retry-safe).
    - Different operation_id → counts as a new reaffirmation.

    Operation IDs must be request-scoped for explicit user reaffirmations.
    Do not reuse a stable per-memory operation ID across distinct user requests.

    Concurrency: when operation_id is provided, the update only matches documents
    that have not already recorded that id in reinforce_operation_ids, so two
    concurrent retries cannot double-increment.
    """
    from bson import ObjectId

    try:
        existing_doc = _load_memory_doc(memory_id)
        if existing_doc is None:
            return {
                "success": False,
                "error": "Active memory not found for this user",
            }
        denied = require_write_or_error(
            user_id,
            existing_doc,
            tenant_id=tenant_id,
            project_id=project_id,
            operation_id=operation_id,
        )
        if denied is not None:
            return denied

        # Fast path: audit already recorded this operation.
        if operation_id and audit_collection is not None:
            prior = audit_collection.find_one(
                {
                    "user_id": user_id,
                    "memory_id": memory_id,
                    "operation_id": operation_id,
                    "event_type": "memory_reinforced",
                }
            )
            if prior:
                return _reinforce_idempotent_payload(
                    user_id, memory_id, operation_id
                )

        # Document-level idempotency (covers concurrent retries of same op).
        # Only treat as claimed when the operation_id is present in the
        # persisted list — never because a mock/empty document was returned.
        if operation_id:
            existing = memory_collection.find_one(
                {
                    "_id": ObjectId(memory_id),
                    "user_id": user_id,
                    "reinforce_operation_ids": operation_id,
                }
            )
            claimed_ids = (
                existing.get("reinforce_operation_ids")
                if isinstance(existing, dict)
                else None
            )
            if isinstance(claimed_ids, list) and operation_id in claimed_ids:
                return _reinforce_idempotent_payload(
                    user_id, memory_id, operation_id
                )

        current_time = datetime.now(timezone.utc)
        doc_user_id = (existing_doc or {}).get("user_id") or user_id
        update_doc = {
            "$inc": {"reinforcement_count": 1},
            "$set": {
                "last_reinforced_at": current_time,
                "is_stale": False,
                "stale_reason": None,
                "updated_at": current_time,
            },
        }
        query = {
            "_id": ObjectId(memory_id),
            "user_id": doc_user_id,
            **_active_status_filter(),
        }
        if operation_id:
            # Atomic claim: only update if this operation_id is new.
            query["reinforce_operation_ids"] = {"$ne": operation_id}
            update_doc["$set"]["last_reinforce_operation_id"] = operation_id
            update_doc["$addToSet"] = {"reinforce_operation_ids": operation_id}

        blocked = _begin_durable_commit()
        if blocked is not None:
            return blocked
        try:
            result = memory_collection.update_one(query, update_doc)
        except Exception:
            _finish_durable_commit(success=False)
            raise

        if result.matched_count == 0:
            _finish_durable_commit(success=False)
            # Missing/inactive, or this operation_id was claimed concurrently.
            if operation_id:
                claimed = memory_collection.find_one(
                    {
                        "_id": ObjectId(memory_id),
                        "user_id": doc_user_id,
                        "reinforce_operation_ids": operation_id,
                    }
                )
                claimed_ids = (
                    claimed.get("reinforce_operation_ids")
                    if isinstance(claimed, dict)
                    else None
                )
                if isinstance(claimed_ids, list) and operation_id in claimed_ids:
                    return _reinforce_idempotent_payload(
                        user_id, memory_id, operation_id
                    )
            return {
                "success": False,
                "error": "Active memory not found for this user",
            }
        _finish_durable_commit(success=True)

        updated = memory_collection.find_one(
            {"_id": ObjectId(memory_id), "user_id": user_id}
        )

        record_audit_event(
            user_id=user_id,
            event_type="memory_reinforced",
            memory_id=memory_id,
            operation_id=operation_id,
            details={
                "reinforcement_count": (
                    updated.get("reinforcement_count") if updated else None
                ),
            },
        )

        # Return ISO datetime strings so tool JSON serialization never fails.
        return {
            "success": True,
            "memory_id": memory_id,
            "last_reinforced_at": current_time.isoformat(),
            "reinforcement_count": (
                updated.get("reinforcement_count") if updated else None
            ),
            "idempotent": False,
            "operation_id": operation_id,
        }
    except Exception as e:
        print(f"[Memory reinforcement error]: {e}")
        return {"success": False, "error": str(e)}


def update_memory_importance(user_id, memory_id, importance):
    """Update importance without changing the memory text or embedding."""
    from bson import ObjectId

    try:
        importance = max(1, min(5, int(importance)))

        result = memory_collection.update_one(
            {
                "_id": ObjectId(memory_id),
                "user_id": user_id,
            },
            {
                "$set": {
                    "importance": importance,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )

        if result.matched_count == 0:
            return {
                "success": False,
                "error": "Memory not found for this user",
            }

        print(f"[Memory importance updated] {memory_id}: {importance}/5")
        return {
            "success": True,
            "memory_id": memory_id,
            "importance": importance,
        }

    except Exception as e:
        print(f"[Memory importance update error]: {e}")
        return {"success": False, "error": str(e)}


def check_duplicate_memory(
    user_id,
    embedding,
    duplicate_threshold=0.90,
    candidate_limit=10,
    *,
    include_needs_review=True,
):
    """Vector duplicate check.

    Stage 19 fix: include needs_review candidates so case/punctuation
    variants of a pending fact are recognized as the same logical memory.
    """
    try:
        status_match = (
            _reviewable_status_filter()
            if include_needs_review
            else _active_status_filter()
        )
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "memory_vector_index",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": max(100, candidate_limit * 20),
                    "limit": max(candidate_limit * 5, 30),
                    "filter": build_vector_search_filter(user_id=user_id),
                }
            },
            {"$match": status_match},
            {
                "$project": {
                    "_id": 1,
                    "memory": 1,
                    "user_id": 1,
                    "importance": 1,
                    "confidence": 1,
                    "status": 1,
                    "fact_key": 1,
                    "normalized_value": 1,
                    "conflict_group_id": 1,
                    "version": 1,
                    "last_accessed_at": 1,
                    "access_count": 1,
                    "last_reinforced_at": 1,
                    "reinforcement_count": 1,
                    "project_id": 1,
                    "tenant_id": 1,
                    "scope": 1,
                    "owner_id": 1,
                    "shared_with": 1,
                    "permission_level": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        results = list(memory_collection.aggregate(pipeline))[:candidate_limit]
        candidates = []

        for item in results:
            if item.get("user_id") is None:
                item["user_id"] = user_id
            if item.get("owner_id") is None:
                item["owner_id"] = item.get("user_id") or user_id
            if not check_read_permission(user_id, item, audit=False).get("allowed"):
                continue
            candidates.append({
                "memory_id": str(item["_id"]),
                "memory": item.get("memory", ""),
                "user_id": item.get("user_id"),
                "importance": item.get("importance", 3),
                "confidence": item.get("confidence", 0.75),
                "status": item.get("status", "active"),
                "fact_key": item.get("fact_key"),
                "normalized_value": item.get("normalized_value"),
                "conflict_group_id": item.get("conflict_group_id"),
                "version": item.get("version", 1),
                "last_accessed_at": item.get("last_accessed_at"),
                "access_count": item.get("access_count", 0),
                "last_reinforced_at": item.get("last_reinforced_at"),
                "reinforcement_count": item.get("reinforcement_count", 0),
                "project_id": item.get("project_id"),
                "tenant_id": item.get("tenant_id"),
                "scope": item.get("scope", "user"),
                "owner_id": item.get("owner_id") or item.get("user_id"),
                "shared_with": list(item.get("shared_with") or []),
                "permission_level": item.get("permission_level") or "private",
                "score": float(item.get("score", 0.0)),
            })

        if not candidates:
            return {
                "success": True,
                "is_duplicate": False,
                "memory": None,
                "score": None,
                "candidates": [],
            }

        # Prefer active vector matches when scoring duplicates.
        ranked = sorted(
            candidates,
            key=lambda item: (
                0 if item.get("status", "active") in ("active", None) else 1,
                -float(item.get("score", 0.0)),
            ),
        )
        best = ranked[0]
        is_duplicate = best["score"] >= duplicate_threshold

        return {
            "success": True,
            "is_duplicate": is_duplicate,
            "memory_id": best["memory_id"],
            "memory": best["memory"],
            "importance": best["importance"],
            "status": best.get("status", "active"),
            "score": best["score"],
            "candidates": ranked,
        }

    except Exception as e:
        print(f"[Duplicate check error]: {e}")
        return {"success": False, "error": str(e)}


def update_memory(user_id, memory_id, memory, embedding):
    """Deprecated: factual changes must use versioned supersede_memory()."""
    log_memory_operation(
        "update_memory_rejected",
        user_id,
        success=False,
        memory_id=memory_id,
        details={
            "error": "Direct overwrite is disabled. Use supersede_memory().",
            "memory_preview": str(memory)[:120],
            "embedding_provided": embedding is not None,
        },
    )
    return {
        "success": False,
        "error": (
            "Direct memory overwrite is disabled to preserve version history. "
            "Use supersede_memory() or confirm_memory_update."
        ),
    }


def _is_retryable_transaction_error(exc: Exception) -> bool:
    if isinstance(exc, ConnectionFailure):
        return True
    if isinstance(exc, OperationFailure):
        # TransientTransactionError / UnknownTransactionCommitResult labels.
        labels = set(getattr(exc, "error_labels", set()) or set())
        return bool(
            labels
            & {
                "TransientTransactionError",
                "UnknownTransactionCommitResult",
            }
        )
    return False


class _SupersedeAbort(Exception):
    """Non-retryable supersede failure raised inside a transaction body."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def supersede_memory(
    user_id,
    memory_id,
    memory,
    embedding,
    fact_key=None,
    normalized_value=None,
    operation_id=None,
    *,
    project_id=None,
    tenant_id=None,
):
    from bson import ObjectId

    existing_by_op = _find_by_operation_id(user_id, operation_id)
    if existing_by_op:
        return {
            "success": True,
            "old_memory_id": existing_by_op.get("supersedes_memory_id"),
            "memory_id": str(existing_by_op["_id"]),
            "memory": existing_by_op.get("memory", memory),
            "version": existing_by_op.get("version", 1),
            "status": existing_by_op.get("status", "active"),
            "idempotent": True,
        }

    blocked = _begin_durable_commit()
    if blocked is not None:
        return blocked

    preloaded = _load_memory_doc(memory_id)
    if preloaded is None:
        _finish_durable_commit(success=False)
        return {
            "success": False,
            "error": "Memory not found for this user",
            "error_code": "not_found",
            "phase": "load_existing",
        }
    denied = require_write_or_error(
        user_id,
        preloaded,
        tenant_id=tenant_id,
        project_id=project_id,
        operation_id=operation_id,
    )
    if denied is not None:
        _finish_durable_commit(success=False)
        return {
            **denied,
            "error_code": "not_found",
            "phase": "load_existing",
        }

    last_error = None
    transaction_started = False
    for attempt in range(1, TRANSACTION_MAX_RETRIES + 1):
        phase = "init"
        try:
            old_object_id = ObjectId(memory_id)
            current_time = datetime.now(timezone.utc)
            new_memory_id = None
            next_version = None

            phase = "start_session"
            with client.start_session() as session:
                phase = "start_transaction"
                transaction_started = True
                with session.start_transaction():
                    phase = "load_existing"
                    existing = memory_collection.find_one(
                        {"_id": old_object_id},
                        session=session,
                    )

                    if not existing:
                        raise _SupersedeAbort(
                            "not_found",
                            "Memory not found for this user",
                        )
                    denied_tx = require_write_or_error(
                        user_id,
                        existing,
                        tenant_id=tenant_id,
                        project_id=project_id,
                        operation_id=operation_id,
                    )
                    if denied_tx is not None:
                        raise _SupersedeAbort(
                            "permission_denied",
                            "Memory not found for this user",
                        )

                    if existing.get("status", "active") != "active":
                        raise _SupersedeAbort(
                            "not_active",
                            (
                                "Only an active memory can be superseded "
                                f"(found status={existing.get('status')})"
                            ),
                        )

                    conflict_group_id = (
                        existing.get("conflict_group_id") or uuid4().hex
                    )
                    next_version = int(existing.get("version", 1)) + 1
                    owner = existing.get("owner_id") or existing.get("user_id") or user_id

                    phase = "validate_new_document"
                    new_document = MemoryDocument(
                        user_id=existing.get("user_id") or user_id,
                        memory=memory,
                        embedding=embedding,
                        importance=existing.get("importance", 3),
                        confidence=existing.get("confidence", 0.75),
                        created_at=current_time,
                        updated_at=current_time,
                        status="active",
                        fact_key=fact_key or existing.get("fact_key"),
                        normalized_value=(
                            normalized_value
                            or existing.get("normalized_value")
                        ),
                        conflict_type="confirmed_update",
                        conflict_group_id=conflict_group_id,
                        supersedes_memory_id=str(old_object_id),
                        version=next_version,
                        last_accessed_at=None,
                        access_count=0,
                        last_reinforced_at=None,
                        reinforcement_count=0,
                        operation_id=operation_id,
                        project_id=existing.get("project_id"),
                        tenant_id=existing.get("tenant_id"),
                        scope=existing.get("scope") or "user",
                        owner_id=owner,
                        shared_with=list(existing.get("shared_with") or []),
                        permission_level=(
                            existing.get("permission_level") or "private"
                        ),
                    ).model_dump()

                    phase = "insert_new"
                    inserted = memory_collection.insert_one(
                        new_document,
                        session=session,
                    )
                    new_memory_id = str(inserted.inserted_id)

                    doc_user_id = existing.get("user_id") or user_id
                    phase = "mark_old_superseded"
                    old_update = memory_collection.update_one(
                        {
                            "_id": old_object_id,
                            "user_id": doc_user_id,
                            **_active_status_filter(),
                        },
                        {
                            "$set": {
                                "status": "superseded",
                                "superseded_by_memory_id": new_memory_id,
                                "conflict_group_id": conflict_group_id,
                                "updated_at": current_time,
                            }
                        },
                        session=session,
                    )

                    if old_update.matched_count != 1:
                        raise RuntimeError(
                            "The previous memory changed before it could be superseded"
                        )

                    phase = "resolve_pending_by_conflict"
                    memory_collection.update_many(
                        {
                            "user_id": doc_user_id,
                            "status": "needs_review",
                            "conflict_with_memory_id": str(old_object_id),
                        },
                        {
                            "$set": {
                                "status": "resolved",
                                "updated_at": current_time,
                            }
                        },
                        session=session,
                    )

                    resolved_key = fact_key or existing.get("fact_key")
                    if resolved_key:
                        phase = "resolve_pending_by_fact_key"
                        memory_collection.update_many(
                            {
                                "user_id": doc_user_id,
                                "status": "needs_review",
                                "fact_key": resolved_key,
                                "_id": {"$ne": inserted.inserted_id},
                            },
                            {
                                "$set": {
                                    "status": "resolved",
                                    "updated_at": current_time,
                                    "superseded_by_memory_id": new_memory_id,
                                }
                            },
                            session=session,
                        )

                    phase = "commit"

            phase = "audit"
            record_audit_event(
                user_id=user_id,
                event_type="memory_superseded",
                memory_id=new_memory_id,
                operation_id=operation_id,
                details={
                    "old_memory_id": memory_id,
                    "version": next_version,
                    "attempt": attempt,
                },
            )

            log_memory_operation(
                "supersede_memory",
                user_id,
                success=True,
                memory_id=new_memory_id,
                operation_id=operation_id,
                details={
                    "old_memory_id": memory_id,
                    "version": next_version,
                    "phase": "committed",
                },
            )

            print(
                f"[Memory version created] old={memory_id}, "
                f"new={new_memory_id}, version={next_version}"
            )

            schedule_summary_refresh(
                user_id,
                reason="memory_superseded",
                memory=memory,
                fact_key=fact_key,
            )

            _finish_durable_commit(success=True)
            return {
                "success": True,
                "old_memory_id": memory_id,
                "memory_id": new_memory_id,
                "memory": memory,
                "version": next_version,
                "status": "active",
                "idempotent": False,
            }

        except _SupersedeAbort as e:
            _finish_durable_commit(success=False)
            log_memory_operation(
                "supersede_memory",
                user_id,
                success=False,
                memory_id=memory_id,
                operation_id=operation_id,
                details={
                    "error_type": type(e).__name__,
                    "error_code": e.code,
                    "error": e.message,
                    "phase": phase,
                    "transaction_started": transaction_started,
                    "transaction_result": "rolled_back_or_aborted",
                },
            )
            return {
                "success": False,
                "error": e.message,
                "error_code": e.code,
                "phase": phase,
            }

        except Exception as e:
            last_error = e
            from app.utils.memory_logging import logger as memory_logger

            memory_logger.exception(
                "supersede_memory failed user_id=%s memory_id=%s phase=%s attempt=%s",
                user_id,
                memory_id,
                phase,
                attempt,
            )
            log_memory_operation(
                "supersede_memory",
                user_id,
                success=False,
                memory_id=memory_id,
                operation_id=operation_id,
                details={
                    "error_type": type(e).__name__,
                    "error": str(e)[:500],
                    "phase": phase,
                    "attempt": attempt,
                    "transaction_started": transaction_started,
                    "transaction_result": "rolled_back_or_aborted",
                },
            )
            if attempt < TRANSACTION_MAX_RETRIES and _is_retryable_transaction_error(e):
                sleep_for = 0.05 * attempt
                print(
                    f"[Memory supersede retry {attempt}/{TRANSACTION_MAX_RETRIES}] "
                    f"phase={phase} waiting {sleep_for:.2f}s: {type(e).__name__}: {e}"
                )
                time.sleep(sleep_for)
                continue
            print(
                f"[Memory supersede error] phase={phase} "
                f"{type(e).__name__}: {e}"
            )
            _finish_durable_commit(success=False)
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "phase": phase,
            }

    _finish_durable_commit(success=False)
    return {
        "success": False,
        "error": f"Transaction failed after retries: {last_error}",
        "phase": "exhausted_retries",
    }


def activate_confirmed_memory(
    user_id,
    memory,
    embedding,
    *,
    fact_key=None,
    normalized_value=None,
    importance=3,
    confidence=0.95,
    operation_id=None,
):
    """Create a new active memory and resolve pending rows for the same fact key.

    Used when the user confirms an update but no active row exists to supersede
    (only needs_review proposals remain).
    """
    save_result = save_memory(
        user_id=user_id,
        memory=memory,
        embedding=embedding,
        importance=importance,
        confidence=confidence,
        status="active",
        fact_key=fact_key,
        normalized_value=normalized_value,
        conflict_type="confirmed_update",
        version=1,
        operation_id=operation_id,
    )
    if not save_result.get("success"):
        return save_result

    new_id = save_result.get("memory_id")
    if fact_key and new_id:
        try:
            current_time = datetime.now(timezone.utc)
            memory_collection.update_many(
                {
                    "user_id": user_id,
                    "fact_key": fact_key,
                    "status": "needs_review",
                },
                {
                    "$set": {
                        "status": "resolved",
                        "superseded_by_memory_id": new_id,
                        "updated_at": current_time,
                    }
                },
            )
            retire_sibling_exclusive_actives(
                user_id=user_id,
                fact_key=fact_key,
                keep_memory_id=new_id,
                superseded_by_memory_id=new_id,
            )
        except Exception as e:
            from app.utils.memory_logging import logger as memory_logger

            memory_logger.exception(
                "activate_confirmed_memory cleanup failed user_id=%s",
                user_id,
            )
            log_memory_operation(
                "activate_confirmed_memory_cleanup",
                user_id,
                success=False,
                memory_id=new_id,
                details={"error_type": type(e).__name__, "error": str(e)[:500]},
            )

    schedule_summary_refresh(
        user_id,
        reason="conflict_resolved",
        memory=memory,
        fact_key=fact_key,
    )

    return {
        "success": True,
        "memory_id": new_id,
        "memory": memory,
        "version": 1,
        "status": "active",
        "activated_from_pending": True,
        "idempotent": bool(save_result.get("idempotent")),
    }


def retire_sibling_exclusive_actives(
    user_id: str,
    fact_key: str,
    *,
    keep_memory_id: str,
    superseded_by_memory_id: str,
):
    """Mark other active rows for an exclusive fact_key as superseded.

    Used when the user explicitly confirms a single new value while multiple
    conflicting active memories still exist for the same key.
    """
    from bson import ObjectId

    if not fact_key or not keep_memory_id or not superseded_by_memory_id:
        return {"success": False, "error": "fact_key and memory ids are required"}

    try:
        keep_oid = ObjectId(keep_memory_id)
        current_time = datetime.now(timezone.utc)
        result = memory_collection.update_many(
            {
                "user_id": user_id,
                "fact_key": {"$in": _fact_key_query_values(fact_key)},
                "status": "active",
                "_id": {"$ne": keep_oid},
            },
            {
                "$set": {
                    "status": "superseded",
                    "superseded_by_memory_id": superseded_by_memory_id,
                    "updated_at": current_time,
                }
            },
        )
        return {
            "success": True,
            "retired_count": int(result.modified_count),
        }
    except Exception as e:
        print(f"[Retire sibling actives error]: {e}")
        return {"success": False, "error": str(e)}


def delete_memory(
    user_id,
    memory_id,
    *,
    soft=True,
    operation_id=None,
    project_id=None,
    tenant_id=None,
):
    """Forget a memory, always scoped to the authenticated user.

    soft=True (default) marks the memory as forgotten without hard deletion.
    """
    from bson import ObjectId

    try:
        object_id = ObjectId(memory_id)
        current_time = datetime.now(timezone.utc)
        existing_doc = _load_memory_doc(memory_id)
        denied = require_write_or_error(
            user_id,
            existing_doc,
            tenant_id=tenant_id,
            project_id=project_id,
            operation_id=operation_id,
        )
        if denied is not None:
            return denied

        owner = memory_owner_id(existing_doc)
        owner_filter: dict = {"_id": object_id}
        if owner:
            owner_filter["owner_id"] = owner
        else:
            owner_filter["user_id"] = existing_doc.get("user_id", user_id)
        if existing_doc.get("tenant_id"):
            owner_filter["tenant_id"] = existing_doc.get("tenant_id")
        if existing_doc.get("project_id"):
            owner_filter["project_id"] = existing_doc.get("project_id")

        if soft:
            owner_filter["status"] = {"$nin": ["forgotten", "superseded"]}
            blocked = _begin_durable_commit()
            if blocked is not None:
                return blocked
            try:
                result = memory_collection.update_one(
                    owner_filter,
                    {
                        "$set": {
                            "status": "forgotten",
                            "updated_at": current_time,
                            "operation_id": operation_id,
                        }
                    },
                )
                _finish_durable_commit(success=True)
            except Exception:
                _finish_durable_commit(success=False)
                raise
            if result.matched_count == 0:
                # Idempotent soft-forget if already forgotten by this user.
                existing = memory_collection.find_one(
                    {
                        "_id": object_id,
                        "user_id": existing_doc.get("user_id", user_id),
                        "status": "forgotten",
                    }
                )
                if existing:
                    return {
                        "success": True,
                        "message": "Memory already forgotten",
                        "memory_id": memory_id,
                        "idempotent": True,
                    }
                return {
                    "success": False,
                    "error": "Memory not found for this user",
                }

            record_audit_event(
                user_id=user_id,
                event_type="memory_forgotten",
                memory_id=memory_id,
                operation_id=operation_id,
                details={"soft": True},
            )
            schedule_summary_refresh(
                user_id,
                reason="memory_forgotten",
            )
            return {
                "success": True,
                "message": "Memory forgotten",
                "memory_id": memory_id,
                "idempotent": False,
            }

        hard_filter = {k: v for k, v in owner_filter.items() if k != "status"}
        result = memory_collection.delete_one(hard_filter)

        if result.deleted_count == 0:
            return {
                "success": False,
                "error": "Memory not found for this user",
            }

        record_audit_event(
            user_id=user_id,
            event_type="memory_deleted",
            memory_id=memory_id,
            operation_id=operation_id,
            details={"soft": False},
        )
        return {
            "success": True,
            "message": "Memory deleted",
            "memory_id": memory_id,
        }

    except Exception as e:
        print(f"[Memory delete error]: {e}")
        return {"success": False, "error": str(e)}


# ============================================================
# SAFE DEDUPLICATION (Stage 19)
# ============================================================

def safe_deduplicate_memories(user_id: str, dry_run: bool = True) -> dict:
    """Identify and optionally resolve duplicate memory records.

    dry_run=True (default) only identifies duplicates without changing data.
    dry_run=False marks extras as 'deduplicated', preserving the original.
    """
    from app.services.memory_conflict_service import normalize_memory_text

    try:
        result = get_memories(user_id=user_id, limit=500)
        if not result.get("success"):
            return result

        memories = result["memories"]

        # Group by normalized text.
        groups: dict[str, list[dict]] = {}
        for mem in memories:
            norm = normalize_memory_text(mem.get("memory", ""))
            if not norm:
                continue
            groups.setdefault(norm, []).append(mem)

        duplicate_groups = [g for g in groups.values() if len(g) >= 2]

        if not duplicate_groups:
            return {
                "success": True,
                "duplicate_groups": 0,
                "duplicates_found": 0,
                "resolved": 0,
                "dry_run": dry_run,
            }

        total_duplicates = sum(len(g) - 1 for g in duplicate_groups)
        resolved = 0

        if not dry_run:
            from bson import ObjectId

            current_time = datetime.now(timezone.utc)
            for group in duplicate_groups:
                # Keep the most recently updated memory as the canonical one.
                group.sort(
                    key=lambda m: m.get("updated_at") or m.get("created_at") or current_time,
                    reverse=True,
                )
                canonical = group[0]

                for extra in group[1:]:
                    extra_id = extra.get("memory_id")
                    if not extra_id:
                        continue
                    memory_collection.update_one(
                        {
                            "_id": ObjectId(extra_id),
                            "user_id": user_id,
                            **_active_status_filter(),
                        },
                        {
                            "$set": {
                                "status": "deduplicated",
                                "updated_at": current_time,
                                "superseded_by_memory_id": canonical["memory_id"],
                            }
                        },
                    )
                    record_audit_event(
                        user_id=user_id,
                        event_type="memory_deduplicated",
                        memory_id=extra_id,
                        details={
                            "canonical_memory_id": canonical["memory_id"],
                            "normalized_text": normalize_memory_text(
                                extra.get("memory", "")
                            )[:100],
                        },
                    )
                    resolved += 1

        return {
            "success": True,
            "duplicate_groups": len(duplicate_groups),
            "duplicates_found": total_duplicates,
            "resolved": resolved,
            "dry_run": dry_run,
            "groups": [
                {
                    "normalized_text": normalize_memory_text(g[0].get("memory", ""))[:100],
                    "count": len(g),
                    "memory_ids": [m.get("memory_id") for m in g],
                }
                for g in duplicate_groups
            ],
        }

    except Exception as e:
        print(f"[Deduplication error]: {e}")
        return {"success": False, "error": str(e)}


def _legacy_compound_fact_candidates(
    *,
    user_id: str,
    fact_key: str,
    statuses: list[str],
    include_missing_active: bool,
    limit: int,
    project_id: str | None,
    tenant_id: str | None,
) -> list[dict]:
    """Recover additional indexed facts from old compound memory rows."""
    from app.services.memory_conflict_service import normalize_fact_key
    from app.services.memory_fact_extraction import extract_facts_from_message

    status_filter: dict = {"status": {"$in": statuses}}
    if include_missing_active:
        status_filter = {
            "$or": [
                {"status": {"$in": statuses}},
                {"status": {"$exists": False}},
            ]
        }
    query = {
        "$and": [
            build_scope_filter(
                user_id=user_id,
                tenant_id=tenant_id,
                project_id=project_id,
                include_shared_with_user=True,
            ),
            status_filter,
        ]
    }
    target_key = normalize_fact_key(str(fact_key))
    matches: list[dict] = []
    for item in memory_collection.find(query).sort("updated_at", -1).limit(
        max(limit * 20, 100)
    ):
        extracted = extract_facts_from_message(
            "Remember that " + str(item.get("memory") or "")
        )
        matching_fact = next(
            (
                fact
                for fact in extracted.facts
                if fact.fact_key
                and normalize_fact_key(fact.fact_key) == target_key
            ),
            None,
        )
        if matching_fact is None:
            continue
        candidate = _doc_to_memory_candidate(item, fallback_user_id=user_id)
        candidate["fact_key"] = target_key
        candidate["normalized_value"] = matching_fact.normalized_value
        matches.append(candidate)
        if len(matches) >= limit:
            break
    return matches
