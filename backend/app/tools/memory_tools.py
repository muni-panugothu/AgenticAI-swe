"""LLM-callable memory tools. user_id is injected by the orchestrator.

Stage 19: Improved conflict detection, idempotent operations,
deduplication-aware saves, and structured audit trail.
"""

from __future__ import annotations

from uuid import uuid4

from app.services.embedding_service import create_embedding
from app.services.memory_conflict_service import (
    classify_duplicate,
    detect_memory_conflict,
    extract_exclusive_fact,
    normalize_memory_text,
)
from app.services.memory_fact_extraction import (
    is_question_only,
    is_summary_artifact_text,
    strip_trailing_question,
)
from app.services.memory_service import (
    check_duplicate_memory,
    delete_memory,
    ensure_conflict_partner_metadata,
    find_active_memories_by_fact_key,
    find_logical_duplicate,
    format_memory_metadata_report,
    inspect_memory_metadata,
    persist_memory_conflict_pair,
    reinforce_memory,
    save_memory,
    semantic_search_memories,
    supersede_memory,
)
from app.services.memory_summary_service import (
    expand_summary_sources,
    search_memory_summaries,
    summarize_user_memories,
)


def _importance_from_text(memory: str) -> int:
    text = memory.lower()
    if any(term in text for term in ("career goal", "long-term goal", "my goal is")):
        return 5
    if any(term in text for term in ("i am learning", "i prefer", "i am working on")):
        return 4
    return 3


def _confidence_from_text(memory: str) -> float:
    text = memory.lower()
    if any(p in text for p in ("i think", "maybe", "not sure", "probably")):
        return 0.60
    if any(p in text for p in ("i prefer", "i am learning", "i work as", "my goal is")):
        return 0.95
    return 0.75


def search_memory(
    user_id: str,
    query: str,
    limit: int = 5,
    *,
    project_id: str | None = None,
    tenant_id: str | None = None,
) -> dict:
    result = semantic_search_memories(
        user_id=user_id,
        query=query,
        limit=max(1, min(int(limit), 10)),
        min_score=0.70,
        project_id=project_id,
        tenant_id=tenant_id,
    )
    if not result.get("success"):
        return result

    memories = []
    for item in result.get("memories", []):
        memories.append({
            "memory_id": item.get("memory_id"),
            "memory": item.get("memory"),
            "importance": item.get("importance"),
            "confidence": item.get("confidence"),
            "freshness_score": item.get("freshness_score"),
            "priority_score": item.get("priority_score"),
            "is_stale": item.get("is_stale", False),
            "version": item.get("version", 1),
            "fact_key": item.get("fact_key"),
            "status": item.get("status"),
        })
    summaries = []
    for item in result.get("summaries") or []:
        if item.get("status") != "active":
            continue
        summaries.append({
            "summary_id": item.get("summary_id"),
            "summary_type": item.get("summary_type"),
            "summary_text": item.get("summary_text"),
            "source_memory_count": item.get("source_memory_count"),
            "conflict_references": item.get("conflict_references") or [],
            "status": item.get("status"),
        })
    return {
        "success": True,
        "memories": memories,
        "summaries": summaries,
        "conflicts": result.get("conflicts") or [],
    }


def summarize_memories_tool(
    user_id: str,
    summary_type: str | None = None,
    scope_key: str | None = None,
    request_id: str | None = None,
    *,
    project_id: str | None = None,
    tenant_id: str | None = None,
) -> dict:
    """Generate or refresh conflict-aware memory summaries for the user."""
    types = [summary_type] if summary_type else None
    if summary_type and summary_type not in {
        "user",
        "project",
        "preferences",
        "general",
    }:
        return {
            "success": False,
            "error": "summary_type must be user, project, preferences, or general",
        }
    op_id = request_id or uuid4().hex
    return summarize_user_memories(
        user_id,
        summary_types=types,
        scope_key=scope_key,
        operation_id=f"summarize-{op_id}",
        use_llm=False,
        force=False,
        project_id=project_id,
        tenant_id=tenant_id,
    )


def search_memory_summaries_tool(
    user_id: str,
    query: str,
    limit: int = 5,
) -> dict:
    return search_memory_summaries(
        user_id=user_id,
        query=query,
        limit=max(1, min(int(limit), 10)),
        include_outdated=False,
    )


def expand_memory_summary_tool(user_id: str, summary_id: str) -> dict:
    if not summary_id:
        return {"success": False, "error": "summary_id is required"}
    return expand_summary_sources(user_id=user_id, summary_id=summary_id)


def save_memory_tool(
    user_id: str,
    memory: str,
    request_id: str | None = None,
    *,
    project_id: str | None = None,
    tenant_id: str | None = None,
    permission_level: str | None = None,
    shared_with: list[str] | None = None,
) -> dict:
    text = strip_trailing_question((memory or "").strip()).rstrip(".!?")
    if not text:
        return {"success": False, "error": "Memory text is required"}
    if is_question_only(text) or is_summary_artifact_text(text):
        return {
            "success": False,
            "error": "Refusing to store questions or summary digests as memories",
        }

    try:
        embedding = create_embedding(text)
    except Exception as exc:
        return {"success": False, "error": f"Embedding failed: {exc}"}

    fact = extract_exclusive_fact(text)

    # Structured duplicate check across active + needs_review first.
    logical = find_logical_duplicate(
        user_id=user_id,
        memory_text=text,
        fact_key=fact[0] if fact else None,
        normalized_value=fact[1] if fact else None,
    )
    if not logical.get("success"):
        return logical

    if logical.get("is_duplicate"):
        match = logical["match"]
        if match.get("status") == "needs_review":
            return {
                "success": True,
                "status": "needs_review",
                "memory_id": match["memory_id"],
                "message": (
                    "Duplicate of an existing pending-review memory. "
                    "No new record created."
                ),
            }

        # Request-scoped reinforce ID: retries share request_id; new calls do not.
        scope = request_id or uuid4().hex
        reinforced = reinforce_memory(
            user_id=user_id,
            memory_id=match["memory_id"],
            operation_id=f"reinforce-tool-{user_id}-{match['memory_id']}-{scope}",
        )
        return {
            "success": True,
            "status": "reinforced",
            "memory_id": match["memory_id"],
            "reinforced": reinforced.get("success", False),
            "idempotent": reinforced.get("idempotent", False),
            "message": "Duplicate memory reinforced instead of inserting a copy.",
        }

    duplicate = check_duplicate_memory(
        user_id=user_id,
        embedding=embedding,
        include_needs_review=True,
    )
    if not duplicate.get("success"):
        return duplicate

    candidates = list(duplicate.get("candidates") or [])

    if fact:
        by_key = find_active_memories_by_fact_key(user_id, fact[0])
        if by_key.get("success"):
            known_ids = {item["memory_id"] for item in candidates}
            for item in by_key.get("memories", []):
                if item["memory_id"] not in known_ids:
                    candidates.append(item)

    conflict = detect_memory_conflict(text, candidates)
    if conflict.get("has_conflict"):
        top = conflict["top_conflict"]
        # Content-scoped conflict op id prevents case-variant pending duplicates.
        op_id = (
            f"save-conflict-{user_id}-"
            f"{conflict.get('fact_key')}-{conflict.get('normalized_value')}"
        )
        group_id = top.get("conflict_group_id") or uuid4().hex
        pending = save_memory(
            user_id=user_id,
            memory=text,
            embedding=embedding,
            importance=_importance_from_text(text),
            confidence=_confidence_from_text(text),
            status="needs_review",
            fact_key=conflict.get("fact_key"),
            normalized_value=conflict.get("normalized_value"),
            conflict_type="possible_contradiction",
            conflict_confidence=top.get("conflict_confidence"),
            conflict_group_id=group_id,
            conflict_with_memory_id=top.get("memory_id"),
            version=1,
            operation_id=op_id,
            project_id=project_id,
            tenant_id=tenant_id,
            permission_level=permission_level,
            shared_with=shared_with,
        )
        if top.get("memory_id") and pending.get("memory_id") and conflict.get("fact_key"):
            pair = persist_memory_conflict_pair(
                user_id,
                str(top["memory_id"]),
                str(pending["memory_id"]),
                fact_key=conflict["fact_key"],
                conflict_group_id=group_id,
                conflict_type="possible_contradiction",
                right_normalized_value=conflict.get("normalized_value"),
                dry_run=False,
            )
            if not pair.get("success"):
                return {
                    "success": False,
                    "error": pair.get("error", "Conflict pair persistence failed"),
                    "pending_memory_id": pending.get("memory_id"),
                }
        return {
            "success": True,
            "status": "needs_review",
            "requires_confirmation": True,
            "conflict": top,
            "pending_memory_id": pending.get("memory_id"),
            "message": (
                "Possible contradiction detected. Existing fact was not replaced. "
                "Call confirm_memory_update after the user confirms."
            ),
        }

    if duplicate.get("is_duplicate"):
        classification = classify_duplicate(
            text,
            duplicate.get("memory") or "",
            vector_score=duplicate.get("score"),
        )
        clf = classification.get("classification")
        if clf == "possible_contradiction":
            top = {
                "memory_id": duplicate.get("memory_id"),
                "memory": duplicate.get("memory"),
                "conflict_confidence": classification.get("confidence"),
                "conflict_group_id": uuid4().hex,
            }
            op_id = (
                f"save-conflict-{user_id}-"
                f"{(fact[0] if fact else 'fact')}-"
                f"{(fact[1] if fact else text)}"
            )
            pending = save_memory(
                user_id=user_id,
                memory=text,
                embedding=embedding,
                importance=_importance_from_text(text),
                confidence=_confidence_from_text(text),
                status="needs_review",
                fact_key=fact[0] if fact else None,
                normalized_value=fact[1] if fact else None,
                conflict_type="possible_contradiction",
                conflict_confidence=classification.get("confidence"),
                conflict_group_id=top["conflict_group_id"],
                conflict_with_memory_id=duplicate.get("memory_id"),
                version=1,
                operation_id=op_id,
                project_id=project_id,
                tenant_id=tenant_id,
                permission_level=permission_level,
                shared_with=shared_with,
            )
            if (
                duplicate.get("memory_id")
                and pending.get("memory_id")
                and fact
            ):
                pair = persist_memory_conflict_pair(
                    user_id,
                    str(duplicate["memory_id"]),
                    str(pending["memory_id"]),
                    fact_key=fact[0],
                    conflict_group_id=top["conflict_group_id"],
                    conflict_type="possible_contradiction",
                    right_normalized_value=fact[1],
                    dry_run=False,
                )
                if not pair.get("success"):
                    return {
                        "success": False,
                        "error": pair.get(
                            "error", "Conflict pair persistence failed"
                        ),
                        "pending_memory_id": pending.get("memory_id"),
                    }
            return {
                "success": True,
                "status": "needs_review",
                "requires_confirmation": True,
                "conflict": top,
                "pending_memory_id": pending.get("memory_id"),
                "message": (
                    "Possible contradiction detected. Existing fact was not "
                    "replaced. Call confirm_memory_update after the user confirms."
                ),
            }

        if clf in {"exact_duplicate", "paraphrase_duplicate"} and (
            duplicate.get("status", "active") == "active"
        ):
            scope = request_id or uuid4().hex
            reinforced = reinforce_memory(
                user_id=user_id,
                memory_id=duplicate["memory_id"],
                operation_id=(
                    f"reinforce-tool-{user_id}-{duplicate['memory_id']}-{scope}"
                ),
            )
            return {
                "success": True,
                "status": "reinforced",
                "memory_id": duplicate["memory_id"],
                "reinforced": reinforced.get("success", False),
                "idempotent": reinforced.get("idempotent", False),
                "message": (
                    "Duplicate memory reinforced instead of inserting a copy."
                ),
            }

    op_id = f"save-tool-{user_id}-{normalize_memory_text(text)[:60]}"
    saved = save_memory(
        user_id=user_id,
        memory=text,
        embedding=embedding,
        importance=_importance_from_text(text),
        confidence=_confidence_from_text(text),
        fact_key=fact[0] if fact else None,
        normalized_value=fact[1] if fact else None,
        operation_id=op_id,
        project_id=project_id,
        tenant_id=tenant_id,
        permission_level=permission_level,
        shared_with=shared_with,
    )
    return saved


def detect_memory_conflict_tool(user_id: str, memory: str) -> dict:
    text = (memory or "").strip().rstrip(".!?")
    if not text:
        return {"success": False, "error": "Memory text is required"}

    try:
        embedding = create_embedding(text)
    except Exception as exc:
        return {"success": False, "error": f"Embedding failed: {exc}"}

    duplicate = check_duplicate_memory(user_id=user_id, embedding=embedding)
    candidates = list(duplicate.get("candidates") or []) if duplicate.get("success") else []

    fact = extract_exclusive_fact(text)
    if fact:
        by_key = find_active_memories_by_fact_key(user_id, fact[0])
        if by_key.get("success"):
            known_ids = {item["memory_id"] for item in candidates}
            for item in by_key.get("memories", []):
                if item["memory_id"] not in known_ids:
                    candidates.append(item)

    return detect_memory_conflict(text, candidates)


def request_memory_update(
    user_id: str,
    memory_id: str,
    new_memory: str,
) -> dict:
    """Stage a factual update for explicit confirmation; do not overwrite yet."""
    text = (new_memory or "").strip().rstrip(".!?")
    if not text or not memory_id:
        return {"success": False, "error": "memory_id and new_memory are required"}

    try:
        embedding = create_embedding(text)
    except Exception as exc:
        return {"success": False, "error": f"Embedding failed: {exc}"}

    fact = extract_exclusive_fact(text)
    op_id = f"request-update-{user_id}-{memory_id}-{normalize_memory_text(text)[:40]}"
    pending = save_memory(
        user_id=user_id,
        memory=text,
        embedding=embedding,
        importance=_importance_from_text(text),
        confidence=_confidence_from_text(text),
        status="needs_review",
        fact_key=fact[0] if fact else None,
        normalized_value=fact[1] if fact else None,
        conflict_type="possible_contradiction",
        conflict_confidence=0.9,
        conflict_group_id=uuid4().hex,
        conflict_with_memory_id=memory_id,
        version=1,
        operation_id=op_id,
    )
    if not pending.get("success"):
        return pending

    return {
        "success": True,
        "status": "needs_review",
        "pending_memory_id": pending["memory_id"],
        "target_memory_id": memory_id,
        "requires_confirmation": True,
        "message": (
            "Update staged for review. Call confirm_memory_update with the "
            "target memory_id after the user confirms the change."
        ),
        "proposed_memory": text,
    }


def confirm_memory_update(
    user_id: str,
    memory_id: str,
    new_memory: str,
) -> dict:
    """Apply a confirmed factual change through versioned supersede."""
    text = (new_memory or "").strip().rstrip(".!?")
    if not text or not memory_id:
        return {"success": False, "error": "memory_id and new_memory are required"}

    try:
        embedding = create_embedding(text)
    except Exception as exc:
        return {"success": False, "error": f"Embedding failed: {exc}"}

    fact = extract_exclusive_fact(text)
    op_id = f"confirm-update-{user_id}-{memory_id}-{normalize_memory_text(text)[:40]}"
    return supersede_memory(
        user_id=user_id,
        memory_id=memory_id,
        memory=text,
        embedding=embedding,
        fact_key=fact[0] if fact else None,
        normalized_value=fact[1] if fact else None,
        operation_id=op_id,
    )


def reinforce_memory_tool(
    user_id: str,
    memory_id: str,
    request_id: str | None = None,
    *,
    project_id: str | None = None,
    tenant_id: str | None = None,
) -> dict:
    if not memory_id:
        return {"success": False, "error": "memory_id is required"}
    # Request-scoped operation ID:
    # - Same chat request / agent retry → same operation_id → idempotent
    # - Distinct user messages → distinct request_id → new reinforcement
    scope = request_id or uuid4().hex
    op_id = f"reinforce-tool-{user_id}-{memory_id}-{scope}"
    return reinforce_memory(
        user_id=user_id,
        memory_id=memory_id,
        operation_id=op_id,
        project_id=project_id,
        tenant_id=tenant_id,
    )


def forget_memory(
    user_id: str,
    memory_id: str,
    *,
    project_id: str | None = None,
    tenant_id: str | None = None,
) -> dict:
    if not memory_id:
        return {"success": False, "error": "memory_id is required"}
    return delete_memory(
        user_id=user_id,
        memory_id=memory_id,
        soft=True,
        operation_id=f"forget-{uuid4().hex}",
        project_id=project_id,
        tenant_id=tenant_id,
    )


def inspect_memory_metadata_tool(user_id: str, fact_key: str) -> dict:
    """Read-only metadata inspection for an exclusive fact key."""
    result = inspect_memory_metadata(user_id=user_id, fact_key=fact_key)
    if not result.get("success"):
        return result
    return {
        **result,
        "message": format_memory_metadata_report(result),
    }
