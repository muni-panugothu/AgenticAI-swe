"""
Dry-run / apply repair for legacy combined factual memory rows.

Identifies memories that contain multiple independent exclusive facts glued
into one document (e.g. \"VS Code.I prefer IntelliJ...\"), proposes splits,
and optionally applies a non-destructive repair:

- Original row is marked status=deduplicated (not deleted).
- Each extracted fact is inserted as needs_review (or skipped if an equivalent
  active/needs_review fact already exists).
- Audit events are written for every change.

Usage:
    python -m scripts.repair_combined_memories --user-id muni
    python -m scripts.repair_combined_memories --user-id muni --apply

Never runs against production without --apply. Default is dry-run.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from uuid import uuid4

from app.services import memory_service
from app.services.memory_audit_service import record_audit_event
from app.services.memory_fact_extraction import (
    extract_fact_from_statement,
    split_statements,
)


def _is_combined_candidate(text: str) -> bool:
    parts = split_statements(text or "")
    if len(parts) < 2:
        return False
    facts = [extract_fact_from_statement(p) for p in parts]
    extracted = [f for f in facts if f is not None]
    exclusive = [f for f in extracted if f.exclusive and f.fact_key]
    # Flag when two exclusive facts are glued, OR any multi-fact glue
    # (e.g. project name + marathon training).
    return len(exclusive) >= 2 or len(extracted) >= 2


def propose_repairs(user_id: str, *, limit: int = 200) -> list[dict]:
    collection = memory_service.memory_collection
    query = {
        "user_id": user_id,
        "status": {"$in": ["active", "needs_review"]},
    }
    proposals = []
    for doc in collection.find(query).limit(limit):
        text = doc.get("memory") or ""
        if not _is_combined_candidate(text):
            continue
        parts = split_statements(text)
        facts = []
        for part in parts:
            fact = extract_fact_from_statement(part)
            if fact and fact.exclusive and fact.fact_key:
                facts.append(
                    {
                        "statement": fact.statement,
                        "fact_key": fact.fact_key,
                        "normalized_value": fact.normalized_value,
                    }
                )
        proposals.append(
            {
                "memory_id": str(doc["_id"]),
                "status": doc.get("status"),
                "original_memory": text,
                "proposed_facts": facts,
            }
        )
    return proposals


def apply_repairs(user_id: str, proposals: list[dict]) -> dict:
    """Apply approved proposals. Does not delete originals."""
    collection = memory_service.memory_collection
    applied = 0
    created = 0
    skipped = 0
    now = datetime.now(timezone.utc)

    for proposal in proposals:
        memory_id = proposal["memory_id"]
        from bson import ObjectId

        original = collection.find_one(
            {"_id": ObjectId(memory_id), "user_id": user_id}
        )
        if not original:
            skipped += 1
            continue

        # Mark original as deduplicated (preserve text + audit trail).
        collection.update_one(
            {"_id": ObjectId(memory_id), "user_id": user_id},
            {
                "$set": {
                    "status": "deduplicated",
                    "stale_reason": "legacy_combined_memory_split",
                    "updated_at": now,
                }
            },
        )
        record_audit_event(
            user_id=user_id,
            event_type="legacy_combined_memory_marked",
            memory_id=memory_id,
            operation_id=f"repair-mark-{memory_id}",
            details={"proposed_fact_count": len(proposal["proposed_facts"])},
        )

        for fact in proposal["proposed_facts"]:
            # Skip if an equivalent fact already exists.
            existing = collection.find_one(
                {
                    "user_id": user_id,
                    "fact_key": fact["fact_key"],
                    "normalized_value": fact["normalized_value"],
                    "status": {"$in": ["active", "needs_review"]},
                }
            )
            if existing:
                skipped += 1
                continue

            insert_doc = {
                "user_id": user_id,
                "memory": fact["statement"],
                "embedding": None,
                "importance": original.get("importance", 3),
                "confidence": 0.7,
                "created_at": now,
                "updated_at": now,
                "status": "needs_review",
                "fact_key": fact["fact_key"],
                "normalized_value": fact["normalized_value"],
                "conflict_type": "possible_contradiction",
                "conflict_group_id": original.get("conflict_group_id")
                or uuid4().hex,
                "version": 1,
                "access_count": 0,
                "reinforcement_count": 0,
                "is_stale": False,
                "operation_id": (
                    f"repair-split-{user_id}-{fact['fact_key']}-"
                    f"{fact['normalized_value']}-{uuid4().hex[:8]}"
                ),
                "supersedes_memory_id": memory_id,
            }
            inserted = collection.insert_one(insert_doc)
            created += 1
            record_audit_event(
                user_id=user_id,
                event_type="legacy_combined_memory_split_fact",
                memory_id=str(inserted.inserted_id),
                operation_id=insert_doc["operation_id"],
                details={
                    "source_memory_id": memory_id,
                    "fact_key": fact["fact_key"],
                    "normalized_value": fact["normalized_value"],
                },
            )

        applied += 1

    return {
        "applied_sources": applied,
        "created_facts": created,
        "skipped": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair legacy combined memory rows (dry-run by default)."
    )
    parser.add_argument("--user-id", required=True, help="Target user_id only")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes. Without this flag, only prints a dry-run report.",
    )
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    proposals = propose_repairs(args.user_id, limit=args.limit)
    report = {
        "user_id": args.user_id,
        "mode": "apply" if args.apply else "dry-run",
        "candidate_count": len(proposals),
        "candidates": proposals,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if not args.apply:
        print(
            "\nDry-run only. Re-run with --apply to mark originals "
            "deduplicated and insert split needs_review facts."
        )
        return

    if not proposals:
        print("Nothing to apply.")
        return

    confirm = input(
        f"Apply repair for {len(proposals)} combined memories "
        f"for user_id={args.user_id}? Type YES to continue: "
    )
    if confirm.strip() != "YES":
        print("Aborted. No changes applied.")
        return

    result = apply_repairs(args.user_id, proposals)
    print(json.dumps({"apply_result": result}, indent=2))


if __name__ == "__main__":
    main()
