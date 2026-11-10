"""
Stage 19 MongoDB Atlas integration test.

Uses a uniquely named temporary collection only.
Does not modify the production memories_history collection.
Makes no Groq or Gemini API calls.

Run:
    python -m tests.test_stage19_mongodb_integration
"""

import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from bson import ObjectId

from app.services import memory_service
from app.services.memory_lifecycle_service import identify_stale_memory


def main() -> None:
    client = memory_service.client
    database = memory_service.database

    hello = client.admin.command("hello")
    transaction_supported = bool(
        hello.get("setName") or hello.get("msg") == "isdbgrid"
    )
    if not transaction_supported:
        raise RuntimeError(
            "MongoDB does not report a replica-set or mongos topology."
        )

    # Do not write production memory_summaries during this suite.
    previous_auto = os.environ.get("MEMORY_SUMMARY_AUTO_REFRESH")
    os.environ["MEMORY_SUMMARY_AUTO_REFRESH"] = "0"

    collection_name = f"stage19_integration_test_{uuid4().hex}"
    user_id = f"stage19-test-{uuid4().hex}"
    other_user = f"stage19-other-{uuid4().hex}"
    original_collection = memory_service.memory_collection
    collection_created = False

    print("=== Stage 19 MongoDB Integration Test ===")
    print(f"Temporary collection: {collection_name}")
    print("Production collection: untouched")

    try:
        test_collection = database.create_collection(collection_name)
        collection_created = True
        memory_service.memory_collection = test_collection

        # --------------------------------------------------------
        # 1. Memory creation with operation_id
        # --------------------------------------------------------
        op_id = f"stage19-save-{uuid4().hex}"
        saved = memory_service.save_memory(
            user_id=user_id,
            memory="My project is Alpha",
            embedding=None,
            importance=3,
            confidence=0.95,
            status="active",
            fact_key="project",
            normalized_value="alpha",
            version=1,
            operation_id=op_id,
        )
        assert saved.get("success"), f"Save failed: {saved}"
        memory_id = saved["memory_id"]
        print("PASS: Memory created with operation_id.")

        # --------------------------------------------------------
        # 2. Idempotent save replay
        # --------------------------------------------------------
        replay = memory_service.save_memory(
            user_id=user_id,
            memory="My project is Alpha",
            embedding=None,
            importance=3,
            confidence=0.95,
            status="active",
            operation_id=op_id,
        )
        assert replay.get("success") and replay.get("idempotent") is True, (
            f"Idempotent replay failed: {replay}"
        )
        assert replay.get("memory_id") == memory_id
        print("PASS: Idempotent save replay returned same memory_id.")

        # --------------------------------------------------------
        # 3. Access tracking
        # --------------------------------------------------------
        memory_service._record_memory_access(
            user_id, [{"memory_id": memory_id}]
        )
        doc = test_collection.find_one({"_id": ObjectId(memory_id), "user_id": user_id})
        assert doc.get("access_count") == 1
        assert isinstance(doc.get("last_accessed_at"), datetime)
        print("PASS: Access tracking persisted.")

        # --------------------------------------------------------
        # 4. Reinforcement + JSON-safe ISO datetime in tool result
        # --------------------------------------------------------
        reinforce_op_id = f"reinforce-{uuid4().hex}"
        reinforced = memory_service.reinforce_memory(
            user_id, memory_id, operation_id=reinforce_op_id
        )
        assert reinforced.get("success"), f"Reinforce failed: {reinforced}"
        assert isinstance(reinforced.get("last_reinforced_at"), str), (
            "last_reinforced_at must be ISO string for JSON serialization"
        )
        doc = test_collection.find_one({"_id": ObjectId(memory_id), "user_id": user_id})
        assert doc.get("reinforcement_count") == 1
        assert isinstance(doc.get("last_reinforced_at"), datetime)
        print("PASS: Reinforcement persisted with JSON-safe ISO timestamp.")

        # 4b. Idempotent reinforce retry (same operation_id)
        reinforce_replay = memory_service.reinforce_memory(
            user_id, memory_id, operation_id=reinforce_op_id
        )
        assert reinforce_replay.get("success")
        assert reinforce_replay.get("idempotent") is True
        doc = test_collection.find_one({"_id": ObjectId(memory_id), "user_id": user_id})
        assert doc.get("reinforcement_count") == 1
        print("PASS: Reinforce retry is idempotent.")

        # 4c. Distinct reaffirmations increment count
        for i in range(2):
            op = f"reinforce-distinct-{i}-{uuid4().hex}"
            result = memory_service.reinforce_memory(
                user_id, memory_id, operation_id=op
            )
            assert result.get("success") and not result.get("idempotent")
        doc = test_collection.find_one({"_id": ObjectId(memory_id), "user_id": user_id})
        assert doc.get("reinforcement_count") == 3
        print("PASS: Distinct reinforce operations increment count.")

        # --------------------------------------------------------
        # 5. Cross-user reinforcement blocked
        # --------------------------------------------------------
        cross = memory_service.reinforce_memory(other_user, memory_id)
        assert not cross.get("success"), f"Cross-user reinforce should fail: {cross}"
        print("PASS: Cross-user reinforcement blocked.")

        # --------------------------------------------------------
        # 6. Cross-user deletion blocked
        # --------------------------------------------------------
        cross_del = memory_service.delete_memory(
            user_id=other_user, memory_id=memory_id, soft=True
        )
        assert not cross_del.get("success"), f"Cross-user delete should fail: {cross_del}"
        print("PASS: Cross-user deletion blocked.")

        # --------------------------------------------------------
        # 7. Duplicate handling — create a textual duplicate
        # --------------------------------------------------------
        dup_op_id = f"stage19-dup-{uuid4().hex}"
        dup_saved = memory_service.save_memory(
            user_id=user_id,
            memory="My project is Alpha",  # same text, different op_id
            embedding=None,
            importance=3,
            status="active",
            fact_key="project",
            normalized_value="alpha",
            operation_id=dup_op_id,
        )
        assert dup_saved.get("success"), f"Dup save failed: {dup_saved}"
        dup_id = dup_saved["memory_id"]
        print("PASS: Duplicate memory inserted for deduplication test.")

        # --------------------------------------------------------
        # 8. Safe deduplication (dry run)
        # --------------------------------------------------------
        dedup_result = memory_service.safe_deduplicate_memories(
            user_id, dry_run=True
        )
        assert dedup_result.get("success"), f"Dedup dry run failed: {dedup_result}"
        assert dedup_result.get("duplicate_groups") >= 1
        assert dedup_result.get("resolved") == 0
        print(f"PASS: Deduplication dry run found {dedup_result['duplicate_groups']} groups.")

        # --------------------------------------------------------
        # 9. Safe deduplication (resolve)
        # --------------------------------------------------------
        dedup_resolve = memory_service.safe_deduplicate_memories(
            user_id, dry_run=False
        )
        assert dedup_resolve.get("success"), f"Dedup resolve failed: {dedup_resolve}"
        assert dedup_resolve.get("resolved") >= 1

        # Verify the duplicate is now marked as deduplicated.
        dup_doc = test_collection.find_one({"_id": ObjectId(dup_id), "user_id": user_id})
        # The canonical one should remain active; the duplicate should be deduplicated.
        active_count = test_collection.count_documents({
            "user_id": user_id,
            "memory": "My project is Alpha",
            "status": "active",
        })
        assert active_count == 1, f"Expected 1 active, found {active_count}"
        print("PASS: Deduplication resolved — one active, extras marked.")

        # --------------------------------------------------------
        # 10. Soft-forget
        # --------------------------------------------------------
        forgotten = memory_service.delete_memory(
            user_id=user_id, memory_id=memory_id, soft=True
        )
        assert forgotten.get("success"), f"Forget failed: {forgotten}"
        doc = test_collection.find_one({"_id": ObjectId(memory_id), "user_id": user_id})
        assert doc.get("status") == "forgotten"
        print("PASS: Soft-forget marks forgotten without hard delete.")

        # --------------------------------------------------------
        # 11. Idempotent soft-forget (already forgotten)
        # --------------------------------------------------------
        again = memory_service.delete_memory(
            user_id=user_id, memory_id=memory_id, soft=True
        )
        assert again.get("success"), f"Idempotent forget failed: {again}"
        assert again.get("idempotent") is True
        print("PASS: Idempotent soft-forget on already-forgotten memory.")

        # --------------------------------------------------------
        # 12. Version update (supersede) with transaction
        # --------------------------------------------------------
        v2_op_id = f"stage19-v2-{uuid4().hex}"
        v2_saved = memory_service.save_memory(
            user_id=user_id,
            memory="My project is Gamma",
            embedding=None,
            importance=4,
            confidence=0.95,
            status="active",
            fact_key="project",
            normalized_value="gamma",
            version=1,
            operation_id=v2_op_id,
        )
        assert v2_saved.get("success"), f"V2 base save failed: {v2_saved}"
        v2_id = v2_saved["memory_id"]

        updated = memory_service.supersede_memory(
            user_id=user_id,
            memory_id=v2_id,
            memory="My project is Delta",
            embedding=None,
            fact_key="project",
            normalized_value="delta",
        )
        assert updated.get("success"), f"Supersede failed: {updated}"
        assert updated.get("version") == 2

        old_doc = test_collection.find_one({"_id": ObjectId(v2_id), "user_id": user_id})
        assert old_doc.get("status") == "superseded"
        new_doc = test_collection.find_one({
            "_id": ObjectId(updated["memory_id"]), "user_id": user_id
        })
        assert new_doc.get("status") == "active"
        assert new_doc.get("version") == 2
        print("PASS: Version update via supersede creates v2 and marks v1 superseded.")

        # --------------------------------------------------------
        # 13. Stale identification (non-destructive)
        # --------------------------------------------------------
        stale_probe = {
            "memory_id": "probe",
            "importance": 1,
            "created_at": datetime.now(timezone.utc) - timedelta(days=400),
        }
        stale = identify_stale_memory(stale_probe)
        assert stale["is_stale"] is True
        print("PASS: Stale identification flags old facts.")

        # --------------------------------------------------------
        # 14. Concurrent update detection
        # --------------------------------------------------------
        concurrent_save = memory_service.save_memory(
            user_id=user_id,
            memory="My company is TestCorp",
            embedding=None,
            importance=3,
            status="active",
            fact_key="company",
            version=1,
            operation_id=f"concurrent-{uuid4().hex}",
        )
        assert concurrent_save.get("success")
        concurrent_id = concurrent_save["memory_id"]

        # Manually mark it superseded to simulate a concurrent update.
        test_collection.update_one(
            {"_id": ObjectId(concurrent_id)},
            {"$set": {"status": "superseded"}},
        )

        # Now try to supersede it — should fail because it's no longer active.
        fail_result = memory_service.supersede_memory(
            user_id=user_id,
            memory_id=concurrent_id,
            memory="My company is NewCorp",
            embedding=None,
        )
        assert not fail_result.get("success"), (
            f"Should fail on already-superseded memory: {fail_result}"
        )
        assert fail_result.get("error_code") in {"not_active", None} or (
            "active" in (fail_result.get("error") or "").lower()
        )
        print("PASS: Concurrent update detection prevents double supersede.")

        # --------------------------------------------------------
        # 15. Activate confirmed memory when only needs_review exists
        # --------------------------------------------------------
        pending = memory_service.save_memory(
            user_id=user_id,
            memory="My project is PendingOne",
            embedding=None,
            importance=3,
            status="needs_review",
            fact_key="project",
            normalized_value="pendingone",
            conflict_type="possible_contradiction",
            operation_id=f"pending-{uuid4().hex}",
        )
        assert pending.get("success")
        activated = memory_service.activate_confirmed_memory(
            user_id=user_id,
            memory="My project is Freshcart",
            embedding=None,
            fact_key="project",
            normalized_value="freshcart",
            operation_id=f"activate-{uuid4().hex}",
        )
        assert activated.get("success"), f"Activate failed: {activated}"
        pending_doc = test_collection.find_one(
            {"_id": ObjectId(pending["memory_id"]), "user_id": user_id}
        )
        active_doc = test_collection.find_one(
            {"_id": ObjectId(activated["memory_id"]), "user_id": user_id}
        )
        assert pending_doc.get("status") == "resolved"
        assert active_doc.get("status") == "active"
        assert active_doc.get("normalized_value") == "freshcart"
        assert pending_doc.get("superseded_by_memory_id") == activated["memory_id"]
        print("PASS: Pending-only confirmation activates Freshcart and resolves pending.")

        # --------------------------------------------------------
        # 16. Cross-user supersede rejected
        # --------------------------------------------------------
        cross_super = memory_service.supersede_memory(
            user_id=other_user,
            memory_id=activated["memory_id"],
            memory="My project is Stolen",
            embedding=None,
        )
        assert not cross_super.get("success")
        print("PASS: Cross-user supersede rejected.")

        # --------------------------------------------------------
        # 17. Supersede relationship fields
        # --------------------------------------------------------
        rel = memory_service.supersede_memory(
            user_id=user_id,
            memory_id=activated["memory_id"],
            memory="My project is RelTest",
            embedding=None,
            fact_key="project",
            normalized_value="reltest",
        )
        assert rel.get("success"), f"Relationship supersede failed: {rel}"
        old_rel = test_collection.find_one(
            {"_id": ObjectId(activated["memory_id"]), "user_id": user_id}
        )
        new_rel = test_collection.find_one(
            {"_id": ObjectId(rel["memory_id"]), "user_id": user_id}
        )
        assert old_rel.get("status") == "superseded"
        assert old_rel.get("superseded_by_memory_id") == rel["memory_id"]
        assert new_rel.get("supersedes_memory_id") == activated["memory_id"]
        assert new_rel.get("version") == int(old_rel.get("version", 1)) + 1
        print("PASS: Supersede relationship fields and version are consistent.")

        # --------------------------------------------------------
        # 18. Primary project classification + non-conflicting name/desc
        # --------------------------------------------------------
        from app.services.memory_conflict_service import classify_project_fact
        from app.services.memory_service import detect_conflicting_active_facts

        p_primary = memory_service.save_memory(
            user_id=user_id,
            memory="My primary project is Nova",
            embedding=None,
            status="active",
            fact_key="primary_project",
            normalized_value="nova",
            operation_id=f"pp-{uuid4().hex}",
        )
        p_named = memory_service.save_memory(
            user_id=user_id,
            memory="Freshcart is my grocery marketplace project",
            embedding=None,
            status="active",
            fact_key="project_name",
            normalized_value="freshcart (grocery marketplace)",
            operation_id=f"pn-{uuid4().hex}",
        )
        assert p_primary.get("success") and p_named.get("success")
        assert classify_project_fact(
            "I am building an Agentic AI bug resolver."
        )[0] == "project_description"
        mixed_conflicts = detect_conflicting_active_facts(
            [
                {
                    "memory": "My primary project is Nova",
                    "status": "active",
                    "fact_key": "primary_project",
                    "normalized_value": "nova",
                },
                {
                    "memory": "Freshcart is my grocery marketplace project",
                    "status": "active",
                    "fact_key": "project_name",
                    "normalized_value": "freshcart (grocery marketplace)",
                },
            ]
        )
        assert mixed_conflicts == [], mixed_conflicts
        print("PASS: Named project does not conflict with primary_project.")

        # --------------------------------------------------------
        # 19. Legacy combined repair dry-run on temp collection
        # --------------------------------------------------------
        from scripts.repair_combined_memories import propose_repairs

        combined = memory_service.save_memory(
            user_id=user_id,
            memory=(
                "my preferred code editor is VS Code."
                "I prefer IntelliJ IDEA as my code editor"
            ),
            embedding=None,
            status="active",
            operation_id=f"combined-{uuid4().hex}",
        )
        assert combined.get("success")
        proposals = propose_repairs(user_id, limit=50)
        assert any(
            p["memory_id"] == combined["memory_id"] for p in proposals
        ), proposals
        print("PASS: Repair dry-run proposes split for combined editor memory.")

        # --------------------------------------------------------
        # 20. Preferred database contradiction (not vector duplicate)
        # --------------------------------------------------------
        from app.services.memory_conflict_service import (
            classify_duplicate,
            detect_memory_conflict,
            extract_exclusive_fact,
        )

        mongo_fact = extract_exclusive_fact(
            "my preferred database is MongoDB"
        )
        assert mongo_fact == ("preferred_database", "mongodb"), mongo_fact
        pg_fact = extract_exclusive_fact(
            "my preferred database is PostgreSQL"
        )
        assert pg_fact == ("preferred_database", "postgresql"), pg_fact

        mongo_saved = memory_service.save_memory(
            user_id=user_id,
            memory="my preferred database is MongoDB",
            embedding=None,
            status="active",
            importance=4,
            fact_key=mongo_fact[0],
            normalized_value=mongo_fact[1],
            operation_id=f"db-mongo-{uuid4().hex}",
        )
        assert mongo_saved.get("success"), mongo_saved

        clf = classify_duplicate(
            "my preferred database is PostgreSQL",
            "my preferred database is MongoDB",
            vector_score=0.97,
        )
        assert clf["classification"] == "possible_contradiction", clf

        conflict = detect_memory_conflict(
            "my preferred database is PostgreSQL",
            [
                {
                    "memory_id": mongo_saved["memory_id"],
                    "memory": "my preferred database is MongoDB",
                    "status": "active",
                    "fact_key": "preferred_database",
                    "normalized_value": "mongodb",
                    "score": 0.97,
                }
            ],
        )
        assert conflict.get("has_conflict"), conflict
        pg_pending = memory_service.save_memory(
            user_id=user_id,
            memory="my preferred database is PostgreSQL",
            embedding=None,
            status="needs_review",
            importance=4,
            fact_key=pg_fact[0],
            normalized_value=pg_fact[1],
            conflict_type="possible_contradiction",
            conflict_group_id=uuid4().hex,
            conflict_with_memory_id=mongo_saved["memory_id"],
            operation_id=f"db-pg-{uuid4().hex}",
        )
        assert pg_pending.get("success"), pg_pending
        mongo_doc = test_collection.find_one(
            {"_id": ObjectId(mongo_saved["memory_id"]), "user_id": user_id}
        )
        pg_doc = test_collection.find_one(
            {"_id": ObjectId(pg_pending["memory_id"]), "user_id": user_id}
        )
        assert mongo_doc.get("status") == "active"
        assert mongo_doc.get("normalized_value") == "mongodb"
        assert pg_doc.get("status") == "needs_review"
        assert pg_doc.get("normalized_value") == "postgresql"
        assert pg_doc.get("conflict_with_memory_id") == mongo_saved["memory_id"]
        print(
            "PASS: Preferred database contradiction keeps MongoDB active "
            "and PostgreSQL needs_review."
        )

        # --------------------------------------------------------
        # 21. Fresh-session retrieval surfaces pending + conflict group
        # --------------------------------------------------------
        from app.services.memory_service import (
            detect_active_pending_conflicts,
            semantic_search_memories,
        )
        from app.chatbot import build_memory_context

        # Structured path must work even when vector search is empty/noisy.
        retrieved = semantic_search_memories(
            user_id=user_id,
            query="Which database do I currently prefer? Identify any conflicts.",
            limit=5,
            min_score=0.99,  # force vector misses; fact-key merge must recover
            include_needs_review=True,
        )
        assert retrieved.get("success"), retrieved
        retrieved_texts = " ".join(
            m.get("memory", "") for m in retrieved.get("memories", [])
        ).lower()
        assert "mongodb" in retrieved_texts, retrieved
        assert "postgresql" in retrieved_texts, retrieved
        assert any(
            m.get("status") == "needs_review" for m in retrieved["memories"]
        ), retrieved
        kinds = {
            c.get("conflict_kind") for c in retrieved.get("conflicts", [])
        }
        assert "active_vs_pending" in kinds, retrieved.get("conflicts")
        context = build_memory_context(
            retrieved["memories"],
            conflicts=retrieved["conflicts"],
        )
        assert "PENDING_REVIEW_FACTS" in context
        assert "PostgreSQL" in context or "postgresql" in context.lower()
        assert "MEMORY_CONFLICT_NOTICE" in context
        # Direct detector sanity.
        direct = detect_active_pending_conflicts(retrieved["memories"])
        assert direct, direct
        print(
            "PASS: Fresh-session retrieval includes pending PostgreSQL "
            "and active_vs_pending conflict group."
        )

        # --------------------------------------------------------
        # 22. Read-only metadata inspection (no writes)
        # --------------------------------------------------------
        from app.services.memory_service import (
            format_memory_metadata_report,
            inspect_memory_metadata,
        )

        before_count = test_collection.count_documents({"user_id": user_id})
        inspection = inspect_memory_metadata(
            user_id=user_id,
            fact_key="preferred_database",
        )
        after_count = test_collection.count_documents({"user_id": user_id})
        assert inspection.get("success"), inspection
        assert inspection.get("read_only") is True
        assert before_count == after_count, "Inspection must not modify records"
        statuses = {r.get("status") for r in inspection["records"]}
        assert "active" in statuses and "needs_review" in statuses, inspection
        # Actual stored values: active MongoDB was saved WITH fact_key in this
        # temp-collection scenario; pending has preferred_database.
        assert any(
            r.get("normalized_value") == "mongodb" and r.get("status") == "active"
            for r in inspection["records"]
        ), inspection
        assert any(
            r.get("normalized_value") == "postgresql"
            and r.get("status") == "needs_review"
            for r in inspection["records"]
        ), inspection
        comparison = inspection["conflict_group_comparison"]
        assert "verdict" in comparison
        assert "same_conflict_group" in comparison
        # Fixture mirrors production asymmetry: pending has conflict_group_id,
        # active MongoDB row does not. Inspection must report that honestly
        # and must NOT fabricate a shared group.
        assert comparison["same_conflict_group"] is False, comparison
        assert comparison["verdict"] == "inconsistent_missing", comparison
        report = format_memory_metadata_report(inspection)
        assert "preferred_database" in report
        assert "No records were modified" in report
        assert "inconsistent_missing" in report
        # Ensure no document fields were mutated (spot-check pending).
        pg_after = test_collection.find_one(
            {"_id": ObjectId(pg_pending["memory_id"]), "user_id": user_id}
        )
        assert pg_after.get("status") == "needs_review"
        assert pg_after.get("normalized_value") == "postgresql"
        print(
            "PASS: Read-only metadata inspection reports stored fields "
            "and conflict_group comparison without writes."
        )

        # --------------------------------------------------------
        # 23. Optional metadata backfill (dry-run then apply)
        # --------------------------------------------------------
        from app.services.memory_service import backfill_memory_metadata

        # Recreate asymmetric pair: active missing fact_key/group, pending complete.
        legacy_active = memory_service.save_memory(
            user_id=user_id,
            memory="my preferred database is Redis",
            embedding=None,
            status="active",
            importance=4,
            # Intentionally omit fact_key / normalized_value / conflict_group_id
            operation_id=f"db-redis-{uuid4().hex}",
        )
        assert legacy_active.get("success"), legacy_active
        group_id = uuid4().hex
        legacy_pending = memory_service.save_memory(
            user_id=user_id,
            memory="my preferred database is Cassandra",
            embedding=None,
            status="needs_review",
            importance=4,
            fact_key="preferred_database",
            normalized_value="cassandra",
            conflict_type="possible_contradiction",
            conflict_group_id=group_id,
            conflict_with_memory_id=legacy_active["memory_id"],
            operation_id=f"db-cass-{uuid4().hex}",
        )
        assert legacy_pending.get("success"), legacy_pending

        dry = backfill_memory_metadata(
            user_id,
            fact_key="preferred_database",
            dry_run=True,
        )
        assert dry.get("success") and dry.get("dry_run") is True, dry
        assert dry.get("proposal_count", 0) >= 1, dry
        # Dry-run must not change the active row yet.
        active_before = test_collection.find_one(
            {"_id": ObjectId(legacy_active["memory_id"]), "user_id": user_id}
        )
        assert active_before.get("fact_key") in (None, "")
        assert active_before.get("conflict_group_id") in (None, "")

        applied = backfill_memory_metadata(
            user_id,
            fact_key="preferred_database",
            dry_run=False,
        )
        assert applied.get("success") and applied.get("dry_run") is False, applied
        assert applied.get("applied_count", 0) >= 1, applied
        active_after = test_collection.find_one(
            {"_id": ObjectId(legacy_active["memory_id"]), "user_id": user_id}
        )
        assert active_after.get("fact_key") == "preferred_database", active_after
        assert active_after.get("normalized_value") == "redis", active_after
        assert active_after.get("conflict_group_id") == group_id, active_after
        # Pending unchanged aside from any prior fields.
        pending_after = test_collection.find_one(
            {"_id": ObjectId(legacy_pending["memory_id"]), "user_id": user_id}
        )
        assert pending_after.get("status") == "needs_review"
        assert pending_after.get("normalized_value") == "cassandra"
        assert pending_after.get("conflict_group_id") == group_id

        inspect_after = inspect_memory_metadata(
            user_id=user_id,
            fact_key="preferred_database",
        )
        # Among redis/cassandra pair, groups should now match when filtered
        # to those two memory ids.
        pair = [
            r
            for r in inspect_after.get("records", [])
            if r.get("memory_id")
            in {legacy_active["memory_id"], legacy_pending["memory_id"]}
        ]
        from app.services.memory_service import compare_conflict_group_ids

        pair_cmp = compare_conflict_group_ids(pair)
        assert pair_cmp["same_conflict_group"] is True, pair_cmp
        assert pair_cmp["verdict"] in {"match", "consistent"}, pair_cmp
        assert active_after.get("conflict_with_memory_id") == legacy_pending[
            "memory_id"
        ], active_after
        assert active_after.get("conflict_type") == "possible_contradiction"
        print(
            "PASS: Optional metadata backfill dry-run then apply aligns "
            "fact_key/normalized_value/conflict_group_id."
        )

        # --------------------------------------------------------
        # 24. Bilateral conflict pair persistence + invariant verify
        # --------------------------------------------------------
        from app.services.memory_service import persist_memory_conflict_pair

        left = memory_service.save_memory(
            user_id=user_id,
            memory="my preferred database is SQLite",
            embedding=None,
            status="active",
            importance=4,
            operation_id=f"db-sqlite-{uuid4().hex}",
        )
        assert left.get("success"), left
        right = memory_service.save_memory(
            user_id=user_id,
            memory="my preferred database is MariaDB",
            embedding=None,
            status="needs_review",
            importance=4,
            fact_key="preferred_database",
            normalized_value="mariadb",
            conflict_type="possible_contradiction",
            conflict_group_id=uuid4().hex,
            conflict_with_memory_id=left["memory_id"],
            operation_id=f"db-maria-{uuid4().hex}",
        )
        assert right.get("success"), right

        dry_pair = persist_memory_conflict_pair(
            user_id,
            left["memory_id"],
            right["memory_id"],
            fact_key="preferred_database",
            conflict_group_id=right.get("conflict_group_id")
            or test_collection.find_one(
                {"_id": ObjectId(right["memory_id"])}
            ).get("conflict_group_id"),
            dry_run=True,
        )
        assert dry_pair.get("success") and dry_pair.get("dry_run"), dry_pair

        applied_pair = persist_memory_conflict_pair(
            user_id,
            left["memory_id"],
            right["memory_id"],
            fact_key="preferred_database",
            dry_run=False,
        )
        assert applied_pair.get("success") and applied_pair.get("verified"), (
            applied_pair
        )
        left_doc = test_collection.find_one(
            {"_id": ObjectId(left["memory_id"]), "user_id": user_id}
        )
        right_doc = test_collection.find_one(
            {"_id": ObjectId(right["memory_id"]), "user_id": user_id}
        )
        assert left_doc.get("status") == "active"
        assert right_doc.get("status") == "needs_review"
        assert left_doc.get("fact_key") == "preferred_database"
        assert left_doc.get("normalized_value") == "sqlite"
        assert left_doc.get("conflict_group_id") == right_doc.get(
            "conflict_group_id"
        )
        assert left_doc.get("conflict_with_memory_id") == right["memory_id"]
        assert right_doc.get("conflict_with_memory_id") == left["memory_id"]
        assert left_doc.get("conflict_type") == "possible_contradiction"

        # Reverse insertion order: pending first then active partner.
        first = memory_service.save_memory(
            user_id=user_id,
            memory="my preferred database is CockroachDB",
            embedding=None,
            status="needs_review",
            fact_key="preferred_database",
            normalized_value="cockroachdb",
            operation_id=f"db-crdb-{uuid4().hex}",
        )
        second = memory_service.save_memory(
            user_id=user_id,
            memory="my preferred database is Spanner",
            embedding=None,
            status="active",
            operation_id=f"db-spanner-{uuid4().hex}",
        )
        rev = persist_memory_conflict_pair(
            user_id,
            second["memory_id"],
            first["memory_id"],
            fact_key="preferred_database",
            dry_run=False,
        )
        assert rev.get("success") and rev.get("verified"), rev
        # Retry idempotency — same group, no failure.
        retry = persist_memory_conflict_pair(
            user_id,
            second["memory_id"],
            first["memory_id"],
            fact_key="preferred_database",
            conflict_group_id=rev["conflict_group_id"],
            dry_run=False,
        )
        assert retry.get("success"), retry
        assert retry["conflict_group_id"] == rev["conflict_group_id"]

        # Cross-user rejection.
        cross = persist_memory_conflict_pair(
            other_user,
            left["memory_id"],
            right["memory_id"],
            fact_key="preferred_database",
            dry_run=False,
        )
        assert not cross.get("success"), cross
        print(
            "PASS: Bilateral conflict pair persistence verifies reciprocal "
            "metadata for both insertion orders and rejects cross-user pairs."
        )

        print("\n=== All Stage 19 MongoDB integration checks passed. ===")

    finally:
        memory_service.memory_collection = original_collection
        if previous_auto is None:
            os.environ.pop("MEMORY_SUMMARY_AUTO_REFRESH", None)
        else:
            os.environ["MEMORY_SUMMARY_AUTO_REFRESH"] = previous_auto
        if collection_created:
            database.drop_collection(collection_name)
            print(f"Cleanup: dropped temporary collection {collection_name}")


if __name__ == "__main__":
    main()
