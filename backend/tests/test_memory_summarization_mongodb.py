"""
Memory summarization MongoDB integration test.

Uses temporary memories + summaries collections only.
No Groq/Gemini calls. Production collections are not modified.

Run:
    python -m tests.test_memory_summarization_mongodb
"""

from __future__ import annotations

import os
from uuid import uuid4

from bson import ObjectId

from app.chatbot import build_memory_context
from app.services import memory_service
from app.services import memory_summary_service as summary_svc


def main() -> None:
    database = memory_service.database
    mem_name = f"summary_mem_test_{uuid4().hex}"
    sum_name = f"summary_sum_test_{uuid4().hex}"
    user_a = f"sum-user-a-{uuid4().hex}"
    user_b = f"sum-user-b-{uuid4().hex}"

    original_mem = memory_service.memory_collection
    original_sum = memory_service.summary_collection
    previous_auto = os.environ.get("MEMORY_SUMMARY_AUTO_REFRESH")
    os.environ["MEMORY_SUMMARY_AUTO_REFRESH"] = "0"

    print("=== Memory Summarization MongoDB Integration ===")
    print(f"Temp memories: {mem_name}")
    print(f"Temp summaries: {sum_name}")

    try:
        mem_coll = database.create_collection(mem_name)
        sum_coll = database.create_collection(sum_name)
        memory_service.memory_collection = mem_coll
        memory_service.summary_collection = sum_coll
        summary_svc.configure_summary_collections(sum_coll, mem_coll)
        summary_svc.ensure_summary_indexes()

        # 1. Basic confirmed preference summary
        memory_service.save_memory(
            user_id=user_a,
            memory="my preferred programming language is Python",
            embedding=None,
            status="active",
            fact_key="preferred_language",
            normalized_value="python",
            operation_id=f"lang-{uuid4().hex}",
        )
        gen = summary_svc.generate_summary_for_type(
            user_a,
            "preferences",
            scope_key="preferred_language",
            force=True,
        )
        assert gen.get("success"), gen
        text = gen["summary"]["summary_text"].lower()
        assert "python" in text
        assert gen["summary"]["source_memory_count"] == 1
        assert gen["summary"]["scope_key"] == "preferred_language"
        print("PASS: Basic summarization with source traceability.")

        # 2. Conflict-aware preferences (MongoDB active / PostgreSQL pending)
        mongo = memory_service.save_memory(
            user_id=user_a,
            memory="my preferred database is MongoDB",
            embedding=None,
            status="active",
            fact_key="preferred_database",
            normalized_value="mongodb",
            operation_id=f"db-m-{uuid4().hex}",
        )
        pg = memory_service.save_memory(
            user_id=user_a,
            memory="my preferred database is PostgreSQL",
            embedding=None,
            status="needs_review",
            fact_key="preferred_database",
            normalized_value="postgresql",
            conflict_type="possible_contradiction",
            conflict_group_id=uuid4().hex,
            conflict_with_memory_id=mongo["memory_id"],
            operation_id=f"db-p-{uuid4().hex}",
        )
        assert mongo.get("success") and pg.get("success")

        conflict_sum = summary_svc.generate_summary_for_type(
            user_a,
            "preferences",
            scope_key="preferred_database",
            force=True,
        )
        assert conflict_sum.get("success"), conflict_sum
        ctext = conflict_sum["summary"]["summary_text"].lower()
        assert "unresolved" in ctext
        assert "mongodb" in ctext and "postgresql" in ctext
        assert "clarification" in ctext
        assert "prefers mongodb" not in ctext
        refs = conflict_sum["summary"]["conflict_references"]
        assert refs and refs[0]["conflict_kind"] == "active_vs_pending"
        ids = set(conflict_sum["summary"]["source_memory_ids"])
        assert mongo["memory_id"] in ids and pg["memory_id"] in ids
        assert conflict_sum["summary"]["source_memory_count"] == len(ids)
        print("PASS: Conflict-aware summarization keeps conflict unresolved.")

        # Force a new version of the same scope → previous becomes outdated.
        forced = summary_svc.generate_summary_for_type(
            user_a,
            "preferences",
            scope_key="preferred_database",
            force=True,
        )
        assert forced.get("success"), forced
        outdated = list(
            sum_coll.find(
                {
                    "user_id": user_a,
                    "summary_type": "preferences",
                    "scope_key": "preferred_database",
                    "status": "outdated",
                }
            )
        )
        assert outdated, "expected previous summary version"
        print("PASS: Previous summary version preserved as outdated.")

        # 3. Idempotent regenerate (same hash → unchanged)
        again = summary_svc.generate_summary_for_type(
            user_a,
            "preferences",
            scope_key="preferred_database",
            force=False,
        )
        assert again.get("success") and again.get("unchanged"), again
        print("PASS: Idempotent regenerate reuses source_hash.")

        # 4. Scoped regenerate after new editor memory
        memory_service.save_memory(
            user_id=user_a,
            memory="my preferred code editor is VS Code",
            embedding=None,
            status="active",
            fact_key="editor",
            normalized_value="vs code",
            operation_id=f"ed-{uuid4().hex}",
        )
        ed = summary_svc.generate_summary_for_type(
            user_a, "preferences", scope_key="editor", force=True
        )
        assert ed.get("success"), ed
        assert "vs code" in ed["summary"]["summary_text"].lower()
        # Database summary must remain isolated.
        db_active = sum_coll.find_one(
            {
                "user_id": user_a,
                "summary_type": "preferences",
                "scope_key": "preferred_database",
                "status": "active",
            }
        )
        assert db_active
        assert "vs code" not in db_active["summary_text"].lower()
        print("PASS: Invalidation regenerates after new memory.")

        # 4b. Listifys project isolation
        listify = memory_service.save_memory(
            user_id=user_a,
            memory="Listifys is my classified marketplace project",
            embedding=None,
            status="active",
            fact_key="project_name",
            normalized_value="listifys (classified marketplace)",
            operation_id=f"listify-{uuid4().hex}",
        )
        memory_service.save_memory(
            user_id=user_a,
            memory="My project is Alpha",
            embedding=None,
            status="active",
            fact_key="primary_project",
            normalized_value="alpha",
            operation_id=f"alpha-{uuid4().hex}",
        )
        memory_service.save_memory(
            user_id=user_a,
            memory="Freshcart is my grocery marketplace project",
            embedding=None,
            status="active",
            fact_key="project_name",
            normalized_value="freshcart (grocery marketplace)",
            operation_id=f"fresh-{uuid4().hex}",
        )
        memory_service.save_memory(
            user_id=user_a,
            memory=(
                "Summary: The user has a classified marketplace project "
                "named Listifys. Source memory ID(s): abc"
            ),
            embedding=None,
            status="active",
            operation_id=f"artifact-{uuid4().hex}",
        )
        listify_sum = summary_svc.generate_summary_for_type(
            user_a, "project", scope_key="listifys", force=True
        )
        assert listify_sum.get("success"), listify_sum
        ltext = listify_sum["summary"]["summary_text"].lower()
        assert "listifys" in ltext
        for foreign in ("alpha", "freshcart", "marathon", "nova", "orion", "beta"):
            assert foreign not in ltext, ltext
        assert listify_sum["summary"]["source_memory_ids"] == [
            listify["memory_id"]
        ]
        assert listify_sum["summary"]["source_memory_count"] == 1
        print("PASS: Listifys project summary excludes unrelated projects.")

        # 5. Cross-user isolation
        memory_service.save_memory(
            user_id=user_b,
            memory="my preferred database is Redis",
            embedding=None,
            status="active",
            fact_key="preferred_database",
            normalized_value="redis",
            operation_id=f"b-{uuid4().hex}",
        )
        summary_svc.generate_summary_for_type(
            user_b, "preferences", scope_key="preferred_database", force=True
        )
        a_search = summary_svc.search_memory_summaries(
            user_a, "database", limit=5
        )
        b_search = summary_svc.search_memory_summaries(
            user_b, "database", limit=5
        )
        assert a_search.get("success") and b_search.get("success")
        assert all(s["user_id"] == user_a for s in a_search["summaries"])
        assert all(s["user_id"] == user_b for s in b_search["summaries"])
        assert all(
            "redis" not in (s.get("summary_text") or "").lower()
            for s in a_search["summaries"]
        )
        print("PASS: Multi-user isolation for summaries.")

        # 6. Outdated summaries excluded from search / context
        sum_coll.update_many(
            {"user_id": user_a, "status": "active"},
            {"$set": {"status": "outdated"}},
        )
        empty = summary_svc.search_memory_summaries(
            user_a, "database", include_outdated=False
        )
        assert empty.get("success")
        assert empty.get("summaries") == []
        context = build_memory_context(
            memories=[],
            summaries=[
                {
                    "summary_text": "should not appear",
                    "status": "outdated",
                    "summary_type": "preferences",
                    "source_memory_count": 1,
                    "conflict_references": [],
                }
            ],
        )
        assert context == ""
        print("PASS: Outdated summaries excluded from retrieval/context.")

        # 7. Expand sources + ownership check
        summary_svc.generate_summary_for_type(
            user_a, "preferences", scope_key="preferred_database", force=True
        )
        active = sum_coll.find_one(
            {
                "user_id": user_a,
                "summary_type": "preferences",
                "scope_key": "preferred_database",
                "status": "active",
            }
        )
        expanded = summary_svc.expand_summary_sources(
            user_a, str(active["_id"])
        )
        assert expanded.get("success"), expanded
        assert expanded.get("memories")
        stolen = summary_svc.expand_summary_sources(
            user_b, str(active["_id"])
        )
        assert not stolen.get("success")
        print("PASS: Source expansion with ownership checks.")

        # 8. Project grouping
        memory_service.save_memory(
            user_id=user_a,
            memory="my primary project is Freshcart",
            embedding=None,
            status="active",
            fact_key="primary_project",
            normalized_value="freshcart",
            operation_id=f"proj-{uuid4().hex}",
        )
        proj = summary_svc.generate_summary_for_type(
            user_a, "project", scope_key="freshcart", force=True
        )
        assert proj.get("success")
        assert "freshcart" in proj["summary"]["summary_text"].lower()
        assert "listifys" not in proj["summary"]["summary_text"].lower()
        print("PASS: Project topic grouping.")

        # 9. Insert verification failure path (mocked)
        from unittest.mock import MagicMock, patch

        mock_coll = MagicMock()
        mock_coll.find_one.side_effect = [None, None, None]  # no op, no active, no verify
        mock_coll.insert_one.return_value = MagicMock(inserted_id=ObjectId())
        with patch.object(
            summary_svc, "_summaries", return_value=mock_coll
        ), patch.object(
            summary_svc, "_memories", return_value=mem_coll
        ):
            failed = summary_svc.generate_summary_for_type(
                user_a, "general", scope_key="_general", force=True
            )
        assert not failed.get("success"), failed
        assert "verif" in (failed.get("error") or "").lower()
        print("PASS: Failed persistence does not report success.")

        print("\n=== All memory summarization Mongo checks passed. ===")

    finally:
        memory_service.memory_collection = original_mem
        memory_service.summary_collection = original_sum
        summary_svc.configure_summary_collections(original_sum, original_mem)
        if previous_auto is None:
            os.environ.pop("MEMORY_SUMMARY_AUTO_REFRESH", None)
        else:
            os.environ["MEMORY_SUMMARY_AUTO_REFRESH"] = previous_auto
        database.drop_collection(mem_name)
        database.drop_collection(sum_name)
        print(f"Cleanup: dropped {mem_name} and {sum_name}")


if __name__ == "__main__":
    main()
