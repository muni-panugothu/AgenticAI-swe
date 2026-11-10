"""Fresh-session pending-review retrieval and conflict-group propagation."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app import chatbot
from app.services.memory_service import (
    detect_active_pending_conflicts,
    semantic_search_memories,
)


class TestActivePendingConflictDetection(unittest.TestCase):
    def test_mongodb_active_postgresql_pending(self):
        memories = [
            {
                "memory_id": "a1",
                "memory": "my preferred database is MongoDB",
                "status": "active",
                "fact_key": "preferred_database",
                "normalized_value": "mongodb",
                "conflict_group_id": "cg-db-1",
            },
            {
                "memory_id": "p1",
                "memory": "my preferred database is PostgreSQL",
                "status": "needs_review",
                "fact_key": "preferred_database",
                "normalized_value": "postgresql",
                "conflict_group_id": "cg-db-1",
            },
        ]
        conflicts = detect_active_pending_conflicts(memories)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["conflict_kind"], "active_vs_pending")
        self.assertEqual(conflicts[0]["fact_key"], "preferred_database")
        self.assertEqual(conflicts[0]["active_values"], ["mongodb"])
        self.assertEqual(conflicts[0]["pending_values"], ["postgresql"])


class TestPendingRetrievalInSemanticSearch(unittest.TestCase):
    def test_fact_key_merge_surfaces_pending_postgresql(self):
        """Fresh session: pending DB fact must appear even if vector misses it."""
        active_mongo = {
            "memory_id": "a-mongo",
            "user_id": "fresh-user",
            "owner_id": "fresh-user",
            "permission_level": "private",
            "shared_with": [],
            "memory": "my preferred database is MongoDB",
            "score": 0.86,
            "status": "active",
            "fact_key": "preferred_database",
            "normalized_value": "mongodb",
            "importance": 4,
            "confidence": 0.75,
            "conflict_group_id": "cg-db",
        }
        pending_pg = {
            "memory_id": "p-pg",
            "user_id": "fresh-user",
            "owner_id": "fresh-user",
            "permission_level": "private",
            "shared_with": [],
            "memory": "my preferred database is PostgreSQL",
            "score": 0.84,
            "status": "needs_review",
            "fact_key": "preferred_database",
            "normalized_value": "postgresql",
            "importance": 4,
            "confidence": 0.75,
            "conflict_group_id": "cg-db",
        }

        def fake_vector(user_id, embedding, *, status_filter, limit, min_score, **_kwargs):
            # Simulate Atlas missing the pending row in top-k for needs_review.
            if status_filter == {"status": "needs_review"}:
                return []
            return [dict(active_mongo)]

        def fake_by_key(user_id, fact_key, *, statuses=("active",), limit=20, **_):
            rows = []
            if "active" in statuses:
                rows.append(dict(active_mongo))
            if "needs_review" in statuses:
                rows.append(dict(pending_pg))
            return {"success": True, "memories": rows}

        with patch(
            "app.services.memory_service.create_embedding",
            return_value=[0.1] * 8,
        ), patch(
            "app.services.memory_service._vector_search_memories",
            side_effect=fake_vector,
        ), patch(
            "app.services.memory_service.find_memories_by_fact_key",
            side_effect=fake_by_key,
        ), patch(
            "app.services.memory_service.annotate_memories_with_lifecycle",
            side_effect=lambda xs: xs,
        ), patch(
            "app.services.memory_service._record_memory_access",
        ), patch(
            "app.services.memory_service.record_audit_event",
        ), patch(
            "app.services.memory_service.memory_collection"
        ) as mock_coll:
            mock_coll.find.return_value.limit.return_value = []
            result = semantic_search_memories(
                "fresh-user",
                "Which database do I currently prefer? Identify any conflicts.",
                limit=5,
                min_score=0.80,
                include_needs_review=True,
            )

        self.assertTrue(result["success"])
        statuses = {m["status"] for m in result["memories"]}
        texts = " ".join(m["memory"] for m in result["memories"]).lower()
        self.assertIn("needs_review", statuses)
        self.assertIn("postgresql", texts)
        self.assertIn("mongodb", texts)
        self.assertGreaterEqual(len(result["conflicts"]), 1)
        kinds = {c.get("conflict_kind") for c in result["conflicts"]}
        self.assertIn("active_vs_pending", kinds)

    def test_build_context_includes_pending_and_conflict_notice(self):
        memories = [
            {
                "memory": "my preferred database is MongoDB",
                "status": "active",
                "fact_key": "preferred_database",
                "normalized_value": "mongodb",
                "importance": 4,
                "confidence": 0.75,
                "is_stale": False,
                "freshness_score": 1.0,
            },
            {
                "memory": "my preferred database is PostgreSQL",
                "status": "needs_review",
                "fact_key": "preferred_database",
                "normalized_value": "postgresql",
                "importance": 4,
                "confidence": 0.75,
                "is_stale": False,
                "freshness_score": 1.0,
            },
        ]
        conflicts = detect_active_pending_conflicts(memories)
        context = chatbot.build_memory_context(memories, conflicts=conflicts)
        self.assertIn("CURRENT_ACTIVE_FACTS", context)
        self.assertIn("PENDING_REVIEW_FACTS", context)
        self.assertIn("MongoDB", context)
        self.assertIn("PostgreSQL", context)
        self.assertIn("MEMORY_CONFLICT_NOTICE", context)
        self.assertIn("pending review", context.lower())


if __name__ == "__main__":
    unittest.main()
