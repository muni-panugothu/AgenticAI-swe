"""Stage 19.1 regression tests for live-test failures.

Covers:
1. Request-scoped reinforcement idempotency
2. Case-insensitive duplicates across needs_review
3. Conflicting active facts trigger clarification
4. Pending-review facts appear in retrieval context when relevant
5. Update aliases like 'primary project' resolve to fact_key=project
"""

import unittest
from unittest.mock import MagicMock, patch

from bson import ObjectId

from app import chatbot
from app.services import memory_service
from app.services.memory_conflict_service import (
    classify_duplicate,
    extract_exclusive_fact,
    normalize_memory_text,
)


class TestCaseInsensitiveDedup(unittest.TestCase):
    def test_case_variants_normalize_identically(self):
        variants = [
            "my project is Alpha",
            "MY PROJECT IS ALPHA",
            "My Project Is Alpha",
            "my project is Alpha.",
        ]
        norms = {normalize_memory_text(v) for v in variants}
        self.assertEqual(len(norms), 1)
        self.assertEqual(norms.pop(), "my project is alpha")

    def test_case_variants_are_exact_duplicates(self):
        result = classify_duplicate(
            "my project is Alpha",
            "MY PROJECT IS ALPHA",
        )
        self.assertEqual(result["classification"], "exact_duplicate")

    def test_project_fact_extraction_is_case_insensitive(self):
        for text in (
            "my project is Alpha",
            "MY PROJECT IS ALPHA",
            "My Project Is Alpha",
        ):
            fact = extract_exclusive_fact(text)
            self.assertEqual(fact, ("primary_project", "alpha"))

    @patch.object(chatbot, "save_memory")
    @patch.object(chatbot, "find_logical_duplicate")
    @patch.object(chatbot, "create_embedding", return_value=[0.1] * 8)
    def test_needs_review_duplicate_does_not_create_another_pending(
        self, _embed, logical_dup, save_memory
    ):
        logical_dup.return_value = {
            "success": True,
            "is_duplicate": True,
            "match": {
                "memory_id": "pending-alpha",
                "memory": "my project is Alpha",
                "status": "needs_review",
                "fact_key": "project",
                "normalized_value": "alpha",
            },
        }

        result = chatbot.try_save_explicit_memory(
            "Remember that MY PROJECT IS ALPHA",
            "user-1",
            request_id="req-1",
        )

        self.assertTrue(result.get("handled"))
        self.assertEqual(result.get("status"), "needs_review")
        save_memory.assert_not_called()


class TestLogicalDuplicateService(unittest.TestCase):
    def test_find_logical_duplicate_matches_needs_review_by_fact(self):
        collection = MagicMock()
        pending_id = ObjectId()
        collection.find.return_value.sort.return_value.limit.return_value = iter([
            {
                "_id": pending_id,
                "user_id": "user-1",
                "owner_id": "user-1",
                "permission_level": "private",
                "shared_with": [],
                "memory": "My project is Alpha",
                "status": "needs_review",
                "fact_key": "project",
                "normalized_value": "alpha",
                "importance": 3,
                "confidence": 0.95,
                "version": 1,
            }
        ])

        with patch.object(memory_service, "memory_collection", collection):
            # First query is find_memories_by_fact_key; second is text scan.
            # Both use collection.find — configure side effects carefully.
            fact_cursor = MagicMock()
            fact_cursor.sort.return_value.limit.return_value = iter([
                {
                    "_id": pending_id,
                    "user_id": "user-1",
                    "owner_id": "user-1",
                    "permission_level": "private",
                    "shared_with": [],
                    "memory": "My project is Alpha",
                    "status": "needs_review",
                    "fact_key": "project",
                    "normalized_value": "alpha",
                    "importance": 3,
                    "confidence": 0.95,
                    "version": 1,
                }
            ])
            text_cursor = MagicMock()
            text_cursor.sort.return_value.limit.return_value = iter([])
            collection.find.side_effect = [fact_cursor, text_cursor]

            result = memory_service.find_logical_duplicate(
                user_id="user-1",
                memory_text="MY PROJECT IS ALPHA",
                fact_key="project",
                normalized_value="alpha",
            )

        self.assertTrue(result["success"])
        self.assertTrue(result["is_duplicate"])
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["memory_id"], str(pending_id))


class TestConflictingActiveFactsClarification(unittest.TestCase):
    def test_build_memory_context_includes_conflict_notice(self):
        memories = [
            {
                "memory": "My project is Alpha",
                "status": "active",
                "fact_key": "project",
                "normalized_value": "alpha",
                "importance": 3,
                "confidence": 0.9,
                "is_stale": False,
            },
            {
                "memory": "My project is Beta",
                "status": "active",
                "fact_key": "project",
                "normalized_value": "beta",
                "importance": 3,
                "confidence": 0.9,
                "is_stale": False,
            },
        ]
        conflicts = memory_service.detect_conflicting_active_facts(memories)
        context = chatbot.build_memory_context(memories, conflicts=conflicts)

        self.assertIn("MEMORY_CONFLICT_NOTICE", context)
        self.assertIn("Ask the user which value is currently correct", context)
        self.assertIn("alpha", context)
        self.assertIn("beta", context)

    @patch.object(chatbot, "load_long_term_memory")
    @patch.object(chatbot, "try_save_explicit_memory", return_value=False)
    @patch.object(chatbot, "try_update_explicit_memory", return_value=None)
    @patch.object(chatbot, "try_delete_explicit_memory", return_value=None)
    def test_chat_asks_clarification_for_conflicting_project_query(
        self, _delete, _update, _save, load_memory
    ):
        load_memory.return_value = {
            "memories": [
                {
                    "memory_id": "1",
                    "memory": "My project is Alpha",
                    "status": "active",
                    "fact_key": "project",
                    "normalized_value": "alpha",
                },
                {
                    "memory_id": "2",
                    "memory": "My project is Beta",
                    "status": "active",
                    "fact_key": "project",
                    "normalized_value": "beta",
                },
            ],
            "conflicts": [
                {
                    "fact_key": "project",
                    "values": ["alpha", "beta"],
                    "memories": [],
                }
            ],
        }

        answer = chatbot.chat_with_ai(
            "What is my project?",
            user_id="conflict-user",
        )

        self.assertIn("conflicting", answer.lower())
        self.assertIn("alpha", answer.lower())
        self.assertIn("beta", answer.lower())
        self.assertIn("which one", answer.lower())


class TestPendingReviewInContext(unittest.TestCase):
    def test_pending_facts_are_labeled_separately(self):
        memories = [
            {
                "memory": "My project is Beta",
                "status": "active",
                "importance": 3,
                "confidence": 0.9,
                "is_stale": False,
            },
            {
                "memory": "I am building Freshcart grocery marketplace",
                "status": "needs_review",
                "importance": 4,
                "confidence": 0.9,
                "is_stale": False,
            },
        ]
        context = chatbot.build_memory_context(memories)
        self.assertIn("CURRENT_ACTIVE_FACTS", context)
        self.assertIn("PENDING_REVIEW_FACTS", context)
        self.assertIn("Freshcart", context)
        self.assertIn("status=needs_review", context)


class TestUpdateAliasResolution(unittest.TestCase):
    @patch.object(chatbot, "retire_sibling_exclusive_actives")
    @patch.object(chatbot, "supersede_memory")
    @patch.object(chatbot, "_collect_update_candidates")
    @patch.object(chatbot, "create_embedding", return_value=[0.2] * 8)
    def test_primary_project_alias_updates_project_fact(
        self, _embed, collect, supersede, _retire
    ):
        collect.return_value = (
            [
                {
                    "memory_id": "beta-id",
                    "memory": "My project is Beta",
                    "status": "active",
                    "fact_key": "primary_project",
                    "normalized_value": "beta",
                    "reinforcement_count": 1,
                }
            ],
            [],
        )
        supersede.return_value = {
            "success": True,
            "memory_id": "new-id",
            "old_memory_id": "beta-id",
            "version": 2,
        }

        result = chatbot.try_update_explicit_memory(
            "Update my primary project to Alpha",
            "user-1",
        )

        self.assertTrue(result["success"])
        supersede.assert_called_once()
        self.assertEqual(
            supersede.call_args.kwargs["fact_key"], "primary_project"
        )


class TestReinforcementAuditIdempotency(unittest.TestCase):
    def test_same_operation_id_is_idempotent_via_audit(self):
        collection = MagicMock()
        audit = MagicMock()
        memory_id = ObjectId()
        op_id = "explicit-reinforce-user-mem-req1"

        audit.find_one.return_value = {
            "user_id": "user",
            "memory_id": str(memory_id),
            "operation_id": op_id,
            "event_type": "memory_reinforced",
        }
        collection.find_one.return_value = {
            "_id": memory_id,
            "user_id": "user",
            "last_reinforced_at": "ts",
            "reinforcement_count": 1,
        }

        with (
            patch.object(memory_service, "memory_collection", collection),
            patch.object(memory_service, "audit_collection", audit),
        ):
            result = memory_service.reinforce_memory(
                "user",
                str(memory_id),
                operation_id=op_id,
            )

        self.assertTrue(result["success"])
        self.assertTrue(result["idempotent"])
        collection.update_one.assert_not_called()

    def test_different_operation_id_increments_again(self):
        collection = MagicMock()
        audit = MagicMock()
        memory_id = ObjectId()

        audit.find_one.return_value = None
        collection.update_one.return_value.matched_count = 1
        collection.find_one.return_value = {
            "_id": memory_id,
            "user_id": "user",
            "reinforcement_count": 2,
        }

        with (
            patch.object(memory_service, "memory_collection", collection),
            patch.object(memory_service, "audit_collection", audit),
        ):
            result = memory_service.reinforce_memory(
                "user",
                str(memory_id),
                operation_id="explicit-reinforce-user-mem-req2",
            )

        self.assertTrue(result["success"])
        self.assertFalse(result.get("idempotent"))
        collection.update_one.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
