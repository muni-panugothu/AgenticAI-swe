"""Stage 17 chatbot-flow tests using mocks; no MongoDB or LLM calls."""

import unittest
from unittest.mock import patch

from app import chatbot


class TestStage17ChatbotMemoryFlow(unittest.TestCase):
    USER_ID = "stage17-test-user"
    OLD_ROLE = {
        "memory_id": "old-role-memory-id",
        "memory": "My role is Junior Developer",
        "importance": 4,
        "confidence": 0.95,
        "status": "active",
        "fact_key": "role",
        "normalized_value": "junior developer",
        "conflict_group_id": None,
        "version": 1,
    }

    @patch.object(
        chatbot,
        "persist_memory_conflict_pair",
        return_value={"success": True, "verified": True},
    )
    @patch.object(chatbot, "save_memory")
    @patch.object(chatbot, "find_active_memories_by_fact_key")
    @patch.object(chatbot, "check_duplicate_memory")
    @patch.object(chatbot, "find_logical_duplicate")
    @patch.object(chatbot, "create_embedding", return_value=[0.01] * 8)
    def test_new_exclusive_value_is_saved_for_review(
        self,
        _embedding,
        logical_dup,
        duplicate_check,
        by_key,
        save_memory,
        persist_pair,
    ):
        logical_dup.return_value = {
            "success": True,
            "is_duplicate": False,
            "match": None,
        }
        duplicate_check.return_value = {
            "success": True,
            "is_duplicate": False,
            "candidates": [self.OLD_ROLE],
        }
        by_key.return_value = {
            "success": True,
            "memories": [self.OLD_ROLE],
        }
        save_memory.return_value = {
            "success": True,
            "memory_id": "pending-role-memory-id",
            "status": "needs_review",
        }

        result = chatbot.try_save_explicit_memory(
            "Remember that my role is Software Engineer.",
            self.USER_ID,
        )
        persist_pair.assert_called()

        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("handled"))
        self.assertEqual(result.get("status"), "needs_review")
        self.assertIn("did not replace", result.get("message", ""))

        saved = save_memory.call_args.kwargs
        self.assertEqual(saved["status"], "needs_review")
        self.assertEqual(saved["fact_key"], "role")
        self.assertEqual(saved["conflict_type"], "possible_contradiction")
        self.assertEqual(saved["conflict_with_memory_id"], "old-role-memory-id")

    @patch.object(chatbot, "save_memory")
    @patch.object(chatbot, "check_duplicate_memory")
    @patch.object(chatbot, "find_logical_duplicate")
    @patch.object(chatbot, "create_embedding", return_value=[0.02] * 8)
    def test_learning_another_language_does_not_create_conflict(
        self, _embedding, logical_dup, duplicate_check, save_memory
    ):
        logical_dup.return_value = {
            "success": True,
            "is_duplicate": False,
            "match": None,
        }
        duplicate_check.return_value = {
            "success": True,
            "is_duplicate": False,
            "candidates": [
                {
                    "memory_id": "learning-js-id",
                    "memory": "I am learning JavaScript",
                    "status": "active",
                }
            ],
        }
        save_memory.return_value = {
            "success": True,
            "memory_id": "learning-python-id",
        }

        result = chatbot.try_save_explicit_memory(
            "Remember that I am learning Python.",
            self.USER_ID,
        )

        self.assertTrue(result)
        if isinstance(result, dict):
            self.assertTrue(result.get("handled"))
        saved = save_memory.call_args.kwargs
        self.assertNotIn("conflict_with_memory_id", saved)

    @patch.object(chatbot, "retire_sibling_exclusive_actives")
    @patch.object(chatbot, "supersede_memory")
    @patch.object(chatbot, "_collect_update_candidates")
    @patch.object(chatbot, "create_embedding", return_value=[0.03] * 8)
    def test_confirmation_sentence_without_final_punctuation_is_accepted(
        self, _embedding, collect, supersede, _retire
    ):
        collect.return_value = ([self.OLD_ROLE], [])
        supersede.return_value = {
            "success": True,
            "memory_id": "new-role-memory-id",
            "old_memory_id": "old-role-memory-id",
            "version": 2,
        }

        result = chatbot.try_update_explicit_memory(
            "Update my role to Software Engineer",
            self.USER_ID,
        )

        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("success"))
        supersede.assert_called_once()
        update_args = supersede.call_args.kwargs
        self.assertEqual(update_args["memory_id"], "old-role-memory-id")
        self.assertEqual(update_args["memory"], "My role is Software Engineer")
        self.assertEqual(update_args["fact_key"], "role")

    @patch.object(chatbot, "supersede_memory")
    @patch.object(chatbot, "_collect_update_candidates")
    def test_ambiguous_update_does_not_supersede_any_memory(
        self, collect, supersede
    ):
        collect.return_value = (
            [
                self.OLD_ROLE,
                {
                    "memory_id": "second-role-id",
                    "memory": "My role is Team Lead",
                    "status": "active",
                    "fact_key": "role",
                    "normalized_value": "team lead",
                },
            ],
            [],
        )

        result = chatbot.try_update_explicit_memory(
            "Update my role to Software Engineer.",
            self.USER_ID,
        )

        self.assertIsInstance(result, dict)
        self.assertFalse(result.get("success"))
        self.assertIn("multiple", result.get("message", "").lower())
        supersede.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
