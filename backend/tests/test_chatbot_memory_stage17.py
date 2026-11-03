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
        "conflict_group_id": None,
        "version": 1,
    }

    @patch.object(chatbot, "save_memory")
    @patch.object(chatbot, "check_duplicate_memory")
    @patch.object(chatbot, "create_embedding", return_value=[0.01] * 8)
    def test_new_exclusive_value_is_saved_for_review(
        self, _embedding, duplicate_check, save_memory
    ):
        duplicate_check.return_value = {
            "success": True,
            "is_duplicate": False,
            "candidates": [self.OLD_ROLE],
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
    @patch.object(chatbot, "create_embedding", return_value=[0.02] * 8)
    def test_learning_another_language_does_not_create_conflict(
        self, _embedding, duplicate_check, save_memory
    ):
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

        self.assertIs(result, True)
        saved = save_memory.call_args.kwargs
        self.assertNotIn("status", saved)
        self.assertNotIn("conflict_with_memory_id", saved)

    @patch.object(chatbot, "supersede_memory")
    @patch.object(chatbot, "semantic_search_memories")
    @patch.object(chatbot, "create_embedding", return_value=[0.03] * 8)
    def test_confirmation_sentence_without_final_punctuation_is_accepted(
        self, _embedding, semantic_search, supersede
    ):
        semantic_search.return_value = {
            "success": True,
            "memories": [self.OLD_ROLE],
        }
        supersede.return_value = {
            "success": True,
            "memory_id": "new-role-memory-id",
            "version": 2,
        }

        # This is the exact form suggested by the conflict response.
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
    @patch.object(chatbot, "semantic_search_memories")
    def test_ambiguous_update_does_not_supersede_any_memory(
        self, semantic_search, supersede
    ):
        semantic_search.return_value = {
            "success": True,
            "memories": [
                self.OLD_ROLE,
                {
                    "memory_id": "second-role-id",
                    "memory": "My previous role was Team Lead",
                    "status": "active",
                },
            ],
        }

        result = chatbot.try_update_explicit_memory(
            "Update my role to Software Engineer.",
            self.USER_ID,
        )

        self.assertIsInstance(result, dict)
        self.assertFalse(result.get("success"))
        self.assertIn("multiple memories", result.get("message", "").lower())
        supersede.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)