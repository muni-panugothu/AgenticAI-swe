"""Stage 19 — Duplicate reinforcement and idempotency tests.

Verifies that:
- Explicit memory requests do NOT fall through to the agent loop.
- Retries with the same request_id are idempotent.
- Distinct user reaffirmations with different request_ids count separately.
- Failed operations do not report false success.
"""

import unittest
from unittest.mock import patch, MagicMock

from app import chatbot


class TestDuplicateReinforcementPrevention(unittest.TestCase):
    """The explicit memory handler must prevent duplicate processing."""

    USER_ID = "stage19-reinforce-test"

    @patch.object(chatbot, "reinforce_memory")
    @patch.object(chatbot, "find_logical_duplicate")
    @patch.object(chatbot, "create_embedding", return_value=[0.01] * 8)
    def test_duplicate_returns_handled_dict(
        self, _embed, logical_dup, reinforce
    ):
        logical_dup.return_value = {
            "success": True,
            "is_duplicate": True,
            "match": {
                "memory_id": "existing-mem-id",
                "memory": "I prefer Python",
                "importance": 4,
                "status": "active",
            },
            "memory_id": "existing-mem-id",
        }
        reinforce.return_value = {"success": True, "memory_id": "existing-mem-id"}

        result = chatbot.try_save_explicit_memory(
            "Remember that I prefer Python",
            self.USER_ID,
            request_id="req-1",
        )

        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("handled"))
        self.assertEqual(result.get("status"), "reinforced")
        self.assertIn("reaffirmed", result.get("message", "").lower())

    @patch.object(chatbot, "reinforce_memory")
    @patch.object(chatbot, "find_logical_duplicate")
    @patch.object(chatbot, "create_embedding", return_value=[0.01] * 8)
    def test_reinforcement_uses_request_scoped_operation_id(
        self, _embed, logical_dup, reinforce
    ):
        """Operation ID must include request_id so retries != new requests."""
        logical_dup.return_value = {
            "success": True,
            "is_duplicate": True,
            "match": {
                "memory_id": "mem-123",
                "memory": "I prefer Python",
                "importance": 4,
                "status": "active",
            },
            "memory_id": "mem-123",
        }
        reinforce.return_value = {"success": True}

        chatbot.try_save_explicit_memory(
            "Remember that I prefer Python",
            self.USER_ID,
            request_id="request-abc",
        )

        reinforce.assert_called_once()
        kwargs = reinforce.call_args.kwargs
        self.assertIn("operation_id", kwargs)
        self.assertIn("mem-123", kwargs["operation_id"])
        self.assertIn("request-abc", kwargs["operation_id"])

    @patch.object(chatbot, "save_memory")
    @patch.object(chatbot, "check_duplicate_memory")
    @patch.object(chatbot, "find_logical_duplicate")
    @patch.object(chatbot, "create_embedding", return_value=[0.02] * 8)
    def test_new_memory_returns_handled_dict(
        self, _embed, logical_dup, dup_check, save
    ):
        logical_dup.return_value = {
            "success": True,
            "is_duplicate": False,
            "match": None,
        }
        dup_check.return_value = {
            "success": True,
            "is_duplicate": False,
            "candidates": [],
        }
        save.return_value = {"success": True, "memory_id": "new-id"}

        result = chatbot.try_save_explicit_memory(
            "Remember that I am learning Rust",
            self.USER_ID,
            request_id="req-new",
        )

        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("handled"))
        self.assertEqual(result.get("status"), "saved")

    @patch.object(chatbot, "save_memory")
    @patch.object(chatbot, "check_duplicate_memory")
    @patch.object(chatbot, "find_logical_duplicate")
    @patch.object(chatbot, "create_embedding", return_value=[0.03] * 8)
    def test_save_failure_returns_false(
        self, _embed, logical_dup, dup_check, save
    ):
        logical_dup.return_value = {
            "success": True,
            "is_duplicate": False,
            "match": None,
        }
        dup_check.return_value = {
            "success": True,
            "is_duplicate": False,
            "candidates": [],
        }
        save.return_value = {"success": False, "error": "DB error"}

        result = chatbot.try_save_explicit_memory(
            "Remember that I am learning Go",
            self.USER_ID,
            request_id="req-fail",
        )

        self.assertFalse(result)


class TestChatWithAiExplicitMemoryRouting(unittest.TestCase):
    USER_ID = "stage19-routing-test"

    def setUp(self):
        # Isolate from Redis / in-process conversation leftovers for this user.
        chatbot.clear_conversation_memory(self.USER_ID)
        chatbot._get_conversation_store(self.USER_ID).clear()

    def tearDown(self):
        chatbot.clear_conversation_memory(self.USER_ID)
        chatbot._get_conversation_store(self.USER_ID).clear()

    @patch.object(chatbot, "try_save_explicit_memory")
    def test_truthy_result_bypasses_agent_loop(self, mock_save):
        mock_save.return_value = {
            "handled": True,
            "status": "saved",
            "message": "Memory saved.",
        }

        mock_llm = MagicMock()
        with patch.object(chatbot, "client", mock_llm):
            result = chatbot.chat_with_ai(
                "Remember that I like Python",
                user_id=self.USER_ID,
            )

        self.assertEqual(result, "Memory saved.")
        mock_llm.chat.completions.create.assert_not_called()

    @patch.object(chatbot, "try_save_explicit_memory")
    def test_false_result_enters_agent_loop(self, mock_save):
        mock_save.return_value = False

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.content = "Hello!"

        with patch.object(chatbot, "client") as mock_client:
            mock_client.chat.completions.create.return_value = mock_response
            result = chatbot.chat_with_ai(
                "Hello, how are you?",
                user_id=self.USER_ID,
            )

        # Greeting must return the model text without a B6 partial-workflow
        # suffix — advisory never-started boards must not annotate.
        self.assertEqual(result, "Hello!")
        self.assertNotIn("not fully finished", result)
        mock_client.chat.completions.create.assert_called()


class TestRetryIdempotency(unittest.TestCase):
    """Same request_id → same op id. Different request_id → different op id."""

    USER_ID = "stage19-retry-test"

    @patch.object(chatbot, "reinforce_memory")
    @patch.object(chatbot, "find_logical_duplicate")
    @patch.object(chatbot, "create_embedding", return_value=[0.01] * 8)
    def test_same_request_id_uses_same_operation_id(
        self, _embed, logical_dup, reinforce
    ):
        logical_dup.return_value = {
            "success": True,
            "is_duplicate": True,
            "match": {
                "memory_id": "stable-id",
                "memory": "I prefer Python",
                "importance": 4,
                "status": "active",
            },
            "memory_id": "stable-id",
        }
        reinforce.return_value = {"success": True}

        chatbot.try_save_explicit_memory(
            "Remember that I prefer Python",
            self.USER_ID,
            request_id="same-request",
        )
        first_op_id = reinforce.call_args.kwargs.get("operation_id")

        chatbot.try_save_explicit_memory(
            "Remember that I prefer Python",
            self.USER_ID,
            request_id="same-request",
        )
        second_op_id = reinforce.call_args.kwargs.get("operation_id")

        self.assertEqual(first_op_id, second_op_id)

    @patch.object(chatbot, "reinforce_memory")
    @patch.object(chatbot, "find_logical_duplicate")
    @patch.object(chatbot, "create_embedding", return_value=[0.01] * 8)
    def test_different_request_ids_use_different_operation_ids(
        self, _embed, logical_dup, reinforce
    ):
        logical_dup.return_value = {
            "success": True,
            "is_duplicate": True,
            "match": {
                "memory_id": "stable-id",
                "memory": "I prefer Python",
                "importance": 4,
                "status": "active",
            },
            "memory_id": "stable-id",
        }
        reinforce.return_value = {"success": True, "idempotent": False}

        chatbot.try_save_explicit_memory(
            "Remember that I prefer Python",
            self.USER_ID,
            request_id="request-1",
        )
        first_op_id = reinforce.call_args.kwargs.get("operation_id")

        chatbot.try_save_explicit_memory(
            "Remember that I prefer Python",
            self.USER_ID,
            request_id="request-2",
        )
        second_op_id = reinforce.call_args.kwargs.get("operation_id")

        self.assertNotEqual(first_op_id, second_op_id)
        self.assertIn("request-1", first_op_id)
        self.assertIn("request-2", second_op_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
