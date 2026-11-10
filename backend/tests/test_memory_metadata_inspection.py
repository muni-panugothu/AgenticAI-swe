"""Read-only memory metadata inspection tests."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
from bson import ObjectId

from app import chatbot
from app.services.memory_service import (
    compare_conflict_group_ids,
    format_memory_metadata_report,
    inspect_memory_metadata,
)


class TestConflictGroupComparison(unittest.TestCase):
    def test_match(self):
        result = compare_conflict_group_ids(
            [
                {"conflict_group_id": "cg-1"},
                {"conflict_group_id": "cg-1"},
            ]
        )
        self.assertTrue(result["same_conflict_group"])
        self.assertEqual(result["verdict"], "match")

    def test_inconsistent_missing(self):
        result = compare_conflict_group_ids(
            [
                {"conflict_group_id": None},
                {"conflict_group_id": "cg-1"},
            ]
        )
        self.assertFalse(result["same_conflict_group"])
        self.assertEqual(result["verdict"], "inconsistent_missing")

    def test_all_missing(self):
        result = compare_conflict_group_ids(
            [
                {"conflict_group_id": None},
                {"conflict_group_id": None},
            ]
        )
        self.assertEqual(result["verdict"], "all_missing")

    def test_mismatch(self):
        result = compare_conflict_group_ids(
            [
                {"conflict_group_id": "cg-a"},
                {"conflict_group_id": "cg-b"},
            ]
        )
        self.assertEqual(result["verdict"], "inconsistent_mismatch")


class TestInspectMemoryMetadataService(unittest.TestCase):
    def test_returns_actual_null_fields_and_does_not_fabricate_group(self):
        mongo_id = ObjectId()
        pg_id = ObjectId()
        docs = [
            {
                "_id": pg_id,
                "user_id": "inspect-user",
                "status": "needs_review",
                "fact_key": "preferred_database",
                "normalized_value": "postgresql",
                "conflict_group_id": "cg-shared",
                "conflict_type": "possible_contradiction",
                "conflict_with_memory_id": mongo_id,
                "memory": "my preferred database is PostgreSQL",
            },
            {
                "_id": mongo_id,
                "user_id": "inspect-user",
                "status": "active",
                "fact_key": None,
                "normalized_value": None,
                "conflict_group_id": None,
                "conflict_type": "none",
                "conflict_with_memory_id": None,
                "memory": "my preferred database is MongoDB",
            },
        ]

        mock_coll = MagicMock()
        mock_coll.find.return_value.sort.return_value.limit.return_value = docs

        with patch(
            "app.services.memory_service.memory_collection",
            mock_coll,
        ):
            result = inspect_memory_metadata(
                "inspect-user",
                fact_key="preferred_database",
            )

        self.assertTrue(result["success"])
        self.assertTrue(result["read_only"])
        self.assertEqual(result["record_count"], 2)

        by_status = {r["status"]: r for r in result["records"]}
        self.assertIsNone(by_status["active"]["fact_key"])
        self.assertIsNone(by_status["active"]["normalized_value"])
        self.assertIsNone(by_status["active"]["conflict_group_id"])
        self.assertEqual(
            by_status["needs_review"]["fact_key"],
            "preferred_database",
        )
        self.assertEqual(
            by_status["needs_review"]["normalized_value"],
            "postgresql",
        )
        comparison = result["conflict_group_comparison"]
        self.assertEqual(comparison["verdict"], "inconsistent_missing")
        self.assertFalse(comparison["same_conflict_group"])

        report = format_memory_metadata_report(result)
        self.assertIn('"fact_key": null', report)
        self.assertIn('"conflict_group_id": null', report)
        self.assertIn("inconsistent_missing", report)
        self.assertIn("No records were modified", report)

    def test_rejects_missing_user(self):
        result = inspect_memory_metadata("", fact_key="preferred_database")
        self.assertFalse(result["success"])


class TestInspectRequestRouting(unittest.TestCase):
    def test_detects_explicit_inspect_query(self):
        msg = (
            "Inspect my MongoDB and PostgreSQL preferred database memories. "
            "Show their actual fact_key, normalized_value, conflict_group_id, "
            "and status. Confirm whether both records belong to the same "
            "conflict group. Do not modify anything."
        )
        self.assertTrue(chatbot.is_memory_metadata_inspect_request(msg))
        self.assertEqual(
            chatbot.infer_inspect_fact_key(msg),
            "preferred_database",
        )

    def test_ordinary_preference_question_is_not_inspect(self):
        msg = "Which database do I currently prefer?"
        self.assertFalse(chatbot.is_memory_metadata_inspect_request(msg))

    def test_memory_architecture_request_is_not_personal_metadata(self):
        msg = (
            "Inspect my memory architecture. Verify implementations, "
            "read complete implementations and relevant tests. Do not modify."
        )
        self.assertTrue(chatbot.is_memory_system_code_inspection_request(msg))
        self.assertFalse(chatbot.is_memory_metadata_inspect_request(msg))
        self.assertIsNone(chatbot.try_inspect_memory_metadata(msg, "u1"))

    def test_personal_metadata_inspect_still_routes_to_memory(self):
        from app.services import agent_orchestration as orch

        msg = (
            "Inspect my MongoDB and PostgreSQL preferred database memories. "
            "Show their actual fact_key, normalized_value, conflict_group_id, "
            "and status. Do not modify anything."
        )
        self.assertTrue(chatbot.is_memory_metadata_inspect_request(msg))
        self.assertFalse(chatbot.is_memory_system_code_inspection_request(msg))
        self.assertEqual(orch.classify_intent(msg), orch.INTENT_MEMORY)

    @patch.object(chatbot, "inspect_memory_metadata")
    @patch.object(chatbot, "restore_task_context_from_redis")
    @patch.object(chatbot, "save_task_context_to_redis")
    @patch.object(chatbot, "add_to_conversation_memory")
    def test_chat_returns_metadata_not_conflict_prompt(
        self,
        _add,
        _redis_save,
        _redis_restore,
        inspect_mock,
    ):
        inspect_mock.return_value = {
            "success": True,
            "read_only": True,
            "user_id": "inspect-user",
            "fact_key": "preferred_database",
            "record_count": 2,
            "records": [
                {
                    "memory_id": "m1",
                    "user_id": "inspect-user",
                    "status": "active",
                    "fact_key": None,
                    "normalized_value": None,
                    "conflict_group_id": None,
                    "conflict_type": "none",
                    "conflict_with_memory_id": None,
                    "memory": "my preferred database is MongoDB",
                },
                {
                    "memory_id": "m2",
                    "user_id": "inspect-user",
                    "status": "needs_review",
                    "fact_key": "preferred_database",
                    "normalized_value": "postgresql",
                    "conflict_group_id": "cg-1",
                    "conflict_type": "possible_contradiction",
                    "conflict_with_memory_id": "m1",
                    "memory": "my preferred database is PostgreSQL",
                },
            ],
            "conflict_group_comparison": {
                "verdict": "inconsistent_missing",
                "same_conflict_group": False,
                "conflict_group_ids": [None, "cg-1"],
                "distinct_non_null_group_ids": ["cg-1"],
                "message": "Some records have a conflict_group_id while others are null.",
            },
        }

        msg = (
            "Inspect my MongoDB and PostgreSQL preferred database memories. "
            "Show their actual fact_key, normalized_value, conflict_group_id, "
            "and status. Confirm whether both records belong to the same "
            "conflict group. Do not modify anything."
        )
        answer = chatbot.chat_with_ai(
            msg,
            user_id="inspect-user",
            request_id="req-inspect",
        )

        inspect_mock.assert_called_once()
        self.assertIn("read-only", answer.lower())
        self.assertIn("preferred_database", answer)
        self.assertIn("inconsistent_missing", answer)
        self.assertIn("null", answer.lower())
        self.assertNotIn("Update my preferred database", answer)


if __name__ == "__main__":
    unittest.main()
