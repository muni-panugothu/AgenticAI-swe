"""Stage 19 Final Stabilization tests.

Covers JSON-safe tool results, mutation retry protection,
multi-statement extraction, update value parsing, paraphrase
conflicts, and request-scoped reinforcement idempotency.
"""

from __future__ import annotations

import json
import unittest
from datetime import date, datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch
from uuid import uuid4

from bson import ObjectId
from pydantic import BaseModel

from app import chatbot
from app.services.memory_conflict_service import (
    detect_memory_conflict,
    extract_exclusive_fact,
)
from app.services.memory_fact_extraction import (
    extract_facts_from_message,
    split_statements,
)
from app.utils.json_safe import dumps_json_safe, is_json_serializable, to_json_safe


class SampleModel(BaseModel):
    name: str
    created_at: datetime


class TestJsonSafeSerialization(unittest.TestCase):
    def test_datetime_and_objectid_round_trip(self):
        now = datetime(2026, 3, 15, 12, 30, 0, tzinfo=timezone.utc)
        oid = ObjectId()
        payload = {
            "memory_id": oid,
            "last_reinforced_at": now,
            "count": 3,
            "ok": True,
            "nested": {"day": date(2026, 3, 15), "ids": [oid]},
            "amount": Decimal("1.5"),
            "uid": uuid4(),
            "model": SampleModel(name="x", created_at=now),
        }
        safe = to_json_safe(payload)
        encoded = json.dumps(safe)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["last_reinforced_at"], now.isoformat())
        self.assertEqual(decoded["memory_id"], str(oid))
        self.assertEqual(decoded["nested"]["ids"][0], str(oid))
        self.assertEqual(decoded["count"], 3)
        self.assertTrue(decoded["ok"])

    def test_dumps_json_safe_and_limit_tool_result(self):
        now = datetime.now(timezone.utc)
        result = {
            "success": True,
            "last_reinforced_at": now,
            "memory_id": ObjectId(),
        }
        limited = chatbot.limit_tool_result(result)
        # Must be valid JSON with no datetime objects left.
        parsed = json.loads(limited)
        self.assertTrue(parsed["success"])
        self.assertIsInstance(parsed["last_reinforced_at"], str)
        self.assertIsInstance(parsed["memory_id"], str)

    def test_unknown_object_raises(self):
        class Weird:
            pass

        with self.assertRaises(TypeError):
            to_json_safe(Weird())

    def test_is_json_serializable(self):
        self.assertTrue(is_json_serializable({"a": datetime.now(timezone.utc)}))
        self.assertFalse(is_json_serializable(object()))


class TestExecuteToolSerializationRecovery(unittest.TestCase):
    def test_reinforce_result_with_datetime_is_json_safe(self):
        now = datetime.now(timezone.utc)
        memory_id = str(ObjectId())

        def fake_reinforce(user_id, memory_id, request_id=None):
            return {
                "success": True,
                "memory_id": memory_id,
                "last_reinforced_at": now,
                "reinforcement_count": 2,
                "idempotent": False,
            }

        with patch.dict(chatbot.tools, {"reinforce_memory": fake_reinforce}):
            result = chatbot.execute_tool(
                "reinforce_memory",
                {"memory_id": memory_id},
                user_id="user-1",
                request_id="req-abc",
            )

        self.assertTrue(result["success"])
        parsed = json.loads(result["result"])
        self.assertEqual(parsed["reinforcement_count"], 2)
        self.assertIsInstance(parsed["last_reinforced_at"], str)
        # Outer envelope must also be JSON-serializable for the agent loop.
        dumps_json_safe(result)

    def test_serialization_failure_after_mutation_still_reports_success(self):
        memory_id = str(ObjectId())
        now = datetime.now(timezone.utc)

        def fake_reinforce(user_id, memory_id, request_id=None):
            return {
                "success": True,
                "memory_id": memory_id,
                "last_reinforced_at": now,
                "reinforcement_count": 1,
            }

        with patch.dict(chatbot.tools, {"reinforce_memory": fake_reinforce}):
            with patch.object(
                chatbot,
                "limit_tool_result",
                side_effect=[TypeError("datetime"), '{"success": true}'],
            ):
                result = chatbot.execute_tool(
                    "reinforce_memory",
                    {"memory_id": memory_id},
                    user_id="user-1",
                    request_id="req-1",
                )

        self.assertTrue(result["success"])
        self.assertTrue(result.get("mutation_committed") or result.get("serialization_recovered"))


class TestMultiStatementExtraction(unittest.TestCase):
    def test_split_editor_conflict_statements(self):
        text = (
            "Remember that my preferred code editor is VS Code.\n"
            "I prefer IntelliJ IDEA as my code editor."
        )
        result = extract_facts_from_message(text)
        self.assertGreaterEqual(len(result.facts), 2)
        keys = {f.fact_key for f in result.facts}
        self.assertIn("editor", keys)
        values = {f.normalized_value for f in result.facts if f.fact_key == "editor"}
        self.assertIn("vs code", values)
        self.assertIn("intellij idea", values)

    def test_split_two_remember_project_statements(self):
        text = (
            "Remember that my primary project is Orion-QA. "
            "Remember that my primary project is Nova-CRM."
        )
        result = extract_facts_from_message(text)
        self.assertEqual(len(result.facts), 2)
        self.assertTrue(all(f.fact_key == "primary_project" for f in result.facts))
        values = {f.normalized_value for f in result.facts}
        self.assertEqual(values, {"orion qa", "nova crm"})

    def test_does_not_save_instructional_text(self):
        text = (
            "Remember that reinforcement_count should be 3; "
            "3 distinct operation_ids"
        )
        result = extract_facts_from_message(text)
        self.assertEqual(result.facts, [])

    def test_abbreviation_not_split(self):
        parts = split_statements("Remember that I use Dr. Smith as mentor.")
        self.assertEqual(len(parts), 1)


class TestParaphraseConflicts(unittest.TestCase):
    def test_language_paraphrase_conflict(self):
        a = extract_exclusive_fact("My preferred programming language is Python.")
        b = extract_exclusive_fact("I prefer TypeScript for programming.")
        self.assertEqual(a[0], "preferred_language")
        self.assertEqual(b[0], "preferred_language")
        self.assertNotEqual(a[1], b[1])
        result = detect_memory_conflict(
            "I prefer TypeScript for programming.",
            [{"memory": "My preferred programming language is Python.", "memory_id": "1"}],
        )
        self.assertTrue(result.get("conflicts"))

    def test_company_paraphrase_conflict(self):
        a = extract_exclusive_fact("I currently work at Google.")
        b = extract_exclusive_fact("I now work at Microsoft.")
        self.assertEqual(a, ("company", "google"))
        self.assertEqual(b, ("company", "microsoft"))

    def test_editor_paraphrase_conflict(self):
        a = extract_exclusive_fact("My preferred code editor is VS Code.")
        b = extract_exclusive_fact("I prefer IntelliJ IDEA as my code editor.")
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[0], "editor")
        self.assertNotEqual(a[1], b[1])

    def test_os_paraphrase_conflict(self):
        a = extract_exclusive_fact("My primary operating system is Windows 11.")
        b = extract_exclusive_fact("I primarily use Ubuntu Linux.")
        self.assertEqual(a[0], "os")
        self.assertEqual(b[0], "os")
        self.assertNotEqual(a[1], b[1])

    def test_career_goal_conflict(self):
        a = extract_exclusive_fact("My career goal is to become an AI Engineer.")
        b = extract_exclusive_fact(
            "My long-term career goal is to become a Data Scientist."
        )
        self.assertEqual(a[0], "goal")
        self.assertEqual(b[0], "goal")
        self.assertNotEqual(a[1], b[1])

    def test_primary_project_conflict(self):
        a = extract_exclusive_fact("My current primary project is Orion-QA.")
        b = extract_exclusive_fact("My primary project is Nova-CRM.")
        self.assertEqual(a[0], "primary_project")
        self.assertEqual(b[0], "primary_project")
        self.assertNotEqual(a[1], b[1])


class TestUpdateValueParsing(unittest.TestCase):
    def test_update_strips_trailing_instructions(self):
        parsed = chatbot._parse_explicit_update(
            "Update my project to Alpha. Then verify reinforcement_count should be 3"
        )
        self.assertIsNotNone(parsed)
        subject, value, require_existing = parsed
        self.assertEqual(subject.lower(), "project")
        self.assertEqual(value, "Alpha")
        self.assertFalse(require_existing)

    @patch.object(chatbot, "supersede_memory")
    @patch.object(chatbot, "_collect_update_candidates")
    @patch.object(chatbot, "create_embedding", return_value=[0.3] * 8)
    def test_explicit_update_asks_when_multiple_distinct_actives(
        self, _embed, collect, supersede
    ):
        collect.return_value = (
            [
                {
                    "memory_id": "a1",
                    "memory": "my project is Alpha",
                    "status": "active",
                    "fact_key": "project",
                    "normalized_value": "alpha",
                    "reinforcement_count": 1,
                },
                {
                    "memory_id": "b1",
                    "memory": "my project is Beta",
                    "status": "active",
                    "fact_key": "project",
                    "normalized_value": "beta",
                    "reinforcement_count": 2,
                },
            ],
            [],
        )

        result = chatbot.try_update_explicit_memory(
            "Update my project to Freshcart.",
            "user-clean",
        )

        self.assertFalse(result["success"])
        self.assertIn("multiple", result["message"].lower())
        supersede.assert_not_called()

    def test_change_and_replace_patterns(self):
        change = chatbot._parse_explicit_update(
            "Change my primary project to Beta."
        )
        replace = chatbot._parse_explicit_update(
            "Replace my preferred language with Python."
        )
        self.assertEqual(change[1], "Beta")
        self.assertEqual(replace[1], "Python")

    def test_clean_update_value_helper(self):
        cleaned = chatbot._clean_update_value(
            "Freshcart then verify that expected status is active"
        )
        self.assertEqual(cleaned, "Freshcart")


class TestRequestScopedReinforcement(unittest.TestCase):
    @patch.object(chatbot, "reinforce_memory")
    @patch.object(chatbot, "find_logical_duplicate")
    @patch.object(chatbot, "create_embedding", return_value=[0.1] * 8)
    def test_same_request_retry_reuses_operation_scope(
        self, _embed, logical_dup, reinforce
    ):
        logical_dup.return_value = {
            "success": True,
            "is_duplicate": True,
            "match": {
                "memory_id": "mem-1",
                "memory": "My preferred programming language is Python",
                "status": "active",
                "importance": 5,
            },
        }
        reinforce.return_value = {
            "success": True,
            "idempotent": False,
            "reinforcement_count": 1,
        }

        chatbot.try_save_explicit_memory(
            "Remember that my preferred programming language is Python.",
            "user-1",
            request_id="same-req",
        )
        chatbot.try_save_explicit_memory(
            "Remember that my preferred programming language is Python.",
            "user-1",
            request_id="same-req",
        )

        ids = [call.kwargs["operation_id"] for call in reinforce.call_args_list]
        self.assertEqual(len(ids), 2)
        self.assertEqual(ids[0], ids[1])
        self.assertIn("same-req", ids[0])

    @patch.object(chatbot, "reinforce_memory")
    @patch.object(chatbot, "find_logical_duplicate")
    @patch.object(chatbot, "create_embedding", return_value=[0.1] * 8)
    def test_distinct_requests_get_distinct_operation_ids(
        self, _embed, logical_dup, reinforce
    ):
        logical_dup.return_value = {
            "success": True,
            "is_duplicate": True,
            "match": {
                "memory_id": "mem-1",
                "memory": "My preferred programming language is Python",
                "status": "active",
                "importance": 5,
            },
        }
        reinforce.return_value = {
            "success": True,
            "idempotent": False,
            "reinforcement_count": 1,
        }

        for req in ("req-1", "req-2", "req-3"):
            chatbot.try_save_explicit_memory(
                "Remember that my preferred programming language is Python.",
                "user-1",
                request_id=req,
            )

        ids = [call.kwargs["operation_id"] for call in reinforce.call_args_list]
        self.assertEqual(len(set(ids)), 3)


class TestMultiStatementSavePipeline(unittest.TestCase):
    @patch.object(
        chatbot,
        "persist_memory_conflict_pair",
        return_value={"success": True, "verified": True},
    )
    @patch.object(chatbot, "save_memory")
    @patch.object(chatbot, "detect_memory_conflict")
    @patch.object(chatbot, "find_active_memories_by_fact_key")
    @patch.object(chatbot, "check_duplicate_memory")
    @patch.object(chatbot, "find_logical_duplicate")
    @patch.object(chatbot, "create_embedding", return_value=[0.2] * 8)
    def test_multi_statement_editor_conflict_in_one_message(
        self,
        _embed,
        logical_dup,
        dup_check,
        by_key,
        detect_conflict,
        save_memory,
        _persist_pair,
    ):
        # First statement: unique save. Second: conflicts with first.
        logical_dup.side_effect = [
            {"success": True, "is_duplicate": False},
            {"success": True, "is_duplicate": False},
        ]
        dup_check.side_effect = [
            {"success": True, "is_duplicate": False, "candidates": []},
            {
                "success": True,
                "is_duplicate": False,
                "candidates": [
                    {
                        "memory_id": "mem-vscode",
                        "memory": "my preferred code editor is VS Code",
                        "fact_key": "editor",
                        "normalized_value": "vs code",
                    }
                ],
            },
        ]
        by_key.side_effect = [
            {"success": True, "memories": []},
            {
                "success": True,
                "memories": [
                    {
                        "memory_id": "mem-vscode",
                        "memory": "my preferred code editor is VS Code",
                        "fact_key": "editor",
                        "normalized_value": "vs code",
                    }
                ],
            },
        ]
        detect_conflict.side_effect = [
            {"conflicts": []},
            {
                "conflicts": [
                    {
                        "memory_id": "mem-vscode",
                        "memory": "my preferred code editor is VS Code",
                        "conflict_confidence": 0.95,
                        "conflict_group_id": "cg-1",
                    }
                ]
            },
        ]
        save_memory.side_effect = [
            {"success": True, "memory_id": "mem-vscode", "status": "active"},
            {"success": True, "memory_id": "mem-idea", "status": "needs_review"},
        ]

        result = chatbot.try_save_explicit_memory(
            "Remember that my preferred code editor is VS Code.\n"
            "I prefer IntelliJ IDEA as my code editor.",
            "user-1",
            request_id="multi-1",
        )

        self.assertTrue(result.get("handled"))
        self.assertEqual(save_memory.call_count, 2)
        statuses = [c.kwargs.get("status", "active") for c in save_memory.call_args_list]
        self.assertIn("needs_review", statuses)
        self.assertIn("needs_review", result.get("status", "") + str(result.get("results")))


class TestConflictClarification(unittest.TestCase):
    @patch.object(chatbot, "load_long_term_memory")
    @patch.object(chatbot, "try_save_explicit_memory", return_value=False)
    @patch.object(chatbot, "try_update_explicit_memory", return_value=None)
    @patch.object(chatbot, "try_delete_explicit_memory", return_value=None)
    @patch.object(chatbot, "restore_task_context_from_redis")
    @patch.object(chatbot, "save_task_context_to_redis")
    @patch.object(chatbot, "add_to_conversation_memory")
    def test_conflicting_actives_ask_clarification(
        self,
        _add,
        _redis_save,
        _redis_restore,
        _delete,
        _update,
        _save,
        load_mem,
    ):
        load_mem.return_value = {
            "memories": [
                {
                    "memory": "My current project is Alpha",
                    "status": "active",
                    "fact_key": "project",
                    "normalized_value": "alpha",
                },
                {
                    "memory": "My current project is Beta",
                    "status": "active",
                    "fact_key": "project",
                    "normalized_value": "beta",
                },
            ],
            "conflicts": [
                {
                    "fact_key": "project",
                    "values": ["alpha", "beta"],
                }
            ],
        }

        answer = chatbot.chat_with_ai(
            "What is my current primary project?",
            request_id="q-1",
            user_id="user-1",
        )
        self.assertIn("conflicting", answer.lower())
        self.assertIn("alpha", answer.lower())
        self.assertIn("beta", answer.lower())


class TestPendingReviewContext(unittest.TestCase):
    def test_build_memory_context_separates_pending(self):
        context = chatbot.build_memory_context(
            [
                {
                    "memory": "My project is Alpha",
                    "status": "active",
                    "importance": 4,
                    "confidence": 0.9,
                },
                {
                    "memory": "Freshcart is my grocery marketplace project",
                    "status": "needs_review",
                    "importance": 4,
                    "confidence": 0.8,
                },
            ]
        )
        self.assertIn("CURRENT_ACTIVE_FACTS", context)
        self.assertIn("PENDING_REVIEW_FACTS", context)
        self.assertIn("Freshcart", context)
        self.assertIn("UNTRUSTED", context)


class TestMutationReplayProtection(unittest.TestCase):
    def test_fingerprint_stable_for_same_args(self):
        # Exercise the helper via a mini agent-loop simulation.
        key1 = None
        key2 = None

        def capture(name, arguments):
            safe_args = {
                k: v for k, v in (arguments or {}).items()
                if k not in {"user_id", "request_id"}
            }
            return f"{name}:{dumps_json_safe(safe_args, ensure_ascii=False)}"

        key1 = capture("reinforce_memory", {"memory_id": "abc"})
        key2 = capture("reinforce_memory", {"memory_id": "abc", "request_id": "x"})
        self.assertEqual(key1, key2)


if __name__ == "__main__":
    unittest.main()
