"""Critical Stage 19 fixes: tool_choice, database contradictions, file reader."""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from app import chatbot
from app.services.memory_conflict_service import (
    classify_duplicate,
    extract_exclusive_fact,
)
from app.tools.file_reader import read_file
from app.tools.calculator import calculator


class TestGroqToolChoice(unittest.TestCase):
    def test_multi_tool_request_uses_auto_not_forced_calculator(self):
        msg = (
            "Search the backend for reinforce_memory, read its implementation "
            "from app/services/memory_service.py, and calculate (250 * 32) + 1800 "
            "using your calculator tool. Combine all three results in one answer."
        )
        self.assertTrue(chatbot.is_multi_tool_request(msg))
        self.assertTrue(chatbot.requires_calculator(msg))
        choice, nudge = chatbot.resolve_tool_choice(
            user_message=msg,
            step=0,
            max_steps=15,
            calculator_used=False,
        )
        self.assertEqual(choice, "auto")
        self.assertIsNotNone(nudge)
        self.assertIn("multiple tools", nudge.lower())

    def test_calculator_only_may_force_calculator(self):
        msg = "Calculate 125 * 48 + 750."
        self.assertFalse(chatbot.is_multi_tool_request(msg))
        choice, _ = chatbot.resolve_tool_choice(
            user_message=msg,
            step=0,
            max_steps=15,
            calculator_used=False,
        )
        self.assertEqual(choice["function"]["name"], "calculator")

    def test_sequential_multi_tool_execution_order(self):
        patches = [
            patch.object(chatbot, "client"),
            patch.object(
                chatbot,
                "load_long_term_memory",
                return_value={"memories": [], "conflicts": []},
            ),
            patch.object(chatbot, "try_save_explicit_memory", return_value=False),
            patch.object(chatbot, "try_update_explicit_memory", return_value=None),
            patch.object(chatbot, "try_delete_explicit_memory", return_value=None),
            patch.object(chatbot, "restore_task_context_from_redis"),
            patch.object(chatbot, "save_task_context_to_redis"),
            patch.object(chatbot, "add_to_conversation_memory"),
            patch.object(chatbot, "get_conversation_memory", return_value=[]),
        ]
        mocks = [p.start() for p in patches]
        try:
            mock_client = mocks[0]

            def _tool_call(call_id, name, args):
                call = MagicMock()
                call.id = call_id
                call.function.name = name
                call.function.arguments = json.dumps(args)
                return call

            step1 = MagicMock()
            step1.tool_calls = [
                _tool_call(
                    "c1",
                    "search_code",
                    {"search_text": "reinforce_memory"},
                )
            ]
            step1.content = None

            step2 = MagicMock()
            step2.tool_calls = [
                _tool_call(
                    "c2",
                    "read_file",
                    {
                        "file_path": "app/services/memory_service.py",
                        "line_start": 1,
                        "line_end": 40,
                    },
                )
            ]
            step2.content = None

            step3 = MagicMock()
            step3.tool_calls = [
                _tool_call(
                    "c3",
                    "calculator",
                    {"expression": "(250 * 32) + 1800"},
                )
            ]
            step3.content = None

            final = MagicMock()
            final.tool_calls = None
            final.content = (
                "Found reinforce_memory, read the implementation, "
                "and calculator result is 9800."
            )

            mock_client.chat.completions.create.side_effect = [
                MagicMock(choices=[MagicMock(message=step1)]),
                MagicMock(choices=[MagicMock(message=step2)]),
                MagicMock(choices=[MagicMock(message=step3)]),
                MagicMock(choices=[MagicMock(message=final)]),
            ]

            executed = []

            def fake_execute(name, args, user_id=None, request_id=None, **_kwargs):
                executed.append(name)
                if name == "calculator":
                    return {
                        "success": True,
                        "result": json.dumps(
                            {
                                "success": True,
                                "expression": "(250 * 32) + 1800",
                                "result": 9800,
                            }
                        ),
                    }
                if name == "read_file":
                    return {
                        "success": True,
                        "result": (
                            json.dumps(
                                {
                                    "success": True,
                                    "file_path": args["file_path"],
                                    "line_start": 1,
                                    "line_end": 5,
                                    "empty": False,
                                }
                            )
                            + "\ncontent:\n"
                            + "def reinforce_memory(...):\n    pass\n"
                        ),
                    }
                return {
                    "success": True,
                    "result": json.dumps(
                        {
                            "success": True,
                            "matches": [
                                {
                                    "file_path": "app/services/memory_service.py",
                                    "line": 10,
                                    "text": "def reinforce_memory",
                                }
                            ],
                        }
                    ),
                }

            msg = (
                "Search the backend for reinforce_memory, read its "
                "implementation from app/services/memory_service.py, and "
                "calculate (250 * 32) + 1800 using your calculator tool."
            )

            with patch.object(
                chatbot, "execute_tool", side_effect=fake_execute
            ):
                answer = chatbot.chat_with_ai(
                    msg,
                    user_id="tool-user",
                    request_id="req-seq",
                )

            self.assertEqual(
                executed,
                ["search_code", "read_file", "calculator"],
            )
            first_kwargs = (
                mock_client.chat.completions.create.call_args_list[0].kwargs
            )
            self.assertEqual(first_kwargs["tool_choice"], "auto")
            self.assertIn("9800", answer)
        finally:
            for p in patches:
                p.stop()


class TestPreferredDatabaseClassification(unittest.TestCase):
    def test_extract_preferred_database(self):
        fact = extract_exclusive_fact(
            "my preferred database is MongoDB"
        )
        self.assertEqual(fact, ("preferred_database", "mongodb"))

        fact2 = extract_exclusive_fact(
            "my preferred database is PostgreSQL"
        )
        self.assertEqual(fact2, ("preferred_database", "postgresql"))

    def test_paraphrase_duplicate_same_database(self):
        result = classify_duplicate(
            "My preferred database is MongoDB.",
            "I prefer MongoDB as my database.",
            vector_score=0.97,
        )
        self.assertEqual(result["classification"], "paraphrase_duplicate")

    def test_contradiction_mongodb_vs_postgresql(self):
        result = classify_duplicate(
            "My preferred database is PostgreSQL.",
            "My preferred database is MongoDB.",
            vector_score=0.96,
        )
        self.assertEqual(result["classification"], "possible_contradiction")
        self.assertGreaterEqual(result["confidence"], 0.80)

    def test_vector_similarity_alone_does_not_override_contradiction(self):
        # Even at near-identical embedding score, different exclusive values
        # must remain contradictions.
        result = classify_duplicate(
            "my preferred database is PostgreSQL",
            "my preferred database is MongoDB",
            vector_score=0.99,
        )
        self.assertEqual(result["classification"], "possible_contradiction")

    def test_project_database_usage_not_preferred_database(self):
        fact = extract_exclusive_fact(
            "I am building a project that uses PostgreSQL"
        )
        self.assertNotEqual(
            (fact or (None, None))[0],
            "preferred_database",
        )


class TestFileReaderCritical(unittest.TestCase):
    def test_missing_main_returns_file_not_found(self):
        # FastAPI entry lives at app/main.py now; use a path that must not exist.
        result = read_file(
            "app/nonexistent_entry_for_tests.py",
            line_start=1,
            line_end=40,
        )
        self.assertIsInstance(result, dict)
        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "FILE_NOT_FOUND")
        self.assertEqual(
            result["message"],
            "The requested file does not exist.",
        )

    def test_valid_file_read(self):
        result = read_file(
            "app/services/memory_service.py",
            line_start=1,
            line_end=20,
        )
        self.assertIsInstance(result, str)
        meta = json.loads(result.split("\ncontent:\n", 1)[0])
        self.assertTrue(meta["success"])

    def test_invalid_line_range(self):
        result = read_file(
            "app/tools/calculator.py",
            line_start=20,
            line_end=5,
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "INVALID_LINE_RANGE")

    def test_directory_rejected(self):
        result = read_file("app/services", line_start=1, line_end=10)
        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "IS_DIRECTORY")

    def test_path_traversal(self):
        result = read_file("../etc/passwd", line_start=1, line_end=5)
        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "PATH_TRAVERSAL_DENIED")

    def test_calculator_expression(self):
        result = calculator("(250 * 32) + 1800")
        self.assertTrue(result["success"])
        self.assertEqual(result["result"], 9800)


if __name__ == "__main__":
    unittest.main()
