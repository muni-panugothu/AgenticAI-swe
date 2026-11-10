"""Regression tests: retrieval precision, file reader, agent loop, calculator."""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from app import chatbot
from app.services.memory_service import (
    _project_intent_boost,
    infer_query_fact_keys,
    semantic_search_memories,
)
from app.tools import file_reader
from app.tools.calculator import calculator
from app.tools.file_reader import read_file


class TestEditorRetrievalPrecision(unittest.TestCase):
    def test_infer_editor_fact_key(self):
        keys = infer_query_fact_keys(
            "What is my current preferred code editor?"
        )
        self.assertEqual(keys, ["editor"])

    def test_language_query_does_not_claim_editor(self):
        keys = infer_query_fact_keys(
            "What is my preferred programming language?"
        )
        self.assertEqual(keys, ["preferred_language"])
        self.assertNotIn("editor", keys)

    def test_editor_query_boosts_editor_over_language(self):
        q = "What is my current preferred code editor?"
        editor_mem = {
            "fact_key": "editor",
            "memory": "My preferred code editor is VS Code",
            "score": 0.82,
        }
        language_mem = {
            "fact_key": "preferred_language",
            "memory": "My preferred programming language is Python",
            "score": 0.85,
        }
        self.assertGreater(
            _project_intent_boost(q, editor_mem),
            _project_intent_boost(q, language_mem),
        )

    def test_fact_key_filtering_prefers_editor_matches(self):
        """Soft filter: matching fact_key stays ahead of mismatches."""
        editor = {
            "memory_id": "e1",
            "user_id": "user-test",
            "owner_id": "user-test",
            "permission_level": "private",
            "shared_with": [],
            "fact_key": "editor",
            "memory": "My preferred code editor is IntelliJ IDEA",
            "score": 0.81,
            "status": "active",
            "importance": 4,
            "confidence": 0.9,
        }
        language = {
            "memory_id": "l1",
            "user_id": "user-test",
            "owner_id": "user-test",
            "permission_level": "private",
            "shared_with": [],
            "fact_key": "preferred_language",
            "memory": "My preferred programming language is Python",
            "score": 0.92,
            "status": "active",
            "importance": 4,
            "confidence": 0.9,
        }

        with patch(
            "app.services.memory_service.create_embedding",
            return_value=[0.1] * 8,
        ), patch(
            "app.services.memory_service._vector_search_memories",
            side_effect=lambda *a, **k: (
                [language, editor]
                if k.get("status_filter") != {"status": "needs_review"}
                else []
            ),
        ), patch(
            "app.services.memory_service.annotate_memories_with_lifecycle",
            side_effect=lambda xs: xs,
        ), patch(
            "app.services.memory_service._record_memory_access",
        ), patch(
            "app.services.memory_service.record_audit_event",
        ):
            result = semantic_search_memories(
                "user-test",
                "What is my current preferred code editor?",
                limit=5,
                min_score=0.80,
                include_needs_review=False,
            )

        self.assertTrue(result["success"])
        memories = result["memories"]
        self.assertGreaterEqual(len(memories), 1)
        self.assertEqual(memories[0]["fact_key"], "editor")


class TestFileReaderValidation(unittest.TestCase):
    def test_missing_file_returns_structured_error(self):
        result = read_file(
            "app/definitely_missing_xyz_123.py",
            line_start=1,
            line_end=10,
        )
        self.assertIsInstance(result, dict)
        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "FILE_NOT_FOUND")
        self.assertIn("does not exist", result["message"])

    def test_invalid_line_range(self):
        result = read_file("app/tools/calculator.py", line_start=10, line_end=5)
        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "INVALID_LINE_RANGE")

    def test_line_start_beyond_eof(self):
        result = read_file(
            "app/tools/calculator.py",
            line_start=99999,
            line_end=100010,
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "INVALID_LINE_RANGE")

    def test_path_traversal_denied(self):
        result = read_file("../Secrets/passwords.txt", line_start=1, line_end=5)
        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "PATH_TRAVERSAL_DENIED")

    def test_directory_rejected(self):
        result = read_file("app/tools", line_start=1, line_end=5)
        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "IS_DIRECTORY")

    def test_empty_file_is_not_missing_but_clear(self):
        empty_path = file_reader.PROJECT_ROOT / "_tmp_empty_test_file.py"
        try:
            empty_path.write_text("", encoding="utf-8")
            rel = str(empty_path.relative_to(file_reader.PROJECT_ROOT)).replace(
                "\\", "/"
            )
            result = read_file(rel, line_start=1, line_end=20)
            self.assertIsInstance(result, dict)
            self.assertTrue(result["success"])
            self.assertTrue(result.get("empty"))
            self.assertIn("empty", result["message"].lower())
        finally:
            if empty_path.exists():
                empty_path.unlink()

    def test_successful_read_is_json_safe_prefix(self):
        result = read_file("app/tools/calculator.py", line_start=1, line_end=20)
        self.assertIsInstance(result, str)
        meta = json.loads(result.split("\ncontent:\n", 1)[0])
        self.assertTrue(meta["success"])
        self.assertFalse(meta.get("empty"))

    def test_function_boundary_stops_before_next_top_level_def(self):
        # Window covering the end of execute_tool must stop before add_to_*.
        result = read_file("app/chatbot.py", line_start=900, line_end=980)
        self.assertIsInstance(result, str)
        meta, _, body = result.partition("\ncontent:\n")
        payload = json.loads(meta)
        self.assertTrue(payload["span_limited"])
        self.assertFalse(payload["needs_continuation"])
        self.assertTrue(payload["function_complete"])
        self.assertNotIn("def add_to_conversation_memory", body)
        self.assertIn("Tool execution error", body)

    def test_span_limited_mid_function_still_needs_continuation(self):
        result = read_file("app/chatbot.py", line_start=820, line_end=890)
        payload = json.loads(result.split("\ncontent:\n", 1)[0])
        self.assertTrue(payload["span_limited"])
        self.assertTrue(payload["needs_continuation"])
        self.assertFalse(payload.get("function_complete"))
        self.assertEqual(payload["continuation"]["next_line"], payload["line_end"] + 1)

    def test_json_safe_tool_errors_via_execute_tool(self):
        result = chatbot.execute_tool(
            "read_file",
            {
                "file_path": "app/no_such_file_zzz.py",
                "line_start": 1,
                "line_end": 10,
            },
            user_id="tool-user",
        )
        self.assertTrue(result["success"])  # transport ok
        payload = json.loads(result["result"])
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error_code"], "FILE_NOT_FOUND")
        # Outer envelope must serialize cleanly for the agent loop.
        from app.utils.json_safe import dumps_json_safe

        dumps_json_safe(result)


class TestCalculatorTool(unittest.TestCase):
    def test_calculator_returns_structured_success(self):
        result = calculator("125 * 48 + 750")
        self.assertTrue(result["success"])
        self.assertEqual(result["result"], 6750)

    def test_calculator_rejects_code_injection(self):
        result = calculator("__import__('os').system('echo hi')")
        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "CALCULATION_ERROR")


class TestAgentLoopToolBehavior(unittest.TestCase):
    def _patch_chat_deps(self):
        return [
            patch.object(chatbot, "client"),
            patch.object(
                chatbot,
                "load_long_term_memory",
                return_value={"memories": [], "conflicts": []},
            ),
            patch.object(
                chatbot, "try_save_explicit_memory", return_value=False
            ),
            patch.object(
                chatbot, "try_update_explicit_memory", return_value=None
            ),
            patch.object(
                chatbot, "try_delete_explicit_memory", return_value=None
            ),
            patch.object(chatbot, "restore_task_context_from_redis"),
            patch.object(chatbot, "save_task_context_to_redis"),
            patch.object(chatbot, "add_to_conversation_memory"),
            patch.object(
                chatbot, "get_conversation_memory", return_value=[]
            ),
        ]

    def test_forces_calculator_on_explicit_calculate_request(self):
        patches = self._patch_chat_deps()
        mocks = [p.start() for p in patches]
        try:
            mock_client = mocks[0]

            calc_call = MagicMock()
            calc_call.id = "call-calc"
            calc_call.function.name = "calculator"
            calc_call.function.arguments = json.dumps(
                {"expression": "125 * 48 + 750"}
            )

            first = MagicMock()
            first.tool_calls = [calc_call]
            first.content = None

            final = MagicMock()
            final.tool_calls = None
            final.content = "The result is 6750."

            mock_client.chat.completions.create.side_effect = [
                MagicMock(choices=[MagicMock(message=first)]),
                MagicMock(choices=[MagicMock(message=final)]),
            ]

            with patch.object(chatbot, "execute_tool") as exec_tool:
                exec_tool.return_value = {
                    "success": True,
                    "result": json.dumps(
                        {
                            "success": True,
                            "expression": "125 * 48 + 750",
                            "result": 6750,
                        }
                    ),
                }
                answer = chatbot.chat_with_ai(
                    "Calculate 125 * 48 + 750.",
                    user_id="tool-user",
                    request_id="req-calc",
                )

            self.assertIn("6750", answer)
            first_kwargs = (
                mock_client.chat.completions.create.call_args_list[0].kwargs
            )
            # Calculator-only requests may force the calculator tool.
            self.assertEqual(
                first_kwargs["tool_choice"]["function"]["name"],
                "calculator",
            )
            exec_tool.assert_called()
            self.assertEqual(exec_tool.call_args.args[0], "calculator")

            # Multi-tool arithmetic must NOT force calculator tool_choice.
            multi = (
                "Search for reinforce_memory, read app/services/memory_service.py, "
                "and calculate 125 * 48 + 750."
            )
            choice, _ = chatbot.resolve_tool_choice(
                user_message=multi,
                step=0,
                max_steps=15,
                calculator_used=False,
            )
            self.assertEqual(choice, "auto")
        finally:
            for p in patches:
                p.stop()

    def test_skips_repeated_unsuccessful_read_and_search(self):
        patches = self._patch_chat_deps()
        mocks = [p.start() for p in patches]
        try:
            mock_client = mocks[0]

            read_call = MagicMock()
            read_call.id = "call-read"
            read_call.function.name = "read_file"
            read_call.function.arguments = json.dumps(
                {
                    "file_path": "app/main.py",
                    "line_start": 1,
                    "line_end": 40,
                }
            )

            search_call = MagicMock()
            search_call.id = "call-search"
            search_call.function.name = "search_code"
            search_call.function.arguments = json.dumps(
                {"search_text": "main.py"}
            )

            read_again = MagicMock()
            read_again.id = "call-read-2"
            read_again.function.name = "read_file"
            read_again.function.arguments = json.dumps(
                {
                    "file_path": "app/main.py",
                    "line_start": 1,
                    "line_end": 40,
                }
            )

            step1 = MagicMock()
            step1.tool_calls = [read_call]
            step1.content = None

            step2 = MagicMock()
            step2.tool_calls = [search_call, read_again]
            step2.content = None

            final = MagicMock()
            final.tool_calls = None
            final.content = (
                "app/main.py does not exist, so I cannot read it."
            )

            mock_client.chat.completions.create.side_effect = [
                MagicMock(choices=[MagicMock(message=step1)]),
                MagicMock(choices=[MagicMock(message=step2)]),
                MagicMock(choices=[MagicMock(message=final)]),
            ]

            missing = {
                "success": True,
                "result": json.dumps(
                    {
                        "success": False,
                        "error_code": "FILE_NOT_FOUND",
                        "message": "The requested file does not exist.",
                        "file_path": "app/main.py",
                    }
                ),
            }

            with patch.object(
                chatbot, "execute_tool", return_value=missing
            ) as exec_tool:
                answer = chatbot.chat_with_ai(
                    "Read app/main.py and summarize it.",
                    user_id="tool-user",
                    request_id="req-missing",
                )

            self.assertIn("does not exist", answer.lower())
            # First read executes; later identical read + search for main.py
            # must be short-circuited inside the agent loop (no re-execute).
            executed_names = [c.args[0] for c in exec_tool.call_args_list]
            self.assertEqual(executed_names.count("read_file"), 1)
            self.assertNotIn("search_code", executed_names)
            # Finalization makes a 4th tool-free LLM call after the draft.
            # When that call fails (exhausted mocks → StopIteration), the
            # agent must still return a safe useful answer from the draft
            # and/or offline FILE_NOT_FOUND evidence — never crash.
            self.assertGreaterEqual(
                mock_client.chat.completions.create.call_count,
                3,
            )
            self.assertTrue(
                "file_not_found" in answer.lower()
                or "does not exist" in answer.lower()
                or "evidence ledger" in answer.lower(),
                msg=answer[:600],
            )
        finally:
            for p in patches:
                p.stop()

    def test_finalization_llm_failure_returns_safe_partial(self):
        """Dedicated tool-free finalization failure → offline/draft partial."""
        patches = self._patch_chat_deps()
        mocks = [p.start() for p in patches]
        try:
            mock_client = mocks[0]

            read_call = MagicMock()
            read_call.id = "call-read"
            read_call.function.name = "read_file"
            read_call.function.arguments = json.dumps(
                {
                    "file_path": "app/main.py",
                    "line_start": 1,
                    "line_end": 40,
                }
            )
            step1 = MagicMock()
            step1.tool_calls = [read_call]
            step1.content = None

            # Planning turn already has a useful draft, then finalization
            # LLM call is forced to fail.
            draft = MagicMock()
            draft.tool_calls = None
            draft.content = (
                "app/main.py does not exist, so I cannot summarize it."
            )

            mock_client.chat.completions.create.side_effect = [
                MagicMock(choices=[MagicMock(message=step1)]),
                MagicMock(choices=[MagicMock(message=draft)]),
                RuntimeError("simulated finalization transport failure"),
            ]

            missing = {
                "success": True,
                "result": json.dumps(
                    {
                        "success": False,
                        "error_code": "FILE_NOT_FOUND",
                        "message": "The requested file does not exist.",
                        "file_path": "app/main.py",
                    }
                ),
            }

            with patch.object(
                chatbot, "execute_tool", return_value=missing
            ):
                answer = chatbot.chat_with_ai(
                    "Read app/main.py and summarize it.",
                    user_id="tool-user-final-fail",
                    request_id="req-final-fail",
                )

            lowered = answer.lower()
            self.assertIn("does not exist", lowered)
            # Must remain an honest partial / evidence-backed answer.
            self.assertTrue(
                "evidence ledger" in lowered
                or "partial" in lowered
                or "file_not_found" in lowered,
                msg=answer[:600],
            )
        finally:
            for p in patches:
                p.stop()

    def test_multi_tool_execution_preserves_parallel_calls(self):
        patches = self._patch_chat_deps()
        mocks = [p.start() for p in patches]
        try:
            mock_client = mocks[0]

            calc_call = MagicMock()
            calc_call.id = "c1"
            calc_call.function.name = "calculator"
            calc_call.function.arguments = json.dumps(
                {"expression": "10 + 5"}
            )

            time_call = MagicMock()
            time_call.id = "c2"
            time_call.function.name = "get_current_time"
            time_call.function.arguments = json.dumps({})

            step1 = MagicMock()
            step1.tool_calls = [calc_call, time_call]
            step1.content = None

            final = MagicMock()
            final.tool_calls = None
            final.content = "15 and the current time were retrieved."

            mock_client.chat.completions.create.side_effect = [
                MagicMock(choices=[MagicMock(message=step1)]),
                MagicMock(choices=[MagicMock(message=final)]),
            ]

            def fake_execute(name, args, user_id=None, request_id=None, **_kwargs):
                if name == "calculator":
                    return {
                        "success": True,
                        "result": json.dumps(
                            {"success": True, "result": 15}
                        ),
                    }
                return {
                    "success": True,
                    "result": json.dumps(
                        {"success": True, "time": "12:00"}
                    ),
                }

            with patch.object(
                chatbot, "execute_tool", side_effect=fake_execute
            ) as exec_tool:
                answer = chatbot.chat_with_ai(
                    "Calculate 10 + 5 and tell me the current time.",
                    user_id="tool-user",
                    request_id="req-multi",
                )

            names = [c.args[0] for c in exec_tool.call_args_list]
            self.assertIn("calculator", names)
            self.assertIn("get_current_time", names)
            self.assertIn("15", answer)
        finally:
            for p in patches:
                p.stop()


if __name__ == "__main__":
    unittest.main()
